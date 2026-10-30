"""Standard LeRobot ACT training with episode holdout checkpoint selection."""
import copy
import json
import logging
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Subset

from lerobot.datasets.compute_stats import get_feature_stats
from lerobot.scripts import lerobot_train as trainer
from lerobot.utils.random_utils import get_rng_state, set_rng_state

original_make_dataset = trainer.make_dataset
original_save_checkpoint = trainer.save_checkpoint
validation_loader = None


class EpisodeSubset(Subset):
    """Keep underlying global indices intact for ACT's future-action queries."""
    @property
    def num_frames(self):
        return len(self.indices)

    def __getattr__(self, name):
        return getattr(self.dataset, name)


def select_episodes(dataset, episodes):
    values = dataset.hf_dataset.with_format(None)["episode_index"]
    selected = set(episodes)
    indices = [i for i, episode in enumerate(values) if episode in selected]
    if not indices:
        raise RuntimeError("Empty episode split")
    return EpisodeSubset(dataset, indices)


def make_training_dataset(cfg):
    dataset = original_make_dataset(cfg)
    # Save and validate the first ten updates before the regular 500-step cadence.
    if not cfg.resume:
        cfg.save_freq = 10
    # Recompute vector normalization from training episodes, excluding the holdout.
    parts = [pq.read_table(p, columns=["episode_index", "action", "observation.state"])
             for p in sorted(Path(cfg.dataset.root).glob("data/*/*.parquet"))]
    selected = set(cfg.dataset.episodes)
    rows = [r for table in parts for r in table.to_pylist() if r["episode_index"] in selected]
    for key in ("action", "observation.state"):
        dataset.meta.stats[key] = get_feature_stats(np.asarray([r[key] for r in rows],
                                                   dtype=np.float32), axis=0, keepdims=False)
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stats = {key: {k: np.asarray(v).tolist() for k, v in dataset.meta.stats[key].items()}
             for key in ("action", "observation.state")}
    (output / "training_vector_stats.json").write_text(json.dumps(stats, indent=2))
    (output / "split.json").write_text(json.dumps(dict(train_episodes=sorted(selected),
        validation_episodes=[9], validation_max_frames=128, selection_metric="normalized_masked_chunk_l1"), indent=2))
    subset = select_episodes(dataset, cfg.dataset.episodes)
    logging.info("Verified training split: episodes=%s frames=%s", cfg.dataset.episodes, len(subset))
    return subset


def save_with_validation(checkpoint_dir, step, cfg, policy, optimizer, scheduler=None,
                         preprocessor=None, postprocessor=None):
    global validation_loader
    cfg.save_freq = 500
    original_save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, scheduler,
                             preprocessor, postprocessor)
    rng = get_rng_state()
    was_training = policy.training
    try:
        if validation_loader is None:
            val_cfg = copy.deepcopy(cfg)
            val_cfg.dataset.episodes = [9]
            val_cfg.dataset.image_transforms.enable = False
            dataset = select_episodes(original_make_dataset(val_cfg), [9])
            indices = np.unique(np.linspace(0, len(dataset)-1, min(128, len(dataset)), dtype=int)).tolist()
            validation_loader = DataLoader(Subset(dataset, indices), batch_size=cfg.batch_size,
                                           shuffle=False, num_workers=0)
        total = count = 0
        per_joint = torch.zeros(6, dtype=torch.float64)
        with torch.inference_mode():
            policy.eval()
            for batch in validation_loader:
                batch = preprocessor(batch)
                predicted = policy.predict_action_chunk(batch)
                error = (predicted - batch["action"]).abs()
                mask = (~batch["action_is_pad"]).unsqueeze(-1).expand_as(error)
                total += error[mask].sum().item()
                count += mask.sum().item()
                per_joint += (error * mask).sum(dim=(0, 1)).cpu().double()
        score = total / count
        if not math.isfinite(score):
            raise RuntimeError("Nonfinite validation error")
        result = dict(step=step, normalized_masked_chunk_l1=score,
                      per_joint_l1=(per_joint / (count / 6)).tolist(), frames=len(validation_loader.dataset))
        output = Path(cfg.output_dir)
        with (output / "validation.jsonl").open("a") as f:
            f.write(json.dumps(result) + "\n")
        best_path = output / "best_validation.json"
        if not best_path.exists() or score < json.loads(best_path.read_text())["normalized_masked_chunk_l1"]:
            best_path.write_text(json.dumps({**result, "checkpoint": str(checkpoint_dir)}, indent=2))
            link = output / "checkpoints" / "best"
            if link.is_symlink():
                link.unlink()
            link.symlink_to(Path(checkpoint_dir).name, target_is_directory=True)
        logging.info("HELDOUT step=%s normalized_masked_chunk_l1=%.6f", step, score)
    finally:
        policy.train(was_training)
        set_rng_state(rng)


if __name__ == "__main__":
    torch.set_num_threads(4)
    trainer.make_dataset = make_training_dataset
    trainer.save_checkpoint = save_with_validation
    trainer.init_logging(log_file=Path("outputs/training_jobs/act_so101_20k_holdout.log"))
    trainer.train()
