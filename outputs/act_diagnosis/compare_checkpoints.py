"""Compare saved checkpoints on fixed heldout opening/closing transition frames."""
import contextlib
import gc
import io
import json
import time
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch

from lerobot.datasets.video_utils import decode_video_frames
from lerobot.scripts.lerobot_so101_act_inference import (
    DATASET_DEFAULT,EXPECTED_JOINTS,_load_runtime,_policy_observation,_infer,
)


def main():
    torch.set_num_threads(4)
    out=Path("outputs/act_diagnosis/checkpoint_comparison_fullchunk")
    out.mkdir(exist_ok=True)
    root=Path("outputs/train/act_so101_c270_pick_place_05_cpu_20k_holdout_002")
    rows=json.loads(Path("outputs/act_diagnosis/episode9.json").read_text())
    events=json.loads(Path("outputs/act_diagnosis/gripper_events.json").read_text())
    # Fix the diagnostic frame set before comparing any candidate.
    uniform=set(np.linspace(0,len(rows)-1,40,dtype=int).tolist())
    event_sets=[set(max(0,min(len(rows)-1,e["frame"]+offset)) for offset in (-9,-6,-3,0,3,6,9)) for e in events]
    selected=sorted(uniform.union(*event_sets))
    statistics=json.loads((root/"training_vector_stats.json").read_text())
    std=np.asarray(statistics["action"]["std"],dtype=np.float64)
    if std.shape!=(6,) or not np.isfinite(std).all() or (std<=0).any():
        raise RuntimeError("Invalid training-only normalization statistics")
    metadata=pq.read_table(DATASET_DEFAULT/"meta/episodes/chunk-000/file-000.parquet").to_pylist()[9]
    offset=metadata["videos/observation.images.top/from_timestamp"]
    frames=decode_video_frames(DATASET_DEFAULT/"videos/observation.images.top/chunk-000/file-000.mp4",
        [offset+rows[i]["timestamp"] for i in selected],.05,backend="torchcodec")
    images=[(f.permute(1,2,0).numpy()*255).round().astype(np.uint8) for f in frames]
    (out/"protocol.json").write_text(json.dumps(dict(episode=9,events=events,frames=selected,
        uniform_frames=sorted(uniform),critical_transition_indices=[1,2],
        selection_metric="equal-weight mean of normalized gripper MAE and normalized mean arm MAE for all valid future actions landing in the pickup-close and release-open windows",
        caveat="Uses the existing validation episode; this is diagnostic model selection, not an independent task-success test."),indent=2))
    for e in events:
        image=images[selected.index(e["frame"])]
        cv2.imwrite(str(out/f"transition_{e['frame']:04d}_{e['to']}.jpg"),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
    checkpoints=sorted((root/"checkpoints").glob("[0-9]*"),key=lambda p:int(p.name))
    score_path=out/"scores.jsonl"
    all_scores=[]
    began=time.monotonic()
    for checkpoint in checkpoints:
        step=int(checkpoint.name)
        args=Namespace(model=checkpoint/"pretrained_model",dataset_root=DATASET_DEFAULT,
            follower_port="/dev/ttyACM1",camera="/dev/video0",position_mode="range_m100_100")
        with contextlib.redirect_stdout(io.StringIO()):
            _,_,policy,pre,post=_load_runtime(args)
        predictions=[]
        chunks=[]
        for index,image in zip(selected,images):
            policy.reset()
            observation=_policy_observation(np.asarray(rows[index]["observation.state"],dtype=np.float32),image)
            first=_infer(observation,policy,pre,post).numpy()
            if len(policy._action_queue)!=49:
                raise RuntimeError("Expected a 50-action ACT chunk")
            with torch.inference_mode():
                future=post(torch.cat(list(policy._action_queue),dim=0)).detach().cpu().numpy()
            chunk=np.vstack((first[None,:],future))
            if chunk.shape!=(50,6) or not np.isfinite(chunk).all():
                raise RuntimeError("Invalid postprocessed action chunk")
            predictions.append(first.tolist())
            chunks.append(chunk)
        predicted=np.asarray(predictions)
        truth=np.asarray([rows[i]["action"] for i in selected])
        errors=np.abs(predicted-truth)
        normalized=errors/std
        base_mask=np.asarray([i in uniform for i in selected])
        critical_frames=event_sets[1]|event_sets[2]
        critical_mask=np.asarray([i in critical_frames for i in selected])
        first_critical_gripper=float(normalized[critical_mask,5].mean())
        first_critical_arm=float(normalized[critical_mask,:5].mean())
        future_indices=np.asarray(selected)[:,None]+np.arange(50)[None,:]
        valid=future_indices<len(rows)
        all_truth=np.asarray([r["action"] for r in rows])
        chunk_errors=np.abs(np.asarray(chunks)-all_truth[np.minimum(future_indices,len(rows)-1)])
        chunk_normalized=chunk_errors/std
        critical_future=np.isin(future_indices,list(critical_frames))&valid
        critical_gripper=float(chunk_normalized[:,:,5][critical_future].mean())
        critical_arm=float(chunk_normalized[:,:,:5][critical_future].mean())
        uniform_future=valid&base_mask[:,None]
        truth_gripper=all_truth[np.minimum(future_indices,len(rows)-1),5]
        severe_closed=(np.asarray(chunks)[:,:,5]>=12)&(truth_gripper<=5)&critical_future
        severe_open=(np.asarray(chunks)[:,:,5]<=5)&(truth_gripper>=12)&critical_future
        transitions=[]
        for e,indices in zip(events,event_sets):
            mask=np.asarray([i in indices for i in selected])
            after=np.asarray([e["frame"]<=i<=e["frame"]+9 for i in selected])
            wrong=(predicted[after,5]>=12) if e["to"]=="closed" else (predicted[after,5]<=5)
            transitions.append(dict(frame=e["frame"],to=e["to"],
                gripper_mae=float(errors[mask,5].mean()),gripper_max_error=float(errors[mask,5].max()),
                severe_opposite_predictions=int(wrong.sum()),after_samples=int(after.sum())))
        score=dict(step=step,checkpoint=str(checkpoint),
            uniform_first_action_normalized_mae=float(normalized[base_mask].mean()),
            uniform_full_chunk_normalized_mae=float(chunk_normalized[uniform_future].mean()),
            first_action_critical_score=(first_critical_gripper+first_critical_arm)/2,
            critical_score=(critical_gripper+critical_arm)/2,
            critical_gripper_normalized_mae=critical_gripper,critical_arm_normalized_mae=critical_arm,
            critical_gripper_mae=float(chunk_errors[:,:,5][critical_future].mean()),
            critical_future_action_count=int(critical_future.sum()),
            critical_opposite_open_predictions=int(severe_closed.sum()),
            critical_opposite_closed_predictions=int(severe_open.sum()),
            transitions=transitions,
            joint_mae=dict(zip(EXPECTED_JOINTS,errors.mean(0).tolist(),strict=True)))
        all_scores.append(score)
        with score_path.open("a") as f:
            f.write(json.dumps(score)+"\n")
        (out/f"predictions_{step:06d}.json").write_text(json.dumps(dict(frames=selected,predictions=predictions,full_chunks=np.asarray(chunks).tolist())))
        print(f"CHECKPOINT step={step} critical_score={score['critical_score']:.4f} gripper_mae={score['critical_gripper_mae']:.3f} elapsed={time.monotonic()-began:.1f}s",flush=True)
        del policy,pre,post,predictions,chunks
        gc.collect()
    ranking=sorted(all_scores,key=lambda r:r["critical_score"])
    (out/"ranking.json").write_text(json.dumps(dict(protocol="protocol.json",ranking=ranking,
        elapsed=time.monotonic()-began,task_success="not evaluated",deployed_model_changed=False),indent=2))
    print("RANKING "+json.dumps([dict(step=s["step"],critical_score=s["critical_score"],gripper_mae=s["critical_gripper_mae"]) for s in ranking[:5]]),flush=True)


if __name__=="__main__":
    main()
