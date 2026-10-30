# SO-101と模倣学習による卓上物体搬送

Physical AI応用1講座 最終課題のコードです。人の実演からACTを学習し、RGB画像と関節位置からSO-101の動作を推論します。対象タスクはオレンジ色のハンドグリップを丸皿へ移すことです。

## 実装の位置づけ

Hugging Faceの[LeRobot](https://github.com/huggingface/lerobot)を基盤にしています。このリポジトリには本課題で追加したコード・学習設定・診断処理を収録し、LeRobot本体は次のコミットを外部依存として使用します。

`6f5bb4d4a49fbdb47acfeaa2c190b5fa125f645a`

基盤を独自開発したものではありません。LeRobot由来のコードのライセンスはApache-2.0です（LICENSE参照）。

## 構成

| ファイル | 用途 |
| --- | --- |
| `src/lerobot/scripts/lerobot_so101_act_inference.py` | 入力・モデル・校正情報の確認、読み取り専用推論、制限付き単発指令 |
| `outputs/training_jobs/train_act_holdout.py` | エピソード分割、学習用データだけでの正規化、検証付きACT学習 |
| `outputs/training_jobs/act_so101_20k_holdout.json` | CPU・2万更新の学習設定 |
| `outputs/guarded_act_rollout.py` | 20・40・90秒の制限付きモデル実行とログ保存 |
| `outputs/return_act_start.py` | 記録した動作経路を低速で逆行して開始姿勢に戻す処理 |
| `outputs/teleop_matching_monitor.py` | 温度・過負荷の記録と、その他のステータス異常の確認 |
| `outputs/act_diagnosis/compare_checkpoints.py` | 保存したモデルの把持・解放局面の比較 |
| `outputs/act_diagnosis/raw_temperature_probe.py` | SDKを使わない読み取り専用温度診断 |
| `outputs/act_diagnosis/packet_audit.py` | 受信パケットの記録・応答長の確認 |
| `outputs/act_diagnosis/check_sdk_reply_validation.py` | 合成パケットによるSDK診断（実機アクセスなし） |
| `patches/cpu-dataloader.patch` | データワーカー0でのCPU学習用変更 |

## 学習・実機での確認結果

- 実演：10エピソード、5,883フレーム、30 fps。
- 学習：エピソード0～8。検証：9。CPU・バッチ8で20,000更新を完了。
- ACT：画像640×480＋6軸位置を入力し、50時刻分の6軸目標位置を推論。
- 通常の検証誤差が最小のモデルは5,000更新時点。把持・解放の局面別比較を経て、17,500更新時点も実機評価に使用。
- 40秒の動作では1,138回の指令を送信。90秒設定の試行は約41秒、別試行では約81秒で制御更新の遅延を検出して停止。
- 対象物の置き直し・解放補助など人の介入を含む試行があり、安定した完全自律搬送や成功率は確立していません。

## セットアップ

本リポジトリを取得したディレクトリを`so101-imitation-learning`とした例です。別途LeRobotの指定版を取得し、追加コードを配置します。元の開発作業フォルダへ上書きせず、専用のコピーを使ってください。

```bash
git clone https://github.com/huggingface/lerobot.git lerobot-runtime
git -C lerobot-runtime checkout 6f5bb4d4a49fbdb47acfeaa2c190b5fa125f645a
cp so101-imitation-learning/src/lerobot/scripts/lerobot_so101_act_inference.py lerobot-runtime/src/lerobot/scripts/
cp -R so101-imitation-learning/outputs lerobot-runtime/
cd lerobot-runtime
git apply ../so101-imitation-learning/patches/cpu-dataloader.patch
python -m pip install -e '.[feetech]'
```

Python・PyTorch・動画デコーダ等の実行環境はLeRobotの指定版に合わせて用意します。実験ではDocker内のPython環境を使用しました。上記セットアップは新しい環境での通し実行を検証していません。

## データと学習

撮影データ・モデル重み・端末の校正情報・認証情報は含めません。学習の再実行には自分の実演データが必要です。データの既定配置は`/datasets/so101_c270_pick_place_05`、画像キーは`observation.images.top`です。別のデータを使う場合は設定・各スクリプトの既定値を変更してください。分割は10エピソード構成を前提としています。

```bash
mkdir -p outputs/training_jobs
python outputs/training_jobs/train_act_holdout.py \
  --config_path=outputs/training_jobs/act_so101_20k_holdout.json
```

局面別比較には、エピソード9の関節・行動データをJSON配列にした`outputs/act_diagnosis/episode9.json`、学習によって出力される正規化統計、各モデルが必要です。生データを含めていないため、そのままでは比較を再実行できません。

## 読み取り専用での確認

実験のデバイス指定はfollowerが`/dev/ttyACM1`、カメラが`/dev/video0`、校正IDが`follower_arm`です。実機に合わせて変更し、保存済みの校正情報とモデル・データを配置します。次のコマンドは位置指令を送りません。

```bash
python -m lerobot.scripts.lerobot_so101_act_inference \
  --mode observe --robot-id follower_arm --position-mode range_m100_100
```

## 実機動作の条件

`guarded_act_rollout.py`は実際に動作指令を送ります。実験の最終設定では、利用者の指定により**ソフト側の温度・過負荷停止を無効化し、記録のみ**としています。本体の保護設定は変更しませんが、これだけで保護や安全性を保証するものではありません。他のステータス異常、追従誤差、校正範囲、制御更新の遅延による停止は維持しています。

実機の電源・配線・作業範囲・校正・現在の姿勢を確認し、人の手を作業範囲から離したうえで操作してください。速度上限50、開始位置からの範囲180は、ドライバの**校正後の位置単位**であり、50度/秒・180度を意味しません。

実験で使用した呼び出し例（実機が動きます）：

```bash
python outputs/guarded_act_rollout.py \
  --model outputs/train/act_so101_c270_pick_place_05_cpu_20k_holdout_002/checkpoints/017500/pretrained_model \
  --output outputs/act_diagnosis/example_run \
  --target-speed 50 --position-envelope 180 \
  --relative-limit 5 --elbow-relative-limit 8 --seconds 40
```

終了時は測定した位置を目標として保持し、トルクを維持します。復帰処理も実機を動かすため、記録した経路や周囲の物体配置を確認してから使用します。

## 限界

少数の実演・単一の検証エピソードを用いた開発です。オフライン誤差は成功率ではなく、モデル選択にも同じ検証エピソードを使っています。未知の物体・配置への汎化や、長時間動作の安定性は未確認です。温度の瞬間的な上振れと制御更新の遅延の原因も未確定です。
