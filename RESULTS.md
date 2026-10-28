# Measured results (all single-seed; synthetic tokens; 1 vCPU, fp32; no codec, no audio I/O)

## A. Smoke task, closed loop (no teacher forcing), 1600 training steps, ~1.0-1.5M params
Fixed leading silence (original task), 200 examples (K=4 re-evaluated on 300):
| model | exact response | acoustic-mode acc | backbone calls/frame | refiner/depth passes/frame |
|---|---|---|---|---|
| BPM K=4 (transformer) | 1.00 | 1.00 | 0.28 | 1.13 |
| BPM K=8 (transformer) | 0.685 | 0.999 | 0.17 | 0.69 |
| BPM K=4 (GRU, 1.0M params) | 0.92 | 1.00 | 0.28 | 1.13 |
| AR baseline (K=1 + AR depth) | 1.00 | 1.00 | 0.97 | 2.91 |
K=8 and GRU were still improving / under-trained at the step budget; their gaps are NOT clean architecture conclusions.

## B. Jittered onset (leading silence 2-5 frames), 300 examples
| inference policy | exact response | max onset delay (frames) |
|---|---|---|
| BPM K=4, default (revision + activity re-plan) | 1.000 | 0 |
| BPM K=4, activity re-plan off | 0.813 | 1 |
| BPM K=4, activity re-plan and revision off | 0.800 | 1 |
| AR baseline | 1.000 | 0 |
The fixed-timing task (A) hid this effect: anchors always fell conveniently relative to end of speech.
Cause of the ~19-point drop was not diagnosed.
Refine steps 1/2/4/8 gave identical accuracy on this toy (acoustic mapping is trivially deterministic): not transferable.

## C. Wake latency at real sizes (random weights; scripts/benchmark.py; 120 frames after warm-up)
| model | params | wake p50 / p95 / max (ms) | model RTF | calls/frame |
|---|---|---|---|---|
| BPM K=4, S=4, ultra-small | 40.0M | 51.4 / 55.8 / 56.7 | 0.161 | 0.25 |
| BPM K=4, S=1, ultra-small | 40.0M | 25.9 / 40.5 / 44.2 | 0.085 | 0.25 |
| AR (K=1, depth AR 2 layers), ultra-small backbone | 38.0M | 33.3 / 41.2 / 45.3 | 0.425 | 1.00 |
| BPM K=4, S=4, small | 80.3M | 118.3 / 142.6 / 142.7 | 0.372 | 0.25 |
One wake must finish before the first frame of the new block is due (80 ms budget): fine at 40M here, NOT at 80M on this CPU.
The AR depth module in bpm/modules.py is not KV-cached, so its time is pessimistic.

## Known issues
* `train_seconds` in results.json is wrong for resumed runs.
* ckpt for runs/ar_jit has no results.json (killed during eval); use scripts/evaluate.py.
* Mimi adapter untested; no real-audio data or training; no text auxiliary loss implemented.
