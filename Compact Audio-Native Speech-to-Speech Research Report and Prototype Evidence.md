# Compact Audio-Native Speech-to-Speech: Research Report and Prototype Evidence

Prepared 28 October 2026. Evidence labels used throughout: **\[V\]** verified by me against a primary source this session; **\[S\]** developer or secondary-source claim, not independently verified; **\[M\]** measured by me in a sandbox (1 vCPU, fp32, synthetic tokens, no codec); **\[E\]** estimate from stated equations; **\[H\]** hypothesis. Nothing labelled \[M\] is evidence about real speech quality.

## 1. Executive summary and direct answer

**Question.** Can a compact model map incoming audio to outgoing audio, with low latency and parallel (non-autoregressive) generation, and show useful conversational behaviour?

**Answer, in three parts.**

1. **Mechanically, yes.** A streaming backbone that predicts K=4 future frames per call, plus a masked parallel acoustic refiner, trains and runs correctly. On a synthetic token task it matched a Moshi-style frame-by-frame AR baseline (100% vs 100% exact responses) while making 0.28 instead of 0.97 backbone calls per frame \[M\].
2. **Parallelism buys fewer sequential passes, not fewer FLOPs.** At the 40M size the parallel refiner costs about 2.9x the FLOPs of a KV-cached AR depth module (3.3 vs 1.16 GFLOPs per audio second) but needs about 6.4x fewer sequential passes (15.6 vs 100 per second) \[E\]. Block commitment also has a real cost: without a turn-end/activity trigger, exact-response accuracy fell from 100% to about 80% on a jittered-timing task \[M\].
3. **The broad claim is not supported by any evidence I found.** I found no evidence that a 20-100M model can do useful open-domain conversation. Even PersonaPlex (7B) scores low on spoken QA in the DiffuPlex tables (roughly the low 30s, as I read them \[V\]). Dessa's claim of learning from speech alone is unverified \[S\]. The defensible small-scale target is a **narrow-domain** conversational agent on frozen Mimi tokens.

**Recommended architecture:** Architecture C-lite, "Block-Parallel Multi-rate" (BPM): 12.5 Hz causal backbone, K=4 block head, masked parallel acoustic refiner, frozen Mimi codec, activity-triggered re-planning. Exact configuration in section 8.

**What is not done:** no real-audio training, no text auxiliary loss, no human or public-benchmark evaluation, no measured time-to-first-audio, and the Mimi adapter has never been run. Sections 12 and 16 list the gaps.

## 2. Verified literature review

- **MOSS-Speech** (arXiv 2510.00499; ICLR 2026 per the brief). Model card: about 9B parameters, built on Qwen3-8B, standard autoregressive generate loop \[V\]. The paper abstract states speech is produced without text guidance; the capability depends on a pretrained text LLM backbone \[V, abstract + card\]. I did not verify its license or any latency figure.
- **Moshi and Mimi** (Kyutai; arXiv 2410.00037). Mimi: 24 kHz audio, 12.5 Hz frames, 8 codebooks x 2048 entries, so 8 x 11 bits x 12.5 = 1,100 bps \[V, repository and model card\]. Moshi trains with a text "inner monologue" stream and generates text tokens alongside audio \[V\]. The paper reports roughly 160 ms theoretical / 200 ms practical latency \[S, not re-checked\]. I did not verify Mimi's parameter count.
- **PersonaPlex-7B-v1** (NVIDIA). Moshi-based, 7B, text and voice prompts, trained on a Fisher-derived corpus plus synthetic data \[V, model card\]. One secondary site describes it as a 70B MoE; that contradicts the model card and is discarded.
- **Dessa** (Block). Announced about 23 October 2026. All coverage I found is secondary; Block's own page could not be fetched. Claims: full-duplex, trained only on speech. No parameter count, dataset size, latency or benchmark has been published \[S\]. I make no inference about its architecture or size.
- **MaskGCT** (ICLR 2025; arXiv 2409.00750). Two-stage masked generative TTS over semantic and acoustic codec tokens, non-autoregressive. The total output length must be specified before generation \[V\]. That is precisely the response-length problem for dialogue.
- **DiffuPlex** (arXiv 2610.12214, October 2026). The closest prior work. As I read it: Moshi and PersonaPlex wake the backbone every 80 ms; a rolling masked-diffusion draft yields about 1.5x backbone speedup; the depth transformer stays autoregressive over codebooks; masking future frames hurts naturalness when the same-frame first-codebook token is not available (UTMOSv2 2.65 vs 2.89) \[V\]. It informed the refiner design here.
- **VocalNet-MDM** (arXiv 2602.08607). Masked-diffusion speech LLM claiming decoding speedups \[S, secondary summary only\].
- **DuplexMamba** (arXiv 2502.11123) and **Mamba-HuBERT** (ACL 2026). Mamba works for duplex speech-to-text and for speech SSL/streaming ASR \[V, abstracts\]. Neither demonstrates speech *generation*, so SSM viability for audio output is unproven.

## 3. Existing-model comparison

| System | Params | Text at inference | Generation | Simultaneous listen/speak | Codec / rate | Measured latency | Code / weights | Main limits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Moshi | \~7B temporal + depth transformer (exact split not verified) | Emits text tokens; trained with text | AR frames + AR depth | Yes | Mimi, 12.5 Hz, 8x2048 \[V\] | \~200 ms reported \[S\] | Code and weights public; weight license reported CC-BY-4.0, confirm | Large; text-guided |
| MOSS-Speech | \~9B \[V\] | Not required for speech output (paper) | AR | Not verified | Not verified | Not found | HF weights public; license not verified | Large; depends on text LLM |
| PersonaPlex-7B | 7B \[V\] | Text prompt used | AR (Moshi-based) | Yes | Mimi | Not verified | HF, gated; license per card | Low spoken-QA scores in DiffuPlex tables |
| Dessa | Unpublished | Claimed none \[S\] | Unpublished | Claimed \[S\] | Unpublished | Unpublished | Not released | Nothing verifiable |
| MaskGCT | Not verified | Text in (TTS) | NAR masked | No | Semantic + acoustic codec | Not applicable | Public | Needs length up front; TTS only |
| DiffuPlex | Applies to Moshi-class | Inherited | Rolling masked draft + AR depth | Yes | Mimi | \~1.5x backbone speedup \[V\] | Not verified | Quality loss from masked futures |
| DuplexMamba | Not verified | Output is text | AR | Yes (text out) | Not applicable | Not verified | Public per paper | No speech output |

## 4. Three competing architectures

All three use frozen Mimi tokens (Q=8, V=2048, 12.5 Hz). Parameter counts below come from the budget tool in the prototype and match the formulas shown. Only C is implemented.

### A. Parallel audio-to-audio generator \[not implemented\]

```
user tokens -> chunked bidirectional encoder -> latent z -> length predictor -> masked decoder (whole response window, S passes) -> Mimi
```

- Strength: fewest sequential steps per response.
- Fatal problem for conversation: it must predict response length before generating and cannot start speaking until the whole window is decoded. Interruptions force a full re-decode. Time-to-first-audio is at least the full refinement of the window. **I rank it last for streaming.**

### B. Recurrent conversational model \[partially implemented: GRU backbone option\]

```
user frame -> embed -> GRU/SSM state (L x d) -> block head -> refiner -> Mimi
```

- State is L x d numbers: 8 layers x 384 = 3,072 values (6 KB in fp16) versus a 3 MB KV cache for the attention version \[E\].
- GRU backbone with 8 layers, d=384: 6 d^2 L = 6 x 147,456 x 8 = 7.08M parameters, 33.0M total \[exact count\].
- In smoke tests the 1.0M-parameter GRU reached 92% vs 100% for the attention model, but its loss was still falling, so this is **not** a fair verdict \[M\].
- Open risk: whether a fixed-size state keeps speaker identity, dialogue history and uncertainty over minutes. Untested.

### C. Multi-rate hybrid, implemented as BPM \[implemented\]

```
user u_p (Q tokens) --\
                       +--> embed sum --> causal backbone (12.5 Hz, KV cache) --> h_p
own a_p (Q tokens) ---/

h_p --> block head (K=4 future frames, in parallel): cb0, speak/silent, user-cb0 (aux)
h_p + cb0[K] --> refiner: masked parallel decode of cb1..cb7 for K frames (S passes)
  --> Mimi decoder (frozen) --> audio
Controller: re-plan when user activity flips, or user cb0 is surprising (revision), or barge-in cancel
```

**Hypothesis (H):** meaning changes more slowly than acoustic detail, so the backbone can be called every K frames while a small refiner fills in acoustics. **This is not tested here**: the synthetic task does not separate semantic from acoustic timescales.

**Ultra-small configuration (exact counts):**

| Module | Formula | Parameters |
| --- | --- | --- |
| User embedding | Q x V x d = 8 x 2048 x 384 | 6.29M |
| Own embedding | same | 6.29M |
| Backbone, 8 layers | 12 d^2 L = 12 x 384^2 x 8 | 14.16M |
| Block heads | d x K x d + 2 x d x V + small | 2.17M |
| Refiner (d=256, 4 layers) | 3.67M acoustic embed + 0.52M cb0 embed + \~3.2M blocks + 3.67M output heads | 11.13M |
| **Total** |  | **40.04M** |

## 5. Limits of non-autoregressive generation

Four properties that are often conflated: parallel prediction **within a chunk**, **incremental** generation of successive chunks, **continuous streaming** output, and **full-duplex** interaction. BPM has the first two. DiffuPlex and BPM both keep dependence *between* blocks autoregressive.

| Approach | Remaining dependencies | Length handling | Passes | Starts before full reply known | Handles new user audio | Interruptible |
| --- | --- | --- | --- | --- | --- | --- |
| Whole-response masked decode (A) | All tokens refined jointly | Must predict length first | S over entire window | No | Re-decode everything | Only by full re-decode |
| Multi-frame prediction, K future frames | cb0 across frames conditionally independent | Speak/silent flag per frame; ends when predicted silent | 1 backbone + S refiner per block | Yes | Re-plan at anchor | Drop unplayed frames |
| Iterative masked refinement of acoustics | cb1-7 conditioned on same-frame cb0 and block | Inherited from cb0 | S per block | Yes | Via re-plan | Yes |
| Diffusion / consistency | Depends on sampler | Usually fixed window | Few-step distilled | Window-bound | Window re-run | Window-bound |

**Measured behaviour \[M, synthetic\].** With activity re-planning on, onset delay was 0 frames. With it off, delay was at most 1 frame but exact-response accuracy dropped to about 80%. The mechanism is undiagnosed; my hypothesis is committing a block before the user's turn ended. An earlier fixed-timing task showed zero cost and was misleading, because block anchors always fell conveniently. **Recommendation:** keep AR across blocks, parallel within blocks, K=4 default, with a required activity trigger.

**Time-to-first-audio** is determined by wake time plus detection delay, not by K. K controls throughput and how many frames are committed ahead (up to K x 80 ms = 320 ms of unplayed audio that must be droppable on barge-in).

## 6. Audio representation

| Candidate | Rate | Codebooks / bitrate | Causal / streaming | Notes |
| --- | --- | --- | --- | --- |
| **Mimi** \[V\] | 12.5 Hz | 8 x 2048, 1.1 kbps | Yes, 80 ms frames | Semantic distillation into first codebook; available with Kyutai weights |
| EnCodec (recalled, not re-verified) | 75 Hz at 24 kHz | Up to 32 x 1024 depending on bandwidth | Streaming-capable | 6x more positions per second than Mimi |
| SoundStream (recalled) | \~50-75 Hz | RVQ | Streaming | Original reference; no public Google weights I verified |
| SpeechTokenizer (recalled) | 50 Hz | 8 RVQ, first level semantic | Not designed for causal streaming | Semantic + acoustic hybrid |
| Continuous latents | Variable | None | Possible | No discrete targets; harder to mask; not evaluated |

**Recommendation: frozen Mimi.** The decisive factor is frame rate: 12.5 Hz means the fewest backbone positions per second of any candidate here, and the 80 ms frame is the codec's native step, so chunk sizes of 20, 40 or 160 ms are not available without changing codec. I have **not** measured Mimi encoder or decoder latency, reconstruction quality, speaker or emotion preservation. These are required first measurements.

## 7. Training data, licensing and staged training

### Data (licences verified only where marked)

| Corpus | Content | Licence / access | Fit |
| --- | --- | --- | --- |
| Fisher English (LDC2004S13, LDC2005S13) \[V catalogue\] | Two-channel telephone conversations, 8 kHz | LDC user agreement; not freely redistributable; commercial use depends on membership terms | Real conversational turn-taking; 8 kHz limits fidelity |
| Switchboard-1 (LDC97S62) \[V\] | Telephone conversations, 8 kHz | LDC | Small; same limits |
| Seamless Interaction \[V\] | 4,000+ hours of dyadic conversations with separate channels | CC BY-NC 4.0: non-commercial only | Good for research; blocks commercial use |
| Synthetic dialogues from a TTS model | Generated question/answer pairs | Depends on the TTS model's licence (not verified) | Needed for the narrow-domain experiment; controls content |
| CANDOR and others | Not verified | Not verified | Do not use until checked |

None of these licences should be assumed to allow redistribution of derived tokens or models. **Check each agreement before training.** Paired audio does not by itself supply factual knowledge or reasoning; expect conversational *form* (timing, backchannels) to be learnable, and *content* to need text-derived supervision, synthetic data, or a narrow domain.

### Stages

1. **Codec.** Reuse frozen Mimi. Do not retrain. Measure reconstruction, speaker preservation and latency first (not yet done).
2. **Understanding / state.** Predictive-coding auxiliary: predict the user's next K cb0 tokens (implemented, weight 0.1). Optional: masked-frame modelling; text-aligned auxiliary loss (not implemented).
3. **Response learning.** Two-channel dialogues, agent channel as target.
4. **Parallel output.** Masked refiner for cb1-7 with a cosine mask schedule (implemented).
5. **Streaming / duplex fine-tune.** Add overlapped speech, backchannels, interruptions, speaker changes (not implemented; the smoke task has no overlap).
6. **Distillation and deployment.** Quantization-aware training and post-training INT8 cut **storage and memory traffic**; distillation and structured pruning cut **compute**; LoRA cuts **fine-tuning memory** only. None is implemented here.

### Losses (as implemented)

For an anchor p, with targets for frames p+1..p+K:

L = CE(cb0) + CE(speak flag) + 0.1 x CE(user cb0) + CE(masked cb1..cb(Q-1) | cb0, h\_p, visible acoustic tokens)

The refiner draws a mask ratio r = cos(pi/2 x u), u uniform in \[0,1\], and computes loss only on masked tokens. **Essential:** cb0, speak flag, refiner. **Optional:** user-cb0 auxiliary (kept for revision, not ablated). **Experimental, not implemented:** text inner-monologue loss, speaker-consistency loss, timing loss. Add a loss only when an ablation shows a gain.

### Compute estimate \[E\]

Training FLOPs \~ 6 x N\_active x frames. For the ultra-small model, N\_active \~ 30M (backbone, heads, refiner blocks), and 100,000 steps x 32 sequences x 1,024 frames = 3.28 billion frames, about 6e17 FLOPs. At an *assumed* sustained 20 TFLOP/s on a single accelerator that is roughly 8 hours. The same schedule is about 73,000 audio-hours of exposure: on a 2,000-hour corpus that is \~36 epochs, so **overfitting is the main risk**, not compute.

## 8. Recommended architecture and parameter budgets

**Recommendation: BPM, ultra-small, K=4, S=4, attention backbone.** Configuration: Q=8, V=2048, d=384, 8 layers, 6 heads, window 250 frames (20 s), refiner d=256 x 4 layers. **40.04M parameters** (breakdown in section 4).

**Why it beats A and B.** A cannot stream and needs a length oracle. B has the smallest state but SSM and GRU speech *generation* is unproven (section 2) and my GRU run was inconclusive. C keeps AR across blocks (stoppable, revisable) and parallelism where the dependence is weakest.

| Config | Total params | Backbone | Embeddings | Refiner | fp32 / fp16 / int8 weights | State (fp16) | GFLOPs per audio s (ingest / generate / total) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ultra-small | 40.04M | 14.16M | 12.58M | 11.13M | 160 / 80 / 40 MB | 3.06 MB | 0.39 / 2.93 / 3.32 |
| Small | 80.34M | 37.76M | 16.78M | 22.66M | 321 / 161 / 80 MB | 6.12 MB | 1.02 / 9.10 / 10.12 |
| Medium | 185.31M | 113.27M | 25.16M | 41.36M | 741 / 371 / 185 MB | 12.24 MB | 2.99 / 20.96 / 23.94 |

Equations: weight bytes = parameters x bytes per parameter; KV state = layers x 2 x (window-1) x d x bytes; ingest FLOPs = 12.5 x (2 x 12 d^2 L + 4 L d x window); refiner FLOPs per block = S x (2 x 12 r^2 x layers x K x Q + head). Real memory is higher than weight storage because of activations, temporary buffers, the KV cache, the Mimi codec, and runtime overhead. **I did not measure peak RAM or VRAM, and the codec is excluded from every number above.**

**Sequential passes per second of audio \[E\]:** BPM K=4, S=4: 3.1 backbone + 12.5 refiner = 15.6. AR (K=1, 7 depth steps per frame): 12.5 + 87.5 = 100. Backbone weight traffic at fp16: about 102 MB/s for BPM vs 397 MB/s for the AR variant, which matters when decoding is memory-bandwidth bound.

**Scale caveat.** I found no evidence that these sizes support open-domain conversation. Plan on a narrow task first.

## 9. Streaming inference algorithm

```
state = None; pending = []; plan = [silent frame 0]
for each 80 ms frame t:
    if plan is empty:                              # WAKE
        h, state = backbone(pending frames, state) # one call for the whole chunk
        blk = head(h[-1])                          # K frames: cb0, speak flag, user-cb0 prediction
        blk.cb1..7 = refiner.decode(h[-1], blk.cb0, S)   # S parallel passes
        plan = blk frames; pending = []
    own = plan.pop(0); play(codec.decode(own))     # frames already played are never edited
    u = receive user frame t
    if plan and surprise(u.cb0 under stored prediction) > tau: plan.clear()   # revision
    if plan and user_active flips:                  plan.clear()              # turn start/end
    pending += (u, own)
barge_in(): plan.clear()                            # next frame wakes with fresh context
```

Working code, unit-tested on token streams (chunked streaming equals the full forward pass to 1e-4 for both KV-cache and GRU backbones).

## 10. Latency budget

**Definitions.** *Time to first useful audio* = from the last non-silent user sample (reference VAD) to the first non-silent output sample at the audio device. *Model wake time* = wall-clock for one backbone call plus planning plus refinement. *Model RTF* = total wake time / audio duration. Report mean, median, p95 and max with warm-up excluded.

TTFA = t\_detect + t\_encode + t\_wake + t\_decode + t\_buffer, plus network and OS terms where relevant.

| Component | Value | Status |
| --- | --- | --- |
| Capture and detect end of speech | at least one 80 ms Mimi frame | structural; VAD delay not measured |
| Mimi encode | not measured | required |
| Wake (ultra-small, K=4, S=4) | p50 51.4 ms, p95 55.8 ms | \[M\] 1 vCPU, fp32, random weights |
| Wake (ultra-small, S=1) | p50 25.9 ms, p95 40.5 ms | \[M\] |
| Wake (AR baseline, per frame) | p50 33.3 ms, p95 41.2 ms | \[M\], uncached depth module so pessimistic |
| Wake (small, 80M, S=4) | p50 118.3 ms, p95 142.6 ms | \[M\] |
| Mimi decode, device buffer, network | not measured | required |

Model-only RTF \[M\]: 0.16 (BPM ultra-small, S=4), 0.085 (S=1), 0.43 (AR), 0.37 (BPM small). **A wake must complete inside the 80 ms frame budget or the first frame of a new block is late.** That holds for the 40M model on this CPU and fails for the 80M model. Sub-500 ms time-to-first-audio is an experimental target; it is **not** demonstrated. The \~500 ms budget minus one detection frame and a 51 ms wake leaves roughly 370 ms for encode, decode, buffering and network, which is plausible but unmeasured.

## 11. Evaluation and ablation protocol

| Baseline | Purpose |
| --- | --- |
| A. STT, LLM, TTS cascade | Cost/latency and quality reference |
| B. Matched AR speech-to-speech (K=1 plus AR depth) | Isolates the parallel-decoding effect (built here) |
| C. Public parallel/NAR baseline (e.g. DiffuPlex-style draft on an open model) | External reference; availability unverified |
| D. Ablated BPM | Component contribution |

Metrics: WER of generated speech via a pinned ASR; spoken-QA accuracy; semantic relevance; naturalness (UTMOSv2 plus human MOS, since automatic scores are limited); speaker similarity; emotion and non-speech preservation; multi-turn coherence; response timing; interruption and overlap handling; RAM/VRAM; robustness to noise, accents and partial input.

Ablations: AR vs parallel; token rate; recurrent vs attention backbone; context window; refinement steps S; model size; with/without text supervision; chunk size K and lookahead. Report mean, median and tail latency with warm-up, batch size, precision, hardware and software versions. Public benchmarks (VoiceBench, Full-Duplex-Bench as used in DiffuPlex) should be used where possible; their licences and suitability were not checked.

## 12. Prototype: status and measured results

The repository (bpm\_s2s.zip, delivered with this report) has 23 passing unit tests. **Working and tested:** attention with KV cache, GRU backbone, block head, masked refiner, AR depth baseline, streaming engine, budget calculator, synthetic task, train/evaluate/benchmark scripts. **Untested:** the Mimi adapter (no access to huggingface.co from the sandbox). **Not implemented:** real-audio loader and training loop, text auxiliary loss, human and benchmark evaluation.

**Smoke task, fixed timing, closed loop, 1,600 steps, single seed \[M\]:**

| Model | Params | Exact response | Backbone calls per frame |
| --- | --- | --- | --- |
| BPM K=4 | 1.40M | 1.00 | 0.28 |
| BPM K=8 | 1.46M | 0.685 | 0.17 |
| BPM K=4, GRU | 1.00M | 0.92 | 0.28 |
| AR baseline | 1.34M | 1.00 | 0.97 |

K=8 and GRU were still improving at the step budget, so their gaps are not clean conclusions.

**Jittered onset, 300 examples \[M\]:**

| Policy | Exact response |
| --- | --- |
| BPM default (revision plus activity re-plan) | 1.000 |
| Activity re-plan off | 0.813 |
| Both off | 0.800 |
| AR baseline | 1.000 |

The fixed-timing task had hidden this effect. Refinement steps 1, 2, 4 and 8 gave identical accuracy because the toy acoustic mapping is trivially deterministic; that result does not transfer to real audio.

## 13. Reproducibility checklist

- Python 3.12.3; PyTorch 2.14.1 (CPU use); NumPy, PyYAML, pytest. 1 vCPU, 3 GB RAM sandbox.
- Commands: pip install -r requirements.txt; python -m pytest tests -q; ./run\_smoke.sh bpm (or ar, bpm\_k8, bpm\_gru, bpm\_jit, ar\_jit); python scripts/evaluate.py --ckpt runs/bpm\_jit/ckpt.pt; python scripts/benchmark.py --config configs/ultra\_small.yaml.
- **Single seed.** No confidence intervals. Three or more seeds are needed before any comparison is trusted.
- Known issues: train\_seconds is wrong for resumed runs; the ar\_jit run lacks a results file (use scripts/evaluate.py); RESULTS.md in the repo lists all numbers.
- Data and weights: licences must be checked before use (section 7).

## 14. Principal risks and fallbacks

| Risk | Evidence | Fallback |
| --- | --- | --- |
| Semantic capacity at 20-100M | No supporting evidence found; 7B models score low on spoken QA | Narrow domain; text-aided distillation from a larger teacher |
| Block commitment hurts turn-taking | About 19-point accuracy loss when the trigger is removed \[M, toy\] | Mandatory activity/turn-end trigger; reduce K to 2 |
| Parallel acoustics degrade quality on real audio | DiffuPlex reports lower naturalness with masked futures \[V\] | AR cb0 plus parallel acoustic refiner only; more refinement steps |
| Wake time exceeds the frame budget | 80M model: 118 ms p50 on 1 vCPU \[M\] | Smaller model, lower S, INT8, GPU |
| Codec latency and quality unmeasured | Not measured | Measure first; alternate codec |
| Overfitting on small corpora | \~36 epochs on 2,000 h \[E\] | Augmentation, synthetic data, early stopping |
| Licence limits | Fisher/Switchboard need LDC; Seamless is non-commercial \[V\] | Licensed or synthetic data |

## 15. Milestones

1. Measure Mimi encode/decode latency, reconstruction and speaker preservation on target hardware.
2. Build the TTS-generated narrow-domain dialogue set and the real-audio loader.
3. Run the decisive experiment (section 16) with at least three seeds.
4. If it passes: scale to 80M, add real conversational data, text-auxiliary ablation, duplex fine-tuning.
5. End-to-end time-to-first-audio and barge-in measurement on a named machine.
6. Human evaluation against the cascade and AR baselines.

## 16. The decisive experiment and final answer

**Experiment.** Train matched \~50M models (AR; BPM K=4, S=4; BPM with S=1) on 100k+ TTS-generated closed-domain spoken Q&A dialogues in several voices, encoded with Mimi, evaluated on a held-out compositional split, with at least three seeds. ASR on the generated audio scores answers.

**Pass (all required):**

1. BPM answer accuracy is at least 90% of AR accuracy and at least 80% absolute.
2. Generated-speech WER is within 3 points (absolute) of AR.
3. Naturalness (UTMOSv2) within 0.15 of AR.
4. Model compute per audio second at least 1.8x lower than a KV-cached AR baseline on the named machine.
5. Median time-to-first-audio at most 500 ms and p95 at most 700 ms, measured as defined in section 10, including codec and playout buffer.
6. Unplayed audio stops within 200 ms of a barge-in trigger.

**Falsified if** criterion 1 fails at equal parameters and data. **Fallback:** keep the AR cb0 stream and parallelise only the acoustic codebooks.

**Final answer to the research question.**

1. **Architecture:** BPM (Architecture C-lite).
2. **Configuration:** d=384, 8 layers, K=4, S=4, refiner d=256 x 4 layers; 40.04M parameters.
3. **Rationale:** streams, interruptible, and measured to cut sequential calls about 3.5x on the toy task; A cannot stream; B is unproven for generation.
4. **Minimum data and training:** a few thousand hours of two-channel dialogue plus a TTS narrow-domain set; about 6e17 training FLOPs \[E\].
5. **Inference memory:** about 80 MB fp16 weights plus 3 MB state, excluding the codec \[E\]; no hardware requirement is claimed.
6. **Principal unresolved risk:** whether the model carries enough semantic content at this size, followed by block-commitment quality on real audio.
7. **Smallest experiment:** the one above.
8. **Pass/fail:** the six criteria above.
9. **If it passes:** scale to 80M and add real conversational data and duplex fine-tuning.
10. **If it fails:** AR cb0 with parallel acoustic decoding, or a narrower task.
