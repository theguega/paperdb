# **MINERVA: How Small Can a Manipulation Policy Be and Still Solve LIBERO?** 

Kohei Sendai<sup>1</sup> , Tatsuya Matsushima<sup>2</sup> , Yusuke Iwasawa<sup>3</sup> 



<!-- Start of picture text -->
100<br>Search for the smallest policy that still solves the task set MINERVA (ours) TurboVLA π 0.5 (LeRobot)<br>iterate:  train  →  evaluate  →  shrink & redesign<br>OpenVLA-OFT<br>0.54M, 95.1% π 0<br>distill<br>90<br>attn  →  mixer, vision  ↑ 7,700× fewer parameters, −2.4 pts<br>chunk 8/16/32,<br>TE/BID/RTC, seeds ×3 SmolVLA<br>too far<br>80<br>9.7M 4.9M 0.99M 0.54M 0.24M OpenVLA<br>Octo<br>breaks<br>most deployment-efficient<br>for the fixed task set (LIBERO)<br>70<br>1M 10M 100M 1B 10B<br>total parameters (log scale)<br>LIBERO average success (%)<br><!-- End of picture text -->

Fig. 1. **Left:** we search for the smallest policy that still solves the fixed LIBERO task set, iterating over model scale, capacity allocation, objectives and inference strategies; the search stops at a 0.54M-parameter empirical capacity floor — one step smaller and the benchmark breaks. **Right:** that floor (blue) lands in the same roughly 95–98% band as fine-tuned generalist VLAs two to four orders of magnitude larger (gray, paper-reported: Octo [1], OpenVLA [2], SmolVLA [3], _π_ 0 [4], TurboVLA [5], OpenVLA-OFT [6], as compiled in [3], [5], [6]); the _π_ 0 _._ 5 point is the LeRobot implementation’s reported result [7], [8]. 

**_Abstract_ — Vision-language-action (VLA) models with billions of parameters now dominate the LIBERO manipulation benchmark, yet how much capacity the benchmark actually demands has not been measured. From a deployment standpoint this is the quantity that matters: once a robot’s task set is fixed, the smallest policy that still solves it governs deployment cost — for example, whether the policy fits on an edge device. We present MINERVA (MINimal Efficient Robotic Vision-Action policy), a family of deliberately minimal visuomotor policies and scale it down until it breaks. A 0.54M-parameter policy reaches 95.1% average success over 2,000 rollouts on the four standard LIBERO suites, 2.4 points below the reported LeRobot** _π_ 0 _._ 5 **result from a model 7,700** _×_ **larger; performance saturates near one million parameters and collapses below a quarter million. Across a broad sweep of architectural, training, and inference choices, only action-chunk length and vision allocation consistently exceed a** _±_ **1-point training-seed band. Flow matching shows no detectable advantage over direct L1 regression across three seeds, while regression is up to 3.8** _×_ **faster on GPU. A permutation probe makes the role of task conditioning causal: rewriting the task-ID mapping alone collapses success to near chance, showing that instruction conditioning on standard LIBERO primarily selects among memorized tasks. The same recipe reaches 94.6% over 89** 

> 1Kohei Sendai is with Matsuo-Iwasawa Lab, Graduate School of Engineering, The University of Tokyo, Japan `kohei.sendai@weblab.t.u-tokyo.ac.jp` 

> 2Tatsuya Matsushima is with Matsuo-Iwasawa Lab, The University of Tokyo, `matsushima@weblab.t.u-tokyo.ac.jp` 

3Yusuke Iwasawa is with Matsuo-Iwasawa Lab, The University of Tokyo, `iwasawa@weblab.t.u-tokyo.ac.jp` 

**LIBERO-90 tasks in a single-seed extension, while LIBEROPlus evaluation drops success to 46–56% under perturbations, with near-zero photometric robustness across all tested scales. The 0.54M policy replans at every control step in 5–9 ms per chunk on a laptop CPU — 113** _×_ **faster than SmolVLA and 1,400** _×_ **faster than** _π_ 0 _._ 5 **— with no GPU. We position this study as a first step toward measuring task-specific capacity floors, which can guide the construction or distillation of deploymentefficient policies. Code, recipes and all checkpoints are open sourced here:** `https://github.com/k1000dai/MINERVA` **.** 

## I. INTRODUCTION 

Imitation-learned manipulation policies are growing rapidly: vision-language-action (VLA) models such as OpenVLA (7B) [2], _π_ 0 [4] and _π_ 0 _._ 5 [7] (3–4B) attach action heads to pretrained vision-language backbones and report nearsaturated success on the LIBERO benchmark [9]. The costs follow the size: multi-GPU training, gigabytes of weights, large VRAM for inference. For system integrators who need a manipulation module on a CPU, beside a planner, under a power budget, the practical question is not whether a larger model helps, but how much capacity the task demands. 

This is a deployment-efficiency question in the sense of [10]: optimize for the constraints that bind at deployment rather than during development. Generalization is what a large VLA buys, and it matters — but it is not required everywhere. Once a robot is deployed its task set is typically fixed, and what then governs cost is the task set’s _minimal_ 

_sufficient capacity_ : the smallest parameter count at which it is still solved. If that capacity floor is known, a policy can be built at it — directly, or by distilling a generalist [11] — and run on the edge. Conceptually, such a floor is a property of the task setting, in the spirit of information-theoretic taskcomplexity measures in RL [12], and it will differ across environments, sensing noise and embodiments; but it has to be measured somewhere first. In this work, we use _empirical capacity floor_ to denote the smallest tested model in our policy family that maintains the target success level under the fixed training and evaluation protocol. 

This paper measures that demand for the most widely used benchmark in the field. Recent analyses [13] show that standard LIBERO evaluation is largely a memorization test: policies are trained and evaluated on the same 40 tasks in the same scenes, and the language instruction mostly serves to disambiguate which of the memorized tasks to execute. We take this observation seriously and build MINERVA (MINimal Efficient Robotic Vision-Action policy), a policy family designed to contain nothing beyond what that reading of the benchmark requires: a from-scratch CNN encoder (no pretrained backbone), a 40-entry learned task-ID embedding (no language encoder), and a flow-matching action-chunk head whose token mixer is an MLP rather than self-attention. We then scale the total budget from 9.7M parameters down to 0.09M and observe where performance breaks. 

The headline result is that **0.54M parameters reach 95.1% average success** over the four standard suites, 2.4 points below the reported LeRobot _π_ 0 _._ 5 result from a model 7,700 _×_ larger; 0.99M parameters land within 0.75 points (Table I, Fig. 1). Success saturates near 1M parameters and collapses below 0.25M — and the collapse consistently arrives through the long-horizon suite first, while shorthorizon tasks survive in models far too small to chain subgoals (Fig. 3). 

The small parameter count makes broad design-space exploration cheap enough to be practical. We sweep architectural and training choices including action-chunk length, vision–action capacity allocation, generative versus regression objectives, and action-token mixing, and re-train the key ablation configurations under three seeds. This reveals a _±_ 1-point seed band on the 4-suite average — comparable to many single-run ablation deltas on this benchmark — and leaves only two robust effects: action-chunk length ( _−_ 3 _._ 2 points at chunk 8, _−_ 2 _._ 2 at chunk 32), with short chunks breaking the goal suite and long chunks the long-horizon suite, and vision allocation, where starving the encoder costs a seed-replicated _−_ 1 _._ 41 points. By contrast, flow matching shows no detectable advantage over direct L1 regression across three seeds (+0 _._ 34, inside the band), while regression requires one forward pass instead of ten Euler steps. Selfattention over the 16 action tokens is likewise unnecessary: a token-mixing MLP matches its score with 26% fewer parameters, and reallocating that capacity to vision is what enables the sub-1M models. 

Three further studies delimit what this level of performance is and is not: a _permutation probe_ isolates the causal 

role of task conditioning (rewriting the task-ID mapping alone collapses success from 96.75% to chance level); the recipe survives 2 _._ 25 _×_ the task count (94.6% on LIBERO90 at 0.995M); and a _LIBERO-Plus_ audit prices what the headline omits (46–56% under perturbation; photometric robustness remains near zero at every tested scale). 

Inference-time strategy matters as much as architecture. Replanning every step and averaging overlapping chunk predictions (ACT-style temporal ensembling [14]) outperforms both BID-style candidate selection [15] and soft-inpainting (RTC [16]) approaches, and mode-seeking sampling (initialnoise temperature 0.85) adds up to 2.3 points — but only on sub-1M models, where the learned velocity field is noisiest. The resulting policies are lightweight enough for CPU deployment: 8.9 ms per action chunk on eight laptop CPU threads for the 0.54M model, against 1.0 s for SmolVLA [3] and 12.8 s for _π_ 0 _._ 5 (Table III). 

Contributions: (i) an empirical capacity floor for standard LIBERO: 0.5M parameters, no language encoder and no pretrained perception suffice for 95%, with a parameterscaling curve localizing where capacity stops mattering; (ii) a seed-replicated ablation exposing a _±_ 1-point training-seed band on this benchmark and showing that only chunk length and vision allocation survive it, while flow matching shows no detectable advantage over one-pass regression; (iii) a budget-allocation study showing vision, not the action head, is where parameters pay; (iv) a comparison of inference-time chunking strategies under one protocol; and (v) a permutation probe of task-ID conditioning, a LIBERO-90 extension, and a LIBERO-Plus robustness audit that together delimit what a memorization-level policy does and does not achieve. 

## II. RELATED WORK 

**VLA policies and LIBERO.** LIBERO [9] provides four ten-task suites (Spatial, Object, Goal, Long/“10”) and paired demonstrations, and has become the default benchmark for VLA evaluation. Current leaders are large: OpenVLA (7B) [2] fine-tunes a Prismatic VLM, and OpenVLA-OFT [6] lifts it to 97.1% with an optimized fine-tuning recipe; _π_ 0 [4] and _π_ 0 _._ 5 [7] attach flow-matching action experts to a PaliGemma backbone; Octo (93M) [1] and SmolVLA (450M) [3] are the efficiency-oriented entries, still two to three orders of magnitude larger than our 0.54M model. LIBERO-PRO [13] shows these scores largely measure memorization — perturbing objects, layouts or instructions collapses standard models to near 0% — which motivates our design: if standard LIBERO is a memorization benchmark, its capacity floor is an empirical question, and language should be replaceable by a task index. 

**Efficient VLAs.** A growing body of work attacks VLA inference cost while keeping the vision-language backbone: TinyVLA [17] pairs a sub-1B VLM with a diffusion head and skips generalist pretraining; MiniVLA [18] swaps OpenVLA’s 7B backbone for a 1B VLM; DeeR-VLA [19] adds dynamic early-exit inference; FAST [20] compresses the action token stream; SmolVLA [3] trims the recipe to 450M. Closest in spirit to us, TurboVLA [5] removes the 

large language model from the control pathway entirely and reaches 97.7% on LIBERO with 0.2B parameters at 32 Hz on an RTX 4090. All of these retain a language pathway and pretrained components, and none goes below 0.2B. MINERVA asks the complementary question — not how to make a VLA cheaper, but how much capacity the benchmark itself demands — and its headline model lands more than two orders of magnitude below the smallest of them, at the declared cost of any open-vocabulary capability. The framing follows deployment-efficient RL [10], which optimizes for the constraints that bind at deployment, and task-complexity measurement [12], which quantifies what a task demands independently of any particular algorithm. 

**Small visuomotor policies.** Sub-100M imitation policies predate the VLA wave: ACT [14] (an 80M transformer with action chunking and temporal ensembling) and Diffusion Policy [21] remain strong baselines, and end-to-end CNN policies with spatial-softmax keypoints [22] date to 2016. MINERVA inherits from all three — chunking and ensembling from ACT, a generative action head from Diffusion Policy (flow matching [23] instead of DDPM), keypoint extraction from [22] — but pushes total capacity one to two orders of magnitude lower than prior work while remaining multi-task across 40 tasks. 

**Efficient generative heads.** Our head follows DiT [24] with AdaLN-Zero conditioning and shares the AdaLN projection across blocks. Replacing token self-attention with an MLP mixer follows [25]; depthwise-separable convolutions follow MobileNets [26]; FiLM conditioning [27] injects the task embedding into the encoder. Distillation [11] transfers velocity targets from a larger teacher. 

**Inference-time chunking strategies.** How a chunked policy is _executed_ is an active design axis: temporal ensembling averages overlapping predictions [14]; Bidirectional Decoding (BID) [15] re-samples candidate chunks and selects for coherence with the committed plan; Real-Time Chunking (RTC) [16] softly inpaints the new chunk toward the old plan during flow integration. We evaluate all three, plus initialnoise temperature control, on one model under one protocol. 

## III. METHOD 

_A. Design principle: nothing the benchmark does not demand_ 

Standard LIBERO evaluates on the training scenes, tasks and instructions. If the instruction is used only to identify one of the 40 known tasks, its functional role can be represented by a 40-way task ID. MINERVA replaces the language pathway with a learned embedding table of 40 entries, and spends the entire parameter budget on the two things the benchmark does exercise: visual state estimation and closedloop action generation. This is explicitly a study of standard LIBERO, not a general-purpose VLA; Section VI returns to what is given up. 

## _B. Architecture_ 

**Perception.** Both camera views (agent-view and wrist, 256<sup>2</sup> , resized to 144<sup>2</sup> and random-cropped to 128<sup>2</sup> at train- 

ing, center-cropped at evaluation) pass through one shared from-scratch CNN with a learned view embedding: four stride-2 stages of depthwise-separable residual blocks [26], FiLM-modulated [27] by the task embedding, followed by spatial-softmax keypoint extraction [22] and a linear projection to a per-view feature. A two-layer MLP encodes the 8- D proprioceptive state. The concatenation of view features, state features and the task embedding, over _n_ obs=2 observation steps, forms the conditioning vector _c_ . Observation history matters: at 7M parameters, moving from one to two observation steps gains +6 points on Goal and +4 on Long. 

**Action head.** Actions are generated as chunks _a_ = _a_ 1: _H_ with _H_ =16 by conditional flow matching [23]. Training draws _ε ∼N_ (0 _, I_ ) and a noise level _τ_ , forms the interpolant _a_<sup>_τ_</sup> = _τa_ + (1 _− τ_ ) _ε_ , and regresses the velocity field toward the constant target: 



with _τ_ sampled from a Beta distribution tilted toward high noise. At inference, a chunk is produced by integrating _vθ_ with 10 Euler steps from _ε ∼N_ (0 _, σ_<sup>2</sup> _I_ ), where _σ_ is the noise temperature of Section III-C. The head is a 4-layer DiT [24] in which self-attention over the 16 action tokens is replaced by a token-mixing MLP [25], and the AdaLN-Zero modulation MLP is shared across blocks [28]. The same head can instead be trained as a direct L1 regression (learned queries in, chunk out, one forward pass at inference); Section V- C shows flow matching has no detectable advantage over direct L1 regression across three seeds on this benchmark. A training-only auxiliary head predicts episode progress ( _<_ 9k parameters, dropped at inference). 

**Model family.** Only the two sub-1M models differ purely in width: MINERVA-0.5M (0.54M: CNN widths 32/64/112/160, 48 keypoints, head width 96) and MINERVA1M (0.99M: 40/80/160/224, 64 keypoints, head width 128); MINERVA-5M (4.89M) restores the full-width CNN and head self-attention, and MINERVA-10M (9.66M) also unshares AdaLN. Each smaller model is trained with velocity distillation [11] from a larger teacher: a frozen teacher _vϕ_ is evaluated at the same ( _a_<sup>_τ_</sup> _, τ, c_ ) and its prediction added as a second regression target, _L_ = _L_ fm + _λ∥vθ − vϕ∥_ 2<sup>2</sup> with _λ_ =1. The teacher is excluded from parameter counts and absent at inference. A single-run comparison at 5M favored distillation strongly (+1 _._ 85 average over the same architecture trained alone; the distilled 4.89M model even beats its own 7.12M teacher), while at 1M it sits inside the seed band of Section V-C — we keep it for the mid-scale models. 

## _C. Inference-time strategy_ 

At deployment the policy replans every control step and applies ACT-style temporal ensembling [14]: each executed action is the exponentially weighted average (decay 0 _._ 01 per step of age) of the predictions from all chunks covering that timestep. For sub-1M models the initial noise of the flow integration is scaled by _σ_ = 0 _._ 85 _<_ 1, making sampling mode-seeking. Both choices are validated in Section V-E. 



<!-- Start of picture text -->
Spatial Object Goal Long (10)<br><!-- End of picture text -->

Fig. 2. The four standard LIBERO task suites: Spatial, Object, Goal and Long/“10”. Scene renderings from the LIBERO benchmark [9]. 

### TABLE I 

SUCCESS RATE (%) ON THE FOUR LIBERO SUITES. MINERVA ROWS USE 2,000 ROLLOUTS (50 EPISODES/TASK), FIXED SEED. THE _π_ 0 _._ 5 ROW IS THE LEROBOT IMPLEMENTATION’S REPORTED RESULT [8] (10 EPISODES/TASK, 400 ROLLOUTS). 

|policy|params|Spat.|Obj.|Goal|Long|**avg**|
|---|---|---|---|---|---|---|
|MINERVA-0.5M|0.54M|94.4|99.6|96.4|89.8|**95.05**|
|+ L1 head|0.54M|96.8|99.6|97.4|89.2|95.75|
|MINERVA-1M|0.99M|97.0|99.8|97.4|92.8|96.75|
|MINERVA-5M|4.89M|99.0|99.2|98.0|92.4|97.15|
|MINERVA-10M|9.66M|98.4|99.0|98.4|94.0|97.45|
|_π_0_._5 (LeRobot)|4.1B|97.0|99.0|98.0|96.0|97.50|



## IV. EXPERIMENTAL SETUP 

**Data and training.** All models train on the `lerobot/libero` dataset (1,693 demonstrations, 273k frames, all 40 tasks pooled across the four suites; Fig. 2), implemented in the LeRobot framework [8]. Training runs 90–120k steps at batch 128 with AdamW (lr 10<sup>_−_4</sup> , cosine decay); one model trains in _≈_ 2 h on a single GH200 or consumer RTX 5080 GPU. Long-suite episodes are loss-upweighted 3 _×_ for the sub-1M models. 

**Evaluation protocol.** Unless otherwise noted, standardLIBERO results use 50 episodes per task _×_ 10 tasks _×_ 4 suites = 2,000 rollouts under fixed seed, with hard environment resets between episodes. Ablation screens use 25 episodes per task where noted; all conclusions were reconfirmed at 50. 

## V. RESULTS 

## _A. Headline: 95% at half a million parameters_ 

Table I shows the released family against the LeRobot implementation of _π_ 0 _._ 5, whose reported result [8] was measured at 10 episodes/task — a reference point rather than a protocol-matched comparison. The 0.54M model reaches 95.05% — 2.4 points behind a model 7,700 _×_ its size — and the 0.99M model closes to 0.75 points. Retrained with the one-pass regression head of Section V-C, the same 0.54M architecture scores 95.75%; this is the configuration we ship. The remaining gap is almost entirely the long-horizon suite (92.8 vs. 96.0 at 1M); the three short-horizon suites are matched or exceeded from 1M upward. Rows are single runs (seed 1000); Section V-C re-trains the 1M model under three seeds (94.60–96.75) and reads every comparison against that band. 



<!-- Start of picture text -->
100 LeRobot-reported  π 0.5 (4.1B): 97.5<br>95.1<br>90<br>0.54M<br>80<br>70<br>60<br>4-suite average<br>LIBERO-Long<br>50<br>0.1 0.25 0.5 1 2.5 5 10<br>parameters (M, log scale)<br>success rate (%)<br><!-- End of picture text -->

Fig. 3. Success versus total parameters (log scale), 2,000 rollouts per point. Performance saturates near 1M parameters; below 0.25M it collapses, and the long-horizon suite (orange) falls first. 

### TABLE II 

ABLATIONS AT THE 1M SCALE, EACH CELL A FULL 2,000-ROLLOUT EVALUATION. TOP: CONFIGURATIONS RE-TRAINED WITH THREE SEEDS. BOTTOM: SINGLE-RUN DELTAS (SEED 1000) — SUGGESTIVE ONLY AGAINST THE _±_ 1-POINT SEED BAND. 

|change|avg _±_ sd (_n_=3)|∆|
|---|---|---|
|baseline: chunk 16, flow matching|95_._63_±_1_._08|—|
|action chunk 16_→_8|92_._47_±_0_._84|_−_**3**_._**16**|
|action chunk 16_→_32|93_._43_±_0_._71|_−_**2**_._**20**|
|vision 0_._49M_→_0_._13M (same total)|94_._22_±_0_._20|_−_**1**_._**41**|
|flow matching _→_L1 regression|95_._97_±_0_._80|+0_._34|
||(_n_=1)||
|AdaLN-Zero _→_concatenation||_−_0_._40|
|AdaLN-Zero _→_FiLM||_−_0_._10|
|no distillation||+0_._10|
|mixer _→_DiT self-attn. head (+26% params)||+0_._05|



## _B. Scaling: saturation at 1M, collapse below 0.25M_ 

Fig. 3 extends the family across two orders of magnitude. From 1M to 10M parameters, average success moves only 0.7 points (96.75 _→_ 97.45): capacity stops being the binding constraint at about one million parameters. Below 0.54M the curve breaks sharply — 88.6% at 0.24M, 68.3% at 0.09M — and the failure is structured: at 0.24M the model still solves 95.2% of Spatial but only 73.0% of Long. Short-horizon competence survives in models far too small to chain the subgoals of long-horizon tasks, suggesting the long suite is where LIBERO actually prices capacity. 

## _C. What survives seed averaging_ 

Re-training the configurations of Table II (top) under three training seeds first calibrates the instrument: **a single training seed moves the 4-suite average by** _±_ **1 point** (the baseline spans 94.60–96.75; its long-suite score alone spans 87.6–92.8). This is large enough to obscure the roughly onepoint deltas common in single-run ablations. 

Two choices survive the band. The first is **chunk length** ( _−_ 3 _._ 16 at chunk 8; _−_ 2 _._ 20 at chunk 32). Its failure is asymmetric and interpretable. Chunk 8 breaks the _goal_ suite 

(86–89% vs. 96.6–97.4% at chunk 16) while leaving the long suite intact: goal tasks pose different targets in one scene, so the policy must commit to a branch, and short chunks let it dither between modes. Chunk 32 breaks the _long_ suite (80– 82% vs. 87.6–92.8%) while leaving goal intact: multi-stage tasks require reacting at subgoal boundaries, and long openloop chunks cannot. **Chunk length chooses which suite you sacrifice** ; 16 is the value that sacrifices neither. 

The second is **vision allocation** : the starved 0.13M-vision configuration, re-trained under three seeds after its singleseed delta ( _−_ 2 _._ 50) proved the last load-bearing _n_ =1 number, replicates at 94 _._ 22 _±_ 0 _._ 20, a _−_ 1 _._ 41-point mean difference, with the damage concentrated on the long suite ( _−_ 4 _._ 6). This is half the single-seed estimate, but the direction is consistent across seeds; notably the starved model is far more seedstable (sd 0.20 vs. 1.08) — with less vision capacity there is less to vary. 

One widely assumed choice does _not_ survive: **flow matching is indistinguishable from direct L1 regression** . The single-seed comparison had regression _−_ 1 _._ 70 worse; over three seeds this reverses to +0 _._ 34. Yet regression requires only one forward pass rather than ten Euler steps, reducing inference time from 8.2 to 2.2 ms per chunk on GPU and from 8.9 to 5.1 ms on CPU at the 0.54M scale (Table III). Flow matching therefore provides no measurable accuracy benefit under this benchmark and protocol, whereas direct regression is **up to 3.8** _×_ **faster** . The same conclusion holds at 0.54M parameters, where retraining with the regression head yields 95.75% versus 95.05% for flow matching, well within the seed band despite the noisier velocity field. We therefore ship the regression variant. 

The single-run rows (Table II, bottom) read accordingly: conditioning style, distillation at 1M, and restoring selfattention (+0 _._ 05 at +26% parameters) are all far inside the band — no measured difference; the mixer head’s value is that it reaches the same score with 26% fewer parameters. 

## _D. Where to spend a fixed budget: the eyes, not the head_ 

A six-point sweep redistributing a fixed _≈_ 1M budget between vision encoder and action head is one-sided: every allocation giving vision 50–80% of the budget lands in a flat 96.3–96.8 band, while starving vision to 0.13M costs the seed-replicated _−_ 1 _._ 41 average and _−_ 4 _._ 6 long of Section V-C — a loss no action-head capacity recovers. During development the same effect appeared as a 10-point longsuite jump (80.8 _→_ 91.0) when moving parameters from an attention head into the CNN at constant total. The design rule for tiny policies is blunt: **shrink the action head, not the eyes** . 

## _E. Inference-time strategy_ 

A comparison of four execution strategies on the same 0.54M checkpoint (25 episodes/task screens) shows two regularities. First, **execution horizon dominates** : for every method, executing 1–2 actions per replan clearly beats 4, which beats 8. Second, at horizon 1, averaging overlapping 

TABLE III 

INFERENCE COST PER ACTION CHUNK (RTX 5080 LAPTOP GPU / 8 CPU THREADS, BATCH 1). 

|policy|params|GPU ms|CPU ms|VRAM|
|---|---|---|---|---|
|MINERVA-0.5M|0.54M|8.2|8.9|0.03 GB|
|+ L1 head|0.54M|**2.2**|**5.1**|0.03 GB|
|MINERVA-1M|0.99M|8.5|10.0|0.03 GB|
|MINERVA-10M|9.66M|11.1|17.8|0.08 GB|
|SmolVLA|450M|118|1,010|0.97 GB|
|_π_0_._5|4.1B|196|12,781|9.36 GB|



chunk predictions (temporal ensembling, 96.3%) beats BIDstyle candidate selection (95.0%), which selects the most coherent of 16 sampled chunks, and both outperform plain chunking (94.1%); RTC-style soft inpainting _hurts_ (91.8%), over-constraining an already-noisy velocity field. The winner was re-confirmed at 50 episodes/task and is the protocol behind Table I. 

Mode-seeking sampling behaves like a capacity-dependent corrector: initial-noise temperature 0.85 adds +0 _._ 2 to +2 _._ 25 points on every sub-1M model — most where the model is weakest — and nothing at 1M and above. Small flow policies do not need a different architecture at inference; they need their noisy velocity field averaged and their sampling sharpened. 

## _F. Efficiency: closing the loop on a CPU_ 

Table III reports wall-clock cost: on identical hardware the 0.54M model is 113 _×_ faster than SmolVLA and 1,400 _×_ faster than _π_ 0 _._ 5. At 8.9 ms per chunk on eight laptop CPU threads — _≈_ 5 ms with the shipped regression head, which skips the ten-step ODE — MINERVA-0.5M can replan at every step of a _∼_ 100 Hz control loop with no GPU at all. The replan-every-step protocol that maximizes success (Section V-E) is therefore affordable on an embedded CPU; for the VLA baselines it is not ( _∼_ 1 Hz SmolVLA, 0.08 Hz _π_ 0 _._ 5). For integration into a larger robot system, the entire perception-to-action module fits in the latency and memory budget usually reserved for a single sensor driver — and task-ID conditioning is a natural interface for it: a planner that already knows which skill to execute selects it by index, no language model in the loop. 

## _G. The embedding causally selects the task_ 

The headline result — task IDs replace language at no cost — is correlational. A permutation probe makes the role of task conditioning causal: we evaluate the released 1M checkpoint, with all weights fixed, while rewriting only its task-ID mapping. Rotating every task’s ID within its suite collapses the average success rate from 96.75% to **6.5%** . Forcing every task to its suite’s task-0 ID yields **16.2%** . Spatial, Goal, and Long fall almost exactly to the 1-in-10 prediction (10.0%, 10.0%, and 8.8%, respectively), under which only the task matching the forced ID succeeds. Object retains a larger 22– 36% residue, indicating that visually distinctive scenes can partially compensate for an incorrect task ID. These results 

### TABLE IV 

LIBERO-PLUS [29] SUCCESS (%) BY PERTURBATION FACTOR (STRATIFIED SAMPLE, 2 EPISODES/VARIANT, IDENTICAL VARIANTS PER MODEL). LANGUAGE VARIANTS ARE VISUALLY IDENTICAL TO BASE SCENES AND DOUBLE AS THE UNPERTURBED BASELINE. 

|factor|0.54M|0.99M|4.89M|
|---|---|---|---|
|language (= baseline)|96.7|96.7|99.2|
|new objects|69.7|62.1|95.5|
|sensor noise|68.5|60.0|73.8|
|robot init|57.1|60.3|63.5|
|object layout|50.0|51.8|58.9|
|camera|11.5|18.5|34.6|
|background|9.3|4.7|7.0|
|light|1.1|2.2|6.5|
|**average**|**46.7**|**46.0**|**55.7**|
|(standard LIBERO)|(95.05)|(96.75)|(97.15)|



show that task-ID conditioning provides the dominant taskselection signal, while visual context can disambiguate the task when the conditioning signal is insufficient. On standard LIBERO, instruction-following therefore reduces primarily to task _selection_ , with task identity conveyed mainly through the conditioning channel. 

## _H. Beyond 40 tasks: LIBERO-90_ 

To test the recipe on a larger closed task set, we train the 1M architecture on LIBERO-90 (3,959 demonstrations, 90 tasks; one task has no demonstrations). Task IDs are indices into the list of distinct instruction strings, and the 90 tasks contain only 74 distinct instruction strings, so some tasks necessarily share an ID. The result: **94.6% average over 89 tasks** at 0.995M parameters — standard-LIBERO-level success at 2 _._ 25 _×_ the task count, with no language encoder, no pretrained vision, and here no distillation. Tasks sharing an instruction string with another task score 93.6% vs. 95.1% for unique-ID tasks: vision supplies scene context when the ID underdetermines it, completing the picture of Section V- G. The failure tail is thin (worst tasks 50–60%; no collapsed scene family). Caveats: single seed, 10 episodes/task, evaluation pinned to the renderer version matching this dataset’s scenes. 

## _I. Robustness under LIBERO-Plus perturbations_ 

Finally we price what the 95% headline does _not_ include, on LIBERO-Plus [29] (10,030 perturbed variants of the 40 tasks across 7 factors; each rephrased instruction is mapped to the corresponding base-task ID). In Table IV the 0.54M, 0.99M and 4.89M models fall to 46.7%, 46.0% and 55.7%. Three findings: (1) language variants lose _≈_ 0 points, so the drop is perturbation-driven, not a renderer artifact; (2) robustness is not capacity-limited below 1M, and 5M buys +10 points, almost all on semantic/geometric factors (new objects 62 _→_ 96); (3) photometric robustness remains near zero across all tested scales (light _≤_ 6.5%): the scratch CNN is highly sensitive to appearance shifts, and no parameter count in this range fixes it — quantifying the memorization thesis. 

## VI. DISCUSSION AND LIMITATIONS 

**What this says about the benchmark.** MINERVA is deliberately incapable of language understanding, openvocabulary perception, or transfer to unseen tasks — yet it loses only 0.75–2.4 points to state-of-the-art VLAs on standard LIBERO. Together with the LIBERO-PRO perturbation analysis [13], the parameter-scaling curve gives a quantitative reading: the standard evaluation is satisfiable by _∼_ 0.5M parameters of task-indexed visuomotor memorization — and the permutation probe of Section V-G shows that the task ID selects the executed task, while the LIBERO-Plus audit prices exactly what the memorization omits. Reported LIBERO gains above _∼_ 97% are therefore unlikely to measure the capabilities that motivate large models; perturbation-based protocols [13] are needed to see those. The training-seed band of Section V-C sharpens this: single-run gains of a point or less on this benchmark are comparable to the observed seed-to-seed variation. Conversely, the long suite’s sharp collapse below 0.5M is the one axis where standard LIBERO does price capacity. 

**A capacity floor for deployment.** The point of scaling down is not the small model itself but the measurement. Generalist VLAs concentrate their value in open-world generalization; a deployed system with a fixed task set does not pay for generalization it will not use — the argument deployment-efficient RL makes for training constraints [10], applied here to capacity. Our result gives one concrete task set its empirical floor ( _≈_ 0.5M parameters, _≈_ 1M with longhorizon headroom), and the recipe — train or reuse a larger model, distill to the floor, execute with one forward pass — is the pipeline an integrator would run against their own task set. The floor is task-set-specific: LIBERO-Plus suggests that robustness may require greater capacity — the 4.89M model substantially improves semantic robustness over the sub-1M models — and it will shift with environment, sensing and embodiment. Measuring how this floor varies — a manipulation analogue of information-theoretic task-complexity measures in RL [12] — is the research program this first step opens. 

**Limitations.** The task-ID embedding cannot address a task it was not trained on; MINERVA is a measurement instrument and a deployable component for closed task sets, not a generalist. The five key ablation configurations are each evaluated across three training seeds, but the headline models, the LIBERO-90 run, the remaining sweep points and the distillation comparison are single runs inside a _±_ 1- point seed band, and everything rests on one simulator and one evaluation seed. Robustness is now measured rather than assumed: under LIBERO-Plus perturbation the models keep roughly half their standard score, failing hardest on photometric shifts — a potentially addressable gap, e.g., via photometric augmentation, that the 95% headline conceals. Real-robot validation remains future work, though the CPU latency and the SO-101-class hardware support in the underlying framework [8] make it direct. 

## VII. CONCLUSION 

We scaled a language-free, from-scratch manipulation policy down until standard LIBERO broke it, and it did not break until half a million parameters. A _±_ 1-point trainingseed band frames every comparison; only chunk length and vision allocation survive it, and one-pass regression matches flow matching at lower inference cost and is therefore the configuration we ship. A permutation probe shows the embedding causally selects the task, the recipe holds on LIBERO-90, and a LIBERO-Plus audit locates the robustness the headline does not include. The outcome is a manipulation module that closes its control loop in _≈_ 5 ms on a laptop CPU, and a calibration of what the community’s default benchmark actually measures — a first step toward task-specific capacity measurement: know a task set’s empirical floor, then build or distill to it. 

## ACKNOWLEDGMENT 

The authors thank the maintainers of LeRobot and LIBERO for the open infrastructure this study builds on. This research also used computational resources of Miyabi provided through the Multidisciplinary Cooperative Research Program at the Center for Computational Sciences, University of Tsukuba. 

- [17] J. Wen _et al._ , “TinyVLA: Towards fast, data-efficient vision-languageaction models for robotic manipulation,” _arXiv:2409.12514_ , 2024. 

- [18] S. Belkhale and D. Sadigh, “MiniVLA: A better VLA with a smaller footprint,” Stanford AI Lab Blog, https://ai.stanford.edu/blog/minivla/, 2024. 

- [19] Y. Yue _et al._ , “DeeR-VLA: Dynamic inference of multimodal large language models for efficient robot execution,” in _NeurIPS_ , 2024. 

- [20] K. Pertsch _et al._ , “FAST: Efficient action tokenization for visionlanguage-action models,” _arXiv:2501.09747_ , 2025. 

- [21] C. Chi _et al._ , “Diffusion policy: Visuomotor policy learning via action diffusion,” in _RSS_ , 2023. 

- [22] S. Levine, C. Finn, T. Darrell, and P. Abbeel, “End-to-end training of deep visuomotor policies,” _JMLR_ , vol. 17, no. 39, pp. 1–40, 2016. 

- [23] Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow matching for generative modeling,” in _ICLR_ , 2023. 

- [24] W. Peebles and S. Xie, “Scalable diffusion models with transformers,” in _ICCV_ , 2023. 

- [25] I. Tolstikhin _et al._ , “MLP-Mixer: An all-MLP architecture for vision,” in _NeurIPS_ , 2021. 

- [26] A. G. Howard _et al._ , “MobileNets: Efficient convolutional neural networks for mobile vision applications,” _arXiv:1704.04861_ , 2017. 

- [27] E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. Courville, “FiLM: Visual reasoning with a general conditioning layer,” in _AAAI_ , 2018. 

- [28] C. Chen _et al._ , “DiT-Air: Revisiting the efficiency of diffusion model architecture design in text to image generation,” _arXiv:2503.10618_ , 2025. 

- [29] S. Fei _et al._ , “LIBERO-Plus: In-depth robustness analysis of visionlanguage-action models,” _arXiv:2510.13626_ , 2025. 

## REFERENCES 

- [1] Octo Model Team _et al._ , “Octo: An open-source generalist robot policy,” in _RSS_ , 2024. 

- [2] M. J. Kim _et al._ , “OpenVLA: An open-source vision-language-action model,” in _CoRL_ , 2024. 

- [3] M. Shukor _et al._ , “SmolVLA: A vision-language-action model for affordable and efficient robotics,” _arXiv:2506.01844_ , 2025. 

- [4] K. Black _et al._ , “ _π_ 0: A vision-language-action flow model for general robot control,” in _RSS_ , 2025. 

- [5] H. Xie _et al._ , “TurboVLA: Real-time vision-language-action model at 32 Hz on an RTX 4090 with _<_ 1 GB VRAM,” _arXiv:2607.27205_ , 2026. 

- [6] M. J. Kim, C. Finn, and P. Liang, “Fine-tuning vision-language-action models: Optimizing speed and success,” in _RSS_ , 2025. 

- [7] Physical Intelligence _et al._ , “ _π_ 0 _._ 5: A vision-language-action model with open-world generalization,” _arXiv:2504.16054_ , 2025. 

- [8] R. Cadene _et al._ , “LeRobot: State-of-the-art machine learning for realworld robotics in PyTorch,” https://github.com/huggingface/lerobot, 2024. 

- [9] B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, and P. Stone, “LIBERO: Benchmarking knowledge transfer for lifelong robot learning,” in _NeurIPS Datasets and Benchmarks_ , 2023. 

- [10] T. Matsushima, H. Furuta, Y. Matsuo, O. Nachum, and S. Gu, “Deployment-efficient reinforcement learning via model-based offline optimization,” in _ICLR_ , 2021. 

- [11] G. Hinton, O. Vinyals, and J. Dean, “Distilling the knowledge in a neural network,” _arXiv:1503.02531_ , 2015. 

- [12] H. Furuta, T. Matsushima, T. Kozuno, Y. Matsuo, S. Levine, O. Nachum, and S. Gu, “Policy information capacity: Informationtheoretic measure for task complexity in deep reinforcement learning,” in _ICML_ , 2021. 

- [13] X. Zhou _et al._ , “LIBERO-PRO: Towards robust and fair evaluation of vision-language-action models beyond memorization,” _arXiv:2510.03827_ , 2025. 

- [14] T. Z. Zhao, V. Kumar, S. Levine, and C. Finn, “Learning fine-grained bimanual manipulation with low-cost hardware,” in _RSS_ , 2023. 

- [15] Y. Liu, J. I. Hamid, A. Xie, Y. Lee, M. Du, and C. Finn, “Bidirectional decoding: Improving action chunking via guided test-time sampling,” _International Conference on Learning Representations (ICLR)_ , 2025. [Online]. Available: https://arxiv.org/abs/2408.17355 

- [16] K. Black, M. Y. Galliker, and S. Levine, “Real-time execution of action chunking flow policies,” in _NeurIPS_ , 2025. 

