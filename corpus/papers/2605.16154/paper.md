# **Learn Where Outcomes Diverge: Efficient VLA RL via Probabilistic Chunk Masking** 

**Vaidehi Bagaria**<sup>_∗_</sup> **, Nikshep Grampurohit**<sup>_∗_</sup> **, Pulkit Verma**<sup>_†_</sup> Indian Institute of Technology Madras 

### **Abstract** 

Reinforcement learning (RL) allows vision-language-action (VLA) policies to generalize beyond their training distribution by optimizing directly for task success, but this post-training stage is computationally expensive. A natural response has been to speed up rollout data collection through faster simulators and world models. In GRPO-based VLA RL, however, we find that the dominant cost lies elsewhere: gradient computation accounts for _∼_ 78% of wall-clock time per step in our runs, while rollout collection accounts for only _∼_ 21%. Gradient cost dominates because much of this computation is spent on phases that contribute little to learning. GRPO’s learning signal is driven by advantage variance: only phases where successful and failed rollouts diverge produce useful learning signal. However, GRPO assigns the same advantage to every chunk in a rollout. As a result, actor-update compute is spent uniformly across the trajectory, including phases the policy already handles after pre-training and supervised fine-tuning. This paper presents Probabilistic Chunk Masking (PCM), a drop-in modification to GRPO that allocates gradient computation to a small, probabilistically selected subset of chunks per trajectory. PCM scores semantic phases using success-failure action variance, a rollout-derived proxy for per-phase gradient variance, and samples a fixed chunk budget with online-updated phase-level keep probabilities. We formalize per-phase gradient variance as the quantity that determines where gradient computation is useful and show that success-failure action variance provides a measurable proxy for it. PCM requires no reward model or learned critic. On three LIBERO benchmarks, PCM matches the final success rate of standard GRPO while achieving 2 _._ 38 _×_ wall-clock speedup, 4 _._ 8 _×_ faster gradient updates, and 60% lower peak activation memory, while backpropagating through fewer than 20% of trajectory chunks. 

### **1 Introduction** 

Reinforcement learning (RL) has become essential for vision-language-action (VLA) models, enabling generalization to new environments and behaviors [Black et al., 2024, Kim et al., 2024]. The field has increasingly converged on Group Relative Policy Optimization (GRPO) [Shao et al., 2024] whose critic-free formulation removes the cost and complexity of training a separate value model [Li et al., 2025a, Lu et al., 2025]. As these systems scale, speeding up post-training has become critical. Efforts to speed up GRPO-based VLA RL have implicitly assumed that rollout collection is the bottleneck. We measure where time actually goes and find the reverse: in our runs gradient computation accounts for _∼_ 78% of wall-clock time per training step, while rollout collection accounts for only _∼_ 21% (Fig. 1, right). 

> *Equal contribution. 

> †Correspondence to `pulkitv@cse.iitm.ac.in` . 

Preprint. 



|**Benchmark**|**Grad (%) **|**Rollout (%)**|
|---|---|---|
|LIBERO-Object|77.91|21.27|
|LIBERO-Goal|77.86|21.29|
|LIBERO-Spatial|77.85|21.29|



Figure 1: Left: per-phase success–failure action variance ( _Cc_ ) concentrates on decision-critical phases. Right: gradient computation dominates wall-clock training cost in simulator-based GRPO with 10 rollouts per prompt. Remaining _∼_ 1% is logging, evaluation, and inter-step overhead. 

This inefficiency arises from a deeper structural feature shared by most RL algorithms. Backpropagation runs over every chunk in the trajectory, and trajectory-level credit assigns the same advantage to each chunk regardless of which determined the outcome. Both inefficiencies share a common cause: the trajectory has been treated as the unit of gradient computation. Even methods that differentiate per-step credit, such as PPO [Schulman et al., 2017] with a learned critic, still apply forward/backward-pass compute to all sampled timesteps. They modify the signal, not the compute allocation. 

This observation suggests a missing axis in VLA RL optimization: _within a trajectory, which chunks should receive gradient computation at all?_ 

The answer is not all of them. GRPO’s learning signal is driven by differences in advantage across rollouts: when rollout rewards are uniform, gradients vanish [Wang et al., 2026, Xu and Ding, 2026]. We argue that variance non-uniformity persists even within a single non-collapsed group. Supervised fine-tuning is effective in densely covered, unimodal regions, but weak under sparse demonstrations or multimodal optimal behavior [Chu et al., 2025]; these are precisely the regimes that RL fine-tuning is intended to refine. This suggests that (a) gradient computation should be treated as an allocatable resource and (b) recomputing gradients on phases the base policy already handles wastes compute and memory. From this perspective, RL training becomes a problem of allocating limited gradient budget to the parts of trajectories that carry useful learning signal. 

We present Probabilistic Chunk Masking (PCM), a drop-in modification to GRPO that updates only a fixed budget of chunks per trajectory. PCM assigns phase-level keep probabilities using the success-failure action variance _Cc_ , which is concentrated in a small number of outcome-critical semantic phases _c_ (Fig. 1, left). It then physically removes the unselected chunks from the backward pass, significantly reducing actor-update compute and activation memory. We further prove that _Cc_ serves as a measurable lower-bound proxy for the true per-phase gradient variance _Vc_ , giving a principled basis for this allocation. The **main contributions** of this paper are as follows: 

- **We show that** _Vc_ **is the right signal for gradient allocation** and derive a Neyman allocation 

- proportional to _Nc√Vc_ ( _Nc_ is the number of chunks in phase _c_ ), which concentrates gradient compute on outcome-divergent phases. 

- **We establish that the per-phase success–failure action variance** _Cc_ **is a sufficient proxy for** _Vc_ , 

- computable directly from rollouts GRPO already produces with no reward model or learned critic. 

- **Empirically, these enable substantial efficiency gains in gradient-bound VLA-RL.** On 

- LIBERO [Liu et al., 2023], PCM achieves 2 _._ 38 _×_ wall-clock speedup to 98% convergence, 4 _._ 8 _×_ faster gradient steps, 60% lower peak activation memory, using less than 20% of chunks per trajectory. 

### **2 Related works** 

**Reinforcement Learning Fine-Tuning of VLA Models.** Reinforcement learning has become an effective post-training stage for vision-language-action (VLA) models [Chu et al., 2025, Liu et al., 2025]. Early approaches relied on learned critics [Hu et al., 2024, Guo et al., 2025, Chen et al., 2025a, Tan et al., 2025], but the field has converged on critic-free GRPO-style optimization as these systems have scaled [Li et al., 2025a, Lu et al., 2025, Chen et al., 2025b, Ye et al., 2025]. 

To support this shift, advances in world models and simulation have significantly reduced rollout cost [Zhu et al., 2025, Liu et al., 2026, Zang et al., 2026] under the implicit assumption that data collection 

2 

is the bottleneck. However, this progress has left gradient computation, which we observe accounts for _∼_ 78% of training time, largely unoptimized. Other work improves the source or structure of the reward signal [Li et al., 2025b, Zhai et al., 2025, Tan et al., 2025] by altering the reward, critic, rollout source, or advantage/credit-assignment rule. We show that this additional machinery can be avoided: _Cc_ (success-failure action variance within a phase), an internal signal already present in GRPO rollouts, directly captures where learning signal concentrates and is sufficient to guide gradient allocation. 

**Efficiency Methods for GRPO.** GRPO’s learning signal is driven by advantage variance [Nan et al., 2025], and prior large language model (LLM) RL work has exploited this at two granularities. At the prompt level, methods filter or reweight zero-variance rollout groups [Yu et al., 2025, Zheng et al., 2025, Hu et al., 2025]. At the token level, work selects which sequence positions receive gradient using policy-internal signals such as token entropy or probability shifts [Wang et al., 2025, Khandoga et al., 2026, Hu et al., 2026]. PCM operates at a third granularity: phase-level allocation within a trajectory. Entropy and probability shifts are indirect proxies that measure where the model is uncertain, not where successful and failed rollouts actually diverge. In settings with verifiable outcomes (like VLA RL), this raises a more fundamental question, _why estimate outcome-criticality indirectly when the outcome signal is already available?_ We use _Cc_ , computed directly from rollout outcomes, to allocate gradient compute according to observed success–failure divergence. Unlike prior work that relies on proxy signals such as entropy or probability shifts, PCM exploits outcome-grounded divergence. 

Parameter-efficient methods like LoRA and its extensions are standard in LLM and VLA fine-tuning, reducing trainable parameters but not the activation cost of backpropagating through long trajectories [Hu et al., 2021, Aghajanyan et al., 2021, Dettmers et al., 2023, Zhang et al., 2025, 2023]. Activationfocused methods reduce this memory cost through checkpointing and streaming backpropagation [Chen et al., 2016, Korthikanti et al., 2023, Luo et al., 2025]. Crucially, prior methods do not address the inefficiency of computing gradients on outcome-invariant phases within a trajectory, which still incur computation cost despite contributing little to learning. 

### **3 Preliminaries** 

We briefly describe the setup for chunk-level GRPO, which our method builds on. 

**VLA Policies and Action Chunks.** Let _πθ_ denote a VLA policy that produces actions in chunks of length _L_ conditioned on observation _si,k_ , where _i_ indexes the trajectory and _k_ the chunk within that trajectory. A rollout of trajectory _i_ decomposes into _Ni_ = _⌈Ti/L⌉_ chunks, where _Ti_ is the total number of timesteps. 

**Group Relative Policy Optimization.** GRPO [Shao et al., 2024] samples a group of _G_ rollouts from the current policy, scores each with a binary reward _ri ∈{_ 0 _,_ 1 _}_ , and forms the group-relative advantage 



Writing _ai,k_ for the action tokens of chunk _k_ in trajectory _i_ , the chunk-level GRPO objective is 



with PPO-style clipping omitted for clarity. 

### **4 Methodology: Probabilistic Chunk Masking** 

#### **4.1 Phase-Conditioned Gradient Variance** 

Following the VLA-RL setup formalized in Appendix A, we partition every trajectory into _K_ semantic phases using a deterministic labeling rule (described in Sec. 5.1). Let _P_ = _{c_ 1 _, . . . , cK}_ denote this phase set, and let _φ_ ( _i, k_ ) _∈P_ denote the phase of chunk ( _i, k_ ). 

3 



Figure 2: Probabilistic Chunk Masking (PCM). We compute per-phase success–failure action variance ( _Cc_ ) from GRPO rollouts, use it to prioritize decision-critical phases, and sample a fixed budget of chunks per trajectory. Gradients are computed only on selected chunks. 

**Phase decomposition of the GRPO gradient.** The gradient of _L_ GRPO decomposes by phase: 



The learning signal available in phase _c_ is characterized by the _per-phase gradient variance_ 



Intuitively, _Vc_ is large for phases where the policy behaves differently across successful and failed rollouts. Throughout, _Vc_ refers to the within-trajectory chunk variance ( _Ai_ fixed); under advantage normalization (Eq. 1), this matches the unconditional variance in Eq. (4) up to an _O_ (1) constant. We formalize this observation as follows. Note that formal proofs appear in Appendix B. 

**Lemma 1** (Phase gradient variance) **.** _Let c ∈P be a phase in which πθ has converged, i.e., the action distributions of successful and failed rollouts are identical in phase c. Then ∥gc_ ( _θ_ ) _∥≈_ 0 _and Vc ≈_ 0 _, so gradient samples from phase c contribute negligible signal. Conversely, for a phase c in which the base policy is underspecified, Vc is large and gradient samples from phase c are informative._ 

**Empirical proxy for Vc.** Computing _Vc_ exactly requires access to the gradient distribution, which is expensive. We instead use the _success-failure action variance_ 



computable directly from the rollout group that GRPO already produces, with no auxiliary model or annotation. We compute _Cc_ only for rollout groups with non-zero reward variance. **Lemma 2** ( _Cc_ as a proxy for _Vc_ ) **.** _For a policy πθ that is locally Gaussian with per-dimension variance σπ_<sup>2</sup><sup>_, the per-phase gradient variance satisfies Vc≥C_</sup> _c_<sup>2</sup><sup>_/_4</sup><sup>_σ_</sup> _π_<sup>2</sup><sup>_._</sup> 

Note that the bound in Lemma 2 is a lower bound that absorbs _∥∇θµθ_ ( _s_ ) _∥_<sup>2</sup> into the constant. It justifies using _Cc_ to recover the _ordering_ of _Vc_ across phases, rather than exact correspondence. We treat _Cc_ as a practical proxy supported by empirical evidence: the realized allocation (Fig. 6) closely matches the<sup>_√_</sup> _Vc_ -weighted prediction, and variance-aware ablations (Fig. 5b) show that _Cc_ -weighted selection outperforms both random masking and highest-variance-only selection. 

#### **4.2 Optimal Budget Allocation** 

We now state the core theoretical result. Given a fixed chunk budget _B_ per trajectory and phase variance estimates _{Vc}c∈P_ , we ask: how should _B_ be distributed across phases to minimize the variance of the GRPO gradient estimator (the sampling noise around the true gradient, distinct from the per-phase signal _Vc_ )? Lower estimator variance means a more accurate estimate of the true gradient per step, which translates directly to faster stochastic gradient descent (SGD) convergence [Bottou et al., 2018]. This follows Neyman-allocation intuition: under a fixed sampling budget, phases should be sampled in proportion to their contribution to estimator variance. 

**Theorem 1** (Optimal phase allocation) **.** _Let bc denote the number of chunks sampled from phase c, with_<sup>�</sup> _c∈P_<sup>_bc_=</sup><sup>_B, and let Ncdenote the expected number of chunks in phase c per trajectory.The_</sup> 

4 

_allocation minimizing the variance of the unbiased GRPO gradient estimator (Eq. (2) subject to the budget constraint is_ 



The speedup over uniform allocation depends on how sharply _Vc_ concentrates across phases. When _Vc_ is uniform across all _K_ phases, Eq. (6) reduces to uniform allocation _bc_ = _B/K_ . When _Vc_ concentrates on a small subset of phases, the Neyman allocation assigns disproportionately more budget to those outcome divergent phases, reducing estimator variance relative to uniform at the same total budget _B_ . The concentration of _Cc_ on contact-rich phases observed in Fig. 1, combined with Lemma 2, implies that _Vc_ is similarly concentrated, making this regime directly applicable here. Section 5 quantifies the resulting wall-clock speedup empirically. A formal convergence-rate derivation and speedup formula are provided in Appendix B.1. 

#### **4.3 Online Phase Score Estimation** 

Theorem 1 requires knowledge of _{Vc}_ , which is not directly observable. By Lemma 2, the relative ordering of _{Vc}_ is captured by the rollout-computable proxy _{Cc}_ . We therefore use _Cc_ to set per-phase weights; since phase _c_ contains _Nc_ candidate chunks, weighted sampling consistent with Eq. (6) gives E[ _bc_ ] _∝ Nc√Vc_ , concentrating gradient computation on outcome-divergent phases. Phases with larger estimated _Vc_ receive higher keep probability, while larger _Nc_ contributes more candidate chunks. Both factors increase sampling frequency within the fixed budget _B_ . 

During training, we compute a batch-level phase score _Cc_<sup>(</sup><sup>_t_)</sup> from the rollout group at step _t_ . We maintain a short buffer _Bc_ for each phase and append the current score to the short refresh buffer. With _G_ = 10 rollouts and binary rewards, individual _Cc_ estimates may be noisy, especially under imbalanced success-failure splits. Therefore, scores are averaged over _T_ rc = 5 batches before refreshing the keep probabilities. 



Every _T_ rc steps (we use _T_ rc = 5), we collapse the buffered scores into a share-normalized phase score: 



Share normalization is scale-invariant: it preserves the relative phase ordering even though the absolute magnitude of _Cc_ can vary across tasks and training stages. The resulting _ρ_ ˜ _c_ is used as the phase-level score for probabilistic chunk selection. For the first update, the keep probabilities are computed from the phase scores of the first rollout group itself; after that, scores are updated every _T_ rc steps using the buffered window. This online update lets PCM track the realized learning signal: if a phase learns faster, its success–failure divergence decreases and keep probability naturally falls. 

#### **4.4 Probabilistic Mask Selection** 

Given phase scores _{ρ_ ˜ _c}_ as proxies for _{_<sup>_√_</sup> _Vc}_ , we map them to per-phase keep-probabilities that implement the allocation of Theorem 1 in expectation: 



with _p_ min = 0 _._ 1. The floor _p_ min prevents any phase from being structurally excluded between buffer recomputes. This is important because _ρ_ ˜ _c_ may be transiently zero for a phase with nonzero true variance _Vc_ if that phase is underrepresented or temporarily underestimated in the current buffer window. Excluding such phases entirely would violate the exploration requirement of the allocation and could cause the estimated _{ρc}_ to diverge from the true _{Vc}_ ordering over time. Each chunk inherits its phase’s probability weight: _wi,k_ = _pφ_ ( _i,k_ ). The set _{pc}_ is held fixed across the next _T_ rc steps. 

#### **4.5 Fixed-Budget Sampling and Physical Shrinking** 

Given the allocation rule derived in Theorem 1, we sample a fixed budget of B chunks per trajectory using weighted sampling without replacement [Plackett, 1975, Luce, 1959], with weights _{wi,k}_ : 

_Ki ∼_ WeightedSample( _wi,_ 1 _, . . . , wi,Ni_ ; min( _B, Ni_ ) _,_ w/o replacement) _._ (10) 

5 

The selected set is _Ki_ = _{k_ 1 _, . . . , kmi}_ . Non-selected chunks are physically removed from the batch tensor before the forward pass. 

The actor update uses the masked objective: 



The masked objective in Eq. (11) omits the 1 _/pc_ importance weights, making it a biased estimator of the full-trajectory GRPO gradient. The bias is<sup>�</sup> _c∈P_<sup>(1</sup><sup>_−pc_)</sup><sup>_gc_:chunks that are not selected</sup> contribute zero to Eq. (11) but _gc_ to the full gradient. Under our allocation _pc ∝_<sup>_√_</sup> _Vc_ , the factor (1 _− pc_ ) is large precisely when _Vc_ is small, and by Lemma 1, small _Vc_ implies small _∥gc∥_ . Each term in the bias is therefore suppressed by at least one small factor, so Theorem 1’s allocation remains approximately optimal under the biased estimator. The trade-off is favorable: at the same chunk budget _B_ , the masked estimator has substantially lower variance than the importance-weighted unbiased alternative, translating to faster SGD convergence (Appendix B.1). A formal bound is given in Appendix B.2. 

**Algorithm 1** Probabilistic Chunk Masking for GRPO 

|**Req**<br>|**uire:** Policy_πθ_; group size_G_; chunk budget_B_; floor_p_min; refresh window_T_rc<br>||
|---|---|---|
|1: <br>|Initialize phase-score buffers_Bc ←_[ ]for all phases_c_<br>||
|2:|**while**not converged**do**<br>||
|3:|Sample_G_rollouts and compute binary rewards_{ri}_<br>|_▷_Eq. (1)|
|4:|Compute GRPO advantages_{Ai}_from the group rewards<br>|_▷_Eq. (1)|
|5:|Label each chunk with a phase_φ_(_i, k_)from the gripper trajectory<br>|_▷_Sec.5.1|
|6:|Compute per-phase success-failure action variance_{C_<sup>(</sup><sup>_t_)</sup><br>_c _<sup>_}_</sup>_c∈P_ <sup>from rollouts</sup><br>|_▷_Eq. (5)|
|7:|Append_C_<sup>(</sup><sup>_t_)</sup><br>_c_<br>to buffer_Bc_for all_c ∈P_|_▷_Eq. (7)|
|8:|**if**first batch**or**_|Bc|_=_T_rc **then**||
|9:|Collapse buffers, share-normalize and max-normalize|_▷_Eq. (8)|
|10:|Apply floor: _pc ←_max(_p_min_,_ ˜_ρc_)for all_c_|_▷_Eq. (9)|
|11:|Reset phase buffers: _Bc ←_[ ]for all_c ∈P_||
|12:|**end if**||
|13:|**for**each trajectory_i_with_Ni_ chunks**do**||
|14:|Set chunk weights_wi,k ←pφ_(_i,k_) for_k_ = 1_, . . . , Ni_|_▷_Sec. (4.4)|
|15:|Sample_Ki_ with_|Ki|_= min(_B, Ni_)via weighted sampling w/o replacement|_▷_Eq. (10)|
|16:<br>17:|Physically remove non-selected chunks from batch; keep only_Ki_for actor upd<br>**end for**|ate_▷_Sec.4.5|
|18:<br>19:|Update_πθ_ using the masked GRPO loss<br> **end while**|_▷_Eq. (11)|



### **5 Empirical Evaluation** 

#### **5.1 Experimental Setup** 

**Models and Benchmarks.** We run our experiments on OpenVLA-OFT [Kim et al., 2025], a 7B vision-language-action model that predicts 7-DoF action chunks with chunk length _L_ =8. We evaluate on three benchmarks: LIBERO-Object, LIBERO-Spatial, and LIBERO-Goal [Liu et al., 2023]. 

**Phase Labeling.** We assign each chunk to one of _K_ =5 semantic phases using a deterministic, multi-grasp-aware rule over the per-chunk gripper-close fraction _gf_ [ _j_ ] _∈_ [0 _,_ 1]. Activegrip chunks ( _gf_ [ _j_ ] _≥_ 0 _._ 5) capture sustained grasp and transport; pre-grasp labels the up-tothree chunks immediately before sustained closure (0 _._ 1 _≤ gf_ [ _j_ ] _<_ 0 _._ 5); release-ramp labels the up-to-three chunks after; approach covers the remaining pre-contact chunks; and tail covers post-release open-gripper chunks; Overlapping windows are resolved by the priority ordering `active-grip` _>_ `pre-grasp` _>_ `release-ramp` _>_ `approach` _>_ `tail` . This rule is applied identically to successful and failed trajectories, enabling _Cc_ to compare outcomes under a shared phase partition. Full details are in Appendix E.2. 

6 

**Training and Evaluation.** We implement PCM on the SimpleVLA-RL [Li et al., 2025a] verl [Sheng et al., 2025] pipeline with GRPO groups of 10 rollouts per prompt, comparing against the fulltrajectory chunk-level GRPO baseline. We fine-tune OpenVLA-OFT with LoRA using 2 NVIDIA H100 GPUs. We evaluate every training step using 50 held-out validation rollouts. Results are averaged across three seeds, and we report per-step update time and peak GPU memory. Detailed experimental settings are in Appendix E. 

**Research Questions.** Our experiments are designed to answer the following: 

- **RQ1** Does variance-aware chunk selection preserve the learning dynamics of full-trajectory GRPO, measured by step-wise accuracy curves? 

- **RQ2** Does selective gradient allocation reduce wall-clock time to reach a target success rate without affecting sample efficiency? 

- **RQ3** How sensitive is performance to the chunk budget _B_ , and is there a stable operating point that balances gradient signal with per-step speedup? 

- **RQ4** Is variance-aware probabilistic selection necessary, or do simpler masking strategies achieve comparable performance under the same chunk budget? 

#### **5.2 Results and Discussion** 



Figure 3: Success rate as a function of wall-clock time **(a)** and training step **(b)** on LIBERO-Object, averaged over three seeds. PCM ( _B_ =12) closely matches full-trajectory GRPO across training steps while converging significantly faster. 

**RQ1: Learning Dynamics.** We compare PCM with chunk budget _B_ =12 against full-trajectory GRPO on LIBERO-Object using the same SFT-initialized OpenVLA-OFT checkpoint, identical rollout groups, rewards, advantages, and optimizer settings. The methods differ only in whether the actor update is computed over all chunks or the subset selected by PCM. Both are trained for a fixed budget of 200 steps. 

As a function of training step (Fig. 3b), PCM and GRPO exhibit nearly identical learning curves and reach matched final accuracy. This shows that PCM preserves per-step learning dynamics while improving efficiency, confirming that _the selective gradient allocation does not degrade step-wise convergence behavior_ . 

**RQ2: Wall-Clock Efficiency.** As shown in Fig. 3a, PCM reaches nearsaturation significantly earlier in wallclock time. Fig. 4a shows that PCM reaches the target success rate of 98% _±_ 0 _._ 02 up to 2 _._ 38 _×_ faster than vanilla GRPO, with the gap widening sharply at higher accuracy thresholds. Table 1 shows that this trend holds across all LIBERO benchmarks, reducing the av- 

||Vanilla GRPO|PCM (Ours)|
|---|---|---|
|LIBERO-Object|45_._78_±_0_._95|**19**_._**23**_±_**0**_._**57**|
|LIBERO-Goal|51_._25_±_1_._05|**21**_._**18**_±_**0**_._**59**|
|LIBERO-Spatial|49_._89_±_0_._98|**21**_._**23**_±_**0**_._**63**|
|Overall|48_._97_±_0_._99|**20**_._**55**_±_**0**_._**60**|



Table 1: Wall-clock time in hours to reach 98% _±_ 0 _._ 02 success rate. Lower is better. 

erage time to 98% _±_ 0 _._ 02 success from 48 _._ 97 _±_ 0 _._ 99 to 20 _._ 55 _±_ 0 _._ 60 hours. To understand how this speedup decomposes, Fig. 4b-c separate per-step compute from overall convergence. PCM reduces 

7 



Figure 4: Efficiency on LIBERO-Object at _B_ =12. **(a)** Wall-clock time to first reach each SR threshold; the PCM/GRPO gap widens at higher thresholds. **(b)** Activation memory is reduced by 60%. **(c)** Cumulative actor-update time over 200 steps is 4 _._ 8 _×_ faster. 

activation memory by 60% (10 _._ 1 _→_ 4 _._ 1 GB) and peak GPU memory by 15% (39 _._ 7 _→_ 33 _._ 6 GB) per step. 

Cumulative gradient-update time over 200 training steps is 4 _._ 8 _×_ faster. Because the per-step learning curves are matched (Fig. 3b), the wall-clock gain is entirely attributable to per-step compute savings rather than improved sample efficiency. Together these results show that the _selective gradient allocation substantially reduces per-step cost while leaving sample efficiency intact_ . 



Figure 5: Ablations on LIBERO-Object. **(a)** Chunk budget sweep shows that _B_ =12 preserves final success while retaining most of the wall-clock speedup. **(b)** At fixed _B_ =12, PCM outperforms random masking and highest-variance-phase selection. 

**RQ3: Sensitivity to Chunk Budget** _B_ **.** PCM samples a fixed budget of _B_ chunks per trajectory. Smaller budgets improve per-step efficiency but reduce gradient coverage, while larger budgets better approximate full GRPO at higher cost. We sweep B _∈_ {8, 12, 16} on LIBERO-Object using the same training configuration. We select this range for _B_ to bracket the inflection of the cumulative _Cc_ curve in Fig. 8b, plotted against the fraction of trajectory chunks retained, which marks the smallest budget capturing most of the available learning signal. Details in Appendix D. 

Fig. 5a shows that all budgets reach comparable final accuracy, but differ in efficiency. _B_ =8 is fastest per step but has degraded sample efficiency and plateaus at a slightly lower accuracy, indicating insufficient gradient signal. _B_ =16 provides no accuracy gain over _B_ =12 while increasing convergence time, showing diminishing returns beyond high- _Cc_ phases. _B_ =12 lies at the inflection point, matching final accuracy while preserving the wall-clock advantage. We use _B_ =12 as our default for all other experiments. _B_ =12 is large enough to consistently sample from high- _Cc_ phases under the probabilistic selection rule while remaining small enough to deliver substantial per-step compute and memory savings. 

**RQ4: Variance Concentration and the Role of Variance-Aware Selection.** To test whether variance-aware probabilistic selection is necessary, we compare PCM at _B_ =12 against two ablations that occupy opposite extremes of the allocation spectrum: _random masking_ , which samples _B_ chunks uniformly at random from each trajectory (no concentration on high- _Cc_ phases), and _full masking_ , which restricts updates to the single highest- _Cc_ phase and discards all other chunks (maximum concentration, no exploration). Together, these bracket PCM, which lies between these extremes via probabilistic _Cc_ -weighted sampling. 

8 

Fig. 5b shows that both ablations underperform PCM. Random masking plateaus at _∼_ 78% success rate, 22 points below PCM at the same chunk budget. Full masking fails more sharply, plateauing at a low accuracy of _∼_ 43%. Since all methods use the same _B_ =12 budget, these gaps isolate the contribution of variance-aware selection from compute reduction. 

Fig. 6 shows PCM’s realized gradient allocation across training, revealing three mechanisms that together explain its performance. 

• **Concentration:** Active-grip ( _∼_ 7–8 chunks per trajectory) and pre-grasp ( _∼_ 3–5 chunks per trajectory) dominate throughout, matching the high- _Cc_ phases (Fig. 1) and the<sup>_√_</sup> _Vc_ -weighted allocation predicted by Theorem 1. The agreement between predicted and realized allocation is the empirical confirmation that _Cc_ tracks _Vc_ in practice, validating Lemma 2’s proxy assumption. 

• **Exploration:** Probabilistic _Cc_ -weighted sampling (Eq. 10) ensures that every phase has nonzero selection probability whenever its weight is above zero, Figure 6: Phase-wise gradient allocation unproviding the primary exploration mechanism. The der PCM over training steps. _p_ min = 0 _._ 1 floor (Eq. 9) acts as a safety net for lowscoring phases (e.g., _tail_ ): it prevents transient zero estimates from collapsing a phase’s weight to zero between buffer recomputes, which would otherwise cause online phase scores to fail to recover when the true _Vc_ ordering shifts during training. 

Figure 6: Phase-wise gradient allocation under PCM over training steps. 

• **Adaptation:** Approach and release-ramp allocations decline from _∼_ 2 to _∼_ 1 chunks and from _∼_ 2.5 to _∼_ 1.5 chunks, respectively, as the policy masters those phases, with budget shifting toward phases where learning signal concentrates. This online dynamic that adapts with evolving success-failure gap of the policy cannot be reproduced by static weighting. 

The ablations occupy degenerate corners of this design space. Random masking explores but lacks concentration, matching the suboptimal uniform-allocation regime in Theorem 1, where _bc_ = _B/K_ (equal budget across phases) is inefficient when _Vc_ is non-uniform across phases. Full masking concentrates updates on the highest- _Cc_ phase but discards weaker adjacent-phase signals and prevents exploration and adaptation. 

Only PCM has all three properties: _Cc_ -weighted keep probabilities with a _p_ min floor to concentrate updates on contact-rich phases while retaining lower-scoring phases. The ablations show that varianceaware probabilistic selection substantially outperforms uninformed masking strategies in exploiting the concentrated structure of _Cc_ . 

### **6 Limitations and Future Work** 

Our evaluation spans the three LIBERO suites (Object, Spatial, Goal), which test transfer of object, spatial, and task knowledge respectively (Table 1). We do not test longer horizons or bimanual coordination; however, the underlying allocation principle applies whenever _Vc_ is non-uniform, a structural feature of any GRPO setting where the policy has not converged uniformly. Experiments use a gripper-based phase partition rule. Extension to other settings is straightforward in principle (any consistent decomposition suffices), though we do not validate alternative phase abstractions empirically. We do not directly benchmark against methods targeting sample efficiency through different objectives (e.g., entropy-based token selection or prompt filtering). These methods differ from PCM in both granularity (token vs. phase) and signal (policy-internal uncertainty vs. outcomegrounded divergence); adapting them to the VLA chunk-level setting is beyond the scope of this work. 

Future work includes extending PCM to longer-horizon and bimanual tasks, validating learned temporal phase segmenters or LLM-derived phase labelers, and applying the variance-allocation principle to LLM reasoning where _Cc_ can be computed against verified outcomes. Combining PCM with orthogonal sample-efficiency methods remains an open direction. 

9 

### **References** 

- Armen Aghajanyan, Luke Zettlemoyer, and Sonal Gupta. Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning. In _Annual Meeting of the Association for Computational Linguistics_ , 2021. doi: 10.18653/v1/2021.acl-long.568. URL `https://arxiv.org/abs/2012.13255v1` . 

- Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Lucy Xiaoyang Shi, James Tanner, Quan Vuong, Anna Walling, Haohuan Wang, and Ury Zhilinsky. _π_ 0: A Vision-Language-Action Flow Model for General Robot Control, 2024. URL `https://arxiv.org/abs/2410.24164v4` . 

- Léon Bottou, Frank E. Curtis, and Jorge Nocedal. Optimization methods for large-scale machine learning. _SIAM Review_ , 60(2):223–311, 2018. doi: 10.1137/16M1080173. 

- Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. Training deep nets with sublinear memory cost. abs/1604.06174, 2016. URL `http://arxiv.org/abs/1604.06174` . 

- Yuhui Chen, Shuai Tian, Shugao Liu, Yingting Zhou, Haoran Li, and Dongbin Zhao. ConRFT: A Reinforced Fine-tuning Method for VLA Models via Consistency Policy, 2025a. URL `https://arxiv.org/abs/ 2502.05450v2` . 

- Zengjue Chen, Runliang Niu, He Kong, Qi Wang, Qianli Xing, and Zipei Fan. TGRPO :Fine-tuning VisionLanguage-Action Model via Trajectory-wise Group Relative Policy Optimization, 2025b. URL `https: //arxiv.org/abs/2506.08440` . 

- Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, Quoc V. Le, Sergey Levine, and Yi Ma. SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Posttraining. In _International Conference on Machine Learning_ , 2025. doi: 10.48550/arxiv.2501.17161. URL `https://arxiv.org/abs/2501.17161v2` . 

- Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. QLoRA: Efficient Finetuning of Quantized LLMs. In _Advances in Neural Information Processing Systems_ , 2023. doi: 10.48550/arxiv.2305. 14314. URL `https://arxiv.org/abs/2305.14314v1` . 

- Yanjiang Guo, Jianke Zhang, Xiaoyu Chen, Xiang Ji, Yen-Jen Wang, Yucheng Hu, and Jianyu Chen. Improving Vision-Language-Action Model with Online Reinforcement Learning. In _IEEE International Conference on Robotics and Automation_ , 2025. URL `https://arxiv.org/abs/2501.16664v1` . 

- Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-Rank Adaptation of Large Language Models. In _International Conference on Learning Representations_ , 2021. URL `https://arxiv.org/abs/2106.09685v2` . 

- Jiaheng Hu, Rose Hendrix, Ali Farhadi, Aniruddha Kembhavi, Roberto Martin-Martin, Peter Stone, Kuo-Hao Zeng, and Kiana Ehsani. FLaRe: Achieving Masterful and Adaptive Robot Policies with Large-Scale Reinforcement Learning Fine-Tuning. In _IEEE International Conference on Robotics and Automation_ , 2024. doi: 10.1109/icra55743.2025.11127934. URL `https://arxiv.org/abs/2409.16578v2` . 

- Yuelin Hu, Zhengxue Cheng, Wei Liu, and Li Song. Entropy-Gated Selective Policy Optimization:Token-Level Gradient Allocation for Hybrid Training of Large Language Models, 2026. URL `https://arxiv.org/ abs/2602.03309` . 

- Zengjie Hu, Jiantao Qiu, Tianyi Bai, Haojin Yang, Binhang Yuan, Qi Jing, Conghui He, and Wentao Zhang. VADE: Variance-Aware Dynamic Sampling via Online Sample-Level Difficulty Estimation for Multimodal RL, 2025. URL `https://arxiv.org/abs/2511.18902` . 

- Mykola Khandoga, Rui Yuan, and Vinay Kumar Sankarapu. Beyond Uniform Credit: Causal Credit Assignment for Policy Optimization, 2026. URL `https://arxiv.org/abs/2602.09331` . 

- Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. OpenVLA: An Open-Source Vision-LanguageAction Model. In _Conference on Robot Learning_ , 2024. doi: 10.48550/arxiv.2406.09246. URL `https: //arxiv.org/abs/2406.09246v3` . 

- Moo Jin Kim, Chelsea Finn, and Percy Liang. Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success. In _Robotics: Science and Systems_ , 2025. URL `https://arxiv.org/abs/2502.19645` . 

10 

- Vijay Anand Korthikanti, Jared Casper, Sangkug Lym, Lawrence McAfee, Michael Andersch, Mohammad Shoeybi, and Bryan Catanzaro. Reducing Activation Recomputation in Large Transformer Models. In _Proceedings of the Machine Learning and Systems Conference (MLSys)_ , 2023. URL `https://openreview. net/forum?id=nvRncppDoD` . 

- Haozhan Li, Yuxin Zuo, Jiale Yu, Yuhao Zhang, Zhaohui Yang, Kaiyan Zhang, Xuekai Zhu, Yuchen Zhang, Tianxing Chen, Ganqu Cui, Dehui Wang, Dingxiang Luo, Yuchen Fan, Youbang Sun, Jia Zeng, Jiangmiao Pang, Shanghang Zhang, Yu Wang, Yao Mu, Bowen Zhou, and Ning Ding. SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning, 2025a. URL `https://arxiv.org/abs/2509.09674v1` . 

- Hengtao Li, Pengxiang Ding, Runze Suo, Yihao Wang, Zirui Ge, Dongyuan Zang, Kexian Yu, Mingyang Sun, Hongyin Zhang, Donglin Wang, and Weihua Su. VLA-RFT: Vision-Language-Action Reinforcement Finetuning with Verified Rewards in World Simulators, 2025b. URL `https://arxiv.org/abs/2510.00406` . 

- Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning. In _Advances in Neural Information Processing Systems_ , 2023. URL `https://arxiv.org/abs/2306.03310` . 

- Jijia Liu, Feng Gao, Bingwen Wei, Xinlei Chen, Qingmin Liao, Yi Wu, Chao Yu, and Yu Wang. What Can RL Bring to VLA Generalization? An Empirical Study, 2025. URL `https://arxiv.org/abs/2505.19789v4` . 

- Xiaokang Liu, Zechen Bai, Hai Ci, Kevin Yuchen Ma, and Mike Zheng Shou. World-VLA-Loop: Closed-Loop Learning of Video World Model and VLA Policy, 2026. URL `https://arxiv.org/abs/2602.06508` . 

- Guanxing Lu, Wenkai Guo, Chubin Zhang, Yuheng Zhou, Haonan Jiang, Zifeng Gao, Yansong Tang, and Ziwei Wang. VLA-RL: Towards Masterful and General Robotic Manipulation with Scalable Reinforcement Learning, 2025. URL `https://arxiv.org/abs/2505.18719v1` . 

- R. Duncan Luce. _Individual Choice Behavior: A Theoretical Analysis_ . Wiley, New York, 1959. 

- Qijun Luo, Mengqi Li, Lei Zhao, and Xiao Li. StreamBP: Memory-Efficient Exact Backpropagation for Long Sequence Training of LLMs. In _Advances in Neural Information Processing Systems_ , 2025. URL `https://arxiv.org/abs/2506.03077v1` . 

- Gongrui Nan, Siye Chen, Jing Huang, Mengyu Lu, Dexun Wang, Chunmei Xie, Weiqi Xiong, Xianzhou Zeng, Qixuan Zhou, Yadong Li, and Xingzhong Xu. NGRPO: Negative-enhanced Group Relative Policy Optimization, 2025. URL `https://arxiv.org/abs/2509.18851` . 

- R. L. Plackett. The analysis of permutations. 24(2):193–202, 1975. doi: 10.2307/2346567. 

- John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal Policy Optimization Algorithms, 2017. URL `https://arxiv.org/abs/1707.06347` . 

- Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models, 2024. URL `https://arxiv.org/abs/2402.03300v3` . 

- Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. HybridFlow: A Flexible and Efficient RLHF Framework. In _Proceedings of the Twentieth European Conference on Computer Systems_ , EuroSys ’25, page 1279–1297, New York, NY, USA, 2025. Association for Computing Machinery. ISBN 9798400711961. doi: 10.1145/3689031.3696075. URL `https://doi.org/10.1145/3689031.3696075` . 

- Shuhan Tan, Kairan Dou, Yue Zhao, and Philipp Krähenbühl. Interactive Post-Training for Vision-LanguageAction Models, 2025. URL `https://arxiv.org/abs/2505.17016v1` . 

- Hongcheng Wang, Yinuo Huang, Sukai Wang, Guanghui Ren, and Hao Dong. Why Tree-Style Branching Matters for Thought Advantage Estimation in GRPO, 2026. URL `https://arxiv.org/abs/2509.24494` . 

- Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xionghui Chen, Jianxin Yang, Zhenru Zhang, Yuqiong Liu, An Yang, Andrew Zhao, Yang Yue, Shiji Song, Bowen Yu, Gao Huang, and Junyang Lin. Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning. In _Advances in Neural Information Processing Systems_ , 2025. URL `https: //arxiv.org/abs/2506.01939` . 

- Zhongwen Xu and Zihan Ding. Single-stream Policy Optimization. In _International Conference on Learning Representations_ , 2026. URL `https://openreview.net/forum?id=b61UW62K7W` . 

11 

- Angen Ye, Zeyu Zhang, Boyuan Wang, Xiaofeng Wang, Dapeng Zhang, and Zheng Zhu. VLA-R1: Enhancing Reasoning in Vision-Language-Action Models, 2025. URL `https://arxiv.org/abs/2510.01623` . 

- Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, and Mingxuan Wang. DAPO: An Open-Source LLM Reinforcement Learning System at Scale. In _Advances in Neural Information Processing Systems_ , 2025. URL `https: //arxiv.org/abs/2503.14476v2` . 

- Hongzhi Zang, Mingjie Wei, Si Xu, Yongji Wu, Zhen Guo, Yuanqing Wang, Hao Lin, Peihong Wang, Liangzhi Shi, Yuqing Xie, Zhexuan Xu, Zhihao Liu, Kang Chen, Wenhao Tang, Quanlu Zhang, Weinan Zhang, Chao Yu, and Yu Wang. RLinf-VLA: A Unified and Efficient Framework for Reinforcement Learning of Vision-Language-Action Models, 2026. URL `https://arxiv.org/abs/2510.06710` . 

- Shaopeng Zhai, Qi Zhang, Tianyi Zhang, Fuxian Huang, Haoran Zhang, Ming Zhou, Shengzhe Zhang, Litao Liu, Sixu Lin, and Jiangmiao Pang. A Vision-Language-Action-Critic Model for Robotic Real-World Reinforcement Learning, 2025. URL `https://arxiv.org/abs/2509.15937` . 

- Jun Zhang, Jue Wang, Huan Li, Lidan Shou, Ke Chen, Yang You, Guiming Xie, Xuejian Gong, and Kunlong Zhou. Train Small, Infer Large: Memory-Efficient LoRA Training for Large Language Models. In _International Conference on Learning Representations_ , 2025. URL `https://arxiv.org/abs/2502.13533v2` . 

- Qingru Zhang, Minshuo Chen, Alexander Bukharin, Nikos Karampatziakis, Pengcheng He, Yu Cheng, Weizhu Chen, and Tuo Zhao. AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning. In _International Conference on Learning Representations_ , 2023. URL `https://arxiv.org/abs/2303.10512` . 

- Haizhong Zheng, Yang Zhou, Brian R. Bartoldson, Bhavya Kailkhura, Fan Lai, Jiawei Zhao, and Beidi Chen. Act Only When It Pays: Efficient Reinforcement Learning for LLM Reasoning via Selective Rollouts. In _Advances in Neural Information Processing Systems_ , 2025. URL `https://openreview.net/forum?id= x5lITYXmW2` . 

- Fangqi Zhu, Zhengyang Yan, Zicong Hong, Quanxin Shou, Xiao Ma, and Song Guo. WMPO: World Modelbased Policy Optimization for Vision-Language-Action Models, 2025. URL `https://arxiv.org/abs/ 2511.09515v1` . 

12 

## **Appendix** 

### **A Problem Formulation and Assumptions** 

For clarity, we explicitly restate the problem formulation and assumptions underlying PCM. While these are implicit in the main text, we formalize them here to precisely define the input setting, the chunk-selection objective, and the structural conditions required for the analysis. 

**Input.** A VLA policy _πθ_<sup>0initializedfromasupervisedfine-tuningcheckpoint,asimulateden-</sup> vironment _E_ , a distribution over tasks _T_ with binary reward _r_ : _τ →{_ 0 _,_ 1 _}_ , and a group of _G_ rollouts _{τi}_<sup>_G_</sup> _i_ =1<sup>where each trajectory decomposes into</sup><sup>_Ni_=</sup><sup>_⌈Ti/L⌉_chunks of length</sup><sup>_L_,and a</sup> fixed gradient compute budget _B ≪_<sup>�</sup> _i_<sup>_Ni_.</sup> 



subject to _|Ki|_ = _B_ for all _i_ (fixed compute budget), and _M_ using only _{τi, ri}_<sup>_G_</sup> _i_ =1<sup>without access to</sup> auxiliary reward models or learned critics. 

**Assumptions.** We assume the following: 

- **Binary reward:** _r_ : _τ →{_ 0 _,_ 1 _}_ is trajectory-level, with each group containing both successes and failures (0 _< µr <_ 1); when all rollouts succeed or fail, advantages collapse and the variance signal is uninformative. 

- **Fixed chunk length:** The policy produces actions in chunks of fixed length _L_ , so the phase decomposition _ϕ_ ( _i, k_ ) is well-defined and consistent across trajectories. 

- **Deterministic phase labeling:** _ϕ_ ( _i, k_ ) _∈P_ is computable from the gripper command trajectory alone, requiring no learned model. 

- **Shared phase structure:** Successful and failed rollouts share the same phase partition _P_ , so per-phase action variance compares like with like across outcome groups. 

- **SFT initialization:** _πθ_<sup>0is initialized from a supervised fine-tuning checkpoint, ensuring gradient</sup> variance concentrates on a small subset of outcome-critical phases. 

### **B Theoretical Results** 

In this section we provide proofs for the formal results in the main paper. 

**Lemma 1** (Phase gradient variance) **.** _Let c ∈P be a phase in which πθ has converged, i.e., the action distributions of successful and failed rollouts are identical in phase c. Then ∥gc_ ( _θ_ ) _∥≈_ 0 _and Vc ≈_ 0 _, so gradient samples from phase c contribute negligible signal. Conversely, for a phase c in which the base policy is underspecified, Vc is large and gradient samples from phase c are informative._ 

_Proof._ Since _Ai_ is zero-mean by construction (Eq. (1)), we have _Vc_ = 0 if and only if _πθ_ ( _ai,k | si,k_ ) is identical across successful and failed rollouts for every chunk ( _i, k_ ) with _φ_ ( _i, k_ ) = _c_ . This holds exactly when outcomes are independent of the actions taken in phase _c_ . When these conditional action distributions match, positive- and negative-advantage score-function terms are drawn from the same distribution and cancel in expectation, leaving _∥gc_ ( _θ_ ) _∥≈_ 0. When they diverge, phase _c_ carries useful learning signal. 

**Lemma 2** ( _Cc_ as a proxy for _Vc_ ) **.** _For a policy πθ that is locally Gaussian with per-dimension variance σπ_<sup>2</sup><sup>_, the per-phase gradient variance satisfies Vc≥C_</sup> _c_<sup>2</sup><sup>_/_4</sup><sup>_σ_</sup> _π_<sup>2</sup><sup>_._</sup> 

_Proof._ Let _µ_<sup>+</sup> _c_<sup>=E[</sup><sup>_ai,k|ri_=1</sup><sup>_,φ_(</sup><sup>_i, k_)=</sup><sup>_c_]and</sup><sup>_µ−_</sup> _c_<sup>=E[</sup><sup>_ai,k|ri_=0</sup><sup>_,φ_(</sup><sup>_i, k_)=</sup><sup>_c_].For</sup> a Gaussian policy _πθ_ ( _· | s_ ) = _N_ ( _µθ_ ( _s_ ) _, σπ_<sup>2</sup><sup>_I_),thescorefunctionis</sup><sup>_∇θ_log</sup><sup>_πθ_(</sup><sup>_a|s_)=(</sup><sup>_a−_</sup> 

13 

_µθ_ ( _s_ )) _/σπ_<sup>2</sup><sup>_·∇θµθ_(</sup><sup>_s_).Conditioning on the outcome group, the variance of the score is lower-bounded</sup> by the variance of the mean action across groups. Specifically, for any random variable _X_ and a binary conditioning event _E_ , Var( _X_ ) _≥_ Var(E[ _X | E_ ]) by the law of total variance. Applying this with _X_ = _Ai · ∇θ_ log _πθ_ ( _ai,k | si,k_ ) and _E_ = _{ri_ = 1 _}_ versus _{ri_ = 0 _}_ , and using _|Ai| ≤_ 1 _/ε_ as a finite bound, yields the equation in the lemma after absorbing the gradient norm _∥∇θµθ∥_<sup>2</sup> and balanced-group constants into _σπ_<sup>2.The bound preserves the relative ordering of</sup><sup>_{Vc}_across</sup> phases. 

**Theorem 1** (Optimal phase allocation) **.** _Let bc denote the number of chunks sampled from phase c, with_<sup>�</sup> _c∈P_<sup>_bc_=</sup><sup>_B, and let Ncdenote the expected number of chunks in phase c per trajectory.The_</sup> _allocation minimizing the variance of the unbiased ratio estimator of the GRPO gradient (Eq. (2)) subject to the budget constraint is_ 



_Proof._ We analyze the idealized stratified ratio estimator with deterministic per-phase budget bc; the masked loss in Eq. (11) implements this allocation in expectation under the sampling rule of Eq. (10), at the cost of a finite-sample bias bounded by the contribution of low-Vc phases (Sec. 4.5). The GRPO gradient decomposes by phase as in Eq. (3). PCM estimates the full phase gradient _gc_ ( _θ_ ), which sums over all _Nc_ chunks in phase _c_ , via the ratio estimator 



where _Kc ⊆{k_ : _φ_ ( _i, k_ ) = _c}_ with _|Kc|_ = _bc_ chunks drawn uniformly without replacement from the _Nc_ available chunks in phase _c_ . The _Nc/bc_ factor scales up the sampled sum to account for the full phase size, making ˆ _gc_ unbiased for _gc_ ( _θ_ ): 



The variance of ˆ _gc_ is 



where _Vc_ is the per-chunk variance within phase _c_ (Eq. (4)), and the _bc_ in the numerator is the variance of a single-chunk sample, reduced by _bc_ draws. 

The total estimator variance across all phases is 



We minimize _σ_<sup>2</sup> ( _{bc}_ ) subject to<sup>�</sup> _c∈P_<sup>_bc_=</sup><sup>_B_,</sup><sup>_bc≥_0.By the Cauchy-Schwarz inequality,</sup> 



with equality if and only if _Nc√Vc/_<sup>_√_</sup> _bc ∝_<sup>_√_</sup> _bc_ , i.e., _bc ∝ Nc√Vc_ . Combining with<sup>�</sup> _c_<sup>_bc_=</sup><sup>_B_</sup> yields Eq. (13). Substituting _b_<sup>_∗_</sup> _c_<sup>=</sup><sup>_B · Nc_</sup> _√Vc/_<sup>�</sup> _c_<sup>_′ Nc′√_</sup> _Vc′_ into Eq. (16), the minimum total variance achieved is 



14 

#### **B.1 Convergence Derivation for Phase Allocation** 

We derive the convergence-rate comparison between the optimal allocation of Theorem 1 and uniform allocation. Under standard SGD assumptions (L-smooth objective, bounded per-phase gradient variance _Vc_ ), the number of steps to reach an _ε_ -stationary point satisfies [Bottou et al., 2018] 



where _σ_<sup>2</sup> is the variance of the gradient estimator at sample budget _B_ ; the 1 _/B_ variance reduction from _B_ -sample averaging is incorporated into _σ_<sup>2</sup> via the per-phase variance formulas. 

**Optimal allocation.** From Theorem 1 and Eq. (17), the minimum achievable estimator variance under budget _B_ is 



Substituting into Eq. (18): 



**Uniform allocation.** Under _bc_ = _B/K_ for all _K_ phases, the total estimator variance is 



Substituting into Eq. (18): 



**Speedup ratio.** Dividing Eq. (22) by Eq. (20): 



By the Cauchy-Schwarz inequality, 



so ∆ _≥_ 1, with equality if and only if all _Nc_<sup>2</sup><sup>_Vc_are equal across phases.When variance concentrates</sup> on few phases such that max _c Nc_<sup>2</sup><sup>_Vc≫_</sup> _K_ <u>1</u> � _c_<sup>_N_</sup> _c_<sup>2</sup><sup>_Vc_, then ∆</sup><sup>_≫_1:the optimal allocation yields</sup> substantially fewer convergence steps than uniform allocation at the same chunk budget _B_ . 

#### **B.2 Bias Analysis of the Masked Estimator** 

The masked objective in Eq. (11) omits the _Nc/bc_ rescaling of the unbiased estimator (Eq. 14), trading a small bias for lower variance. We bound this bias. 

**Lemma 3 (Bias of the masked estimator).** _Let pc denote the keep probability for phase c. The bias of the masked estimator relative to the unbiased estimator satisfies_ 



15 

_Proof._ Under weighted sampling without replacement with inclusion probability _pc_ , the expected contribution of phase _c_ to the masked estimator is _pc · gc_ , while the unbiased estimator yields _<u>gc</u>_ . Summing across phases and applying the triangle inequality gives the bound. 

Under the allocation _pc ∝_<sup>_√_</sup> _Vc_ , (1 _− pc_ ) is large only when _Vc_ is small, and by Lemma 1, small _Vc_ implies _∥gc∥→_ 0. Each term in the bound is therefore suppressed by at least one small factor, so Theorem 1’s allocation remains approximately optimal under the biased estimator. 

### **C Methods We Tried That Did Not Work** 

The motivating observation behind this paper was that VLA rollouts on a given task are remarkably similar across trajectories. Most timesteps execute nearly identical motion patterns regardless of whether the trajectory eventually succeeds. A reaching phase looks like a reaching phase across all rollouts; the divergence concentrates on a small number of decision points. This led us to hypothesize that gradient compute was being misallocated: the same per-step budget was being spent on phases that contributed little to the policy update as on phases that determined the outcome. Before arriving at PCM, we explored two alternative directions for exploiting this observation. Both failed, and the failures shaped the eventual design. 

#### **C.1 Branching at Decision Critical Timesteps** 

Our initial approach attempted to increase the rollout signal at high-uncertainty points by branching: at each candidate decision point, fork the rollout into K alternative continuations to give GRPO more contrastive signal in the regions that mattered. Two problems made this infeasible. First, each branch required actually executing a downstream rollout in the simulator to be useful for GRPO, which meant returning to the branch point and continuing the simulation; this added significant rollout-side overhead and partially undid the efficiency motivation. Second, naive branching is exponential in depth, refining beyond the first branch point requires nested branching, and the rollout count grows as _K_<sup>_d_</sup> where d is the number of branches per trajectory. We found no principled way to bound depth without re-introducing the same uniform-allocation problem we were trying to escape. 



Figure 7: (a) Success vs. training steps for vanilla GRPO vs branching GRPO. (b) Action entropy over trajectory progress at different stages of training, showing weak alignment with outcome-critical phases. The training and evaluation follow the same procedures as Sec. 5.2 

The lesson from this attempt was that any practical method had to operate within the existing rollout structure rather than expanding it. PCM instead reduces gradient computation over rollouts that have already been generated. 

#### **C.2 Entropy as a Signal to Concentrate Learning** 

Our second attempt used per-chunk policy entropy as a proxy for "where the model has room to learn." The hypothesis was the standard one: high-entropy chunks indicate uncertainty, uncertainty indicates a learning opportunity, and concentrating gradient on these chunks should yield faster convergence. We expected that as training progressed, entropy would decrease in initially-uncertain chunks and the gradient allocation would naturally shift toward the remaining noisy regions. 

Two empirical findings led us to abandon this approach. First, entropy was poorly structured across trajectories: we did not observe consistent concentration on decision-critical phases the way we 

16 



Figure 8: Cumulative _Cc_ captured as a function of the fraction of trajectory chunks retained, with chunks sorted in descending order by phase-level _Cc_ . The solid curve shows the empirical cumulative _Cc_ , while the dashed diagonal shows the uniform baseline under which _Cc_ would scale linearly with chunk count. The gap between the two reflects the uneven distribution of learning signal across the trajectory: a small fraction of chunks captures a disproportionate share of the total _Cc_ . 

eventually observed for _Cc_ (Figure 1). Per-chunk entropy varied substantially across rollouts even within the same task and phase, and the high-entropy chunks did not align with the points where successful and failed rollouts actually diverged. We attribute this to the fact that VLA action distributions are typically already low-entropy after SFT. The supervised pretraining drives the policy toward confident, low-entropy actions, and the residual entropy reflects modeling noise rather than genuine multimodal uncertainty over correct behavior. Second, and more decisively, entropy did not meaningfully decrease over training. Across multiple runs on LIBERO, we measured per-chunk entropy at the start and end of GRPO fine-tuning and found negligible reduction. This is consistent with the known property that GRPO is a relatively weak signal for changing the underlying action distribution: the group-relative advantage primarily reweights existing modes rather than collapsing the policy onto a sharper one. Entropy under SFT-initialized policies is therefore close to a fixed property of the model rather than a quantity RL can drive down. Selecting on entropy would mean repeatedly allocating gradient to the same chunks throughout training regardless of whether they had become well-learned, and would not adapt to the shifting locus of the success–failure gap as training progressed. 

### **D Analysis of Chunk Budget** 

**The compute-signal trade-off.** Choosing _B_ reflects a trade-off between two opposing pressures. Lowering _B_ reduces per-step gradient computation linearly: fewer chunks in the backward pass reduce activation memory and accelerate actor updates, which is the source of PCM’s wall-clock advantage. Increasing _B_ captures more of the available learning signal: a larger budget includes more chunks from outcome-divergent phases and approaches the gradient quality of full-trajectory GRPO. The cumulative _Cc_ curve in Fig. 8 is sharply concave: early chunks, corresponding to high- _Cc_ phases, deliver disproportionate signal, while additional chunks beyond the knee contribute progressively less. The _knee_ of this curve, the point at which marginal _Cc_ capture per chunk drops sharply, occurs at approximately 20% of trajectory chunks. This concavity makes the trade-off non-trivial. Below the knee, each additional chunk yields substantial signal at fixed compute cost, so excluding it sacrifices accuracy for marginal speedup. Above the knee, each additional chunk yields diminishing signal at the same cost, so including it sacrifices speedup for marginal accuracy. The knee marks the point where these pressures balance: the smallest _B_ for which captured _Cc_ has largely saturated and further increases primarily incur additional compute cost. 

**Budget range selection.** The cumulative _Cc_ curve in Fig. 8 is sharply concave, with a knee at approximately 20% of trajectory chunks. We compute captured _Cc_ by ranking chunks according 

17 

to their phase score and measuring the fraction of total chunk-level score mass retained by the top _B_ chunks. Beyond the knee, each additional chunk yields progressively smaller marginal gains in captured _Cc_ . We select _B ∈{_ 8 _,_ 12 _,_ 16 _}_ to bracket this knee, corresponding to 12%, 19%, and 25% of trajectory chunks, respectively. 

- _B_ = 8 (12% of chunks, 50% of _Cc_ captured): Below the knee. Tests whether aggressive truncation that falls short of the inflection leads to signal under-coverage and degraded final accuracy. 

- _B_ = 12 (19% of chunks, 59% of _Cc_ captured): At the knee. Tests the predicted operating point where _Cc_ capture is high and marginal returns begin to flatten. 

- _B_ = 16 (25% of chunks, 64% of _Cc_ captured): Beyond the knee. Tests whether additional budget yields meaningful gains. The extra 5 percentage points of _Cc_ capture over _B_ = 12 come at substantially higher per-step compute. 

The empirical sweep in Sec. 5.2 (RQ3) confirms this curvature: _B_ = 8 underperforms in final accuracy, _B_ = 16 matches _B_ = 12 but at higher cost, and _B_ = 12 emerges as the operating point consistent with the knee of the cumulative _Cc_ curve. 

**Choosing** _B_ **for other domains.** The knee of the cumulative _Cc_ curve provides a principled approximation for _B_ in any RL setting with a defined phase partition. The reasoning generalizes directly from the compute–signal trade-off: _Cc_ is computed from rollouts that GRPO already produces, so the curve can be measured in any domain where outcomes are verifiable and rollouts decompose into phases. The structural assumptions of the method are therefore not specific to manipulation. The knee identifies the operating point at which the two pressures balance, independent of the absolute magnitude of _Cc_ or the number of phases, because it is determined by the curvature of the signal. 

The procedure is the same across domains. For a target domain, compute _Cc_ per phase from a single rollout group, plot the cumulative _Cc_ curve as in Fig. 8, and select _B_ at the knee, the smallest fraction of chunks at which captured _Cc_ has largely saturated. The phase partition itself is domain-specific, such as gripper-state heuristics for manipulation, semantic decomposition for LLM reasoning, or episode segmentation for dialogue. Once defined, the curve and its knee are computed identically. Domains with sharper curvature, where signal concentrates in a few phases, admit smaller _B_ and larger speedups. Flatter curves indicate more uniform signal and require larger _B_ to capture comparable _Cc_ , yielding smaller speedups. Only the resulting value of _B_ varies; the selection rule remains unchanged. 

### **E Implementation Details** 

This appendix provides additional experimental details omitted from the main text, including benchmark construction, training hyperparameters, PCM-specific settings, and evaluation protocol. Unless stated otherwise, PCM and the full-trajectory GRPO baseline use the same rollout, reward, advantage, optimizer, and evaluation configuration. 

#### **E.1 Benchmarks** 

**Models and Benchmarks.** We evaluate on three LIBERO suites to cover diverse manipulation regimes and test generalization. LIBERO-Object tests object-centric pick-and-place, LIBEROSpatial tests spatial-relation following, and LIBERO-Goal tests goal-conditioned manipulation. The manipulation tasks are specified by natural-language instructions. The following examples illustrate the types of prompts used across the benchmarks: 

- Object-centric manipulation: “pick up the mug and place it in the bowl.” 

- Spatial manipulation: “place the object on the left side of the tray.” 

- Goal-conditioned manipulation: “move the object to the target location.” 

The policy receives the language instruction and RGB observation and predicts a sequence of 7-DoF end-effector actions. These suites vary in object identity, spatial layout, goal specification, and task structure, bringing diversity in manipulation regimes and causing phase-level failure patterns to 

18 

appear with different frequencies across benchmarks. Each benchmark contains 10 tasks with 50 trials per task under different initial object configurations, yielding 500 evaluation trials per benchmark. Trajectories contain a total of 64 chunks. During RL fine-tuning, we use visual augmentations such as brightness changes and image jitter to improve robustness. 

**Training.** Our method is implemented based on the SimpleVLA-RL `verl` pipeline [Sheng et al., 2025]. We initialize from the SimpleVLA-RL 1-trajectory SFT checkpoint, where OpenVLA-OFT is supervised-fine-tuned with one demonstration per task. We then perform RL fine-tuning with LoRA rank _r_ =32 and _α_ =32 on 2 NVIDIA H100 GPUs. We use a prompt batch size of 8 and sample 10 rollouts per prompt, giving 80 trajectories per training step. The trajectory mini-batch size is 4, and the PPO mini-batch size is 2 with micro-batch size 1. We optimize the actor with AdamW using a constant learning rate of 1 _×_ 10<sup>_−_5</sup> , ( _β_ 1 _, β_ 2) = (0 _._ 9 _,_ 0 _._ 999), weight decay 0, and gradient clipping at 2 _._ 0. Following SimpleVLA-RL, we use one PPO epoch per update, asymmetric clipping ( _ϵ_ low _, ϵ_ high) = (0 _._ 2 _,_ 0 _._ 4), entropy coefficient 10<sup>_−_3</sup> , and disable the KL penalty. Rollouts are sampled with temperature 1 _._ 6, top- _p_ = 1 _._ 0, and top- _k_ = _−_ 1 without truncation. 

For PCM, we use a fixed chunk budget of _B_ =12 chunks per trajectory. Phase keep probabilities are initialized from the success–failure phase scores _Cc_ computed from the rollout group, and then recomputed every _T_ rc=5 training steps from the recent phase-score buffer. We clip keep probabilities below by _p_ min=0 _._ 1, ensuring that low-variance phases are not completely discarded and preserving a small amount of exploration throughout training. Non-selected chunks are physically removed before the actor forward and backward pass. PCM is simple to integrate into existing GRPO pipelines: it only adds phase-score computation from rollout statistics and a chunk-shrinking step before the actor update. We use the same PCM hyperparameters across all benchmarks without task-specific tuning. 

#### **E.2 Phase Allotment Rules** 

We assign trajectory chunks to semantic phases using a deterministic gripper-based heuristic. For each chunk _j_ , we compute _gf_ [ _j_ ] _∈_ [0 _,_ 1], the fraction of timesteps in the chunk where the gripper-close command is active. We use a sustained-close threshold _τ_ = 0 _._ 75 to locate grasp intervals, and then assign phase labels around those intervals. 

- **Active-grip.** Chunks with substantial gripper closure are labeled _active-grip_ . We use a softer threshold _gf_ [ _j_ ] _≥_ 0 _._ 5 for labeling, so this phase includes both sustained grasp chunks and nearby partial-close chunks. In successful rollouts, this covers grasp and transport while the object is held, or recovery after drop; in failed rollouts, it can also capture drops, re-grasp attempts, or repeated failed closure near the object. 

- **Pre-grasp.** Up to three chunks immediately before a sustained-close interval are labeled _pre-grasp_ when the gripper has begun closing but has not yet reached sustained closure (0 _._ 1 _≤ gf_ [ _j_ ] _<_ 0 _._ 5). This phase often captures final alignment and orientation adjustment before contact. 

- **Release-ramp.** Up to three chunks immediately after a sustained-close interval are labeled _release-ramp_ ( _gf_ [ _j_ ] _<_ 0 _._ 5), capturing the transition from holding the object to releasing it. This phase is important when failures occur because the object is released at the wrong location or does not fall into the target bin. 

- **Approach.** Chunks before the pre-grasp/active-grip region of a grasp cycle are labeled _approach_ ( _gf_ [ _j_ ] _<_ 0 _._ 5), corresponding to navigation toward the object. 

- **Tail** Chunks after the final release with an open gripper are labeled _tail_ ( _gf_ [ _j_ ] _<_ 0 _._ 1). These often capture post-release behavior, including cases where the object was not successfully placed and the policy continues hovering or moving after release or attempts. 

The unified phase set is resolved by priority: 

_active-grip > pre-grasp > release-ramp > approach > tail_ 

When multiple phase rules apply due to overlapping windows, we assign the highest-priority label. Phases are distinguished first by their position relative to sustained-close intervals and then by the gripper thresholds above: open-gripper chunks before a future grasp attempt are _approach_ , whereas open-gripper chunks after the final release-ramp window (after a sustained grasp interval) are _tail_ . 

19 



Figure 9: Successful and failed manipulation rollouts look similar for most of the trajectory, diverging mainly around outcome-critical grasp phases. 

Since labels are assigned at the chunk level, averaging the gripper command over the chunk reduces sensitivity to single-timestep noise and makes the thresholding rule more stable. 

The same rule is applied to successful and failed trajectories. The terminal success label is never used by the phase classifier, so _Cc_ compares success and failure trajectories under a shared phase partition. The labeling rule is lightweight, requires no learned phase model, and adds no inference overhead. Other phase allocators, such as task-specific heuristics, temporal semantic phase segmenters, or LLM-based labelers, could be used when gripper state is insufficient. 

**Phase-based failure localization:** Figure 9 illustrates how common failures fall naturally into this phase partition. In a missed-grasp failure, the robot may approach the object similarly to successful rollouts but close with poor alignment. The closing chunks are labeled _active-grip_ , while the immediately preceding orientation-adjustment chunks are labeled _pre-grasp_ . This explains why both pre-grasp and active-grip can receive high _Cc_ : the failure may be caused not only by the closure itself, but also by the pose and alignment right before contact. After such failures, however, many remaining chunks are downstream consequence states (hovering, searching, colliding, or moving without the object) which are typically labeled _tail_ . These chunks can occupy a large fraction of failed rollouts, but they are less causally informative than the contact error that produced them. PCM therefore keeps them with low nonzero probability while concentrating budget on the grasp phases that prevent the policy from entering these failed states. 

Other failures are captured by the same heuristic without additional case rules. If the object is grasped but later dropped or repeatedly re-grasped during transport, the corresponding close/hold chunks remain _active-grip_ , making this phase sensitive to transport instability. These chunks provide useful RL signal by penalizing slips, failed recovery and unstable holds while rewarding successful recovery. If the grasp succeeds but placement fails, the error appears around _release-ramp_ : the robot releases at the wrong location or the object misses the bin, making its _Cc_ moderately high. Subsequent _tail_ chunks mostly capture open-gripper hovering after release, and are therefore less outcome-discriminative and provide little learning signal. When computing _Cc_ , we compare only phases that contain both successful and failed samples in the rollout group; phases absent from one outcome group in a batch are skipped for that update. In the observed rollout groups, failures often diverge during active-grip, making it the most outcome-discriminative phase and causing _Cc_ to become largest there. This concentration emerges automatically from the gripper-based phase labels and rollout outcomes, without hand-designed phase weights or task-specific tuning. 

20 

