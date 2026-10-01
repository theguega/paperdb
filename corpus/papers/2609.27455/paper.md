# LATENT EVOLVING WORLD ACTION MODEL 

**Xueji Fang**<sup>1</sup><sup>_,_2</sup><sup>_,_3</sup> **Boqiang Duan**<sup>3</sup> **Hua Wu**<sup>3</sup> **Jingdong Wang**<sup>3</sup><sup>_,†,‡_</sup> **Guo-Jun Qi**<sup>2</sup><sup>_,†_</sup> 1Zhejiang University 2Westlake University 3Baidu Inc. 

ABSTRACT 

World Action Models (WAMs) jointly model action generation and environment dynamics and are mostly built on pretrained Video Diffusion Models (VDMs). In VDM-based WAMs, observations are first encoded by a VAE, and the resulting compressed latents are then processed by large video diffusion backbones to extract effective features for action generation. However, this paradigm ties WAM performance and training cost to large-scale video generation pretraining, limiting WAM efficiency and scalability. In this paper, we theoretically and empirically investigate how visual representations affect action generation in WAMs. Our results show that predictive embeddings from Joint-Embedding Predictive Architecture (JEPA) encoders better support action generation than compressed VAE latents, with I-JEPA performing best in our encoder comparison. Based on these findings, we propose LeWAM, which conditions action generation on JEPA embeddings and models environment evolution by predicting future embeddings in the same space, without relying on a video diffusion backbone. We further find that imitation learning matches demonstrated actions but does not distinguish better actions from worse ones, even though small action deviations can greatly affect task success. To address this limitation without additional environment interaction or the human oversight required for resets and safety, we introduce Demonstration-Guided DPO (DemoDPO), an offline preference refinement stage that derives preference supervision directly from demonstrations. With only 0.4B trainable parameters, LeWAM achieves an average success rate of 92.28% on RoboTwin 2.0, comparable to that of state-of-the-art VLAs and WAMs, and maintains practical effectiveness on real-world manipulation tasks. 



#### github.com/XuejiFang/LeWAM 



huggingface.co/XuejiFang/LeWAM 



<!-- Start of picture text -->
95 LeWAM FastWAM LingBot-VLA-v2 π 0.5 LeWAM<br>100<br>90 Reported VLA/WAM range 84.9<br>85 I-JEPA-H 80 75.7<br>V-JEPA2-H SigLIP<br>80 60.7<br>60 55.7<br>53.0<br>75 V-JEPA2-L 44.0 42.0 42.348.0<br>70 DINOv3-L 40<br>28.9<br>23.0<br>65 MAE-L 20 15.7<br>Wan2.2-VAE: 10.81<br>12 (data size not reported)<br>10<br>0<br>1M 10M 100M 1B 10B Stack Fold Arrange<br>Pretraining data size (log scale) Blocks Towel Flowers<br>(a) (b)<br>Task Progress (%)<br>Avg. success rate on RoboTwin 2.0<br><!-- End of picture text -->

Figure 1: _(a)_ **Frozen vision encoder comparison and LeWAM performance on RoboTwin 2.0.** Frozen encoders are evaluated with the same simple action predictor architecture and training recipe. I-JEPA-H, pretrained only on ImageNet-22K, performs best among the evaluated encoders, while LeWAM achieves an average success rate of 92.28%. _(b)_ **Real-world task progress across three manipulation tasks.** LeWAM achieves the highest task progress on all three tasks, outperforming _π_ 0 _._ 5, LingBot-VLA-v2, and FastWAM. 

> _†_ Corresponding authors _‡_ Project leader 

1 

## 1 INTRODUCTION 

Vision-Language-Action (VLA) models (Brohan et al., 2022; Belkhale et al., 2024; Kim et al., 2024; Wen et al., 2025; Fan et al., 2025; Physical Intelligence et al., 2025; Chen et al., 2025a; Black et al., 2026; GigaBrain Team et al., 2026; Zhou et al., 2025) have become a widely used paradigm for generalist robotic policies, mapping visual observations and language instructions directly to actions via behavioral cloning. By leveraging large pretrained vision-language backbones, VLAs generalize across language instructions, but they do not explicitly model how the physical world evolves under actions. World Action Models (WAMs) (Ye et al., 2026b;a; Ma et al., 2026; Yuan et al., 2026) address this limitation by coupling action prediction with visual dynamics learning, often using Video Diffusion Model (VDM) backbones to model how actions change the physical scene. These WAMs provide additional supervision for embodied policies and report competitive results on manipulation tasks. 

Despite these advances, recent WAMs reveal a tension between how they are trained and deployed. Some WAMs retain explicit joint video and action generation at inference time (Ye et al., 2026b; Bi et al., 2026; Ma et al., 2026), but action-centered WAMs increasingly remove future visual generation during deployment (Ye et al., 2026a; Yuan et al., 2026). These WAMs still benefit from video supervision during training, but at test time often generate actions directly from the current observation and task instruction. At deployment, they use VLA-style inference, predicting actions directly without forecasting future state changes. For VDM-based WAMs, the visual representation can therefore become a bottleneck for action generation: compressed VAE posterior means can be poorly suited to lightweight action predictors, and extracting useful features from them relies on a large pretrained video diffusion backbone, limiting efficiency and scalability. 

Prior studies provide evidence on visual representations for action generation, but their settings are not directly comparable. Existing encoder comparisons cover a limited set of representations or evaluate them as additions to VLA features (Wu et al., 2024; Egbe et al., 2025; Hu et al., 2024; Miao et al., 2026; Wang et al., 2025). Existing latent world models use predictive embeddings for goalconditioned planning rather than direct action generation (Assran et al., 2025; Maes et al., 2026). Existing VLAs and VDM-based WAMs rely on different visual pipelines, using pretrained VLM features and VAE latents processed by video diffusion backbones, respectively (Kim et al., 2024; Black et al., 2024; Physical Intelligence et al., 2025; Wen et al., 2025; Zheng et al., 2025; Ye et al., 2026b; Bi et al., 2026; Ma et al., 2026; Yuan et al., 2026; Ye et al., 2026a). None of these studies compares representative frozen encoders learned through different pretraining objectives as direct inputs to the same action predictor. These objectives include reconstruction (He et al., 2022; Chen et al., 2024), self-supervised feature learning (Caron et al., 2021; Zhang et al., 2019; Qi et al., 2019), vision-language alignment (Zhai et al., 2023), and image or video prediction (Assran et al., 2023; 2025; Zhang et al., 2023). Such a comparison is necessary to isolate the contribution of the visual representation from the policy and training procedure. 

We therefore analyze WAMs built on VDMs from the perspective of information theory and compare representative frozen encoders under the same training recipe. Our analysis separates the action information a representation retains from the part a downstream model can use, while the controlled comparison shows that compressed VAE latents are poorly suited to direct action generation and that I-JEPA performs best among the evaluated encoders. These findings motivate LeWAM, a **L** atent **e** volving **W** orld **A** ction **M** odel that performs both action generation and world modeling in JEPA embedding space. To jointly learn action generation and future embedding prediction conditioned on actions, LeWAM organizes the current embedding, noisy and demonstrated action tokens, and future embedding queries in a unified token sequence and controls the information flow with a structured attention mask. This design allows both objectives to be learned by a single predictor without information leakage. To further exploit the information distributed across the frozen encoder, we introduce AdaFuse, which adaptively combines representations from different encoder layers for each feature dimension. At inference time, LeWAM removes the demonstrated actions and future queries and generates actions only from the current observation and task context. 

We observe that LeWAM generates actions close to demonstrations on the validation set, yet small deviations during interaction with the environment can cause task failure. We further find that allowing the policy multiple attempts increases its success rate. This suggests that successful action trajectories lie within the policy distribution but receive insufficient probability mass. We therefore aim 

2 



<!-- Start of picture text -->
Inference<br>IDM<br>WM loss gradient WM<br>WAM<br>WM<br>Policy JEPA JEPA JEPA JEPA<br>WAM (VDM backbone) VLA Enc VLA Enc Enc cond Enc<br>VAE latent<br>cond cond cond<br>(a) (b) (c) (d)<br><!-- End of picture text -->

Figure 2: Comparison of WAM architectures. **(a) Action-centered VDM-based WAMs** use compressed VAE latents as visual inputs and jointly learn action generation and future frame prediction, but generate only actions at inference. **(b) LaWAM** (Chen et al., 2026) predicts latent actions from VLM representations and uses a separate world model to predict future DINOv3 embeddings, which condition action generation at inference. **(c) VLA-JEPA** (Sun et al., 2026) generates actions from VLM representations and uses latent action tokens to condition a separate world model supervised by V-JEPA2 embeddings during training. **(d) LeWAM** ses frozen I-JEPA embeddings as both visual inputs for action generation and targets for action-conditioned future prediction, learning both tasks with a single predictor while generating only actions at inference. 

to shift the policy distribution toward these trajectories. A natural approach is to maximize expected return through online reinforcement learning with rollout rewards (Ren et al., 2025; Yi et al., 2026). However, on physical robots, this requires repeated environment interaction and human oversight for resets and safety, making it difficult to scale (Eysenbach et al., 2017; Sharma et al., 2021; Walke et al., 2023; Patil et al., 2024). To avoid this requirement, we introduce Demonstration-Guided DPO (DemoDPO) as the offline preference refinement stage of LeWAM. DemoDPO uses demonstrations to construct preferences among action candidates from a frozen reference policy, providing offline preference supervision without additional environment interaction or manually designed rewards. 

Our contributions are summarized as follows: 

- We theoretically and empirically study visual representations for action generation in WAMs, identify I-JEPA as an effective visual encoder for action generation, and further examine its representations through attention visualization and probing. 

- We propose LeWAM, which combines AdaFuse with joint action generation and future embedding prediction in a lightweight model, while generating only actions at inference. 

- We refine LeWAM with DemoDPO, using demonstrations to provide preference supervision without additional environment interaction or manually designed rewards. 

## 2 RELATED WORK 

**Visual Representations and World Modeling for Action Generation.** Diffusion Policy (Chi et al., 2025) and Octo (Octo Model Team et al., 2024) generate action sequences through conditional denoising without an explicit world model. VDM-based WAMs typically couple action learning with future video modeling through pretrained video diffusion backbones, as illustrated in Figure 2 _(a)_ (Ye et al., 2026b; Bi et al., 2026; Ma et al., 2026; Yuan et al., 2026; Ye et al., 2026a). Recent methods instead predict future states in pretrained representation spaces. VLA-JEPA uses latent action tokens produced by a VLM to condition a separate world model trained against frozen V-JEPA2 targets (Sun et al., 2026). The flow matching action head is conditioned on VLM representations, whereas V- JEPA2 embeddings supervise only dynamics learning and do not serve as direct visual inputs for action generation, as shown in Figure 2 _(c)_ . LaWAM first learns a latent action model in a frozen DINOv3 space and then factorizes control into a VLA that predicts a latent action, a separate latent world model that decodes it into a future visual subgoal, and an action expert conditioned on that subgoal (Chen et al., 2026). Future prediction therefore remains an explicit intermediate at deployment, as shown in Figure 2 _(b)_ . LeWAM instead uses frozen I-JEPA embeddings as both the direct visual input for action generation and the target for future prediction. A single predictor jointly generates actions and predicts future embeddings conditioned on demonstrated action prefixes during training, while future prediction is removed at deployment, as shown in Figure 2 _(d)_ . 

3 

**Robot Policy Refinement beyond Imitation.** Online reinforcement learning has been used to refine diffusion and flow matching policies using rewards collected through environment interaction (Ren et al., 2025; Yi et al., 2026). However, applying these methods to physical robots requires repeated environment rollouts, reliable task rewards, and human oversight for resets and safety, making them difficult to scale across tasks (Eysenbach et al., 2017; Sharma et al., 2021; Walke et al., 2023; Patil et al., 2024). Preference-based methods provide an alternative, but existing approaches derive supervision from predefined task rewards or reward-based evaluators (Zhang et al., 2024a; Shan et al., 2025), model-generated evaluation criteria (Zhang et al., 2024b), or direct human feedback, including pairwise comparisons, interventions, and manually specified behavior preferences (Chen et al., 2025b; Xia et al., 2026; Vatsa et al., 2026). These signals still require additional supervision or interaction beyond the original offline demonstrations. DemoDPO instead uses each existing demonstration to rank a group of candidates sampled from a frozen reference policy and filters groups without sufficiently distinct preferences. It therefore enables offline preference refinement without additional environment interaction, reward supervision, or human preference annotation. 

## 3 REPRESENTATION-GUIDED LATENT WORLD ACTION MODELING 

In this section, we analyze visual representations for action generation and the limitations of VDMbased WAMs. We then present LeWAM and its DemoDPO preference refinement stage. 

### 3.1 PROBLEM FORMULATION 

We study visuomotor policy learning from expert demonstrations. At time _t_ , the policy maps a visual observation _Xt_ and task context _ct_ to an expert action chunk _At_ = ( _at, . . . , at_ + _H_ a _−_ 1), where _H_ a is the number of control steps per generation pass. LeWAM parameterizes the conditional policy _πθ_ ( _At | Xt, ct_ ) using flow matching Lipman et al. (2023). World Action Models augment this objective by modeling environment evolution over the same horizon. We select _N_ future offsets 0 _< δ_ 1 _< · · · < δN ≤ H_ a and define _Zi_ = _E_ ( _Xt_ + _δi_ ), where the visual representation function _E_ produces _Zi ∈_ R<sup>_hw×d_</sup> as an _h × w_ grid of _d_ -dimensional patch embeddings. LeWAM jointly generates _At_ and predicts _{Zi}_<sup>_N_</sup> _i_ =1<sup>, as detailed in Section 3.3.Below, we omit time indices and let</sup> _X_ , _A_ , _c_ , and _Z_ = _E_ ( _X_ ) denote random variables drawn from the demonstrations. Unless stated otherwise, all probabilities, expectations, and information quantities refer to their joint distribution. 

### 3.2 ACTION-SUFFICIENT WORLD MODELING 

We analyze how a visual representation affects action uncertainty. 

**Assumption 1** (Action sufficiency) **.** _The factors S_ = _S_ ( _X_ ) _form a sufficient statistic for the action given the context, that is, p_ ( _A | X, c_ ) = _p_ ( _A | S, c_ ) _._ 

**Proposition 1** (Action relevance gap) **.** _Under Assumption 1, let Z_ = _E_ ( _X_ ) _, where E is deterministic. When the conditional action entropies are finite, the additional action uncertainty satisfies_ 

∆ _E_ ( _A_ ) := _H_ ( _A | Z, c_ ) _− H_ ( _A | X, c_ ) = _I_ ( _A_ ; _X | Z, c_ ) = _I_ ( _A_ ; _S | Z, c_ ) _≥_ 0 _,_ (1) _where H denotes entropy for discrete actions and differential entropy for continuous actions, and I denotes conditional mutual information._ 

A useful representation for action generation should retain the information in _S_ relevant to _A_ . 

We next examine VAE compression and the reliance on large video diffusion backbones in VDMbased WAMs. First, a linear Gaussian analysis makes explicit how VAE training can suppress observation directions in the posterior mean (Wang & Ziyin, 2022). 

**Proposition 2** (VAE spectral compression) **.** _For fixed c, consider the linear Gaussian VAE in Appendix A.2, optimized to a global minimum with latent width k, KL weight β_ vae _>_ 0 _, and fixed decoder variance σ_ dec<sup>2</sup><sup>_>_0</sup><sup>_.Let_(</sup><sup>_λi, ui_)</sup><sup>_denote the observation covariance eigenpairs in decreas-_</sup> _ing eigenvalue order, and set λ_ cut = _β_ vae _σ_ dec<sup>2</sup><sup>_.The posterior mean Z_vae</sup><sup>_has effective dimension_</sup> _r_ eff := rank Cov( _Z_ vae _| c_ ) = min� _k,_ # _{i_ : _λi > λ_ cut _}_ � _._ (2) _Let J_ = _{i ≤ k_ : _λi > λ_ cut _}. If A_ = _BcX_ + _ξ, where ξ is independent Gaussian noise with positive definite covariance, then_ ∆ _E_ ( _A_ ) _>_ 0 _exactly when Bcui̸_ = 0 _for some i ∈/ J._ 

4 

Proposition 2 and Corollary 1 show that reconstruction based selection need not retain actionrelevant directions. Appendix A.3 further distinguishes information loss from the difficulty of extracting retained information with a fixed predictor (Xu et al., 2020). This distinction motivates our controlled comparison of frozen representations for action generation in Section 4.2. 

Second, VDM-based WAMs rely on a large video diffusion backbone pretrained at scale for useful visual features. Indeed, Wang et al. (2026) show that such a backbone can be repurposed into a strong feedforward perception model. This result demonstrates the value of video generation pretraining, but does not establish the effectiveness of raw VAE latents as direct representations for action generation. The diffusion backbone extracts the information retained in _Z_ vae and exploits learned visual priors. Section 4.2 evaluates whether a lightweight predictor on a frozen JEPA encoder matches these systems without such a backbone. 

### 3.3 WORLD ACTION MODELING IN REPRESENTATION SPACE 



<!-- Start of picture text -->
, noisy act. tokens<br>, clean act. tokens ··· ···<br>act. head state head<br>🔥<br>LeWAM<br>AdaLN + FFN<br>AdaLN + Self Attn.<br>🔥 Vis Proj. 🔥 Act Proj. w/ QK Norm and 3D RoPE<br>❄ JEPA 🔥 AdaFuse<br>Encoder future<br>LN + MLP MLP quires<br>obs +<br>··· ··· embed actions<br>(a) (b) (c)<br>··· ···<br>cond.<br>··· ···<br>···<br>···<br><!-- End of picture text -->

Figure 3: LeWAM architecture for joint action generation and future embedding prediction. _(a)_ A frozen JEPA encoder extracts multilayer embeddings, and AdaFuse adaptively combines them. _(b)_ The predictor combines observation and action tokens with future queries to predict action velocities and future embeddings. _(c)_ The mask prevents clean actions from leaking into action prediction and restricts each future query to the action prefix for its randomly sampled horizon. 

Figure 3 shows how LeWAM learns an action policy and a representation space world model within a single predictor: 



where _A_<sup>(</sup> _t_<sup>_δi_)</sup> = ( _at, . . . , at_ + _δi−_ 1) is the clean action prefix leading to _Xt_ + _δi_ and _Zi_ = _E_ ( _Xt_ + _δi_ ). The first conditional generates an action chunk from the current observation, while the second predicts how the visual embedding evolves under the demonstrated action prefix. Given a training segment starting at time _t_ , let _Eℓ_ ( _X_ ) _∈_ R<sup>_hw×d_</sup> denote the patch embeddings from layer _ℓ_ of a frozen JEPA encoder with _L_ layers. AdaFuse learns logits Γ _∈_ R<sup>_L×d_</sup> , normalizes them across layers separately for each feature dimension, and computes 



where _⊙_ denotes elementwise multiplication, and LNin and LNout apply LayerNorm before and after fusion. The channelwise fusion weights are shared across patch tokens and observations. We use AdaFuse for the current observation and obtain future targets directly from the frozen encoder: _Z_ 0 = _F_ Ada( _Xt_ ) and _Zi_ = _E_ ( _Xt_ + _δi_ ), where _E_ denotes the frozen encoder. For conditional flow matching, we sample _τ ∼_ Unif[0 _,_ 1] and form 



To jointly model action generation and future embedding prediction, LeWAM processes the token sequence [ _Z_ 0 _, A_<sup>_′_</sup> _τ_<sup>_, At, Q_1:</sup><sup>_N_] with a single predictor. Each</sup><sup>_Qi_is formed by repeating a learned query</sup> token over a patch grid matching _Z_ 0 and adding learned position embeddings. Each attention layer uses QK normalization and 3D RoPE with one temporal and two spatial coordinates. _Z_ 0 and _Qi_ use 

5 

their spatiotemporal positions, whereas noisy and clean action tokens use their temporal positions with both spatial coordinates set to _−_ 1. A structured attention mask allows each noisy action token to attend only to _Z_ 0 and its causal noisy action prefix, makes the clean action tokens causal, and allows _Qi_ to attend only to _Z_ 0, its own query grid, and the first _δi_ clean actions. This prevents clean actions from leaking into action generation and actions after _δi_ from affecting the prediction of _Zi_ , either directly or through earlier clean action states. The causal noisy action stream also supports different action horizons at inference. Under this mask, the noisy action stream realizes _πθ_ ( _At | Xt, ct_ ), while the future query stream realizes _pθ_ ( _Zi | Xt, A_<sup>(</sup> _t_<sup>_δi_)</sup> _, ct_ ). 

Separate heads map the hidden states of the noisy action tokens to the flow velocity ˆ _vθ_ and those of the future query tokens to the predicted future embeddings _Z_<sup>ˆ</sup> 1: _N_ . We optimize both objectives with 



where E _i_ averages over the _N_ future targets and sg denotes stop gradient. The joint objective is _L_ LeWAM = _L_ act + _αL_ wm, where _α_ controls the contribution of future embedding prediction. 

**Inference.** At inference, LeWAM encodes _Xt_ into _Z_ 0 and denoises an action chunk of the desired length conditioned on _Z_ 0 and _ct_ . Clean action tokens and future queries are omitted, so future embedding prediction provides training supervision without additional inference cost. 

### 3.4 DEMONSTRATION-GUIDED DPO 

We observe that LeWAM can generate action chunks that closely match successful trajectories, yet small action deviations can still cause a rollout to fail. This exposes a limitation of supervised imitation learning: flow matching learns the demonstrated action distribution but provides no explicit signal that distinguishes better generated actions from worse ones. We therefore refine LeWAM with Demonstration-Guided DPO (DemoDPO), which uses each demonstrated action chunk to rank a group of candidates sampled from a frozen reference policy. We initialize the trainable policy _πθ_ and a frozen reference policy _π_ ref from the supervised LeWAM checkpoint. For each training condition _xt_ = ( _Xt, ct_ ) and demonstrated action chunk _At_ , we draw _G_ candidates _A_<sup>�</sup> _i_ i _.∼_ i _._ d _. π_ ref ( _· | xt_ ) from independent Gaussian noise. We measure their similarity to the demonstration using action MSE, denoted by _di_ , and define the candidates with the smallest and largest MSE as _A_<sup>_w_</sup> and _A_<sup>_l_</sup> , respectively. The demonstration therefore provides the ranking signal, while both candidates are sampled from the same frozen reference policy. Let _i_<sup>_w_</sup> and _i_<sup>_l_</sup> be the indices of _A_<sup>_w_</sup> and _A_<sup>_l_</sup> , respectively, and let _ε_ pair denote the minimum MSE gap. We define the pair retention mask as 



We retain only pairs with _M_ pair = 1, filtering out groups whose best and worst candidates have nearly indistinguishable demonstration errors. 

For a preference pair ( _A_<sup>_w_</sup> _, A_<sup>_l_</sup> ), standard DPO (Rafailov et al., 2023) minimizes 



where the expectation is over preference pairs, _σ_ is the sigmoid function, _β_ is the DPO regularization coefficient, and _ρθ_ ( _A | xt_ ) is the log policy ratio of the trainable policy to the frozen reference policy. A larger _ρθ_ ( _A | xt_ ) means that the trainable policy assigns higher likelihood to _A_ relative to the reference policy. For a flow matching policy, _ρθ_ cannot be evaluated directly from the velocity predictor. Diffusion-DPO (Wallace et al., 2024) derives a tractable surrogate objective for diffusion models from denoising errors at a sampled noise level, while Flow-DPO (Liu et al., 2026) adapts this objective to flow matching policies using velocity prediction errors. We follow this construction by sampling ( _τ, ϵ_ ) independently of the candidate generation noises _{ηi}_<sup>_G_</sup> _i_ =1<sup>and applying Equation 5</sup> to _A_<sup>_w_</sup> and _A_<sup>_l_</sup> with the same ( _τ, ϵ_ ). For _ϕ ∈{θ,_ ref _}_ and _A ∈{A_<sup>_w_</sup> _, A_<sup>_l_</sup> _}_ , let _A_<sup>_′_</sup> _τ_<sup>denote the resulting</sup> noisy action. We define the mean squared flow matching error as 





6 

Substituting the surrogate _sθ_ for _ρθ_ in Equation 8 gives 

_L_ DemoDPO = _−_ E �log _σ_ � _β_ � _sθ_ ( _A_<sup>_w_</sup> _| xt_ ; _τ, ϵ_ ) _− sθ_ ( _A_<sup>_l_</sup> _| xt_ ; _τ, ϵ_ )���� _M_ pair = 1� _._ (11) The expectation is over candidate groups sampled from the frozen reference policy and the shared flow noise, conditioned on _M_ pair = 1. The loss encourages the trainable policy to assign a higher relative score to the candidate with lower demonstration MSE than to the candidate with higher demonstration MSE. Appendix B.1 derives this objective from the Diffusion-DPO denoising surrogate and the Flow-DPO velocity formulation, and Algorithm 1 summarizes one update step. 

## 4 EXPERIMENTS 

### 4.1 EXPERIMENTAL SETUP 

**Simulation and real-world settings.** We evaluate LeWAM on both simulated and real-world manipulation tasks. For simulation, we use RoboTwin 2.0, a bimanual robotic manipulation benchmark comprising 50 tasks, following the multitask training setup of recent RoboTwin evaluations (Ye et al., 2026a; Yuan et al., 2026). The training data contain 2,500 clean and 25,000 heavily randomized demonstrations, corresponding to 50 and 500 demonstrations per task, respectively. We train all policies in our controlled comparisons for 10 epochs using AdamW with a learning rate of 10<sup>_−_4</sup> and a global batch size of 1024 on 16 NVIDIA H800 GPUs. Following Fast-WAM, we use an action chunk of 32 steps in simulation. For future embedding prediction, we uniformly sample one future offset from steps 1 to 32 for each training sample and set _α_ = 0 _._ 1. For DemoDPO, we sample _G_ = 4 candidates from the frozen reference policy using 10 explicit Euler steps and retain pairs satisfying _dil − diw ≥_ 10<sup>_−_3</sup> . We train DemoDPO for 1,000 optimization steps with a learning rate of 10<sup>_−_5</sup> , _β_ = 5000, and an EMA decay of 0 _._ 9995. In the real world, we evaluate three bimanual tasks on an AgileX dual Piper robot: stacking three blocks, folding a towel, and arranging three flowers in a vase. 

**Evaluation protocol.** Simulation performance is measured by task success rate. For the real-world evaluation, each method is evaluated for 50 rollouts per task, and we report task progress measured by milestone completion. The complete real-world experiment details are provided in Appendix D. Table 3 in Appendix C.4 reports inference latency and peak memory. 

### 4.2 ROBOTWIN 2.0 RESULTS AND ABLATIONS 

We organize the RoboTwin 2.0 evaluation around three empirical questions: representation choice, latent world modeling, and preference refinement. 

- _RQ1_ : How do frozen visual representations affect action generation? 

- _RQ2_ : How should future embeddings be predicted in JEPA space? 

- _RQ3_ : What makes offline preference refinement effective for LeWAM? 

**_RQ1_ : Frozen visual representations.** Proposition 1 and the decomposition in Appendix A.3 attribute action uncertainty to the information a representation omits and the information a fixed predictor cannot extract, so we compare frozen vision encoders under the same action predictor and training recipe. We use a discrete task ID as _ct_ across all controlled comparisons to isolate visual representation effects from differences in text representations. We first compare the Wan2.2 VAE, SigLIP, MAE-Large, DINOv3-Large, and V-JEPA2-Large, as shown in Figure 1 _(a)_ and Table 1. The Wan2.2 VAE performs worst at 10.81%, while V-JEPA2-Large performs best among the three Large encoders. Notably, I-JEPA-Huge outperforms V-JEPA2-Huge despite being pretrained only on ImageNet-22K (Deng et al., 2009). 

To further understand why I-JEPA supports action generation effectively, we visualize the attention from action tokens to vision tokens during inference. Figure 4 shows that the attention concentrates on the target objects and target regions and shifts between relevant regions as the manipulation progresses. Prior work links segmentation ability to manipulation generalization (Burns et al., 2023) and shows benefits of geometric information for robot learning (Jeon et al., 2026; Wang et al., 2024), so we examine how readily each frozen encoder exposes geometry and object information. We train lightweight depth and segmentation probes on the frozen encoder outputs with teacher targets from 

7 

DA3MONO-LARGE (Lin et al., 2025) and SAM3 (Carion et al., 2026), following the protocol in Appendix C.5. I-JEPA-Huge performs best among the evaluated individual encoders on both probes while the Wan2.2 VAE performs worst, indicating that I-JEPA makes the geometry and object information emphasized during action generation readily accessible. 



Figure 4: Attention from action tokens to vision tokens during a representative hammering rollout. 

Table 1: Frozen encoder comparison on RoboTwin 2.0 with representation probing. Bold, underline, and italics mark the best, second, and third results, respectively. 

||S|uccess ra|te|Dep|th|Segme|ntation|
|---|---|---|---|---|---|---|---|
|Encoder|Clean|Rand.|Avg.|RMSE _↓_|Corr. _↑_|AP@50 _↑_|AP@75 _↑_|
|Wan2.2-VAE|12.52|9.10|10.81|0.708|0.600|0.206|0.025|
|SigLIP|81.60|76.80|79.20|0.625|0.711|0.378|0.044|
|MAE-Large|64.30|63.80|64.05|0.556|0.774|0.535|0.147|
|DINOv3-Large|69.58|65.50|67.54|0.572|0.765|0.515|0.146|
|V-JEPA2-Large|73.40|70.94|72.17|0.477|0.832|_0.571_|_0.247_|
|V-JEPA2-Huge|_83.76_|_79.64_|_81.70_|_0.446_|_0.852_|0.532|0.238|
|I-JEPA-Huge|88.42|83.92|86.17|0.445|0.853|0.613|0.323|
|+ AdaFuse|**88.76**|**85.00**|**86.88**|**0.440**|**0.854**|**0.616**|**0.345**|



Table 2: RoboTwin baselines and LeWAM variants. We report average success rates over 50 tasks in clean and randomized settings, with 100 evaluation cases per task in each setting. Embodied PT. indicates whether additional robot data are used for pretraining before benchmark training. 

|Model|Encoder|Trainable<br>Params.|Embodied<br>PT.|Clean|Rand.|Avg.|
|---|---|---|---|---|---|---|
|_VLM-based VLAs_|||||||
|_π_0|SigLIP|3B|✓|65.92|58.40|62.16|
|_π_0_._5|SigLIP|3B|✓|82.74|76.76|79.75|
|_VDM-based WAMs_|||||||
|Motus w/o Pretrain|Wan2.2-VAE|6B|✗|77.56|77.00|77.28|
|LingBot-VA w/o Pretrain|Wan2.2-VAE|5B|✗|80.60|–|80.60|
|<br>GigaWorld-Policy|Wan2.2-VAE|5B|✓|86.36|85.04|85.70|
|Motus|Wan2.2-VAE|6B|✓|88.66|87.02|87.84|
|Fast-WAM|Wan2.2-VAE|6B|✗|91.88|91.78|91.83|
|LingBot-VA|Wan2.2-VAE|5B|✓|92.90|91.50|92.20|
|_Latent-based WAMs_|||||||
|LaWAM|DINOv3-B/16|1.7B|✓|92.64|89.80|91.22|
||_LeWA_|_M Variants_|||||
|Action Only|I-JEPA-H/14|0.4B|✗|88.42|83.92|86.17|
|<br>+ WM (FM)|I-JEPA-H/14|0.4B|✗|88.18|84.24|86.21|
|<br>+ WM|I-JEPA-H/14|0.4B|✗|88.86|85.18|87.02|
|+ WM + AdaFuse|I-JEPA-H/14|0.4B|✗|91.32|90.06|90.69|
|LeWAM|I-JEPA-H/14|0.4B|✗|**93.14**|**91.42**|**92.28**|



**_RQ2_ : Future embedding prediction.** We keep LeWAM’s I-JEPA encoder frozen and compare direct prediction of future embeddings from learnable future queries, as described in Section 3.3, with a flow matching alternative. The latter adds Gaussian noise to future embeddings and predicts their transport directions conditioned on the corresponding clean action prefixes. 

Table 2 shows that direct prediction achieves 87.02%, outperforming flow matching at 86.21% and the Action Only baseline at 86.17%. Prior work shows that JEPA representations encode rich se- 

8 

mantic and spatial structure (Assran et al., 2023; Mur-Labadia et al., 2026). In our setting, the frozen encoder maps each observed future frame to a fixed, high-dimensional target embedding, whereas flow matching introduces an additional noise-conditioned transport problem over this space. We hypothesize that direct regression makes more direct use of this structured supervision, explaining its stronger performance. We then evaluate AdaFuse, which adaptively integrates representations from multiple encoder layers. Without future embedding prediction, AdaFuse improves success from 86.17% to 86.88%, as shown in Table 1. Adding future embedding prediction further raises success to 90.69% in Table 2, with a larger improvement than without AdaFuse. These results suggest that AdaFuse and future embedding prediction are complementary. Table 1 further shows that AdaFuse improves depth and segmentation probing over the final I-JEPA layer, suggesting that the learned fusion makes complementary information across encoder layers more accessible. 

**_RQ3_ : DemoDPO refinement.** DemoDPO improves the average success rate from 90.69% to 92.28% across all 50 RoboTwin tasks without additional environment interaction, as shown in Table 2. To examine what makes this refinement effective, Figure 5 compares pair filtering thresholds, DPO coefficients, and candidate group sizes on a fixed subset of eight challenging tasks. The evaluation protocol is provided in Appendix C.3. Pair filtering retains candidates with a sufficient difference in demonstration similarity. A moderate threshold gives the strongest improvement, whereas using all pairs or imposing an overly strict threshold reduces performance. This suggests that effective refinement requires both distinguishable preferences and sufficient training pairs. The DPO coefficient also affects refinement: performance improves as _β_ increases to 5000, then declines at 6000. Increasing the candidate group size from 2 to 4 improves success, while increasing it further to 8 provides little additional benefit. We therefore use _ε_ pair = 10<sup>_−_3</sup> , _β_ = 5000, and _G_ = 4. 

To distinguish the benefit of DemoDPO from that of additional training, we apply 1,000 additional supervised updates to the checkpoint used to initialize DemoDPO. Success decreases slightly from 73.31% to 73.25%, whereas DemoDPO reaches 76.63% after the same number of updates, indicating that its gains are not explained by additional training alone. LeWAM therefore matches the strongest baselines in Table 2 with 0.4B trainable parameters and no video diffusion backbone, at the lowest latency and peak memory among the evaluated models in Table 3. 



<!-- Start of picture text -->
SR (%) SR (%) SR (%) 76.63 76.75<br>76 73.56 74.50 77 76.56 76.63 76<br>73.31<br>72 71.69 75 74.50 75.00 74.88 73 73.31<br>73.75<br>67.88 73.31 70.44<br>68<br>73 70<br>ε pair 0 10 −4 10 −3 10 −2 β 1000 2000 3000 4000 5000 6000 G 2 4 8<br>(a) (b) (c)<br><!-- End of picture text -->

Figure 5: DemoDPO analyses and ablations for RQ3. _(a)_ Pair filtering threshold. _(b)_ DPO coefficient. _(c)_ Candidate group size. The dashed line marks the 73.31% success rate before DemoDPO. 

**Real-world experiments.** Figure 1 _(b)_ summarizes the real-world results, with further results and implementation details provided in Appendix D. 

## 5 CONCLUSION 

Our theoretical analysis motivates examining both the information retained by visual representations and its accessibility for action generation. Attention analysis shows that the policy focuses on target objects and regions during manipulation, while controlled encoder comparisons and representation probing suggest that I-JEPA makes the relevant geometric and object information more accessible than raw VAE representations. These findings motivate LeWAM, which couples action generation with future embedding prediction in I-JEPA space rather than relying on a pretrained video diffusion backbone. AdaFuse strengthens this representation by integrating complementary information across encoder layers, while DemoDPO further refines the policy using preferences derived from demonstrations. Together, these components enable LeWAM to achieve an average success rate of 92.28% on RoboTwin 2.0 and strong performance on real-world manipulation tasks, with only 0.4B trainable parameters and no embodied pretraining. 

9 

### AI USE STATEMENT 

In this work, we used generative AI tools to assist with codebase development and manuscript polishing. Generative AI was not used to produce conclusions unsupported by objective experimental evidence. All AI-assisted content, including but not limited to code and manuscript text, was manually reviewed and verified by the authors. We take full responsibility for the final content of this work. 

### REPRODUCIBILITY STATEMENT 

To support reproducibility, we provide the training and inference code in the supplementary material. Our simulation experiments use the official RoboTwin 2.0 training dataset. The real-world experiments use an in-house dataset collected through teleoperation, which we will publicly release. Detailed model configurations, training procedures, and evaluation protocols are provided in the paper and appendix. 

## REFERENCES 

- Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 15619–15629, 2023. 

- Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, et al. V-JEPA 2: Self-supervised video models enable understanding, prediction and planning. _arXiv preprint arXiv:2506.09985_ , 2025. 

- Suneel Belkhale, Tianli Ding, Ted Xiao, Pierre Sermanet, Quon Vuong, Jonathan Tompson, Yevgen Chebotar, Debidatta Dwibedi, and Dorsa Sadigh. RT-H: Action hierarchies using language. _arXiv preprint arXiv:2403.01823_ , 2024. 

- Hongzhe Bi, Hengkai Tan, Shenghao Xie, Zeyuan Wang, Shuhe Huang, Haitian Liu, Ruowen Zhao, Yao Feng, Chendong Xiang, Yinze Rong, et al. Motus: A unified latent action world model. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 35101–35113, 2026. 

- Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. _π_ 0: A vision-language-action flow model for general robot control. _arXiv preprint arXiv:2410.24164_ , 2024. 

- Kevin Black, Manuel Galliker, and Sergey Levine. Real-time execution of action chunking flow policies. _Advances in Neural Information Processing Systems_ , 38:33383–33407, 2026. 

- Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. RT-1: Robotics transformer for real-world control at scale. _arXiv preprint arXiv:2212.06817_ , 2022. 

- Kaylee Burns, Zach Witzel, Jubayer Ibn Hamid, Tianhe Yu, Chelsea Finn, and Karol Hausman. What makes pre-trained visual representations successful for robust manipulation? _arXiv preprint arXiv:2312.12444_ , 2023. 

- Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris Coll-Vinent, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, et al. SAM 3: Segment anything with concepts. In _International Conference on Learning Representations_ , volume 2026, pp. 138846–138923, 2026. 

- Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 9650–9660, 2021. 

10 

- Jialei Chen, Kai Wang, Kang Chen, Shuaihang Chen, Feng Gao, Wenhao Tang, Zhiyuan Li, Weilin Liu, Zhuyu Yao, Boxun Li, et al. LaWAM: Latent world action models for efficient dynamicsaware robot policies. _arXiv preprint arXiv:2606.15768_ , 2026. 

- Peng Chen, Pi Bu, Yingyao Wang, Xinyi Wang, Ziming Wang, Jie Guo, Yingxiu Zhao, Qi Zhu, Jun Song, Siran Yang, et al. CombatVLA: An efficient vision-language-action model for combat tasks in 3D action role-playing games. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 10919–10928, 2025a. 

- Xiaokang Chen, Mingyu Ding, Xiaodi Wang, Ying Xin, Shentong Mo, Yunhao Wang, Shumin Han, Ping Luo, Gang Zeng, and Jingdong Wang. Context autoencoder for self-supervised representation learning. _International Journal of Computer Vision_ , 132(1):208–223, 2024. 

- Yuxin Chen, Devesh K Jha, Masayoshi Tomizuka, and Diego Romeres. FDPP: Fine-tune diffusion policy with human preference. In _2025 IEEE International Conference on Robotics and Automation (ICRA)_ , pp. 12010–12016. IEEE, 2025b. 

- Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion Policy: Visuomotor policy learning via action diffusion. _The International Journal of Robotics Research_ , 44(10-11):1684–1704, 2025. 

- Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A large-scale hierarchical image database. In _2009 IEEE Conference on Computer Vision and Pattern Recognition_ , pp. 248–255. IEEE, 2009. 

- ThankGod Egbe, Peng Wang, Zhihao Guo, and Zidong Chen. DINOv3-diffusion policy: Self-supervised large visual model for visuomotor diffusion policy learning. _arXiv preprint arXiv:2509.17684_ , 2025. 

- Benjamin Eysenbach, Shixiang Gu, Julian Ibarz, and Sergey Levine. Leave no trace: Learning to reset for safe and autonomous reinforcement learning. _arXiv preprint arXiv:1711.06782_ , 2017. 

- Yiguo Fan, Pengxiang Ding, Shuanghao Bai, Xinyang Tong, Yuyang Zhu, Hongchao Lu, Fengqi Dai, Wei Zhao, Yang Liu, Siteng Huang, et al. Long-VLA: Unleashing long-horizon capability of vision language action model for robot manipulation. _arXiv preprint arXiv:2508.19958_ , 2025. 

- GigaBrain Team, Boyuan Wang, Bohan Li, Chaojun Ni, Guan Huang, Guosheng Zhao, Hao Li, Jie Li, Jindi Lv, Jingyu Liu, et al. GigaBrain-0.5M*: a VLA that learns from world model-based reinforcement learning. _arXiv preprint arXiv:2602.12099_ , 2026. 

- Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll´ar, and Ross Girshick. Masked autoencoders are scalable vision learners. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 16000–16009, 2022. 

- Yucheng Hu, Yanjiang Guo, Pengchao Wang, Xiaoyu Chen, Yen-Jen Wang, Jianke Zhang, Koushil Sreenath, Chaochao Lu, and Jianyu Chen. Video Prediction Policy: A generalist robot policy with predictive visual representations. _arXiv preprint arXiv:2412.14803_ , 2024. 

- Byungwoo Jeon, Dongyoung Kim, Huiwon Jang, Insoo Kim, and Jinwoo Shin. SpatialBoost: Enhancing visual representation through language-guided reasoning. _arXiv preprint arXiv:2603.22057_ , 2026. 

- Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. OpenVLA: An open-source vision-language-action model. _arXiv preprint arXiv:2406.09246_ , 2024. 

- Haotong Lin, Sili Chen, Junhao Liew, Donny Y Chen, Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth Anything 3: Recovering the visual space from any views. _arXiv preprint arXiv:2511.10647_ , 2025. 

- Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In _The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023_ . OpenReview.net, 2023. URL https://openreview.net/forum?id=PqvMRDCJT9t. 

11 

- Jie Liu, Gongye Liu, Jiajun Liang, Ziyang Yuan, Xiaokun Liu, Mingwu Zheng, Xiele Wu, Qiulin Wang, Menghan Xia, Xintao Wang, et al. Improving video generation with human feedback. _Advances in Neural Information Processing Systems_ , 38:82155–82192, 2026. 

- Teli Ma, Jia Zheng, Zifan Wang, Chunli Jiang, Andy Cui, Junwei Liang, and Shuo Yang. DiT4DiT: Jointly modeling video dynamics and actions for generalizable robot control. _arXiv preprint arXiv:2603.10448_ , 2026. 

- Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, and Randall Balestriero. LeWorldModel: Stable end-to-end joint-embedding predictive architecture from pixels. _arXiv preprint arXiv:2603.19312_ , 2026. 

- Shangchen Miao, Ningya Feng, Jialong Wu, Ye Lin, Xu He, Dong Li, and Mingsheng Long. JEPAVLA: Video predictive embedding is needed for VLA models. _arXiv preprint arXiv:2602.11832_ , 2026. 

- Lorenzo Mur-Labadia, Matthew Muckley, Amir Bar, Mido Assran, Koustuv Sinha, Mike Rabbat, Yann LeCun, Nicolas Ballas, and Adrien Bardes. V-JEPA 2.1: Unlocking dense features in video self-supervised learning. _arXiv preprint arXiv:2603.14482_ , 2026. 

- Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, et al. Octo: An open-source generalist robot policy. _arXiv preprint arXiv:2405.12213_ , 2024. 

- Darshan Patil, Janarthanan Rajendran, Glen Berseth, and Sarath Chandar. Intelligent switching for reset-free RL. _arXiv preprint arXiv:2405.01684_ , 2024. 

- William Peebles and Saining Xie. Scalable diffusion models with transformers. In _IEEE/CVF International Conference on Computer Vision, ICCV 2023, Paris, France, October 1-6, 2023_ , pp. 4172–4182. IEEE, 2023. doi: 10.1109/ICCV51070.2023.00387. URL https://doi.org/ 10.1109/ICCV51070.2023.00387. 

- Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al. _π_ 0 _._ 5: A vision-language-action model with open-world generalization. _arXiv preprint arXiv:2504.16054_ , 2025. 

- Guo-Jun Qi, Liheng Zhang, Chang Wen Chen, and Qi Tian. AVT: Unsupervised learning of transformation equivariant representations by autoencoding variational transformations. In _2019 IEEE/CVF International Conference on Computer Vision (ICCV)_ , pp. 8129–8138. IEEE, 2019. 

- Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. _Advances in Neural Information Processing Systems_ , 36:53728–53741, 2023. 

- Allen Ren, Justin Lidard, Lars Ankile, Anthony Simeonov, Pulkit Agrawal, Anirudha Majumdar, Benjamin Burchfiel, Hongkai Dai, and Max Simchowitz. Diffusion policy policy optimization. In _International Conference on Learning Representations_ , volume 2025, pp. 77288–77329, 2025. 

- Zhao Shan, Chenyou Fan, Shuang Qiu, Jiyuan Shi, and Chenjia Bai. Forward KL regularized preference optimization for aligning diffusion policies. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 39, pp. 14386–14395, 2025. 

- Archit Sharma, Kelvin Xu, Nikhil Sardana, Abhishek Gupta, Karol Hausman, Sergey Levine, and Chelsea Finn. Autonomous reinforcement learning: Formalism and benchmarking. _arXiv preprint arXiv:2112.09605_ , 2021. 

- Jingwen Sun, Wenyao Zhang, Zekun Qi, Shaojie Ren, Zezhi Liu, Hanxin Zhu, Guangzhong Sun, Xin Jin, and Zhibo Chen. VLA-JEPA: Enhancing vision-language-action model with latent world model. _arXiv preprint arXiv:2602.10098_ , 2026. 

- Amitesh Vatsa, Zhixian Xie, and Wanxin Jin. RoDiF: Robust direct fine-tuning of diffusion policies with corrupted human feedback. _arXiv preprint arXiv:2602.00886_ , 2026. 

12 

- Homer Rich Walke, Jonathan Heewon Yang, Albert Yu, Aviral Kumar, Jedrzej Orbik, Avi Singh, and Sergey Levine. Don’t start from scratch: Leveraging prior data to automate robotic reinforcement learning. In _Conference on Robot Learning_ , pp. 1652–1662. PMLR, 2023. 

- Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 8228–8238, 2024. 

- Letian Wang, Chuhan Zhang, Rishabh Kabra, Jasper Uijlings, Steven Waslander, Andrew Zisserman, Joao Carreira, Kaiming He, Misha Andriluka, Eduard Gabriel Bazavan, et al. Video generation models are general-purpose vision learners. _arXiv preprint arXiv:2607.09024_ , 2026. 

- Wanying Wang, Jinming Li, Yichen Zhu, Zhiyuan Xu, Zhengping Che, Yaxin Peng, Chaomin Shen, Dong Liu, Feifei Feng, and Jian Tang. Visual robotic manipulation with depth-aware pretraining. _arXiv preprint arXiv:2401.09038_ , 2024. 

- Zihao Wang and Liu Ziyin. Posterior collapse of a linear latent variable model. _Advances in Neural Information Processing Systems_ , 35:37537–37548, 2022. 

- Ziru Wang, Mengmeng Wang, Jade Dai, Teli Ma, Guo-Jun Qi, Yong Liu, Guang Dai, and Jingdong Wang. DynaMind: Reasoning over abstract video dynamics for embodied decision-making. In _Forty-second International Conference on Machine Learning_ , 2025. 

- Yuqing Wen, Hebei Li, Kefan Gu, Yucheng Zhao, Tiancai Wang, and Xiaoyan Sun. LLaDA-VLA: Vision language diffusion action models. _arXiv preprint arXiv:2509.06932_ , 2025. 

- Hongtao Wu, Ya Jing, Chilam Cheang, Guangzeng Chen, Jiafeng Xu, Xinghang Li, Minghuan Liu, Hang Li, and Tao Kong. Unleashing large-scale video generative pre-training for visual robot manipulation. In _International Conference on Learning Representations_ , volume 2024, pp. 10641–10662, 2024. 

- Wenke Xia, Yichu Yang, Hongtao Wu, Xiao Ma, Tao Kong, and Di Hu. Human-assisted robotic policy refinement via action preference optimization. _Advances in Neural Information Processing Systems_ , 38:36746–36768, 2026. 

- Yilun Xu, Shengjia Zhao, Jiaming Song, Russell Stewart, and Stefano Ermon. A theory of usable information under computational constraints. In _8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020_ . OpenReview.net, 2020. URL https://openreview.net/forum?id=r1eBeyHFDH. 

- Angen Ye, Boyuan Wang, Chaojun Ni, Guan Huang, Guosheng Zhao, Hao Li, Hengtao Li, Jie Li, Jindi Lv, Jingyu Liu, et al. GigaWorld-Policy: An efficient action-centered world–action model. _arXiv preprint arXiv:2603.17240_ , 2026a. 

- Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, Shenyuan Gao, Sihyun Yu, George Kurian, Suneel Indupuru, You Liang Tan, Chuning Zhu, Jiannan Xiang, et al. World action models are zero-shot policies. _arXiv preprint arXiv:2602.15922_ , 2026b. 

- Brent Yi, Hongsuk Choi, Himanshu Gaurav Singh, Xiaoyu Huang, Takara E Truong, Carmelo Sferrazza, Yi Ma, Rocky Duan, Pieter Abbeel, Guanya Shi, et al. Flow policy gradients for robot control. _arXiv preprint arXiv:2602.02481_ , 2026. 

- Tianyuan Yuan, Zibin Dong, Yicheng Liu, and Hang Zhao. Fast-WAM: Do world action models need test-time future imagination? _arXiv preprint arXiv:2603.16666_ , 2026. 

- Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pp. 11975–11986, 2023. 

- Liheng Zhang, Guo-Jun Qi, Liqiang Wang, and Jiebo Luo. AET vs. AED: Unsupervised representation learning by auto-encoding transformations rather than data. In _2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pp. 2542–2550. IEEE, 2019. 

13 

- Tianle Zhang, Jiayi Guan, Lin Zhao, Yihang Li, Dongjiang Li, Zecui Zeng, Lei Sun, Yue Chen, Xuelong Wei, Lusong Li, et al. Preferred-action-optimized diffusion policies for offline reinforcement learning. _arXiv preprint arXiv:2405.18729_ , 2024a. 

- Xinyu Zhang, Jiahui Chen, Junkun Yuan, Qiang Chen, Jian Wang, Xiaodi Wang, Shumin Han, Xiaokang Chen, Jimin Pi, Kun Yao, et al. CAE v2: Context autoencoder with CLIP latent alignment. _Transactions on Machine Learning Research_ , 2023. 

- Zijian Zhang, Kaiyuan Zheng, Zhaorun Chen, Joel Jang, Yi Li, Siwei Han, Chaoqi Wang, Mingyu Ding, Dieter Fox, and Huaxiu Yao. GRAPE: Generalizing robot policy via preference alignment. _arXiv preprint arXiv:2411.19309_ , 2024b. 

- Jinliang Zheng, Jianxiong Li, Zhihao Wang, Dongxiu Liu, Xirui Kang, Yuchun Feng, Yinan Zheng, Jiayin Zou, Yilun Chen, Jia Zeng, et al. X-VLA: Soft-prompted transformer as scalable crossembodiment vision-language-action model. _arXiv preprint arXiv:2510.10274_ , 2025. 

- Han Zhou, Jinjin Cao, Liyuan Ma, Xueji Fang, and Guo-Jun Qi. From human hands to robot arms: Manipulation skills transfer via trajectory alignment. _CoRR_ , abs/2510.00491, 2025. doi: 10. 48550/ARXIV.2510.00491. URL https://doi.org/10.48550/arXiv.2510.00491. 

- Jie Zhu, Jiyang Qi, Mingyu Ding, Xiaokang Chen, Ping Luo, Xinggang Wang, Wenyu Liu, Leye Wang, and Jingdong Wang. Understanding self-supervised pretraining with part-aware representation learning. _arXiv preprint arXiv:2301.11915_ , 2023. 

14 

## APPENDIX 

## A THEORETICAL ANALYSIS 

This section proves Propositions 1 and 2 and analyzes the effective dimension and action information of VAE posterior means. 

### A.1 ACTION-RELEVANT INFORMATION BOUNDS 

We use the same notation as the main text: _X_ is the current visual observation, _S_ denotes the actionrelevant explanatory factors, _A_ is the expert action chunk, _c_ is task context, and _Z_ = _E_ ( _X_ ) is the visual representation consumed by the action predictor. 

_Proof of Proposition 1._ Let _Z_ = _E_ ( _X_ ) and suppose _H_ ( _A | Z, c_ ) and _H_ ( _A | X, c_ ) are finite. Because _Z_ is determined by _X_ , conditioning on ( _X, Z, c_ ) is equivalent to conditioning on ( _X, c_ ). 

_I_ ( _A_ ; _X | Z, c_ ) = _H_ ( _A | Z, c_ ) _− H_ ( _A | X, Z, c_ ) = _H_ ( _A | Z, c_ ) _− H_ ( _A | X, c_ ) = ∆ _E_ ( _A_ ) _._ (12) Since _S_ = _S_ ( _X_ ), the mutual information chain rule gives 

_I_ ( _A_ ; _X | Z, c_ ) = _I_ ( _A_ ; _S, X | Z, c_ ) = _I_ ( _A_ ; _S | Z, c_ ) + _I_ ( _A_ ; _X | S, Z, c_ ) _._ (13) Assumption 1 states that _A_ is conditionally independent of _X_ given ( _S, c_ ). Since _Z_ is a function of _X_ , this also gives _I_ ( _A_ ; _X | S, Z, c_ ) = 0. Combining the two identities and using nonnegativity of conditional mutual information yields 



This identity motivates evaluating a representation by the action-relevant factor information it preserves rather than by its image reconstruction quality. 

### A.2 SPECTRAL COMPRESSION OF VAE POSTERIOR MEANS 

We analyze how VAE training selects the effective dimension of the posterior mean, then determine when the removed directions contain action information. The spectral analysis specializes the linear VAE result of Wang & Ziyin (2022). The action uncertainty result below follows by conditioning on the retained directions. This analysis concerns a linear Gaussian model at a global optimum, with a fixed decoder variance. The linear Gaussian model is an analyzable instance, not a model of the pretrained Wan2.2 VAE, whose encoder is nonlinear and whose pretraining distribution differs from the demonstration distribution. We use it to establish what reconstruction based selection does and does not guarantee about action information, and we treat the pretrained encoder empirically in Section 4.2. 

Fix a task context _c_ and let the centered observation vector be _X ∈_ R<sup>_dx_</sup> with 



The _ui_ are orthonormal principal directions, and _λi_ measures observation variance along _ui_ . Consider a VAE with _k ≤ dx_ latent coordinates, standard normal prior, and 

_q_ (˜ _z | x_ ) = _N_ ( _Mx,_ diag( _s_ 1<sup>2</sup><sup>_, . . . , s_2</sup> _k_<sup>))</sup><sup>_,_</sup> _p_ ( _x | z_ ˜) = _N_ ( _W_ ˜ _z, σ_ dec<sup>2</sup><sup>**I**)</sup><sup>_,_</sup> (16) where _M ∈_ R<sup>_k×dx_</sup> , _W ∈_ R<sup>_dx×k_</sup> , and _s_<sup>2</sup> _i_<sup>_>_0arelearned,while</sup><sup>_σ_</sup> dec<sup>2</sup><sup>_>_0isfixed.Theencoder</sup> variances are independent of _x_ . Let _Z_<sup>�</sup> denote the sampled latent variable and _z_ ˜ its realization. The training objective is, up to a constant, 



Here _β_ vae _>_ 0 is the VAE regularization weight. Although training samples _Z_<sup>�</sup> , the representation supplied to the action predictor is the deterministic posterior mean _Z_ vae = E[ _Z_<sup>�</sup> _| X_ ] = _MX_ . Let _λ_ cut = _β_ vae _σ_ dec<sup>2.Ata globaloptimum expressed in principal coordinates,the posterior mean has</sup> coordinates 



15 

where ( _a_ )+ = max( _a,_ 0). Its effective dimension, defined as the rank of its covariance, is 

_r_ eff := rank Cov( _Z_ vae _| c_ ) = min� _k,_ # _{i_ : _λi > λ_ cut _}_ � _._ (19) Let _J_ be the index set of retained directions, with _J_ = _{i ≤ k_ : _λi > λ_ cut _}_ , and define Σomit = � _i/∈J_<sup>_λiuiu_</sup> _i_<sup>_⊤_.If the expert action chunk, viewed as a vector, satisfies</sup> 



then the additional action uncertainty is 



It is strictly positive exactly when _Bcui̸_ = 0 for at least one omitted direction. 

_Proof of Proposition 2._ **1. Reduce training to principal directions.** The spectral solution for the linear VAE aligns the decoder with principal directions of Σ _X_ (Wang & Ziyin, 2022, Theorem 2). For completeness, we derive the resulting threshold and its effect on the posterior mean. Write _xi_ = _u_<sup>_⊤_</sup> _i_<sup>_X_, and denote the scalar decoder loading, encoder mean coefficient, and posterior variance</sup> along this direction by _wi_ , _mi_ , and _s_<sup>2</sup> _i_<sup>.The contribution of this direction to Equation 17 is</sup> 



The first term rewards reconstruction, while the second penalizes the encoder mean and deviation of the posterior variance from the prior variance. 

**2. Identify which directions survive regularization.** Fix _wi_ . Minimizing over _mi_ and _s_<sup>2</sup> _i_<sup>gives</sup> 



Thus ( _wi_<sup>2)</sup><sup>_∗_=(</sup><sup>_λi−λ_cut)+.ChoosingnonnegativedecoderloadingsgivesEquation18.Sign</sup> changes or permutations of latent coordinates do not affect the result. When _λi ≤ λ_ cut, both _wi_<sup>_∗_and</sup> _m_<sup>_∗_</sup> _i_<sup>vanish,and(</sup><sup>_s_2</sup> _i_<sup>)</sup><sup>_∗_=1:thiscoordinatematchestheprioranditsmeancarriesnoobservation</sup> information. When _λi > λ_ cut, its mean coefficient is nonzero. The reduction in the optimized loss relative to omitting a direction increases with _λi_ above the threshold, so the _k_ available coordinates retain the largest eligible eigenvalues. Their number is given by Equation 19. 

**3. Determine the information missing from the posterior mean.** For every _i ∈ J_ , the nonzero coefficient in Equation 18 makes _u_<sup>_⊤_</sup> _i_<sup>_X_recoverable from</sup><sup>_Z_vae. For</sup><sup>_i∈/J_, the corresponding coordinate</sup> is absent. The Gaussian principal coordinates are independent, so 



This identifies the lost directions for the deterministic mean, rather than a noisy latent sample. 

**4. Translate missing observation directions into action uncertainty.** Applying Equation 20 gives 

Cov( _A | X, c_ ) = Σ _ξ,_ Cov( _A | Z_ vae _, c_ ) = Σ _ξ_ + _Bc_ Σomit _Bc_<sup>_⊤._</sup> (26) 

Both conditional action distributions are Gaussian with positive definite covariance. Their differential entropies are finite, and subtracting them yields 



which equals the expression in Equation 21. Here _H_ denotes differential entropy. No nonnegativity assumption on differential entropy is used. Finally, _Bc_ Σomit _Bc_<sup>_⊤_</sup> =<sup>�</sup> _i/∈J_<sup>_λi_(</sup><sup>_Bcui_)(</sup><sup>_Bcui_)</sup><sup>_⊤_is</sup> positive semidefinite and is nonzero exactly when an omitted direction affects the action. The log determinant gap is therefore strictly positive precisely under the stated condition. 

**Corollary 1** (Reconstruction ordering does not order action relevance) **.** _In the setting of Proposition 2 with k < dx, let Bc_ = _b u_<sup>_⊤_</sup> _dx_<sup>_,so the action depends only on the least energetic observation_</sup> _direction. Then_ ∆ _E_ ( _A_ ) = _I_ ( _A_ ; _X | c_ ) _, that is, Z_ vae _retains no action information, whereas the single direction u_<sup>_⊤_</sup> _dx_<sup>_Xattains_∆</sup><sup>_E_(</sup><sup>_A_)=0</sup><sup>_whilehavingthelargestreconstructionerroramong_</sup> _principal directions. There exist sequences of observation and action noise covariances for which the minimum reconstruction error from Z_ vae _tends to zero while_ ∆ _E_ ( _A_ ) _diverges._ 

16 

_Proof of Corollary 1._ Since _k < dx_ , the index _dx_ is omitted, so _λdxudxu_<sup>_⊤_</sup> _dx ⪯_ Σomit and _Bc_ Σomit _Bc_<sup>_⊤_</sup> = _λdxbb_<sup>_⊤_</sup> = _Bc_ Σ _X Bc_<sup>_⊤_,using</sup><sup>_Bc_=</sup><sup>_b u⊤_</sup> _dx_<sup>and</sup><sup>_u_</sup> _d_<sup>_⊤_</sup> _x_<sup>Σ</sup><sup>_Xud_</sup> _x_<sup>=</sup><sup>_λd_</sup> _x_<sup>.Substituting</sup> into Equation 21 and applying det( **I** + _vv_<sup>_⊤_</sup> ) = 1 + _∥v∥_ 2<sup>2gives</sup> 



so the posterior mean retains none of the action information in _X_ . For the representation _u_<sup>_⊤_</sup> _dx_<sup>_X_, the</sup> action is conditionally independent of _X_ given ( _u_<sup>_⊤_</sup> _dx_<sup>_X, c_), so ∆</sup><sup>_E_(</sup><sup>_A_)=0, while its reconstruction</sup> error tr Σ _X − λdx_ is the largest among projections onto a single principal direction. Fix _λ_ 1 _> · · · > λk > λ_ cut and take 0 _< ϵ < λ_ cut. Choose distinct _λi ∈_ ( _ϵ/_ 2 _, ϵ_ ] for all _i > k_ , together with Σ _ξ_ = _ϵ_<sup>2</sup> **I** and _∥b∥_ 2 = 1. Then _J_ = _{_ 1 _, . . . , k}_ . Then tr Σomit _≤_ ( _dx − k_ ) _ϵ →_ 0 while _λdxb_<sup>_⊤_</sup> Σ<sup>_−_</sup> _ξ_<sup>1</sup><sup>_b ≥_1</sup><sup>_/_(2</sup><sup>_ϵ_)</sup><sup>_→∞_.</sup> 

The result separates two sources of dimensional reduction: the architectural width _k_ and the threshold _β_ vae _σ_ dec<sup>2inducedbytraining.Withinthismodel,directionsareretainedaccordingtotheir</sup> contribution to visual reconstruction, while their effect on action uncertainty depends additionally on _Bc_ . The effective dimension can be smaller than the latent width, and losing a direction increases action uncertainty only when that direction matters for the task. Corollary 1 and Proposition 3 bound the two ends of this behavior: reconstruction based selection can retain no action information when the action depends on a low variance direction, and it can lose only a controlled amount when reconstruction is accurate and the action sensitivity is bounded. 

### A.3 USABLE ACTION UNCERTAINTY 

Proposition 1 measures information content and is indifferent to whether a downstream model can extract it. We record here the decomposition used in Section 3.2 and the bound that separates its two terms. Let _V_ be a family of conditional densities for _A_ given ( _Z, c_ ) and let _HV_ ( _A | Z, c_ ) = inf _f ∈V_ E[ _−_ log _f_ ( _A | Z, c_ )] be the usable conditional entropy of Xu et al. (2020). For any _f_ , E[ _−_ log _f_ ( _A | Z, c_ )] _− H_ ( _A | Z, c_ ) = E _Z,cD_ KL� _p_ ( _· | Z, c_ ) _∥ f_ ( _· | Z, c_ )� _≥_ 0, so _HV_ ( _A | Z, c_ ) _≥ H_ ( _A | Z, c_ ), with equality when _V_ contains the true conditional. Adding and subtracting _H_ ( _A | Z, c_ ) yields 



whose first term is nonnegative by Proposition 1 and whose second term is nonnegative by the inequality above. The first term depends only on the representation, while the second depends jointly on the representation and the downstream model. Here _A_ denotes the vectorized action chunk of dimension _H_ a _Da_ . If _V_ consists of predictors _N_ ( _g_ ( _Z, c_ ) _, σ_<sup>2</sup> **I** ) with _g_ ranging over the realizable regressors and _σ_<sup>2</sup> _>_ 0 free, and 0 _< M_ ( _Z_ ) _< ∞_ , then taking the infimum over _g_ and optimizing _σ_<sup>2</sup> gives _σ_<sup>2</sup> = _M_ ( _Z_ ) _/_ ( _H_ a _Da_ ) and 



For this Gaussian family, usable conditional entropy is monotone in the minimum achievable action MSE. 

**Proposition 3** (Loss channel bound) **.** _In the setting of Proposition 2 with A_ = _BcX_ + _ξ as in Equation 20,_ 



_Proof._ The identity for MSErec follows from Equation 25. Since Σomit _⪰_ 0, we have Σomit _⪯ λ_ max(Σomit) **I** _⪯_ (tr Σomit) **I** , hence _Bc_ Σomit _Bc_<sup>_⊤⪯_MSErec</sup><sup>_BcB_</sup> _c_<sup>_⊤⪯_MSErec</sup><sup>_∥Bc∥_</sup> 2<sup>2</sup><sup>**I**.Conjugat-</sup> ing by Σ<sup>_−_</sup> _ξ_<sup>1</sup><sup>_/_2</sup> and using Σ<sup>_−_</sup> _ξ_<sup>1</sup> _⪯ λ_ min(Σ _ξ_ )<sup>_−_1</sup> **I** gives 



17 

The log determinant is monotone in the positive semidefinite order, so substituting this bound into Equation 21 and evaluating the determinant of the resulting multiple of the identity gives the claim. 

Because MSErec is the minimum mean squared error of predicting _X_ from _Z_ vae, it is bounded above by the error of any particular decoder, including one acting on the sampled latent _Z_<sup>�</sup> , since _X → Z_ vae _→ Z_<sup>�</sup> forms a Markov chain. Measured reconstruction fidelity therefore upper bounds the information loss term within this model. Corollary 1 operates in the regime where the action sensitivity _∥Bc∥_<sup>2</sup> 2<sup>_/λ_min(Σ</sup><sup>_ξ_)divergesandtheboundbecomesvacuous,sothetworesultsare</sup> compatible. The bound relates reconstruction error to action information loss through the action sensitivity and noise level in this linear Gaussian model. 

### A.4 LIMITS OF DOWNSTREAM RECOVERY 

Let a downstream model construct a hidden state _R_ deterministically from ( _Z, c_ ). The data processing inequality gives 



Thus, a trainable transformer can reorganize retained information into a predictor state, but it cannot recreate action-relevant information discarded jointly by the representation and context. 

## B DEMODPO DERIVATION AND ALGORITHM 

We derive the DemoDPO objective by following Diffusion-DPO from action likelihoods to a surrogate based on denoising errors, then converting these errors to flow matching errors. The derivation separates the path approximation, the bound used for sampling a single step, and the weighting choice used in training. 

### B.1 OBJECTIVE DERIVATION 

**1. Preference objective.** Let _D_ = _{_ ( _xt, A_<sup>_w_</sup> _, A_<sup>_l_</sup> ) _}_ contain preference pairs under the training condition _xt_ = ( _Xt, ct_ ). Using the notation of Equation 8, standard DPO optimizes 



This objective favors the preferred action relative to the reference policy. Its direct evaluation requires action likelihoods, which are not outputs of the velocity predictor. We therefore follow the diffusion surrogate construction of Wallace et al. (2024), before adapting it to flow matching. 

**2. From actions to generation paths.** In the diffusion construction, let _A_ 0 = _A_ and let _A_ 1: _K_ denote intermediate noisy actions. Here _k_ indexes diffusion steps, not control steps. For _ϕ ∈ {θ,_ ref _}_ , the path distribution and its endpoint marginal are 





The marginal integrates over all paths ending at _A_ , whereas the path distribution factorizes into individual transitions. Diffusion-DPO constructs a preference objective on paths, using the joint KL to upper bound the endpoint KL: 

_D_ KL ( _pθ_ ( _A_ 0 _| xt_ ) _∥p_ ref ( _A_ 0 _| xt_ )) _≤ D_ KL ( _pθ_ ( _A_ 0: _K | xt_ ) _∥p_ ref ( _A_ 0: _K | xt_ )) _._ (36) The corresponding score averages the path log ratio conditional on the endpoint: 



Replacing _ρθ_ by this score gives a surrogate preference objective on paths. The bound in Equation 36 applies to the endpoint KL, not the original DPO loss. The shared terminal prior cancels in the path ratio, leaving a sum of transition log ratios. 

18 

**3. From paths to a sampled step.** Sampling paths conditional on a given endpoint remains intractable. Following Diffusion-DPO, we approximate this posterior by the known forward noising process _q_ ( _A_ 1: _K | A_ ). Define the expected transition log ratio 



At _k_ = 1, _A_ 0 = _A_ is fixed, so this is the log density ratio for the final denoising step. The approximated path score becomes 

_ρ_<sup>_q_</sup> _θ_<sup>(</sup><sup>_A | xt_) := E</sup><sup>_q_(</sup><sup>_A_</sup> 1: _K_<sup>_|A_) log</sup><sup>_<u>pθ</u>_</sup><sup><u>(</u></sup><sup>_A_0:</sup><sup>_K_</sup><sup>_<u>| xt</u>_</sup><sup><u>)</u></sup> (39) _p_ ref ( _A_ 0: _K | xt_ )<sup>=</sup><sup>_K_E</sup><sup>_k,Ak∼q_(</sup><sup>_·|A_)</sup><sup>_gθ,k_(</sup><sup>_A, Ak| xt_)</sup><sup>_,_</sup> where _k_ is uniform on _{_ 1 _, . . . , K}_ . Thus, the path sum can be estimated by sampling a single step. For a fixed pair, write _gθ,k_<sup>_w_and</sup><sup>_g_</sup> _θ,k_<sup>_l_forthetwotransitionscoresatthesamesampledstep.Since</sup> _−_ log _σ_ is convex, Jensen’s inequality gives 

_Lq_ := _−_ E _D_ log _σ_ � _βK_ E _k,q_ [ _gθ,k_<sup>_w−g_</sup> _θ,k_<sup>_l_]</sup> � _≤−_ E _D,k,q_ log _σ_ � _βK_ [ _gθ,k_<sup>_w−g_</sup> _θ,k_<sup>_l_]</sup> � =: _L_ step _._ (40) Here _q_ supplies the noisy marginals for both actions. This bound makes training possible with one sampled noise level per pair. It applies to the objective after the forward process approximation. 

**4. From Gaussian transitions to denoising errors.** The remaining transition scores can be evaluated through noise prediction. For _k ≥_ 2, inserting the forward posterior into the log ratio gives 



In the Gaussian diffusion construction, the forward posterior has mean _µq,k_ , and the reverse models have means _µϕ,k_ and a shared fixed covariance _νk_<sup>2</sup><sup>**I**.Terms involving the posterior covariance cancel</sup> between the two KLs. Under the usual noise parameterization, _µϕ,k − µq,k_ = _bk_ ( _ϵ − ϵ_ ˆ _ϕ_ ), where _bk_ depends only on the noise schedule. Consequently, 



The noise predictors are evaluated on the same _Ak_ , condition _xt_ , and step _k_ . For a Gaussian endpoint decoder with shared fixed variance, the _k_ = 1 log density ratio likewise reduces to a difference of reconstruction errors, which can be expressed as noise errors with its corresponding weight. Substituting these scores into Equation 40 yields the denoising error surrogate: the preferred action should have a smaller error relative to the reference model than the rejected action. 

**5. From denoising to flow matching.** To adapt this surrogate to our velocity predictor, we follow Liu et al. (2026) and use the linear interpolation from Equation 5: 





Subtracting the true noise gives the exact error conversion 



Thus, the denoising surrogate becomes a preference loss on velocity errors with a weight that depends on _τ_ . Following the constant weighting used by Flow-DPO, we replace the resulting weight by a constant. This is a choice of training objective, separate from the exact error conversion above. 

**6. DemoDPO objective.** We now use the normalization and score from Equations 9 and 10: 







All constant scale factors are absorbed into _β_ , with 1 _/_ 2 kept explicit to match the main text. For each candidate pair, we sample _τ ∼_ Unif[0 _,_ 1] and a shared _ϵ ∼N_ (0 _,_ **I** ), independently of the 

19 

noises used to generate the candidates. We form _A_<sup>_′_</sup> _τ_<sup>_,s_</sup> = (1 _− τ_ ) _A_<sup>_s_</sup> + _τϵ_ for _s ∈{w, l}_ . Using demonstration MSE to rank candidates and retaining pairs with _M_ pair = 1 gives 

_L_ DemoDPO = _−_ E _D,τ,ϵ_ �log _σ_ � _β_ � _sθ_ ( _A_<sup>_w_</sup> _| xt_ ; _τ, ϵ_ ) _− sθ_ ( _A_<sup>_l_</sup> _| xt_ ; _τ, ϵ_ )���� _M_ pair = 1� _._ (48) This is Equation 11: the sampled preference loss uses velocity errors, a frozen reference policy, and demonstration rankings, without evaluating action likelihoods or complete generation paths. 

### B.2 TRAINING ALGORITHM 

Algorithm 1 summarizes one DemoDPO update. Sampling and candidate selection are performed independently for each training example. mse averages over valid action entries while retaining the batch dimension and, for candidate ranking, the candidate dimension. M denotes _M_ pair. sample ~~t~~ ime() draws _τ ∼_ Unif[0 _,_ 1] independently for each example, with the same _τ_ and _ϵ_ shared by both candidates. 

**Algorithm 1** DemoDPO with Demonstration-Guided Group Ranking 

```
#fn,ref:velocitypredictorandfrozenreference
#x,A_demo:conditionanddemonstratedactions
#G,pair_epsilon,beta:candidatecount,pairthreshold,DPOcoefficient
```

```
A=stopgrad(sample(ref,x,G))#Gcandidatespercondition
d=mse(A,A_demo)#demonstrationerror
w,l=argmin(d),argmax(d)
M=d[l]-d[w]>=pair_epsilon#retaindistinctpairs
A_w,A_l=A[w],A[l]
tau=sample_time()
eps=randn_like(A_demo)#sharedbybothcandidates
z_w=(1-tau)*A_w+tau*eps
z_l=(1-tau)*A_l+tau*eps
v_w,v_l=eps-A_w,eps-A_l
err_w=mse(fn(z_w,x,tau),v_w)
err_l=mse(fn(z_l,x,tau),v_l)
ref_w=mse(ref(z_w,x,tau),v_w)
ref_l=mse(ref(z_l,x,tau),v_l)
```

```
s_w=-0.5*(err_w-stopgrad(ref_w))#relativepreferencescore
s_l=-0.5*(err_l-stopgrad(ref_l))
loss=-logsigmoid(beta*(s_w-s_l))
loss=(loss*M).sum()/M.sum().clamp(min=1)
```

## C IMPLEMENTATION DETAILS 

We provide the training sample construction, frozen encoder comparison, evaluation protocols, visual representation analysis, and per task RoboTwin results. 

### C.1 TRAINING SAMPLE CONSTRUCTION 

Following Fast-WAM (Yuan et al., 2026), we concatenate images from the head and two wrist cameras into a single image before feeding it to the vision encoder. Because our study aims to isolate the effect of visual representations, we instantiate the task context _ct_ as a discrete task ID and use the same conditioning across all controlled comparisons, rather than natural language instructions. At control step _t_ , each training sample contains the current visual observation _Xt_ and an action chunk _At_ = ( _at, . . . , at_ + _H_ a _−_ 1), where _at ∈_ R<sup>_Da_</sup> and _Da_ is the action dimension. During training, we randomly sample one future offset _δ ∈{_ 1 _, . . . , H_ a _}_ and use the corresponding observation _Xt_ + _δ_ for future embedding prediction. Thus, we use _N_ = 1 future target per training sample. 

20 

Sampling offsets across training provides supervision over the prediction horizon while limiting the number of future query tokens. Empirically, increasing _N_ provides limited performance gains while introducing more future query tokens and slowing training. All LeWAM variants are trained only on the RoboTwin demonstrations described in Section 4.1, while retaining the original visual pretraining of the frozen I-JEPA encoder. 

### C.2 FROZEN ENCODER COMPARISON 

The goal of this comparison is to select a frozen visual encoder for LeWAM under a fixed downstream action predictor. We compare representative checkpoints from each encoder family, with different architectures and parameter counts. We first establish two representative reference points: the Wan2.2 VAE, which provides the compressed latent used by WAMs built on VDMs, and SigLIP (Zhai et al., 2023), a widely adopted visual encoder in VLAs. We then compare MAELarge (He et al., 2022), DINOv3-Large, and V-JEPA2-Large (Assran et al., 2025). Based on this comparison, we further evaluate V-JEPA2-Huge and I-JEPA-Huge (Assran et al., 2023). 

For each encoder, we freeze all pretrained parameters and independently train the same lightweight DiT-style action predictor on RoboTwin. Future embedding prediction is disabled so that the experiment evaluates each representation solely through action generation. The action predictor architecture, current observations from multiple views, task conditioning, action horizon, optimizer, batch size, and training schedule are kept fixed across encoders. Only the encoder-specific input projection and the downstream action predictor are trained. All encoder variants use a 12-layer action predictor with a hidden size of 1280, and each encoder output is mapped to this width using a learned linear projection. After selecting I-JEPA-H/14, we use its frozen 0.6B-parameter encoder in LeWAM with an input resolution of 224 _×_ 224 and a patch size of 14 _×_ 14, producing 256 image tokens. The predictor and AdaFuse contain 402M trainable parameters, giving LeWAM 1.0B parameters in total. Following DiT (Peebles & Xie, 2023), we use AdaLN-Zero to condition the predictor on noise timestep and task embeddings. Noisy action tokens use the sampled noise timestep, while visual tokens, clean action tokens, and future queries use the time embedding evaluated at zero. Before fusion, a shared LayerNorm independently normalizes each encoder layer’s tokens across channels. 

Each observation is processed with the input range, spatial resolution, and normalization for each channel expected by its pretrained encoder. For consistency with the other encoders, we use only the current image, repeated 16 times following V-JEPA2’s official single-image protocol (Assran et al., 2025). The complete action generation and probing results are reported in Table 1 and discussed in RQ1. 

### C.3 DEMODPO ABLATION EVALUATION PROTOCOL 

The DemoDPO hyperparameter ablations in Figure 5 use a fixed evaluation subset of eight challenging RoboTwin tasks to reduce evaluation cost. All policies are trained on the complete RoboTwin 2.0 training set covering 50 tasks, comprising 2,500 clean and 25,000 randomized demonstrations, as described in Section 4.1. The subset is used only for evaluation and is not a training split by task. For each task, we evaluate 100 clean and 100 randomized cases, resulting in 1,600 evaluation cases in total. In alphabetical order, the task list is blocks ~~r~~ anking ~~s~~ ize, hanging ~~m~~ ug, open ~~m~~ icrowave, pick ~~d~~ iverse ~~b~~ ottles, place ~~a~~ 2b ~~r~~ ight, place ~~c~~ an ~~b~~ asket, stamp ~~s~~ eal, and turn ~~s~~ witch. 

Under this evaluation protocol, the sweeps in Figure 5, from _(a)_ to _(c)_ , vary _ε_ pair with _β_ = 2000 and _G_ = 4, _β_ with _ε_ pair = 10<sup>_−_3</sup> and _G_ = 4, and _G_ with _ε_ pair = 10<sup>_−_3</sup> and _β_ = 5000, respectively. 

### C.4 INFERENCE EFFICIENCY 

We benchmark all models on an NVIDIA H800 with batch size 1 and 10 denoising steps. Latency includes visual encoding, language encoding and KV-cache prefilling where applicable, and complete action denoising, with images re-encoded for every request. CPU image preprocessing, tokenization, CPU–GPU transfers, robot execution, and one-time CUDA Graph capture are excluded. We report peak PyTorch-allocated GPU memory with CUDA Graph enabled, including model parameters and graph-held tensors but excluding capture-phase peaks. 

21 

Table 3 shows that LeWAM achieves the lowest latency with CUDA Graph disabled or enabled and the lowest peak memory among the evaluated models. 

Table 3: GPU inference efficiency. Peak allocated memory is measured with CUDA Graph enabled. 

|Model|Latency|(ms) _↓_|Peak memory|Action chunk|
|---|---|---|---|---|
||CUDA Graph off|CUDA Graph on|(GiB) _↓_|length|
|_π_0_._5 (PyTorch)|162.45|53.14|7.037|50|
|LingBot-VLA-v2|711.31|126.53|12.089|50|
|Fast-WAM|325.55|102.38|23.198|32|
|LeWAM|**103.53**|**31.90**|**1.988**|32|



C.5 VISUAL REPRESENTATION ANALYSIS 

**Policy attention visualization.** Attention scores are averaged across the action tokens, all attention heads, and all transformer layers at each inference step. Figure 6 presents attention maps across several manipulation tasks using the same aggregation protocol as Figure 4. In Blocks Ranking RGB and Blocks Ranking Size, attention shifts between blocks as the policy grasps and places them in sequence. Prominent responses appear around the active block, gripper, and placement region in both clean and cluttered scenes. These observations suggest that action generation emphasizes object locations and spatial relationships, motivating the depth and segmentation probes below. 

Prior work probes pretrained visual encoders (Zhu et al., 2023; Chen et al., 2024; Assran et al., 2023; Mur-Labadia et al., 2026). We analyze the information accessible from each frozen visual representation using depth and segmentation probes. We randomly sample one timestep from each RoboTwin 2.0 training episode, obtaining 27,500 samples with synchronized front, left, and right views. The samples are divided into training, validation, and test sets using an 8:1:1 ratio. All visual encoders remain frozen. For AdaFuse, probing uses frozen fusion weights learned by LeWAM, while action success in Table 1 uses action-only training. Each probe applies LayerNorm, a linear projection to a width of 1280, and GELU to the frozen visual tokens. The depth and segmentation mask heads operate on the spatial tokens, while the mask confidence head applies mean pooling before linear prediction. Only the projection and probe heads are trained. 

For depth probing, DA3MONO-LARGE (Lin et al., 2025) generates a relative depth target independently for each view. The targets are standardized within each view and arranged in the same spatial layout as the visual input. The depth probe is trained with mean squared error over valid pixels. We evaluate depth prediction using scale and shift invariant root mean squared error (SSI-RMSE, shown as RMSE in Table 1) and Spearman rank correlation (Corr.) between the predicted and teacher depth values. 

SAM3 (Carion et al., 2026) generates instance masks and confidence targets independently for each view. Predicted masks are assigned to the teacher masks using Hungarian matching. The training objective combines binary cross entropy (BCE) for mask and confidence prediction with Dice loss. We report average precision at intersection over union (IoU) thresholds of 0.50 (AP@50) and 0.75 (AP@75). A predicted mask is correct when its IoU with a teacher mask exceeds the threshold. 

Table 1 reports the probing results together with action success. Among the individual encoders, I-JEPA-Huge performs best on both depth and segmentation. The AdaFuse representation further improves all depth and segmentation metrics relative to the final I-JEPA layer. Figure 7 presents representative depth and segmentation predictions. 

### C.6 ROBOTWIN PER-TASK RESULTS 

Table 4 reports success rates on all 50 RoboTwin 2.0 tasks under clean and randomized settings. In Table 4, results for GigaWorld-Policy and LaWAM are taken from their original papers (Ye et al., 2026a; Chen et al., 2026), while results for all other external baselines are taken from Yuan et al. (2026). LeWAM achieves an overall success rate of 92.28%, with 93.14% in clean settings and 91.42% under randomization, matching the performance of leading VLA and WAM baselines. 

22 



<!-- Start of picture text -->
Blocks Ranking RGB<br><!-- End of picture text -->



<!-- Start of picture text -->
Place the red block, green block, and blue block in the order of<br>red, green, and blue from left to right, placing in a row.<br><!-- End of picture text -->







<!-- Start of picture text -->
Blocks Ranking Size<br><!-- End of picture text -->



<!-- Start of picture text -->
There are three blocks on the table, the color of the blocks is random, move the blocks<br>to the center of the table, and arrange them from largest to smallest, from left to right.<br><!-- End of picture text -->





Figure 6: Policy attention visualizations under clean and randomized settings. Attention follows the active objects and interaction regions as the manipulation progresses. 

23 



<!-- Start of picture text -->
DINOv3 MAE V-JEPA2 V-JEPA2 I-JEPA<br>Input Target VAE SigLIP LeWAM<br>Large Large Large Huge Huge<br>DINOv3 MAE V-JEPA2 V-JEPA2 I-JEPA<br>Input Target VAE SigLIP LeWAM<br>Large Large Large Huge Huge<br><!-- End of picture text -->

Figure 7: Depth and segmentation probing across visual representations. LeWAM denotes I-JEPA with learned AdaFuse weights. Depth maps share a fixed color scale, while segmentation colors distinguish local mask instances rather than semantic categories. 

24 

Table 4: Per-task success rates (%) on the 50 RoboTwin 2.0 tasks under clean and randomized evaluation settings. Best results are shown in bold. 

|Task|_π_0|_._5|Giga<br>Pol|World-<br>icy|M|otus|LaW|AM|Fast-|WAM|LingB|ot-VA|LeW|AM|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||Clean|Rand.|Clean|Rand.|Clean|Rand.|Clean|Rand.|Clean|Rand.|Clean|Rand.|Clean|Rand.|
|Adjust Bottle|**100**|99|**100**|**100**|89|93|**100**|**100**|**100**|**100**|90|94|**100**|**100**|
|Beat Block Hammer|96|93|86|86|95|88|90|93|**99**|97|96|**98**|**99**|97|
|Blocks Ranking RGB|92|85|92|96|99|97|97|**100**|**100**|**100**|99|98|**100**|99|
|Blocks Ranking Size|49|26|44|48|75|63|93|89|**94**|**98**|**94**|96|81|80|
|Click Alarmclock|98|89|**100**|**100**|**100**|**100**|**100**|**100**|**100**|**100**|99|**100**|**100**|98|
|Click Bell|99|66|**100**|**100**|**100**|**100**|**100**|**100**|**100**|**100**|**100**|**100**|**100**|**100**|
|Dump Bin Bigbin|92|97|92|**100**|95|91|**97**|95|**97**|96|89|96|**97**|98|
|Grab Roller<br>|**100**<br>|**100**<br>|**100**<br>|**100**<br>|**100**<br>|**100**<br>|**100**<br>|**100**<br>|**100**<br>|**100**<br>|**100**<br>|**100**<br>|**100**<br>|**100**<br>|
|Handover Block|66|57|80|80|86|73|96|87|95|81|99|78|**100**|**93**|
|Handover Mic|98|97|72|72|78|63|93|98|**99**|**100**|94|96|**99**|99|
|Hanging Mug|18|17|16|12|38|38|51|43|58|**62**|40|28|**65**|53|
|<br>Lift Pot|96|85|98|98|96|99|**100**|99|**100**|**100**|**100**|99|**100**|**100**|
|Move Can Pot|51|55|76|78|34|74|**98**|93|90|88|94|**97**|93|93|
|Move Pillbottle Pad|84|61|90|90|93|96|97|90|**100**|**99**|99|**99**|97|96|
|Move Playingcard Away|96|84|78|72|**100**|96|**100**|**100**|**100**|**100**|**100**|99|**100**|**100**|
|Move Stapler Pad|56|42|92|82|83|85|**94**|**87**|77|64|91|79|93|85|
|<br>Open Laptop|90|96|96|98|95|91|**100**|**100**|98|**100**|92|94|99|**100**|
|Open Microwave|34|77|74|66|**95**|**91**|41|43|62|45|82|86|87|86|
|Pick Diverse Bottles|81|71|82|70|90|**91**|**91**|88|80|85|89|82|87|81|
|Pick Dual Bottles|93|63|86|86|96|90|**100**|95|**100**|96|**100**|**99**|99|94|
|Place A2B Left|87|82|94|88|88|79|**98**|91|95|**93**|97|**93**|81|76|
|Place A2B Right|87|84|90|92|91|87|89|94|93|**99**|**97**|95|85|80|
|Place Bread Basket|77|64|82|82|91|94|92|85|91|93|**97**|**95**|**97**|**95**|
|Place Bread Skillet|85|66|94|90|86|83|90|83|90|**93**|**95**|90|94|85|
|Place Burger Fries|94|87|98|96|98|98|93|96|96|**99**|97|95|**99**|98|
|Place Can Basket|62|62|78|74|81|76|**92**|65|71|69|81|**84**|82|67|
|Place Cans Plasticbox<br>|94<br>|84<br>|**100**<br>|**100**<br>|98<br>|94<br>|**100**<br>|95<br>|99<br>|96<br>|**100**<br>|99<br>|**100**<br>|99<br>|
|Place Container Plate|99|95|98|96|98|99|**100**|**100**|96|**100**|99|97|**100**|**100**|
|Place Dual Shoes|75|75|96|84|93|87|**98**|**94**|94|88|94|89|86|90|
|Place Empty Cup|**100**|99|90|90|99|98|99|**100**|**100**|**100**|**100**|**100**|**100**|**100**|
|Place Fan<br>|87<br>|85<br>|92<br>|94<br>|91<br>|87<br>|92<br>|93<br>|96<br>|**96**<br>|**99**<br>|93<br>|96<br>|94<br>|
|Place Mouse Pad|60|39|88|90|66|68|91|84|83|89|**93**|**96**|**93**|85|
|Place Object Basket|80|76|90|**92**|81|87|**92**|90|89|88|91|88|89|90|
|Place Object Scale|86|80|88|80|88|85|95|88|90|**97**|**96**|95|89|95|
|Place Object Stand|91|85|**100**|**98**|98|97|92|93|90|94|99|96|98|96|
|<br>Place Phone Stand|81|81|82|72|87|86|93|94|**97**|**99**|**97**|97|**97**|98|
|Place Shoe|92|93|98|96|99|97|**100**|**100**|96|99|98|98|99|97|
|Press Stapler|87|83|96|96|93|**98**|**98**|97|90|97|85|82|83|85|
|Put Bottles Dustbin|84|79|72|70|81|79|94|**92**|**95**|90|87|91|94|90|
|Put Object Cabinet|80|79|74|74|88|71|90|82|**94**|**89**|85|87|91|83|
|<br>Rotate QRcode|89|87|90|84|89|73|94|89|93|89|**96**|91|93|**92**|
|Scan Object<br>|72<br>|65<br>|60<br>|64<br>|67<br>|66<br>|**96**<br>|90<br>|89<br>|**92**<br>|**96**<br>|91<br>|89<br>|90<br>|
|Shake Bottle<br>Shake Bottle Horizontally|99<br>99|97<br>99|**100**<br>**100**|98<br>**100**|**100**<br>**100**|97<br>98|**100**<br>**100**|**100**<br>**100**|**100**<br>**100**|**100**<br>**100**|**100**<br>**100**|97<br>99|**100**<br>**100**|99<br>98|
|<br>Stack Blocks Three|91|76|70|78|91|95|90|75|95|97|99|**98**|**100**|96|
|Stack Blocks Two|97|**100**|**100**|94|**100**|98|**100**|97|**100**|**100**|**100**|98|**100**|**100**|
|Stack Bowls Three|77|71|70|72|79|**87**|**90**|80|80|81|86|83|85|86|
|Stack Bowls Two|95|96|96|92|98|98|**100**|**99**|92|98|94|98|99|95|
|S Sl|79|55|**96**|**98**|93|92|89|88|90|94|**96**|97|82|92|
|tamp ea|||||||||||||||
|Turn Switch|62|54|82|**84**|**84**|78|47|56|61|59|44|45|60|68|
|**Average**|8274|7676|8636|8504|8866|8702|9264|8980|9188|**91.78**|9290|9150|**93.14**|9142|
|**Overall**|. <br>79|.<br>.75|. <br>85|.<br>.70|. <br>87|.<br>.84|. <br>91|.<br>.22|. <br>91|<br>.83|. <br>92|.<br>.20|<br>**92**|.<br>**.28**|



25 

### C.7 LEWAM PER-TASK ABLATIONS 

Table 5 reports per-task results for the LeWAM ablations. Direct future embedding prediction provides a modest overall gain over the Action Only baseline, improving the success rate from 86.17% to 87.02%, while the flow matching variant reaches 86.21%. Adding AdaFuse produces a larger improvement to 90.69%. DemoDPO further raises the success rate to 92.28%. Compared with the AdaFuse variant, it improves 25 clean and 24 randomized per-task results. Relative to the Action Only baseline, the complete LeWAM improves 36 clean and 41 randomized per-task scores, showing that the gain is distributed across many tasks rather than concentrated in a few cases. 

Table 5: Per-task LeWAM ablations on RoboTwin 2.0. LeWAM uses I-JEPA-H/14 with 402M trainable parameters. WM (FM) uses flow matching for future embedding prediction. The remaining variants progressively add WM, AdaFuse, and DemoDPO. Best results are shown in bold. 

|Task|Act<br>On|ion<br>ly|+<br>(F|WM<br>M)|+|WM|+<br>+ Ad|WM<br>aFuse|Le|WAM|
|---|---|---|---|---|---|---|---|---|---|---|
||Clean|Rand.|Clean|Rand.|Clean|Rand.|Clean|Rand.|Clean|Rand.|
|Adjust Bottle|99|**100**|**100**|99|**100**|99|**100**|**100**|**100**|**100**|
|<br>Beat Block Hammer|98|92|98|94|96|**97**|**99**|94|**99**|**97**|
|Blocks Ranking RGB|98|96|99|98|99|98|**100**|**99**|**100**|**99**|
|<br>Blocks Ranking Size|78|63|77|76|77|68|79|**80**|**81**|**80**|
|<br>Click Alarmclock|**100**|96|**100**|98|**100**|**99**|**100**|98|**100**|98|
|Click Bell<br>|**100**<br>|99<br>|**100**<br>|**100**<br>|**100**<br>|**100**<br>|**100**<br>|**100**<br>|**100**<br>|**100**<br>|
|Dump Bin Bigbin|95|95|**100**|97|94|97|97|94|97|**98**|
|<br>Grab Roller|99|99|99|**100**|**100**|99|**100**|**100**|**100**|**100**|
|Handover Block|96|82|78|78|93|72|96|81|**100**|**93**|
|Handover Mic|**99**|98|98|98|**99**|94|98|**99**|**99**|**99**|
|<br>Hanging Mug|**65**|46|24|12|47|38|46|**54**|**65**|53|
|<br>Lift Pot|**100**|69<br>|**100**|68<br>|99<br>|72<br>|**100**|**100**|**100**|**100**|
|Move Can Pot|79|75|88|86|**98**|**94**|89|92|93|93|
|<br>Move Pillbottle Pad|84|89|94|92|96|94|**97**|**96**|**97**|**96**|
|Move Playingcard Away|99|99|**100**|98|**100**|**100**|**100**|99|**100**|**100**|
|<br>Move Stapler Pad|71|62|71|70|75|80|**94**|**86**|93|85|
|<br>Open Laptop|96|**100**|98|98|94|**100**|98|**100**|**99**|**100**|
|<br>Open Microwave|76|69|**89**|**91**|67|67|76|82|87|86|
|<br>Pick Diverse Bottles|72|75|66|68|73|66|83|78|**87**|**81**|
|Pick Dual Bottles|98|91|94|90|91|92|**99**|**94**|**99**|**94**|
|<br>Place A2B Left|82|70|81|73|**84**|**76**|79|71|81|**76**|
|Place A2B Right|76|63|81|65|79|67|83|79|**85**|**80**|
|<br>Place Bread Basket|91|90|90|86|88|93|92|92|**97**|**95**|
|Place Bread Skillet|94|79|91|81|91|86|**97**|**87**|94|85|
|Place Burger Fries|97|98|98|93|96|95|**99**|**100**|**99**|98|
|<br>Place Can Basket|46|53|73|61|73|61|75|62|**82**|**67**|
|Place Cans Plasticbox<br>|99<br>|95<br>|**100**<br>|**99**<br>|99<br>|93<br>|**100**<br>|97<br>|**100**<br>|**99**<br>|
|Place Container Plate|99|94|97|94|99|93|**100**|**100**|**100**|**100**|
|<br>Place Dual Shoes|83|80|**88**|86|76|80|81|84|86|**90**|
|<br>Place Empty Cup|**100**|**100**|**100**|98|**100**|99|**100**|**100**|**100**|**100**|
|<br>Place Fan|87|86|**97**|**94**|96|89|**97**|90|96|**94**|
|Place Mouse Pad|76|75|77|74|87|**85**|89|84|**93**|**85**|
|Place Object Basket|**94**|83|79|77|80|74|90|**90**|89|**90**|
|<br>Place Object Scale|83|75|86|80|82|80|85|92|**89**|**95**|
|Place Object Stand|85|93|92|90|92|90|94|95|**98**|**96**|
|<br>Place Phone Stand|96|84|94|81|**97**|82|**97**|**98**|**97**|**98**|
|Place Shoe|93|93|93|92|94|94|**99**|94|**99**|**97**|
|Press Stapler|**83**|79|80|75|80|83|79|**85**|**83**|**85**|
|<br>Put Bottles Dustbin|88|74|92|76|87|73|**94**|89|**94**|**90**|
|Put Object Cabinet|**92**|**84**|81|81|81|80|90|77|91|83|
|<br>Rotate QRcode|94|90|**96**|**93**|84|90|93|87|93|92|
|Scan Object|88|73|87|75|88|72|87|87|**89**|**90**|
|<br>Shake Bottle|99|**100**|99|**100**|99|**100**|99|98|**100**|99|
|<br>Shake Bottle Horizontally|**100**|**100**|**100**|**100**|99|**100**|99|99|**100**|98|
|<br>Stack Blocks Three|96|95|99|91|**100**|**99**|99|97|**100**|96|
|<br>Stack Blocks Two|99|99|**100**|**100**|**100**|98|**100**|**100**|**100**|**100**|
|<br>Stack Bowls Three|83|78|78|79|81|70|78|**89**|**85**|86|
|Stack Bowls Two|97|95|97|90|98|95|98|**97**|**99**|95|
|Stamp Seal<br>|53<br>|53<br>|53<br>|48<br>|77<br>|66<br>|79<br>|86<br>|**82**<br>|**92**<br>|
|Turn Switch|**66**|70|57|69|58|70|63|**71**|60|68|
|**Average**|8842|8392|8818|8424|8886|8518|9132|9006|**9314**|**9142**|
|**Overall**|.<br>86.|.<br>17|.<br>86|.<br>.21|.<br>87|.<br>.02|.<br>90|.<br>.69|**.**<br>**92**|**.**<br>**.28**|



26 

## D REAL-WORLD EXPERIMENTS 

**Data collection.** We conduct the real-world experiments on an AgileX dual arm platform consisting of two Piper manipulators. Demonstrations are collected through leader-follower teleoperation. The policy observes synchronized RGB images from a fixed front camera and cameras mounted on the left and right robot arms. We collect 400 episodes for each of Stack Blocks, Fold Towel, and Arrange Flowers, resulting in 1,200 episodes and approximately 11 hours of data collected in house. Both observations and actions are recorded at 30 Hz. The initial object positions are varied across episodes. All demonstrations use delta joint positions as the action representation. 

**Policy training.** All evaluated methods are trained on the same demonstrations for 20,000 optimization steps. The baselines are finetuned from their publicly available pretrained checkpoints. We use AdamW with a learning rate of 5 _×_ 10<sup>_−_5</sup> and a batch size of 256 for all methods. Each policy predicts an action chunk of 50 steps. We use the checkpoints after 20,000 supervised optimization steps for baseline evaluation. For LeWAM, we further apply DemoDPO for 1,000 updates using the same hyperparameters as in Section 4.1, including _G_ = 4 and _β_ = 5000, and evaluate the refined policy. 

**Robot evaluation.** All policies are deployed on an NVIDIA A100 GPU with 40 GB of memory and executed at 30 Hz. Before each rollout, both robot arms are returned to the same home configuration. The objects are then placed in the initial configuration sampled for that rollout. Each method is evaluated for 50 rollouts on each of the three tasks. Each rollout is capped at three minutes. 

**Task progress.** The three real-world tasks contain several sequential manipulation stages. We measure task progress by scoring each recorded rollout according to completed milestones. 

Let _sm,q,i_ denote the score of rollout _i_ produced by method _m_ on task _q_ , and let _Mq_ denote the maximum score for that task. We report task progress as 



where _Nq_ = 50. Task progress is the average rollout score as a percentage of the maximum score. 

Each milestone contributes one point and can be counted at most once in a rollout. Task dependent penalties are applied after adding the milestone points. The resulting score is clipped to the interval [0 _, Mq_ ]. All methods are evaluated from their recorded rollouts using the same scoring rules. 

**Stack Blocks (** _Mq_ = 7 **).** The objective is to form a three block stack with the red block at the bottom, the green block in the middle, and the blue block on top. The rollout receives one point for each of the following milestones: 

1. The robot lifts the red block and moves it toward the target location. 

2. The red block is placed stably. 

3. The robot lifts the green block and moves it toward the red block. 

4. The green block is placed stably. 

5. The robot lifts the blue block and moves it toward the green block. 

6. The blue block is placed stably. 

The six manipulation milestones contribute at most six points. The final configuration receives one additional point when the stack is stable and neatly aligned, and no additional point when it is stable but untidy. Two points are deducted if the final structure collapses. One point is deducted if the structure collapses during execution and the robot does not continue the task. 

**Fold Towel (** _Mq_ = 6 **).** The objective is to fold a towel and hang it on a rack. The rollout receives one point for each of the following milestones: 

1. Most of the towel is unfolded. 

27 

2. The towel is fully flattened. 

3. The towel is folded in half. 

4. The robot adjusts the orientation of the folded towel. 

5. The robot lifts the towel and moves it toward the rack. 

6. The towel is placed on the rack. 

No adjustment is applied when the towel remains stably and neatly placed on the rack. One point is deducted if the towel falls from the rack or its final placement is irregular. 

**Arrange Flowers (** _Mq_ = 6 **).** The objective is to insert three flowers into a vase while keeping the vase upright. For each flower, one point is awarded when the robot lifts it and moves it toward the vase, and another point is awarded when it is inserted successfully. The three flowers therefore contribute at most six points. If the vase is knocked over and the robot does not restore it, the rollout terminates and one point is deducted. 

### D.1 QUALITATIVE COMPARISONS 

DemoDPO raises task progress from 81.1%, 58.3%, and 50.7% to 84.9%, 60.7%, and 53.0% on Stack Blocks, Fold Towel, and Arrange Flowers, respectively, with example rollouts in Figure 8. 



<!-- Start of picture text -->
Stack Blocks<br>w/o DemoDPO<br>w/ DemoDPO<br>Fold Towel<br>w/o DemoDPO<br>w/ DemoDPO<br>Arrange Flowers<br>w/o DemoDPO<br>w/ DemoDPO<br><!-- End of picture text -->

Figure 8: Qualitative effect of DemoDPO on the three real-world tasks. For each task, the two rows compare LeWAM before and after DemoDPO refinement under the same initial configuration. 

Figure 9 compares representative rollouts for all four methods on the three real-world tasks. These examples visualize the partial task completion captured by the task progress metric. 

28 



<!-- Start of picture text -->
Stack Blocks<br>FastWAM<br>LingBot-VLA-v2<br>π 0.5<br>LeWAM<br>Fold Towel<br>FastWAM<br>LingBot-VLA-v2<br>π 0.5<br>LeWAM<br>Arrange Flowers<br>FastWAM<br>LingBot-VLA-v2<br>π 0.5<br>LeWAM<br><!-- End of picture text -->

Figure 9: Representative real-world rollouts for Stack Blocks, Fold Towel, and Arrange Flowers. For each task, the sequences show task execution from the fixed front camera under the same initial configuration across methods. 

29 

