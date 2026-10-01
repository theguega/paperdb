-:=-.==.. � --—­ 上海人工智能实验室<sup>Shanghai Artificial Intelligence Laboratory</sup> 



**InternW0-** ∆ **: A World Action Model Bridging Predictive Dynamics and Actions with 20K+ Hours of Open Data** 

**Physical Intelligence Team, Shanghai AI Laboratory** 

**Project page:** https://internrobotics.github.io/InternW0-Delta/ 



<!-- Start of picture text -->
Multi-source data TOTAL: 23,072h Key methods<br>A R C VLM<br>semantics<br>…<br>Robot UMI Ego Ego2Robot Video expert<br>11,302 h DATA FILTERING & VERIFICATION 2,075 h 4,061 h 5,634 h Observed tokensCausal Imprint (Δ) …… MoT Action expert…<br>signal + state-action visual quality<br>training only<br>boundary trimming instruction + video<br>action magnitude manual sample 4D-aware<br>distillation<br>CANONICAL 80-D STATE-ACTION<br>Full-stack<br>Benchmark evaluations open source Real-world manipulation<br>Success rate (%) ↑<br>Data tools Pretraining<br>Post- Checkpoints<br>training<br>Cup filling Cup stacking Dropper transfer<br>Code · Recipes · Weights<br>Grippers Dexterous hands<br><!-- End of picture text -->

# **Abstract** 

World Action Models (WAMs) have emerged as a promising paradigm for generalist robot manipulation by jointly modeling visual dynamics and action generation. A central challenge is how to effectively integrate complementary priors from large-scale pretrained models—including visual dynamics, scene semantics, and geometric and motion understanding—into a unified framework for robot action generation. We introduce InternW0-∆, a unified World Action Model that meets this challenge: pretrained on a large-scale heterogeneous corpus, it outperforms prior methods across diverse simulation benchmarks and real-robot platforms. InternW0-∆ brings together pretrained visual dynamics, scene-level semantic understanding, 4D geometric and motion priors, and action generation within a _Mixture-of-Transformers (MoT) framework_ . Within the World–Action MoT, a pretrained video expert and an action expert interact under scene-grounded semantic guidance from a frozen VLM, while a pretrained 4D foundation model injects geometric and motion priors through training-only distillation. To translate predictive visual dynamics into representations useful for action prediction, we introduce _Causal Imprint_ , which learns future-relevant scene changes from training-only future supervision and makes these predictive representations directly available to the action expert without requiring future-video rollout at inference. 

To support large-scale joint training, we construct a heterogeneous corpus spanning robot demonstrations, UMI data, egocentric human demonstrations, and Ego2Robot data—carefully curated and filtered, unified under a common state- 

action representation, and temporally aligned—yielding over **20K** hours of processed training data, _to our knowledge the largest open-source corpus of its kind_ . We pretrain InternW0-∆ on this heterogeneous corpus and demonstrate strong performance across diverse simulation benchmarks and real-robot platforms. We will open source training code and the model weights, infrastructure, and dataprocessing pipeline, together with processed data where licenses permit, to accelerate progress in embodied intelligence and physical AI. 

# **1 Introduction** 

World Action Models (WAMs) offer a promising approach to generalist robot manipulation by jointly modeling visual dynamics and robot actions (Ye et al., 2026b; Bi et al., 2025). A key motivation behind this formulation is that large-scale video pretraining can provide strong visual and temporal knowledge about how scenes evolve over time. However, the ability to predict future observations does not directly translate into effective robot control. Action generation further requires identifying task-relevant changes, understanding object geometry and motion, and grounding these cues in the current instruction and scene. The central challenge<sup>1</sup> is therefore to transfer predictive knowledge from visual dynamics modeling into representations that are directly useful for action generation, without requiring explicit future generation during online control (Yuan et al., 2026b). 

To face this challenge, we introduce InternW0-∆, a directed world-action architecture that combines predictive visual dynamics, temporal context, and task-conditioned scene semantics for action generation. At its core, a pretrained video expert and an action expert are coupled through a directed Mixture-of-Transformers architecture. The video expert processes a lightweight sparse memory of anchor, recent, and current observations, providing both episode-level context and recent interaction history, while a frozen vision-language model (VLM) supplies task-conditioned scene semantics to the action expert. To make predictive dynamics directly useful for control, Causal Imprint learns future-relevant scene changes from training-only future supervision and makes these representations available to the action expert. In parallel, training-only 4D-aware distillation from a Track4World (Lu et al., 2026) teacher injects geometric and motion priors into the video expert through auxiliary supervision. The directed information flow ensures that neither Causal Imprint nor the action expert takes future observations as input, with future information used only as training supervision. This allows InternW0-∆ to directly predict actions at inference without sampling future videos or invoking the distillation branch. 

To support large-scale joint training, we curate a heterogeneous corpus spanning robot demonstrations, UMI data, egocentric human demonstrations, and Ego2Robot data, drawing primarily on public datasets. These sources differ substantially in robot embodiment, control space, camera configuration, and temporal convention. We therefore convert them into a canonical state-action representation and apply systematic quality filtering and temporal alignment, resulting in over **20K** hours of processed training data. On this corpus, we adopt a two-stage training recipe that first pretrains InternW0-∆ to jointly learn visual dynamics and action generation, and then adapts the resulting checkpoint to target embodiments and tasks through post-training. 

In addition, we develop complementary infrastructure to support efficient model iteration and online execution. For model development, we optimize the training pipeline to reduce the cost of repeated architecture and hyperparameter experiments. Caching video autoencoder latents and frozen vision-language features avoids redundant encoding across repeated training runs, while layerwise compilation and activation checkpointing improve backbone throughput and memory efficiency. Together, these optimizations substantially accelerate model iteration and make largescale experimentation more practical. For deployment, context caching and compiled action execu- 

> 1This challenge aligns with the broader goal of the InternW series: connecting perception, physical prediction, and action under limited sensing and computation (Chen et al., 2026d). 

2 

tion reduce inference overhead and support asynchronous action-chunk execution. On the physical dexterous-hand deployment, the optimized runtime achieves an average controller-observed round-trip latency of 152.8 ms on a single NVIDIA RTX 5090 GPU, corresponding to a 5.11 _×_ speedup over the standard runtime. 

We will release code, checkpoints, recipes, and infrastructure covering the entire pipeline, from data processing and two-stage training to evaluation and deployment. Processed data will be shared where licenses permit, and versioned indices will reference the original samples and record our filtering decisions, so that the community can more easily reproduce our work. 

We evaluate InternW0-∆ on LIBERO-Plus (Fei et al., 2026), RoboTwin 2.0 (Chen et al., 2026c), EBench (Gao et al., 2026), and RoboDojo (Chen et al., 2026b), spanning different embodiments, task demands, and distribution shifts. Post-trained only on unperturbed demonstrations, InternW0∆ achieves the best success rates under distribution shift on both LIBERO-Plus (92.8%) and RoboTwin 2.0 Clean2Random (71.9%), while also leading on Clean2Clean (90.0%). On EBench, which targets mobile bimanual manipulation, it obtains the highest overall score of 66.0. On RoboDojo, whose memory, precision, and long-horizon tasks remain challenging for all methods, it achieves the best average success rate of 23.9%, nearly double that of the strongest prior WAM. We further demonstrate real-robot deployment on two gripper-based and two dexterous-hand platforms, adapting the same pretrained checkpoint to their respective control interfaces through posttraining. 

The main contributions of this work are as follows: 

1. **A World Action Model with action-relevant predictive representations.** InternW0-∆ couples a pretrained video expert and an action expert through a directed Mixture-ofTransformers. Causal Imprint learns future-relevant scene changes for the action expert without future-video sampling at inference, while training-only 4D-aware distillation adds geometric and motion priors to the video expert. 

2. **A scalable and reproducible data-to-deployment recipe.** We curate and unify over 20K hours of heterogeneous robot and human demonstrations under a canonical state-action representation, pretrain on this corpus, and adapt the resulting checkpoint to target embodiments through post-training. Our infrastructure further speeds up both model iteration and online execution. We will release the code, checkpoints, recipes, and filtered-data indices. 

3. **Comprehensive evaluation across embodiments, from simulation to real robots.** We evaluate InternW0-∆ on LIBERO-Plus, RoboTwin 2.0, EBench, and RoboDojo, covering single-arm, bimanual, and mobile manipulation under diverse task demands and distribution shifts. We further deploy it on gripper-based and dexterous-hand real robots, adapting the same pretrained checkpoint to each platform through post-training. 

# **2 Related Work** 

## **2.1 Vision–Language–Action and World Action Models** 

Vision–language–action (VLA) models transfer the semantic knowledge of pretrained vision– language models to robot control. RT-2 (Brohan et al., 2023) and OpenVLA (Kim et al., 2024) adapt these backbones to predict tokenized actions, while _π_ 0 (Black et al., 2025b) and GR00T N1 (NVIDIA et al., 2025) couple multimodal understanding with continuous action generation. Subsequent efforts expand the breadth of policy learning: _π_ 0.5 (Black et al., 2025a) combines heterogeneous training sources for open-world generalization, LingBot-VLA (Wu et al., 2026a;b) studies large-scale cross-embodiment learning and practical adaptation, and Qwen-RobotManip (Yuan et al., 2026a) emphasizes alignment across heterogeneous manipulation data. These works establish strong semantic and instruction-following foundations for generalist policies. 

3 

World Action Models (WAMs) additionally couple action learning with predictions of how the visual world evolves. DreamZero (Ye et al., 2026b) transfers pretrained video priors through joint video–action prediction, while Motus (Bi et al., 2025) integrates understanding, video generation, and action modeling within a mixture-of-transformers architecture. LingBot-VA (Li et al., 2026b) adopts causal video–action modeling for streaming control, and LingBot-VA 2.0 (Zhang et al., 2026b) further explores native video–action pretraining. A key design choice is whether action inference requires generating future video. Fast-WAM (Yuan et al., 2026b) separates future-video supervision from the action inference path, showing that video prediction can benefit policy learning without test-time future imagination. Following this separation, InternW0-∆ retains a trainable video expert and video-generation supervision, while using a frozen VLM for scene semantics. Its directed video–action interface allows the action expert to exploit learned predictive representations without sampling future video. 

## **2.2 Visual Representations for Robot Control** 

Beyond the policy architecture, the choice of representation determines which aspects of visual dynamics are made available to action generation. Video Prediction Policy (Hu et al., 2025) extracts predictive visual features from a video diffusion model for control. V-JEPA (Bardes et al., 2024) instead learns video representations by predicting in embedding space, and V-JEPA 2 (Assran et al., 2025) extends this approach to latent planning through action-conditioned post-training. Within robot policies, VLA-JEPA (Sun et al., 2026) uses future-state embedding prediction for pretraining, while JEPA-WAM (Lin et al., 2026) couples spatially structured transition prediction and action generation through a shared predictor. InternVLA-A1.5 (Ma et al., 2026a) distills a frozen video generator into foresight queries attached to a VLM-based policy. ST-WAM (Wang et al., 2026b) combines future VAE-latent and DINO-feature (Caron et al., 2021) prediction with semantic history retrieval, illustrating how generative and semantic prediction targets can complement each other. 

Geometric supervision provides another source of action-relevant structure. Spatial Forcing (Li et al., 2026a) aligns intermediate VLA features with pretrained 3D representations, and LingBotVLA 2.0 (Wu et al., 2026b) supervises current and future queries with depth and causal video features. Moving beyond per-frame geometry, 4D-WAM (Yang et al., 2026a) transfers trajectory-field knowledge through temporal feature-difference alignment and source-to-destination correspondence. Track4Action (Wang et al., 2026a) predicts pooled Track4World (Lu et al., 2026) descriptors from policy observations and uses the resulting features to condition action generation. 

InternW0-∆ combines predictive and geometric supervision within the video expert, with distinct roles for the two representations. Causal Imprint predicts clean video-latent changes and aligns with future video-expert features; its hidden states directly inform the action expert. Separately, we use Track4World (Lu et al., 2026) as a training-only teacher to distill clip-level geometry and motion information into the video expert. The teacher and distillation branch are discarded at inference, so this supervision does not add an extra policy inference path. 

## **2.3 Open-Source Systems for Robot Learning** 

Open robot learning depends on reusable data interfaces, training implementations, and evaluation and deployment tools in addition to model checkpoints. LeRobot (Cadene et al., 2026) provides shared infrastructure for dataset handling, policy training, and robot interaction. OpenVLA (Kim et al., 2024) and openpi (Physical Intelligence, 2025) make pretrained policies accessible through public implementations and adaptation workflows, while StarVLA (StarVLA Community, 2026) modularizes VLA development to support interchangeable components and controlled experimentation. These systems lower the cost of reproducing and extending robot policies, although their supported models, data pipelines, and deployment settings differ. 

Recent foundation-model efforts also expose larger-scale training recipes. LingBot-VLA (Wu et al., 2026a) releases checkpoints and a training and evaluation codebase. Qwen-RobotManip (Yuan et al., 2026a) documents the data-alignment pipeline underlying its manipulation policy. OpenWAM (Wang et al., 2026d) brings modular architectures, data processing, training, and deployment 

4 



<!-- Start of picture text -->
Anchor  A Recent  R Current C videonoisy ImprintCausal 30 × World–Action MoT<br>tokens tokens<br>Predicted Causal Imprint<br>Video expert<br>❄ Wan VAE … Wan 2.2-TI2V-5B<br>future world<br>video context …<br>task instruction T5 cross ⋯ ⋯<br>attention …<br>current proprio Linear masked self-attention… action chunk<br>Linear<br>cross Action expert 1 8 16 24 32<br>attention noisy action tokens<br>current observation ❄ VLM … timestep<br>action context predict what changes — and act accordingly<br><!-- End of picture text -->

Figure 1: **Architecture overview of InternW0-** ∆ **.** A frozen Wan VAE encodes sparse visual memory comprising anchor (A), recent (R), and current (C) observations. The pretrained Wan2.2-TI2V5B video expert and the ActionDiT action expert are coupled through 30 directed Mixture-ofTransformers (MoT) blocks. T5 instruction embeddings condition the video expert, while a frozen VLM provides task-conditioned scene semantics to the action expert; current proprioception is independently projected for both experts. Causal Imprint tokens encode change-oriented predictive features from recent and current observations to guide action generation. 

into a common framework for systematic WAM research. We share this emphasis on accessible research infrastructure. Alongside the policy, InternW0-∆ documents a unified data representation, multi-stage training, reusable frozen-encoder and teacher-feature caches, and compilation and memory optimizations. We will release these components together with the data-processing and filtering code, versioned filtered-data indices, training and evaluation code, model checkpoints, and deployment utilities, so that subsequent work can investigate model and representation choices with a concrete, inspectable training recipe. 

# **3 InternW0-** ∆ **Model Design** 

## **3.1 Architecture Overview** 

As shown in Figure 1, InternW0-∆ integrates pretrained visual dynamics, task-conditioned scene semantics, and action generation within a directed world-action architecture. A pretrained video expert and an action expert are coupled through a directed Mixture-of-Transformers, while a frozen vision-language model provides scene-grounded task semantics to the action expert. The video expert processes a lightweight sparse memory of anchor, recent, and current observations, with proprioceptive states conditioning both experts. These components are detailed in Sections 3.2, 3.3 and 3.6. 

To learn action-relevant dynamic representations, Causal Imprint uses training-only future supervision to capture future-relevant scene changes, while 4D-aware distillation from a frozen Track4World (Lu et al., 2026) teacher further introduces geometric and motion priors. The directed information flow prevents any forward activation path from realized future observations to the action prediction path, allowing InternW0-∆ to directly generate actions at inference without future-video rollout. We describe these representation-learning objectives and the resulting inference procedure in Sections 3.4, 3.5, 3.7 and 3.8. 

5 

## **3.2 Multimodal Context Encoding** 

**Multi-view visual encoding.** At each time step _τ_ , InternW0-∆ receives observations from up to _K_ camera views. We denote the _k_ -th view by _oτ_<sup>(</sup><sup>_k_)</sup> and use a binary indicator _mτ_<sup>(</sup><sup>_k_)</sup> _∈{_ 0, 1 _}_ to specify whether the corresponding view is available. To support heterogeneous camera configurations across embodiments, valid views are first resized and arranged into a unified visual canvas through an embodiment-dependent composition operator Π, 



The composed observation is then encoded by the pretrained video VAE, 



where _zτ_ serves as the visual latent input to the video expert. 

**Proprioceptive encoding.** In addition to visual observations, InternW0-∆ conditions on the current proprioceptive state _st_ , expressed in the canonical representation defined in Section 4.1. The canonical state stores the current joint positions, absolute end-effector (EEF) pose, and gripper or hand configuration in fixed semantic slots. We independently project the canonical state into the embedding spaces of the video and action experts, 



where _Es_<sup>_v_and</sup><sup>_E_</sup> _s_<sup>_a_areseparatelyparameterizedstateencoders,eachimplementedasasinglelin-</sup> ear layer, and _et_<sup>_v_and</sup><sup>_e_</sup> _t_<sup>_a_denotetheresultingproprioceptiveembeddingsforthevideoandaction</sup> experts, respectively. 

**Action encoding.** To support heterogeneous embodiments, robot-specific control signals are mapped into the canonical action representation defined in Section 4.1, following the fixed semantic layout in Table 1. Joint, gripper, and hand actions specify absolute target configurations, whereas EEF actions specify motion relative to the current EEF pose. We denote an action chunk and its validity mask by 



where _Ha_ denotes the action horizon, _Da_ = 80 is the dimensionality of the canonical action space, and _MA_ indicates the valid timesteps and action dimensions for the active embodiment. The canonical action vectors are projected into the action expert’s embedding space through an action encoder, 



where _EA_ is implemented as a single linear layer and _et_<sup>_A_denotes the resulting action-token embed-</sup> dings. 

**Task and scene semantic encoding.** The pretrained video expert inherits the T5-based (Raffel et al., 2020) language conditioning pathway from Wan2.2-TI2V-5B (Wan Team, 2025). Given the instruction _ℓ_ , we construct the video-side conditioning sequence as 



Retaining this pathway preserves the language conditioning learned during video pretraining. However, T5 only encodes the linguistic content of the instruction and does not directly perceive the current environment. As a result, it provides limited information about the scene in which the instruction must be executed, including the objects present in the scene, their spatial configuration, the robot–object relationships, and the current interaction state. Such scene understanding is essential for action prediction, since the same language instruction may correspond to different actions under different visual states. 

6 

We therefore introduce an additional frozen vision–language model to complement the T5 pathway with scene-level visual understanding. The VLM jointly processes all valid current views together with their view identities and the instruction, 



The resulting multimodal tokens provide the action expert with a richer understanding of the current scene while relating this visual context to the task instruction. We combine these features with the action-side proprioceptive embedding as 



which is used to condition action prediction. 

The two pathways are therefore complementary. T5 preserves the pretrained language prior of the video expert, while the VLM supplies the scene understanding required for observationconditioned control. Importantly, introducing the VLM also establishes a multimodal semantic interface that can support future agentic capabilities, such as incorporating subtask descriptions, intermediate goals, persistent memory, and execution feedback (Ichter et al., 2023; Brohan et al., 2023; Jiang et al., 2023; Huang et al., 2023). 

## **3.3 Sparse Memory Context** 

Effective action prediction requires awareness of both recent execution dynamics and the broader task context, while retaining a dense observation history introduces unnecessary computational overhead. Motivated by the observation that nearby history is most informative for short-term motion and interaction continuity, whereas distant history mainly serves as a coarse global reference, we equip InternW0-∆ with a lightweight sparse memory that preserves both levels of temporal context. Specifically, InternW0-∆ maintains a sparse visual memory consisting of an episode-level global memory, a short-term memory from the previous action chunk, and the current observation, 



where _xa_ denotes the composed multi-view observation at the beginning of the episode, _xt−Hc_ provides a reference to the state before the previous action chunk, and _xt_ is the current observation. Their corresponding video latents, _{za_ , _zt−Hc_ , _zt}_ , provide sparse episode-level and recent execution context to the video expert without maintaining a dense observation history. 

## **3.4 Causal Imprint** 

Causal Imprint is motivated by the idea that supervision from future outcomes can leave a trainingtime imprint on representations formed from the observed context.<sup>2</sup> Instead of exposing realized future observations to the action prediction path, we use them only as supervision, encouraging the model to encode future-relevant scene changes from observations available at inference time. 

Concretely, we introduce a set of learnable Causal Imprint tokens into the self-attention layers of the video expert. These tokens aggregate recent and current visual features to capture motion, interaction, and state transitions that are informative for action generation, complementing the appearance information contained in the observed visual features. Their spatial layout follows that of the current video latent, allowing each Causal Imprint token to interact with observation tokens at the corresponding spatial resolution. 

During training, we construct explicit change targets from the clean video latents before noise injection. Given the current latent _zt_ and future latent slices _{zt_ + _iρv }i_<sup>_T_</sup> =<sup>_v_</sup> 1<sup>, we define the adjacent latent</sup> differences as 



> 2The name is loosely inspired by the temporal perspective portrayed in _Interstellar_ , where information is depicted as leaving traces across different moments in time. 

7 



<!-- Start of picture text -->
Teacher targets · offline<br>GT video clip 4D-aware<br>teacher descriptor<br>⋯ Track4World Geometry 2D Motion 3D Motion Camera Visibility<br>1024D 128D 256D 18D 4D<br>Action-aligned window Aggregation + per-block L₂ norm Teacher cache 푟￿￿<br>MSE<br>Student branch<br>30 × 16 learnable queries 2-layer Transformer decoderSelf Q Cross FFN Pool& 푟￿￿<br>World–Action MoT attn attn MLP<br>Student descriptor<br>Mid layer 15<br>K, V<br>A R C Linear<br><!-- End of picture text -->

Figure 2: **4D-aware representation distillation.** A query-based student branch aggregates clean A/R/C features from the video expert and aligns them with cached Track4World descriptors via an auxiliary MSE loss, transferring geometric and motion priors without additional policy inference cost. 

where _zt_ +0 _ρv_ = _zt_ . We stack these differences along the temporal dimension to form the supervision target ∆ _Zt_ . These adjacent latent differences encourage Causal Imprint to encode visual changes along the future trajectory. The resulting Causal Imprint representations are made available to the action expert, while the clean future latents are used only to construct training targets. The complementary future-feature alignment is illustrated in Figure 4, with the corresponding objectives detailed in Section 3.7. 

## **3.5 4D-Aware Representation Distillation** 

To enrich the video expert with geometric and motion-aware representations, we introduce a training-only distillation objective using a frozen Track4World (Lu et al., 2026) teacher (See Figure 2). For each training sample, the teacher processes a ground-truth video window aligned with the action horizon. We aggregate its geometry and 2D/3D motion features, together with camera-motion and visibility statistics, into a clip-level 4D-aware descriptor _rt_<sup>T</sup><sup>_∈_</sup><sup>**R**1430following</sup> the scheme of (Wang et al., 2026a). Each component is independently _ℓ_ 2-normalized before concatenation. Teacher descriptors are precomputed and cached offline, avoiding teacher evaluation during policy training. 

The student branch uses only observations available to the policy at inference. Specifically, we extract the clean-condition hidden tokens corresponding to the anchor, recent, and current frames from the 15th VideoDiT block and linearly project them into the decoder embedding space. We then introduce 16 learnable queries and aggregate the projected visual tokens through a two-layer Transformer decoder. Each layer comprises query self-attention, cross-attention to the projected visual tokens, and a feed-forward network. Finally, the decoded queries are mean-pooled and projected to the teacher feature dimension, producing the student descriptor _rt_<sup>S.</sup> 

We align the student descriptor with the cached teacher descriptor using an auxiliary MSE loss, jointly optimized with the original WAM objectives, to distill geometric and motion priors into the video expert. This distillation introduces no additional policy inference cost, as neither the teacher nor the student branch is required at inference. 

8 



<!-- Start of picture text -->
(a)  TOKEN TYPES video stream action stream (c)  ATTENTION MASK<br>key<br>A R C F Δ Act<br>A R C F Δ Act<br>A<br>Anchor frame Recent frame Current frame Future frame Causal Imprint Action<br>R<br>(b)  WORLD–ACTION MoT LAYER C<br>masked self-attention  F<br>A R C F Δ + QKVNorm A A Cross-attn FFN 𝑥! "$% Δ<br>Video  𝑥! " R R<br>C C video context Act<br>Action 𝑥# "<br>Act Act Act + QKVNorm ΔF ΔF Cross-attn FFN 𝑥# "$% F  Δ  ←← {A, R, C, F}{R, C, Δ}<br>⋯ ⋯ Act  ← {A, R, C, Δ, Act}<br>Act Act action context<br>visible blocked<br>query<br><!-- End of picture text -->

Figure 3: **World-Action MoT layer and attention mask.** (a) The video stream contains anchor (A), recent (R), current (C), future (F), and Causal Imprint (∆) tokens, while action tokens (Act) form a separate stream. (b) The two experts compute separate query, key, and value projections for joint masked self-attention, followed by expert-specific cross-attention, feed-forward, and residual pathways. (c) Rows and columns index query and key token groups, respectively. 

## **3.6 Mixture-of-Transformers and Attention Mask** 

The video and action experts retain separate modality-specific parameters, while exchanging information through masked joint attention. The token groups and a coupled World-Action MoT layer are illustrated in Figure 3(a,b). At each coupled transformer block, the two experts independently compute their query, key, and value projections, which are concatenated for a single attention operation: 



The resulting features are then routed back to their corresponding streams, where the video and action experts continue with their own cross-attention, feed-forward, and residual pathways. Thus, joint attention serves as the communication interface between the two experts without merging their modality-specific parameters. The cross-attention modules use the video-side context _Ct_<sup>_v_and</sup> action-side context _Ct_<sup>_a_definedinEquations(6)and(8),respectively.AfterthefinalMoTblock,</sup> the action decoder uses a single linear layer to map the action expert’s output features to _Da_ - dimensional flow-velocity predictions in the canonical action space. 

As illustrated in Figure 3, the attention mask further controls the direction of information flow among the anchor, recent, current, future, Causal Imprint, and action tokens. Future-frame tokens can use the observed context for future prediction, while Causal Imprint and action tokens are prevented from directly accessing the realized future. Instead, the Causal Imprint tokens learn predictive change representations from the recent and current observations under future-derived supervision, and expose these representations to the action stream. The action expert therefore conditions on observed context and the learned Causal Imprint representation, rather than privileged future frames. 

This directed information flow separates future supervision from policy inference. Future observations provide a training signal to the video expert, but no forward activation path exposes the realized future to the action policy at inference time. 

9 



<!-- Start of picture text -->
World–Action MoT<br>Block 8 Block 20 Final ground truth G G G G G G<br>video, Δ Δ Δ Δ F F F ⋯ Δ Δ Δ F F F ⋯ Δ Δ Δ F F F video latents t+32 t+16 t<br>latents<br>Δ  student F  teacher · stop-grad<br>latentsaction Act Act Act ⋯ Act Act Act ⋯ Act Act Act Causal Imprintpredicted Δ Δ Δ MSE<br><!-- End of picture text -->

Figure 4: **Causal Imprint supervision.** Causal Imprint is trained with two complementary objectives. For future-feature alignment (left), Causal Imprint features at block 8 are aligned with spatially corresponding features of the terminal future slice at block 20 using the cosine alignment loss. For direct Causal Imprint supervision (right), the final Causal Imprint features regress adjacent differences between clean ground-truth video latents using a squared-error loss. 

## **3.7 Training Objectives** 

We jointly optimize InternW0-∆ for video generation, action generation, Causal Imprint learning, and 4D-aware representation distillation. 

**Flow matching objectives.** Following the flow-matching formulation of the pretrained video model, for a target variable _y_ , we sample Gaussian noise _ϵ ∼_ N(0, _I_ ) and a flow time _σ ∈_ (0, 1), and construct 



The corresponding flow-matching objective is 



We apply this objective to both future video generation and action prediction: 



where _Zt_<sup>_F_denotes the future video latent sequence and</sup><sup>_At_denotes the target action chunk.</sup> 

**Causal Imprint objectives.** We train Causal Imprint with the two complementary objectives illustrated in Figure 4. First, the final-layer Causal Imprint representation _h_<sup>∆</sup> _t_<sup>is directly supervised</sup> by the adjacent clean-latent differences ∆ _Zt_ defined in Equation (10). The corresponding objective is 



This objective provides direct supervision for Causal Imprint, encouraging it to encode changes associated with the subsequent evolution of the scene. In addition, video diffusion transformers have been shown to develop rich semantic, temporal, and physical representations in their intermediate features (King et al., 2026; Esmati et al., 2026). We therefore introduce a representation-alignment objective to transfer such predictive information from the video expert to Causal Imprint. Specifically, each Causal Imprint token at block _ℓs_ = 8 is aligned with the spatially corresponding feature of the terminal future slice at block _ℓt_ = 20: 



where sg[ _·_ ] denotes stop-gradient. The future-video features serve only as detached training targets and are not connected to Causal Imprint through the forward attention path. 

10 



<!-- Start of picture text -->
1 Observe 2 Prefill once 3 Action-only denoising 4 Execute<br>Causal Imprint learned  future video<br>Anchor  A Recent  R Current  C with future supervision absent online a σ ∈ℝ 32×80<br>⋯<br>no future<br>video<br>Video expert read cached Action expert N flowsteps decoding<br>A + R + C + Δ<br>1×video prefill<br>Frozen video context visual K/Vper-layer action context  N×action update<br>❄ Wan VAE<br>one pass per observation only action tokens are recomputed<br>execute → observe again<br><!-- End of picture text -->

Figure 5: **Efficient inference with cached visual context.** (1) Anchor (A), recent (R), and current (C) observations are encoded by the frozen Wan VAE. (2) Conditioned on the video context, the video expert processes the observation tokens and Causal Imprint tokens once, caching their key/value (K/V) projections at each MoT layer. Future-video tokens are omitted. (3) Starting from Gaussian noise, the action expert generates an action chunk through _N_ flow steps, reusing the cached A/R/C/∆ features and the action-side context. Only the action stream is recomputed during denoising. (4) The predicted actions are executed, and the process repeats with an updated cache when a new observation is acquired. Each observation-to-action cycle requires one video-expert prefill and _N_ action-expert updates, without future-video sampling or decoding. 

**4D-aware representation distillation objective.** We align the student descriptor _rt_<sup>Swiththe</sup> cached teacher 4D-aware descriptor _rt_<sup>Tusing an auxiliary mean squared error loss:</sup> 



where _D_ = 1430 is the descriptor dimension and sg[ _·_ ] denotes stop-gradient. The loss is averaged over training samples with valid teacher descriptors. This auxiliary supervision transfers geometric and motion priors to the video expert by backpropagating through the student branch and the selected VideoDiT hidden tokens, while the teacher remains frozen. 

**Overall objective.** The complete training objective is 



## **3.8 Efficient Inference** 

As illustrated in Figure 5, InternW0-∆ generates action chunks without sampling future video. Inference consists of a single video-expert prefill followed by iterative action-only denoising. The visual context is computed once for each new observation and reused throughout the action denoising process. 

**Visual prefill and caching.** At each policy step, we encode the anchor, recent, and current observations defined in Equation (9) using the frozen Wan VAE. The resulting observation tokens, together with the learned Causal Imprint tokens, are processed by the video expert conditioned on _Ct_<sup>_v_, without instantiating future-video tokens.We cache the self-attention key/value (K/V) projec-</sup> tions of the A/R/C/∆ tokens at each MoT layer: 



11 

where _L_ is the number of coupled MoT layers. The directed mask in Figure 3(c) prevents observation and Causal Imprint tokens from attending to action tokens. With the video-side inputs and conditioning held fixed during action denoising, these cached features are independent of the evolving action tokens and can be reused at every flow step. 

**Action-only denoising and execution.** We initialize a noisy action chunk _A_<sup>1</sup> _t_<sup>_∼_N(0,</sup><sup>_I_) with shape</sup> _Ha × Da_ , where _Ha_ = 32 and _Da_ = 80. At each flow step, the action expert recomputes only the action stream, attending to the cached visual K/V and the action tokens, while using _Ct_<sup>_a_through</sup> its cross-attention pathway. The predicted action flow velocity is 



where _fθ_<sup>_a_denotestheaction-sideflowpredictor.FollowingtheflowconventioninEquation(13),</sup> we perform _N_ integration steps from _σ_ = 1 to _σ_ = 0 to obtain the predicted action chunk _A_<sup>ˆ</sup> _t_ = _A_<sup>0</sup> _t_<sup>.</sup> No video-side token features are recomputed within this loop, and neither future-video sampling nor VAE decoding is required. 

The predicted action chunk is used for robot execution. When a new observation is acquired, the sparse memory and conditioning are updated, and the visual cache is recomputed for the next cycle. Each cycle therefore consists of one video-expert prefill and _N_ action-expert updates. 

# **4 Data** 

In this section, we introduce the data recipe of InternW0-∆, which unifies heterogeneous embodied data under a common representation for joint training. We first describe the unified representation used across different data sources. We then describe the data sources, processing rules, and filtering statistics for robot manipulation data, egocentric and ego-to-robot data, and Universal Manipulation Interface (UMI) data. 

## **4.1 Unified Representation** 

We train InternW0-∆ to use heterogeneous data sources that span diverse robot morphologies, degrees of freedom, control interfaces, coordinate conventions, and action dimensions. Directly mixing their native representations would assign inconsistent physical meanings to the same dimensions, hindering cross-embodiment learning. We therefore convert all trajectories into a canonical state–action representation with fixed semantic slots. In particular, all robot actions are mapped into a shared 80-dimensional action space, whose detailed layout is provided in Table 1. Each slot is associated with a predefined physical quantity rather than the original ordering used by the source dataset. 

For the robot state, we retain absolute quantities that describe the current robot configuration. These include the current joint positions, the absolute end-effector (EEF) pose, and the current gripper or hand configuration. The EEF pose is represented by a 3D position together with a continuous 6D rotation representation. For parallel-jaw grippers, the hand state is represented by the gripper opening, while dexterous hands use the corresponding hand joint configuration. All EEF poses are first transformed into a consistent coordinate convention before being assigned to the canonical slots. 

For the action representation, we distinguish between joint-space and task-space control signals. Joint actions are represented as absolute target joint positions, corresponding to the configuration that the low-level controller is expected to reach at the next or a future control step. EEF actions consist of a 3D translation increment and a 3D rotation vector. The latter parameterizes the relative orientation change in the local coordinate frame of the current EEF. Thus, each EEF uses a 9D absolute pose representation in the state and a 6D relative motion representation in the action, before being embedded into the canonical format. An action segment contains _Ha_ consecutive canonical action vectors, each following the same fixed semantic layout. 

12 

Table 1: Canonical layout of the 80-dimensional robot action representation. Slot ranges use zerobased, half-open indexing. Paired arm ranges are listed in left–right order. 

|Component|Slot range(s)|Dim.|
|---|---|---|
|_Arm-specific slots_|||
|Arm joints|[0, 7), [40, 47)|2_×_7|
|End-effector|[7, 16), [47, 56)|2_×_9|
|Gripper|[16, 17), [56, 57)|2_×_1|
|Hand|[17, 29), [57, 69)|2_×_12|
|_Body and mobility s_|_lots_||
|Torso joints|[29, 34)|5|
|Independent lift|[39, 40)<br>|1|
|Head|[74, 77)|3|
|Mobile base|[77, 80)|3|
|Reserved slots|[34, 39), [69, 74)|2_×_5|
|**Total**|[0, 80)|**80**|



## **4.2 Robot Data** 

## **_4.2.1 Data Sources_** 

Robot manipulation demonstrations serve as the main source of embodied supervision in our training corpus. We integrate a total of 15 datasets collected from both simulated and real-world environments, covering a wide range of manipulation scenarios, including single-arm and dual-arm tabletop tasks, dexterous manipulation, mobile manipulation, and humanoid interaction. The resulting collection spans diverse robot embodiments, task settings, sensing configurations, and control interfaces, providing broad coverage of manipulation behaviors for model training. 

**AgiBotWorld (contributors, 2024)** AgiBotWorld is a large-scale real-world manipulation dataset collected using a fleet of dual-arm humanoid robots. Its full release contains over one million trajectories and approximately 3,000 hours of demonstrations, covering 217 tasks, 87 manipulation skills, more than 3,000 objects, and over 100 real-world scenes. The data includes dual-arm manipulation, dexterous-hand interaction, tool use, and mobile manipulation, with multimodal observations from multiple cameras and tactile sensors. 

**InternData-A1 (contributors, 2025)** InternData-A1 provides large-scale manipulation data across both simulated and real-world settings. It contains more than 630K trajectories and 7,400 hours of interaction across four robot embodiments, 70 tasks, and 227 scenes, covering rigid, articulated, deformable, and fluid-object manipulation. It further includes long-horizon tasks, multi-arm collaboration, and human–robot interaction, providing complementary supervision beyond standard tabletop manipulation. 

**RoboMIND (Wu et al., 2025a) and RoboMIND 2.0 (Hou et al., 2025)** RoboMIND contains 107K human-teleoperated trajectories across 479 tasks and four robot embodiments, ranging from singlearm manipulators to dual-arm and dexterous humanoid platforms. It also provides failure demonstrations in addition to successful trajectories. RoboMIND 2.0 further extends this collection to more than 310K real-world dual-arm trajectories across six embodiments and 739 tasks, together with tactile-enhanced and mobile-manipulation episodes, substantially increasing the coverage of contact-rich and spatially extended behaviors. 

**RW-RL Dataset (Intelligence et al., 2026)** The RW-RL Dataset focuses on real-world interaction data for policy improvement beyond offline imitation learning. It contains over 1,000 hours of 

13 

interaction across multiple robot families, scenario domains, and task templates, combining teleoperated demonstrations with autonomous rollouts, human interventions, reward signals, and termination labels. These data expose the model to both successful behaviors and states encountered during policy execution and recovery. 

**Dexora (Zhang et al., 2026e)** Dexora targets high-DoF bimanual dexterous manipulation with dual robot arms and dual dexterous hands. It combines 100K embodiment-matched simulated trajectories with 10K real-world teleoperated episodes, covering both coarse arm motion and finegrained finger control. This data provides dense supervision for manipulation behaviors that require coordinated arm and finger motion beyond parallel-jaw grippers. 

**ABC-130K (Allshire et al., 2026)** ABC-130K contains more than 130K real-world bimanual teleoperation episodes, corresponding to over 3,500 hours of interaction across 195 manipulation tasks. Collected with dual-arm YAM platforms, it covers manipulation primitives including pick-andplace, folding, handover, insertion, tool use, and assembly. Its scale and task composition provide extensive supervision for coordinated bimanual manipulation. 

**Galaxea Open-World Dataset (Team, 2025)** The Galaxea Open-World Dataset contains more than 500 hours of real-world mobile manipulation collected with a consistent robot embodiment. The demonstrations span residential, kitchen, retail, and office environments and include fine-grained subtask-level language annotations. Compared with fixed tabletop data, it provides additional supervision for spatially extended and long-horizon manipulation that couples navigation with object interaction. 

**RoboCOIN (Wu et al., 2025b)** RoboCOIN is a multi-embodiment bimanual manipulation dataset containing more than 180K demonstrations collected across 15 robot platforms. It covers 421 tasks in 16 real-world scenarios and organizes bimanual behaviors according to coordination patterns and object properties. Its hierarchical annotations provide supervision at trajectory, subtask, and frame levels, while its embodiment diversity complements datasets collected using a single robot platform. 

**RH20T (Fang et al., 2024)** RH20T contains more than 110K contact-rich real-world manipulation sequences covering 147 tasks and seven robot configurations. In addition to RGB observations and robot proprioception, it records force, audio, depth, and, for part of the data, tactile information. Each robot trajectory is also associated with a human demonstration and language description, providing rich multimodal supervision for contact-sensitive manipulation. 

**RDT-1B (Liu et al., 2024)** The robot-data corpus used by RDT-1B aggregates 46 datasets into more than one million episodes spanning multiple robot embodiments and manipulation settings. It is further complemented by more than 6K demonstrations collected on the ALOHA dual-arm platform. We incorporate this data to increase the diversity of robot morphologies and to strengthen the coverage of coordinated bimanual behaviors. 

**RoboSet (Kumar et al., 2023)** RoboSet combines real-world teleoperation data with simulated human and expert-policy trajectories under a common data format. Its main real-world component contains approximately 31K kitchen manipulation trajectories across 40 tasks, accompanied by multi-view visual observations, actions, robot states, and rewards. The dataset therefore contributes both large-scale real-world tabletop interactions and additional simulated manipulation behaviors. 

**RealSource-World (RealSource, 2025)** RealSource-World contains more than 14 million frames from over 11K real-world dual-arm manipulation episodes collected using the RS-02 humanoid robot. The dataset covers 35 tasks across household, kitchen, office, retail, and industrial environments, with synchronized head and wrist cameras and rich robot proprioception. It addition- 

14 

Table 2: Dataset-level statistics before and after filtering. FPS denotes the frame rate declared in the training data metadata; comma-separated values indicate subsets with different frame rates. 

|Dataset|FPS|Before i|filtering|i|After filterin|ig|
|---|---|---|---|---|---|---|
|||Hours|Episodes|Hours|Episodes|Frames (M)|
|_Robot data_|||||||
|InternData-A1 (contributors,2025)|30|3,528.27|573,389|3,375.51|559,062|364.555|
|AgiBotWorld (contributors,2024)|30|2,358.63|141,563|1,698.21|105,227|183.407|
|<br>Galaxea (Team,2025)|15, 62|681.22|29,994|327.42|17,703|17.684|
|RoboMIND (Wu et al.,2025a)|30|777.01|215,240|625.79|195,869|67.586|
|<br>ABC-130K (Allshire et al.,2026)|30|3,533.30|128,996|2,587.94|94,727|279.497|
|RoboSet (Kumar et al.,2023)|_≈_5.12|21.64|9,500|20.05|9,382|0.370|
|MolmoAct2 (Fang et al.,2026)|30|68.68|3,331|33.28|1,927|3.594|
|RoboCOIN (Wu et al.,2025b)|30, 50|533.17|77,843|485.77|77,102|53.095|
|RDT-1B (Liu et al.,2024)|25|35.66|6,110|33.10|6,110|2.979|
|<br>RH20T (Fang et al.,2024)|10|1,132.10|82,894|1,130.92|82,823|40.713|
|ActionNet (Fourier ActionNet Team,2025)|30|157.72|32,121|157.22|32,120|16.980|
|Dexora (Zhang et al.,2026e)|20|26.30|7,867|26.30|7,867|1.894|
|HABIT (Song et al.,2026)|10|165.18|10,623|122.71|8,389|4.417|
|RealSource (RealSource,2025)|30|345.46|26,671|177.92|25,504|19.216|
|RWRL (Intelligence et al.,2026)|15, 30|503.37|23,861|500.06|23,844|45.060|
|**Total**|–|**13,867.71**|**1,370,003**|**11,302.20**|**1,247,656**|**1,101.047**|
|_UMI data_|||||||
|Hy-UMI-10K (Zhang et al.,2026a)|30|2,161.74|250,135|2,075.42|416,053|224.146|
|_Egocentric human data_|||||||
|EgoDex (Hoque et al.,2025)|30|821.48|333,682|772.16|310,418|83.393|
|<br>EgoVerse (Punamiya et al.,2026)|30|3,818.56|1,484,714|3,289.19|1,312,338|355.233|
|**Total**|–|**4,640.04**|**1,818,396**|**4,061.35**|**1,622,756**|**438.626**|



ally provides atomic-skill segmentation and trajectory-quality annotations, making long-horizon demonstrations accessible at finer temporal granularity. 

**MolmoAct2-BimanualYAM (Fang et al., 2026)** MolmoAct2-BimanualYAM provides more than 720 hours of real-world bimanual manipulation demonstrations collected using dual YAM arms. The dataset contains multi-view observations and language-annotated tasks covering behaviors such as garment manipulation, cable handling, grocery organization, packing, and table bussing. These demonstrations add substantial coverage of long-horizon bimanual tasks involving coordinated interactions between both arms. 

**HABIT (Song et al., 2026)** HABIT focuses on robot manipulation in environments where humans are actively present. It contains more than 10K episodes and 160 hours of demonstrations across 60 tasks, organized around collaborative, shared-space, and human-directed interactions. This setting introduces behaviors such as human–robot synchronization, yielding, and gesture-conditioned manipulation that are largely absent from robot-only demonstration datasets. 

**ActionNet (Fourier ActionNet Team, 2025)** ActionNet focuses on dexterous bimanual manipulation using humanoid robots. It contains more than 30K teleoperated trajectories, corresponding to approximately 140 hours of interaction, and covers both basic manipulation primitives and more complex two-hand behaviors using dexterous hands. The demonstrations are collected across multiple Fourier humanoid platforms and paired with manually verified language instructions, providing additional supervision for high-DoF humanoid manipulation. 

15 

## **_4.2.2 Processing Rules_** 

Our data curation builds on practices used in prior work, including Qwen-RobotManip (Yuan et al., 2026a). We apply signal anomaly and consistency filtering, static boundary trimming, visual quality filtering, and an action magnitude guard to the robot demonstrations. We additionally perform automated checks of instruction correctness and video–instruction consistency. The resulting data are further checked through manual inspection of sampled episodes to verify robot motion and representation conventions across datasets and embodiments. We release the complete filtering pipeline and its implementation as an open-source resource to support reproducible data curation and facilitate community reuse and adaptation to additional datasets and embodiments. 

**Signal anomaly and consistency filtering.** Following Qwen-RobotManip (Yuan et al., 2026a), we perform sudden-change detection and state–action consistency checks. Frames containing non-finite values or abrupt outliers, identified through deviations from smoothed trajectories and higher-order temporal changes, are removed while contiguous valid segments are retained. For physically comparable state–action dimensions with sufficient variation, we assess directional agreement after local cross-correlation alignment. We use a default agreement threshold of 0.65 and reject episodes if any required dimension falls below this threshold or if state changes systematically precede the associated actions. Comparisons involving delta actions are adapted to the corresponding control representation. 

**Static boundary trimming.** We apply an adaptive trimming rule to remove prolonged inactivity at episode boundaries. Motion is estimated from normalized frame-to-frame state differences, with rotations represented continuously over time to avoid artificial discontinuities. An adaptive threshold derived from each episode’s motion distribution identifies sustained active segments. We retain the interval spanning the first and last reliable active segments, together with a small context margin at both ends. Intermediate pauses are preserved to maintain the temporal structure of task execution. Short episodes and those without a clear separation between static and active motion are left unchanged. 

**Visual quality filtering.** We perform frame-level visual quality checks to detect black frames, blur, compression artifacts, and other image-level anomalies, using frames resized to a unified resolution for all subsequent measurements. Black frames are identified from the mean luminance and the fraction of pixels below a dark-intensity threshold. For blur detection, Sobel gradients are first used to locate edge pixels, and the mean absolute Laplacian response is then computed only over sufficiently populated edge regions to avoid false positives in textureless manipulation scenes. Compression artifacts are quantified by comparing neighboring-pixel differences at periodic 8 _×_ 8 block boundaries with those at non-boundary locations. Abrupt temporal changes are measured using frame differences normalized by the episode-level median and median absolute deviation. Extreme temporal changes, blockiness, and saturation anomalies are combined to identify suspected corruption, while decoding and shape errors are recorded separately. 

**Action magnitude guard.** After the preceding signal and visual quality checks, we apply an action magnitude guard to remove residual large-amplitude commands. In some datasets, initialization and reset routines near episode boundaries introduce sustained large actions, such as rapid arm retraction. These motions can remain temporally smooth and consistent with the recorded states, allowing them to survive the preceding filters. When present over consecutive frames, they can also distort action statistics despite normalization based on the 1st and 99th percentiles ( _q_ 01 and _q_ 99). We therefore impose additional magnitude limits in physical units. For end-effector (EEF) actions, we discard samples whose single-step translation magnitude exceeds 0.2 m or whose single-step rotation magnitude exceeds 0.5 rad. 

**Instruction correctness and video-instruction consistency filtering.** We apply an LLM-based filter to assess the correctness, clarity, and executability of each natural-language instruction. Instructions that are empty, malformed, gibberish, incomplete, or lack a clear action, object, or tar- 

16 



<!-- Start of picture text -->
1 Action Alignment Map human motion to EEF actions<br>3D Hand Keypoints Wrist-to-EEF Retargeting Filtering<br>Camera Frame Action Valid Motion Static motion Anomalous Motion<br>2 Kinematic Alignment Solve feasible robot trajectories<br>Aligned Trajectories Base search + IK<br>tracking<br>Left Right<br>joint-limit<br>joint-discontinuity<br>trajectory-completion<br>Feasible Trajectories<br>Base Candidates (Joint & Task Space ) collision<br>3 Visual Alignment Replace arm with rendered robot<br>Original Video Human Segmentation (SAM3) Remove Human( Propainter) Render & Composite (Robot)<br><!-- End of picture text -->

Figure 6: Ego and ego-to-robot data processing pipeline. Ego2Robot data undergo all stages, whereas egocentric data undergo only action alignment and action speed alignment. Action alignment maps hand motion to end-effector targets and gripper commands. Kinematic alignment couples base search with trajectory IK to obtain base poses and joint trajectories. Visual alignment combines the rendered robot with the human-removed scene using depth-aware compositing. 

get are removed before visual verification. For the remaining instructions, we perform task-level video-instruction consistency checking with a vision-language model. Using gripper-state signals recorded in the robot data, we detect opening and closing events and split each episode into actioncentered video clips. The VLM first interprets these local clips individually to obtain descriptions of the corresponding manipulation actions. We then combine the clip-level descriptions with sparsely sampled global frames from the full episode and the original instruction, and ask the VLM to determine whether the observed behavior matches the instruction and whether the task is completed. 

**Manual semantic verification.** After all automated filtering stages, we manually inspect sampled episodes from every dataset–embodiment pair to verify the interpretation of robot states and actions. We replay the joint trajectories decoded by our data loaders using the corresponding URDF and visualize the EEF poses alongside the recorded videos from all available camera views. We assess whether the visualized robot motion, end-effector poses, and gripper behavior agree with the observations. Particular attention is paid to dataset-specific conventions, including gripper opening and closing definitions and coordinate-frame conventions. This inspection helps identify semantic mismatches and verify that our data loaders correctly interpret heterogeneous annotations and map them consistently into the unified representation. 

We apply the filtering and verification stages in a fixed order. First, signal anomaly and consistency filtering removes invalid or inconsistent data while retaining contiguous valid segments. We then trim static episode boundaries, perform visual quality filtering, and apply the action magnitude 

17 

guard. The remaining data undergo LLM-based instruction screening, followed by VLM-based checks of video–instruction consistency and task completion. Finally, we manually inspect sampled episodes from each dataset–embodiment pair to verify the decoded robot motion, its agreement with the recorded observations, and the mapping of dataset-specific conventions into the unified representation. 

## **4.3 Ego and Ego2Robot Data** 

## **_4.3.1 Data Sources_** 

**EgoDex (Hoque et al., 2025)** EgoDex contains 829 hours of egocentric human demonstrations collected using Apple Vision Pro across 194 tabletop manipulation tasks. It pairs 30 Hz RGB video with 3D head, upper-body, hand poses, camera intrinsics and poses, and language descriptions, providing fine-grained supervision for dexterous manipulation. 

**EgoVerse (Punamiya et al., 2026)** EgoVerse is a continuously expanding collection of egocentric human demonstrations contributed by academic and industrial partners. We use an expanded EgoVerse snapshot of approximately 473K recordings, yielding 1.48M segmented episodes spanning 3,819 hours. The data pair egocentric video with 3D hand keypoints, 6-DoF head poses, and task descriptions, including subtask-level language annotations for industry-contributed recordings. 

## **_4.3.2 Processing Rules_** 

The processing pipeline is organized into action alignment, kinematic alignment, and visual alignment, as shown in Figure 6. Our conversion procedure is inspired by Ego2Robot (Wang et al., 2026c) and Qwen-RobotManip (Yuan et al., 2026a). Unlike their representative-keyframe kinematic checks, we incorporate coarse-to-fine trajectory validation into base selection. Ego2Robot data undergo all stages, whereas egocentric data undergo only action alignment and action speed alignment. 

**Action alignment.** Action alignment converts heterogeneous egocentric human data into a standardized representation of bimanual wrist states and actions. Specifically, we treat the wrist pose of each 3D hand as the corresponding end-effector pose. To obtain a unified camera-relative representation, we use the camera extrinsics to transform each end-effector pose from the source frame into the instantaneous camera frame and construct the corresponding state: 



Here, **p**<sup>_s_</sup> _Et_<sup>and</sup><sup>_Rs_</sup> _Et_<sup>denotetheend-effectorpositionandorientationinthesourceframe</sup><sup>_s_,respec-</sup> tively. _R_<sup>_C_</sup> _s_<sup>_t_and</sup><sup>**t**</sup><sup>_C_</sup> _s_<sup>_t_specify the transformation from the source frame to the camera frame</sup><sup>_C_</sup> _t_<sup>, and</sup><sup>_g_</sup> _t_ denotes gripper openness. Notably, we estimate gripper openness from human fingertip geometry. The canonical mapping uses the distance between the thumb tip and the midpoint of the index and middle fingertips: 



All fingertip positions are measured in meters and expressed in a common coordinate frame at the same timestamp. Distances of 5 cm and 7 cm map to fully closed ( _gt_ = 0) and fully open ( _gt_ = 1) gripper targets, respectively. Following the action parameterization used for robot data, we define actions using camera-relative position deltas, EEF-local rotation deltas, and absolute next-frame gripper targets: 



After action alignment, we retain an episode only if at least one hand has valid pose observations spanning more than 1 s and covering more than 70% of the episode duration. We then discard 

18 

**Algorithm 1** Coupled Base Search and Trajectory IK for Bimanual Arms 

**Require:** Robot model M; _{_ T _a_ , W _a}a∈{L_ , _R}_ ; acceptance criteria and search limits **Ensure:** Base poses and joint trajectories ( _bL_ , _bR_ , **Q** _L_ , **Q** _R_ ), or ∅ 1: **for** _a ∈{L_ , _R}_ **do** 2: B _a ←_ Candidates(T _a_ , W _a_ ) 3: **for** _s ∈_ (coarse, medium) **do** 4: B _a ←_ SearchArmTrajectories(B _a_ , T _a_ , W _a_ ; _s_ ) 5: **end for** 6: **end for** 7: P _←_ PairBases(B _L_ , B _R_ ) 8: **for** _p_ = ( _bL_ , _bR_ ) _∈_ P in order **do** 9: ( _v_ , **Q** _L_ , **Q** _R_ ) _←_ SolveFullTrajectory( _p_ , T _L_ , T _R_ ) 10: **if** _v_ **then** 11: **return** ( _bL_ , _bR_ , **Q** _L_ , **Q** _R_ ) 12: **end if** 13: **end for** 14: **return** ∅ 

episodes in which both hands are static, defined by each hand having position variance below 5 _×_ 10<sup>_−_4</sup> m<sup>2</sup> and rotation variance below 0.1 rad<sup>2</sup> . For hand trajectories, we apply the trajectory outlier detection described in the robot data section and additionally reject episodes with excessive linear or angular speeds. We apply the same anomaly checks to the available camera trajectories. To limit the effects of substantial locomotion during manipulation, we also remove episodes with camera displacement greater than 1 m. 

**Kinematic alignment.** Kinematic alignment couples base selection with inverse kinematics (IK): the outer search proposes base configurations, while the inner solver holds each base fixed and computes joint trajectories to assess its suitability. Unlike the representative-keyframe checks described in Ego2Robot and Qwen-RobotManip, we incorporate coarse-to-fine trajectory validation into the search to detect tracking failures and discontinuous changes between IK solution branches. 

Algorithm 1 summarizes the coupled base search and trajectory IK procedure for bimanual arms. The shared robot model M is loaded in MuJoCo (Todorov et al., 2012) from a robot description file and includes the kinematic structure, joint limits, collision geometry, and visual geometry required for IK, collision checking, and rendered occupancy evaluation. For arm _a_ , T _a_ contains the endeffector targets and gripper commands, W _a_ is its workspace index, and B _a_ contains base candidates with their available joint trajectories and evaluation statistics. The stage index _s_ takes the values coarse and medium in sequence, with progressively denser trajectory evaluation frames. 

SearchArmTrajectories uses workspace seeds and prior trajectories for base-fixed IK, retaining a ranked shortlist subject to stage-specific tracking, joint-limit, joint-discontinuity, and trajectorycompletion criteria and search limits. PairBases forms candidate pairs, rejects those violating minimum base separation or exhibiting medium-stage cross-arm collisions, and orders the survivors. Both functions use stage-specific priorities based on motion quality, rendered image occupancy, and layout preferences without relaxing acceptance criteria. Visual preferences favor less base and proximal-arm intrusion while also considering overall robot coverage. SolveFullTrajectory then solves full-frame trajectories with cached arm/base solutions reused when available and returns a pass flag _v_ after trajectory-quality and cross-arm collision checks. Both IK routines use hierarchical updates combining position and approach tracking, contact-point and orientation refinement, and projected joint-space regularization, as detailed in Appendix A. The search returns the first passing pair in the scheduled order. 

**Visual alignment.** Visual alignment uses SAM3 (Carion et al., 2025) to segment human regions and ProPainter (Zhou et al., 2023) to remove them from the source video. Using the selected base configurations and joint trajectories, the target robot is rendered from the original camera viewpoint and composited into the inpainted scene. 

19 



Figure 7: Qualitative Ego2Robot synthesis results. After action and kinematic alignment, the robot is rendered from the original egocentric viewpoint and composited into the human-removed scene with depth-aware occlusion. 

Table 3: Source-video coverage and robot training data produced by the Ego2Robot pipeline. 

|Dataset|Sourc|e data|Conver|ted data||Final data||
|---|---|---|---|---|---|---|---|
||Hours|Episodes|Hours|Episodes|Robot-hours|Episodes|Frames (M)|
|EgoDex (Hoque et al.,2025)|772.16|310,418|310.03|149,753|1,702.33|1,041,346|183.852|
|EgoVerse (Punamiya et al.,2026)|3,289.19|1,312,338|791.51|133,697|3,931.43|2,689,167|424.410|
|**Total**|**4,061.35**|**1,622,756**|**1,101.55**|**283,450**|**5,633.77**|**3,730,513**|**608.262**|



Depth-aware compositing calibrates estimated scene depth using metric hand keypoints and checks scale consistency near the manipulated objects. Object masks constrain occlusion decisions, and temporal hysteresis reduces unstable visibility changes across frames. The resulting videos are paired with the corresponding smoothed end-effector targets and gripper commands, with validity masks retained for downstream training, as shown in Figure 7. 

**Action speed alignment.** To better match the motion speed of the robot data, we slow down EgoVerse and EgoDex trajectories by a factor of two. This doubles the trajectory duration while preserving the spatial path. 

## **4.4 UMI Data** 

## **_4.4.1 Data Sources_** 

**Hy-UMI-10K (Zhang et al., 2026a)** The public release of Hy-UMI-10K contains approximately 250K episodes spanning 2,162 hours of bimanual manipulation. It pairs head- and wrist-camera RGB video with 6-DoF gripper poses, gripper openness, and task descriptions. 

## **_4.4.2 Filtering_** 

Inspired by recent UMI data processing practices (Zhang et al., 2026a; Team et al., 2026), we apply the following filtering procedure. We first perform basic segment cleanup and reject any segment containing a gripper-pose quaternion _q_ with _∥q∥_ 2 _≤_ 10<sup>_−_8</sup> . We then discard segments in which both hands are static, defined by each hand having position variance below 5 _×_ 10<sup>_−_4</sup> m<sup>2</sup> and rotation variance below 0.1 rad<sup>2</sup> . We follow the trajectory anomaly filtering procedure described in section 4.2.2 and additionally reject trajectories with excessive linear or angular speeds or abrupt position or orientation jumps between adjacent frames. 

## **4.5 Data Filtering and Conversion Statistics** 

Table 2 reports dataset-level hours and episode counts before and after filtering, together with retained frame counts, for robot, egocentric human, and UMI data. Table 3 reports the Ego2Robot 

20 

conversion of EgoDex and EgoVerse only: the filtered human videos used as input, the unique source coverage that completes synthesis, and the robot training data obtained from conversion. 

**Data filtering.** After the processing rules above, the robot collection is reduced from 13,867.71 hours across 1,370,003 episodes to 11,302.20 hours across 1,247,656 episodes, comprising 1,101.047 million frames. EgoDex and EgoVerse jointly retain 4,061.35 of 4,640.04 hours and 1,622,756 of 1,818,396 episodes, totaling 438.626 million frames; the retained durations are 772.16 and 3,289.19 hours, respectively. Hy-UMI-10K is reduced from 2,161.74 hours across 250,135 episodes to 2,075.42 hours and 224.146 million frames. Retained UMI trajectories are segmented into multiple training episodes, so the episode count rises to 416,053. 

**Ego2Robot conversion.** The source-data column of Table 3 is the filtered EgoDex and EgoVerse videos from Table 2. The converted-data column is successful source coverage: the union of intervals, across robot embodiments and processing batches, for which at least one embodiment completes synthesis. Each covered original source episode is counted once, even if only part of it is converted; only the successful intervals contribute to covered duration. The final-data column is the exported robot training set. 

Completing synthesis requires a selected base configuration and a joint trajectory that satisfy the tracking, joint-limit, joint-discontinuity, and trajectory-completion criteria in Kinematic Alignment, together with base-separation and cross-arm collision checks for bimanual setups. An interval with no such configuration cannot be converted. Failures in subsequent synthesis stages, including visual alignment, likewise omit an interval from successful coverage, so unsuccessful coverage is not attributed solely to kinematic infeasibility. 

Robot-hours sum usable durations across embodiments, reported before the factor-of-two slowdown in Action Speed Alignment, and robot episodes count the exported training segments of at least 32 action steps. One source episode can therefore yield multiple robot episodes, and the final training volume is larger than the unique source coverage. Across EgoDex and EgoVerse, the pipeline covers 1,101.55 hours from 283,450 unique source episodes and yields 5,633.77 robot-hours across 3,730,513 training episodes, totaling 608.262 million frames. 

# **5 Training** 

We train InternW0-∆ in two stages. The first stage jointly optimizes a pretrained video expert and a randomly initialized action expert to a heterogeneous mixture of robot trajectories. The second stage specializes the resulting checkpoint to a target embodiment and task distribution. 

## **5.1 Pretraining** 

**Data and temporal sampling.** Pretraining uses a heterogeneous hybrid dataset, including real robot demonstrations (80%), ego-to-robot (Ego2Robot) data (10%), UMI-collected demonstrations (8%), and egocentric (Ego) demonstrations (2%). These data sources cover different robot forms, camera layouts, state conventions, and task vocabularies. Each data source is converted to a canonical 80-dimensional state and action space by its adapter before entering the hybrid dataset. The dataset streams are weighted according to the number of valid start frames, and a trajectory-level balancing mechanism prevents a small number of longer segments from dominating the hybrid dataset. We use fixed time windows for training instead of using complete segments. A training sample contains 33 video frames and a 32-step action block. The video stream and action stream are sampled at a fixed frequency ratio of 4; the most recent observation lags the current observation by 32 action steps, and the anchor point is taken from the beginning of the episode. Images are packed into a 384×256 training canvas used by the pretraining scheme. We initially pretrain InternW0-∆ without 4D-aware representation distillation, and then continue pretraining with the auxiliary distillation objective described in Section 3.5 alongside the existing training objectives. During this final phase, each per-GPU mini-batch of 16 samples includes one sample paired with 

21 

Table 4: Pretraining settings. 

|Setting|Value|
|---|---|
|Canonical state/action dimension|80|
|Video resolution|384_×_256|
|Video frames|33|
|Action horizon|32|
|Video-to-action frequency ratio|4|
|Recent-frame offset|32 action steps|
|Batch size|16|
|Gradient accumulation|1|
|Optimizer|AdamW<br>|
|Learning rate|5_×_10<sup>_−_5</sup><br>|
|Weight decay|1_×_10<sup>_−_2</sup>|
|Learning-rate schedule|5% warm-up followed by cosine decay|
|Mixed precision|bfloat16|
|Video expert|Wan2.2-TI2V-5B (Wan Team,2025)|
|Action expert|ActionDiT with random initialization|
|VLM|RynnBrain1.1-2B (frozen) (Dang et al.,2026)|
|Video flow-matching loss weight|0.5|
|Action loss weight(Robot/UMI/Ego2Robot/Ego)|1.0/0.5/0.1/0.1|
|Causal Imprint loss weight|0.5|
|Future-feature alignment loss weight|0.1|
|Distillation loss weight|0.1|



an offline-cached Track4World (Lu et al., 2026) descriptor. The auxiliary distillation loss is applied only to this sample. 

**Initialization.** The video expert model is initialized using Wan2.2-TI2V-5B (Wan Team, 2025). The action expert model is randomly initialized. The visual language encoder is initialized using RynnBrain1.1-2B (Dang et al., 2026) and remains frozen during training. We use two language pathways, T5 (Raffel et al., 2020) and VLM. The two language paths assume different responsibilities during training. Instructions are converted into T5 embeddings for use by the video expert, thus preserving the pre-trained language-video interface of Wan2.2. T5 receives instructions but not the current scene, so its embeddings alone cannot determine which object is being referred to, its position in the camera view, or the progress of the task. To provide this missing observational basis, a valid current view and the same instructions are jointly passed to the RynnBrain1.1-2B vision language model. Its dense hidden states are appended with a proprioceptive token and provided to the action expert. 

**Implementation and hyperparameters.** We pre-train our model on 256 NVIDIA A800 GPUs using bfloat16 mixed precision. The per-GPU batch size is 16, with no gradient accumulation, resulting in an effective global batch size of 4,096. We use the AdamW optimizer with a learning rate of 5 _×_ 10<sup>_−_5</sup> and a weight decay of 1 _×_ 10<sup>_−_2</sup> . The learning-rate schedule consists of a 5% warm-up phase followed by cosine decay. The weights for the video, Causal Imprint, and future-feature alignment loss are set to 0.5, 1.0, 0.5, and 0.1, respectively. We use data-source-dependent action loss weights: 1.0 for robot demonstrations, 0.5 for UMI-collected demonstrations, and 0.1 for both ego-to-robot (Ego2Robot) data and egocentric (Ego) demonstrations. Each weight scales only the action loss of samples from the corresponding data category. Pretraining consists of two consecutive phases. We first train for 235K optimization steps without 4D-aware representation distillation. We then continue training for an additional 10K steps, adding the distillation objective described in Section 3.5 with a weight of 0.1 while retaining the existing training objectives. The complete 

22 

Table 5: Simulation post-training configurations. LIBERO-Plus is used only for evaluation. RoboTwin 2.0 is trained on the clean split and evaluated on the randomized. 

|Benchmark|Training split|Views|Resolution|Normalization|Epochs|
|---|---|---|---|---|---|
|EBench (Gao et al.,2026)|EBench train|3|384_×_256|Z-score|10|
|RoboDojo (Chen et al.,2026b)|RoboDojo train|3|384_×_256|Z-score|10|
|LIBERO-Plus (Fei et al.,2026)|LIBERO (Liu et al.,2023) train|2|384_×_256|Min–max|15|
|RoboTwin 2.0 (Chen et al.,2026c)|Clean|3|384_×_256|Z-score|10|



pretraining run comprises 245K optimization steps and takes approximately 14 days. The main hyperparameters are summarized in Table 4. 

## **5.2 Post-Training for Simulation** 

We post-train InternW0-∆ independently on four simulation benchmarks: EBench (Gao et al., 2026), RoboDojo (Chen et al., 2026b), LIBERO-Plus (Fei et al., 2026), and RoboTwin 2.0 (Chen et al., 2026c). Each experiment starts from **the same pretrained checkpoint** and uses a benchmark-specific data adapter to convert the native observation, state, and action interfaces into our unified representation. Native action dimensions are mapped into the canonical 80-dimensional action space, while dimensions that are not defined for a given embodiment are masked from the training objective. 

Unless otherwise stated, we use a batch size of 16, eight data-loading workers per process, bfloat16 mixed precision, and a learning rate of 5 _×_ 10<sup>_−_5</sup> . Each sample is constructed from a 33-step trajectory window and contains a 32-step action chunk. Visual observations are sampled every four action steps, resulting in nine observation frames for each training sample. Multi-view observations are packed into a common 384 _×_ 256 canvas across all four benchmarks. The benchmark-specific configurations are summarized in Table 5. 

**EBench (Gao et al., 2026).** EBench is a mobile bimanual manipulation benchmark containing 26 tasks that cover tabletop manipulation, pick-and-place, and long-horizon interaction. The benchmark is designed to test a broad range of manipulation capabilities and generalization factors, including tasks that require coordinated dual-arm manipulation and mobile-base motion. We use the three RGB observations provided by the benchmark, consisting of one external view and two wrist views, and pack them into the common 384 _×_ 256 input canvas. 

The control interface contains end-effector targets for both arms, gripper commands, and planar mobile-base motion. We convert each arm pose into the unified end-effector representation, reduce the two finger coordinates of each gripper to a single gripper value, and retain the threedimensional base motion command. These components are then mapped into their corresponding entries in the canonical action space. We use z-score normalization and post-train for 10 epochs. 

**RoboDojo (Chen et al., 2026b).** RoboDojo provides 42 simulation tasks for bimanual manipulation and organizes them along five capability dimensions: generalization, memory, precision, longhorizon execution, and open-vocabulary instruction following. The tasks are designed to evaluate more than short-horizon pick-and-place behavior and include challenging multi-stage and precision-sensitive interactions. For post-training, we use the high camera together with the left and right wrist cameras and pack the three views into a 384 _×_ 256 canvas. 

The standard simulated embodiment uses a 14-dimensional bimanual joint-space interface, with six arm joints and one gripper coordinate for each arm. The benchmark-specific adapter maps the valid joint and gripper channels into the canonical action representation and masks dimensions that are not used by this embodiment. We use z-score normalization and post-train for 10 epochs. 

**LIBERO (Liu et al., 2023).** LIBERO is a single-arm manipulation benchmark built around a Franka robot and contains four standard task suites: LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, and 

23 

Table 6: Real-robot datasets used for post-training and evaluation. 

|Task|Episodes|Frames|Duration (min)|Epochs|
|---|---|---|---|---|
|**AC-One (gripper)**|||||
|Luminol reaction|162|433,715|240.95|10|
|Get a drink|137|244,440|135.80|10|
|Toast bread|304|311,631|173.13|10|
|MOF experiment|350|1,173,626|652.02|10|
|**Arx5 (gripper)**|||||
|Magnetic stirrer|202|66,353|36.86|10|
|Place tube|2,664|555,702|308.72|10|
|**Franka+XHand (de**|**xterous han**|**d)**|||
|Pour water|50|44,704|24.84|100|
|Place fruit into box|48|39,253|21.81|100|
|Stack cups|54|48,264|26.81|100|
|**TianJi Marvin+Wuj**|**i Hand (dex**|**terous han**|**d)**||
|Use dropper|101|107,703|59.84|50|
|<br>Make a sandwich|101|106,731|59.30|50|



LIBERO-10. We post-train jointly on demonstrations from these four suites. Each observation contains an external scene view and a wrist view, which are packed into the common 384 _×_ 256 input representation. 

The native action consists of a six-dimensional end-effector command and a scalar gripper command, which are mapped into the corresponding entries of the canonical action space. We use min–max normalization and post-train for 15 epochs. 

We evaluate the resulting checkpoint on the LIBERO-Plus robustness setting. While standard LIBERO measures performance close to the post-training distribution, LIBERO-Plus preserves the underlying task semantics while introducing controlled distribution shifts along seven dimensions, including object layout, camera viewpoint, robot initialization, language instruction, lighting, background appearance, and sensor noise. This provides a direct measure of how well the post-trained policy generalizes when the visual, linguistic, and execution conditions deviate from those observed during training. 

**RoboTwin 2.0 (Chen et al., 2026c).** RoboTwin 2.0 is a bimanual manipulation benchmark containing 50 tasks for the ALOHA-AgileX embodiment. We use three RGB observations, consisting of one high camera and two wrist cameras, and pack them into a 384 _×_ 256 training canvas. The native 14-dimensional action space contains six joint coordinates and one gripper coordinate for each arm, which are mapped to the corresponding bimanual entries of our canonical action representation. 

Following the Clean2Random evaluation protocol, we post-train exclusively on demonstrations collected under the clean environment configuration and evaluate the same checkpoint under both clean and progressively randomized environments. The randomized evaluation introduces controlled changes in background appearance, lighting, scene clutter, and tabletop height, with the Hard setting combining multiple sources of variation. This setting directly measures whether the policy can retain its manipulation capability when the visual and physical environment deviates from the post-training distribution. We use z-score normalization and post-train for 10 epochs. 

## **5.3 Post-Training for Real Robots** 

**Setup.** We post-train InternW0-∆ on 4 robot setups, including two **gripper-based** ones and two **dexterous-hand** ones. For **gripper-based** setups, task-specific datasets are collected on the ARX 

24 

Table 7: Post-training settings for real robots. 

|Setting|Value|
|---|---|
|Canonical state/action dimension|80|
|Video resolution|384_×_256|
|Video frames|33|
|Action horizon|32|
|Video-to-action frequency ratio|4|
|Recent-frame offset|32 action steps|
|Batch size|16|
|Gradient accumulation|1|
|Optimizer|AdamW<br>|
|Learning rate|5_×_10<sup>_−_5</sup><br>|
|Weight decay|1_×_10<sup>_−_2</sup>|
|Learning-rate schedule|5% warm-up followed by cosine decay|
|Mixed precision|bfloat16|
|Video expert|Wan2.2-TI2V-5B (Wan Team,2025)|
|Action expert|ActionDiT|
|VLM|RynnBrain1.1-2B (frozen) (Dang et al.,2026)|
|Video loss weight|0.5|
|Action loss weight|1.0|
|Causal Imprint loss weight|0.5|
|Future-feature alignment loss weight|0.1|
|Distillation loss weight|0.1|



AC-One real-robot platform and Arx5 single-arm robot. For **dexterous-hand** setups, we post-train the same pretrained checkpoint on the bimanual Franka+XHand platform and TianJi Marvin+Wuji Hand combination. We have collected 11 tasks across all 4 setups, whose details are summarized in Table 6. The post-training settings for real robots are included in Table 7. It is worth mentioning that all post-training share the same settings and start from the same pretrained checkpoint. 

**RTC post-training.** During real-robot post-training, we train the policy to predict 32-step action chunks. Following training-time RTC (Black et al., 2025d), we simulate inference delay by sampling a committed prefix length _d ∼_ U _{_ 0, . . . , 16 _}_ for each full-length training sample; for padded samples, the maximum prefix length is capped by the valid sequence length to retain at least one supervised suffix action. The first _d_ ground-truth actions remain clean, with their per-token flow timesteps set to zero, while noise is applied to the remaining actions. The action loss is computed only over valid suffix tokens and action dimensions. This trains the policy to predict an action continuation conditioned on a known prefix. At deployment, the prefix is supplied by the previously generated plan, as described in Section 6.2. 

# **6 Infrastructure** 

## **6.1 Training Infrastructure for Model Development** 

During model iteration, we identified two main sources of overhead: repeated extraction of visual features from samples revisited across epochs, and the compute and activation memory required by the MoT backbone. We address these costs at complementary stages of the training pipeline. A unified cache reuses the outputs of the frozen VAE and VLM, while layerwise compilation and checkpointing improve the execution of the MoT backbone. Together, these optimizations reduce redundant encoder work and allow the backbone to be configured for either throughput or activation memory. 

25 



<!-- Start of picture text -->
Online Cache Training<br>Cache Manager<br>VAE<br>encoder VAE latent Key generation memory  disk<br>hit hit<br>Training  Current  Metadata validation In-memory Disk cache<br>data loader sample cache<br>Frozen VLM  VLM context Hit/miss/fallback<br>encoder<br>capacity limit<br>insert into<br>Offline Cache Online Training<br>Cache Manager<br>Offline precompute<br>Key generation<br>Dataset  VAE encoder shard writing Persistent  Training  Training<br>samples Metadata validation shards data loader model input<br>Frozen VLM encoder<br>Shard write/update<br><!-- End of picture text -->

Figure 8: Feature caching for exploratory training. The Cache Manager serves VAE latents and VLM contexts to training through a common interface for online and offline materialization, reducing repeated feature extraction during model development. 

## **_6.1.1 Feature Caching_** 

For a training sample _xi_ , let 



where _zi_ is the video latent, _hi_ is the VLM context, and ( _pi_ , _si_ ) denote the task prompt and state. Because the same samples are revisited during training, repeatedly evaluating the frozen encoders expends computation without changing these representations. Feature caching therefore aims to amortize encoding across training steps while preserving preprocessing, temporal alignment, camera ordering, and model semantics. 

Figure 8 shows the Cache Manager between the data loader, feature encoders, and storage backends. Built on LiteGen (contributors, 2026), it performs lookups, validates artifact metadata, dispatches reads and writes, and records hits, misses, and fallbacks. To prevent reuse across incompatible data or encoder configurations, each artifact is identified by its sample and representation provenance. A representative key is 



where _D_ , _e_ , _k_ , and _v_ identify the dataset version, episode, temporal chunk, and camera view, and _P_ and _M_ identify the preprocessing and model versions. VLM artifacts must additionally account for any prompt or state inputs that affect _hi_ . Metadata validation rejects incompatible or corrupted artifacts; an online miss can then fall back to encoding rather than silently consuming an invalid feature. 

The same artifact contract supports two materialization strategies. In _online caching_ , the manager first checks memory and then disk for each sample. A miss invokes the corresponding VAE or frozen VLM encoder, after which the feature is inserted into memory and, when needed, persisted to disk. Entries are thus populated as the dataset is traversed: this avoids a separate preprocessing job, although the first pass still pays the encoding cost. Asynchronous persistence limits the 

26 



<!-- Start of picture text -->
layer 0 layer 1 layer 2 layer 29<br>Video Expert Video Expert Video Expert Video Expert<br>Mixed Attention Mixed Attention Mixed Attention ... Mixed Attention<br>Action Expert Action Expert Action Expert Action Expert<br>Layerwise Inductor Graph Layerwise Inductor Graph Layerwise Inductor Graph Layerwise Inductor Graph<br>Groupwise Activation Checkpointing  ❌Disable CUDA Graph<br>ZeRO Communication / Computeoverlap CUDA Graph Capture Triton Fusion Compile<br>Data Loading Tokenization Scheduling Loss & Optimization layer 0 layer 1 layer n<br>Eager Orchestrator CUDA Graph Launch<br><!-- End of picture text -->

Figure 9: Layerwise compilation of the two-stream MoT backbone. An eager orchestrator invokes independently compiled layers and groups three consecutive layers for external non-reentrant checkpointing when memory reduction is required. VLM contexts are padded before action crossattention. 

impact of writes on training. In _offline caching_ , batched encoder inference produces persistent feature shards before training begins. A manifest records artifact keys, coverage, tensor shapes and data types, preprocessing versions, and model fingerprints; a startup check validates the manifest and index. The one-time preprocessing cost can then be amortized over subsequent epochs and repeated experiments using the same data and encoder versions. 

## **_6.1.2 Layerwise MoT Optimization_** 

The MoT backbone contains 30 blocks. In each block, a video expert and an action expert update separate token streams while sharing mixed self-attention. The experimental configuration uses a 3,072-dimensional Wan video stream and a 1,024-dimensional action stream; the latter also attends to text, the current VLM context, and proprioceptive tokens. Our objective is to improve the throughput and memory efficiency of this backbone while preserving its attention routing and intended gradients. 

As illustrated in Figure 9, we compile each MoT layer as an independent PyTorch Inductor graph and invoke the layers from an eager orchestrator. Layer boundaries constrain recompilation, allow per-layer profiling, and retain opportunities to overlap ZeRO communication with computation. We compile supported regions while leaving operations that Inductor cannot lower in eager execution. Mixed attention uses PyTorch FlexAttention with a 64-token block size; constructing the block mask outside the compiled layer keeps the causal and stream-specific routing rules consistent across execution modes. To stabilize the compiled action-layer shapes, we pad VLM contexts and validity masks to 640 tokens. The current LIBERO (Liu et al., 2023) cache reaches 614 tokens after appending the proprioceptive token; added positions are masked, and longer contexts are rejected rather than truncated. 

For throughput-oriented training, the reduce-overhead mode uses layerwise compilation with CUDA Graph capture where eligible. When activation memory is the limiting factor, the eager 

27 

orchestrator instead places three consecutive compiled layers inside each external non-reentrant checkpoint region. This recomputes the grouped activations during backward while retaining independent compiler graphs for individual layers; the expert-internal and eager mixed-attention checkpoint switches are disabled for this configuration. CUDA Graph capture is also disabled on the checkpointed path because its interaction with AOTAutograd and non-reentrant checkpointing produced incorrect gradients in validation. 

## **_6.1.3 Training Efficiency_** 

The proposed optimizations can be enabled independently according to the data characteristics and resource constraints at different stages of model development. On LIBERO (Liu et al., 2023) and RoboTwin (Chen et al., 2026c), combining VAE/VLM feature caching with the workload-specific layerwise MoT configuration yields end-to-end training throughput speedups of 3.02 _×_ and 2.11 _×_ , respectively, over the corresponding unoptimized baselines. When stochastic inputs preclude feature reuse, or when the dataset makes cache materialization impractical, the throughput-oriented layerwise MoT optimization remains applicable on its own and improves end-to-end throughput by 20–30%. These optimizations reduce the time for architecture and hyperparameter exploration across training regimes. 

## **6.2 Real-Robot Deployment** 

**Deployment setup.** We deploy InternW0-∆ with inference running locally on a single NVIDIA RTX 5090 GPU with 32 GiB of memory and a control rate of 30 Hz. The observation configuration varies by platform: AC-One uses one head camera and two wrist cameras; Arx5 uses one wrist camera and two external cameras; bimanual Franka+XHand uses a single external camera; and TianJi Marvin+Wuji Hand uses one head camera and one wrist camera. 

**Asynchronous execution strategy.** Asynchronous execution requires each new action chunk to remain consistent with actions already committed to execution. Existing strategies address this requirement through action blending, inference-time guidance, or training-time prefix conditioning. A recent empirical study of real-time World Action Models compares these approaches and documents their trade-offs in precision, smoothness, and execution efficiency (Motubrain Team, 2026). 

We use the prefix-conditioned policy trained in Section 5.3, supplying committed actions as clean inputs during sampling. This avoids additional gradient-based RTC guidance (Black et al., 2025c) and is compatible with our compiled deployment pipeline. 

The controller retains a 32-step prediction horizon and launches a single background inference request every 16 execution steps while the remaining actions of the current plan execute. The worker receives a fresh observation and a temporally aligned prefix from the previous plan. Prefix tokens are held clean at timestep zero and re-clamped to the supplied actions throughout sampling. The prefix length is estimated conservatively from recent inference delays and capped at the trained maximum of 16 steps. 

**Inference optimization.** Asynchronous inference must complete within a bounded time window. After a replan is issued, the controller must receive the next action chunk before the remaining 16 actions are exhausted. At 30 Hz, this window is 533 ms; observation transport, policy inference, prefix alignment, and controller-side command dispatch all consume it, and robot-side scheduling jitter reduces the remaining margin. The policy must therefore meet a round-trip latency constraint, not merely attain high nominal throughput. 

At inference, InternW0-∆ executes the action-generation path of the directed MoT. The runtime first isolates policy inference in a separate rclpy-free process, preventing ROS callbacks, timers, and Python GIL scheduling in the robot node from interfering with CUDA launch execution. The remaining runtime exploits two invariants of this path. First, conditioning features and their attention 

28 

Table 8: Cumulative inference-latency ablation on dexterous-hand deployment. Success rates are from separate LIBERO-Plus (Fei et al., 2026) simulations. 

|Configuration|Round-trip time (ms)_↓_|Speedup|Success rate (LIBERO-plus, %)|
|---|---|---|---|
|Standard runtime|780.5|1.00_×_|92.78|
|+ process isolation|374.1|2.09_×_|92.67|
|+ feature caching & compilation|249.8|3.12_×_|92.46|
|+ context caching|217.3|3.59_×_|92.41|
|+ grouped action execution|186.2|4.19_×_|92.33|
|+ CUDA-graph replay|152.8|5.11_×_|92.23|



K/V projections are fixed within one action invocation and are reused across diffusion steps. Second, the action hot path has fixed tensor shapes for a given deployment configuration. Cached K/V tensors are packed across layers, groups of consecutive MoT action layers are compiled as fixedshape callables, and CUDA Graph Trees replay these groups to reduce launch overhead. These options do not change the model inputs, diffusion schedule, attention semantics, action interface, or RTC prefix contract. 

Table 8 reports a cumulative ablation of controller-observed round-trip latency on the dexteroushand deployment. Each request is timed from when the controller issues a replan to when the returned action chunk is available. The RTT includes controller-side request handling and transport to the isolated policy process, as well as observation preprocessing, state normalization, recent-frame/prefix alignment, conditioning, action inference, action mapping, and output postprocessing. Action inference runs on one RTX 5090 GPU. Three warm-up requests are discarded, and the following 50 requests are averaged per configuration. 

# **7 Experiments** 

## **7.1 Evaluation Protocol** 

We evaluate InternW0-∆ on four simulation benchmarks, including LIBERO-Plus (Fei et al., 2026), RoboTwin 2.0 (Chen et al., 2026c), EBench (Gao et al., 2026), and RoboDojo (Chen et al., 2026b). The post-training setup for each benchmark has been described in Section 5.2. For each benchmark, we follow its official simulator, task definitions, and success criteria without additional adaptation on the evaluation environments. Baseline results are taken from the corresponding papers or public benchmark results under matching evaluation protocols. 

## **7.2 Simulation Benchmark Results** 

## **_7.2.1 LIBERO-Plus_** 

We train InternW0-∆ on the standard LIBERO (Liu et al., 2023) training set and directly evaluate it on LIBERO-Plus (Fei et al., 2026) without adaptation to the evaluation environments. As shown in Table 9, InternW0-∆ achieves the best overall success rate of 92.8%, outperforming QwenRobotManip-Context (Yuan et al., 2026a) by 1.4 percentage points and the strongest prior WAM, Being-H0.7 (Luo et al., 2026), by 8.0 points. The largest gains appear under robot perturbations, where InternW0-∆ reaches 91.1%, compared with 87.4% for the previous best result. It also achieves the best performance under camera and language perturbations, reaching 90.6% and 92.9%, respectively. In addition, InternW0-∆ remains competitive on background and layout perturbations, with 99.4% and 88.4% success rates. 

29 

Table 9: Zero-shot robustness on LIBERO-Plus. All values are success rates (%). 

|Method|Camera|Robot|Language|Light|Background|Noise|Layout|Total_↑_|
|---|---|---|---|---|---|---|---|---|
|||**VLA**|||||||
|_π_0 (Black et al.,2025b)|13.8|6.0|58.8|85.0|81.4|79.0|68.9|53.6|
|_π_0-FAST (Pertsch et al.,2025)|65.1|21.6|61.0|73.2|73.2|74.4|68.8|61.6|
|RIPT-VLA (Tan et al.,2025)|55.2|31.2|77.6|88.4|91.6|73.5|74.2|68.4|
|OpenVLA-OFT (Kim et al.,2025)|56.4|31.9|79.5|88.7|93.3|75.8|74.2|69.6|
|<br>StarVLA (StarVLA Community,2026)|52.5|49.8|88.5|95.7|95.7|73.0|76.9|74.1|
|<br>VLA-JEPA (Sun et al.,2026)|63.3|67.1|85.4|95.6|93.6|66.3|85.1|79.5|
|VLAct (Yang et al.,2026b)|73.9|68.4|81.5|96.7|96.7|86.0|83.3|82.6|
|<br>_π_0.5 (Black et al.,2025a)|78.4|73.6|80.8|96.2|94.1|89.0|84.5|84.4|
|InternVLA-A1.5 (Ma et al.,2026a)|83.1|55.1|86.9|96.4|98.2|95.6|85.2|84.8|
|Qwen-RobotManip-Context (Yuan et al.,2026a)|89.9|83.9|86.5|**98.6**|**99.9**|**97.9**|87.5|91.4|
|||**WAM**|||||||
|Fast-WAM (Yuan et al.,2026b)|16.4|44.5|68.9|78.2|53.7|37.7|60.7|51.5|
|OpenWAM-_α_(Wang et al.,2026d)|33.8|76.1|88.0|97.0|87.1|39.8|77.5|69.2|
|<br>4D-WAM (Yang et al.,2026a)|45.2|64.3|90.6|94.3|57.7|69.1|79.2|71.0|
|<br>ST-WAM (Wang et al.,2026b)|55.4|60.1|79.3|93.0|74.2|79.5|74.3|72.8|
|Faster-WAM (Ma et al.,2026b)|67.9|49.0|92.1|94.3|57.0|82.3|82.7|75.0|
|JEPA-WAM (Lin et al.,2026)|79.2|59.2|68.2|93.3|94.6|83.6|76.1|79.2|
|<br>Cosmos-Policy (Kim et al.,2026)|75.8|63.3|81.7|96.5|88.9|92.7|82.2|82.2|
|<br>ImageWAM (Zhang et al.,2026d)|80.8|50.3|91.4|98.1|85.5|93.8|80.5|83.1|
|<br>ABot-M0.5 (Chen et al.,2026a)|70.5|87.4|88.6|94.0|89.7|75.5|85.2|83.4|
|<br>Being-H0.7 (Luo et al.,2026)|82.0|59.0|82.8|97.8|90.0|93.5|**88.5**|84.8|
|**InternW0-**∆|**90.6**|**91.1**|**92.9**|95.8|99.4|94.0|88.4|**92.8**|



## **_7.2.2 RoboTwin 2.0_** 

Under the Clean2Random protocol, models are trained on clean demonstrations only and evaluated on both Clean2Clean and Clean2Random settings. As shown in Table 10, InternW0∆ achieves the best performance in both settings, reaching 90.0% on Clean2Clean and 71.9% on Clean2Random, with an overall success rate of 81.0%. Compared with Qwen-RobotManipContext (Yuan et al., 2026a), the strongest VLA baseline, InternW0-∆ improves Clean2Random by 2.5 percentage points and the overall score by 3.9 points. The advantage over existing WAMs is larger: compared with OpenWAM- _α_ (Wang et al., 2026d), InternW0-∆ improves Clean2Random from 48.7% to 71.9%, while also slightly improving Clean2Clean from 89.4% to 90.0%. These results show that the improvement on randomized environments is achieved without sacrificing performance on the original clean distribution. 

## **_7.2.3 EBench_** 

We further evaluate InternW0-∆ on EBench (Gao et al., 2026), which covers precision-sensitive Table Top tasks, mobile pick-and-place tasks, and multi-stage Long Horizon manipulation. As shown in Table 11, InternW0-∆ achieves an overall success rate of 49.2% and the highest overall score of 66.0, demonstrating competitive performance across different manipulation settings. 

The advantage of InternW0-∆ is particularly clear on Long Horizon tasks, where it achieves 49.4% success rate and a score of 76.5, both the highest among the compared methods. Since these tasks require multiple dependent operations over extended interaction sequences, the results indicate that InternW0-∆ can effectively maintain task progress and execution consistency over longhorizon manipulation. Together with its strong performance on Simple PnP, these results show that InternW0-∆ generalizes well from basic mobile manipulation to more temporally extended behaviors. 

30 

Table 10: Evaluation results on RoboTwin 2.0 (Chen et al., 2026c) Clean2Random under clean-only training. Success rates (%) are reported on Clean2Clean and Clean2Random. 

|Method|Clean2Clean_↑_|Clean2Random_↑_|Overall_↑_|
|---|---|---|---|
||**VLA**|||
|GR00T-N1.7 (NVIDIA,2026)|43.6|20.7|32.2|
|StarVLA (StarVLA Community,2026)|58.1|10.6|34.4|
|X-VLA (Zheng et al.,2026)|68.0|20.9|44.5|
|Spatial Forcing (Li et al.,2026a)|77.2|26.7|52.0|
|ABot-M0 (Yang et al.,2026c)|70.7|36.0|53.4|
|<br>_π_0.5 (Black et al.,2025a)|73.1|47.9|60.5|
|GigaBrain-0.7 (GigaBrain Team et al.,2026)|66.8|67.9|67.4|
|Qwen-RobotManip-Context (Yuan et al.,2026a)|84.7|69.4|77.1|
||**WAM**|||
|AHA-WAM (Cai et al.,2026a)|64.3|3.2|33.8|
|Fast-WAM (Yuan et al.,2026b)|77.8|1.9|39.9|
|X-WAM (Guo et al.,2026)|70.0|25.8|47.9|
|4D-WAM (Yang et al.,2026a)|81.5|41.8|61.7|
|<br>OpenWAM-_α_(Wang et al.,2026d)|89.4|48.7|69.0|
|**InternW0-**∆|**90.0**|**71.9**|**81.0**|



Table 11: Evaluation results on EBench (Gao et al., 2026) 

|Method|Tab|le Top|Sim|ple PnP|Long|Horizon|O|verall|
|---|---|---|---|---|---|---|---|---|
||SR_↑_|Score_↑_|SR_↑_|Score_↑_|SR_↑_|Score_↑_|SR_↑_|Score_↑_|
|||**VL**|**A**||||||
|StarVLA-OFT (StarVLA Community,2026)|–|–|–|–|–|–|0.0|0.2|
|_π_0 (Black et al.,2025b)|15.7|30.0|35.0|39.0|17.0|41.0|23.6|37.0|
|X-VLA (Zheng et al.,2026)|8.6|24.0|50.0|54.0|6.2|25.0|23.7|36.0|
|InternVLA-A1 (Cai et al.,2026b)|4.3|11.0|43.0|47.0|17.9|46.0|23.9|36.0|
|_π_0.5 (Black et al.,2025a)|12.9|32.0|45.0|50.0|18.1|39.0|27.1|41.0|
|GigaBrain-0.7 (GigaBrain Team et al.,2026)|–|–|–|–|–|–|33.3|46.0|
|<br>Qwen-RobotManip (Yuan et al.,2026a)|**50.0**|**70.0**|56.5|60.0|29.9|55.0|45.6|60.0|
|||**WA**|**M**||||||
|Fast-WAM (Yuan et al.,2026b)|–|–|–|–|–|–|4.7|7.6|
|OpenWAM-_α_(Wang et al.,2026d)|30.0|44.2|**67.5**|**72.0**|44.3|72.6|**49.4**|64.7|
|**InternW0-**∆|33.6|55.2|60.0|64.3|**49.4**|**76.5**|49.2|**66.0**|



## **_7.2.4 RoboDojo_** 

As shown in Table 12, InternW0-∆ achieves highly competitive overall performance, reaching an average SR of 23.91% and a score of 30.77. Compared with DM0.5 (Dexmal, 2026), a strong VLA baseline in terms of average performance, InternW0-∆ improves the average SR by 4.57 percentage points and the score by 5.87 points. The improvement over existing WAMs is more substantial: compared with OpenWAM- _α_ (Wang et al., 2026d), InternW0-∆ increases the average SR from 11.92% to 23.91% and the score from 17.18 to 30.77. In individual categories, InternW0-∆ shows particularly strong performance on Gen-Std and Precision, reaching SRs of 33.78% and 23.25%, respectively, while also achieving a Long-Horizon score of 46.29. It remains competitive on Gen-Rand and Open tasks, where GPT-6 Astra (Zhang et al., 2026c) shows stronger performance. Overall, InternW0-∆ also exceeds GPT-6 Astra in average SR and score, reaching 23.91% and 30.77 compared with 22.48% and 28.97, respectively. 

31 

Table 12: Evaluation results on RoboDojo (Chen et al., 2026b) across six evaluation categories. 

|Method|Gen|-Std|Gen|-Rand|Pre|cision|Long-|Horizon|Me|mory|O|pen|A|vg|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||SR_↑_|Score_↑_|SR_↑_|Score_↑_|SR_↑_|Score_↑_|SR_↑_|Score_↑_|SR_↑_|Score_↑_|SR_↑_|Score_↑_|SR_↑_|Score_↑_|
|||||**V**|**LA**||||||||||
|StarVLA-_α_(StarVLA Community,2026)|5.00|7.54|0.00|0.33|4.33|9.90|6.50|14.15|2.44|3.34|0.58|0.68|3.24|6.40|
|<br>X-VLA (Zheng et al.,2026)|12.00|17.90|1.00|3.04|12.00|18.32|9.75|16.53|3.56|4.76|0.50|0.55|6.52|10.13|
|<br>_π_0.5 (Black et al.,2025a)|15.00|20.93|1.00|5.82|5.50|12.40|14.67|23.54|4.56|5.78|1.67|1.98|6.91|11.41|
|Spatial Forcing (Li et al.,2026a)|15.00|21.25|4.00|6.98|10.58|17.33|14.58|23.26|4.11|5.43|1.58|1.78|8.04|12.38|
|<br>Hy-Embodied-0.5-VLA (Zhang et al.,2026a)|17.00|21.98|0.00|1.57|8.00|13.81|14.92|25.74|12.11|13.37|0.58|0.65|8.80|13.07|
|<br>Xiaomi-Robotics-1 (Team et al.,2026)|28.00|35.65|6.00|11.44|18.83|26.69|23.67|38.39|6.56|7.81|3.58|3.94|13.93|20.07|
|Galaxea G0.5 (Liu et al.,2026)|20.00|26.74|6.00|11.16|20.42|28.25|**32.25**|44.12|7.33|8.61|1.58|1.73|14.88|20.23|
|DM0.5 (Dexmal,2026)|18.00|23.49|4.00|8.06|16.75|24.82|19.50|33.70|**47.44**|**47.74**|2.08|2.43|19.34|24.90|
|||||**LLM a**|**s Policy**||||||||||
|GPT-6 Astra (Zhang et al.,2026c)|32.67|35.32|**28.33**|**31.40**|4.00|12.65|8.25|21.45|38.67|43.04|**31.00**|**34.36**|22.48|28.97|
|||||**W**|**AM**||||||||||
|Fast-WAM (Yuan et al.,2026b)|2.00|4.33|0.00|0.34|0.00|1.96|5.17|9.14|3.44|3.55|0.42|0.42|2.03|3.48|
|AHA-WAM (Cai et al.,2026a)|6.00|10.32|0.00|1.26|2.42|5.86|2.67|8.61|2.78|2.97|0.83|0.88|2.39|4.82|
|GigaWorld-Policy (Ye et al.,2026a)|6.00|10.28|0.00|0.41|1.83|6.15|8.92|15.51|2.22|3.46|0.50|0.54|3.27|6.20|
|<br>X-WAM (Guo et al.,2026)|5.00|11.24|1.00|3.54|1.83|6.72|9.08|17.47|4.67|6.32|0.25|0.57|3.83|7.69|
|<br>OpenWAM-_α_(Wang et al.,2026d)|25.56|33.16|4.11|8.26|9.25|18.45|25.33|34.93|9.11|10.41|1.08|1.41|11.92|17.18|
|**InternW0-**∆|**33.78**|**40.98**|11.78|19.19|**23.25**|**31.98**|29.33|**46.29**|34.00|34.67|10.17|10.84|**23.91**|**30.77**|



## **7.3 Real-Robot Experiments** 

**Tasks and evaluation procedure.** We evaluate eleven tasks across four real-robot platforms: ACOne, Arx5, bimanual Franka+XHand, and TianJi Marvin+Wuji Hand. On AC-One, the tasks comprise _get a drink_ , _toast bread_ , _Luminol reaction_ , and _MOF experiment_ . _Get a drink_ episodes place a cup under the dispenser and fill it to the marked level before moving it to the right side of the dispenser; _toast bread_ episodes place two bread slices into the toaster in sequence and press its switch; _Luminol reaction_ episodes add NaOH, luminol, hydrogen peroxide, and potassium ferricyanide in sequence and mix the reaction; _MOF experiment_ episodes pour a measured solution through a funnel into a flask, return the funnel, and place and seal the flask on the stirrer. On Franka+XHand, we evaluate _pour water_ , _place fruit into box_ , and _stack cups_ . These tasks respectively pour water from a bottle into a disposable paper cup, place the language-specified fruit into a box, and stack disposable cups. On Arx5, _magnetic stirrer_ places a magnetic stir bar into a beaker and then places the beaker on the magnetic stirrer; _place tube_ transfers a test tube to a wooden test-tube rack. On TianJi Marvin+Wuji Hand, _use dropper_ picks up a rubber-bulb dropper, draws liquid from a reagent bottle, and transfers it into a test tube held in a rack. The _make a sandwich_ task places a bread slice from the bread rack onto a plate, sequentially layers cheese, egg, and lettuce on top, and adds a second bread slice to complete the sandwich. On these four platforms, each rollout starts from the robot’s home pose with a text instruction and runs under asynchronous training-time RTC (Section 6.2), which achieves real-time deployment. Figure 10 illustrates representative execution sequences for eight of these tasks. 

**Effect of pretraining on real-robot performance.** We compare models with and without our pretraining stage after task-specific post-training on two AC-One tasks, using 20 evaluation trials per task and condition. As shown in Table 13, pretraining increases success rates from 20% to 95% on _toast bread_ and from 0% to 95% on _Luminol reaction_ , supporting its benefit for these downstream real-robot tasks. 

**Generalization across physical conditions.** Beyond the training conditions, the post-trained policy exhibits qualitative generalization capabilities to unseen target water levels. For the AC-One _get a drink_ task, all 137 post-training demonstrations use the same marked target water level. At deployment, the policy successfully fills cups to different marked target levels, indicating that its 

32 



Figure 10: **Real-robot task execution.** Selected video frames illustrate eight tasks, with each row ordered chronologically from left to right. From top to bottom: _get a drink_ , _toast bread_ , _Luminol reaction_ , _MOF experiment_ , _place tube_ , _pour water_ , _stack cups_ , and _use dropper_ . The final frame of the Luminol sequence shows a close-up of the reaction outcome. 

Table 13: Real-robot performance with and without our pretraining stage, followed by task-specific post-training. Each entry reports successful trials out of 20 and the corresponding success rate. 

|Task|Withoutpretraining|Withpretraining|
|---|---|---|
|Toast bread|4/20 (20%)|19/20 (95%)|
|Luminol reaction|0/20 (0%)|19/20 (95%)|



behavior is not restricted to the single target level represented in the demonstrations. Figure 11 shows three successful executions alongside the training target-level reference. 

**Cross-embodiment adaptation.** The same pretrained checkpoint is post-trained separately for the **gripper-based** and **dexterous hands** interfaces. For grippers, AC-One uses 14-dimensional bimanual absolute joint targets and Arx5 utilizes 6-dimensional end-effector poses plus a 1-dimensional gripper. For dexterous hands, Franka+XHand uses 6-dimensional end-effector commands per Franka arm together with 12-dimensional dexterous hand commands per hand, whereas TianJi Marvin+Wuji Hand is controlled by 7-dimensional arm joint action and 20-dimensional hand action. All four settings therefore require different post-training action adapters while sharing the same pretrained model. 

33 



Figure 11: **Generalization to different target water levels.** For the AC-One _get a drink_ task, the rightmost cup shows the marked target level used throughout post-training data collection. The three cups on the left show the outcomes of three successful executions at different marked target levels. Black markings indicate the target levels. 

**RTC continuity comparison.** The three RTC strategies exhibit distinct continuity behaviors in the recorded AC-One runs (Figure 12). In the illustrated trace, VJP (Black et al., 2025c) shows an abrupt change in the published action commands at the chunk boundary, with a boundary-to-within-plan change ratio of 8.94. Hard prefix (Motubrain Team, 2026) preserves the committed actions at the chunk boundary, but introduces a sharp jetting event at the prefix-to-suffix boundary, where the model switches from the committed prefix to freely generated actions. Its corresponding ratio is 7.13. Training-time conditioning (Black et al., 2025d) keeps changes at the prefix-to-suffix boundary comparable to ordinary within-plan changes, with a ratio of 1.12. These diagnostic traces support our use of training-time prefix conditioning for continuous asynchronous execution. 

## **7.4 Ablation** 

## **_7.4.1 Component-Wise Ablations_** 

We conduct cumulative ablations on LIBERO-Plus to investigate the contribution of the main components in InternW0-∆. All configurations are trained independently from scratch using the same training data, optimization recipe, and evaluation protocol. Therefore, each row in Table 14 corresponds to a separate training run rather than continued training from the preceding configuration. Starting from the baseline, we progressively introduce Sparse Memory Context (SMC), VLM, Causal Imprint (CI), representation alignment, and 4D-aware representation distillation. 

**Sparse Memory Context.** We first investigate whether lightweight temporal context benefits action prediction. Adding SMC improves the success rate from 49.59% to 53.47%, a gain of 3.88 percentage points. SMC augments the current observation with an episode-level anchor and a recent observation preceding the previous action chunk, providing both coarse task context and shortterm execution history. The improvement indicates that a single current observation is insufficient to fully capture the interaction state, while a small number of carefully selected historical observations already provides useful temporal information without maintaining a dense visual history. 

34 



<!-- Start of picture text -->
(a) Action jetting visualization (b) Relative boundary action change<br>0 . 25 10<br>8 . 94<br>0 . 2 8<br>7 . 13<br>0 . 15 6<br>0 . 1 4<br>0 . 05 2<br>1 . 12<br>0 0<br>0 5 11 16 20 24 VJP Hard prefix Training-time<br>Control steps from inference request RTC<br>Hard prefix VJP Training-time RTC<br>ℓ commandchange2<br>/Boundarywithin-planchange<br><!-- End of picture text -->

Figure 12: **Action jetting visualization.** (a) _ℓ_ 2 changes between consecutive published 14dimensional action commands, aligned to the inference request at _t_ = 0, with _t_ measured in published control steps. Hard prefix and training-time RTC are shown as mean curves with chunk switches at _t_ = 10; VJP is shown as an individual trace with its chunk switch at _t_ = 16. The gray dashed line at _t_ = 0 also marks the previous hard-prefix plan’s prefix-to-suffix boundary. The blue dashed line at _t_ = 11 marks the training-time prefix-to-suffix boundary. The red dashed line at _t_ = 16 marks both the hard-prefix prefix-to-suffix boundary and the VJP chunk boundary. (b) Mean boundary command change divided by the mean consecutive-command change within the executed new-plan segments, excluding the evaluated boundary. The evaluated boundary is the chunk boundary for VJP and the prefix-to-suffix boundary for hard prefix and training-time RTC. The dashed horizontal line denotes a ratio of one. 

Table 14: Cumulative ablation study on LIBERO-Plus (Fei et al., 2026). All variants are independently trained from scratch under the same training and evaluation settings. Starting from the baseline, we progressively introduce Sparse Memory Context (SMC), vision-language conditioning, Causal Imprint (CI), the additional CI alignment objective Lalign, and 4D-aware representation distillation L4D. Success rate denotes the overall success rate (%). 

|Method|Success rate (%)_↑_|
|---|---|
|Baseline|49.59|
|Baseline + SMC|53.47|
|Baseline + SMC + Qwen3.5-2B|60.37|
|Baseline + SMC + RynnBrain1.1-2B|69.08|
|Baseline + SMC + RynnBrain1.1-2B + CI|70.80|
|Baseline + SMC + RynnBrain1.1-2B + CI +Lalign|76.45|
|Baseline + SMC + RynnBrain1.1-2B + CI +Lalign +L4D|**78.37**|



**VLM.** We next investigate the effect of the vision-language representation supplied to the action expert. With SMC fixed, introducing Qwen3.5-2B improves the success rate from 53.47% to 60.37%, showing the benefit of jointly reasoning over the current visual scene and task instruction. Replacing Qwen3.5-2B with RynnBrain1.1-2B further increases the success rate to 69.08%, corresponding to an additional gain of 8.71 percentage points. Since all variants use the same training and evaluation settings, this comparison shows that the representation provided by the VLM has a strong effect on downstream manipulation performance. We therefore adopt RynnBrain1.1-2B in the remaining experiments. 

35 

Table 15: Ablation of distillation design on LIBERO-Plus (Fei et al., 2026). All variants include SMC, RynnBrain1.1-2B (Dang et al., 2026), CI, and Lalign. Action injection indicates whether the student descriptor is additionally fed into the action expert. 

|Teacher|Action injection|Success rate (%)_↑_|
|---|---|---|
|None|No|76.45|
|CoWTracker|No|76.15|
|Pi3X|No|73.63|
|Track4World|Yes|77.78|
|Track4World|No|**78.37**|



**Causal Imprint.** We then investigate Causal Imprint, which is designed to encode information about how the observed scene is likely to evolve while using only observations available to the policy at inference time. CI is trained with its latent-difference objective L∆, which directly supervises the imprint representation using changes between consecutive future video latents. Introducing CI improves the success rate from 69.08% to 70.80%. We further introduce the representation alignment objective Lalign, which aligns Causal Imprint tokens with spatially corresponding representations from the future video features. This increases the success rate substantially from 70.80% to 76.45%, a gain of 5.65 percentage points. The two objectives provide complementary signals: L∆ directly describes the temporal changes along the future trajectory, whereas Lalign transfers the richer semantic and temporal representation learned by the video expert. Together, they encourage Causal Imprint to capture not only low-level visual changes but also higher-level information about the subsequent evolution of the scene. Importantly, the future observations are used only to construct supervision during training. The Causal Imprint tokens themselves cannot directly attend to the realized future, and the action expert therefore receives no privileged future observation during either training or inference. This allows future trajectories to shape the learned representation without requiring explicit future-video generation for online action prediction. 

**4D-aware representation distillation.** As shown in Table 14, adding L4D improves the overall success rate on LIBERO-Plus from 76.45% to 78.37%, a gain of 1.92 percentage points. We further investigate the teacher choice and action feature injection in Table 15. Distillation from Track4World (Lu et al., 2026) achieves the highest success rate of 78.37%, compared with 76.15% for CoWTracker (Lai et al., 2026) and 73.63% for Pi3X (Wang et al., 2025). Additionally, injecting the student descriptor into the action expert yields a slightly lower success rate of 77.78%. We therefore adopt auxiliary distillation of the video expert without action injection, allowing the student branch to be removed at inference with no additional policy inference cost. 

## **_7.4.2 Data Source Ablations_** 

We investigate how different data sources contribute to model performance. We consider four types of data, including raw robot demonstrations, raw human egocentric videos (Ego), human demonstrations converted through our Ego2Robot pipeline (Ego2Robot), and UMI data. All variants use the same InternW0-∆ architecture and are pretrained using data from different sources. The subsequent training and evaluation pipelines are kept identical across all settings, allowing us to study the impact of data composition under a controlled setup. 

Across the evaluated settings, as shown in Table 16 and Table 17, robot demonstration data provides the strongest and most consistent benefit. On LIBERO-Plus, robot pretraining improves the overall success rate from 78.59% to 83.73%. The effect is particularly clear on RoboTwin 2.0 under the Clean2Random setting, where the success rate increases from 4.38% to 32.34%, indicating a substantial gain in robustness to visual and environmental variations. In comparison, raw egocentric video brings only a modest improvement on the current benchmarks, increasing LIBERO-Plus performance to 80.45% and RoboTwin Clean2Random from 2.80% to 3.13%. Ego2Robot data yields a larger gain, reaching 81.86% on LIBERO-Plus and 6.66% on RoboTwin Clean2Random. UMI 

36 

data provides an even stronger improvement, achieving 83.24% on LIBERO-Plus and increasing RoboTwin Clean2Random from 2.80% to 13.90%. Overall, these results suggest that human data become more effective when their observation and action spaces are better aligned with those of robots. 

The results are specific to our current data scale and evaluation setting, with both LIBERO-Plus and RoboTwin primarily focusing on gripper-based manipulation. Under this setting, raw egocentric video provides limited gains in task success rate, but this does not rule out stronger benefits at a larger scale or on tasks that are more closely aligned with human hand-object interaction. In particular, human demonstrations may be more useful for dexterous manipulation, where fine-grained contact patterns and hand motions are less well covered by standard robot datasets. Moreover, our current evaluation only measures downstream success rate. Therefore, the limited improvement from Ego data should not be interpreted as evidence that it does not improve the learned representation, since such effects may not be fully reflected by success rate on the current benchmarks. 

Table 16: Pretraining data-source ablation on LIBERO-Plus (Fei et al., 2026). The baseline uses no pretraining, whereas Robot, Ego, Ego2Robot, and UMI are pretrained exclusively on their respective data sources. All values are success rates (%). 

|Method|Camera|Robot|Language|Light|Background|Noise|Layout|Overall_↑_|
|---|---|---|---|---|---|---|---|---|
|Baseline|60.10|76.77|88.68|95.01|81.23|75.58|78.69|78.59|
|Robot|**74.36**|68.39|93.69|96.67|81.97|**93.07**|80.85|**83.73**|
|Ego|65.60|74.26|91.93|94.83|**84.20**|79.01|78.82|80.45|
|Ego2Robot|63.10|79.81|93.69|96.76|82.62|82.70|79.15|81.86|
|UMI|65.92|**80.26**|**96.49**|**97.20**|79.83|84.82|**81.38**|83.24|



Table 17: Pretraining data-source ablation on RoboTwin 2.0 (Chen et al., 2026c). Baseline-Joint and Baseline-EEF use no pretraining, whereas Robot-Joint, Ego2Robot-Joint, Ego-EEF and UMI-EEF are pretrained exclusively on their respective data sources. All values are success rates (%). 

|Method|Clean2Clean_↑_|Clean2Random_↑_|Overall_↑_|
|---|---|---|---|
|Baseline-Joint|78.20|4.38|41.29|
|Robot-Joint|**81.36**|**32.34**|**56.85**|
|Ego2Robot-Joint|76.92|6.66|41.79|
|Baseline-EEF|61.58|2.80|32.08|
|Ego-EEF|61.49|3.13|32.31|
|UMI-EEF|**66.05**|**13.90**|**39.98**|



## **7.5 Exploration of GPT-Guided Policy** 

Recent work on GPT as an embodied policy has explored using GPT-6 Astra (Zhang et al., 2026c) together with _π_ 0.5, where GPT monitors the execution of a VLA policy and provides corrective actions when necessary (Su et al., 2026). Their experiments on RoboDojo show that such test-time correction can improve the execution of a pretrained VLA policy (Su et al., 2026). Motivated by these results, we conduct a small-scale exploration to examine whether the same mechanism can also complement a World Action Model. 

We select five RoboDojo tasks on which the standalone InternW0-∆ exhibits relatively low performance under seed 0, covering both semantic reasoning and manipulation challenges. As shown in Table 18, InternW0-∆ obtains an average success rate of 10.80% and an average score of 13.92 across these tasks. We then introduce GPT-6 Astra as an execution-time assistant that observes the interaction and provides end-effector (EEF) corrections when necessary. This increases the average success rate to 47.20% and the average score to 52.00, corresponding to absolute gains of 

37 

Table 18: Comparison of Ours and Ours+Astra on five RoboDojo (Chen et al., 2026b) tasks. 

|Method|classif|y objects|press b|y number|solve|equation|genera|l pickup|arrange larg|est number|Av|erage|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
||SR_↑_|Score_↑_|SR_↑_|Score_↑_|SR_↑_|Score_↑_|SR_↑_|Score_↑_|SR_↑_|Score_↑_|SR_↑_|Score_↑_|
|Ours|14.00|22.80|4.00|4.00|0.00|0.00|36.00|36.00|0.00|6.80|10.80|13.92|
|**Ours+Astra**|**80.00**|**88.00**|**20.00**|**20.00**|**20.00**|**20.00**|**76.00**|**76.00**|**40.00**|**56.00**|**47.20**|**52.00**|



36.40 percentage points in success rate and 38.08 score points, respectively. The improvement is particularly clear on “classify objects”, where the success rate increases from 14% to 80%, and on “general pickup”, from 36% to 76%. Tasks with zero standalone success also become partially solvable: “solve equation” and “arrange largest number” reach 20% and 40% success rates, respectively. These results suggest that GPT-based execution correction can provide complementary high-level reasoning and online adjustment even when the underlying policy is a world-action model rather than a conventional VLA. 

We further observe that _the reasoning configuration of GPT-6 Astra can have a noticeable effect on correction quality, especially for tasks that require semantic understanding._ For example, on “classify objects by language”, using GPT-6 Astra with the xhigh reasoning setting together with InternW0-∆ successfully completes all five evaluated episodes. In contrast, the medium setting obtains only partial task scores and does not complete the full task in these trials. Although this comparison is based on a small number of episodes, it indicates that stronger test-time reasoning may be particularly useful when policy correction requires interpreting language instructions, identifying task-relevant objects, and deciding how the underlying policy should be adjusted. 

# **8 Conclusion** 

**Summary.** We have presented InternW0-∆, a world-action model that learns action-relevant dynamic representations by combining pretrained visual dynamics, task-conditioned scene semantics, and robot action generation within a directed Mixture-of-Transformers architecture. A pretrained video expert is coupled with an action expert grounded in scene semantics from a frozen VLM, while sparse visual memory provides both episode-level context and recent interaction history. Causal Imprint uses future supervision to capture changes relevant to subsequent interaction, and training-only 4D-aware distillation further introduces geometric and motion priors into the video representation. Together, these designs allow predictive information learned during training to directly support action generation without requiring future-video rollout at inference. Beyond the model, we have developed a scalable data, training, and deployment recipe built on over 20K hours of processed robot, UMI, egocentric, and Ego2Robot data. Canonical state and action representations, systematic quality filtering, and temporal alignment enable heterogeneous pretraining followed by target-embodiment post-training. Across multiple simulation benchmarks and real-robot platforms, the same pretrained checkpoint can be adapted to different task distributions, embodiments, and control interfaces, including both gripper-based and dexterous-hand manipulation. 

We have further developed training and inference infrastructure for efficient model iteration and online execution, and will release the associated code, checkpoints, training recipes, dataprocessing tools, and evaluation and deployment utilities. 

**Limitations.** The current work does not exhaustively explore all components of the proposed training and deployment framework. First, although egocentric human demonstrations constitute an important part of our pretraining mixture, our study of their contribution remains limited. We have not yet systematically examined how different egocentric data sources, conversion strategies, scaling ratios, and supervision forms affect downstream robot performance. Second, our investigation of agent-assisted control is still preliminary. The current experiments only consider a limited corrective setting, and do not systematically study agent invocation policies, long-horizon plan- 

38 

ning, hierarchical decision making, or tighter integration between external agents and the worldaction model. We leave a broader study of egocentric data utilization and agent-enhanced control to future work. 

# **9 Team** 

## **Core Contributors** 

**Real-Robot Data:** Zizun Li<sup>*</sup> , Xingyu Miao<sup>*</sup> , Xueyuan Wei **Ego & UMI Data:** Kaiwen Song<sup>*</sup> , Tenghui Wang<sup>*</sup> , Yuping He 

**Model, Pretraining & Infrastructure:** Xingyu Miao<sup>*</sup> , Zizun Li<sup>*</sup> , Hanxue Zhang, Yating Wang **Post-Training & Real-Robot Deployment:** Baole Fang<sup>*</sup> , Xudong Li<sup>*</sup> , Xijie Yang<sup>*</sup> 

**Technical Lead (listed alphabetically):** Junting Dong<sup>†</sup> , Haoyu Guo<sup>†</sup> , Tao Lu<sup>†</sup> , Mulin Yu<sup>†</sup> 

**Project Lead:** Bowen Zhou, Bin Zhao, Tianfan Xue, Weinan Zhang<sup>†</sup> , Chunhua Shen<sup>†</sup> 

## **Contributors** 

Xueyue Zhu, Chao Gao, Zeyu He, Yuanzhen Zhou, Jianyang Zhang, Siwei Cui, Xing Gao, Yifei Yao, Qiaojun Yu, Kailin Li, Ming Zhou, Xinzhe Wang, Yingxiang Xu, Mu Huang, Zetao Cai, Fuxian Huang, Yunsong Zhou, Yufei Xue, Wenqi Guo, Jianjun Zhou, Xinyue Li, Kerui Ren, Weiguang Zhao, Ni Yang, Wenze Cui, Bingqi Jiang, Rong Fu, Hengjie Li 

> *Equal contribution. 

> †Corresponding authors. 

39 

# **References** 

- Arthur Allshire, Himanshu Gaurav Singh, Ritvik Singh, Adam Rashid, Hongsuk Choi, David McAllister, Justin Yu, Yiyuan Chen, Huang Huang, Pieter Abbeel, Xi Chen, Rocky Duan, Phillip Isola, Jitendra Malik, Fred Shentu, Guanya Shi, Philipp Wu, and Angjoo Kanazawa. Scalable behavior cloning with open data, training, and evaluation, 2026. URL https://arxiv.org/abs/ 2606.27375. 

- Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba Komeili, Matthew J. Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, and Nicolas Ballas. V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning. _arXiv preprint arXiv:2506.09985_ , 2025. URL https://arxiv.org/abs/2506.09985. 

- Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, and Nicolas Ballas. Revisiting Feature Prediction for Learning Visual Representations from Video. _arXiv preprint arXiv:2404.08471_ , 2024. URL https://arxiv.org/abs/2404.08471. 

- Hongzhe Bi, Hengkai Tan, Shenghao Xie, Zeyuan Wang, Shuhe Huang, Haitian Liu, Ruowen Zhao, Yao Feng, Chendong Xiang, Yinze Rong, Hongyan Zhao, Hanyu Liu, Zhizhong Su, Lei Ma, Hang Su, and Jun Zhu. Motus: A Unified Latent Action World Model. _arXiv preprint arXiv:2512.13030_ , 2025. URL https://arxiv.org/abs/2512.13030. 

- Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Robert Equi, Chelsea Finn, Niccolo Fusai, Manuel Y. Galliker, Dibya Ghosh, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z. Ren, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, James Tanner, Quan Vuong, Homer Walke, Anna Walling, Haohuan Wang, Lili Yu, and Ury Zhilinsky. _π_ 0.5: a Vision-Language-Action Model with Open-World Generalization. In _Proceedings of The 9th Conference on Robot Learning_ , volume 305 of _Proceedings of Machine Learning Research_ , pages 17–40. PMLR, 2025a. URL https://proceedings.mlr.press/v305/black25a.html. 

- Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Robert Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Lucy Xiaoyang Shi, Laura Smith, James Tanner, Quan Vuong, Anna Walling, Haohuan Wang, and Ury Zhilinsky. _π_ 0: A Vision-Language-Action Flow Model for General Robot Control. In _Proceedings of Robotics: Science and Systems_ , 2025b. doi: 10.15607/RSS.2025.XXI.010. URL https://www.roboticsproceedings.org/rss21/p010.html. 

- Kevin Black, Manuel Y. Galliker, and Sergey Levine. Real-time execution of action chunking flow policies, 2025c. URL https://arxiv.org/abs/2506.07339. 

- Kevin Black, Allen Z. Ren, Michael Equi, and Sergey Levine. Training-time action conditioning for efficient real-time chunking, 2025d. URL https://arxiv.org/abs/2512.05964. 

- Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence, Chuyuan Fu, Montse Gonzalez Arenas, Keerthana Gopalakrishnan, Kehang Han, Karol Hausman, Alexander Herzog, Jasmine Hsu, Brian Ichter, Alex Irpan, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Lisa Lee, Tsang-Wei Edward Lee, Sergey Levine, Yao Lu, Henryk Michalewski, Igor Mordatch, Karl Pertsch, Kanishka Rao, Krista Reymann, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Pierre Sermanet, Jaspiar Singh, Anikait Singh, Radu Soricut, Huong Tran, Vincent Vanhoucke, Quan Vuong, Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Jialin Wu, 

40 

Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and Brianna Zitkovich. RT-2: Vision-languageaction models transfer web knowledge to robotic control. _arXiv preprint arXiv:2307.15818_ , 2023. URL https://arxiv.org/abs/2307.15818. 

- Remi Cadene, Simon Alibert, Francesco Capuano, Michel Aractingi, Adil Zouitine, Pepijn Kooijmans, Jade Choghari, Martino Russi, Caroline Pascal, Steven Palma, Mustafa Shukor, Jess Moss, Alexander Soare, Dana Aubakirova, Quentin Lhoest, Quentin Gallouédec, and Thomas Wolf. LeRobot: An Open-Source Library for End-to-End Robot Learning. In _The Fourteenth International Conference on Learning Representations_ , 2026. URL https://arxiv.org/abs/2602.22818. 

- Jisong Cai, Long Ling, Shiwei Chu, Zhongshan Liu, Jiayue Kang, Zhixuan Liang, Wenjie Xu, Yinan Mao, Weinan Zhang, Xiaokang Yang, Ru Ying, Ran Zheng, and Yao Mu. AHA-WAM: Asynchronous Horizon-Adaptive World-Action Modeling with Observation-Guided Context Routing. _arXiv preprint arXiv:2606.09811_ , 2026a. URL https://arxiv.org/abs/2606.09811. 

- Junhao Cai, Zetao Cai, Jiafei Cao, Yilun Chen, Zeyu He, Lei Jiang, Hang Li, Hengjie Li, Yang Li, Yufei Liu, et al. InternVLA-A1: Unifying understanding, generation and action for robotic manipulation. _arXiv preprint arXiv:2601.02456_ , 2026b. URL https://arxiv.org/abs/2601.02456. 

- Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, Jie Lei, Tengyu Ma, Baishan Guo, Arpit Kalla, Markus Marks, Joseph Greer, Meng Wang, Peize Sun, Roman Rädle, Triantafyllos Afouras, Effrosyni Mavroudi, Katherine Xu, Tsung-Han Wu, Yu Zhou, Liliane Momeni, Rishi Hazra, Shuangrui Ding, Sagar Vaze, Francois Porcher, Feng Li, Siyuan Li, Aishwarya Kamath, Ho Kei Cheng, Piotr Dollár, Nikhila Ravi, Kate Saenko, Pengchuan Zhang, and Christoph Feichtenhofer. SAM 3: Segment Anything with Concepts, 2025. URL https://arxiv.org/abs/2511.16719. Version Number: 2. 

- Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In _Proceedings of the International Conference on Computer Vision (ICCV)_ , 2021. 

- Ronghan Chen, Yandan Yang, Zuojin Tang, Dongjie Huo, Tong Lin, Haoning Wu, Haoyun Liu, Yuzhi Chen, Lulu Zheng, Botai Yuan, Tianlun Li, Mingxin Wang, Dekang Qi, Bin Hu, Wei Mei, Yuze Xuan, Haolong Yang, Yanqing Zhu, Mu Xu, Zhiheng Ma, and Xinyuan Chang. ABot-M0.5: Unified Mobility-and-Manipulation World Action Model. _arXiv preprint arXiv:2607.00678_ , 2026a. URL https://arxiv.org/abs/2607.00678. 

- Tianxing Chen, Yue Chen, Zixuan Li, Junyuan Tang, Kailun Su, Haoran Lu, Weijie Wan, Baijun Chen, Songling Liu, Haowen Yan, Honghao Su, Zhiyang Dou, Kaixuan Wang, Dandan Zhang, Yunze Liu, Yan Qin, Qiwei Liang, Qiwei Wu, Zijian Lin, Wenwei Lin, Yuran Wang, Minghua He, Tianshu Wu, Ruihai Wu, Jingquan Zhou, Kai-Chong Lei, Haibao Yu, Yuanfeng Ji, Weiyang Jin, Guanyu Lin, Xiaofan Li, Qi Xiong, Renjing Xu, Zhongyu Li, Wenhao Chai, Enze Xie, Ziwei Wang, Yao Mu, Hao Dong, Wojciech Matusik, Mingyu Ding, Wenbo Ding, Ping Luo, and Masayoshi Tomizuka. RoboDojo: A Unified Sim-and-Real Benchmark for Comprehensive Evaluation of Generalist Robot Manipulation Policies. _arXiv preprint arXiv:2607.04434_ , 2026b. URL https: //arxiv.org/abs/2607.04434. 

- Tianxing Chen, Zanxin Chen, Baijun Chen, Zijian Cai, Yibin Liu, Zixuan Li, Qiwei Liang, Xianliang Lin, Yiheng Ge, Zhenyu Gu, Weiliang Deng, Yubin Guo, Tian Nian, Xuanbing Xie, Qiangyu Chen, Kailun Su, Tianling Xu, Guodong Liu, Mengkang Hu, Huan ang Gao, Kaixuan Wang, Zhixuan Liang, Yusen Qin, Xiaokang Yang, Ping Luo, and Yao Mu. RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation. In _Proceedings of the International Conference on Machine Learning_ , 2026c. URL https://arxiv.org/abs/2506.18088. 

- Xinyuan Chen, Haoyu Guo, Shi Guo, Bingqi Jiang, Chunhua Shen, Xing Shen, Tianfan Xue, Yufei Xue, Mulin Yu, Weinan Zhang, Bin Zhao, Bowen Zhou, and Ming Zhou. A definition and roadmap for world models, 2026d. URL https://arxiv.org/abs/2607.06401. 

41 

- AgiBot World Colosseum contributors. OpenDriveLab/AgiBot-World, 2024. 

Agibot world colosseum. https://github.com/ 

- InternData-A1 contributors. Interndata-a1. https://github.com/InternRobotics/InternManip, 2025. 

- LiteGen contributors. Litegen. https://github.com/DeepLink-org/LiteGen, 2026. 

- Ronghao Dang, Jiayan Guo, Bohan Hou, Sicong Leng, Kehan Li, Xin Li, Jiangpin Liu, Yunxuan Mao, Zhikai Wang, Yuqian Yuan, Minghao Zhu, Xiao Lin, Yang Bai, Qian Jiang, Yaxi Zhao, Minghua Zeng, Junlong Gao, Yuming Jiang, Jun Cen, Siteng Huang, Liuyi Wang, Wenqiao Zhang, Chengju Liu, Jianfei Yang, Shijian Lu, and Deli Zhao. Rynnbrain: Open embodied foundation models, 2026. URL https://arxiv.org/abs/2602.14979. 

- Dexmal. DM0.5: From the Lab to the Open World. Technical blog and model release, 2026. URL https://www.dexmal.com/blog/dm0.5?lang=en-US. Released 2026-07-09. Accessed: 2026-09-18. 

- Parsa Esmati, Somjit Nath, Katja Hofmann, Derek Nowrouzezahrai, Samira Ebrahimi Kahou, and Majid Mirmehdi. The invisible hand of physics: When video diffusion models know more than they show, 2026. 

- Hao-Shu Fang, Hongjie Fang, Zhenyu Tang, Jirong Liu, Chenxi Wang, Junbo Wang, Haoyi Zhu, and Cewu Lu. Rh20t: A comprehensive robotic dataset for learning diverse skills in one-shot. In _2024 IEEE International Conference on Robotics and Automation (ICRA)_ , pages 653–660. IEEE, 2024. 

- Haoquan Fang, Jiafei Duan, Donovan Clay, Sam Wang, Shuo Liu, Weikai Huang, Xiang Fan, WeiChuan Tsai, Shirui Chen, Yi Ru Wang, Shanli Xing, Jaemin Cho, Jae Sung Park, Ainaz Eftekhar, Peter Sushko, Karen Farley, Angad Wadhwa, Cole Harrison, Winson Han, Ying-Chun Lee, Eli VanderBilt, Rose Hendrix, Suveen Ellawela, Lucas Ngoo, Joyce Chai, Zhongzheng Ren, Ali Farhadi, Dieter Fox, and Ranjay Krishna. Molmoact2: Action reasoning models for real-world deployment, 2026. URL https://arxiv.org/abs/2605.02881. 

- Senyu Fei, Siyin Wang, Junhao Shi, Zihao Dai, Jikun Cai, Pengfang Qian, Li Ji, Xinzhe He, Shiduo Zhang, Zhaoye Fei, Jinlan Fu, Jingjing Gong, and Xipeng Qiu. LIBERO-Plus: A Progressive Robustness Benchmark for Visual-Language-Action Models. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , 2026. URL https://openaccess.thecvf.com/content/CVPR2026/html/Fei_LIBERO-Plus_A_Progressive_ Robustness_Benchmark_for_Visual-Language-Action_Models_CVPR_2026_paper.html. 

- Yao Mu Fourier ActionNet Team. Actionnet: A dataset for dexterous bimanual manipulation. 2025. 

- Ning Gao, Jinliang Zheng, Xing Gao, Haoxiang Ma, Hanqing Wang, Yukai Wang, Jiantong Chen, Zanxin Chen, Shujie Zhang, Mingda Jia, Xuekun Jiang, Zihou Zhu, Xinyu Li, Shuai Wang, Hao Li, Wenzhe Cai, Yuqiang Yang, Xudong Xu, Zhaoyang Lyu, Yao Mu, Tai Wang, Jiangmiao Pang, Jia Zeng, Weinan Zhang, and Chunhua Shen. EBench: Elemental Diagnosis of Generalist Mobile Manipulation Policies. _arXiv preprint arXiv:2606.18239_ , 2026. URL https://arxiv.org/abs/2606. 18239. 

- GigaBrain Team, Angen Ye, Axiang Sun, Can Jin, Chenxi Cheng, Chong Shi, Dengke Shang, Dingqian Zhang, Guan Huang, Guangqiang Wang, Guangqing Ding, Guo Li, Hangcong Li, Hengyu Zhong, Hongtao Lu, Jianbo Qin, Jiming Mao, Jing Zhu, Jindi Lv, Jingzhi Cui, Junjie Xie, Junyi Bao, Kai Liu, Lei Yuan, Limin Long, Lv Feng, Mingming Yu, Peng Li, Pengfei Yi, Qi Li, Qianli Zhang, Qingfang Li, Qitang Hu, Rui Zhang, Shaoyan Sun, Shibo Sun, Shiying Duan, Tenghui Chen, Tianze Liu, Weijie Ke, Wenyao Xue, Xiaofeng Wang, Xiaoyu Tian, Xinyu Liu, Xinze Chen, Yang Wang, Yankai Wang, Yejun Zeng, Yifan Li, Yifei Nie, Yilong Li, Yilong Liu, Yongchao Feng, Yumeng Wang, Yun Ye, Zhichao Liu, Ziheng He, Zonghai Yang, and Zheng Zhu. GigaBrain-0.7: Scaling Embodied Foundation Models to Emergent Capabilities with a Three-System Architecture. _arXiv preprint arXiv:2608.15875_ , 2026. URL https: //arxiv.org/abs/2608.15875. 

42 

- Jun Guo, Qiwei Li, Peiyan Li, Zilong Chen, Nan Sun, Yifei Su, Heyun Wang, Yuan Zhang, Xinghang Li, and Huaping Liu. Unified 4D World Action Modeling from Video Priors with Asynchronous Denoising. _arXiv preprint arXiv:2604.26694_ , 2026. URL https://arxiv.org/abs/2604.26694. 

- Ryan Hoque, Peide Huang, David J. Yoon, Mouli Sivapurapu, and Jian Zhang. EgoDex: Learning dexterous manipulation from large-scale egocentric video, 2025. URL https://arxiv.org/abs/ 2505.11709. 

- Chengkai Hou, Kun Wu, Jiaming Liu, Zhengping Che, Di Wu, Fei Liao, Guangrun Li, Jingyang He, Qiuxuan Feng, Zhao Jin, Chenyang Gu, Zhuoyang Liu, Nuowei Han, Xiangju Mi, Yaoxu Lv, Yankai Fu, Gaole Dai, Langzhe Gu, Tao Li, Yuheng Zhang, Yixue Zhang, Xinhua Wang, Shichao Fan, Meng Li, Zhen Zhao, Ning Liu, Zhiyuan Xu, Pei Ren, Junjie Ji, Haonan Liu, Kuan Cheng, Shanghang Zhang, and Jian Tang. Robomind 2.0: A multimodal, bimanual mobile manipulation dataset for generalizable embodied intelligence, 2025. URL https://arxiv.org/abs/2512.24653. 

- Yucheng Hu, Yanjiang Guo, Pengchao Wang, Xiaoyu Chen, Yen-Jen Wang, Jianke Zhang, Koushil Sreenath, Chaochao Lu, and Jianyu Chen. Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations. In _Proceedings of the 42nd International Conference on Machine Learning_ , volume 267 of _Proceedings of Machine Learning Research_ , pages 24328–24346. PMLR, 2025. URL https://proceedings.mlr.press/v267/hu25g.html. 

- Wenlong Huang, Fei Xia, Ted Xiao, Harris Chan, Jacky Liang, Pete Florence, Andy Zeng, Jonathan Tompson, Igor Mordatch, Yevgen Chebotar, Pierre Sermanet, Tomas Jackson, Noah Brown, Linda Luu, Sergey Levine, Karol Hausman, and Brian Ichter. Inner monologue: Embodied reasoning through planning with language models. In _Proceedings of the 6th Conference on Robot Learning_ , volume 205 of _Proceedings of Machine Learning Research_ , pages 1769–1782. PMLR, 2023. URL https://proceedings.mlr.press/v205/huang23c.html. 

- Brian Ichter, Anthony Brohan, Yevgen Chebotar, Chelsea Finn, Karol Hausman, Alexander Herzog, Daniel Ho, Julian Ibarz, Alex Irpan, Eric Jang, Ryan Julian, Dmitry Kalashnikov, Sergey Levine, Yao Lu, Carolina Parada, Kanishka Rao, Pierre Sermanet, Alexander T. Toshev, Vincent Vanhoucke, Fei Xia, Ted Xiao, Peng Xu, Mengyuan Yan, Noah Brown, Michael Ahn, Omar Cortes, Nicolas Sievers, Clayton Tan, Sichun Xu, Diego Reyes, Jarek Rettinghouse, Jornell Quiambao, Peter Pastor, Linda Luu, Kuang-Huei Lee, Yuheng Kuang, Sally Jesmonth, Nikhil J. Joshi, Kyle Jeffrey, Rosario Jauregui Ruano, Jasmine Hsu, Keerthana Gopalakrishnan, Byron David, Andy Zeng, and Chuyuan Kelly Fu. Do as i can, not as i say: Grounding language in robotic affordances. In _Proceedings of the 6th Conference on Robot Learning_ , volume 205 of _Proceedings of Machine Learning Research_ , pages 287–318. PMLR, 2023. URL https://proceedings.mlr.press/ v205/ichter23a.html. 

- Boden Intelligence, Junpu Innovation Center, and Shanghai Jiao Tong University MINT Lab. Rw-rl dataset: Real-world reinforcement learning dataset. https://huggingface.co/datasets/MINTSJTU/RW-RL-Dataset, 2026. 

- Yunfan Jiang, Agrim Gupta, Zichen Zhang, Guanzhi Wang, Yongqiang Dou, Yanjun Chen, Li FeiFei, Anima Anandkumar, Yuke Zhu, and Linxi Fan. VIMA: Robot manipulation with multimodal prompts. In _Proceedings of the 40th International Conference on Machine Learning_ , volume 202 of _Proceedings of Machine Learning Research_ , pages 14975–15022. PMLR, 2023. URL https://proceedings.mlr.press/v202/jiang23b.html. 

- Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. OpenVLA: An Open-Source Vision-Language-Action Model. _arXiv preprint arXiv:2406.09246_ , 2024. URL https://arxiv.org/abs/2406.09246. 

- Moo Jin Kim, Chelsea Finn, and Percy Liang. Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success. In _Proceedings of Robotics: Science and Systems_ , 2025. doi: 10.15607/RSS.2025.XXI.017. URL https://www.roboticsproceedings.org/rss21/p017.html. 

43 

- Moo Jin Kim, Yihuai Gao, Tsung-Yi Lin, Yen-Chen Lin, Yunhao Ge, Grace Lam, Percy Liang, Shuran Song, Ming-Yu Liu, Chelsea Finn, and Jinwei Gu. Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning. _arXiv preprint arXiv:2601.16163_ , 2026. URL https://arxiv. org/abs/2601.16163. 

- Michael King, Aravindh Mahendran, Matthew Koichi Grimes, Fedor Kitashov, Adham Elarabawy, Pedro Velez, Maks Ovsjanikov, and Viorica P˘atr˘aucean. Gen4u: Unifying video generation and understanding via diffusion, 2026. 

- Vikash Kumar, Rutav Shah, Gaoyue Zhou, Vincent Moens, Vittorio Caggiano, Jay Vakil, Abhishek Gupta, and Aravind Rajeswaran. Robohive – a unified framework for robot learning. In _NeurIPS: Conference on Neural Information Processing Systems_ , 2023. URL https://sites.google.com/view/ robohive. 

- Zihang Lai, Eldar Insafutdinov, Edgar Sucar, and Andrea Vedaldi. CoWTracker: Tracking by warping instead of correlation. _arXiv preprint arXiv:2602.04877_ , 2026. 

- Fuhao Li, Wenxuan Song, Han Zhao, Jingbo Wang, Pengxiang Ding, Donglin Wang, Long Zeng, and Haoang Li. Spatial Forcing: Implicit Spatial Representation Alignment for Vision-LanguageAction Model. In _International Conference on Learning Representations_ , 2026a. URL https://arxiv. org/abs/2510.12276. 

- Lin Li, Qihang Zhang, Yiming Luo, Shuai Yang, Ruilin Wang, Fei Han, Mingrui Yu, Zelin Gao, Nan Xue, Xing Zhu, Yujun Shen, and Yinghao Xu. Causal world modeling for robot control. _arXiv preprint arXiv:2601.21998_ , 2026b. 

- Yihan Lin, Jiawei He, Shifeng Bao, Chen Zhao, Yang Li, Xiaobo Wang, Yan Wang, Cheng Chi, and Jing Zhang. JEPA-WAM: Learning Vision-Language-Action Policies with Joint-Embedding World Modeling. _arXiv preprint arXiv:2608.09381_ , 2026. URL https://arxiv.org/abs/2608. 09381. 

- Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning. In _Advances in Neural Information Processing Systems_ , volume 36, 2023. URL https://proceedings.neurips.cc/paper_files/ paper/2023/hash/8c3c666820ea055a77726d66fc7d447f-Abstract-Datasets_and_Benchmarks. html. 

- Songming Liu, Lingxuan Wu, Bangguo Li, Hengkai Tan, Huayu Chen, Zhengyi Wang, Ke Xu, Hang Su, and Jun Zhu. Rdt-1b: a diffusion foundation model for bimanual manipulation. _arXiv preprint arXiv:2410.07864_ , 2024. 

- Yicheng Liu, Zibin Dong, Baijun Ye, Tianyuan Yuan, Tao Jiang, Anqi Yang, Shicheng Cao, Haonan Liu, Yue Sun, Zihan Guo, Xiao Liu, Dong Ke, Changxun Pan, Chenru Wu, Tailai Cheng, Xiaoshu Ren, Xinlei Zhang, Jianning Cui, Zijie Zhao, Haoyu Zhang, Kaiming Xu, Haodong Yang, Bowen Zhang, Jiahui Niu, Shaoting Zhu, Shiduo Zhang, and Hang Zhao. G0.5: One Autoregressive Stream for Robot Reasoning and Action. _arXiv preprint arXiv:2608.11739_ , 2026. URL https: //arxiv.org/abs/2608.11739. 

- Jiahao Lu, Jiayi Xu, Wenbo Hu, Ruijie Zhu, Chengfeng Zhao, Sai-Kit Yeung, Ying Shan, and Yuan Liu. Track4world: Feedforward world-centric dense 3d tracking of all pixels. _arXiv preprint arXiv:2603.02573_ , 2026. 

- Hao Luo, Wanpeng Zhang, Yicheng Feng, Sipeng Zheng, Haiweng Xu, Chaoyi Xu, Ziheng Xi, Yuhui Fu, and Zongqing Lu. Being-H0.7: A Latent World-Action Model from Egocentric Videos. _arXiv preprint arXiv:2605.00078_ , 2026. URL https://arxiv.org/abs/2605.00078. 

- Haoxiang Ma, Junhao Cai, Xiaoxu Xu, Hao Li, Yuyin Yang, Yang Tian, Jiafei Cao, Hongrui Zhu, Zherui Qiu, Zhaxizhuoma, Yuqiang Yang, Jiaqi Peng, Xueyuan Wei, Yangkun Zhu, Jiahao Jiang, Xing Gao, Hanqing Wang, Feng Yuan, Kailin Li, Xueyue Zhu, Tai Wang, Yan Ding, Jiangmiao 

44 

Pang, Jia Zeng, Jingjing Zhang, Bowen Zhou, Yao Mu, Chunhua Shen, and Weinan Zhang. InternVLA-A1.5: Unifying Understanding, Latent Foresight, and Action for Compositional Generalization. _arXiv preprint arXiv:2607.04988_ , 2026a. URL https://arxiv.org/abs/2607.04988. 

- Liheng Ma, Rui Heng Yang, Zhanguang Zhang, Mateo Clemente, Ziwen Hu, Tongtong Cao, and Yingxue Zhang. Faster-WAM: Do World Action Models Need Deep Action Modules? _arXiv preprint arXiv:2608.02365_ , 2026b. URL https://arxiv.org/abs/2608.02365. 

- Motubrain Team. World action models in real time: An empirical study of smooth execution via asynchronous deployment, 2026. URL https://arxiv.org/abs/2608.01880. 

- NVIDIA. NVIDIA Isaac GR00T N1.7-3B. Hugging Face model card, 2026. URL https:// huggingface.co/nvidia/GR00T-N1.7-3B. Model version N1.7. Accessed: 2026-09-18. 

- NVIDIA, Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi "Jim" Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, Joel Jang, Zhenyu Jiang, Jan Kautz, Kaushil Kundalia, Lawrence Lao, Zhiqi Li, Zongyu Lin, Kevin Lin, Guilin Liu, Edith Llontop, Loic Magne, Ajay Mandlekar, Avnish Narayan, Soroush Nasiriany, Scott Reed, You Liang Tan, Guanzhi Wang, Zu Wang, Jing Wang, Qi Wang, Jiannan Xiang, Yuqi Xie, Yinzhen Xu, Zhenjia Xu, Seonghyeon Ye, Zhiding Yu, Ao Zhang, Hao Zhang, Yizhou Zhao, Ruijie Zheng, and Yuke Zhu. GR00T N1: An Open Foundation Model for Generalist Humanoid Robots. _arXiv preprint arXiv:2503.14734_ , 2025. URL https://arxiv.org/abs/2503.14734. 

- Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. FAST: Efficient Action Tokenization for Vision-LanguageAction Models. In _Proceedings of Robotics: Science and Systems_ , 2025. doi: 10.15607/RSS.2025. XXI.012. URL https://www.roboticsproceedings.org/rss21/p012.html. 

- Physical Intelligence. OpenPI: Open-Source Models and Packages for Robotics. GitHub repository, 2025. URL https://github.com/Physical-Intelligence/openpi. 

- Ryan Punamiya, Simar Kareer, Zeyi Liu, et al. EgoVerse: An egocentric human dataset for robot learning from around the world, 2026. URL https://arxiv.org/abs/2604.07607. 

- Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. _Journal of Machine Learning Research_ , 21(140):1–67, 2020. URL https://www.jmlr. org/papers/v21/20-074.html. 

- RealSource. Realsource world: A large-scale real-world dual-arm manipulation dataset. https: //huggingface.co/datasets/RealSourceData/RealSource-World, 2025. 

- Jaehwi Song, Suchae Jeong, Byeongguk Jeon, Sungdong Kim, Minjoon Seo, Hyungmok Son, and Kimin Lee. Habit: Human-aware behavior and interaction training dataset for robot manipulation. _arXiv preprint arXiv:2606.31682_ , 2026. 

- StarVLA Community. StarVLA: A Lego-like Codebase for Vision-Language-Action Model Developing. _arXiv preprint arXiv:2604.05014_ , 2026. URL https://arxiv.org/abs/2604.05014. 

- Jiayi Su, Yixin Zheng, Mi Yan, Li Yi, Zhizheng Zhang, and He Wang. GPT 6 Astra as an embodied policy. Technical report and code, 2026. URL https://github.com/anonymous-report-421/ eval-of-gpt-6-astra-as-policy. 

- Jingwen Sun, Wenyao Zhang, Zekun Qi, Shaojie Ren, Zezhi Liu, Hanxin Zhu, Guangzhong Sun, Xin Jin, and Zhibo Chen. VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model. In _European Conference on Computer Vision_ , 2026. URL https://arxiv.org/abs/ 2602.10098. 

- Shuhan Tan, Kairan Dou, Yue Zhao, and Philipp Krähenbühl. Interactive Post-Training for VisionLanguage-Action Models. _arXiv preprint arXiv:2505.17016_ , 2025. URL https://arxiv.org/abs/ 2505.17016. 

45 

- Galaxea Team. Galaxea g0: Open-world dataset and dual-system vla model. _arXiv preprint arXiv:2509.00576_ , 2025. 

- Xiaomi Robotics Team, Jun Guo, Piaopiao Jin, Jason Li, Peiyan Li, Yingyan Li, Futeng Liu, Wanli Peng, Optimus Qin, Yifei Su, et al. Xiaomi-robotics-1: Scaling vision-language-action models with over 100k hours of real-world trajectories. _arXiv preprint arXiv:2607.15330_ , 2026. 

- Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A physics engine for model-based control. In _2012 IEEE/RSJ International Conference on Intelligent Robots and Systems_ , pages 5026–5033. IEEE, 2012. doi: 10.1109/IROS.2012.6386109. 

- Wan Team. Wan: Open and advanced large-scale video generative models. _arXiv preprint arXiv:2503.20314_ , 2025. URL https://arxiv.org/abs/2503.20314. 

- Chenyi Wang, Xinkai Wang, Bokai Lin, Jialin Tian, Fucheng Zhang, Cewu Lu, and Lixin Yang. Track4Action: Distilling World-Centric 3D Tracker into Vision-Language-Action Policies. _arXiv preprint arXiv:2608.03727_ , 2026a. URL https://arxiv.org/abs/2608.03727. 

- Mingxin Wang, Bin Hu, Bin Qian, Kaitao Jiang, Haoning Wu, Feng Yan, Bowen Jing, Ruiyang Hao, Enyi Wang, Kangning Niu, Yandan Yang, Mu Xu, Yan Wang, Houde Liu, and Tianlun Li. ST-WAM: Semantic-Temporal World Action Model for Robust Manipulation under Visual Distribution Shifts. _arXiv preprint arXiv:2607.28993_ , 2026b. URL https://arxiv.org/abs/2607. 28993. 

- Ye Wang, Pei Lin, Xiong-Hui Chen, Haoqi Yuan, Zhixuan Liang, Yiyang Huang, Anzhe Chen, Zixing Lei, Jie Zhang, Tao Zhang, Haoyang Li, Tong Zhang, Chenxi Xiao, Ziyuan Jiao, and Qin Jin. Ego2Robot: Scalable Robot Data Synthesis from Egocentric Human Data, 2026c. URL http: //arxiv.org/abs/2608.02580. arXiv:2608.02580 [cs.RO]. 

- Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chunhua Shen, and Tong He. _π_<sup>3</sup> : Permutation-equivariant visual geometry learning. _arXiv preprint arXiv:2507.13347_ , 2025. 

- Yuran Wang, Siqiao Huang, Mingleyang Li, Chenhao Zhang, Jiaqi Liang, Weiyang Jin, Yue Chen, Xuemin Chi, Donghao Zhou, Qize Yu, Yu-Kai Wang, Yuhan Rui, Shenzhe Yao, Zhen Yuan, Zhenhao Shen, Kefei Zhu, Zijie Zhu, Ning Gao, Xiaowei Chi, Guanqi He, Shanghang Zhang, Hao Dong, Lin Shao, and Hang Zhao. OpenWAM: An Open, Modular Exploration Towards Systematic World-Action Model Pretraining. _arXiv preprint arXiv:2609.07398_ , 2026d. URL https://arxiv.org/abs/2609.07398. 

- Kun Wu, Chengkai Hou, Jiaming Liu, Zhengping Che, Xiaozhu Ju, Zhuqin Yang, Meng Li, Yinuo Zhao, Zhiyuan Xu, Guang Yang, et al. Robomind: Benchmark on multi-embodiment intelligence normative data for robot manipulation. In _Robotics: Science and Systems (RSS) 2025_ . Robotics: Science and Systems Foundation, 2025a. URL https://www.roboticsproceedings.org/rss21/ p152.pdf. 

- Shihan Wu, Xuecheng Liu, Shaoxuan Xie, Pengwei Wang, Xinghang Li, Bowen Yang, Zhe Li, Kai Zhu, Hongyu Wu, Yiheng Liu, Zhaoye Long, Yue Wang, Chong Liu, Dihan Wang, Ziqiang Ni, Xiang Yang, You Liu, Ruoxuan Feng, Runtian Xu, Lei Zhang, Denghang Huang, Chenghao Jin, Anlan Yin, Xinlong Wang, Zhenguo Sun, Junkai Zhao, Mengfei Du, Mingyu Cao, Xiansheng Chen, Hongyang Cheng, Xiaojie Zhang, Yankai Fu, Ning Chen, Cheng Chi, Sixiang Chen, Huaihai Lyu, Xiaoshuai Hao, Yequan Wang, Bo Lei, Dong Liu, Xi Yang, Yance Jiao, Tengfei Pan, Yunyan Zhang, Songjing Wang, Ziqian Zhang, Xu Liu, Ji Zhang, Caowei Meng, Zhizheng Zhang, Jiyang Gao, Song Wang, Xiaokun Leng, Zhiqiang Xie, Zhenzhen Zhou, Peng Huang, Wu Yang, Yandong Guo, Yichao Zhu, Suibing Zheng, Hao Cheng, Xinmin Ding, Yang Yue, Huanqian Wang, Chi Chen, Jingrui Pang, YuXi Qian, Haoran Geng, Lianli Gao, Haiyuan Li, Bin Fang, Gao Huang, Yaodong Yang, Hao Dong, He Wang, Hang Zhao, Yadong Mu, Di Hu, Hao Zhao, Tiejun Huang, Shanghang Zhang, Yonghua Lin, Zhongyuan Wang, and Guocai Yao. Robocoin: 

46 

An open-sourced bimanual robotic data collection for integrated manipulation. 2025b. URL https://github.com/FlagOpen/RoboCOIN. 

- Wei Wu, Fan Lu, Yunnan Wang, Shuai Yang, Shi Liu, Fangjing Wang, Qian Zhu, He Sun, Yong Wang, Shuailei Ma, Yiyu Ren, Kejia Zhang, Hui Yu, Jingmei Zhao, Shuai Zhou, Zhenqi Qiu, Houlong Xiong, Ziyu Wang, Zechen Wang, Ran Cheng, Yong-Lu Li, Yongtao Huang, Xing Zhu, Yujun Shen, and Kecheng Zheng. A Pragmatic VLA Foundation Model. _arXiv preprint arXiv:2601.18692_ , 2026a. URL https://arxiv.org/abs/2601.18692. 

- Wei Wu, Fangjing Wang, Fan Lu, He Sun, Shi Liu, Yunnan Wang, Yibin Yan, Yong Wang, Shuailei Ma, Xinyang Wang, Yibin Liu, Shuai Yang, Tianxiang Zhou, Kejia Zhang, Lei Zhou, Cheng Su, Nan Xue, Bin Tan, Han Zhang, Youchao Zhang, Fei Liao, Xing Zhu, Yujun Shen, and Kecheng Zheng. From Foundation to Application: Improving VLA Models in Practice. _arXiv preprint arXiv:2607.06403_ , 2026b. URL https://arxiv.org/abs/2607.06403. 

- Lishan Yang, Wenxuan Song, Xi Wang, Pingyue Sheng, Zheng Fang, Ziyang Zhou, Junjie He, Haodong Yan, Jiayi Chen, Nan Sun, Qiao Sun, Pengwei Wang, Lingqiao Liu, Yan Wang, Yuxiang Gao, Feras Dayoub, and Haoang Li. 4D-WAM: Infusing Spatiotemporal Awareness into World Action Models through Trajectory Fields. _arXiv preprint arXiv:2608.08023_ , 2026a. URL https://arxiv.org/abs/2608.08023. 

- Senqiao Yang, Chengyao Wang, Yuxin Chen, Zixuan Wang, Longxiang Tang, Haokun Gui, Jinhui Ye, Changsheng Lu, Xiaoyang Wu, Mingkang Zhu, Pengguang Chen, Shu Liu, Zhuotao Tian, Hengshuang Zhao, Bei Yu, and Jiaya Jia. Beyond Data Scaling: Representation-Centric Continued Pre-training for Vision-Language-Action Models. _arXiv preprint arXiv:2608.27550_ , 2026b. URL https://arxiv.org/abs/2608.27550. 

- Yandan Yang, Shuang Zeng, Tong Lin, Xinyuan Chang, Dekang Qi, Junjin Xiao, Haoyun Liu, Ronghan Chen, Yuzhi Chen, Dongjie Huo, Feng Xiong, Xing Wei, Zhiheng Ma, and Mu Xu. ABot-M0: VLA Foundation Model for Robotic Manipulation with Action Manifold Learning. _arXiv preprint arXiv:2602.11236_ , 2026c. URL https://arxiv.org/abs/2602.11236. 

- Angen Ye, Boyuan Wang, Chaojun Ni, Guan Huang, Guosheng Zhao, Hao Li, Hengtao Li, Jie Li, Jindi Lv, Jingyu Liu, Min Cao, Peng Li, Qiuping Deng, Wenjun Mei, Xiaofeng Wang, Xinze Chen, Xinyu Zhou, Yang Wang, Yifan Chang, Yifan Li, Yukun Zhou, Yun Ye, Zhichao Liu, and Zheng Zhu. GigaWorld-Policy: An Efficient Action-Centered World–Action Model. _arXiv preprint arXiv:2603.17240_ , 2026a. URL https://arxiv.org/abs/2603.17240. 

- Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, Shenyuan Gao, Sihyun Yu, George Kurian, Suneel Indupuru, You Liang Tan, Chuning Zhu, Jiannan Xiang, Ayaan Malik, Kyungmin Lee, William Liang, Nadun Ranawaka, Jiasheng Gu, Yinzhen Xu, Guanzhi Wang, Fengyuan Hu, Avnish Narayan, Johan Bjorck, Jing Wang, Gwanghyun Kim, Dantong Niu, Ruijie Zheng, Yuqi Xie, Jimmy Wu, Qi Wang, Ryan Julian, Danfei Xu, Yilun Du, Yevgen Chebotar, Scott Reed, Jan Kautz, Yuke Zhu, Linxi "Jim" Fan, and Joel Jang. World Action Models are Zero-shot Policies. _arXiv preprint arXiv:2602.15922_ , 2026b. URL https://arxiv.org/abs/2602.15922. 

- Haoqi Yuan, Zhixuan Liang, Anzhe Chen, Ye Wang, Haoyang Li, Pei Lin, Yiyang Huang, Zixing Lei, Tong Zhang, Jiazhao Zhang, Jie Zhang, Jingyang Fan, Gengze Zhou, Qihang Peng, Chenxu Lv, Xiaoyue Chen, An Yang, Fei Huang, Junyang Lin, Dayiheng Liu, Jingren Zhou, Chenfei Wu, and Xiong-Hui Chen. Qwen-RobotManip Technical Report: Alignment Unlocks Scale for Robotic Manipulation Foundation Models. _arXiv preprint arXiv:2606.17846_ , 2026a. URL https://arxiv. org/abs/2606.17846. 

- Tianyuan Yuan, Zibin Dong, Yicheng Liu, and Hang Zhao. Fast-wam: Do world action models need test-time future imagination?, 2026b. URL https://arxiv.org/abs/2603.16666. 

- He Zhang, Lingzhu Xiang, Haitao Lin, Zeyu Huang, Minghui Wang, Dingyan Zhong, Yubo Dong, Yihao Wu, Yongming Rao, Dongsheng Zhang, et al. Hy-Embodied-0.5-VLA: From visionlanguage-action models to a real-world robot learning stack. _arXiv preprint arXiv:2606.14409_ , 2026a. 

47 

- Qihang Zhang, Lin Li, Luyao Zhang, Shuai Yang, Yiming Luo, Shuaiting Li, Ruilin Wang, Junke Wang, Jiahao Shao, Gangwei Xu, Jiaming Zhou, Yishu Shen, Yudong Jin, Fangyi Xu, Shuailei Ma, Jiaqi Liao, Guanxing Lu, Zifan Shi, Yongkun Wen, Yujie Zhao, Weixuan Tang, Xinyang Wang, Chaojian Li, Jiapeng Zhu, Ka Leong Cheng, Nan Xue, Xing Zhu, Yujun Shen, and Yinghao Xu. Native Video-Action Pretraining for Generalizable Robot Control. _arXiv preprint arXiv:2607.08639_ , 2026b. URL https://arxiv.org/abs/2607.08639. 

- Wenbo Zhang, Kaixuan Wang, Yutao Ouyang, Xiaoyu Huang, Liyang Li, Kailun Su, Weiyang Jin, Wenhao Chai, Haotian Liang, Zhiyang Dou, Yue Chen, and Tianxing Chen. An unexpected robot policy: Early evaluations of gpt-6 astra on robodojo and beyond, 2026c. URL https://arxiv.org/ abs/2609.24170. 

- Yuyang Zhang, Wenyao Zhang, Zekun Qi, He Zhang, Haitao Lin, Jingbo Zhang, Yao Mu, Xiaokang Yang, Wenjun Zeng, and Xin Jin. ImageWAM: Do World Action Models Really Need Video Generation, or Just Image Editing? _arXiv preprint arXiv:2606.19531_ , 2026d. URL https://arxiv. org/abs/2606.19531. 

- Zongzheng Zhang, Jingrui Pang, Zhuo Yang, Kun Li, Minwen Liao, Saining Zhang, Guoxuan Chi, Jinbang Guo, Huan ang Gao, Modi Shi, Dongyun Ge, Yao Mu, Jiayuan Gu, Rui Chen, Hao Dong, Huazhe Xu, Li Yi, Yixin Zhu, Hang Zhao, Pengwei Wang, Shanghang Zhang, Guocai Yao, Jianyu Chen, Hongyang Li, and Hao Zhao. Dexora: Open-source vla for high-dof bimanual dexterity, 2026e. URL https://arxiv.org/abs/2605.18722. 

- Jinliang Zheng, Jianxiong Li, Zhihao Wang, Dongxiu Liu, Xirui Kang, Yuchun Feng, Yinan Zheng, Jiayin Zou, Yilun Chen, Jia Zeng, Ya-Qin Zhang, Jiangmiao Pang, Jingjing Liu, Tai Wang, and Xianyuan Zhan. X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment VisionLanguage-Action Model. In _International Conference on Learning Representations_ , 2026. URL https://arxiv.org/abs/2510.10274. 

- Shangchen Zhou, Chongyi Li, Kelvin C.K. Chan, and Chen Change. ProPainter: Improving Propagation and Transformer for Video Inpainting. In _2023 IEEE/CVF International Conference on Computer Vision (ICCV)_ , pages 10443–10452, Paris, France, October 2023. IEEE. doi: 10.1109/ICCV51070.2023.00961. URL https://ieeexplore.ieee.org/document/10378438/. 

48 

# **A Human-to-Robot IK Details** 

At each local IK iteration for a fixed robot base placement, the solver combines position and approach tracking, contact-point and orientation refinement, and projected joint-space regularization: 



where **q** is the current joint configuration and ∆ **q** is its local update. The increment ∆ **q** _p_ tracks TCP position and approach direction, ∆ **q** _s_ refines contact-point alignment and orientation through the damped projection **N** _p_ , and **N** _p_ **g** aux applies projected joint-space preferences. The subscript _p_ denotes position and approach tracking, _s_ denotes contact-point and orientation refinement, and _µ_ is the pseudoinverse damping. The individual terms and update bounds are specified below. 

_Position and approach tracking._ This step reduces TCP position error and approach-direction error beyond the allowed tolerance: 



Here ( **J** _p_ , **e** _p_ ) contains the weighted position-and-approach Jacobian and residual. The damped inverse is defined by **J**<sup>#</sup> _µ_<sup>=</sup><sup>**J**</sup><sup>_⊤_(</sup><sup>**JJ**</sup><sup>_⊤_+</sup><sup>_µ_</sup><sup>**I**)</sup><sup>_−_1, with adaptive damping</sup><sup>_µ >_0.</sup> 

_Contact-point and orientation refinement._ Optional gripper contact-point and full-orientation targets are refined after the position-and-approach step. Their correction accounts for the residual left by that step and uses the position-and-approach projection **N** _p_ to attenuate interference: 



where ( **J** _s_ , **e** _s_ ) stacks the weighted Jacobians and residuals of the enabled contact-point and orientation terms. Contact-point alignment drives the left and right gripper-pad centers toward two virtual points on the human pinch axis. These targets specify geometric alignment. This term is zero when neither refinement is active. 

_Joint-space regularization._ The auxiliary adjustment encourages continuity, a preferred home configuration, and separation from joint limits: 



The reference **q**<sup>ref</sup> is the warm-reference configuration when available, or otherwise the continuation seed. Attraction to this reference is proportional to the negative gradient of<sup><u>1</u></sup> 2<sup>_∥_</sup><sup>**q**</sup><sup>_−_</sup><sup>**q**ref</sup><sup>_∥_</sup> 2<sup>2.The</sup> remaining terms pull toward the home configuration **q**<sup>home</sup> and follow the joint-limit avoidance direction **d** lim. The weights _λr_ , _λh_ , and _λl_ control the respective contributions, with disabled terms set to zero. All three contributions pass through **N** _p_ in the total update. 

The damped projection is not an exact null-space projector. The combined increment is clipped before updating the joint configuration, followed by joint-limit enforcement and bounds on displacement from the continuation seed. Accepted iterates are selected using task-priority criteria rather than a single aggregate loss. Continuity depends on reference attraction, solution reuse, step bounds, and trajectory-level validation. 

49 

