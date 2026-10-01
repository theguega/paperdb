# Pri4R: Learning World Dynamics for Vision-Language-Action Models with Privileged 4D Representation 

Jisoo Kim<sup>_∗_1</sup><sup>_,_2</sup> , Jungbin Cho<sup>_∗_3</sup><sup>_,_5</sup> , Sanghyeok Chu<sup>2</sup><sup>_,_4</sup> , Ananya Bal<sup>5</sup> , Jinhyung Kim<sup>2</sup> , Gunhee Lee<sup>2</sup> , Sihaeng Lee<sup>2</sup> , Seung Hwan Kim<sup>2</sup> Bohyung Han<sup>_†_4</sup> , Hyunmin Lee<sup>_†_2</sup> , Laszlo A. Jeni<sup>_†_5</sup> , Seungryong Kim<sup>_†_1</sup> 

> 1KAIST AI 2LG AI Research, 3Yonsei University, 4Seoul National University, 5Carnegie Mellon University https://jiiiisoo.github.io/Pri4R/ 



<!-- Start of picture text -->
Expert Demonstration Privileged 4D Representation<br>Vanilla VLA<br>World dynamics<br>Pri4R<br>Action-World  + 4D<br>dynamics Supervision<br>Spatiotemporal<br>awareness<br>LIBERO Real World Robocasa<br>100 92.7 96.3 80 73.8 50 46.3<br>90 70 60.3 40 33.1<br>80 60 30<br>0 0 0<br>Vanilla  +Pri4R Vanilla +Pri4R Vanilla +Pri4R<br><!-- End of picture text -->

Fig. 1: **Pri4R** equips Vision–Language–Action (VLA) models with an implicit awareness of action–world dynamics via _privileged_ 4D geometric supervision. Unlike standard VLAs (top) trained only by action imitation (left), Pri4R extracts 3D point tracks from demonstrations (right) and adds an auxiliary head to predict future point trajectories alongside actions. This training signal encourages the policy to model how scene geometry evolves under interaction, improving robustness and task success (bottom), while preserving the original test-time interface with zero inference overhead. 

**_Abstract_ —Humans learn not only how their bodies move, but also how the surrounding world responds to their actions. In contrast, while recent Vision-Language-Action (VLA) models exhibit impressive semantic understanding, they often fail to capture the spatiotemporal dynamics governing physical interaction. In this paper, we introduce Pri4R, a simple yet effective approach that endows VLA models with an implicit understanding of world dynamics by leveraging privileged 4D information during training. Specifically, Pri4R augments VLAs with a lightweight point track head that predicts 3D point tracks. By injecting VLA features into this head to jointly predict future 3D trajectories, the model learns to incorporate the evolving scene geometry within its shared representation space. This allows the action prediction components to leverage a more physically-aware context for precise control. Due to its architectural simplicity, Pri4R is seamlessly compatible with dominant VLA design patterns through minimal changes. During inference, we run the model using the original VLA architecture unchanged; Pri4R adds no extra inputs, outputs, or computational overhead during inference. Across simulation and real-world evaluations, Pri4R** 

**significantly improves performance on challenging manipulation tasks, including a +10% gain on LIBERO-Long and a +40% gain on RoboCasa. We further show that 3D point track prediction is an effective supervision target for learning actionworld dynamics, while also validating our design choices through extensive ablations. Code and checkpoints will be released.** 

#### I. INTRODUCTION 

Generalizable robot policies require not only semantic understanding of the surrounding environment, but also knowledge of the dynamics governing action–world interactions. Recent advances in Vision–Language Models (VLMs) [2, 37], together with the emergence of large-scale robot datasets [27, 42, 54], have enabled Vision–Language–Action (VLA) models [7, 24, 28] to leverage pretrained VLMs for robotic control. A key design principle of these approaches is to minimize architectural changes, introducing only a lightweight action head to transfer the high-level semantic capabilities of VLMs 

to robot manipulation. 

However, this adaptation uses a sole learning signal: simple action labels, which primarily encourage _imitation of demonstrated motions_ , with no knowledge about _the dynamics of the world_ . **Put simply, action labels specify** **_how to move_ , but not** **_what will happen_ .** As a result, the learned policies often produce semantically plausible actions that lack an understanding of underlying world dynamics (e.g., attempting to grasp a handle without accounting for the door’s kinematic constraints), leading to brittle or inaccurate object interactions and ultimately, task failure. 

In contrast, humans reason about actions through an internal understanding of geometry and dynamics, anticipating _how their embodiment and surrounding objects will move, deform, or make contact in response to actions_ [15, 19, 25, 39, 58]. Some efforts sought to incorporate such predictive capabilities by introducing additional models that generate future images or states [5, 16, 21, 59]. More recent approaches instead train a uniform policy to predict actions together with additional signals, including high-level abstractions (e.g., language [48, 69], feature embeddings [10, 65], object-centric representations [24, 66]), and even dense observations (e.g., images [51, 66] and depth maps [65]). 

Although these signals provide extra knowledge for action prediction, they are not directly aligned with the _spatiotemporal metric space_ in which action–world interactions unfold, yielding only indirect supervision where the model must still learn to utilize these cues for precise control. This mismatch suggests that truly modeling world dynamics requires a representation that is both metric and temporally grounded, namely _how 3D geometry evolves over time (4D)_ , rather than predicting static abstractions or goal observations. Furthermore, the aforementioned methods require additional computation during inference, along with complex design choices and hyperparameters. 

In this paper, we propose **Pri4R** , a simple yet effective framework that equips VLA models with an implicit knowledge of world dynamics, enabling more accurate and robust behavior. The key idea is to utilize 4D geometry as a privileged supervision signal during training to revise the VLM’s latent representations. By injecting the VLM’s internal embeddings into a dedicated point tracking head, we guide the backbone to encode the causal relationship between actions and the resulting geometric evolution of the scene. This shared, dynamics-aware representation, then directly empowers the action prediction head to reason with a deeper physical context, without introducing additional inputs or architectural changes at inference time. Figure 1 illustrates the concept of Pri4R. 

Specifically, we first precompute 3D point tracks for the demonstration trajectories used for training. We then augment the VLA with a lightweight point tracking head that predicts future 3D trajectories, conditioned on the model’s internal embeddings. This design ensures that the gradients from the point tracking task inform and enrich the shared feature space used for action prediction. Pri4R is broadly applicable and easy to integrate into existing VLA frameworks with minimal 

architectural changes, for both backbone-centric VLAs [28] and expert-style VLAs [7, 24]. At test time, the auxiliary units are discarded, allowing the original VLA architecture to run unchanged while benefiting from the integrated information about world dynamics. 

Through extensive experiments on multi-task simulation benchmarks (LIBERO and RoboCasa) and real-world evaluations, we show that Pri4R consistently outperforms stateof-the-art baselines across all settings. We further analyze supervision target variants and find that 3D point track prediction most effectively encourages learning of action-world dynamics. Finally, we ablate the core design choices of Pri4R and show that each component is necessary for learning world dynamics and improving downstream performance. 

Our contributions are summarized as follows: 

- We propose Pri4R, which facilitates learning world dynamics by leveraging 3D point tracks as privileged supervision [18] during training to enrich the shared representation of VLA models. 

- We show that Pri4R consistently improves the performance of SOTA VLA models across simulation and real-world tasks without any inference time overhead or architectural changes. 

- Through systematic analysis, we demonstrate that predicting 3D point tracks is particularly effective for learning action-world dynamics and extensively ablate each design choice of Pri4R. 

#### II. RELATED WORK 

**Vision-Language-Action models.** Pioneered by Robotics Transformers [8, 70], Vision–Language–Action (VLA) models [7, 10, 11, 22, 24, 28, 32, 34, 38, 50] adapt large pretrained Vision–Language Models (VLMs) [2, 13, 37] for robotic control by predicting actions conditioned on visual observations and language instructions, typically via action-specific heads or modules. OpenVLA [28] trains a 7B VLM [26] by autoregressively predicting discrete action tokens on the OpenX [42] dataset. Follow-up works improve OpenVLA by reducing the model footprint (e.g., to 1B parameters) [4] or by incorporating visual trace prompting [68]. In contrast, other methods [7, 33, 50, 57] use generative action heads (e.g., diffusion or flow matching) to produce continuous actions. More recently, OpenVLA-OFT [29] shows that L1 regression on continuous actions with parallel decoding can both improve performance and substantially increase inference throughput. **Improving VLAs with forecasting.** Inspired by human anticipatory reasoning, forecasting has played an important role in VLAs. One line of work [1, 5, 9, 16, 21, 59] employs generative models to explicitly predict future states, images, or videos, and then conditions action policies on these predicted modalities. This can be seen as the generative model serving as a high-level planner, guiding the low-level policy to improve task achievement. However, these approaches are often constrained by the accuracy of the predicted signals and the additional inference-time latency they introduce. In contrast, more recent methods leverage the VLA itself to 

produce auxiliary signals within the same model, using them as intermediate reasoning either before or alongside action prediction at test time. Concretely, this is achieved either by first generating intermediate signals and then causally predicting actions [10, 24, 51, 66, 67, 69], or by generating signals and actions jointly in parallel [46, 65]. Although effective, these auxiliary signals (e.g., language, feature embeddings, images, videos) are not expressed in the same spatiotemporal metric space as actions, and therefore provide only indirect supervision for action learning. 

**Point tracking for manipulation.** 2D point tracks have been actively used in robot learning [62, 64, 68], but they provide limited geometric information. With the emergence of strong 3D point tracking models [17, 61], recent work has begun to adopt 3D point tracks for robotic manipulation as a structured spatiotemporal representation for downstream learning and control. In particular, point tracks have been used for policy learning [45, 56, 63], reward modeling [43, 47, 62], and even as action representations [23, 41]. In our work, we use 3D point tracks as a privileged training signal to encourage the model to learn geometric 4D world dynamics that are directly relevant for action prediction, without requiring point tracks at inference time. III. PRELIMINARIES 

This section formalizes the Vision-Language-Action (VLA) framework and reviews representative baselines, providing the background to discuss how Pri4R incorporates world dynamics into these formulations. 

_A. VLA Formulation_ 

Vision-Language-Action (VLA) models are typically developed by fine-tuning pretrained VLMs via imitation learning [3]. At each time step _t_ , the policy receives an observation **o** _t_ ≜ ( **I** _t,_ **t** _t,_ **q** _t_ ) and predicts an action chunk **a** _t_ : _t_ + _H_ over a horizon _H_ , where **I** _t_ = _{_ **I**<sup>_i_</sup> _t_<sup>_}N_</sup> _i_ =1<sup>denotesasetof</sup><sup>_N_multi-</sup> view camera images, **t** _t_ is a tokenized language instruction, and **q** _t_ is the robot proprioceptive state. Note that, if _H_ = 1, the chunk reduces to a single action **a** _t_ . 

In this framework, the images **I** _t_ and robot state **q** _t_ are encoded by modality-specific encoders and projected onto the backbone’s token embedding space alongside **t** _t_ . Given these inputs, the VLM backbone produces high-level multimodal embeddings. Since standard VLMs are not designed to output action chunks, VLAs attach an additional _actionprediction component_ (e.g., a lightweight action head) that consumes these embeddings to parameterize an action distribution conditioned on the backbone’s internal representations (and optionally **q** _t_ ). The full set of model parameters _θ_ is trained end-to-end on a demonstration dataset _D_ with behavior cloning [44], maximizing the log-likelihood of the demonstrated action chunks, which is given by 



#### _B. VLA Baselines_ 

To adopt our approach, we consider two state-of-the-art VLA architectures with distinct design patterns: OpenVLAOFT [29] and the _π_ series [7, 24]. 

OpenVLA-OFT employs an Optimized Fine-Tuning (OFT) recipe that enhances the original OpenVLA through (i) parallel decoding with bidirectional attention, (ii) action chunking, (iii) a continuous action representation, and (iv) a simple _ℓ_ 1 regression objective. Architecturally, OpenVLA-OFT replaces the discrete action-token output with an MLP regression head. This MLP head processes the final-layer hidden states at the action query token positions and maps them directly to continuous action chunks **a** _t_ : _t_ + _H_ , enabling efficient, nonautoregressive prediction. 

The _π_ series, in contrast, generates action chunks through a flow-matching framework. It augments the pretrained VLM backbone with a dedicated transformer action expert. This expert takes the robot proprioceptive state **q** _t_ and a noisy action chunk as inputs, predicting a corresponding vector field (velocity) for iterative denoising. The action expert integrates with the backbone by attending to its internal hidden states via shared self-attention. To preserve the VLM’s pretrained representations, _π_ employs a blockwise causal attention mask; while the robotics-specific action tokens (including the noisy action chunk) can attend to the full multi-modal context, the language and image tokens are restricted from attending to these action-related tokens. 

#### IV. PRI4R: LEARNING WORLD DYNAMICS VIA PRIVILEGED 4D REPRESENTATIONS 

Pri4R is a framework that incorporates privileged geometric information to improve the world dynamics understanding of VLA models. During training, we use high-fidelity 4D signals as an auxiliary supervision to refine the internal representations of the VLM backbone. By supervising the model to predict the physical evolution of the scene, we enable the VLA to develop a physically-aware context for robot control, without requiring any additional inputs or computational overhead during inference. 

#### _A. Learning from Privileged Point Track Head_ 

To implement this privileged learning strategy, we augment the VLM backbone with a lightweight _point track head_ that predicts _future 3D trajectories_ of scene points. By feeding the backbone’s multi-modal embeddings into this head, the shared representations are forced to encode the spatiotemporal constraints of the environment. The point track head is deliberately lightweight, consisting of two small MLPs that interface with the backbone without altering its original architecture. 

At time _t_ , the _point MLP_ embeds the current point set _Pt ∈_ R<sup>_Np×_3</sup> into per-point features **e** _t ∈_ R<sup>_Np×d_</sup> . To integrate scene context, we introduce a sequence of multi-modal embeddings over the action horizon, **z** _t_ = _ϕ_ ( **o** _t_ ) _∈_ R<sup>_H×d_</sup> , which is derived from the backbone _ϕ_ and consumed by the action head. We broadcast **z** _t_ across points to obtain a tensor in R<sup>_H×Np×d_</sup> and concatenate it with the per-point features **e** _t_ (broadcast across the horizon _H_ ). The resulting tensor is fed into a _fusion MLP_ to predict per-step 3D displacements: 





<!-- Start of picture text -->
a) OpenVLA-OFT b)  𝝅  Series c) Point Track Head<br>𝑎! 𝑎!"#. . . 𝑎!"$ Δ 𝑃 ' !:!"$ 𝑧! . . . Δ 𝑃 ' !:!"$ Δ 𝑃 ' !:!"$<br>Action 𝑎! 𝑎!"#. . . 𝑎!"$ Embedding . . .<br>Head<br>Module<br>𝑧! . . . Point Pretrained Action𝑎! 𝑎!"# . . . Cross x 𝐿 Point 𝑧! Fusion<br>Track VLM Expert Attention Track MLP<br>Pretrained VLM Head Self Head +<br>Attention<br>. . . Images Text Noise PointMLP<br>Images Text Action . . .<br>𝑃!<br>Query<br>Tokens Δ 𝑃 ' ! = 𝑃 ' !"# − 𝑃 ' ! : Original VLA : Train  Only + : Concat Query TokensEmbedding 𝑃! 𝑃!<br>. . .<br><!-- End of picture text -->

Fig. 2: **Overview of Pri4R.** We augment two common VLA architectures with an auxiliary point track head that predicts per-step 3D point displacements ∆<sup>�</sup> _P t_ : _t_ + _H_ from backbone embeddings **z** _t_ and the current point set _Pt_ . (a) For backbone-centric VLAs (e.g., OpenVLA-OFT [28]), we set **z** _t_ to the final layer action-query token embeddings. (b) For expert-style VLAs (e.g., _π_ [7, 24]), we condition an embedding module on the backbone’s final layer hidden states to produce **z** _t_ . (c) The point track head encodes _Pt_ with a PointMLP, then fuses the resulting point features with **z** _t_ via a FusionMLP to predict future point tracks. Privileged 3D point track supervision during training forces the VLA to model how scene geometry evolves, yielding more reliable interaction and higher task success, while leaving the test-time interface and compute completely unchanged. 

where _⊕_ denotes the feature concatenation operator. 

Through this architecture, the auxiliary loss gradients from the point track head are backpropagated into the VLM backbone, encouraging it to refine its shared representation to capture essential world dynamics. By jointly optimizing for both action and 3D trajectory prediction, the VLA model understands a richer, geometry-aware context that is inherently grounded in the physical evolution of the scene. While the core prediction mechanism remains consistent, the interface between the point track head and the VLA backbone is tailored to the specific design of each model family. We examine the OpenVLA-OFT and _π_ family models; their detailed designs are illustrated in Figure 2 and described below. 

**OpenVLA-OFT** : We set **z** _t_ to the final-layer hidden states of the action query tokens (i.e., the empty action embeddings) produced by the backbone. Since these are the exact embeddings mapped to actions by the action head, injecting them into the point track head allows the backbone to encode the underlying scene dynamics necessary for precise control. 

**_π_ family** : Since the action expert in _π_ models interacts with the backbone via masked self-attention rather than a fixed embedding, the choice of **z** _t_ is less straightforward. We introduce a lightweight transformer embedding module that takes a set of learnable query tokens and applies crossattention over the final-layer image and language tokens from the VLM backbone. This yields an action-horizon embedding sequence **z** _t ∈_ R<sup>_H×d_</sup> , which is then processed by the point track head identically to the OpenVLA-OFT case. 

#### _B. Why 3D Point Tracks as Privileged Supervision?_ 

Although prior works have explored a variety of predictive representations to train VLAs [10, 24, 46, 51, 65–67, 69], these signals are often ill-suited for learning world dynamics. Traditional representations are frequently (1) _temporally sparse_ , predicting only a goal observation at the end of the action 

horizon; (2) _lack explicit spatial structure_ , as language or latent feature embeddings do not preserve metric geometry; and (3) _spatially redundant_ , since dense predictions such as images or depth maps largely reproduce the input observation [65]. 

We therefore use _3D point tracks_ as the target representation. Compared to other representations, 3D point tracks are: (1) _temporally dense_ , matching the action horizon to capture fine-grained world interaction; (2) _geometric_ , providing metric 3D structure that promotes spatial awareness; and (3) _spatially sparse_ , enabling an efficient learning objective by focusing on a compact set of informative points rather than redundant dense grids like video or depth maps [65]. Most importantly, 3D point tracks reside in the same _spatiotemporal metric space_ as robot actions, providing a supervisory signal that is naturally aligned with control. 

#### _C. Construction of 3D Point Track Supervision_ 

To realize the proposed learning with a privileged information framework, we obtain 3D point tracks for every demonstration in _D_ to serve as privileged supervision. For each training sample ( **a** _t_ : _t_ + _H ,_ **o** _t_ ) _∼D_ , we assume access to _Np_ tracked points with fixed identities over the same horizon and denote the point set at time _τ_ by 



where _τ ∈{t, . . . , t_ + _H_ + 1 _}_ and _j ∈{_ 1 _, . . . , Np}_ . We refer to the sequence _{Pτ }_<sup>_t_</sup> _τ_<sup>+</sup> =<sup>_H_</sup> _t_<sup>+1</sup> as the 3D point tracks for the demonstration window starting at time _t_ . 

Rather than directly regressing absolute positions, we supervise the point track head with per-step 3D displacements, 



and denote ∆ _Pτ_ = _{_ ∆ _p_<sup>_τ_</sup> _j_<sup>_}N_</sup> _j_ =1<sup>_p_and∆</sup><sup>_Pt_:</sup><sup>_t_+</sup><sup>_H_=</sup><sup>_{_∆</sup><sup>_Pτ}_</sup> _τ_<sup>_t_+</sup> =<sup>_H_</sup> _t_<sup>.</sup> Given the backbone embeddings derived from the current observation **o** _t_ and the current point set _Pt_ , the point track head 

TABLE I: **Results on LIBERO.** Success rates (SRs) are reported across four task suites (500 trials per suite). Entries marked with * are taken from the OpenVLA-OFT paper [29]. Best performances for each task are marked in **bold** . Pri4R improves average success rates of every state-of-the-art VLAs ( _π_ series and OpenVLA-OFT) and almost every task in LIBERO. 

||Average|LIBERO - Spatial|LIBERO - Object|LIBERO - Goal|LIBERO - Long|
|---|---|---|---|---|---|
||SR (_↑_)|SR (_↑_)|SR (_↑_)|SR (_↑_)|SR (_↑_)|
|Diffusion Policy [14]*|72.4|78.3|92.5|68.3|50.5|
|Octo [50]*|75.1|78.9|85.7|84.6|51.1|
|DiT Policy [20]*|82.4|84.2|96.3|85.4|63.8|
|OpenVLA [28]*|76.5|84.7|88.4|79.2|53.7|
|_π_0 [7]|87.4 _±_ 0.2|87.8 _±_ 0.2|84.9 _±_ 0.4|91.2 _±_ 0.9|85.7 _±_ 0.7|
|_π_0[7] **+ Pri4R**|90.6 _±_ 0.2|92.8 _±_ 0.7|88.6 _±_ 0.3|95.3 _±_ 0.4|85.6 _±_ 0.2|
|_π_0_._5 [24]|92.6 _±_ 0.4|96.1 _±_ 0.8|88.3 _±_ 0.8|95.6 _±_ 0.5|90.5 _±_ 1.2|
|_π_0_._5 [24] **+ Pri4R**|94.0 _±_ 0.2|**97.2** _±_ 0.6|88.9 _±_ 0.6|95.6 _±_ 0.5|94.3 _±_ 0.6|
|OpenVLA-OFT [29]|92.7 _±_ 0.1|90.8 _±_ 0.3|98.2 _±_ 0.0|96.4 _±_ 0.4|85.5 _±_ 0.2|
|OpenVLA-OFT [29] **+ Pri4R**|**96.3** _±_ 0.2|93.2 _±_ 0.4|**98.6** _±_ 0.0|**98.1** _±_ 0.5|**95.3** _±_ 0.3|



predicts future 3D displacements ∆<sup>�</sup> _P t_ : _t_ + _H_ in parallel with the action prediction **a** _t_ : _t_ + _H_ . This auxiliary branch is temporally aligned with the action horizon and utilized exclusively during training. By discarding the point track head after training, the final policy architecture remains identical to the original VLA, ensuring no additional computational overhead or input requirements at inference time. 

#### _D. Training_ 

**Data construction.** We construct 3D point tracks for both simulation and real world. In simulation, we can access the ground-truth scene mesh from the simulator [52]. Therefore, we initialize _Np_ query points only at the first frame by cropping the mesh within a robot-centered 3D cube and sampling points over mesh faces. To track these points, we store the corresponding face indices and barycentric coordinates at initialization, and roll out each action sequence while retrieving the 3D locations of the same surface points at every timestep, yielding _{Pτ }_<sup>_T_</sup> _τ_ =1<sup>.</sup> 

For real scenes, we use an off-the-shelf 3D point tracking model [61] to annotate pseudo-labeled tracks. To focus sampling toward foreground regions, we use a segmentation model to sample more points on robots and objects, while sampling background points uniformly. Since our real-world dataset is recorded with a fixed camera setup, we can recover stable world coordinates for the tracked points over time. 

**Objectives.** We follow the original action training objective of each VLA, _ℓ_ 1 regression for OpenVLA-OFT and flow matching for _π_ models. We add an auxiliary _ℓ_ 1 loss on 3D point track displacements ∆<sup>�</sup> _P t_ : _t_ + _H_ . The total loss is 



where _ω_ pt controls the balance between two loss terms. 

#### V. EXPERIMENTS 

We evaluate Pri4R’s ability to inject knowledge about world dynamics during fine-tuning by using 3D point tracks as privileged supervision. Specifically, we seek to answer the following questions: 

- 1) Does Pri4R improve state-of-the-art VLAs across diverse robot tasks? 

- 2) Which characteristics of 3D point tracks are critical for world-dynamics learning and downstream control performance? 

- 3) How do Pri4R design choices affect the performance of the fine-tuned policy? 

- 4) Does Pri4R improve performance on real world tasks that require awareness about world dynamics? 

#### _A. Experimental Setup_ 

**Baselines.** We compare Pri4R against recent VLA and imitation-learning baselines with publicly available code: Diffusion Policy [14], Octo [50], DiT Policy [20], OpenVLA [28], OpenVLA-OFT [29], _π_ 0 [7], and _π_ 0 _._ 5 [24]. For Pri4R, we use OpenVLA-OFT, _π_ 0, and _π_ 0 _._ 5 as backbones, as they (i) achieve SOTA results across diverse benchmarks and (ii) span two representative VLA architectures: backbone-centric policies with action heads and expert-style action models. 

**Implementation details.** For OpenVLA-OFT [29], we apply LoRA adapters to the pretrained backbone and jointly train the point track head. We set the point supervision weight to _w_ pt = 1 and sample _Np_ = 1024 points per trajectory. All remaining model hyperparameters follow the OpenVLAOFT default configuration. For _π_ 0 and _π_ 0 _._ 5 [7, 24], we use the official PyTorch implementation and fully fine-tune the base model together with the additional point track embedding module and our point track head. We set _w_ pt = 1 and use _Np_ =1024. All other model hyperparameters follow the default settings used for the corresponding _π_ models. 

#### _B. Mutitask Simulation Benchmarks_ 

**Experimental setup.** We evaluate Pri4R on two multi-task simulation benchmarks: LIBERO [36] and RoboCasa [40]. LIBERO is a language-conditioned tabletop manipulation benchmark that tests multitask generalization across task families with varying spatial layouts, target goals, and object configurations; we report results on four suites (Spatial, Object, Goal, and Long), each with 10 tasks and 50 demonstrations per task. RoboCasa is a large-scale kitchen manipulation benchmark with diverse scenes and articulated interactions; we evaluate 24 atomic tasks using the Human-50 dataset, which provides 50 demonstrations per task. 

TABLE II: **Results on Robocasa.** Success rate (SR) for each method across 24 task suites, grouped following [40], and averaged over 50 trials. Pri4R improves average success rates of every state-of-the-art VLAs and almost every task in Robocasa. 

||Average|Pick and Place|Open/Close doors|Open/Close drawers|Twisting knobs|Turning levers|Pressing buttons|Insertion|
|---|---|---|---|---|---|---|---|---|
||SR (_↑_)|SR (_↑_)|SR (_↑_)|SR (_↑_)|SR (_↑_)|SR (_↑_)|SR (_↑_)|SR (_↑_)|
|_π_0 [7]|38.8|24.0|45.0|78.0|29.0|57.3|57.3|0.0|
|_π_0 [7] **+ Pri4R**|42.2 (+3.4)|24.0 (+0.0)|49.0 (+4.0)|84.0 (+6.0)|30.0 (+1.0)|71.3 (+14.0)|59.3 (+2.0)|2.0 (+2.0)|
|_π_0_._5 [24]|52.9|**54.3**|51.0|75.0|28.0|79.3|60.0|4.0|
|_π_0_._5 [24] **+ Pri4R**|**57.0** (+4.1)|52.0 (-2.3)|**68.5** (+17.5)|**89.0** (+14.0)|**33.0** (+5.0)|**86.7** (+7.4)|54.7 (-5.3)|5.0 (+1.0)|
|OpenVLA-OFT [29]|33.1|21.8|45.7|59.0|8.0|36.0|56.0|27.0|
|OpenVLA-OFT [29]**+Pri4R**|46.3 (+13.2)|23.0 (+1.2)|61.7 (+16.0)|80.0 (+21.0)|25.0 (+17.0)|66.7 (+30.7)|**79.3** (+23.3)|**34.0** (+7.0)|



TABLE III: **Analysis of supervision.** Average success rates on RoboCasa. 3D point track is the most effective supervision, while using both robot and scene points is crucial. 

|**Method**|**SR** _↑_|∆|
|---|---|---|
|OpenVLA-OFT [29]|33.1|–|
|_What supervision signal?_|||
|+ Goal point set|33.8|+0.7|
|+ 2D point track|37.0|+3.9|
|+ Depth|42.3|+8.3|
|**+ Ours (3D point track)**|**46.3**|**+13.2**|
|_What to track?_|||
|+ Only scene points|35.2|+2.1|
|+ Only robot points|43.8|+10.7|
|**+ Ours (both scene and robot)**|**46.3**|**+13.2**|



For both baselines, we train a single multitask policy per benchmark. Inputs include language instruction, proprioception and two camera views (thrid-person and wrist) for LIBERO and three views (left, right, wrist) for RoboCasa. For OpenVLA-OFT, we fine-tune with batch size 128 for 90k steps on LIBERO [36], and batch size 64 for 120k steps on RoboCasa [40]. For _π_ series, we finetune with batch size 128 for 30k steps on both benchmark. Additional implementation details are provided in the appendix. 

**Results.** Table I shows that Pri4R improves the average success rate of all strong VLA baselines on LIBERO. The gains are especially pronounced on LIBERO-Long with OpenVLAOFT, where Pri4R yields a 9.8% absolute improvement, indicating that privileged geometric 4D supervision encourages better modeling of action–world interactions. On the more challenging RoboCasa benchmark, which requires robust execution under randomized scene configurations, Pri4R achieves even larger success rate gains than on LIBERO (Table II), demonstrating that _privileged_ point track supervision transfers beyond temporally structured suites and improves generalization under stronger distribution shift. Moreover, Figure 3 illustrates Pri4R’s training dynamics: performance improves more slowly during the first _∼_ 20K steps due to the added point track prediction objective, but then increases rapidly, reaching the baseline’s peak performance 2 _._ 7 _×_ faster. This speedup corresponds to approximately 8 _×_ H200 GPU-days (about 24 hours on 8 H200s). Finally, we visualize the predicted point tracks in Figure 4. Additional results are shown in the Appendix. 



Fig. 3: **Training dynamics.** Pri4R learns slowly at the early stage due to the 3D point track objective, but improves performance rapidly, reaching the baseline peak 2 _._ 7 _×_ faster. 

#### _C. Analysis of 3D Point Track Supervision_ 

**Temporality & Spatiality.** To isolate the roles of temporal density and metric geometry in point track supervision, we evaluate two controlled variants of our 3D full-horizon displacement target, ∆ _Pt_ : _t_ + _H_ . For _temporality_ , we replace ∆ _Pt_ : _t_ + _H_ with a goal-only objective that predicts only the terminal point set _Pt_ + _H_ ; concretely, we mean-pool backbone embeddings before the fusion MLP and keep the rest unchanged. For _spatiality_ , we preserve temporal density but remove metric 3D structure by projecting 3D tracks to a single camera view and supervising with 2D tracks. Table III shows that both goal-only prediction and 2D tracks yield only modest improvements, whereas supervising temporally dense _and_ metrically grounded 3D tracks over the full horizon leads to substantially larger gains, indicating that the benefits are strongest when **temporality and spatiality are jointly present** in the supervision signal. 

**Spatial redundancy** To assess whether 3D point tracks provide advantages over spatially dense supervision such as depth, we replace point track prediction with future depth map prediction. Because depth maps are high-resolution, we first train a depth variational autoencoder (VAE) and use its latent codes as supervision instead of raw depth. Table III shows that predicting depth improves performance over the baseline, but it consistently underperforms Pri4R with 3D point tracks. We attribute this gap to two limitations of depth supervision: (i) depth maps are spatially dense and therefore temporally 

TABLE IV: **Ablation on** _Pt_ **as input.** ∆ denotes absolute improvement over the OpenVLA-OFT baseline [29]. “ _Pt_ input” indicates that the point set _Pt_ is appended to the backbone as additional tokens alongside image and language inputs. “ _Pt_ : _t_ + _H_ track” refers to our proposed displacement supervision over the action horizon. 

||Need _Pt_ at test?|**Avg.**|PnP|Doors|Drawers|Knobs|Lever|Button|Insertion|
|---|---|---|---|---|---|---|---|---|---|
|||SR / ∆|SR / ∆|SR / ∆|SR / ∆|SR / ∆|SR / ∆|SR / ∆|SR / ∆|
|OpenVLA-OFT [29]|No|33.1 / +0.0|21.8 / +0.0|45.7 / +0.0|59.0 / +0.0|8.0 / +0.0|36.0 / +0.0|56.0 / +0.0|27.0 / +0.0|
|+ _Pt_ input|Yes|33.3 / +0.2|14.0 / -7.8|48.0 / +2.3|59.0 / +0.0|10.0 / +2.0|49.3 / +13.3|59.3 / +3.3|26.0 / -1.0|
|+ _Pt_ input + _Pt_:_t_+_H_ track|Yes|34.5 / +1.4|14.3 / -7.5|49.0 / +3.3|59.0 / +0.0|10.0 / +2.0|54.0 / +18.0|60.7 / +4.7|29.0 / +2.0|
|**+** _Pt_:_t_+_H_ **track (Ours)**|**No**|**46.3 / +13.2**|**23.0 / +1.2**|**61.7 / +16.0**|**80.0 / +21.0**|**25.0 / +17.0**|**66.7 / +30.7**|**79.3 / +23.3**|**34.0 / +7.0**|



TABLE V: **Ablations on embedding module with** _π_ 0 _._ 5 **.** Average success rate (SR) on RoboCasa. Each design choice of our embedding module is important. 

|**Method**|**SR** _↑_|∆|
|---|---|---|
|_π_0_._5 [24]|52.9|–|
|_Design choice of embedding module_|||
|+ Point expert|53.4|+0.5|
|+ Backbone query token (attend action)|54.4|+1.5|
|+ Backbone query token|54.8|+1.9|
|**+ Ours**|**57.0**|**+4.1**|



redundant under a static camera, and (ii) they do not provide identity registration across time. Consequently, depth provides a weaker signal for learning how the scene evolves under interaction compared to tracking consistent 3D points over time. 

**Scene-robot interaction.** Finally, we compare tracking only environment points or only robot points to our world tracking that couples both. Tracking only the robot mainly captures self-motion, while tracking only the scene weakens sensitivity to contact and articulation. Table III shows that neither matches world tracking, suggesting that modeling robot– environment interaction is key to improving performance in complex manipulation tasks. 

#### _D. Ablations on Design Choices_ 

**Effect of** _Pt_ **input.** A key design choice in Pri4R is to feed the current point set _Pt only to the point track head_ , rather than to the backbone [6] or the action head [30, 49]. This has two practical advantages: (1) it requires no 3D observations at inference time, and (2) it mitigates distribution shift from pretraining, where **o** _t_ consists only of images and language. As shown in Table IV, appending _Pt_ to the backbone can improve several suites by injecting explicit 3D cues, but it reduces performance on Pick-and-Place, where language plays a larger role [35]. We attribute this drop to introducing a new token type into the VLM embedding space, which can perturb the pretrained representation [30, 49]. Adding point track supervision on top of the backbone _Pt_ input improves the average success rate, but the PnP degradation persists. In contrast, conditioning _only_ the point track head on _Pt_ while supervising it with point tracks yields consistent gains across suites and even improves PnP performance. 



<!-- Start of picture text -->
“Pick up the Santa hat.”<br>“Close the cabinet door.”<br><!-- End of picture text -->

Fig. 4: **Predicted point tracks in simulation and the real world.** Future trajectories are visualized in a rainbow color map. As highlighted in the red boxes, Pri4R accurately predicts point tracks for both scene elements and the robot. 

TABLE VI: **Point-track head ablations.** SR of OpenVLAOFT on LIBERO Long. Our point-track head has two slots: _Point MLP_ and _Fusion MLP_ . We plug in different architectures for each slot and report SR; ∆ is w.r.t. the no-head baseline. 

|**Point Encoder**|**Fusion Module**|**SR** _↑_|∆|
|---|---|---|---|
|**OpenVLA-OFT (no **|**point-track head)**|89.2|–|
|PointNet [12]|**Ours**|80.8|_−_8_._4|
|PtTrans. [60]|**Ours**|92.4|+3_._2|
|**Ours**|Transformer [53]|92.2|+3_._0|
|**Ours**|**Ours**|**94.4**|+**5**_._**2**|



**Ablations on the embedding module.** While OpenVLA-OFT only requires adding a point track head, the _π_ family additionally needs an embedding module to produce **z** _t ∈_ R<sup>_H×d_</sup> . Table V compares several ways of constructing **z** _t_ . A simple baseline introduces a _point expert_ mirroring the action expert, conditioned via layer-wise self-attention with the backbone. We also evaluate learnable query tokens injected into the backbone [65], either allowing attention to action tokens or masking this interaction. All alternatives underperform our embedding module, highlighting the importance of our design for learning world dynamics via point tracking. 

**Ablations on point track head.** We conduct an ablation study on the tracking head architecture, which consists of two components: (i) a point embedding module and (ii) a fusion module that combines action embeddings with point embeddings to predict point tracks. Table VI compares alternative designs, including PointNet and point-transformer point 

TABLE VII: **Quantitative results on the real-world setup.** 

|Model|Height|Spa|tial|Depth|Trackin|g|
|---|---|---|---|---|---|---|
||Unseen|Seen|Unseen|Seen Unseen|Seen Unseen|OOD|
|OpenVLA-OFT|83.3|100.0|60.0|45.9 50.0/ 0.0|75.0<br>41.7|50.0|
|**+ Pri4R**|**96.7**|**100.0**|**80.0**|**79.2**<br>**50.0**/**37.5**|**100.0**<br>**66.7**|**66.7**|
|_π_0_._5|60.0|90.0|30.0|66.7 50.0/ 0.0|50.0<br>0.0|33.8|
|**+ Pri4R**|**66.7**|**100.0**|**60.0**|**95.8**<br>**100.0**/0.0|**66.7**<br>0.0|**50**|



embedding, as well as a transformer-based fusion module. Overall, we find that our lightweight MLP design for both modules achieves the best performance, outperforming heavier point encoders and fusion transformers. 

#### _E. Real World Evaluation_ 

To evaluate the Pri4R’s spatiotemporal capabilities in real world settings, we conduct four real world tasks on the OMYF3M robot, a one-arm manipulation platform. **Experimental setup** OMY-F3M is a one-arm manipulation platform with a 6-DoF arm actuated by DYNAMIXEL-Y motors. We operate the robot at 10Hz with a 7-dimensional action space comprising six target absolute joint angles and one gripper command. Observations are collected from three camera viewpoints (top, wrist-mounted, and left third-person). We finetune OpenVLA-OFT for 60k-150k steps per task with an action chunk size of 10, and execute the full action chunk at inference time. We evaluate models on four real world tasks that require world-dynamics understanding: 

- 1) **Pick-and-place over an obstacle.** The robot must place an object over obstacles of three distinct heights. We collected 45 demonstrations and evaluated 30 trials. 

- 2) **Pick-and-place into a bin.** The robot picks up an object and places it into a bin. The object start location is either seen or unseen during training. We collected 45 demonstrations and evaluated 20 trials. 

- 3) **Pick the farthest object.** The robot must pick up the object farthest from the robot among multiple candidates. Both the target object identity and object positions are randomized. We collected 48 demonstrations and evaluated 40 trials. 

- 4) **Pick a moving object.** The object is relocated while the robot is approaching, requiring the robot to track the moving object and grasp at the updated position. We collected 40 demonstrations and evaluated 36 trials. 

**Results.** Table VII shows the performance on four real world tasks designed to evaluate the understanding of worlddynamics under contact, obstacles, randomized object configurations, reaching with depth-dependent geometry, and target relocation. Pri4R consistently outperforms the OpenVLAOFT and _π_ 0 _._ 5 baselines across all tasks, demonstrating that geometric 4D supervision improves real world spatiotemporal control. As shown in Figure 6, the baseline model tends to collide with obstacles rather than routing around them, approaches incorrect grasp locations when object positions are randomized, and in some cases closes the gripper on empty space yet still executes the remainder of the action chunk as if contact were made. In contrast, Pri4R exhibits stronger 









Fig. 5: **Real-world setup.** Blue boxes indicate the target. In Pick the farthest object, the red box marks a closer distractor object; in Pick up the doll and place in the white bin, the red box marks a distractor. 



<!-- Start of picture text -->
(a) OpenVLA-OFT [29] (b) OpenVLA-OFT + Pri4R<br>Libero<br>Robocasa<br>Realworld<br>Realworld<br><!-- End of picture text -->

Fig. 6: **Qualitative comparison across tasks.** Baseline (left) failures vs. Our method (right) successes. 

spatiotemporal and interaction awareness. It avoids collisions, re localizes targets, grasps objects from unseen locations, maintains geometrically consistent approaches, and executes grasps at the updated object position. This advantage is most evident in the moving-target setting, where Pri4R continues to update the grasp plan as the object relocates, whereas the baseline often stops early at an outdated location, reflecting limited spatiotemporal awareness. Overall, these results suggest that Pri4R transfers spatiotemporal understanding effectively to real world settings and improves robustness under distribution shifts in object placement and dynamics. More details on real world experiments are provided in the appendix. 

#### VI. DISCUSSION, LIMITATIONS, AND FUTURE WORK 

We presented Pri4R, a framework that enhances the world dynamics understanding of VLA models through privileged 4D representations. By supervising the model to predict 3D point tracks during training, we demonstrated that VLA backbones can develop a more physically-aware context, leading to improved control performance without any inference-time overhead. Our results across various benchmarks suggest that capturing the spatiotemporal evolution of a scene is a critical component for robust robot manipulation. 

Despite its effectiveness, Pri4R makes no changes at inference time. While this simplicity is a strength, it may also limit performance: additional test-time computation or explicit geometric inputs could further improve robustness. Moreover, we evaluate Pri4R primarily in the fine-tuning setting on demonstration benchmarks and a small set of realworld rollouts, rather than in large-scale pretraining regimes (e.g., Embodiment X). We believe incorporating 3D pointtrack supervision during pretraining would have an even larger impact on learning action–world dynamics than using it only for fine-tuning. Finally, since Pri4R transfers to real-world settings, and 4D point tracks can be obtained from off-theshelf tracking models, our approach is directly applicable to large-scale robotics datasets. 

#### ACKNOWLEDGMENTS 

This work was supported by LG AI Research. 

#### REFERENCES 

- [1] Anurag Ajay, Seungwook Han, Yilun Du, Shuang Li, Abhi Gupta, Tommi Jaakkola, Josh Tenenbaum, Leslie Kaelbling, Akash Srivastava, and Pulkit Agrawal. Compositional foundation models for hierarchical planning. _Advances in Neural Information Processing Systems_ , 36: 22304–22325, 2023. 

- [2] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. _Advances in neural information processing systems_ , 35: 23716–23736, 2022. 

- [3] Brenna Argall, Sonia Chernova, Manuela Veloso, and Brett Browning. A survey of robot learning from demonstration. _Robotics and Autonomous Systems_ , 57(5):469– 483, 2009. ISSN 0921-8890. 

- [4] Suneel Belkhale and Dorsa Sadigh. Minivla: A better vla with a smaller footprint, 2024. URL https://github.com/ Stanford-ILIAD/openvla-mini. 

- [5] Homanga Bharadhwaj, Debidatta Dwibedi, Abhinav Gupta, Shubham Tulsiani, Carl Doersch, Ted Xiao, Dhruv Shah, Fei Xia, Dorsa Sadigh, and Sean Kirmani. Gen2act: Human video generation in novel scenarios enables generalizable robot manipulation. _arXiv preprint arXiv:2409.16283_ , 2024. 

- [6] Vineet Bhat, Yu-Hsiang Lan, Prashanth Krishnamurthy, Ramesh Karri, and Farshad Khorrami. 3d cavla: Leveraging depth and 3d context to generalize vision language action models for unseen tasks. _arXiv preprint arXiv:2505.05800_ , 2025. 

- [7] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. _π_ 0: A visionlanguage-action flow model for general robot control. _arXiv preprint arXiv:2410.24164_ , 2024. 

- [8] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. _arXiv preprint arXiv:2212.06817_ , 2022. 

- [9] Qingwen Bu, Jia Zeng, Li Chen, Yanchao Yang, Guyue Zhou, Junchi Yan, Ping Luo, Heming Cui, Yi Ma, and Hongyang Li. Closed-loop visuomotor control with generative expectation for robotic manipulation. _Advances in Neural Information Processing Systems_ , 37:139002– 139029, 2024. 

- [10] Qingwen Bu, Jisong Cai, Li Chen, Xiuqi Cui, Yan Ding, Siyuan Feng, Shenyuan Gao, Xindong He, Xuan Hu, Xu Huang, et al. Agibot world colosseo: A largescale manipulation platform for scalable and intelligent embodied systems. _arXiv preprint arXiv:2503.06669_ , 2025. 

- [11] Jun Cen, Chaohui Yu, Hangjie Yuan, Yuming Jiang, Siteng Huang, Jiayan Guo, Xin Li, Yibing Song, Hao Luo, Fan Wang, et al. Worldvla: Towards autoregressive action world model. _arXiv preprint arXiv:2506.21539_ , 2025. 

- [12] R. Qi Charles, Hao Su, Mo Kaichun, and Leonidas J. Guibas. Pointnet: Deep learning on point sets for 3d classification and segmentation. In _2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , pages 77–85, 2017. 

- [13] Xi Chen, Josip Djolonga, Piotr Padlewski, Basil Mustafa, Soravit Changpinyo, Jialin Wu, Carlos Riquelme Ruiz, Sebastian Goodman, Xiao Wang, Yi Tay, et al. Pali-x: On scaling up a multilingual vision and language model. _arXiv preprint arXiv:2305.18565_ , 2023. 

- [14] Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. _The International Journal of Robotics Research_ , 44(10-11):1684–1704, 2025. 

- [15] Michel Desmurget and Scott Grafton. Forward modeling allows feedback control for fast reaching movements. _Trends in Cognitive Sciences_ , 4(11):423–431, 2000. ISSN 1364-6613. 

- [16] Yilun Du, Sherry Yang, Bo Dai, Hanjun Dai, Ofir Nachum, Josh Tenenbaum, Dale Schuurmans, and Pieter Abbeel. Learning universal policies via text-guided video generation. _Advances in neural information processing systems_ , 36:9156–9172, 2023. 

- [17] Haiwen Feng, Junyi Zhang, Qianqian Wang, Yufei Ye, Pengcheng Yu, Michael J Black, Trevor Darrell, and Angjoo Kanazawa. St4rtrack: Simultaneous 4d reconstruction and tracking in the world. _arXiv preprint arXiv:2504.13152_ , 2025. 

- [18] Jan Feyereisl, Suha Kwak, Jeany Son, and Bohyung Han. Object localization based on structural svm using privileged information. In _Advances in Neural Information Processing Systems_ , pages 208–216, 2014. 

- [19] J. Randall Flanagan and Alan M. Wing. The role of internal models in motion planning and control: Evidence from grip force adjustments during movements of handheld loads. _Journal of Neuroscience_ , 17(4):1519–1528, 1997. ISSN 0270-6474. 

- [20] Zhi Hou, Tianyi Zhang, Yuwen Xiong, Hengjun Pu, Chengyang Zhao, Ronglei Tong, Yu Qiao, Jifeng Dai, and Yuntao Chen. Diffusion transformer policy. _arXiv preprint arXiv:2410.15959_ , 2024. 

- [21] Yucheng Hu, Yanjiang Guo, Pengchao Wang, Xiaoyu Chen, Yen-Jen Wang, Jianke Zhang, Koushil Sreenath, Chaochao Lu, and Jianyu Chen. Video prediction policy: A generalist robot policy with predictive visual representations. _arXiv preprint arXiv:2412.14803_ , 2024. 

- [22] Jiangyong Huang, Silong Yong, Xiaojian Ma, Xiongkun Linghu, Puhao Li, Yan Wang, Qing Li, Song-Chun Zhu, Baoxiong Jia, and Siyuan Huang. An embodied generalist agent in 3d world. _arXiv preprint arXiv:2311.12871_ , 2023. 

- [23] Wenlong Huang, Yu-Wei Chao, Arsalan Mousavian, Ming-Yu Liu, Dieter Fox, Kaichun Mo, and Li FeiFei. Pointworld: Scaling 3d world models for in-the-wild robotic manipulation. _arXiv preprint arXiv:2601.03782_ , 2026. 

- [24] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al. _π_ 0 _._ 5: a vision-language-action model with open-world generalization. _arXiv preprint arXiv:2504.16054_ , 2025. 

- [25] Roland Johansson. Sensory input and control of grip. _Novartis Foundation symposium_ , 218:45–59; discussion 59, 02 1998. doi: 10.1002/9780470515563.ch4. 

- [26] Siddharth Karamcheti, Suraj Nair, Ashwin Balakrishna, Percy Liang, Thomas Kollar, and Dorsa Sadigh. Prismatic vlms: Investigating the design space of visuallyconditioned language models. In _Forty-first International Conference on Machine Learning_ , 2024. 

- [27] Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, et al. Droid: A large-scale in-the-wild robot manipulation dataset. _arXiv preprint arXiv:2403.12945_ , 2024. 

- [28] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. _arXiv_ 

_preprint arXiv:2406.09246_ , 2024. 

- [29] Moo Jin Kim, Chelsea Finn, and Percy Liang. Finetuning vision-language-action models: Optimizing speed and success. _arXiv preprint arXiv:2502.19645_ , 2025. 

- [30] Chengmeng Li, Junjie Wen, Yan Peng, Yaxin Peng, Feifei Feng, and Yichen Zhu. Pointvla: Injecting the 3d world into vision-language-action models. _arXiv preprint arXiv:2503.07511_ , 2025. 

- [31] Fuhao Li, Wenxuan Song, Han Zhao, Jingbo Wang, Pengxiang Ding, Donglin Wang, Long Zeng, and Haoang Li. Spatial forcing: Implicit spatial representation alignment for vision-language-action model, 2025. URL https://arxiv.org/abs/2510.12276. 

- [32] Peiyan Li, Yixiang Chen, Hongtao Wu, Xiao Ma, Xiangnan Wu, Yan Huang, Liang Wang, Tao Kong, and Tieniu Tan. Bridgevla: Input-output alignment for efficient 3d manipulation learning with vision-language models. _arXiv preprint arXiv:2506.07961_ , 2025. 

- [33] Qixiu Li, Yaobo Liang, Zeyu Wang, Lin Luo, Xi Chen, Mozheng Liao, Fangyun Wei, Yu Deng, Sicheng Xu, Yizhong Zhang, et al. Cogact: A foundational vision-language-action model for synergizing cognition and action in robotic manipulation. _arXiv preprint arXiv:2411.19650_ , 2024. 

- [34] Xinghang Li, Minghuan Liu, Hanbo Zhang, Cunjun Yu, Jie Xu, Hongtao Wu, Chilam Cheang, Ya Jing, Weinan Zhang, Huaping Liu, et al. Vision-language foundation models as effective robot imitators. _arXiv preprint arXiv:2311.01378_ , 2023. 

- [35] Shijie Lian, Bin Yu, Xiaopeng Lin, Laurence T. Yang, Zhaolong Shen, Changti Wu, Yuzhuo Miao, Cong Huang, and Kai Chen. Langforce: Bayesian decomposition of vision language action models via latent action queries, 2026. 

- [36] Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. _Advances in Neural Information Processing Systems_ , 36:44776– 44791, 2023. 

- [37] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. _Advances in neural information processing systems_ , 36:34892–34916, 2023. 

- [38] Jiaming Liu, Hao Chen, Pengju An, Zhuoyang Liu, Renrui Zhang, Chenyang Gu, Xiaoqi Li, Ziyu Guo, Sixiang Chen, Mengzhen Liu, et al. Hybridvla: Collaborative diffusion and autoregression in a unified vision-languageaction model. _arXiv preprint arXiv:2503.10631_ , 2025. 

- [39] R.C. Miall and D.M. Wolpert. Forward models for physiological motor control. _Neural Networks_ , 9(8):1265– 1279, 1996. ISSN 0893-6080. Four Major Hypotheses in Neuroscience. 

- [40] Soroush Nasiriany, Abhiram Maddukuri, Lance Zhang, Adeet Parikh, Aaron Lo, Abhishek Joshi, Ajay Mandlekar, and Yuke Zhu. Robocasa: Large-scale simulation of everyday tasks for generalist robots. _arXiv preprint arXiv:2406.02523_ , 2024. 

- [41] Dantong Niu, Yuvan Sharma, Haoru Xue, Giscard Biamby, Junyi Zhang, Ziteng Ji, Trevor Darrell, and Roei Herzig. Pre-training auto-regressive robotic models with 4d representations. _arXiv preprint arXiv:2502.13142_ , 2025. 

- [42] Abby O’Neill, Abdul Rehman, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, Ajinkya Jain, et al. Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0. In _2024 IEEE International Conference on Robotics and Automation (ICRA)_ , pages 6892–6903. IEEE, 2024. 

- [43] Shivansh Patel, Xinchen Yin, Wenlong Huang, Shubham Garg, Hooshang Nayyeri, Li Fei-Fei, Svetlana Lazebnik, and Yunzhu Li. A real-to-sim-to-real approach to robotic manipulation with vlm-generated iterative keypoint rewards. _arXiv preprint arXiv:2502.08643_ , 2025. 

- [44] Dean A Pomerleau. Alvinn: An autonomous land vehicle in a neural network. _Advances in neural information processing systems_ , 1, 1988. 

- [45] Daniel Seita, Yufei Wang, Sarthak J Shetty, Edward Yao Li, Zackory Erickson, and David Held. Toolflownet: Robotic manipulation with tools via predicting tool flow from point clouds. In _Conference on Robot Learning_ , pages 1038–1049. PMLR, 2023. 

- [46] Yichao Shen, Fangyun Wei, Zhiying Du, Yaobo Liang, Yan Lu, Jiaolong Yang, Nanning Zheng, and Baining Guo. Videovla: Video generators can be generalizable robot manipulators. 2025. 

- [47] Junyao Shi, Joshua Smith, Jianing Qian, and Dinesh Jayaraman. Points2reward: Robotic manipulation rewards from just one video. 

- [48] Lucy Xiaoyang Shi, Brian Ichter, Michael Equi, Liyiming Ke, Karl Pertsch, Quan Vuong, James Tanner, Anna Walling, Haohuan Wang, Niccolo Fusai, et al. Hi robot: Open-ended instruction following with hierarchical vision-language-action models. _arXiv preprint arXiv:2502.19417_ , 2025. 

- [49] Lin Sun, Bin Xie, Yingfei Liu, Hao Shi, Tiancai Wang, and Jiale Cao. Geovla: Empowering 3d representations in vision-language-action models. _arXiv preprint arXiv:2508.09071_ , 2025. 

- [50] Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, et al. Octo: An open-source generalist robot policy. _arXiv preprint arXiv:2405.12213_ , 2024. 

- [51] Yang Tian, Sizhe Yang, Jia Zeng, Ping Wang, Dahua Lin, Hao Dong, and Jiangmiao Pang. Predictive inverse dynamics models are scalable learners for robotic manipulation. _arXiv preprint arXiv:2412.15109_ , 2024. 

- [52] Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A physics engine for model-based control. In _2012 IEEE/RSJ International Conference on Intelligent Robots and Systems_ , pages 5026–5033, 2012. 

- [53] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob 

Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In _Proceedings of the 31st International Conference on Neural Information Processing Systems_ , NIPS’17, page 6000–6010, Red Hook, NY, USA, 2017. Curran Associates Inc. ISBN 9781510860964. 

- [54] Homer Rich Walke, Kevin Black, Tony Z Zhao, Quan Vuong, Chongyi Zheng, Philippe Hansen-Estruch, Andre Wang He, Vivek Myers, Moo Jin Kim, Max Du, et al. Bridgedata v2: A dataset for robot learning at scale. In _Conference on Robot Learning_ , pages 1723– 1736. PMLR, 2023. 

- [55] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer, 2025. 

- [56] Chuan Wen, Xingyu Lin, John So, Kai Chen, Qi Dou, Yang Gao, and Pieter Abbeel. Any-point trajectory modeling for policy learning. _arXiv preprint arXiv:2401.00025_ , 2023. 

- [57] Junjie Wen, Yichen Zhu, Jinming Li, Minjie Zhu, Zhibin Tang, Kun Wu, Zhiyuan Xu, Ning Liu, Ran Cheng, Chaomin Shen, et al. Tinyvla: Towards fast, data-efficient vision-language-action models for robotic manipulation. _IEEE Robotics and Automation Letters_ , 2025. 

- [58] Daniel M. Wolpert, Zoubin Ghahramani, and Michael I. Jordan. An internal model for sensorimotor integration. _Science_ , 269(5232):1880–1882, 1995. 

- [59] Hongtao Wu, Ya Jing, Chilam Cheang, Guangzeng Chen, Jiafeng Xu, Xinghang Li, Minghuan Liu, Hang Li, and Tao Kong. Unleashing large-scale video generative pretraining for visual robot manipulation. _arXiv preprint arXiv:2312.13139_ , 2023. 

- [60] Xiaoyang Wu, Li Jiang, Peng-Shuai Wang, Zhijian Liu, Xihui Liu, Yu Qiao, Wanli Ouyang, Tong He, and Hengshuang Zhao. Point transformer v3: Simpler, faster, stronger. In _CVPR_ , 2024. 

- [61] Yuxi Xiao, Jianyuan Wang, Nan Xue, Nikita Karaev, Iurii Makarov, Bingyi Kang, Xin Zhu, Hujun Bao, Yujun Shen, and Xiaowei Zhou. Spatialtrackerv2: 3d point tracking made easy. In _ICCV_ , 2025. 

- [62] Mengda Xu, Zhenjia Xu, Yinghao Xu, Cheng Chi, Gordon Wetzstein, Manuela Veloso, and Shuran Song. Flow as the cross-domain manipulation interface. _arXiv preprint arXiv:2407.15208_ , 2024. 

- [63] Zhao-Heng Yin, Sherry Yang, and Pieter Abbeel. Objectcentric 3d motion field for robot learning from human videos. _arXiv preprint arXiv:2506.04227_ , 2025. 

- [64] Wentao Yuan, Jiafei Duan, Valts Blukis, Wilbert Pumacay, Ranjay Krishna, Adithyavairavan Murali, Arsalan Mousavian, and Dieter Fox. Robopoint: A visionlanguage model for spatial affordance prediction for robotics. _arXiv preprint arXiv:2406.10721_ , 2024. 

- [65] Wenyao Zhang, Hongsi Liu, Zekun Qi, Yunnan Wang, XinQiang Yu, Jiazhao Zhang, Runpei Dong, Jiawei He, He Wang, Zhizheng Zhang, et al. Dreamvla: A vision-language-action model dreamed with comprehen- 

sive world knowledge. In _The Thirty-ninth Annual Conference on Neural Information Processing Systems_ , 2025. 

- [66] Qingqing Zhao, Yao Lu, Moo Jin Kim, Zipeng Fu, Zhuoyang Zhang, Yecheng Wu, Zhaoshuo Li, Qianli Ma, Song Han, Chelsea Finn, et al. Cot-vla: Visual chainof-thought reasoning for vision-language-action models. In _Proceedings of the Computer Vision and Pattern Recognition Conference_ , pages 1702–1713, 2025. 

- [67] Haoyu Zhen, Xiaowen Qiu, Peihao Chen, Jincheng Yang, Xin Yan, Yilun Du, Yining Hong, and Chuang Gan. 3dvla: A 3d vision-language-action generative world model. _arXiv preprint arXiv:2403.09631_ , 2024. 

- [68] Ruijie Zheng, Yongyuan Liang, Shuaiyi Huang, Jianfeng Gao, Hal Daum´e III, Andrey Kolobov, Furong Huang, and Jianwei Yang. Tracevla: Visual trace prompting enhances spatial-temporal awareness for generalist robotic policies. _arXiv preprint arXiv:2412.10345_ , 2024. 

- [69] Zhongyi Zhou, Yichen Zhu, Minjie Zhu, Junjie Wen, Ning Liu, Zhiyuan Xu, Weibin Meng, Yaxin Peng, Chaomin Shen, Feifei Feng, et al. Chatvla: Unified multimodal understanding and robot control with visionlanguage-action model. In _Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing_ , pages 5377–5395, 2025. 

- [70] Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, et al. Rt-2: Vision-languageaction models transfer web knowledge to robotic control. In _Conference on Robot Learning_ , pages 2165–2183. PMLR, 2023. 

## **Pri4R: Learning World Dynamics for Vision-Language-Action Models with Privileged 4D Representation** 

### Appendix 

In this Appendix, we additionally provide, 

- **S.I. Real World Experiment Details** 

- A. Real World Setup 

- B. Real World Experiments Details 

- C. Hyperparameters and Training Details 

- **S.II. Simulation Experiment Details** 

- A. Implementation Details for Depth Map Prediction 

- B. Robocasa Full Benchmark Results 

- C. Hyperparameters and Training Details 

- **S.III. Additional Experiments** 

- A. Additional Analysis on _Pt_ 

- B. Comparison to VLAs with implicit 3D awareness 

- C. Additional Ablations on _π_ 0 _._ 5 

- **S.IV. Additional Qualitative Results** 

- A. 3D Point Track Visualization 

**–** B. Real world rollout images 

Moreover, we provide real world experiment videos in the supplementary materials. 

#### APPENDIX A 

REAL WORLD EXPERIMENT DETAILS 

#### _A. Real World Setup_ 

Our robot system overview is illustrated in Figure S1. We use an OMY robot equipped with a 6-DoF manipulator driven by high precision DYNAMIXEL-Y actuators, supporting payloads up to 3kg, along with a 1-DoF gripper. For perception, we mount an Intel RealSense D405 wrist camera and use Intel RealSense D435 cameras for third-person and top-down views. 

#### _B. Real World Experiment Details_ 

We conduct four real world manipulation tasks to evaluate the model’s spatiotemporal understanding and control. In the main paper Table VII and appendix Table **??** , the columns **Height, Spatial, Depth, and Tracking** correspond to the four tasks described below. 

**Pick-and-place over an obstacle (Height)** The instruction is _“Pick up the doll and place it on the right”_ . The robot must pick up a doll and place it to the right side of the workspace while avoiding an obstacle in between. We evaluate three obstacle settings: no obstacle, mid-height obstacle, and high obstacle (twice the mid-height). We collect 45 demonstrations and evaluate 10 trials for each setting (30 total). 

**Pick-and-place into a bin (Spatial)** The instruction is _“Pick up the doll and put it in the white bin”_ . We evaluate two conditions. Seen uses in-distribution object placements without distractors. Unseen includes either (i) an additional distractor object at a seen placement or (ii) a target object placed at an 

unseen location. We collect 45 demonstrations and evaluate 10 trials for Seen and 10 trials for Unseen. 

**Pick the farthest object (Depth)** The instruction is _“Pick up the hat farthest from the robot”_ . Two hats are placed on the table, and the robot must select and grasp the one with the larger distance from the robot base. The target identity is balanced across trials (each hat is the target in half of the trials). For Seen, both hats are placed at in-distribution locations. For Unseen, we report results for two sub-settings indicated in the table as “a/b”: a denotes the case where one of the two hats is at an unseen location, and b denotes the case where both hats are at unseen locations. We collect 48 demonstrations and evaluate 24 trials for Seen and 16 trials for Unseen. 

**Pick a moving object (Tracking)** The instruction is _“Pick up the Santa hat”_ . The target hat is relocated while the robot is approaching, requiring the robot to track the target and execute the grasp at the updated position. Seen uses in-distribution initial and relocated positions. Unseen uses an unseen relocated position (initial position remains in-distribution). OOD additionally relocates the hat twice, inducing stronger distribution shift in the target’s motion. Our dataset includes sequences with a fixed target and sequences with a single relocation. We collect 40 demonstrations and evaluate 12 trials each for Seen, Unseen, and OOD (total 36). 



<!-- Start of picture text -->
Third view  Top camera<br>camera<br>Wrist<br>camera<br>6 DoF arm<br>1 DoF gripper<br><!-- End of picture text -->

Fig. S1: **Realworld setup** 

#### _C. Hyperparameters and Training Details_ 

Table S1 and Table S2 report the hyperparameters used in our real-world training. We follow a shared training protocol across tasks (Height/Spatial/Depth/Tracking) and methods, and modify only the tracking loss weight for Pri4R. 

TABLE S1: **Hyperparameter details for real-world training.** Unless otherwise specified, OpenVLA-OFT and OpenVLA-OFT+Pri4R share the same settings. 

||**OpenVLA-OFT**|**OpenVLA-OFT + Pri4R**|
|---|---|---|
|**Shared settings**|||
|# GPUs|4 _×_|NVIDIA H200|
|learning rate (LR)||5e_−_4|
|total batch size<br>|32 <br>|(8 per GPU)<br>|
|input images<br>input image size<br>LoRA rank|3 camera images: <br>|third-person, wrist, top-down<br>224 _×_ 224<br>32|
|action chunk size<br>||10|
|use proprio (robot state)<br>use FiLM||yes<br>no|
|**Task-specific training steps**|||
|Height||60K|
|Spatial||150K|
|Depth||70K|
|Tracking||80K|
|**Method-specific setting**|||
|point track loss weight (_ω_pt)|–|1.0|



TABLE S2: **Hyperparameter details for training** _π_ **series on real-world tasks.** Unless otherwise specified, _π_ and _π_ +Pri4R share the same settings. 

||_π_0_._5<br>_π_0_._5 **+ Pri4R**|
|---|---|
|**Shared settings**||
|# GPUs<br>optimizer|4 _×_ NVIDIA H200<br>AdamW|
|input images<br>input image size|3 camera images: third-person, wrist, top-down<br>224 _×_ 224|
|learning rate scheduler|Cosine decay|
|warmup steps|10,000|
|learning rate (LR)|5e_−_5|
|total batch size<br>EMA decay|32 (8 per GPU)<br>0.999|
|**Task-specific training steps**||
|Height|40K|
|Spatial|30K|
|Depth|40K|
|Tracking|30K|
|**Method-specific setting**||
|point track loss weight (_ωpt_)|–<br>1.0|



#### APPENDIX B 

#### SIMULATION EXPERIMENT DETAILS 

#### _A. Implementation Details for Depth Map Prediction_ 

In Table III, we conduct experiments with alternative supervision targets to analyze the benefit of 3D point tracks. Specifically, to compare against spatially dense depth supervision, we predict future depth maps using a compact latent target. Because raw depth maps are high-resolution, we first train a depth variational autoencoder (VAE) and use its latent codes as supervision. This compresses the representation from 256 _×_ 256 to 32 _×_ 32 _×_ 4 (4096 dimensions), which is comparable to the dimensionality of our 3D point-track representation. Figure S2 visualizes the depth-VAE reconstructions. 

#### _B. Robocasa Full Benchmark Results_ 

We show the full results on Robocasa on Table S4. 

TABLE S3: **Simulation training details.** We report settings that differ from real-world training. 



<!-- Start of picture text -->
Dataset Method Input images # GPUs Batch size # steps<br>LIBERO OpenVLA-OFT third-view + wrist-view 8 total 64 90K<br>π 0 / π 0 . 5 third-view + wrist-view 4 total 128 30K<br>RoboCasa OpenVLA-OFT left-view + right-view + wrist-view 8 total 64 120K<br>π 0 / π 0 . 5 left-view + right-view + wrist-view 4 total 128 30K<br>Ground Truth Depths<br>Reconstructed Depths<br>T=0 T=1 T=2 T=3<br><!-- End of picture text -->

Fig. S2: **Depth reconstruction: (Top)** ground-truth depth maps from the simulator, and **(Bottom)** depth maps reconstructed by our variational autoencoder (VAE). 

#### _C. Hyperparameters and Training Details_ 

We follow the same hyperparameter and training protocol as in our real world experiments (Table S1 and Table S2) unless otherwise specified. Table S3 summarizes the simulationspecific differences, including camera inputs, compute budget, and training steps for LIBERO and RoboCasa. 

#### APPENDIX C 

#### ADDITIONAL EXPERIMENTS 

#### _A. Additional Analysis on input Pt_ 

Extending Table IV, we study a variant that removes the current point set _Pt_ entirely, from both the point track head input and the backbone. This variant still requires no 3D observations at inference time, consistent with Pri4R. However, without _Pt_ , the point track head must _generate_ future point tracks rather than predict _how a given scene evolves over time_ . As shown in Table S5, predicting 3D point tracks without _Pt_ degrades success and can underperform the baseline. This highlights that conditioning the point track head on _Pt_ is crucial for learning world dynamics and, in turn, improving performance. 

#### _B. Comparison with VLAs with implicit 3D awareness_ 

**SpatialForcing.** SpatialForcing [31] recently proposed a complementary approach for improving 3D awareness in VLAs by aligning intermediate VLA features with representations from a 3D geometric foundation model, VGGT [55]. Similar to Pri4R, this does not require explicit 3D observations as additional inputs at test time, while also endowing VLAs with additional knowledge. 

Table S6 compares SpatialForcing and Pri4R on OpenVLAOFT. While SpatialForcing yields consistent gains over the baseline, Pri4R achieves higher success rates. We attribute this gap to the nature of the supervision: SpatialForcing primarily 

TABLE S4: **Robocasa success rates by task.** We report the average success rate ( **Total** ) and per-task success rates on RoboCasa for all experiments. 

|**Method**|**Total**|**PnP**|**Doors**|**Drawers**|**Knobs**|**Lever**|**Press**|**Insert**|
|---|---|---|---|---|---|---|---|---|
|OpenVLA-OFT|0.331|0.218|0.457|0.590|0.080|0.360|0.560|0.270|
|OpenVLA-OFT + 2D tracks|0.370|0.210|0.520|0.620|0.140|0.473|0.600|0.290|
|OpenVLA-OFT + Goal point set|0.338|0.165|0.533|0.810|0.080|0.413|0.587|0.220|
|OpenVLA-OFT + Environment points|0.352|0.190|0.487|0.470|0.170|0.540|0.493|0.280|
|OpenVLA-OFT + Robot-only points|0.438|0.223|0.607|0.790|0.280|0.660|0.673|0.270|
|OpenVLA-OFT + **Ours (Pri4R)**|**0.463**|**0.230**|**0.617**|**0.800**|**0.250**|**0.667**|**0.793**|**0.340**|
|_π_0_._5|0.529|0.543|0.510|0.750|0.280|0.793|**0.600**|0.040|
|_π_0_._5 + backbone query|0.544|**0.553**|0.605|0.750|0.330|0.827|0.507|0.030|
|_π_0_._5 + query attend action|0.548|0.535|0.605|0.840|**0.350**|0.767|0.560|0.050|
|_π_0_._5 + **Ours (Pri4R)**|**0.570**|0.520|**0.685**|**0.890**|0.330|**0.867**|0.547|**0.050**|



TABLE S5: **Analysis of Point Set** _Pt_ **input.** We present the average success rates on RoboCasa. 

|**Method**|**SR** _↑_|∆|
|---|---|---|
|OpenVLA-OFT [29]|33.1|–|
|+ 3D point sequence w/o input pointcloud|28.7|-4.4|
|**+ Ours (3D point track)**|**46.3**|**+13.2**|



TABLE S6: **Comparison with SpatialForcing.** We compare Pri4R with a recent SOTA method, SpatialForcing. 

||Average|Spatial|Object|Goal|Long|
|---|---|---|---|---|---|
|OpenVLA-OFT|92.7|92.8|98.4|96.4|83.0|
|+SpatialForcing|94.2|**96.8**|**99.0**|96.2|84.8|
|+**Pri4R**|**95.0**|93.2|96.6|**96.8**|**93.2**|



injects static 3D structure from an off-the-shelf model, whereas Pri4R leverages temporally dense 4D supervision through 3D point tracks, explicitly encouraging the policy to model interaction dynamics. 

#### _C. Additional Ablations_ 

TABLE S7: **Ablations on tracking loss weight for** _π_ 0 _._ 5 **.** We present the average success rate (SR) across 24 task suites on RoboCasa, averaged over 50 trials per suite. 

|weight|Avg|PnP|Doors|Drawers|Turn|Twist|Press|Insert|
|---|---|---|---|---|---|---|---|---|
|0.1|54.7|54.3|62.5|90.0|28.0|73.3|55.3|3.0|
|1.0|57.0|52.0|68.5|89.0|33.0|86.7|54.7|5.0|
|10.0|50.7|47.0|54.5|77.0|35.0|82.0|47.3|5.0|



TABLE S8: **Ablations on number of points for 3D track prediction for** _π_ 0 _._ 5 **.** We present the average success rate (SR) across 24 task suites on RoboCasa, averaged over 50 trials per suite. 

|number|Avg|PnP|Doors|Drawers|Turn|Twist|Press|Insert|
|---|---|---|---|---|---|---|---|---|
|256|48.8|49.3|55.5|77.0|28.0|73.3|41.3|1.0|
|512|53.9|48.8|60.0|87.0|27.0|81.3|59.3|7.0|
|1024|57.0|52.0|68.5|89.0|33.0|86.7|54.7|5.0|



#### _B. Real world rollout images_ 

In Fig. S4, we show Pri4R performing real world tasks. Additional qualitative results are provided in the supplementary videos. 

**Ablation on** _ω_ pt **.** Table S7 ablates the weight of the auxiliary point track loss, _ω_ pt. We find that _ω_ pt = 1 _._ 0 performs best among the tested values. Notably, because our point track supervision is formulated as displacements (analogous to the action displacement targets), the Pri4R is not highly sensitive to _ω_ pt and requires minimal tuning. **Ablation on** _Np_ **.** Table S8 ablates the number of tracked points, _Np_ . Although 3D point tracks are intentionally sparse to serve as an efficient supervision signal, using too few points (e.g., 256 or 512) degrades performance compared to our default choice of 1024 points. This suggests that a moderate point density is necessary to sufficiently capture interactionrelevant geometry. 

#### APPENDIX D 

#### ADDITIONAL QUALITATIVE RESULTS 

#### _A. 3D point track visualization_ 

In Fig. S3, we visualize the 3D point track data extracted from simulation. 



<!-- Start of picture text -->
T=0 T=40 T=80 T=120 T=160 T=200 T=240<br>open the microwave door<br>open the right drawer<br><!-- End of picture text -->

Fig. S3: **Visualization of point track data.** Purple lines denote 3D point tracks. Points are sub-sampled for clarity. 



<!-- Start of picture text -->
0s 3s 6s 9s 12s 15s 18s<br>0s 4s 8s 12s 14s 18s 20s<br>Pick the farthest object<br>PnP into a bin<br><!-- End of picture text -->

Fig. S4: **Pri4R on real-world tasks.** Additional qualitative results are provided in the supplementary videos. 

