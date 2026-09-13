# **ZETA: A Controlled Study of Zero-Shot Cross-Embodiment VLA Transfer for Tabletop Manipulation** 

**Mi Yan**<sup>1</sup><sup>_,_2</sup><sup>_∗_</sup> **Wenhao Zhang**<sup>1</sup><sup>_,_3</sup><sup>_∗_</sup> **Zhiqi Zhang**<sup>1</sup><sup>_,_3</sup><sup>_∗_</sup> **Yu Peng**<sup>1</sup><sup>_,_4</sup><sup>_∗_</sup> **Tangxinyu Wang**<sup>1</sup><sup>_,_3</sup><sup>_∗_</sup> **Lingfei Zhai**<sup>1</sup><sup>_,_3</sup> **Jiayi Su**<sup>1</sup><sup>_,_6</sup> **Shengliang Deng**<sup>1</sup><sup>_,_5</sup> **Lin Peng**<sup>1</sup><sup>_,_7</sup> **Yaowei Liu**<sup>1</sup><sup>_,_3</sup> **Yuxing Chen**<sup>1</sup><sup>_,_2</sup> **Zhiyuan Wei**<sup>1</sup><sup>_,_3</sup> **Jilong Wang**<sup>1</sup> **Jiayi Chen**<sup>1</sup><sup>_,_2</sup> **Jiangran Lyu**<sup>1</sup><sup>_,_2</sup> **Zhizheng Zhang**<sup>1</sup><sup>_†_</sup> **He Wang**<sup>1</sup><sup>_,_2</sup><sup>_†_</sup> 

1Galbot 2CFCS, School of CS, Peking University 

3Peking University 4Renmin University of China 5The University of Hong Kong 6Xiamen University Malaysia 7Beihang University 



<!-- Start of picture text -->
Pretrain Data Post-Train Data Evaluation Embodiment Shifts<br>Pretrain Exposed Zero-Shot(RQ4) Simulation Tasks Appearance-only Changes (APP) Gripper-only Changes (GRP) Arm-only Changes (ARM) Full Changes (FULL)<br>Strict Zero-Shot(RQ1-3)<br>Source Robots Target Robot<br>Controlled Data Distribution<br>Simulation<br>Embodiment<br>Simulation Setting<br>Real-World Tasks<br>Task Environment Embodiment<br>VLM Action<br>Real-World<br>Controlled Architecture Controlled Quantity Embodiment Real-World Setting<br>RQ1 Representation RQ2 Embodiment  RQ3 Co-Train Tasks RQ4 Embodiment<br>Abs. Delta Diversity Language ActionGoal Pose “0.2, 0.5, 0.1, 0.6, 0.7, 0.1, 1”“Move backward 3cm,…” Exposure<br>x Bounding Box “100, 131, 185, 20”<br>z y State SR<br>y z World EEF … Diversity VLM Action Source RobotsPretrain DataTarget Robot<br>x Action 1 Robot 2 Robots … N Robots + 5% Target Robot + 13.4% Performance<br><!-- End of picture text -->

Figure 1: Overview of our controlled study. We distinguish _strict zero-shot transfer_ , where the target robot is absent from all training data, from _pretrain-exposed zero-shot transfer_ . Under a controlled benchmark, we conduct fine-grained analyses across four embodiment shift categories, covering four research questions (RQ) on state-action representations, source-embodiment diversity, auxiliary cotraining, and target-embodiment exposure. 

**Abstract:** Zero-shot generalization to unseen embodiments is important for generalizable vision-language-action (VLA) models as robot hardware evolves and task-specific data collection remains costly. However, a systematic understanding of this problem remains limited, in part because the literature lacks a unified zero-shot transfer definition and controlled evaluation settings that isolate embodiment changes from differences in tasks, scenes, or protocols. To address this gap, we first distinguish _strict zero-shot transfer_ , where the target embodiment is absent from all training data, from _pretrain-exposed zero-shot transfer_ , where it appears only during pretraining. We then introduce a controlled benchmark spanning 14 held-out target embodiments across simulation and real-world validation. Within this framework, we conduct a controlled analysis of four factors: state-action representations, pretraining embodiment diversity, auxiliary cotraining objectives, and target-embodiment exposure. Experimental results show that local end-effector (EEF) state-action representations, the source embodiment diversity, and auxiliary co-training improve cross-embodiment transfer by around 

> _∗_ Equal contribution. 

> _†_ Corresponding authors. Correspondence to zhangzz@galbot.com, hewang@pku.edu.cn. 

15, 18, and 7 percentage points, respectively. We further find that adding only 5% target-embodiment data during pretraining improves average target-embodiment progress by 13.4 percentage points, showing that strict and pretrain-exposed zeroshot transfer are distinct and should be reported separately. Together, these findings provide practical guidance for evaluating and improving cross-embodiment VLA transfer in stationary tabletop manipulation with two-finger grippers, while motivating future investigation of broader settings including mobile-base control, dexterous hands, and long-horizon tasks. 

**Keywords:** cross-embodiment, vision-language-action, zero-shot transfer 

## **1 Introduction** 

Zero-shot cross-embodiment transfer is central to generalizable vision-language-action (VLA) models: policies trained on source robots are expected to generalize to new target robots without targettask demonstrations. As robot hardware continues to evolve, such transfer becomes increasingly important for avoiding costly task-specific data collection on every new platform. 

Despite growing interest, a systematic understanding of this problem remains limited. First, “zeroshot transfer” is used inconsistently across studies to denote two protocols with different generalization difficulty: some keep the target embodiment absent from all training, while others include it in pretraining but exclude it from task-specific post-training. Another challenge is that existing evaluations often couple embodiment shift together with differences in tasks, environments, cameras, data quantities, and protocols, making it difficult to attribute failures to embodiment mismatch itself. In this work, we focus on stationary tabletop manipulation with two-finger grippers, allowing us to study embodiment changes in a controlled setting while holding the broader manipulation regime fixed. 

To address the first challenge, we treat target inclusion during pretraining as an explicit experimental variable. We define _strict zero-shot transfer_ as the setting where the target embodiment is absent from all training, and _pretrain-exposed zero-shot transfer_ as the setting where it appears only during pretraining. These settings reflect two common deployment scenarios: newly designed robots with no collected data (e.g., a team iterates on its hardware and has data only from earlier robot versions), and publicly available robots with no private post-training data, respectively. 

To address the second challenge, we construct a controlled simulation benchmark over seven heldout target embodiments within this tabletop manipulation setting, matching the backbone, downstream source data, and evaluation setting, with experiment-specific budget controls designed to characterize each factor. We group embodiment shifts into appearance-only, gripper-only, arm-only, and full-embodiment changes to provide a more fine-grained analysis, with each simulation comparison evaluated on 6,300 rollouts per model. We further complement this simulation study with real-world validation under matched categories. 

Within this framework, our work studies four research questions (RQ) covering the major design axes of cross-embodiment transfer. RQ1: Which state-action representation best supports strict zero-shot transfer to unseen embodiments? RQ2: Under different pretraining budget controls, how does varying the number of source embodiments within a procedurally generated pool affect strict transfer? RQ3: Do co-training objectives improve transfer beyond imitation learning alone? RQ4: How much does seeing the target embodiment during pretraining change the difficulty of zero-shot transfer? RQ1–RQ3 are evaluated under strict zero-shot transfer, while RQ4 compares the strict and pretrain-exposed settings. 

The resulting simulation experiments show that all four factors substantially affect crossembodiment transfer. Local EEF-centered state-action representations improve strict transfer by around 15 percentage points on average. Under a fixed 640K-trajectory pretraining budget, the 512source setting within our procedurally generated Franka-style source pool outperforms the singlesource model by around 18 percentage points; auxiliary co-training further raises average progress 

2 

from 75.7% to 82.3%. Finally, adding only 5% target-embodiment data during pretraining improves average target-embodiment progress by 13.4 percentage points, showing that strict zero-shot transfer and pretrain-exposed zero-shot transfer should be reported separately. For RQ1 and RQ2, real-world validation further shows the same overall trends. 

In summary, we contribute: (i) **Protocol clarification** . We separate strict and pretrain-exposed zero-shot cross-embodiment transfer and argue that they should be reported separately; (ii) **Controlled benchmark** . We introduce a benchmark that isolates embodiment mismatch across four shift categories; (iii) **Factorized study** . We vary state-action representations, source-embodiment diversity, auxiliary co-training, and target-embodiment exposure to quantify their effects; and (iv) **Practical guidance** . We derive recommendations for evaluating and improving VLA transfer within stationary tabletop manipulation with two-finger grippers, favoring local EEF-frame representations, controlled source-diversity evaluation, and auxiliary co-training. 

## **2 Related Work** 

### **2.1 Data Composition in Multi-Embodiment Pretraining** 

Large robot foundation models increasingly rely on pretraining over heterogeneous robot data. Federated multi-embodiment corpora [1] pool demonstrations across institutions and robot types, while follow-on datasets expand coverage with in-the-wild teleoperation [2], skill-rich manipulation [3, 4], standardized multi-embodiment benchmarks [5], large-scale real-world dual-arm platforms [6], unified cross-robot collections [7], and high-fidelity synthetic pretraining data [8]. Together, these resources show that pooling demonstrations across robots can improve downstream learning, and subsequent work has scaled generalist vision-language-action policies across heterogeneous platforms for downstream adaptation and open-world manipulation [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]. Related systems further extend this trend through plug-in action experts [24], diffusionbased foundation policies [25, 26], human-centric cross-embodiment learning [27], multi-domain cross-embodied policies [28], multi-embodiment locomotion [29, 30], and latent-action learning from non-robot or weakly labeled video sources [31, 32]. Empirical studies on robotic data diversity [33] and embodiment scaling [34] further suggest that the composition of pretraining data can strongly affect transfer. However, these works often change data scale, task distribution, embodiment coverage, and architecture simultaneously. In contrast, our study treats pretraining composition as a controlled variable, examining procedural source-embodiment diversity under complementary fixedtotal and fixed-per-embodiment budget controls, and target-embodiment exposure under a fixed total pretraining budget. 

### **2.2 Cross-Embodiment Policy Design: Representations, Interfaces, and Supervision** 

Orthogonal to data composition, cross-embodiment transfer depends on how a policy represents and maps behavior across robots. One line of work learns shared skill or action spaces from crossembodiment demonstrations, human videos, image-space motion tracks, universal action codes, and task-centric latent actions [35, 36, 37, 38, 39, 40, 41, 32, 42, 43, 44, 45, 46]. Another line conditions a shared policy backbone on embodiment-specific information through dedicated interfaces or adaptation modules, including heterogeneous stems and action heads, soft prompts, unified control interfaces, latent guidance, shared hardware tools, and visual robot or viewpoint augmentation [47, 21, 48, 49, 50, 51, 52, 53, 54, 55]. Beyond architectures and action spaces, auxiliary supervision can provide transferable semantic, spatial, or motion-level structure through languageaction prediction, broader co-training objectives, chain-of-thought reasoning, and spatial-temporal grounding [56, 57, 58, 59, 60]. Recent work also revisits whether explicit proprioceptive state is necessary for visuomotor policies [61]. These methods motivate our focus on state-action representations and auxiliary co-training, but prior evaluations often combine the policy design with different datasets and protocols. We instead compare representation and supervision choices under matched architecture, data, and evaluation settings. 

### **2.3 Zero-Shot Cross-Embodiment Evaluation and Benchmarks** 

Several benchmarks and empirical studies evaluate how robot policies generalize across embodiments. Pushing the Limits of Cross-Embodiment Learning studies manipulation and navigation 

3 

Table 1: Classification of representative prior cross-embodiment manipulation evaluations described as zero-shot or out-of-the-box in the literature, reclassified according to target-embodiment exposure under our protocol taxonomy. 

|**Work**|**Target embodiment**<br>**in pretraining?**|**Target embodiment**<br>**in post-training?**|**Protocol**|
|---|---|---|---|
|LAP [56]|✗|✗|**Strict**|
|Cloak [64]|✗|✗|**Strict**|
|RDT2 [25]|✗|✗|**Strict**|
|Octo [11]|✓|✗|**Pretrain-exposed**|
|_π_0_._7 [65]|✓|✗|**Pretrain-exposed**|
|Qwen-RobotManip [66]|✓|✗|**Pretrain-exposed**|
|OpenVLA [12]|✓|✗|**Pretrain-exposed**|
|Gemini Robotics 1.5 [67]<sup>_∗_</sup><br>|✓|✗|**Pretrain-exposed**|
|Being-H0.5 [27]<sup>_†_</sup>|✓|✓|**Unseen task–emb. pair**|



> _∗_ Gemini Robotics 1.5 reports zero-shot transfer to tasks observed only on another embodiment, while the evaluated target embodiments are already included in its multi-embodiment pretraining and receive no robotspecific post-training. 

> _†_ Being-H0.5 reports unseen task–embodiment pairs as zero-shot even though the target embodiment participates in post-training, so the evaluation falls outside both strict and pretrain-exposed protocols. 

transfer [62]; RoboMIND provides multi-embodiment manipulation data [5]; AnyBody introduces a benchmark suite for cross-embodiment manipulation [63]; and recent diversity [33] and scalinglaw studies [34] analyze how embodiment coverage affects scalable robotic learning. These efforts broaden the evaluation landscape, but zero-shot transfer is still not always defined consistently: in some settings the target robot is absent from all training, while in others it appears during pretraining but not during task-specific post-training. Moreover, embodiment changes can be entangled with task, scene, camera, data-budget, or protocol differences. Our benchmark complements prior work by separating strict zero-shot transfer from pretrain-exposed zero-shot transfer and by evaluating appearance-only, gripper-only, arm-only, and full-embodiment shifts under controlled tasks, environments, cameras, data quantity, and training protocol. 

## **3 Problem Formulation and Preliminaries** 

**Notation.** Let _E_ denote a set of embodiments and _T_ denote a set of tasks. An embodiment _e ∈E_ specifies robot-dependent factors such as appearance, morphology and kinematics. For an embodiment–task pair ( _e, t_ ), let _D_ ( _e, t_ ) denote the corresponding demonstration dataset. For sets _E_<sup>_′_</sup> _⊆E_ and _T_<sup>_′_</sup> _⊆T_ , we write _D_ ( _E_<sup>_′_</sup> _, T_<sup>_′_</sup> ) =<sup>�</sup> ( _e,t_ ) _∈E_<sup>_′_</sup> _×T_<sup>_′ D_(</sup><sup>_e, t_), with singleton sets omitted when</sup> clear. Subscripts pre and post denote pretraining and post-training, respectively. 

**Transfer Setup.** We study supervised imitation learning under a two-stage training protocol. A policy is first pretrained on _D_ pre, whose embodiment and task composition depends on the research question. Let _E_ pre denote the embodiments appearing in _D_ pre. It is then post-trained on downstream tasks using only a source embodiment _es_ , i.e., _D_ post = _D_ ( _es, T_ post). Evaluation is performed on a target embodiment _et̸_ = _es_ for the same downstream tasks _T_ post. 

**Strict and Pretrain-Exposed Zero-Shot Protocols.** We distinguish two zero-shot protocols according to target-embodiment exposure during pretraining. In _strict zero-shot transfer_ , the target embodiment is absent from all training data: _et ∈E/_ pre. In _pretrain-exposed zero-shot transfer_ , _et ∈E_ pre but _et_ is absent from post-training. Table 1 applies this distinction to representative prior cross-embodiment evaluations. As the table shows, evaluations described as “zero-shot” or “outof-the-box” in prior work span different target-embodiment exposure protocols; some other settings even evaluate unseen task–embodiment pairs rather than unseen target embodiments. Distinguishing these settings is therefore necessary for interpreting and comparing reported “zero-shot transfer” results. 

**Embodiment Shift Taxonomy.** To provide a more fine-grained analysis of cross-embodiment transfer, we group embodiment shifts into four categories. _Appearance-only_ shifts change visual appearance while preserving morphology and kinematics. _Gripper-only_ shifts alter end-effector geometry while keeping the arm fixed. _Arm-only_ shifts change arm morphology or kinematics while preserv- 

4 

ing the end effector. _Full-embodiment_ shifts change both arm/platform and end-effector factors. Sec. 4 instantiates these categories in simulation and real-world settings, with additional details provided in the supplementary material. 

## **4 Study Design** 

### **4.1 Representative State-Action Representations** 

Existing policies often use either joint-space or Cartesian EEF interfaces. Because strict zero-shot transfer evaluates on target robots absent from all training data, joint-indexed representations may not share dimensionality, limits, or kinematic semantics across robots. We therefore focus on fixeddimensional Cartesian end-effector representations. 

As summarized in Table 2, we comTable 2: State and action representations studied in RQ1. pare two action frames and two state **Action representation** encodings. We use _b_ to denote the World-Delta action _{V_ ( _Tb,et′_ +1 ) _−V_ ( _Tb,et′_ ) _}t_<sup>_<u>t′</u>_</sup><sup><u>+</u></sup> =<sup>_<u>K</u>_</sup> _t_<sup>_<u>−</u>_</sup><sup><u>1</u></sup> robot base frame, _et_ to denote the EEF-Delta action _{V_ ( _Tb_<sup>_−_</sup> _<u>,e</u>_<sup>1</sup> _<u>t′</u>_<sup>_· Tb,e_</sup> _t_<sup>_′_</sup> <u>+1</u><sup>) =</sup><sup>_V_(</sup><sup>_Te_</sup> _t_<sup>_′ ,e_</sup> _t_<sup>_′_</sup> <u>+1</u><sup>)</sup><sup>_}_</sup> _t_<sup>_t′_+</sup> =<sup>_K_</sup> _t_<sup>_−_1</sup> gripper frame at time _t_ , _Tx,y_ to denote **State representation** the transform from frame _x_ to frame Abs. EEF state _{V_ ( _Tb,et−i_ ) _}i_<sup>_<u>h</u>_</sup> =0 EEF-Delta state _<u>{V</u>_ <u>(</u> _Tb,e_<sup>_−_1</sup> _<u>t</u>_<sup>_· Tb,e_</sup> _t−i_<sup><u>) =</u></sup><sup>_V_</sup><sup><u>(</u></sup><sup>_Tet,e_</sup> _t−i_<sup><u>)</u></sup><sup>_<u>}</u>_</sup> _i_<sup>_h_</sup> =0 _y_ , and _V_ ( _T_ ) to denote the vectoriza- _R p_ tion of a transform _T_ . For _T_ = 0 1 , we define _V_ ( _T_ ) = [ _p,_ Euler( _R_ )]. World-Delta and EEF� <u>�</u> 

Delta actions differ in whether the next motion is expressed in the robot base frame or in the current gripper frame. EEF-centered actions reduce dependence on embodiment-specific base definitions and better match the local motion semantics observed from a wrist camera. 

A similar EEF-centered principle also applies to state representations. Recent state-free policies [61] remove explicit end-effector pose history and report improved cross-embodiment transfer. Under the notation in Table 2, this design can be viewed as a degenerate EEF-Delta state: when _h_ = 0, the history contains only _Tet,et_ = _I_ , a constant identity transform that carries no state information, so state-free policies can be viewed as a degenerate EEF-Delta state. 

### **4.2 Controlled Data Construction** 

We clearly separate pretraining from post-training and impose _T_ pre _∩ T_ post = ∅, so downstream source-to-target ( _es → et_ ) embodiment transfer gap can be measured under controlled task exposure. Unless otherwise specified, our primary comparisons use a fixed pretraining budget of 640K trajectories and post-train only on source-embodiment downstream data: 40K trajectories per task in simulation and 50 demonstrations per task in the real world. RQ2 additionally includes a complementary fixedper-embodiment-budget sweep, in which the total pretraining budget varies with the number of source embodiments. 



Figure 2: Procedurally generated embodiment pool for pretraining. 

To control non-embodiment factors, we use a simulator to construct a Figure 2: Procedurally large-scale pretraining dataset with controlled data distributions. Folgenerated embodiment pool for pretraining. lowing [30, 34], we construct 512 procedurally generated Franka-family source embodiments by introducing geometric and visual variation. Fig. 2 shows representative samples from this procedurally generated source-embodiment pool. The seven commercially used test robots are imported as external held-out targets and excluded from this source pool, ensuring that strict zero-shot evaluations use target embodiments absent from all training data. Inspired by prior work [20, 8, 68, 69] that perform direct sim-to-real transfer, we apply domain randomization to visual properties (camera poses, lighting, background, and layout) and physical properties (mass and friction) under simple quasi-static assumptions, reducing the impact of sim-to-real gaps on our experimental analysis. We further complement the simulation evaluation with real-world validation using real post-training data and real-world trials. Details are provided in the Appendix. 

5 

### **4.3 Auxiliary Co-training Objectives** 

Motivated by prior VLA work showing that auxiliary supervision adds transferable semantic supervision beyond imitation learning [10, 70, 14, 71, 72, 73, 56, 58, 57, 59, 60], we include representative tasks as controlled study variants rather than default components of all models. LAP predicts structured language-action descriptions, exposing policies to low-level motion semantics in the VLM text space. Subgoal prediction supervises task progress with a next-step goal representation, emphasizing transferable spatial and state-change structure. Task-conditioned bounding-box prediction grounds instructions in object-centric image regions, testing whether explicit spatial grounding improves transfer. When enabled, auxiliary objectives are applied during both pretraining and post-training. 



<!-- Start of picture text -->
Source Robot APP GRP ARM FULL<br>Simulation Setting<br>Source Robot APP GRP ARM FULL<br>Real-World Setting<br>Aligned Main Camera S1 S2 S3 S1: Serve the sausage in a paper box.<br>S2: Stack the bowl.<br>Aligned Environment<br>S3: Stack the {color} cube on<br>the {color} cube.<br>Simulation Tasks<br>Aligned Wrist Camera R1 R2<br>R1: Water the flower.<br>Aligned Task R2: Open the fryer.<br>Source Robot Target Robot Real-World Tasks Instructions<br><!-- End of picture text -->

Figure 3: Evaluation setup for controlled cross-embodiment transfer. We evaluate seven held-out target embodiments across four shift categories in simulation and the real world, carefully matching post-training and test scenes to align non-embodiment factors (cameras, environments, and tasks). 

### **4.4 Embodiment Shift Categories** 

We focus on common two-finger-gripper embodiments because they cover substantial appearance, morphology, and kinematic variation while keeping the end-effector interface comparable for indepth analysis. We instantiate the shift taxonomy from Sec. 3; additional embodiment details are provided in the supplementary material. Fig. 3 summarizes the held-out target embodiments and their grouping into these four shift categories. In simulation, we use Franka as the source robot; appearance-only shifts modify only the Franka texture (Franka with logo and Franka with green fingers). These target appearances are near the source visual distribution but remain held out from pretraining, with their distinguishing colors lying outside the source palette primarily along the saturation dimension (Appendix D). Gripper-only shifts replace the end-effector geometry while keeping the Franka arm (Franka with UMI gripper). Arm-only shifts replace the arm while keeping a Franka-style hand (UR5e and Google Robot with a Franka hand), and full-embodiment shifts replace all configuration (UR5e+UMI and GoogleRobot). The real-world setup mirrors these categories using Franka-Robotiq as the source embodiment and matched hardware changes for each shift type. 

6 

Table 3: RQ1 zero-shot simulation results: EEF-Delta states improve arm-only and full-embodiment transfer, while EEF-Delta actions yield better average score. 

|**State**|**Action**|**es**|**APP**|**GRP**|**ARM**|**FULL**|**Average**|
|---|---|---|---|---|---|---|---|
|Abs. EEF|World-Delta|87.7_±_2.1|86.3_±_1.3|82.6_±_1.7|38.5_±_0.8|33.9_±_2.3|60.3_±_1.1|
|Abs. EEF|EEF-Delta|**92.6**_±_1.0|**91.8**_±_1.1|**88.2**_±_2.6|39.9_±_0.5|38.6_±_0.8|64.6_±_0.4|
|EEF-Delta|World-Delta|87.8_±_1.4|88.0_±_0.9|77.2_±_1.0|69.8_±_1.9|58.4_±_1.3|73.4_±_0.7|
|EEF-Delta|EEF-Delta|91.5_±_0.6|90.0_±_1.1|78.3_±_2.4|**74.3**_±_0.9|**60.4**_±_0.9|**75.7**_±_1.3|



## **5 Controlled Experiments** 

We organize the experiments around four controlled comparisons. For each research question, we first define the variable under study and the fixed conditions, then report the corresponding transfer results across embodiment-shift categories. 

### **5.1 Shared Experimental Setup** 

**Evaluation protocol.** To isolate embodiment shift, we align non-embodiment factors between evaluation and post-training as closely as possible, and we keep the initial evaluation scene consistent across all models. For each downstream task, the test objects, initial-pose sampling region (30 cm _×_ 40 cm), background, table layout, and lighting are kept consistent with the corresponding post-training setting. Both simulation and real-world evaluation use two visual observations: one wrist-mounted view and one third-person view. We further align these camera poses so that, under a canonical robot pose, the image-plane gripper distribution remains close to the post-training distribution. Additional calibration and setup details are provided in the supplementary material. 

**Model backbone and training.** We instantiate _πθ_ with a representative VLA backbone following _π_ 0 _._ 5 [14]. Each model follows the same two-stage protocol: pretraining on _D_ pre followed by post-training on _D_ post. All comparisons keep the backbone, input configuration, action horizon, losses, data budget, and optimization schedule fixed unless explicitly stated; detailed backbone and optimization hyperparameters are provided in the supplementary material. 

**Metrics and evaluation scale.** We score each rollout by task progress in [0 _,_ 100], with 100% denoting full completion and partial credit assigned to necessary subgoals, such as contacting the air-fryer handle before opening. All tables and plots report mean task progress as a percentage. Simulation is our primary evaluation platform because it supports controlled, high-throughput comparisons across tasks, embodiments, and repeated trials. For each model, we evaluate 3 downstream tasks on 7 test embodiments, with 100 rollouts per task–embodiment pair over 3 independent training runs, yielding 3 _×_ 7 _×_ 100 _×_ 3 = 6300 simulation rollouts per model; we report mean task progress _±_ standard deviation across runs. Each shift-category score aggregates all embodiments in that category and all tasks. Real-world evaluation serves as an external-validity check: because physical trials require hardware setup, resets, and safety checks, we evaluate 2 downstream tasks on 7 test embodiments with 10 trials per task–embodiment pair, yielding 2 _×_ 7 _×_ 10 = 140 real-world rollouts per model. 

### **5.2 RQ1: How Do State-Action Representations Affect Transfer?** 

RQ1 evaluates how the state-action representation defined in Sec. 4.1 affects strict zero-shot transfer. Actions can be expressed either as **World-Delta** or **EEF-Delta** , while state can be expressed either as **Abs. EEF** or as **EEF-Delta** . Their pairwise combinations yield four state-action representations. We evaluate these four variants using the pretraining data of the full set of 512 source embodiments and the same downstream post-training data. While several embodied foundation models use mixed-representation to maximize the utilization of all dataset, we exclude it from this controlled comparison to isolate the effect of each state-action design choice. 

|Table 3 shows that EEF repre-<br>|Table 4:|Real-world|results|for R|Q1. Ea|ch ent|ry aver|ages task|
|---|---|---|---|---|---|---|---|---|
|sentations improve strict zero-<br>shot transfer. Switching action|progress o<br>**State**|n_open the_<br>**Action**|_fryer_a<br>**es**|nd_wat_<br>**APP**|_er the_<br>**GRP**|_flower_.<br>**ARM**|**FULL**|**Average**|
|from world to eef raises the<br>f603%|Abs. EEF<br>Abs. EEF|World-Delta<br>EEF-Delta|87.5<br>90.0|87.5<br>90.0|67.5<br>80.0|47.5<br>60.0|21.3<br>16.3|56.0<br>61.6|
|progress score rom . to|EEF-Delta|World-Delta|85.0|77.5|60.0|62.5|43.1|60.8|
|64.6% with absolute state. Re-|EEF-Delta|EEF-Delta|**100.0**|**97.5**|**92.5**|**87.5**|**81.9**|**89.9**|



7 

placing Abs. EEF state with EEF-Delta raises arm-only transfer from 38.5–39.9% to 69.8–74.3% and full-embodiment transfer from 33.9–38.6% to 58.4–60.4%, suggesting that local state preserve more transferable motion structure. Real-world validation in Table 4 follows the same pattern: EEFDelta state with EEF-Delta actions achieves the best average score, improving from 56.0–61.6% for the other representations to 89.9%. We therefore use this representation for RQ2 and RQ3. 

### **5.3 RQ2: How Does Source-Embodiment Diversity Affect Transfer Under Different Budget Controls?** 

RQ2 studies procedural source-embodiment diversity under two complementary budget controls. Our source pool consists of procedurally generated Franka-style embodiments, as described in Appendix D. In the primary comparison, we keep the total pretraining budget fixed at 640K pickand-place trajectories while varying the number of source embodiments. Consequently, the perembodiment data decreases from 640K trajectories with a single source embodiment to 1,250 trajectories per embodiment with all 512 sources. To complement this fixed-total-budget comparison, we conduct a second sweep that fixes the per-embodiment trajectory budget while varying the source pool over 32, 128, and 512 embodiments. In this setting, each source embodiment contributes the same amount of pretraining data, so the total pretraining budget grows with _|E_ pre _|_ while all other factors remain unchanged. All RQ2 models use the best-average state-action representation from RQ1, with downstream post-training and evaluation held fixed. 

Fig. 5(a) shows that the 512-source model outperforms the single-source model, with the largest gains on gripper-only, arm-only, and full-embodiment shifts. Attention visualizations further show that the 512-source model concentrates more strongly on task-relevant objects and the gripper than the single-source model, suggesting stronger task-relevant visual grounding. A complementary UMAP analysis in Fig. 4 projects mean-pooled last-layer VLM features from 3,600 frames sampled across 1,200 trajectories of two unseen target embodiments, UR5eUMI and GoogleRobot, using cosine distance. The 512-source model shows qualitatively greater overlap between the two embodiments than the single-source model, providing preliminary evidence of stronger cross-embodiment feature alignment. 



Figure 4: UMAP of mean-pooled last-layer VLM features from heldout UR5eUMI and GoogleRobot observations. The 512-source model exhibits greater overlap between the two target embodiments than the singlesource model. 

In real-world validation, Fig. 5(b) reveals a non-monotonic intermediate regime: with 8 sources, arm-only and full-embodiment transfer falls below the single-source setting before recovering at 512 sources. This pattern may reflect a fixed-budget trade-off in which increasing source coverage reduces per-embodiment data density before the source pool becomes sufficiently broad to support robust cross-embodiment transfer. 

The complementary fixed-per-embodiment sweep in Fig. 6 shows increasing average progress as the source pool grows from 32 to 128 to 512 embodiments in simulation, with the same overall trend in real-world evaluation. 

Together, the two budget-control settings show that broader source coverage within our procedurally generated Franka-style embodiment pool can improve transfer, while the measured effect depends on how the pretraining budget is allocated. Under a fixed total budget, increasing the source count trades off against per-embodiment data density and produces a non-monotonic real-world trend; under a fixed per-embodiment budget, performance improves over the evaluated 32–512 source range. These results therefore support treating source-embodiment diversity and data-budget allocation as coupled design variables rather than assuming a universal monotonic benefit from increasing the number of source embodiments. 

### **5.4 RQ3: How Do Co-training Tasks Affect Transfer?** 

RQ3 evaluates whether the auxiliary co-training objectives defined in Sec. 4.3 further improve strict zero-shot transfer after using the best-average state-action representation from RQ1 and the full 512- 

8 



<!-- Start of picture text -->
1 Embodiment 512 Embodiments<br>(a) Simulation transfer (b) Real-world transfer (c) Attention visualization<br>UR5eUMI<br>Google Robot<br><!-- End of picture text -->

Figure 5: RQ2: effect of source-embodiment diversity under a fixed pretraining budget. The 512source setting improves over the single-source setting in both simulation (a) and the real world (b). Embodiment-diverse pretraining also shifts attention from diffuse robot–scene regions toward taskrelevant objects and the gripper. 







<!-- Start of picture text -->
(a) Simulation (b) Real world<br><!-- End of picture text -->

Figure 6: RQ2 under a fixed per-embodiment pretraining budget. We sweep the number of source embodiments among 32, 128, and 512 while keeping the per-embodiment trajectory count fixed, so the total pretraining budget grows with _|E_ pre _|_ . Average zero-shot progress is reported in simulation (a) and real-world evaluation (b). 

embodiment pretraining pool from RQ2. We evaluate these variants in simulation because several objectives require annotations beyond standard demonstrations. 

Table 5 shows that all auxiliary objectives improve over the no-co-training baseline, raising average progress from 75.7% to 77.7–82.3%, with gains concentrated on gripper-only, arm-only, and full-embodiment shifts. Among language-action variants, the EEF-only objective outperforms the mixed-frame LAP variant, suggesting that auxiliary supervision is most effective when its action semantics align with the local EEF representation favored by RQ1. 

Table 5: RQ3 simulation results: auxiliary co-training objectives. All co-training variants improve over imitation-only training, with the largest gains under larger embodiment shifts. 

|**Post train data**|_es_|**APP**|**GRP**|**ARM**|**FULL**|**Average**|
|---|---|---|---|---|---|---|
|No co-training|**91.5**_±_0.6|90.0_±_1.1|78.3_±_2.4|74.3_±_0.9|60.4_±_0.9|75.7_±_1.3|
|Language-action (eef-frame)|90.5_±_1.1|88.6_±_0.2|86.7_±_1.4|79.3_±_0.9|70.5_±_0.5|81.3_±_0.5|
|Language-action (both frames)|89.5_±_1.5|88.6_±_0.4|79.8_±_0.8|73.8_±_1.6|68.4_±_1.5|77.7_±_0.3|
|Subgoal|89.1_±_1.8|**90.3**_±_0.8|**87.4**_±_2.2|77.8_±_0.5|71.5_±_1.5|81.8_±_0.8|
|Task-conditioned BBox|90.4_±_1.4|90.2_±_0.4|87.0_±_0.9|**80.1**_±_0.8|**71.9**_±_1.4|**82.3**_±_0.2|



### **5.5 RQ4: How Does Target-Embodiment Exposure Affect Transfer?** 

RQ4 studies _pretrain-exposed zero-shot transfer_ on two representative targets, UR5eUMI and GoogleRobot. For each target-exposure ratio, we keep the total pretraining budget fixed at 640K trajectories by replacing a controlled fraction of source data with target-embodiment pretraining data; target downstream tasks remain excluded from all training. We also report a non-zero-shot oracle initialized from the shared backbone pretrained on the full pretraining dataset _D_ ( _E_ pre _, T_ pre) and subsequently post-trained on the target-embodiment downstream data _D_ ( _et, T_ post), serving as an upper reference. 

Fig. 7 shows that even limited target embodiment exposure reduces the transfer gap. Adding only 5% target-embodiment data during pretraining yields the largest marginal gain, improving UR5eUMI from 69.3% to 78.6% and GoogleRobot from 51.4% to 68.9%. However, both curves remain be- 

9 



(a) RQ4 pretraining composition setting 







<!-- Start of picture text -->
(b) UR5eUMI (c) Google Robot<br><!-- End of picture text -->

Figure 7: RQ4: target-embodiment exposure during pretraining. Only 5% of target embodiment pretraining data substantially reduce the transfer gap on both UR5eUMI and Google Robot, but remain below the oracle performance of post-training on the target embodiment and test tasks. 

low the oracle, indicating that target-embodiment pretraining mitigates but does not eliminate the need for target-task adaptation. These results support reporting strict and pretrain-exposed zero-shot transfer separately, since target inclusion during pretraining changes the difficulty of the evaluation even when downstream target tasks are excluded. 

We further test whether the effect of targetembodiment exposure depends on the state-action representation studied in RQ1. Table 6 compares all four representations under strict zero-shot transfer and 5% target-embodiment exposure. Target exposure improves all four representations, while substantially narrowing their performance gaps. This suggests that representation choice matters most under strict transfer, whereas even limited targetembodiment exposure can partially compensate for less transferable representations. 

Table 6: Interaction between state-action representation and target-embodiment exposure. Average progress (%) on the RQ4 targets. 

|**State**|**Action**|**0%**|**5%**|
|---|---|---|---|
|Abs. EEF|World-Delta|33.9|66.9|
|Abs. EEF|EEF-Delta|38.6|69.2|
|EEF-Delta|World-Delta|58.4|67.5|
|EEF-Delta|EEF-Delta|**60.4**|**71.1**|



### **5.6 Practical Guidance for Cross-Embodiment Transfer within the Studied Setting** 

Taken together, these research questions suggest a practical recipe for evaluating and improving cross-embodiment VLA transfer. First, zero-shot claims should specify whether the target embodiment is absent from all training data or only absent from task-specific post-training, because these conditions can differ substantially in transfer difficulty. Second, for transfers with large embodiment gaps, we recommend embodiment-compatible representations that use both EEF-Delta state and actions, while absolute EEF state with EEF-Delta actions can remain competitive when the shift is limited to the end effector or appearance. Third, source-embodiment diversity should be considered jointly with data-budget allocation. Within our procedurally generated Franka-style source pool, the 512-source setting yields the strongest overall transfer, although the fixed-total-budget trend is non-monotonic in real-world validation. Finally, auxiliary co-training objectives generally improve transfer. 

## **6 Limitations and Conclusion** 

**Limitations.** This work studies cross-embodiment transfer under a controlled experimental scope, focusing on stationary tabletop manipulation with representative two-finger-gripper manipulators. This design enables matched comparisons that isolate embodiment differences, but does not cover the full range of robot morphologies or manipulation settings. In particular, extending the study to mobile manipulation involving base motion and dexterous hands is an important future direction; the latter introduce substantially more complex contact topology, action dimensionality, and skill structure. Our current evaluation also focuses primarily on relatively short-horizon manipulation tasks, and it remains to be seen whether the observed transfer patterns persist in long-horizon settings. Finally, our experiments are designed to characterize policy behavior and cross-embodiment transfer rather than to optimize every system-level component required for deployment. Together, these design choices keep the analysis focused on embodiment transfer under controlled and comparable conditions. 

10 

**Conclusion.** This paper presents a controlled study of zero-shot cross-embodiment transfer in VLA models for stationary tabletop manipulation with two-finger-gripper manipulators, separating strict zero-shot transfer from pretrain-exposed zero-shot transfer and evaluating embodiment shifts across appearance, gripper, arm, and full-embodiment changes. Our results show that local EEF-centered state-action representations, greater source-embodiment diversity under the studied comparison, and auxiliary co-training improve strict transfer, while even small amounts of target-embodiment exposure during pretraining substantially reduce the transfer gap. Together, these findings provide practical guidance for cross-embodiment VLA transfer within this controlled setting: report target-embodiment exposure explicitly, use local EEF-centered state-action representations, evaluate source-embodiment diversity as a controlled design variable, and incorporate auxiliary co-training. 

11 

## **References** 

- [1] Open X-Embodiment Collaboration, A. O’Neill, A. Rehman, A. Gupta, A. Maddukuri, A. Gupta, A. Padalkar, A. Lee, A. Pooley, A. Gupta, A. Mandlekar, A. Jain, A. Tung, A. Bewley, A. Herzog, A. Irpan, A. Khazatsky, A. Rai, A. Gupta, A. Wang, A. Kolobov, A. Singh, A. Garg, A. Kembhavi, A. Xie, A. Brohan, A. Raffin, A. Sharma, A. Yavary, A. Jain, A. Balakrishna, A. Wahid, B. Burgess-Limerick, B. Kim, B. Sch¨olkopf, B. Wulfe, B. Ichter, C. Lu, C. Xu, C. Le, C. Finn, C. Wang, C. Xu, C. Chi, C. Huang, C. Chan, C. Agia, C. Pan, C. Fu, C. Devin, D. Xu, D. Morton, D. Driess, D. Chen, D. Pathak, D. Shah, D. B¨uchler, D. Jayaraman, D. Kalashnikov, D. Sadigh, E. Johns, E. Foster, F. Liu, F. Ceola, F. Xia, F. Zhao, F. V. Frujeri, F. Stulp, G. Zhou, G. S. Sukhatme, G. Salhotra, G. Yan, G. Feng, G. Schiavi, G. Berseth, G. Kahn, G. Yang, G. Wang, H. Su, H.-S. Fang, H. Shi, H. Bao, H. B. Amor, H. I. Christensen, H. Furuta, H. Bharadhwaj, H. Walke, H. Fang, H. Ha, I. Mordatch, I. Radosavovic, I. Leal, J. Liang, J. Abou-Chakra, J. Kim, J. Drake, J. Peters, J. Schneider, J. Hsu, J. Vakil, J. Bohg, J. Bingham, J. Wu, J. Gao, J. Hu, J. Wu, J. Wu, J. Sun, J. Luo, J. Gu, J. Tan, J. Oh, J. Wu, J. Lu, J. Yang, J. Malik, J. Silv´erio, J. Hejna, J. Booher, J. Tompson, J. Yang, J. Salvador, J. J. Lim, J. Han, K. Wang, K. Rao, K. Pertsch, K. Hausman, K. Go, K. Gopalakrishnan, K. Goldberg, K. Byrne, K. Oslund, K. Kawaharazuka, K. Black, K. Lin, K. Zhang, K. Ehsani, K. Lekkala, K. Ellis, K. Rana, K. Srinivasan, K. Fang, K. P. Singh, K.-H. Zeng, K. Hatch, K. Hsu, L. Itti, L. Y. Chen, L. Pinto, L. Fei-Fei, L. Tan, L. J. Fan, L. Ott, L. Lee, L. Weihs, M. Chen, M. Lepert, M. Memmel, M. Tomizuka, M. Itkina, M. G. Castro, M. Spero, M. Du, M. Ahn, M. C. Yip, M. Zhang, M. Ding, M. Heo, M. K. Srirama, M. Sharma, M. J. Kim, M. Z. Irshad, N. Kanazawa, N. Hansen, N. Heess, N. J. Joshi, N. Suenderhauf, N. Liu, N. D. Palo, N. M. M. Shafiullah, O. Mees, O. Kroemer, O. Bastani, P. R. Sanketi, P. T. Miller, P. Yin, P. Wohlhart, P. Xu, P. D. Fagan, P. Mitrano, P. Sermanet, P. Abbeel, P. Sundaresan, Q. Chen, Q. Vuong, R. Rafailov, R. Tian, R. Doshi, R. Mart´ın-Mart´ın, R. Baijal, R. Scalise, R. Hendrix, R. Lin, R. Qian, R. Zhang, R. Mendonca, R. Shah, R. Hoque, R. Julian, S. Bustamante, S. Kirmani, S. Levine, S. Lin, S. Moore, S. Bahl, S. Dass, S. Sonawani, S. Tulsiani, S. Song, S. Xu, S. Haldar, S. Karamcheti, S. Adebola, S. Guist, S. Nasiriany, S. Schaal, S. Welker, S. Tian, S. Ramamoorthy, S. Dasari, S. Belkhale, S. Park, S. Nair, S. Mirchandani, T. Osa, T. Gupta, T. Harada, T. Matsushima, T. Xiao, T. Kollar, T. Yu, T. Ding, T. Davchev, T. Z. Zhao, T. Armstrong, T. Darrell, T. Chung, V. Jain, V. Kumar, V. Vanhoucke, V. Guizilini, W. Zhan, W. Zhou, W. Burgard, X. Chen, X. Chen, X. Wang, X. Zhu, X. Geng, X. Liu, X. Liangwei, X. Li, Y. Pang, Y. Lu, Y. J. Ma, Y. Kim, Y. Chebotar, Y. Zhou, Y. Zhu, Y. Wu, Y. Xu, Y. Wang, Y. Bisk, Y. Dou, Y. Cho, Y. Lee, Y. Cui, Y. Cao, Y.-H. Wu, Y. Tang, Y. Zhu, Y. Zhang, Y. Jiang, Y. Li, Y. Li, Y. Iwasawa, Y. Matsuo, Z. Ma, Z. Xu, Z. J. Cui, Z. Zhang, Z. Fu, and Z. Lin. Open x-embodiment: Robotic learning datasets and rt-x models. In _2024 IEEE International Conference on Robotics and Automation (ICRA)_ , pages 6892–6903, 2024. doi:10.1109/ICRA57147.2024.10611477. URL `https://arxiv.org/abs/2310.08864` . 

- [2] A. Khazatsky, K. Pertsch, S. Nair, A. Balakrishna, S. Dasari, S. Karamcheti, S. Nasiriany, M. K. Srirama, L. Y. Chen, K. Ellis, P. D. Fagan, J. Hejna, M. Itkina, M. Lepert, Y. J. Ma, P. T. Miller, J. Wu, S. Belkhale, S. Dass, H. Ha, A. Jain, A. Lee, Y. Lee, M. Memmel, S. Park, I. Radosavovic, K. Wang, A. Zhan, K. Black, C. Chi, K. B. Hatch, S. Lin, J. Lu, J. Mercat, A. Rehman, P. R. Sanketi, A. Sharma, C. Simpson, Q. Vuong, H. R. Walke, B. Wulfe, T. Xiao, J. H. Yang, A. Yavary, T. Z. Zhao, C. Agia, R. Baijal, M. G. Castro, D. Chen, Q. Chen, T. Chung, J. Drake, E. P. Foster, J. Gao, V. Guizilini, D. A. Herrera, M. Heo, K. Hsu, J. Hu, M. Z. Irshad, D. Jackson, C. Le, Y. Li, K. Lin, R. Lin, Z. Ma, A. Maddukuri, S. Mirchandani, D. Morton, T. Nguyen, A. O’Neill, R. Scalise, D. Seale, V. Son, S. Tian, E. Tran, A. E. Wang, Y. Wu, A. Xie, J. Yang, P. Yin, Y. Zhang, O. Bastani, G. Berseth, J. Bohg, K. Goldberg, A. Gupta, A. Gupta, D. Jayaraman, J. J. Lim, J. Malik, R. Mart´ın-Mart´ın, S. Ramamoorthy, D. Sadigh, S. Song, J. Wu, M. C. Yip, Y. Zhu, T. Kollar, S. Levine, and C. Finn. Droid: A large-scale in-the-wild robot manipulation dataset, 2024. URL `https://arxiv.org/abs/ 2403.12945` . 

12 

- [3] H.-S. Fang, H. Fang, Z. Tang, J. Liu, C. Wang, J. Wang, H. Zhu, and C. Lu. Rh20t: A comprehensive robotic dataset for learning diverse skills in one-shot, 2023. URL `https: //arxiv.org/abs/2307.00595` . 

- [4] H. Walke, K. Black, A. Lee, M. J. Kim, M. Du, C. Zheng, T. Zhao, P. Hansen-Estruch, Q. Vuong, A. He, V. Myers, K. Fang, C. Finn, and S. Levine. Bridgedata v2: A dataset for robot learning at scale, 2023. URL `https://arxiv.org/abs/2308.12952` . 

- [5] K. Wu, C. Hou, J. Liu, Z. Che, X. Ju, Z. Yang, M. Li, Y. Zhao, Z. Xu, G. Yang, S. Fan, X. Wang, F. Liao, Z. Zhao, G. Li, Z. Jin, L. Wang, J. Mao, N. Liu, P. Ren, Q. Zhang, Y. Lyu, M. Liu, J. He, Y. Luo, Z. Gao, C. Li, C. Gu, Y. Fu, D. Wu, X. Wang, S. Chen, Z. Wang, P. An, S. Qian, S. Zhang, and J. Tang. Robomind: Benchmark on multi-embodiment intelligence normative data for robot manipulation. In _Robotics: Science and Systems XXI_ . Robotics: Science and Systems Foundation, June 2025. doi:10.15607/RSS.2025.XXI.152. URL `https://www.roboticsproceedings.org/rss21/p152.html` . 

- [6] AgiBot-World-Contributors, Q. Bu, J. Cai, L. Chen, X. Cui, Y. Ding, S. Feng, S. Gao, X. He, X. Hu, X. Huang, S. Jiang, Y. Jiang, C. Jing, H. Li, J. Li, C. Liu, Y. Liu, Y. Lu, J. Luo, P. Luo, Y. Mu, Y. Niu, Y. Pan, J. Pang, Y. Qiao, G. Ren, C. Ruan, J. Shan, Y. Shen, C. Shi, M. Shi, M. Shi, C. Sima, J. Song, H. Wang, W. Wang, D. Wei, C. Xie, G. Xu, J. Yan, C. Yang, L. Yang, S. Yang, M. Yao, J. Zeng, C. Zhang, Q. Zhang, B. Zhao, C. Zhao, J. Zhao, and J. Zhu. Agibot world colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems, 2025. URL `https://arxiv.org/abs/2503.06669` . 

- [7] Z. Wang, H. Zheng, Y. Nie, W. Xu, Q. Wang, H. Ye, Z. Li, K. Zhang, X. Cheng, W. Dong, C. Cai, L. Lin, F. Zheng, and X. Liang. All robots in one: A new standard and unified dataset for versatile, general-purpose embodied agents, 2024. URL `https://arxiv.org/abs/2408. 10899` . 

- [8] Y. Tian, Y. Yang, Y. Xie, Z. Cai, X. Shi, N. Gao, H. Liu, X. Jiang, Z. Qiu, F. Yuan, Y. Li, P. Wang, J. Cai, J. Zeng, H. Dong, and J. Pang. Interndata-a1: Pioneering high-fidelity synthetic data for pre-training generalist policy, 2025. URL `https://arxiv.org/abs/2511.16651` . 

- [9] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog, J. Hsu, J. Ibarz, B. Ichter, A. Irpan, T. Jackson, S. Jesmonth, N. J. Joshi, R. Julian, D. Kalashnikov, Y. Kuang, I. Leal, K.-H. Lee, S. Levine, Y. Lu, U. Malla, D. Manjunath, I. Mordatch, O. Nachum, C. Parada, J. Peralta, E. Perez, K. Pertsch, J. Quiambao, K. Rao, M. Ryoo, G. Salazar, P. Sanketi, K. Sayed, J. Singh, S. Sontakke, A. Stone, C. Tan, H. Tran, V. Vanhoucke, S. Vega, Q. Vuong, F. Xia, T. Xiao, P. Xu, S. Xu, T. Yu, and B. Zitkovich. Rt-1: Robotics transformer for real-world control at scale, 2022. URL `https://arxiv.org/abs/2212.06817` . 

- [10] B. Zitkovich, T. Yu, S. Xu, P. Xu, T. Xiao, F. Xia, J. Wu, P. Wohlhart, S. Welker, A. Wahid, Q. Vuong, V. Vanhoucke, H. Tran, R. Soricut, A. Singh, J. Singh, P. Sermanet, P. R. Sanketi, G. Salazar, M. S. Ryoo, K. Reymann, K. Rao, K. Pertsch, I. Mordatch, H. Michalewski, Y. Lu, S. Levine, L. Lee, T.-W. E. Lee, I. Leal, Y. Kuang, D. Kalashnikov, R. Julian, N. J. Joshi, A. Irpan, B. Ichter, J. Hsu, A. Herzog, K. Hausman, K. Gopalakrishnan, C. Fu, P. Florence, C. Finn, K. A. Dubey, D. Driess, T. Ding, K. M. Choromanski, X. Chen, Y. Chebotar, J. Carbajal, N. Brown, A. Brohan, M. G. Arenas, and K. Han. Rt-2: Vision-languageaction models transfer web knowledge to robotic control. In J. Tan, M. Toussaint, and K. Darvish, editors, _Proceedings of The 7th Conference on Robot Learning_ , volume 229 of _Proceedings of Machine Learning Research_ , pages 2165–2183. PMLR, 06–09 Nov 2023. URL `https://proceedings.mlr.press/v229/zitkovich23a.html` . 

- [11] D. Ghosh, H. R. Walke, K. Pertsch, K. Black, O. Mees, S. Dasari, J. Hejna, T. Kreiman, C. Xu, J. Luo, Y. L. Tan, L. Y. Chen, Q. Vuong, T. Xiao, P. R. Sanketi, D. Sadigh, C. Finn, and S. Levine. Octo: An open-source generalist robot policy. In _Proceedings of Robotics: Science and Systems_ , Delft, Netherlands, July 2024. doi:10.15607/RSS.2024.XX.090. 

13 

- [12] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, Q. Vuong, T. Kollar, B. Burchfiel, R. Tedrake, D. Sadigh, S. Levine, P. Liang, and C. Finn. Openvla: An open-source vision-language-action model, 2024. URL `https://arxiv.org/abs/2406.09246` . 

- [13] K. Black, N. Brown, D. Driess, A. Esmail, M. R. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, S. Jakubczak, T. Jones, L. Ke, S. Levine, A. Li-Bell, M. Mothukuri, S. Nair, K. Pertsch, L. X. Shi, L. Smith, J. Tanner, Q. Vuong, A. Walling, H. Wang, and U. Zhilinsky. _π_ 0: A Vision-Language-Action Flow Model for General Robot Control. In _Proceedings of Robotics: Science and Systems_ , Los Angeles, CA, USA, June 2025. doi: 10.15607/RSS.2025.XXI.010. 

- [14] Physical Intelligence, K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, M. Y. Galliker, D. Ghosh, L. Groom, K. Hausman, B. Ichter, S. Jakubczak, T. Jones, L. Ke, D. LeBlanc, S. Levine, A. Li-Bell, M. Mothukuri, S. Nair, K. Pertsch, A. Z. Ren, L. X. Shi, L. Smith, J. T. Springenberg, K. Stachowicz, J. Tanner, Q. Vuong, H. Walke, A. Walling, H. Wang, L. Yu, and U. Zhilinsky. _π_ 0 _._ 5: a vision-languageaction model with open-world generalization, 2025. URL `https://arxiv.org/abs/2504. 16054` . 

- [15] NVIDIA, J. Bjorck, F. Casta˜neda, N. Cherniadev, X. Da, R. Ding, L. J. Fan, Y. Fang, D. Fox, F. Hu, S. Huang, J. Jang, Z. Jiang, J. Kautz, K. Kundalia, L. Lao, Z. Li, Z. Lin, K. Lin, G. Liu, E. Llontop, L. Magne, A. Mandlekar, A. Narayan, S. Nasiriany, S. Reed, Y. L. Tan, G. Wang, Z. Wang, J. Wang, Q. Wang, J. Xiang, Y. Xie, Y. Xu, Z. Xu, S. Ye, Z. Yu, A. Zhang, H. Zhang, Y. Zhao, R. Zheng, and Y. Zhu. Gr00t n1: An open foundation model for generalist humanoid robots, 2025. URL `https://arxiv.org/abs/2503.14734` . 

- [16] Q. Li, Y. Liang, Z. Wang, L. Luo, X. Chen, M. Liao, F. Wei, Y. Deng, S. Xu, Y. Zhang, X. Wang, B. Liu, J. Fu, J. Bao, D. Chen, Y. Shi, J. Yang, and B. Guo. Cogact: A foundational vision-language-action model for synergizing cognition and action in robotic manipulation, 2024. URL `https://arxiv.org/abs/2411.19650` . 

- [17] H. Wu, Y. Jing, C. Cheang, G. Chen, J. Xu, X. Li, M. Liu, H. Li, and T. Kong. Unleashing largescale video generative pre-training for visual robot manipulation. In _International Conference on Learning Representations_ , 2024. URL `https://arxiv.org/abs/2312.13139` . 

- [18] X. Li, M. Liu, H. Zhang, C. Yu, J. Xu, H. Wu, C. Cheang, Y. Jing, W. Zhang, H. Liu, H. Li, and T. Kong. Vision-language foundation models as effective robot imitators, 2023. URL `https://arxiv.org/abs/2311.01378` . 

- [19] M. Shukor, D. Aubakirova, F. Capuano, P. Kooijmans, S. Palma, A. Zouitine, M. Aractingi, C. Pascal, M. Russi, A. Marafioti, S. Alibert, M. Cord, T. Wolf, and R. Cadene. Smolvla: A vision-language-action model for affordable and efficient robotics, 2025. URL `https: //arxiv.org/abs/2506.01844` . 

- [20] S. Deng, M. Yan, S. Wei, H. Ma, Y. Yang, J. Chen, Z. Zhang, T. Yang, X. Zhang, W. Zhang, H. Cui, Z. Zhang, and H. Wang. Graspvla: a grasping foundation model pre-trained on billionscale synthetic action data, 2025. URL `https://arxiv.org/abs/2505.03233` . 

- [21] J. Zheng, J. Li, Z. Wang, D. Liu, X. Kang, Y. Feng, Y. Zheng, J. Zou, Y. Chen, J. Zeng, Y.-Q. Zhang, J. Pang, J. Liu, T. Wang, and X. Zhan. X-vla: Soft-prompted transformer as scalable cross-embodiment vision-language-action model, 2025. URL `https://arxiv.org/ abs/2510.10274` . 

- [22] D. Qu, H. Song, Q. Chen, Y. Yao, X. Ye, Y. Ding, Z. Wang, J. Gu, B. Zhao, D. Wang, and X. Li. Spatialvla: Exploring spatial representations for visual-language-action model, 2025. URL `https://arxiv.org/abs/2501.15830` . 

14 

- [23] M. J. Kim, C. Finn, and P. Liang. Fine-tuning vision-language-action models: Optimizing speed and success, 2025. URL `https://arxiv.org/abs/2502.19645` . 

- [24] J. Wen, Y. Zhu, J. Li, Z. Tang, C. Shen, and F. Feng. Dexvla: Vision-language model with plug-in diffusion expert for general robot control, 2025. URL `https://arxiv.org/abs/ 2502.05855` . 

- [25] S. Liu, B. Li, K. Ma, L. Wu, H. Tan, X. Ouyang, H. Su, and J. Zhu. Rdt2: Exploring the scaling limit of umi data towards zero-shot cross-embodiment generalization, 2026. URL `https://arxiv.org/abs/2602.03310` . 

- [26] S. Liu, L. Wu, B. Li, H. Tan, H. Chen, Z. Wang, K. Xu, H. Su, and J. Zhu. Rdt-1b: a diffusion foundation model for bimanual manipulation, 2025. URL `https://arxiv.org/abs/2410. 07864` . 

- [27] H. Luo, Y. Wang, W. Zhang, S. Zheng, Z. Xi, C. Xu, H. Xu, H. Yuan, C. Zhang, Y. Wang, Y. Feng, and Z. Lu. Being-H0.5: Scaling human-centric robot learning for cross-embodiment generalization, 2026. URL `https://arxiv.org/abs/2601.12993` . 

- [28] R. Doshi, H. Walke, O. Mees, S. Dasari, and S. Levine. Scaling cross-embodied learning: One policy for manipulation, navigation, locomotion and aviation, 2024. URL `https://arxiv. org/abs/2408.11812` . 

- [29] N. Bohlinger, G. Czechmanowski, M. Krupka, P. Kicki, K. Walas, J. Peters, and D. Tateo. One policy to run them all: an end-to-end learning approach to multi-embodiment locomotion, 2025. URL `https://arxiv.org/abs/2409.06366` . 

- [30] M. Liu, D. Pathak, and A. Agarwal. Locoformer: Generalist locomotion via long-context adaptation, 2025. URL `https://arxiv.org/abs/2509.23745` . 

- [31] S. Ye, J. Jang, B. Jeon, S. Joo, J. Yang, B. Peng, A. Mandlekar, R. Tan, Y.-W. Chao, B. Y. Lin, L. Liden, K. Lee, J. Gao, L. Zettlemoyer, D. Fox, and M. Seo. Latent action pretraining from videos, 2025. URL `https://arxiv.org/abs/2410.11758` . 

- [32] Q. Bu, Y. Yang, J. Cai, S. Gao, G. Ren, M. Yao, P. Luo, and H. Li. Univla: Learning to act anywhere with task-centric latent actions, 2025. URL `https://arxiv.org/abs/2505. 06111` . 

- [33] M. Shi, L. Chen, J. Chen, Y. Lu, C. Liu, G. Ren, P. Luo, D. Huang, M. Yao, and H. Li. Is diversity all you need for scalable robotic manipulation?, 2025. URL `https://arxiv.org/ abs/2507.06219` . 

- [34] B. Ai, L. Dai, N. Bohlinger, D. Li, T. Mu, Z. Wu, K. Fay, H. I. Christensen, J. Peters, and H. Su. Towards embodiment scaling laws in robot locomotion, 2025. URL `https://arxiv. org/abs/2505.05753` . 

- [35] M. Xu, Z. Xu, C. Chi, M. Veloso, and S. Song. Xskill: Cross embodiment skill discovery, 2023. URL `https://arxiv.org/abs/2307.09955` . 

- [36] H. Zhi, W. Tan, L. Zhu, F. Li, J. Li, G. Yang, and H. T. Shen. Motif: Learning action motifs for few-shot cross-embodiment transfer, 2026. URL `https://arxiv.org/abs/2602.13764` . 

- [37] H. Kim, J. Kang, H. Kang, M. Cho, S. J. Kim, and Y. Lee. Uniskill: Imitating human videos via cross-embodiment skill representations, 2025. URL `https://arxiv.org/abs/2505. 08787` . 

- [38] R. Yang, Q. Yu, Y. Wu, R. Yan, B. Li, A.-C. Cheng, X. Zou, Y. Fang, X. Cheng, R.-Z. Qiu, H. Yin, S. Liu, S. Han, Y. Lu, and X. Wang. Egovla: Learning vision-language-action models from egocentric human videos, 2025. URL `https://arxiv.org/abs/2507.12440` . 

15 

- [39] J. Ren, P. Sundaresan, D. Sadigh, S. Choudhury, and J. Bohg. Motion tracks: A unified representation for human-robot transfer in few-shot imitation learning, 2025. URL `https://arxiv.org/abs/2501.06994` . 

- [40] J. Zheng, J. Li, D. Liu, Y. Zheng, Z. Wang, Z. Ou, Y. Liu, J. Liu, Y.-Q. Zhang, and X. Zhan. Universal actions for enhanced embodied foundation models, 2025. URL `https://arxiv. org/abs/2501.10105` . 

- [41] G. Jiang, Y. Liang, J. Ye, J.-Y. Huang, C. Jing, R. Duan, P. Abbeel, X. Wang, and X. Zou. Cross-hand latent representation for vision-language-action models, 2026. URL `https:// arxiv.org/abs/2603.10158` . 

- [42] T. Wang, D. Bhatt, X. Wang, and N. Atanasov. Cross-embodiment robot manipulation skill transfer using latent space alignment, 2024. URL `https://arxiv.org/abs/2406.01968` . 

- [43] E. Bauer, E. Nava, and R. K. Katzschmann. Latent action diffusion for cross-embodiment manipulation, 2025. URL `https://arxiv.org/abs/2506.14608` . 

- [44] J. Mu, S. Yang, H. Bae, F. Jia, Q. Ben, B. Li, H. Xu, and J. Pang. One-policy-fits-all: Geometryaware action latents for cross-embodiment manipulation, 2026. URL `https://arxiv.org/ abs/2603.14522` . 

- [45] H. Yuan, B. Zhou, Y. Fu, and Z. Lu. Cross-embodiment dexterous grasping with reinforcement learning. In _International Conference on Learning Representations_ , 2025. URL `https:// arxiv.org/abs/2410.02479` . 

- [46] Y. Yan and D. Lee. Learning a unified latent space for cross-embodiment robot control, 2026. URL `https://arxiv.org/abs/2601.15419` . 

- [47] L. Wang, X. Chen, J. Zhao, and K. He. Scaling proprioceptive-visual learning with heterogeneous pre-trained transformers. In _Advances in Neural Information Processing Systems_ , 2024. URL `https://arxiv.org/abs/2409.20537` . 

- [48] Y. Zhang, C. Yan, J. Yu, J. Xiao, and M. Feroskhan. Learning adaptive cross-embodiment visuomotor policy with contrastive prompt orchestration, 2026. URL `https://arxiv.org/ abs/2602.01040` . 

- [49] T. Wu, S. Li, J. Gong, C. Guo, X. Li, S. Mu, and W. Ding. Cei: A unified interface for crossembodiment visuomotor policy learning in 3d space, 2026. URL `https://arxiv.org/abs/ 2601.09163` . 

- [50] Y. Zhang, C. Wang, O. Lu, Y. Zhao, Y. Ge, Z. Sun, X. Li, C. Zhang, C. Bai, and X. Li. Alignthen-steer: Adapting the vision-language action models through unified latent guidance, 2025. URL `https://arxiv.org/abs/2509.02055` . 

- [51] M. Seo, H. A. Park, S. Yuan, Y. Zhu, and L. Sentis. Legato: Cross-embodiment imitation using a grasping tool. _IEEE Robotics and Automation Letters_ , 10(3):2854–2861, 2025. doi: 10.1109/LRA.2025.3535182. URL `https://arxiv.org/abs/2411.03682` . 

- [52] L. Y. Chen, K. Hari, K. Dharmarajan, C. Xu, Q. Vuong, and K. Goldberg. Mirage: Crossembodiment zero-shot policy transfer with cross-painting, 2024. URL `https://arxiv.org/ abs/2402.19249` . 

- [53] L. Y. Chen, C. Xu, K. Dharmarajan, M. Z. Irshad, R. Cheng, K. Keutzer, M. Tomizuka, Q. Vuong, and K. Goldberg. Rovi-aug: Robot and viewpoint augmentation for crossembodiment robot learning, 2024. URL `https://arxiv.org/abs/2409.03403` . 

- [54] M. Lepert, R. Doshi, and J. Bohg. Shadow: Leveraging segmentation masks for crossembodiment policy transfer, 2025. URL `https://arxiv.org/abs/2503.00774` . 

16 

- [55] M. Lepert, J. Fang, and J. Bohg. Phantom: Training robots without robots using only human videos, 2025. URL `https://arxiv.org/abs/2503.00779` . 

- [56] L. Zha, A. J. Hancock, M. Zhang, T. Yin, Y. Huang, D. Shah, A. Z. Ren, and A. Majumdar. Lap: Language-action pre-training enables zero-shot cross-embodiment transfer, 2026. URL `https://arxiv.org/abs/2602.10556` . 

- [57] F. Lin, K. Arora, J. Mercat, H. Nishimura, P. Shah, C. Xu, M. Zhang, M. Zolotas, M. Angeles, O. Pfannenstiehl, A. Beaulieu, and J. Barreiros. A systematic study of data modalities and strategies for co-training large behavior models for robot manipulation, 2026. URL `https: //arxiv.org/abs/2602.01067` . 

- [58] M. Zawalski, W. Chen, K. Pertsch, O. Mees, C. Finn, and S. Levine. Robotic control via embodied chain-of-thought reasoning, 2025. URL `https://arxiv.org/abs/2407.08693` . 

- [59] Q. Zhao, Y. Lu, M. J. Kim, Z. Fu, Z. Zhang, Y. Wu, Z. Li, Q. Ma, S. Han, C. Finn, A. Handa, M.-Y. Liu, D. Xiang, G. Wetzstein, and T.-Y. Lin. Cot-vla: Visual chain-of-thought reasoning for vision-language-action models, 2025. URL `https://arxiv.org/abs/2503.22020` . 

- [60] R. Zheng, Y. Liang, S. Huang, J. Gao, H. Daum´e III, A. Kolobov, F. Huang, and J. Yang. Tracevla: Visual trace prompting enhances spatial-temporal awareness for generalist robotic policies, 2025. URL `https://arxiv.org/abs/2412.10345` . 

- [61] J. Zhao, W. Lu, D. Zhang, Y. Liu, Y. Liang, T. Zhang, Y. Cao, J. Xie, Y. Hu, S. Wang, J. Guo, D. Wang, and Y. Gao. Do you need proprioceptive states in visuomotor policies?, 2025. URL `https://arxiv.org/abs/2509.18644` . 

- [62] J. H. Yang, C. Glossop, A. Bhorkar, D. Shah, Q. Vuong, C. Finn, D. Sadigh, and S. Levine. Pushing the limits of cross-embodiment learning for manipulation and navigation. In _Proceedings of Robotics: Science and Systems_ , 2024. doi:10.15607/RSS.2024.XX.093. 

- [63] M. Parakh, A. Kirchmeyer, B. Han, and J. Deng. Anybody: A benchmark suite for crossembodiment manipulation, 2025. URL `https://arxiv.org/abs/2505.14986` . 

- [64] M. Piseno, G. Tevet, and C. K. Liu. Cloak: Zero-shot cross-embodiment manipulation by masking the end-effector from the vla, 2026. URL `https://arxiv.org/abs/2606.22836` . 

- [65] Physical Intelligence, B. Ai, A. Amin, R. Aniceto, A. Balakrishna, G. Balke, K. Black, G. Bokinsky, S. Cao, T. Charbonnier, V. Choudhary, F. Collins, K. Conley, G. Connors, J. Darpinian, K. Dhabalia, M. Dhaka, J. DiCarlo, D. Driess, M. Equi, A. Esmail, Y. Fang, C. Finn, C. Glossop, T. Godden, I. Goryachev, L. Groom, H. Habeeb, H. Hancock, K. Hausman, G. Hussein, V. Hwang, B. Ichter, C. Jacobsen, S. Jakubczak, R. Jen, T. Jones, G. Kammerer, B. Katz, L. Ke, M. Khadikov, C. Kuchi, M. Lamb, D. LeBlanc, B. LeCount, S. Levine, X. Li, A. Li-Bell, V. Lialin, Z. Liang, W. Lim, Y. Lu, E. Luo, V. Mano, N. Marwaha, A. Mongush, L. Murphy, S. Nair, T. Patterson, K. Pertsch, A. Z. Ren, G. Schelske, C. Sharma, B. Shi, L. X. Shi, L. Smith, J. T. Springenberg, K. Stachowicz, W. Stoeckle, J. Tang, J. Tanner, S. Tekeste, M. Torne, K. Vedder, Q. Vuong, A. Walling, H. Wang, J. Wang, X. Wang, C. Whalen, S. Whitmore, B. Williams, C. Xu, S. Yoo, L. Yu, W. Zhang, Z. Zhang, and U. Zhilinsky. _π_ 0 _._ 7: a steerable generalist robotic foundation model with emergent capabilities, 2026. URL `https://arxiv.org/abs/2604.15483` . 

- [66] H. Yuan, Z. Liang, A. Chen, Y. Wang, H. Li, P. Lin, Y. Huang, Z. Lei, T. Zhang, J. Zhang, J. Zhang, J. Fan, G. Zhou, Q. Peng, C. Lv, X. Chen, A. Yang, F. Huang, J. Lin, D. Liu, J. Zhou, C. Wu, and X.-H. Chen. Qwen-robotmanip technical report: Alignment unlocks scale for robotic manipulation foundation models, 2026. URL `https://arxiv.org/abs/2606. 17846` . 

17 

- [67] Gemini Robotics Team, A. Abdolmaleki, S. Abeyruwan, J. Ainslie, J.-B. Alayrac, M. G. Arenas, A. Balakrishna, N. Batchelor, A. Bewley, J. Bingham, M. Bloesch, K. Bousmalis, P. Brakel, A. Brohan, T. Buschmann, A. Byravan, S. Cabi, K. Caluwaerts, F. Casarini, C. Chan, O. Chang, L. Chappellet-Volpini, J. E. Chen, X. Chen, H.-T. L. Chiang, K. Choromanski, A. Collister, D. B. D’Ambrosio, S. Dasari, T. Davchev, M. K. Dave, C. Devin, N. D. Palo, T. Ding, C. Doersch, A. Dostmohamed, Y. Du, D. Dwibedi, S. T. Egambaram, M. Elabd, T. Erez, X. Fang, C. Fantacci, C. Fong, E. Frey, C. Fu, R. Gao, M. Giustina, K. Gopalakrishnan, L. Graesser, O. Groth, A. Gupta, R. Hafner, S. Hansen, L. Hasenclever, S. Haves, N. Heess, B. Hernaez, A. Hofer, J. Hsu, L. Huang, S. H. Huang, A. Iscen, M. G. Jacob, D. Jain, S. Jesmonth, A. Jindal, R. Julian, D. Kalashnikov, M. E. Karagozler, S. Karp, M. Kecman, J. C. Kew, D. Kim, F. Kim, J. Kim, T. Kipf, S. Kirmani, K. Konyushkova, L. Y. Ku, Y. Kuang, T. Lampe, A. Laurens, T. A. Le, I. Leal, A. X. Lee, T.-W. E. Lee, G. Lever, J. Liang, L.-H. Lin, F. Liu, S. Long, C. Lu, S. Maddineni, A. Majumdar, K.-K. Maninis, A. Marmon, S. Martinez, A. H. Michaely, N. Milonopoulos, J. Moore, R. Moreno, M. Neunert, F. Nori, J. Ortiz, K. Oslund, C. Parada, E. Parisotto, A. Paryag, A. Pooley, T. Power, A. Quaglino, H. Qureshi, R. V. Raju, H. Ran, D. Rao, K. Rao, I. Reid, D. Rendleman, K. Reymann, M. Rivas, F. Romano, Y. Rubanova, P. P. Sampedro, P. R. Sanketi, D. Shah, M. Sharma, K. Shea, M. Shridhar, C. Shu, V. Sindhwani, S. Singh, R. Soricut, R. Sterneck, I. Storz, R. Surdulescu, J. Tan, J. Tompson, S. Tunyasuvunakool, J. Varley, G. Vesom, G. Vezzani, M. B. Villalonga, O. Vinyals, R. Wagner, A. Wahid, S. Welker, P. Wohlhart, C. Wu, M. Wulfmeier, F. Xia, T. Xiao, A. Xie, J. Xie, P. Xu, S. Xu, Y. Xu, Z. Xu, J. Yan, S. Yang, S. Yang, Y. Yang, H. H. Yu, W. Yu, W. Yuan, Y. Yuan, J. Zhang, T. Zhang, Z. Zhang, A. Zhou, G. Zhou, and Y. Zhou. Gemini robotics 1.5: Pushing the frontier of generalist robots with advanced embodied reasoning, thinking, and motion transfer, 2025. URL `https://arxiv.org/abs/2510.03342` . 

- [68] C. Yin, D. Huang, D. Yang, J. Wang, N. Zhao, C. Xu, W. Sun, L. Hou, Z. Li, J. Wu, Z. Liu, Z. Xiao, S. Zhang, L. Bao, R. Feng, Z. Pang, J. Li, Q. Wang, and M. Yao. Genie sim 3.0 : A high-fidelity comprehensive simulation platform for humanoid robot, 2026. URL `https: //arxiv.org/abs/2601.02078` . 

- [69] A. Maddukuri, Z. Jiang, L. Y. Chen, S. Nasiriany, Y. Xie, Y. Fang, W. Huang, Z. Wang, Z. Xu, N. Chernyadev, S. Reed, K. Goldberg, A. Mandlekar, L. Fan, and Y. Zhu. Sim-and-real cotraining: A simple recipe for vision-based robotic manipulation, 2025. URL `https://arxiv. org/abs/2503.24361` . 

- [70] J. Liu, M. Liu, Z. Wang, P. An, X. Li, K. Zhou, S. Yang, R. Zhang, Y. Guo, and S. Zhang. Robomamba: Efficient vision-language-action model for robotic reasoning and manipulation, 2024. URL `https://arxiv.org/abs/2406.04339` . 

- [71] F. Lin, R. Nai, Y. Hu, J. You, J. Zhao, and Y. Gao. Onetwovla: A unified vision-language-action model with adaptive reasoning, 2026. URL `https://arxiv.org/abs/2505.11917` . 

- [72] D. Qu, H. Song, Q. Chen, Z. Chen, X. Gao, D. Wang, X. Ye, Q. Lv, M. Shi, G. Ren, C. Ruan, M. Yao, H. Yang, J. Bao, B. Zhao, and X. Li. Eo-1: An open unified embodied foundation model for general robot control, 2026. URL `https://arxiv.org/abs/2508.21112` . 

- [73] X. Chen, Y. Chen, Y. Fu, N. Gao, J. Jia, W. Jin, H. Li, Y. Mu, J. Pang, Y. Qiao, Y. Tian, B. Wang, B. Wang, F. Wang, H. Wang, T. Wang, Z. Wang, X. Wei, C. Wu, S. Yang, J. Ye, J. Yu, J. Zeng, J. Zhang, J. Zhang, S. Zhang, F. Zheng, B. Zhou, and Y. Zhu. Internvla-m1: A spatially guided vision-language-action framework for generalist robot policy, 2025. URL `https://arxiv.org/abs/2510.13778` . 

18 

## **A Formal Definitions of the Two Zero-Shot Protocols** 

Table 7: Formal distinction between the two zero-shot cross-embodiment transfer protocols. 

|**Protocol**|**Pre-train data**|**Post-train data**|**Eval.**|
|---|---|---|---|
|**Strict zero-shot**|_D_pre =_D_(_E_pre_, T_pre)<br>**_et /∈E_pre**|_D_post =_D_(_{es}, T_post)<br>_es̸_ =_et_|emb.: _et_<br>task: _T_post|
||_T_pre_∩T_post =∅|||
|**Pretrain-**|_D_pre =_D_(_E_pre_, T_pre)|_D_post =_D_(_{es}, T_post)|emb.: _et_|
|**exposed zero-shot**|**_et ∈E_pre**<br>_T_pre_∩T_post =∅|_es̸_ =_et_|task: _T_post|



Table 7 separates the two protocols by whether the target embodiment appears in the pretraining embodiment set. Both protocols use the same post-training data, _D_ post = _D_ ( _{es}, T_ post) with _es̸_ = _et_ , and both evaluate on the target embodiment _et_ for downstream tasks _T_ post. In strict zeroshot transfer, _et ∈E/_ pre, so evaluation measures transfer to a robot absent from all training data. In pretrain-exposed zero-shot transfer, _et ∈E_ pre, but pretraining tasks remain disjoint from downstream tasks and post-training still uses only the source embodiment. Thus the evaluation remains zero-shot with respect to target-embodiment downstream demonstrations, because _D_ ( _{et}, T_ post) is never used for training, but it is not strict zero-shot because the policy has already observed the target embodiment during pretraining. 

## **B Per-RQ Experimental Design Matrix** 

Table 8 summarizes the experimental configuration used for each research question. The purpose of this matrix is to make explicit which factor is isolated in each comparison, which zero-shot protocol is used, what pretraining data enters the model, and whether the evidence comes from simulation, real-world evaluation, or both. Across all RQs, post-training uses the same source-embodiment downstream data, and the policy backbone, visual inputs, action horizon, optimization schedule, downstream task set, source post-training embodiment, and evaluation scenes are held fixed unless explicitly varied. 

Table 8: Per-RQ experimental design matrix. The table summarizes the protocol, pretraining data, representation choice, and evaluation platform used to isolate each experimental factor. 

|**RQ**|**Factor**<br>**studied**|**Zero-shot**<br>**protocol**|**Pretraining data**|**Representation**|**Eval.**|
|---|---|---|---|---|---|
|RQ1|State-action<br>representation|Strict<br>zero-shot|512 non-target source<br>embodiments|Varied across four<br>designs|Sim. +<br>real|
|RQ2|Source-<br>embodiment<br>diversity|Strict<br>zero-shot|Primary: 1/8/512 sources,<br>fixed total budget.<br>Complementary:<br>32/128/512 sources, fixed<br>per-embodiment budget.|Fixed to<br>best-average RQ1|Sim. +<br>real|
|RQ3|Auxiliary<br>co-training<br>objective|Strict<br>zero-shot|512 non-target source<br>embodiments with<br>auxiliaryobjectives|Fixed to<br>best-average RQ1|Sim.|
|RQ4|Target-<br>embodiment<br>exposure|Pretrain-<br>exposed<br>zero-shot|Fixed-size pretraining<br>pool with controlled<br>target-embodiment<br>replacement|Fixed to<br>best-average RQ1|Sim.|



## **C Detailed Experimental Results** 

The simulation tables in Tables 9–20 expand the aggregate results in the main paper into per-task, per-embodiment progress scores. Unless otherwise stated, each entry reports the three simulation repeats as run-1/ run-2/ run-3 progress. We report the source embodiment and the seven held-out target embodiments used in the main simulation study. For real-world evaluation, Tables 21–24 

19 

report the completed RQ1 and RQ2 runs on _open the fryer_ and _water the flower_ . Each task is reported in a separate table, target embodiments are rows, and the controlled experimental variable is shown in columns. 

## **D Data-Construction Details** 

**Simulation embodiment generation.** The 512 source embodiments are procedurally generated from a Franka-style tabletop arm template. We introduce embodiment diversity along geometry, end-effector morphology, and visual appearance. For arm-level variation, we scale the six movable arm-link regions with factors sampled from shrink and expansion modes, spanning approximately 0 _._ 7–1 _._ 3, and consistently deform the corresponding visual and collision meshes. For end-effector variation, we procedurally generate gripper fingers by sampling finger length, width and thickness profiles, gripper rotation, and optional fingertip cuboids. We also randomize the contact plank attached to the gripper, including its height, width, single- or double-layer structure, and upper-surface shape. For visual variation, each embodiment is assigned a balanced low-saturation multi-color palette with 6–8 colors, which is applied to the arm, finger, and plank meshes; fingertip colors are also varied across embodiments. Figure 9 reports post-generation statistics with histograms of arm scale factors, gripper dimensions, contact-plank geometry, and color-hue coverage. These statistics show that the generated embodiments span diverse arm geometries, gripper shapes, contact surfaces, and appearances while sharing a common Franka-style tabletop manipulation setup. The seven commercially used test robots are imported as external held-out targets and are excluded from this procedurally generated source pool. 



<!-- Start of picture text -->
500<br>0<br>0.0 0.2 0.4 0.6 0.8 1.0<br>HSV saturation S<br>Unique source colors<br><!-- End of picture text -->

Figure 8: HSV color distribution of the procedurally generated source palette versus the appearanceonly evaluation targets. The held-out green and logo-orange colors lie outside the source palette primarily along the saturation dimension. 

The appearance-only evaluation targets are near the source visual distribution but are not directly included in the pretraining pool. In HSV space, the distinguishing green and logo-orange colors have saturation values of 0.77 and 0.99, respectively, compared with a maximum saturation of 0.46 in the generated source palette. Thus, these targets are held out primarily along the saturation dimension, as shown in Figure 8. Under our exposure-based definition, they satisfy strict zero-shot transfer because the target appearances are absent from all training data, although they should not be interpreted as strongly out-of-distribution appearance shifts. 

**Domain randomization.** Inspired by prior sim-to-real studies [20, 8, 68, 69], we apply extensive domain randomization during pretraining to reduce the sim-to-real gap and prevent the policy from overfitting to simulator-specific visual or physical regularities. For visual appearance, we use more than 1,000 materials to randomize the background and tabletop textures. We also randomize illumination intensity, color, type, the number of lights, and light positions, covering a broad range of lighting conditions that can arise in real robot workspaces. For objects and scene layouts, we filter Objaverse to obtain more than 10,000 manipulation objects for pretraining. We further randomize tabletop height, size, and pose, and randomly drop distractor objects into the scene to approximate cluttered real-world setups. For camera configuration, we randomize the extrinsics of both thirdperson and wrist cameras. To emulate the diverse camera placements found in real datasets, the 

20 

Table 9: Detailed RQ1 simulation task progress for serving sausages. Rows report individual robots, while columns vary the state/action representation. Each entry reports run-1/ run-2/ run-3 task <u>progress (%).</u> 

|**Target embodiment**|**Abs. EEF / World-Delta**|**Abs. EEF / EEF-Delta**|**EEF-Delta / World-Delta**|**EEF-Delta / EEF-Delta**|
|---|---|---|---|---|
|Franka|80.8/ 83.8/ 79.8|89.9/ 82.8/ 89.9|85.9/ 80.8/ 86.9|84.8/ 91.9/ 90.9|
|Franka+logo|83.8/ 79.8/ 82.8|83.8/ 90.9/ 93.9|86.9/ 85.9/ 86.9|88.9/ 85.9/ 86.9|
|Franka+green|84.8/ 83.8/ 82.8|87.9/ 88.9/ 89.9|81.8/ 82.8/ 86.9|85.9/ 91.9/ 86.9|
|Franka+UMI|79.8/ 78.8/ 77.8|83.8/ 90.9/ 83.8|70.7/ 80.8/ 76.8|74.8/ 81.8/ 86.9|
|UR5e+Franka|65.7/ 68.7/ 70.7|70.7/ 69.7/ 70.7|76.8/ 81.8/ 74.8|78.8/ 81.8/ 75.8|
|Google+Franka|0.0/ 0.0/ 0.0|2.0/ 1.0/ 1.0|54.5/ 53.5/ 53.5|50.5/ 58.6/ 56.6|
|UR5e+UMI|69.7/ 60.6/ 58.6|69.7/ 76.8/ 75.8|69.7/ 62.7/ 59.6|69.7/ 69.7/ 69.7|
|GoogleRobot|1.0/ 1.0/ 0.0|0.0/ 1.0/ 0.0|59.6/ 58.6/ 64.7|54.5/ 55.6/ 53.5|



Table 10: Detailed RQ1 simulation task progress for stacking bowls. Rows report individual robots, while columns vary the state/action representation. Each entry reports run-1/ run-2/ run-3 task <u>progress (%).</u> 

|**Target embodiment**|**Abs. EEF / World-Delta**|**Abs. EEF / EEF-Delta**|**EEF-Delta / World-Delta**|**EEF-Delta / EEF-Delta**|
|---|---|---|---|---|
|Franka|82.0/ 90.0/ 87.0|95.0/ 93.0/ 89.0|84.0/ 84.0/ 88.0|91.0/ 87.0/ 86.0|
|Franka+logo|87.0/ 83.0/ 83.0|95.0/ 87.0/ 89.0|87.0/ 86.0/ 89.0|79.0/ 88.0/ 87.0|
|Franka+green|87.0/ 89.0/ 78.0|95.0/ 85.0/ 85.0|82.0/ 82.0/ 84.0|87.0/ 89.0/ 90.0|
|Franka+UMI|77.0/ 83.0/ 77.0|86.0/ 86.0/ 81.0|82.0/ 73.0/ 75.0|73.0/ 77.0/ 69.0|
|UR5e+Franka|78.0/ 75.0/ 77.0|71.0/ 79.0/ 73.0|72.0/ 68.0/ 65.0|83.0/ 83.0/ 85.0|
|Google+Franka|0.0/ 0.0/ 2.0|0.0/ 0.0/ 0.0|64.0/ 66.0/ 58.0|77.0/ 70.0/ 73.0|
|UR5e+UMI|70.0/ 64.0/ 61.0|77.0/ 65.0/ 71.0|62.0/ 53.0/ 50.0|74.0/ 74.0/ 69.0|
|GoogleRobot|0.0/ 1.0/ 0.0|0.0/ 0.0/ 0.0|41.0/ 55.0/ 48.0|29.0/ 54.0/ 54.0|



Table 11: Detailed RQ1 simulation task progress for stacking cubes. Rows report individual robots, while columns vary the state/action representation. Each entry reports run-1/ run-2/ run-3 task <u>progress (%).</u> 

|**Target embodiment**|**Abs. EEF / World-Delta**|**Abs. EEF / EEF-Delta**|**EEF-Delta / World-Delta**|**EEF-Delta / EEF-Delta**|
|---|---|---|---|---|
|Franka|94.0/ 98.0/ 94.0|97.0/ 99.0/ 98.0|97.0/ 93.0/ 91.0|98.0/ 98.0/ 96.0|
|Franka+logo|92.0/ 97.0/ 90.0|99.0/ 98.0/ 98.0|95.0/ 93.0/ 93.0|96.0/ 98.0/ 96.0|
|Franka+green|87.0/ 92.0/ 90.0|96.0/ 92.0/ 98.0|97.0/ 91.0/ 93.0|94.0/ 93.0/ 96.0|
|Franka+UMI|95.0/ 89.0/ 86.0|93.0/ 98.0/ 91.0|82.0/ 79.0/ 75.7|77.0/ 81.0/ 84.0|
|UR5e+Franka|79.0/ 82.0/ 81.0|89.0/ 92.0/ 91.0|75.0/ 88.0/ 79.0|71.0/ 77.0/ 84.0|
|Google+Franka|2.0/ 10.0/ 2.0|3.0/ 2.0/ 3.0|79.0/ 74.0/ 74.0|79.0/ 75.0/ 78.0|
|UR5e+UMI|79.0/ 71.0/ 67.0|84.0/ 80.0/ 85.0|59.0/ 58.0/ 61.0|69.0/ 63.0/ 66.0|
|GoogleRobot|2.0/ 2.0/ 2.0|6.0/ 2.0/ 2.0|70.0/ 57.0/ 63.0|59.0/ 47.0/ 56.0|



Table 12: Detailed RQ2 simulation task progress for serving sausages. Rows report individual robots, while columns vary the number of non-target source embodiments in the fixed-budget pretraining pool. Each entry reports run-1/ run-2/ run-3 task progress (%). 

|**Target embodiment**|**1 source**|**8 sources**|**512 sources**|
|---|---|---|---|
|Franka|84.8/ 80.8/ 81.8|82.8/ 81.8/ 85.9|84.8/ 91.9/ 90.9|
|Franka+logo|80.8/ 76.8/ 84.8|84.8/ 80.8/ 81.8|88.9/ 85.9/ 86.9|
|Franka+green|78.8/ 81.8/ 79.8|80.8/ 84.8/ 85.9|85.9/ 91.9/ 86.9|
|Franka+UMI|53.5/ 53.5/ 61.6|72.7/ 75.8/ 77.8|74.8/ 81.8/ 86.9|
|UR5e+Franka|40.4/ 50.5/ 45.5|52.5/ 52.5/ 58.6|78.8/ 81.8/ 75.8|
|Google+Franka|41.4/ 45.5/ 45.5|56.6/ 41.4/ 53.5|50.5/ 58.6/ 56.6|
|UR5e+UMI|40.4/ 33.3/ 40.4|63.6/ 54.5/ 56.6|69.7/ 69.7/ 69.7|
|GoogleRobot|31.3/ 28.3/ 26.3|47.5/ 45.5/ 43.4|54.5/ 55.6/ 53.5|



Table 13: Detailed RQ2 simulation task progress for stacking bowls. Rows report individual robots, while columns vary the number of non-target source embodiments in the fixed-budget pretraining pool. Each entry reports run-1/ run-2/ run-3 task progress (%). 

|**Target embodiment**|**1 source**|**8 sources**|**512 sources**|
|---|---|---|---|
|Franka|87.0/ 81.0/ 84.0|92.0/ 87.0/ 85.0|91.0/ 87.0/ 86.0|
|Franka+logo|84.0/ 84.0/ 83.0|89.0/ 87.0/ 93.0|79.0/ 88.0/ 87.0|
|Franka+green|85.0/ 85.0/ 86.0|86.0/ 86.0/ 86.0|87.0/ 89.0/ 90.0|
|Franka+UMI|64.0/ 65.0/ 58.0|68.0/ 71.0/ 67.0|73.0/ 77.0/ 69.0|
|UR5e+Franka|57.0/ 60.0/ 56.0|76.0/ 73.0/ 76.0|83.0/ 83.0/ 85.0|
|Google+Franka|55.0/ 64.0/ 61.0|70.0/ 71.0/ 62.0|77.0/ 70.0/ 73.0|
|UR5e+UMI|51.0/ 49.0/ 39.0|63.0/ 66.0/ 64.0|74.0/ 74.0/ 69.0|
|GoogleRobot|15.0/ 28.0/ 29.0|45.0/ 35.0/ 49.0|29.0/ 54.0/ 54.0|



21 

Table 14: Detailed RQ2 simulation task progress for stacking cubes. Rows report individual robots, while columns vary the number of non-target source embodiments in the fixed-budget pretraining pool. Each entry reports run-1/ run-2/ run-3 task progress (%). 

|**Target embodiment**|**1 source**|**8 sources**|**512 sources**|
|---|---|---|---|
|Franka|93.0/ 92.0/ 91.0|95.0/ 95.0/ 96.0|98.0/ 98.0/ 96.0|
|Franka+logo|89.0/ 87.0/ 93.0|94.0/ 98.0/ 95.0|96.0/ 98.0/ 96.0|
|Franka+green|91.0/ 89.0/ 87.0|96.0/ 97.0/ 98.0|94.0/ 93.0/ 96.0|
|Franka+UMI|53.0/ 51.0/ 59.0|86.0/ 81.0/ 87.0|77.0/ 81.0/ 84.0|
|UR5e+Franka|55.0/ 59.0/ 60.0|68.0/ 69.0/ 63.0|71.0/ 77.0/ 84.0|
|Google+Franka|60.0/ 64.0/ 59.0|66.0/ 69.0/ 67.0|79.0/ 75.0/ 78.0|
|UR5e+UMI|39.0/ 36.0/ 33.0|61.0/ 55.0/ 55.0|69.0/ 63.0/ 66.0|
|GoogleRobot|39.0/ 35.0/ 28.0|43.0/ 46.0/ 52.0|59.0/ 47.0/ 56.0|



Table 15: Detailed RQ3 simulation task progress for serving sausages. Rows report individual robots, while columns vary the auxiliary co-training objective. Each entry reports run-1/ run-2/ run3 task progress (%). 

|**Target embodiment**|**No co-training**|**LAP(EEF frame) **|**LAP(both frames)**|**Subgoal**|**BBox**|
|---|---|---|---|---|---|
|Franka|84.8/ 91.9/ 90.9|90.9/ 84.8/ 91.9|88.9/ 91.9/ 91.9|91.9/ 95.0/ 93.9|85.9/ 91.9/ 87.9|
|Franka+logo|88.9/ 85.9/ 86.9|87.9/ 85.9/ 86.9|85.9/ 87.9/ 90.9|97.0/ 91.9/ 95.0|92.9/ 92.9/ 91.9|
|Franka+green|85.9/ 91.9/ 86.9|80.8/ 87.9/ 80.8|87.9/ 86.9/ 89.9|92.9/ 96.0/ 95.0|88.9/ 88.9/ 88.9|
|Franka+UMI|74.8/ 81.8/ 86.9|84.8/ 82.8/ 85.9|76.8/ 79.8/ 76.8|84.8/ 88.9/ 93.9|85.9/ 86.9/ 88.9|
|UR5e+Franka|78.8/ 81.8/ 75.8|78.8/ 84.8/ 81.8|72.7/ 76.8/ 68.7|89.9/ 91.9/ 92.9|84.8/ 83.8/ 83.8|
|Google+Franka|50.5/ 58.6/ 56.6|77.8/ 74.8/ 73.7|58.6/ 58.6/ 61.6|68.7/ 69.7/ 70.7|71.7/ 72.7/ 64.7|
|UR5e+UMI|69.7/ 69.7/ 69.7|77.8/ 72.7/ 74.8|70.7/ 74.8/ 70.7|80.8/ 78.8/ 79.8|79.8/ 85.9/ 73.7|
|GoogleRobot|54.5/ 55.6/ 53.5|67.7/ 61.6/ 68.7|62.6/ 50.5/ 55.6|62.6/ 62.6/ 64.7|59.6/ 69.7/ 57.6|



Table 16: Detailed RQ3 simulation task progress for stacking bowls. Rows report individual robots, while columns vary the auxiliary co-training objective. Each entry reports run-1/ run-2/ run-3 task <u>progress (%).</u> 

|**Target embodiment**|**No co-training**|**LAP(EEF frame) **|**LAP(both frames)**|**Subgoal**|**BBox**|
|---|---|---|---|---|---|
|Franka|91.0/ 87.0/ 86.0|84.0/ 87.0/ 89.0|81.0/ 86.0/ 87.0|80.0/ 82.0/ 86.0|87.0/ 90.0/ 86.0|
|Franka+logo|79.0/ 88.0/ 87.0|85.0/ 89.0/ 84.0|92.0/ 89.0/ 84.0|85.0/ 79.0/ 84.0|81.0/ 89.0/ 85.0|
|Franka+green|87.0/ 89.0/ 90.0|88.0/ 85.0/ 85.0|82.0/ 82.0/ 83.0|85.0/ 76.0/ 84.0|89.0/ 83.0/ 86.0|
|Franka+UMI|73.0/ 77.0/ 69.0|86.0/ 80.0/ 82.0|74.0/ 74.0/ 75.0|78.0/ 82.0/ 86.0|79.0/ 79.0/ 82.0|
|UR5e+Franka|83.0/ 83.0/ 85.0|83.0/ 86.0/ 80.0|73.0/ 84.0/ 82.0|83.0/ 80.0/ 86.0|79.0/ 76.0/ 79.0|
|Google+Franka|77.0/ 70.0/ 73.0|69.0/ 68.0/ 69.0|76.0/ 72.0/ 61.0|73.0/ 68.0/ 60.0|76.0/ 83.0/ 74.0|
|UR5e+UMI|74.0/ 74.0/ 69.0|73.0/ 80.0/ 71.0|70.0/ 65.0/ 68.0|83.0/ 72.0/ 81.0|74.0/ 72.0/ 80.0|
|GoogleRobot|29.0/ 54.0/ 54.0|65.0/ 68.0/ 58.0|62.0/ 55.0/ 65.0|31.0/ 55.0/ 57.0|64.0/ 67.0/ 64.0|



Table 17: Detailed RQ3 simulation task progress for stacking cubes. Rows report individual robots, while columns vary the auxiliary co-training objective. Each entry reports run-1/ run-2/ run-3 task <u>progress (%).</u> 

|**Target embodiment**|**No co-training**|**LAP(EEF frame) **|**LAP(both frames)**|**Subgoal**|**BBox**|
|---|---|---|---|---|---|
|Franka|98.0/ 98.0/ 96.0|96.0/ 96.0/ 95.0|93.0/ 91.0/ 95.0|89.0/ 90.0/ 94.0|96.0/ 95.0/ 94.0|
|Franka+logo|96.0/ 98.0/ 96.0|94.0/ 94.0/ 96.0|91.0/ 94.0/ 95.0|94.0/ 94.0/ 93.0|96.0/ 95.0/ 94.0|
|Franka+green|94.0/ 93.0/ 96.0|97.0/ 91.0/ 97.0|92.0/ 89.0/ 92.0|92.0/ 98.0/ 94.0|96.0/ 94.0/ 92.0|
|Franka+UMI|77.0/ 81.0/ 84.0|95.0/ 93.0/ 91.0|86.0/ 85.0/ 91.0|91.0/ 92.0/ 90.0|94.0/ 93.0/ 94.0|
|UR5e+Franka|71.0/ 77.0/ 84.0|92.0/ 90.0/ 88.0|86.0/ 88.0/ 83.0|80.0/ 83.0/ 83.0|91.0/ 84.0/ 89.0|
|Google+Franka|79.0/ 75.0/ 78.0|74.0/ 79.0/ 77.0|68.0/ 77.0/ 82.0|73.0/ 77.0/ 70.0|84.0/ 80.0/ 85.0|
|UR5e+UMI|69.0/ 63.0/ 66.0|86.0/ 84.0/ 86.0|87.0/ 80.0/ 86.0|86.0/ 89.0/ 87.0|83.0/ 89.0/ 87.0|
|GoogleRobot|59.0/ 47.0/ 56.0|56.0/ 58.0/ 60.0|66.0/ 73.0/ 70.0|75.0/ 71.0/ 71.0|61.0/ 59.0/ 68.0|



Table 18: Detailed RQ4 simulation task progress for serving sausages. Rows vary the targetembodiment pretraining exposure ratio and the target post-training oracle; columns report the two target embodiments used in the sweep. Each entry reports run-1/ run-2/ run-3 task progress (%). 

|**Setting**|**UR5e+UMI**|**GoogleRobot**|
|---|---|---|
|Strict zero-shot (0%)|69.7/ 69.7/ 69.7|54.5/ 55.6/ 53.5|
|Pretrain-exposed (5%)|75.8/ 73.7/ 74.8|68.7/ 64.7/ 68.7|
|Pretrain-exposed (30%)|86.9/ 83.8/ 75.8|71.7/ 72.7/ 67.7|
|Pretrain-exposed (100%)|89.9/ 84.8/ 85.9|73.7/ 68.7/ 71.7|
|Targetpost-trainingoracle|83.8/ 78.8/ 80.8|69.7/ 62.6/ 60.6|



22 

Table 19: Detailed RQ4 simulation task progress for stacking bowls. Rows vary the targetembodiment pretraining exposure ratio and the target post-training oracle; columns report the two target embodiments used in the sweep. Each entry reports run-1/ run-2/ run-3 task progress (%). 

|**Setting**|**UR5e+UMI**|**GoogleRobot**|
|---|---|---|
|Strict zero-shot (0%)|74.0/ 74.0/ 69.0|29.0/ 54.0/ 54.0|
|Pretrain-exposed (5%)|80.0/ 77.0/ 80.0|70.0/ 61.0/ 67.0|
|Pretrain-exposed (30%)|87.0/ 82.0/ 85.0|70.0/ 69.0/ 59.0|
|Pretrain-exposed (100%)|88.0/ 81.0/ 86.0|71.0/ 61.0/ 71.0|
|Targetpost-trainingoracle|97.0/ 92.0/ 92.0|81.0/ 79.0/ 76.0|



Table 20: Detailed RQ4 simulation task progress for stacking cubes. Rows vary the targetembodiment pretraining exposure ratio and the target post-training oracle; columns report the two target embodiments used in the sweep. Each entry reports run-1/ run-2/ run-3 task progress (%). 

|**Setting**|**UR5e+UMI**|**GoogleRobot**|
|---|---|---|
|Strict zero-shot (0%)|69.0/ 63.0/ 66.0|59.0/ 47.0/ 56.0|
|Pretrain-exposed (5%)|87.0/ 79.0/ 80.0|75.0/ 68.0/ 77.0|
|Pretrain-exposed (30%)|91.0/ 90.0/ 87.0|76.0/ 68.0/ 67.0|
|Pretrain-exposed (100%)|92.0/ 84.0/ 84.0|77.0/ 71.0/ 68.0|
|Targetpost-trainingoracle|98.0/ 95.0/ 93.0|90.0/ 88.0/ 89.0|



Table 21: Detailed RQ1 real-world task progress for _open the fryer_ . Rows report individual robots, while columns vary the state/action representation. Each entry reports mean task progress (%) over <u>10 rollouts.</u> 

|**Target embodiment**|**Abs. EEF / World-Delta**|**Abs. EEF / EEF-Delta**|**EEF-Delta / World-Delta**<br>**EEF-Delta / EEF-Delta**|
|---|---|---|---|
|Franka+Robotiq|90.0|95.0|80.0<br>100.0|
|Franka+Robotiq (CoRL logo)|90.0|90.0|90.0<br>100.0|
|Franka+Umi|80.0|85.0|90.0<br>100.0|
|UR|55.0|35.0|55.0<br>90.0|
|Humanoid robot|0.0|0.0|35.0<br>100.0|
|UR+Umi|60.0|50.0|60.0<br>80.0|
|Humanoid robot+Umi|0.0|0.0|55.0<br>85.0|
|Piper-on-Quadruped|35.0|0.0|40.0<br>80.0|



Table 22: Detailed RQ1 real-world task progress for _water the flower_ . Rows report individual robots, while columns vary the state/action representation. Each entry reports mean task progress (%) over 

### <u>10 rollouts.</u> 

|**Target embodiment**|**Abs. EEF / World-Delta**|**Abs. EEF / EEF-Delta**|**EEF-Delta / World-Delta**<br>**EE**|**F-Delta / EEF-Delta**|
|---|---|---|---|---|
|Franka+Robotiq|85.0|85.0|90.0|100.0|
|Franka+Robotiq (CoRL logo)|85.0|90.0|65.0|95.0|
|Franka+Umi|55.0|75.0|30.0|85.0|
|UR|40.0|85.0|70.0|85.0|
|Humanoid robot|0.0|0.0|40.0|80.0|
|UR+Umi|55.0|60.0|60.0|75.0|
|Humanoid robot+Umi|0.0|0.0|25.0|65.0|
|Piper-on-Quadruped|20.0|20.0|30.0|90.0|



Table 23: Detailed RQ2 real-world task progress for _open the fryer_ . Rows report individual robots, while columns vary the number of pretraining embodiments _|E_ pre _|_ . Each entry reports mean task progress (%) over 10 rollouts. 

|**Target embodiment**|**1 source**|**8 sources**|**512 sources**|
|---|---|---|---|
|Franka+Robotiq|80.0|100.0|100.0|
|Franka+Robotiq (CoRL logo)|90.0|100.0|100.0|
|Franka+Umi|80.0|100.0|100.0|
|UR|75.0|75.0|90.0|
|Humanoid robot|95.0|90.0|100.0|
|UR+Umi|65.0|75.0|80.0|
|Humanoid robot+Umi|75.0|70.0|85.0|
|Piper-on-Quadruped|90.0|45.0|80.0|



23 



Figure 9: Empirical distributions of the procedurally generated source-embodiment pool. We summarize the arm-scale factors, sampled gripper dimensions, contact-plank geometry, and color-hue coverage for the 512 generated Franka-style tabletop embodiments used for source pretraining. 

third-person camera is randomly rotated around the tabletop center by [ _−_ 150<sup>_◦_</sup> _,_ 150<sup>_◦_</sup> ], with a pitch angle sampled from [15<sup>_◦_</sup> _,_ 45<sup>_◦_</sup> ] and a distance to the tabletop center sampled from [0 _._ 8 _,_ 1 _._ 4] m. For the wrist view, we randomize the camera extrinsics within a 0 _._ 1 m cube along the gripper direction while rejecting configurations in which the view is occluded by the robot arm itself. Finally, we randomize physical parameters such as object mass and friction coefficients. In contrast, post-training data are constructed to mimic small-scale real-world data collection: for each downstream setting, we keep a single scene, object combination, lighting condition, and camera placement fixed, and evaluate the policy under the corresponding matched condition. 

## **E Embodiment Shift Taxonomy** 

### **E.1 Simulation** 

**Franka.** This is the simulation source embodiment, consisting of a Franka-style 7-DoF arm and a Franka two-finger hand; it defines the reference morphology, kinematics, and end-effector interface for the simulated taxonomy. 

**Franka with CoRL logo.** This target is an appearance-only shift: it uses the same Franka arm and Franka hand as the source, but changes the robot texture by adding the CoRL logo. 

**Franka with green fingers.** This target is also an appearance-only shift: the arm, hand geometry, and kinematics remain those of the source Franka, while only the visual appearance of the fingers is changed. 

**Franka with UMI gripper.** This target is a gripper-only shift: it keeps the Franka arm fixed and replaces the Franka hand with a UMI-style two-finger gripper, isolating the effect of end-effector geometry. 

**UR5e with Franka hand.** This target is an arm-only shift: it replaces the Franka arm with a 6-DoF UR5e arm while retaining a Franka-style two-finger hand, changing arm morphology and kinematics without changing the end effector. 

**GoogleRobot with Franka hand.** This target is an arm-only shift: it uses the GoogleRobot mobilemanipulator platform with its 7-DoF arm and a Franka-style two-finger hand, changing the manipulator morphology while preserving the evaluated end-effector geometry. Although GoogleRobot has a mobile base, base states and commands are excluded from the model inputs and outputs to keep the evaluated control interface consistent. 

**UR5e with UMI gripper.** This target is a full-embodiment shift: both the arm and the end effector differ from the source, combining a 6-DoF UR5e arm with a UMI-style two-finger gripper. 

**GoogleRobot.** This target is a full-embodiment shift: it uses the GoogleRobot mobile-manipulator platform with a 7-DoF arm and native two-finger gripper, thereby changing both manipulator morphology and end-effector geometry. As above, mobile-base states and commands are not included in the model inputs or outputs. 

24 

### **E.2 Real World** 

**Franka with Robotiq gripper.** This is the real-world source embodiment, composed of a Franka arm and a Robotiq two-finger gripper. 

**Franka with CoRL logo and Robotiq gripper.** This target is an appearance-only shift: it keeps the Franka arm and Robotiq gripper unchanged while modifying only the visible robot appearance with a CoRL logo. 

**Franka with UMI gripper.** This target is a gripper-only shift: it keeps the Franka arm fixed and replaces the Robotiq gripper with a UMI-style two-finger gripper. 

**UR5e with Robotiq gripper.** This target is an arm-only shift: it replaces the Franka arm with a 6-DoF UR5e arm while retaining the Robotiq two-finger gripper, isolating the change in arm morphology and kinematics. 

**Humanoid robot.** This target is a full-embodiment shift: it uses a mobile humanoid platform with a 7-DoF right arm and native two-finger gripper, changing both the manipulator morphology and end-effector geometry relative to the Franka-Robotiq source. Although the platform is a dual-arm mobile robot, we retain only right-arm information in the model inputs and outputs; the left arm is fixed in a natural lowered pose, and mobile-base states and commands are excluded. 

**UR5e with UMI gripper.** This target is a full-embodiment shift: it combines a 6-DoF UR5e arm with a UMI-style two-finger gripper, so both the arm and end-effector geometry differ from the source embodiment. 

**Humanoid robot with UMI gripper.** This target is a full-embodiment shift: it keeps the mobile humanoid platform and 7-DoF right arm while replacing the native gripper with a UMI-style twofinger gripper. As with the native-gripper humanoid setting, only the right arm is exposed to the policy, the left arm is fixed in a natural lowered pose, and mobile-base states and commands are excluded from the model interface. 

**Piper-on-Quadruped.** This target is a full-embodiment shift: it mounts a 6-DoF Piper robotic arm with its matched two-finger gripper on the back of a Unitree B2 quadruped platform, introducing a legged mobile base together with a different arm and end effector. The quadruped base is not represented in the model inputs or outputs, so evaluation remains focused on the manipulator and gripper control interface. 

## **F Experimental Setup Details** 

### **F.1 Simulation Platform** 

We use MuJoCo for physics simulation and IsaacSim for rendering throughout the simulation experiments. 

### **F.2 Camera Calibration Details** 

To reduce embodiment-dependent visual discrepancies, we calibrate camera poses so that, under a canonical robot pose, the image-plane gripper distribution remains close to the visual distribution observed during post-training. We define a shared tabletop task frame as an environment-anchored coordinate frame fixed to the tabletop workspace, rather than to any particular robot embodiment. Its horizontal axes lie on the tabletop plane, and its vertical axis is aligned with the table normal. Each robot expresses object poses, goal poses, and task-relevant spatial quantities in this common frame through a calibrated transformation from its own base frame. This convention decouples task specification and camera alignment from embodiment-specific base coordinates and kinematic structures. 

In simulation, all third-person cameras are strictly aligned with respect to this task frame. For each evaluation trajectory, the extrinsic pose of the third-person camera is identical across robots when expressed in the corresponding task frame. This ensures that visual differences across embodiments primarily reflect robot appearance and motion rather than avoidable viewpoint changes. For wristmounted cameras, we apply an analogous alignment rule in the gripper frame: the camera extrinsics 

25 

are matched across robots when expressed relative to each robot’s gripper coordinate frame. For real-world experiments, we follow the same principle during physical camera setup and calibration, aligning the cameras so that the visible workspace and the gripper’s image-plane coverage are approximately consistent across embodiments. 

### **F.3 Environment and Object Set Details** 

To isolate embodiment shift from variation in the workspace, we standardize the evaluation environment as much as possible across robots. In simulation, all robot embodiments are evaluated under strictly matched scene configurations: the lighting setup, background material, tabletop material, tabletop workspace, and table arrangement are kept identical for the corresponding test condition. In the real-world experiments, we similarly standardize the physical setup by enclosing the experimental area with black cloth and covering the table with a gray tablecloth, while keeping the table arrangement and workspace layout aligned across robots. These controls reduce appearance and layout differences that are unrelated to the robot embodiment being evaluated. 

We also keep the object set fixed between post-training and evaluation. For both simulation and real-world experiments, all objects used in the same evaluation trial are identical across robot embodiments. Moreover, the evaluation objects are exactly the same object instances as those used in the corresponding post-training data, so no additional object-identity shift is introduced at test time. This design ensures that differences in zero-shot performance primarily reflect embodiment transfer rather than changes in objects, scene layout, or workspace appearance. 

### **F.4 Details of Metrics** 

Each rollout is assigned a task-progress score _s ∈{_ 0 _,_ 0 _._ 5 _,_ 1 _}_ . A rollout receives _s_ = 1 if it fully completes the task, _s_ = 0 _._ 5 if the robot reaches the necessary interaction subgoal by contacting the task-relevant object, and _s_ = 0 otherwise. For example, in the serving-sausages task, contacting the sausage receives half credit, while successfully serving the sausage receives full credit. All reported progress values are expressed as percentages, i.e., 100 _s_ averaged over rollouts. 

We use a macro-averaging protocol so that each task and each embodiment-shift category contributes uniformly to the aggregate score. For a given training run, model, and robot embodiment, we first average progress over the downstream tasks. Within each of the four shift categories, we then average these per-robot task means over all target robots in that category. The overall average is the arithmetic mean of the four category scores. For simulation experiments with three independent training runs, we compute the same aggregation separately for each run and report the mean over the three run-level aggregates; when an uncertainty term is shown, it is the standard deviation across these three run-level aggregates. 

### **F.5 Downstream Tasks** 

For real-world post-training, we use two tasks, opening an air fryer and watering a plant, with 50 demonstrations per task. For the simulation benchmark, we use three downstream tasks, serving sausages, stacking bowls, and stacking cubes, with 40K trajectories per task. To ensure fair comparison, downstream post-training data is collected on the same source Franka arm. 

## **G Auxiliary Co-Training Objective Specifications** 

All auxiliary co-training variants use the same visual input as the imitation policy: the third view and wrist view images, the task caption _ℓ_ , and the current state when required. For each variant, the data loader samples imitation-action examples and auxiliary-query examples with weights 0.75 and 0.25, respectively. The imitation branch remains supervised by the flow-matching action loss, while auxiliary responses are supervised with next-token cross entropy. The construction rules for the auxiliary answer strings are described below, and Table 25 summarizes the exact query templates and example response formats used by the official RQ3 models. 

We follow [56] to generate the auxiliary language targets for the language-action objectives. Each target is generated from the delta action vector. In the EEF-frame variant, the action is represented as [∆ _x,_ ∆ _y,_ ∆ _z,_ ∆ _r,_ ∆ _p,_ ∆ _ψ, g_ ], with translations converted to centimeters and rotations to degrees, 

26 

then rounded to integers. Nonzero translation components are rendered as language clauses, nonzero rotation components are rendered as rotation clauses, and the gripper command is rendered as an open-or-close clause. The EEF-frame translation directions are ordered as up/down, right/left, and forward/backward. The mixed-frame variant uses the same clause-level format, with the prompt specifying whether the delta is interpreted in the end-effector frame or the robot base frame. For base-frame labels, translation directions are ordered as forward/backward, left/right, and up/down; roll, pitch, and yaw use right/left, back/forward, and counterclockwise/clockwise, respectively. This variant tests whether mixing frame semantics in the auxiliary language target helps or hurts the EEF-centered policy representation. 

For the subgoal-prediction objective, the answer is selected from the next manipulation-relevant waypoint in the trajectory and encoded as a decimal list in the current EEF coordinate frame. Translation values are formatted to 0.001 precision and rotation values to 0.01 precision. 

For the task-conditioned bounding-box objective, the answer contains one object box per input view, ordered by the configured camera list. Each box is represented in XYXY order, normalized from image pixels to [ _−_ 1 _,_ 1], and emitted as quantized coordinate tokens using 256 uniform bins. The task-conditioned wording requires the model to identify the object relevant to the instruction rather than predict generic object boxes. 

## **H Additional RQ4 Target-Exposure Details** 

RQ4 uses UR5eUMI and GoogleRobot as representative target embodiments for the pretrainexposed zero-shot setting. UR5eUMI tests transfer to an arm with different degrees of freedom from the source embodiments, while GoogleRobot tests transfer to a mobile-manipulator platform with more humanoid-like morphology. For each target, we construct a separate 640K-trajectory pick-and-place pretraining pool. At each exposure ratio, a controlled fraction of non-target source data is replaced with target-embodiment pretraining data, so the total pretraining budget remains fixed. The target embodiment’s downstream post-training tasks are never used in these scaling experiments; thus the evaluation remains zero-shot with respect to target-embodiment downstream tasks, but differs from strict zero-shot transfer because the target embodiment appears during pretraining. 

The target-embodiment post-training oracle is initialized from the same shared pretrained checkpoint as the zero-shot variants and subsequently post-trained on _D_ ( _et, T_ post). It is not a zero-shot setting and serves only as a performance upper reference. In the RQ4 target-exposure plot, the pretrainexposed curves remain below this oracle, indicating that target-embodiment pretraining mitigates but does not eliminate the need for target-task adaptation. 

## **I Model Training Details** 

We use a Mixture-of-Transformers (MoT) backbone following _π_ 0 _._ 5 [14], initializing the visionlanguage stream from pretrained VLM weights and coupling it with a separate action expert through layer-wise attention. The visual input contains the current observation image resized to 224 _×_ 224, and proprioceptive inputs use a 4-step state history with 99th-percentile normalization for continuous states and actions. The policy predicts a 12-step action chunk. Action outputs are supervised with a flow-matching imitation loss; auxiliary co-training targets, when enabled, use cross-entropy loss with weight 0.25 relative to the imitation loss. Pretraining uses batch size 320 on 16 H800 GPUs for 120K steps, and post-training uses batch size 160 on 8 H800 GPUs for 100K steps. 

## **J Policy Inference and Action Execution Details** 

At inference time, each request contains the language instruction _ℓ_ , two RGB observations (one wrist-mounted view and one third-person view), and a four-entry EEF state history. Images are resized to 224 _×_ 224. We use the notation from the main-text state-action representation table: _b_ denotes the robot base frame, _et_ denotes the gripper frame at time _t_ , _Tx,y_ denotes the transform from 

27 

_R p_ frame _x_ to frame _y_ , and _V_ ( _T_ ) = [ _p,_ Euler( _R_ )] for _T_ = 0 1 . The gripper state or command is � � 

appended as a scalar. 

The server interface can be written as a request–response map. For each setting, the request specifies the language instruction, images, and one of the two EEF state histories, while the response is a 12step action chunk, 



Here ∆ _ϕj_ uses the same Euler-angle convention as _V_ ( _T_ ), and the continuous gripper output _uj_ is quantized to _{−_ 1 _,_ 0 _,_ +1 _}_ , corresponding to close, no-op, and open. The four request–response forms are: 

### 1. **Abs. EEF state / World-Delta action.** The server request is 



The server response is 

which gives the absolute gripper-frame target by 



2. **Abs. EEF state / EEF-Delta action.** The server request is 



The server response is 



### 3. **EEF-Delta state / World-Delta action.** The server request is 



with 

The server response is 

which gives the absolute gripper-frame target by 



4. **EEF-Delta state / EEF-Delta action.** The server request is 



For efficiency, the server batches up to 10 requests. In our serving setup, one warmed-up policy query on a server with a single NVIDIA GeForce RTX 4090 GPU takes 168 ms and uses 6168 MiB of GPU memory. 

28 

## **K Additional Problem-Setup Details** 

Each demonstration consists of observation–action pairs collected from embodiment _e_ performing task _t_ . Observations include images and language instructions, while proprioceptive states and actions are expressed in the representation specified by each experimental condition. The policy _πθ_ maps the past observation history and task specification to an action chunk, _πθ_ : ( _o<t, ℓ_ ) _�→ at_ : _t_ + _K_ , where _o<t_ is the observation history before the current action step _t_ , _ℓ_ is the language instruction, and _at_ : _t_ + _K_ denotes an action chunk of length _K_ . We use _E_ pre and _T_ pre to denote the embodiments and tasks represented in _D_ pre. Across comparisons, the policy backbone, data budget, optimization protocol, testing tasks, scenes, cameras, and objects are held fixed unless explicitly stated. 

29 

Table 24: Detailed RQ2 real-world task progress for _water the flower_ . Rows report individual robots, while columns vary the number of pretraining embodiments _|E_ pre _|_ . Each entry reports mean task progress (%) over 10 rollouts. 

|**Target embodiment**|**1 source**|**8 sources**|**512 sources**|
|---|---|---|---|
|Franka+Robotiq|80.0|95.0|100.0|
|Franka+Robotiq (CoRL logo)|85.0|85.0|95.0|
|Franka+Umi|30.0|65.0|85.0|
|UR|75.0|35.0|85.0|
|Humanoid robot|85.0|30.0|80.0|
|UR+Umi|55.0|35.0|75.0|
|Humanoid robot+Umi|50.0|30.0|65.0|
|Piper-on-Quadruped|65.0|70.0|90.0|



Table 25: Auxiliary co-training objective specifications. Braces denote fields filled from the trajectory annotation. The target column lists example answer formats. 

|**Objective**|**Prompt template**|**Target format**|
|---|---|---|
|Language-<br>action (EEF<br>frame)|`The task is` _{_`instruction`_}_`.`<br>`Predict the robot’s action in`<br>`the end-effector frame;`<br>`State:`<br>_{_`state tokens`_}_`;`<br>`Answer:`|`move` _{_`direction`_} {_`value`_}_ `cm`;<br>`tilt/rotate` _{_`direction`_} {_`value`_}_<br>`degrees`;`open gripper`;`close gripper`.|
|Language-<br>action (mixed<br>frame)|`The task is` _{_`instruction`_}_`.`<br>`Predict the robot’s action in`<br>`the` _{_`action frame`_}_`; State:`<br>_{_`state tokens`_}_`; Answer:`|`move` _{_`direction`_} {_`value`_}_ `cm`;`rotate`<br>_{_`direction`_} {_`value`_}_ `degrees`;`close`<br>`gripper`.|
|Subgoal<br>prediction|`In order to` _{_`instruction`_}_`,`<br>`what is the next goal pose as`<br>`xyz and rpy?`<br>`Answer:`|`[x,y,z,roll,pitch,yaw]`, e.g.,<br>`[0.012,-0.004,0.085,1.57,0.00,-0.12]`.|
|Task-<br>conditioned<br>BBox|`In order to` _{_`instruction`_}_`,`<br>`what is the bounding box of`<br>`the object to be manipulated`<br>`in the images?`<br>`Answer:`|`[x`<br>`min,y`<br>`min,x`<br>~~`m`~~`ax,y`<br>`max]`per view, e.g.,<br>`[[q`<br>~~`x`~~`min,q`<br>`ymin,q`<br>~~`x`~~`max,q`<br>~~`y`~~`max], ...]`.|



30 

