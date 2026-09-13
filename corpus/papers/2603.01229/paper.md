# **RMBench: Memory-Dependent Robotic Manipulation Benchmark with Insights into Policy Design** 

**Tianxing Chen**<sup>* 1</sup> **Yuran Wang**<sup>* 2 3</sup> **Mingleyang Li**<sup>* 2</sup> **Yan Qin**<sup>* 4</sup> **Hao Shi**<sup>5</sup> **Zixuan Li**<sup>6</sup> **Yifan Hu**<sup>2</sup> **Yingsheng Zhang**<sup>5</sup> **Kaixuan Wang**<sup>1</sup> **Yue Chen**<sup>2</sup> **Hongcheng Wang**<sup>2</sup> **Tianhang Yang**<sup>5</sup> **Junjie Wang**<sup>5</sup> **Tianhang Yang**<sup>2</sup> **Renjing Xu**<sup>4</sup> **Ruihai Wu**<sup>2</sup> **Yao Mu**<sup>7</sup> **Yaodong Yang**<sup>2 3</sup> **Hao Dong**<sup>† 2</sup> **Ping Luo**<sup>† 1</sup> 

_∗_ Equal contribution, _†_ Corresponding authors 

Website: https://RMBench.github.io Code: https://github.com/robotwin-Platform/rmbench 

## **Abstract** 

Robotic manipulation policies have made rapid progress in recent years, yet most existing approaches give limited consideration to memory capabilities. Consequently, they struggle to solve tasks that require reasoning over historical observations and maintaining task-relevant information over time, which are common requirements in real-world manipulation scenarios. Although several memory-aware policies have been proposed, systematic evaluation of memory-dependent manipulation remains underexplored, and the relationship between architectural design choices and memory performance is still not well understood. To address this gap, we introduce **RMBench** , a simulation benchmark comprising 9 manipulation tasks that span multiple levels of memory complexity, enabling systematic evaluation of policy memory capabilities. We further propose **Mem-0** , a modular manipulation policy with explicit memory components designed to support controlled ablation studies. Through extensive simulation and real-world experiments, we identify memoryrelated limitations in existing policies and provide empirical insights into how architectural design choices influence memory performance. 

## **1. Introduction** 

Recent progress in robotic manipulation has demonstrated strong capabilities across a wide range of tasks. Modern policies such as Pi0.6 (Intelligence et al., 2025a) and 

1MMLab@HKU 2PKU 3PsiBot 4HKUST (GZ) 5THU 6SZU 7SJTU. Correspondence to: Tianxing Chen _<_ chentianxing@connect.hku.hk _>_ , Ping Luo _<_ pluo@hku.hk _>_ , Hao Dong _<_ hao.dong@pku.edu.cn _>_ . 

_Preprint. Mar 1, 2026._ 

RDT2 (Team, 2025) achieve impressive performance in flexible object manipulation and fine-grained skills, including complex activities like coffee making. Nevertheless, most existing robotic policies are primarily designed for short-horizon tasks and fine-grained manipulation. These approaches typically rely on a fixed-length window of recent observations, implicitly assuming that the underlying decision process is approximately Markovian. 

In contrast, memory-dependent tasks are ubiquitous in realworld robotic applications. Such tasks are inherently nonMarkovian, as past observations and actions may influence future decisions over extended temporal horizons. Examples include remembering the location of previously placed objects or reasoning over multiple attempts after forgetting a password. These tasks are both challenging and practically important, as they require robots to retain, retrieve, and utilize information beyond short-term sensory inputs. 

Motivated by this challenge, several recent works have begun to explore memory-aware robotic policies. Approaches such as MemoryVLA (Shi et al., 2025), MemER (Sridhar et al., 2025), CronusVLA (Li et al., 2025), and SAM2Act (Fang et al., 2025) incorporate explicit memory mechanisms to address long-horizon and memorydependent decision making. Despite these efforts, the field currently lacks a systematically designed experimental platform for evaluating and analyzing robotic policies under long-term memory requirements. In particular, there is limited understanding of the underlying mechanisms that make memory strategies effective for robotic manipulation. 

Existing benchmarks only partially bridge this gap. MemoryBench (Fang et al., 2025) comprises seven single-arm manipulation tasks that involve memory, yet only three can be reliably reproduced in simulation, and the benchmark provides limited guidance on principled task design. MIKASA (Cherepanov et al., 2025) introduces 32 memoryrelated manipulation tasks, but its formulation is largely tailored to reinforcement learning rather than general im- 

1 

**RMBench: Memory-Dependent Robotic Manipulation Benchmark with Insights into Policy Design** 

itation learning. LIBERO-Long (Liu et al., 2023) offers ten long-horizon tasks, though these tasks do not explicitly demand memory since all task-relevant information remains observable throughout execution. 

To address these limitations, we first introduce Task Memory Complexity, a principled metric for characterizing memory requirements in robotic manipulation tasks. This metric provides a systematic way to classify memory-dependent tasks and serves as a guideline for task design. Based on this formulation, we propose **RMBench** , a robotic manipulation benchmark built on RoboTwin 2.0 platform. RMBench consists of 9 dual-arm manipulation tasks spanning different levels of task memory complexity, enabling large-scale and controlled studies of memory retention and utilization in robotic manipulation. 

Furthermore, we propose **Mem-0** , a novel memory-oriented robotic policy designed with modular components that can be easily replaced or reconfigured. Mem-0 adopts a dualsystem architecture with a task-phase classifier that explicitly distinguishes different stages of a task, allowing structured memory usage across long horizons. Through systematic ablation studies of Mem-0, we analyze which design components are critical for effective memory in robotic manipulation and derive insights for future policy design. 

Our main contributions are summarized as follows: 

- We introduce Task Memory Complexity, a principled metric for categorizing memory-dependent robotic manipulation tasks, and propose **RMBench** , a simulation benchmark comprising 9 memory-centric tasks based on this metric. 

- We propose **Mem-0** , a memory-oriented robotic manipulation policy featuring a dual-system architecture with a task-phase classifier for flexible memory usage. 

- We conduct comprehensive evaluations of representative state-of-the-art policies on RMBench and perform detailed ablation studies of Mem-0, revealing which policy design mechanisms are most beneficial for memory in robotic manipulation. 

## **2. Related Work** 

### **2.1. Robotic Manipulation Benchmarks** 

Physics-based simulators underpin modern robotic manipulation research, and numerous simulation benchmarks have been proposed in recent years. RoboTwin (Mu et al., 2025; Chen et al., 2025a), RoboCasa (Nasiriany et al., 2024), ManiSkill3 (Tao et al., 2025), AutoBio (Lan et al., 2025), UniVTAC (Chen et al., 2026), DexGarmentLab (Wang et al., 2025), BEHAVIOR-1K (Li et al., 2024a), and SIMPLER (Li et al., 2024c) provide diverse manipulation tasks, yet most scenarios emphasize short-horizon interactions or can be 

solved without relying on historical observations. Several benchmarks have begun to consider memory-related manipulation tasks. MemoryBench (Fang et al., 2025) includes a limited number of memory-dependent tasks but suffers from poor reproducibility and lacks clear task design principles. MIKASA (Cherepanov et al., 2025) introduces a larger collection of memory-related tasks, though its design is primarily tailored to reinforcement learning. LIBERO-Long (Liu et al., 2023) and RoboCerebra (Han et al., 2025) features long-horizon tasks, but task-relevant information remains observable throughout execution and therefore does not explicitly require memory. In contrast, RMBench explicitly stratifies memory requirements in manipulation tasks, enabling systematic analysis of memory-based policies across different levels of task difficulty. 

### **2.2. Robotic Manipulation Policies** 

Recent advances in generative models and imitation learning have produced a wide range of robotic manipulation policies that achieve strong performance on individual tasks (Zhao et al., 2023; Chi et al., 2025; Ze et al., 2024; Chen et al., 2025b; Lu et al., 2024; Su et al., 2025). Inspired by large visual foundation models, many approaches adopt VisionLanguage-Action (VLA) formulations (Wen et al., 2025a; Lin et al., 2025; Liang et al., 2025; Shen et al., 2025; Wen et al.; 2025b), where policies are pretrained on large-scale robot datasets and exhibit improved generalization. Representative examples include Pi0.5 and Pi0.6 (Intelligence et al., 2025b;a), RDT2 (Team, 2025), and X-VLA (Zheng et al., 2025), as well as methods that incorporate future observation prediction such as Motus (Bi et al., 2025), CogACT (Li et al., 2024b), and CronusVLA (Li et al., 2025). 

Despite these advances, most existing policies rely on fixed-length observation histories, which limits their ability to selectively retain task-relevant information over long time horizons. This limitation motivates recent memoryaware approaches, including MemoryVLA (Shi et al., 2025), MemER (Sridhar et al., 2025), and SAM2Act (Fang et al., 2025), which explicitly incorporate memory mechanisms for memory-dependent manipulation. Building on this line of work, we propose Mem-0, a modular memory-enabled policy designed to facilitate systematic ablation and analysis of memory components and architectural choices. 

## **3. RMBench** 

In this section, we introduce **Task Memory Complexity** , a principled criterion for characterizing memory requirements in robotic manipulation tasks and guiding benchmark design. Based on this, we propose **RMBench** , a simulation benchmark comprising nine manipulation tasks with varying memory demands, designed to support controlled evaluation across different levels of task memory complexity. 

2 

**RMBench: Memory-Dependent Robotic Manipulation Benchmark with Insights into Policy Design** 























<!-- Start of picture text -->
Observe and Pick Up Put Back Block Rearrange Blocks<br>Swap Blocks Swap T<br>Battery Try Blocks Ranking Try<br>Cover Blocks Press Button<br><!-- End of picture text -->







_Figure 1._ **RMBench Tasks.** We illustrate the nine memory-dependent tasks in RMBench along with their key execution steps. Tasks detailed description are shown in Appendix. A. 

### **3.1. Task Memory Complexity (TMC)** 

Robotic manipulation often operates under partial observability, where the current observation alone may be insufficient to determine task progress or the correct next action without access to past information. Such partial observability may arise from occlusions, delayed effects, or state aliasing. Importantly, the required past information does not necessarily correspond to a contiguous sequence of recent observations, but rather to a small set of task-relevant observations occurring at arbitrary time steps. Existing benchmarks typically assess memory through specific policy architectures, but lack a task-centric criterion that characterizes how much task-relevant information must be retained over time. To address this gap, we introduce **Task Memory Complexity (TMC)** , which measures the minimal amount of past information required for optimal decision-making. 

**Setup.** We model a manipulation task as a partially observable Markov decision process (POMDP) with latent state _st ∈S_ , observation _ot ∈O_ , and action _at ∈A_ . Let the full interaction history up to time _t_ be 



Rather than assuming access to the full history, we consider a memory state that summarizes task-relevant information from past observations. Let _Mt_ denote a memory representation constructed from _ht_ , and let _M_<sup>(</sup> _t_<sup>_k_)</sup> denote a memory state that encodes information from at most _k_ task-relevant past observations. 

**Definition. Task Memory Complexity** is defined as the smallest integer _m ≥_ 0 such that there exists an optimal pol- 

icy _π_<sup>_∗_</sup> whose action at time _t_ depends only on the memory state _M_<sup>(</sup> _t_<sup>_m_)</sup> . Formally, 



**Task Annotation.** Tasks are annotated according to their Task Memory Complexity using the notation _M_ (0), _M_ (1), or more generally _M_ ( _n_ ), where the index indicates the number of task-relevant past observations that must be retained to solve the task optimally. 

**Interpretation.** _M_ (0) denotes memory-free tasks, for which the current observation is sufficient to uniquely determine task progress and the optimal action. _M_ (1) denotes tasks that require retaining a single task-relevant past observation to disambiguate the current state. More generally, _M_ ( _n_ ) denotes tasks whose optimal decisions depend on retaining _n_ task-relevant past observations, capturing nonlocal and multi-step temporal dependencies. 

### **3.2. RMBench System Design** 

RMBench is developed within the RoboTwin 2.0 (Chen et al., 2025a) system framework. It is built on top of the SAPIEN (Xiang et al., 2020) simulation engine and supports both automated data synthesis and integrated policy evaluation within a unified pipeline. This design enables scalable data generation as well as consistent and reproducible benchmarking of robotic manipulation policies. 

In addition, we provide fine-grained language annotations that align with each action–observation pair. These annotations assign explicit linguistic descriptions to low-level interactions and state transitions, offering structured and 

3 

**RMBench: Memory-Dependent Robotic Manipulation Benchmark with Insights into Policy Design** 

dense supervision signals for downstream training of highlevel reasoning or memory modules. 

### **3.3. RMBench Benchmark Tasks** 

Based on the proposed Task Memory Complexity (Sec. 3.1), we design a total of nine memory-dependent manipulation tasks. These tasks are grouped into two categories: five _M_ (1) tasks, and four _M_ ( _n_ ) tasks. Representative key frames for each task are illustrated in Fig. 1, and detailed task specifications are provided in the Appendix A. 

The _M_ (1) tasks consist of _Observe and Pick Up_ , _Rearrange Blocks_ , _Put Back Block_ , _Swap Blocks_ and _Swap T_ . These tasks require the policy to retain a single past observation or a fixed, limited number of historical frames. Successful execution depends on dynamically attending to task-relevant information across different stages of the task. 

The _M_ ( _n_ ) tasks include _Blocks Ranking Try_ , _Press Button_ , _Cover Blocks_ , and _Battery Try_ . These tasks require repeated active exploration, trial-and-error interactions, or repeated execution for a task-specific number of attempts, often guided by external feedback. As a result, they demand strong long-term memory retention and effective retrieval mechanisms to accumulate and utilize historical information over extended horizons. 

## **4. Mem-0 Policy** 

In this section, we present **Mem-0** , a modular memoryoriented robotic policy designed for systematic analysis of memory mechanisms. As shown in Fig. 2, Mem-0 consists of a **Planning Module** that performs subtask-level reasoning based on key memory and an **Execution Module** that executes subtasks using sliding-window and anchor memories. The two modules are connected by a **Subtask End Classifier** , which enables closed-loop planning and execution. This modular design supports fine-grained analysis of the roles of different memory components in memorydependent robotic manipulation. 

### **4.1. Planning Module** 

In long-horizon manipulation tasks, end-to-end VLA models are prone to error accumulation and trajectory drift during inference. Prior work (Sridhar et al., 2025; Wen et al., 2024) mitigates these issues through subtask decomposition. However, such approaches become insufficient in memorydependent settings, such as the _M_ ( _n_ )-type tasks in RMBench, where accurate subtask inference requires reasoning over multiple previously completed subtasks, for example by tracking the number of button presses. To address this limitation, we introduce a key memory window module within the Planning Module, which aggregates completed subtasks and enables memory-aware subtask reasoning. 

The Planning Module performs subtask-level reasoning using a vision language model conditioned on visual observations and structured memory. At planning step _t_ , the model receives as input the initial observation _o_ 0, the task goal _g_ , and the memory state _Mt−_ 1, and predicts the next subtask as 



Here, _o_ 0 _∈_ R<sup>_H×W ×_3</sup> denotes the initial RGB observation at the beginning of an episode, and _g ∈T_ denotes the global task instruction. The finished-task memory _Mt−_ 1, also referred to as the key memory window, aggregates all previously completed subtasks and is defined as 



where _si ∈T_ is the textual description of the _i_ -th subtask and _o_<sup>end</sup> _i ∈_ R<sup>_H×W ×_3</sup> is the RGB observation at the termination of that subtask. 

Conditioning on _Mt−_ 1 enables the vision–language model to explicitly reason over previously executed subtasks and their corresponding visual outcomes, thereby supporting memory-dependent subtask inference beyond single-frame observation-based planning. 

In contrast to existing approaches that infer a new subtask at every observation frame and thus require _O_ ( _T_ ) planning calls over a horizon of _T_ timesteps, the Planning Module performs subtask reasoning only upon subtask termination, as identified by a Subtask End Classifier (Section 4.3). For a long-horizon task consisting of _N_ subtasks, where _N ≪ T_ , this design reduces the number of planning invocations to _O_ ( _N_ ). As a result, the Execution Module can operate at a high control frequency within each subtask without being constrained by planning latency, substantially reducing computational overhead and accelerating task execution. 

### **4.2. Execution Module** 

Although subtask-level planning is effective for many memory-dependent tasks, some tasks are not well suited for explicit subtask decomposition. For instance, in _M_ (1)type tasks such as _Swap T_ , the target placement orientation of object _T_ cannot be specified reliably through language. In addition, overly fine-grained subtask decomposition increases annotation effort and planning latency, which degrades overall execution efficiency. To address these limitations, we incorporate an **anchor memory module** and a **sliding memory window module** within the Execution Module. This design allows the Execution Module to handle _M_ (1)-type tasks without additional subtask decomposition by maintaining a persistent anchor memory together with short-term transient memories. 

The **Execution Module** executes the current subtask using a diffusion-based policy conditioned on multimodal per- 

4 

**RMBench: Memory-Dependent Robotic Manipulation Benchmark with Insights into Policy Design** 



<!-- Start of picture text -->
Planning Module<br>In-Episode Memory (key)<br>Initial Observation Task Instruction key frame 1 key frame 2 key frame 3 key frame 4<br>Cover the blocks from left to right using<br>the lids, and then uncover them again in<br>the sequence red, green, and blue.<br>Vision Language Model (Qwen3-VL-8B-Instruct)<br>Execution Module Design Details about  Design Details about<br>State Noised Action Anchor Memory Fusion Sliding Memory Fusion<br>SubTask Current Observation<br>Open the left cover to uncover  anchor window sliding window …<br>the blocks in the order of red,  Image token Image token<br>green, and blue. Timestep PE<br>Q K, V Q K, V<br>Vision Language Model (Qwen3-VL-2B-Instruct) Cross Attention Cross Attention<br>Anchor Memory  anchor  Residual Residual<br>Fusion fused token<br>Image<br>token text token<br>new key frame<br>Sliding Memory  sliding  Subtask End  if the subtask has<br>Fusion fused token Denoised Action Classifier been finished<br>conditioning<br>Diffusion Transformer (DiT)<br><!-- End of picture text -->

_Figure 2._ **Mem-0 Pipeline** . Mem-0 comprises a Planning Module and an Execution Module linked by a Subtask End Classifier. The Planning Module generates high-level subtasks from task instructions, observations, and key-frame memory, while the Execution Module produces low-level actions using the current observation, the subtask, and fused anchor and sliding memories in a diffusion-based policy. Upon subtask completion, a key frame is stored to enable iterative planning and execution until task completion. 

ception and memory. A vision–language model (VLM), denoted by _V_ exec( _·_ ), encodes the current RGB observation _ot ∈_ R<sup>_H×W ×_3</sup> and the subtask instruction _st ∈T_ into image and text token embeddings: 



Mean pooling is then applied to obtain compact latent = = representations **z**<sup>img</sup> _t_ MeanPool( **Z**<sup>img</sup> _t_ ) and **z**<sup>text</sup> _t_ MeanPool( **Z**<sup>text</sup> _t_<sup>).</sup> 

To incorporate memory, the image latent attends to two memory buffers: an _anchor memory A_ and a _sliding memory window St_ . Memory-conditioned representations are computed via cross-attention: 



where _l ∈{_ anchor _,_ slide _}_ , _M_<sup>anchor</sup> _t_ = _A_ and _M_<sup>slide</sup> _t_ = _St_ . The fused token are concatenated with the text token to form the conditioning vector **c** _t_ = [ ˜ **z**<sup>anchor</sup> _t_ ; ˜ **z**<sup>slide</sup> _t_ ; **z**<sup>text</sup> _t_ ]. After attention at timestep _t_ , the image latent is appended to the sliding memory window: 



where Trunc _K_ ( _·_ ) retains the most recent _K_ elements. At the beginning of a subtask ( _t_ = 0), the image latent **z**<sup>img</sup> 0 is 

stored as the anchor memory _A_ = _{_ **z**<sup>img</sup> 0<sup>_}_, which remains</sup> fixed throughout the subtask. Upon subtask termination, both memory buffers are reset: _A ←_ ∅ _, St ←_ ∅ _._ 

For action generation, we employ a diffusion transformer (DiT) with a fixed action horizon _H_ = 30. Let **a**<sup>_ϵ_</sup> _t_ : _t_ + _H−_ 1<sup>_∈_</sup> R<sup>_H×da_</sup> denote a noisy action sequence obtained by perturbing a ground-truth action sequence with Gaussian noise, and let ˆ **a** _t_ : _t_ + _H−_ 1 _∈_ R<sup>_H×da_</sup> denote the corresponding denoised prediction produced by the model. At timestep _t_ , the DiT predicts a denoised action sequence 



A prefix of the predicted sequence, denoted as **a** ˆ _t_ : _t_ +∆ _−_ 1 with 1 _≤_ ∆ _≤ H_ , is executed as control commands before the next replanning step. 

### **4.3. Subtask End Classifier** 

To enable closed-loop interaction between the Planning and Execution Modules, we introduce a _Subtask End Classifier_ to detect subtask completion. The classifier is implemented as a lightweight multilayer perceptron (MLP) that operates on the conditioning vector **c** _t_ and outputs a binary signal _C_ end( **c** _t_ ) _∈{_ 0 _,_ 1 _}_ , indicating whether the current subtask is ongoing or terminated at timestep _t_ . 

To improve robustness and avoid premature termination 

5 

**RMBench: Memory-Dependent Robotic Manipulation Benchmark with Insights into Policy Design** 



Once this condition is satisfied, the subtask is terminated, and the final observation _o_<sup>end</sup> _t_ together with the corresponding subtask description _st_ is passed to the Planning Module to trigger the next round of subtask-level reasoning. This mechanism establishes a closed loop between high-level planning and low-level execution, enabling coordinated and iterative subtask inference and execution. 

## **5. Experiment** 

We design a set of experiments to validate three key objectives: (1) to evaluate the performance of existing manipulation policies and Mem-0 on RMBench, thereby characterizing their ability to handle memory-dependent tasks across different levels of difficulty; (2) to conduct systematic ablation studies on the Mem-0 architecture in order to analyze how different module designs affect performance on memory-intensive manipulation tasks; and (3) to perform real-world robotic experiments to assess the effectiveness and generalization of Mem-0 beyond simulation. 

In addition to the SAPIEN platform, RMBench also implemented on NVIDIA Isaac Lab - Arena<sup>1</sup> . 

### **5.1. Evaluation of Policies on RMBench** 

we benchmark a diverse set of policies on RMBench, including non-pretrained methods, pretrained methods, and Mem-0, our memory-centric policy. Specifically, DP and ACT are used as non-pretrained baselines, while Pi0.5 and X-VLA represent pretrained approaches. For each task under both the _M_ (1) and _M_ ( _n_ ) settings, all models are trained with 50 expert demonstrations and evaluated over 100 rollout episodes. 

For _M_ (1)-type tasks, Mem-0 operates without subtask decomposition, and the reported results primarily reflect the capability of the Execution Module. In contrast, for _M_ ( _n_ )- type tasks, subtask decomposition is performed at key decision points, and the results capture the joint performance of the Planning and Execution Modules. All baseline methods are trained without subtask decomposition. We report success rates for all methods in Table 1. 

Experimental results show that both non-pretrained and pretrained baselines consistently underperform on memorydependent tasks. This behavior can be attributed to the fact that most existing models are designed under a Markovian 

assumption, where the next action is determined solely by the current observation. When applied to the non-Markovian tasks in RMBench, these models fail to infer the correct action without access to task-relevant past information, leading to substantial performance degradation. Representative failure cases of baseline methods are illustrated in Fig. 3. 

In contrast, Mem-0 demonstrates substantial performance gains across the majority of tasks, indicating the effectiveness of explicitly incorporating memory mechanisms. On average, Mem-0 improves success rates by 38.4% on _M_ (1) tasks and 21.2% on _M_ ( _n_ ) tasks relative to the baselines, underscoring the critical role of memory modules in addressing memory-dependent manipulation in RMBench. 

Despite these gains, Mem-0 exhibits limitations on tasks that require strong semantic understanding, such as _Observe and Pick Up_ , where pretrained models retain an advantage due to large-scale pretraining. In fine-grained manipulation tasks such as _Swap T_ , Mem-0 exhibits limited placement accuracy, resulting in only marginal performance gains. In the _Press Button_ task, the small magnitude of individual press actions further complicates reliable termination detection: the Subtask End Classifier may fail to consistently recognize task completion, causing repeated presses or missed contacts and ultimately zero success. 

These results highlight several open challenges in the current Mem-0 design. More Visualization and Analysis can be found in the Appendix C and Supplementary Material. Nevertheless, the overall performance trends clearly demonstrate that explicit memory modeling yields significant improvements on the majority of memory-dependent tasks in RMBench. 

### **5.2. Analysis on Memory-Related Module** 

In this section, we analyze the contribution of individual components in Mem-0 to provide insights into effective memory module design. To this end, we conduct four ablation studies: 

(1) **w/o Anchor** : The anchor memory module in the Execution Module is removed, so image tokens are no longer fused with anchor memory tokens. 

(2) **w/o Sliding** : The sliding memory module in the Execution Module is removed, and image tokens are not fused with historical sliding memory tokens. 

(3) **w/o Key** : The key memory window in the Planning Module is removed, and subtask inference relies solely on a single-frame observation. 

(4) **GT Classifier** : The Subtask End Classifier is removed, and subtask termination is determined using ground-truth signals provided by the simulator. 

> 1https://github.com/isaac-sim/IsaacLab-Arena 

6 

**RMBench: Memory-Dependent Robotic Manipulation Benchmark with Insights into Policy Design** 

_Table 1._ **RMBench benchmark results.** RMBench includes nine manipulation tasks across the _M_ (1) and _M_ ( _n_ ) levels of Task Memory Complexity. We report success rates for five policies, each trained with 50 synthesized demonstrations and evaluated over 100 rollouts. ( **<mark>Bold</mark>** <mark>:</mark> best; Underlined: second-best; Green: relative improvement over the second-best). 

|Tasks|TMC|DP|ACT|Pi0.5|X-VLA|Mem-0 (ours)|
|---|---|---|---|---|---|---|
|Observe and Pick Up|_M_(1)<br>|1%|1%|**9%**|**9%**|4%|
|Rearrange Blocks|_M_(1)|0%|29%|13%|13%|**89%**|
|Put Back Block|_M_(1)|0%|0%|11%|18%|**90%**|
|Swap Blocks|_M_(1)<br>|11%|2%|24%|16%|**67%**|
|Swap T|_M_(1)<br>|**20%**|2%|15%|3%|14%|
|**_Average_**|_M_(1)|6.4%|6.8%|14.4%|11.8%|**52.8%(+38.4%)**|
|Battery Try|_M_(_n_)<br>|10%|19%|16%|26%|**28%**|
|Blocks Ranking Try|_M_(_n_)|10%|0%|6%|1%|**18%**|
|Cover Blocks|_M_(_n_)|0%|0%|0%|2%|**68%**|
|Press Button|_M_(_n_)|0%|0%|0%|0%|0%|
|**_Average_**|_M_(_n_)|5%|4.8%|5.5%|7.3%|**28.5%(+21.2%)**|
|**_Total Average_**|/|5.8%|5.9%|10.4%|9.8%|**42.0%(+31.6%)**|



_Table 2._ **Ablation Studies.** <mark>(</mark> **<mark>Bold</mark>** <mark>:</mark> the best results; Underlined: the second-best results). 

|_M_(1)Tasks|Observe and Pick Up|Rearrange Blocks|Put Back Block|Swap Blocks|Swap T|**_Average_**|
|---|---|---|---|---|---|---|
|Vanilla (ours)|**4%**|**89%**|**90%**|**67%**|14%|**52.8%**|
|w/o Anchor|**4%**|73%|35%|15%|7%|26.8%|
|w/o Sliding|3%|62%|78%|39%|**20%**|40.4%|
|_M_(_n_)Tasks|Battery Try|Blocks Ranking Try|Cover Blocks|Press Button||**_Average_**|
|Vanilla (ours)|28%|18%|68%|0%||28.5%|
|w/o Key|13%|1%|5%|0%||4.8%|
|w/o Anchor|14%|0%|**92%**|1%||26.8%|
|w/o Sliding|17%|0%|84%|0%||25.3%|
|GT Classifier|**30%**|**45%**|**92%**|**14%**||**45.3%**|



Because Mem-0 does not perform subtask decomposition for _M_ (1)-type tasks, the ablation study for _M_ (1) includes only the **w/o Anchor** and **w/o Sliding** settings. All ablation results are reported in Table 2. 

**Analysis on Anchor Memory Performance.** Compared to the vanilla setting, removing the anchor memory ( **w/o Anchor** ) leads to a substantial reduction in success rates across most tasks. Qualitative inspection of evaluation videos indicates that, although the sliding memory window remains active, Mem-0 progressively loses access to task-critical information as relevant memories are evicted over time. As a result, the policy fails to attend to essential cues required for correct decision-making and exhibits erroneous behaviors similar to those observed in Fig. 3. These findings indicate that, for memory-dependent manipulation tasks, it is crucial for a policy to explicitly identify and retain task-critical information throughout execution in order to achieve reliable task completion. 

**Analysis on Sliding Memory Performance.** The Sliding Memory Window primarily captures short-term historical motion trends. Experiments show that, even with the support of anchor information, removing this module still degrades performance across most tasks, with success rates 

falling below those of the vanilla model. Qualitative results further indicate that, without sliding memory, Mem-0 exhibits unstable and oscillatory behaviors; for example, in the button-pressing task, the policy cannot infer whether the button has already been pressed from a single observation, leading to premature termination or redundant actions and eventual failure. 

Interestingly, on the _Swap T_ task, the w/o Sliding setting outperforms the vanilla model. This improvement likely stems from the fixed motion patterns in the training data and the task’s high sensitivity to the initial orientation of the T-shaped object. Removing sliding memory shifts the policy’s reliance toward anchor information and reduces interference from transient motion cues, leading to better performance. This observation suggests that sliding memory can function as either a facilitator or a source of interference, underscoring the importance of coordinating sliding and anchor memories. 

**Analysis on Key Memory Performance.** Under the **w/o Key** setting, where subtask inference is based solely on single-frame observations, success rates decrease substantially relative to the vanilla configuration. Qualitative analysis shows that, for _M_ ( _n_ )-type tasks requiring long-term 

7 



<!-- Start of picture text -->
RMBench: Memory-Dependent Robotic Manipulation Benchmark with Insights into Policy Design<br><!-- End of picture text -->



<!-- Start of picture text -->
0 Swap T T 0 Observe and Pick Up T 0 Swap Blocks T<br>0 Put Back Block T 0 Rearrange Blocks T<br>0 Battery Try T 0 Press Button T<br>0 Blocks Ranking Try T 0 Cover Blocks T<br><!-- End of picture text -->

_Figure 3._ **Visualization of Baseline Typical Error.** Because the baseline predicts the next action solely from the current observation, it struggles to perform reliably on non-Markovian tasks that require persistent memory over time. 

information to inform subsequent motion decisions, the Planning Module is unable to reliably infer the correct next subtask when restricted to the current observation alone, resulting in task failure. These results indicate that retaining key memories is critical for accurate and reliable subtask inference in memory-dependent manipulation tasks. 

**Analysis on the Classifier between the Planning and Execution Modules.** The Subtask End Classifier in Mem-0 serves two primary functions. First, it connects the Planning and Execution Modules by triggering high-level reasoning only when necessary, thereby reducing inference cost and latency. Second, it enables task simplification through subtask decomposition. Compared with MemER (Sridhar et al., 2025), which performs high-level subtask reasoning at every timestep, Mem-0 operates at a planning frequency of approximately 5–10 Hz, whereas MemER runs at 1–2 Hz. 

In addition, the strong performance achieved with the **GT Classifier** highlights the effectiveness of subtask decomposition relative to fully end-to-end approaches. Nevertheless, the current classifier design in Mem-0 is relatively simple and lacks precision in detecting subtask transitions. Inaccurate transition timing can negatively impact subtask inference in the Planning Module. These results indicate that more refined classifier designs are necessary to achieve tighter coordination between the Planning and Execution Modules through more accurate subtask termination signals. 

### **5.3. Real World Experiment** 

_Table 3._ **Real-world Experiment results.** 

|Tasks|ACT|Pi0.5|Mem-0 (ours)|
|---|---|---|---|
|Put Back Block|0.0%|10.0%|**17.5%**|
|Rearrange Blocks|0.0%|7.5%|**37.5%**|
|Cover Blocks|0.0%|0.0%|**12.5%**|
|**_Average_**|0.00%|5.83%|**22.50%**|





<!-- Start of picture text -->
Put Back Block Rearrange Blocks Cover Blocks<br><!-- End of picture text -->

_Figure 4._ **Real-world Experiment Tasks.** The real-world experimental setup is illustrated above. 

To assess the real-world performance of Mem-0, we evaluate it on three physical manipulation tasks aligned with RMBench: Put Back Blocks, Rearrange Blocks and Cover Blocks. We compare Mem-0 against ACT and Pi0.5, with all data collection and evaluation conducted on the X-One dual-arm robotic platform. For each task, we collect 100 real-world demonstrations and evaluate the trained policies over 40 rollout trials, reporting success rate as the primary metric. The evaluation results are summarized in Table 3. 

8 

**RMBench: Memory-Dependent Robotic Manipulation Benchmark with Insights into Policy Design** 

The results show that Mem-0 outperforms both baseline policies in real-world experiments. Upon closer inspection, we observe that most failures of Mem-0 arise from imprecise block manipulation, rather than high-level task planning. This behavior is likely attributable to two factors. First, real-world data collection involves diverse human behaviors, which introduces additional variability and increases the difficulty of learning consistent low-level manipulation skills. Second, Mem-0 is trained without dedicated pretraining on robotic manipulation, which may limit its ability to generalize fine-grained motor behaviors. Addressing these limitations through improved low-level pretraining and more structured real-world data collection constitutes an important direction for future work. 

## **6. Conclusion** 

In this paper, we present RMBench (benchmark) and Mem-0 (policy) to systematically evaluate memory in robotic manipulation, revealing the memory limitations of existing policies and how different architectural choices (such as anchor memory, sliding memory, and key memory) affect memory performance, thereby providing preliminary insights into the principled integration of memory mechanisms for effective memory-dependent robotic manipulation. 

As for future work, promising directions include improved memory representation and fusion, more robust subtask termination criteria, and the integration of pretraining to enhance semantic understanding and generalization. We hope RMBench fosters principled progress toward scalable, memory-aware robotic manipulation. 

## **Acknowledgements** 

We would like to thank Xspark AI for supporting our realworld experiments, and D-Robotics for providing the computing resources. Also thank NVIDIA Isaac Lab - Arena Team for technical support. 

## **Impact Statement** 

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here. 

## **References** 

- Bi, H., Tan, H., Xie, S., Wang, Z., Huang, S., Liu, H., Zhao, R., Feng, Y., Xiang, C., Rong, Y., et al. Motus: A unified latent action world model. _arXiv preprint arXiv:2512.13030_ , 2025. 

- Chen, B., Wan, W., Chen, T., Guo, X., Xu, C., Qi, Y., Zhang, 

H., Wu, L., Xu, T., Li, Z., et al. Univtac: A unified simulation platform for visuo-tactile manipulation data generation, learning, and benchmarking. _arXiv preprint arXiv:2602.10093_ , 2026. 

- Chen, T., Chen, Z., Chen, B., Cai, Z., Liu, Y., Li, Z., Liang, Q., Lin, X., Ge, Y., Gu, Z., et al. Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. _arXiv preprint arXiv:2506.18088_ , 2025a. 

- Chen, T., Mu, Y., Liang, Z., Chen, Z., Peng, S., Chen, Q., Xu, M., Hu, R., Zhang, H., Li, X., et al. G3flow: Generative 3d semantic flow for pose-aware and generalizable object manipulation. In _Proceedings of the Computer Vision and Pattern Recognition Conference_ , pp. 1735–1744, 2025b. 

- Cherepanov, E., Kachaev, N., Kovalev, A. K., and Panov, A. I. Memory, benchmark & robots: A benchmark for solving complex tasks with reinforcement learning. _arXiv preprint arXiv:2502.10550_ , 2025. 

- Chi, C., Xu, Z., Feng, S., Cousineau, E., Du, Y., Burchfiel, B., Tedrake, R., and Song, S. Diffusion policy: Visuomotor policy learning via action diffusion. _The International Journal of Robotics Research_ , 44(10-11): 1684–1704, 2025. 

- Fang, H., Grotz, M., Pumacay, W., Wang, Y. R., Fox, D., Krishna, R., and Duan, J. Sam2act: Integrating visual foundation model with a memory architecture for robotic manipulation. _arXiv preprint arXiv:2501.18564_ , 2025. 

- Han, S., Qiu, B., Liao, Y., Huang, S., Gao, C., Yan, S., and Liu, S. Robocerebra: A large-scale benchmark for longhorizon robotic manipulation evaluation. _arXiv preprint arXiv:2506.06677_ , 2025. 

- Intelligence, P., Amin, A., Aniceto, R., Balakrishna, A., Black, K., Conley, K., Connors, G., Darpinian, J., Dhabalia, K., DiCarlo, J., Driess, D., Equi, M., Esmail, A., Fang, Y., Finn, C., Glossop, C., Godden, T., Goryachev, I., Groom, L., Hancock, H., Hausman, K., Hussein, G., Ichter, B., Jakubczak, S., Jen, R., Jones, T., Katz, B., Ke, L., Kuchi, C., Lamb, M., LeBlanc, D., Levine, S., Li-Bell, A., Lu, Y., Mano, V., Mothukuri, M., Nair, S., Pertsch, K., Ren, A. Z., Sharma, C., Shi, L. X., Smith, L., Springenberg, J. T., Stachowicz, K., Stoeckle, W., Swerdlow, A., Tanner, J., Torne, M., Vuong, Q., Walling, A., Wang, H., Williams, B., Yoo, S., Yu, L., Zhilinsky, U., and Zhou, Z. _π_ 0<sup>_∗_</sup> _._ 6<sup>:a vla that learns from experience, 2025a.URL</sup> https://arxiv.org/abs/2511.14759. 

- Intelligence, P., Black, K., Brown, N., Darpinian, J., Dhabalia, K., Driess, D., Esmail, A., Equi, M., Finn, C., 

9 

**RMBench: Memory-Dependent Robotic Manipulation Benchmark with Insights into Policy Design** 

- Fusai, N., Galliker, M. Y., Ghosh, D., Groom, L., Hausman, K., Ichter, B., Jakubczak, S., Jones, T., Ke, L., LeBlanc, D., Levine, S., Li-Bell, A., Mothukuri, M., Nair, S., Pertsch, K., Ren, A. Z., Shi, L. X., Smith, L., Springenberg, J. T., Stachowicz, K., Tanner, J., Vuong, Q., Walke, H., Walling, A., Wang, H., Yu, L., and Zhilinsky, U. _π_ 0 _._ 5: a vision-language-action model with open-world generalization, 2025b. URL https: //arxiv.org/abs/2504.16054. 

- Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., and Stoica, I. Efficient memory management for large language model serving with pagedattention. In _Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles_ , 2023. 

- Lan, Z., Jiang, Y., Wang, R., Xie, X., Zhang, R., Zhu, Y., Li, P., Yang, T., Chen, T., Gao, H., et al. Autobio: A simulation and benchmark for robotic automation in digital biology laboratory. _arXiv preprint arXiv:2505.14030_ , 2025. 

- Li, C., Zhang, R., Wong, J., Gokmen, C., Srivastava, S., Mart´ın-Mart´ın, R., Wang, C., Levine, G., Ai, W., Martinez, B., Yin, H., Lingelbach, M., Hwang, M., Hiranaka, A., Garlanka, S., Aydin, A., Lee, S., Sun, J., Anvari, M., Sharma, M., Bansal, D., Hunter, S., Kim, K.-Y., Lou, A., Matthews, C. R., Villa-Renteria, I., Tang, J. H., Tang, C., Xia, F., Li, Y., Savarese, S., Gweon, H., Liu, C. K., Wu, J., and Fei-Fei, L. Behavior-1k: A human-centered, embodied ai benchmark with 1,000 everyday activities and realistic simulation. _arXiv preprint arXiv:2403.09227_ , 2024a. 

- Li, H., Yang, S., Chen, Y., Tian, Y., Yang, X., Chen, X., Wang, H., Wang, T., Zhao, F., Lin, D., et al. Cronusvla: Transferring latent motion across time for multi-frame prediction in manipulation. _arXiv preprint arXiv:2506.19816_ , 2025. 

- Li, Q., Liang, Y., Wang, Z., Luo, L., Chen, X., Liao, M., Wei, F., Deng, Y., Xu, S., Zhang, Y., et al. Cogact: A foundational vision-language-action model for synergizing cognition and action in robotic manipulation. _arXiv preprint arXiv:2411.19650_ , 2024b. 

- Li, X., Hsu, K., Gu, J., Pertsch, K., Mees, O., Walke, H. R., Fu, C., Lunawat, I., Sieh, I., Kirmani, S., Levine, S., Wu, J., Finn, C., Su, H., Vuong, Q., and Xiao, T. Evaluating real-world robot manipulation policies in simulation, 2024c. URL https://arxiv.org/abs/ 2405.05941. 

- Liang, Z., Li, Y., Yang, T., Wu, C., Mao, S., Nian, T., Pei, L., Zhou, S., Yang, X., Pang, J., et al. Discrete 

diffusion vla: Bringing discrete diffusion to action decoding in vision-language-action policies. _arXiv preprint arXiv:2508.20072_ , 2025. 

- Lin, M., Ding, P., Wang, S., Zhuang, Z., Liu, Y., Tong, X., Song, W., Lyu, S., Huang, S., and Wang, D. Hif-vla: Hindsight, insight and foresight through motion representation for vision-language-action models. _arXiv preprint arXiv:2512.09928_ , 2025. 

- Liu, B., Zhu, Y., Gao, C., Feng, Y., Liu, Q., Zhu, Y., and Stone, P. Libero: Benchmarking knowledge transfer for lifelong robot learning, 2023. URL https://arxiv. org/abs/2306.03310. 

- Lu, G., Gao, Z., Chen, T., Dai, W., Wang, Z., Ding, W., and Tang, Y. Manicm: Real-time 3d diffusion policy via consistency model for robotic manipulation. _arXiv preprint arXiv:2406.01586_ , 2024. 

- Mu, Y., Chen, T., Chen, Z., Peng, S., Lan, Z., Gao, Z., Liang, Z., Yu, Q., Zou, Y., Xu, M., et al. Robotwin: Dual-arm robot benchmark with generative digital twins. In _Proceedings of the Computer Vision and Pattern Recognition Conference_ , pp. 27649–27660, 2025. 

- Nasiriany, S., Maddukuri, A., Zhang, L., Parikh, A., Lo, A., Joshi, A., Mandlekar, A., and Zhu, Y. Robocasa: Largescale simulation of everyday tasks for generalist robots. _arXiv preprint arXiv:2406.02523_ , 2024. 

- Shen, W., Liu, Y., Wu, Y., Liang, Z., Gu, S., Wang, D., Nian, T., Xu, L., Qin, Y., Pang, J., et al. Expertise need not monopolize: Action-specialized mixture of experts for vision-language-action learning. _arXiv preprint arXiv:2510.14300_ , 2025. 

- Shi, H., Xie, B., Liu, Y., Sun, L., Liu, F., Wang, T., Zhou, E., Fan, H., Zhang, X., and Huang, G. Memoryvla: Perceptual-cognitive memory in vision-languageaction models for robotic manipulation. _arXiv preprint arXiv:2508.19236_ , 2025. 

- Sridhar, A., Pan, J., Sharma, S., and Finn, C. Memer: Scaling up memory for robot control via experience retrieval. _arXiv preprint arXiv:2510.20328_ , 2025. 

- Su, Y., Zhan, X., Fang, H., Xue, H., Fang, H.-S., Li, Y.-L., Lu, C., and Yang, L. Dense policy: Bidirectional autoregressive learning of actions. _arXiv preprint arXiv:2503.13217_ , 2025. 

- Tao, S., Xiang, F., Shukla, A., Qin, Y., Hinrichsen, X., Yuan, X., Bao, C., Lin, X., Liu, Y., kai Chan, T., Gao, Y., Li, X., Mu, T., Xiao, N., Gurha, A., Rajesh, V. N., Choi, Y. W., Chen, Y.-R., Huang, Z., Calandra, R., Chen, R., Luo, S., and Su, H. Maniskill3: Gpu parallelized robotics simulation and rendering for generalizable embodied ai, 2025. URL https://arxiv.org/abs/2410.00425. 

10 

**RMBench: Memory-Dependent Robotic Manipulation Benchmark with Insights into Policy Design** 

- Team, R. Rdt2: Enabling zero-shot cross-embodiment generalization by scaling up umi data, September 2025. URL https://github.com/thu-ml/RDT2. 

- Wang, Y., Wu, R., Chen, Y., Wang, J., Liang, J., Zhu, Z., Geng, H., Malik, J., Abbeel, P., and Dong, H. Dexgarmentlab: Dexterous garment manipulation environment with generalizable policy, 2025. URL https: //arxiv.org/abs/2505.11032. 

_62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations)_ , Bangkok, Thailand, 2024. Association for Computational Linguistics. URL http://arxiv.org/abs/2403. 13372. 

- Wen, J., Zhu, Y., Zhu, M., Tang, Z., Li, J., Zhou, Z., Liu, X., Shen, C., Peng, Y., and Feng, F. Diffusionvla: Scaling robot foundation models via unified diffusion and autoregression. In _Forty-second International Conference on Machine Learning_ . 

- Wen, J., Zhu, M., Zhu, Y., Tang, Z., Li, J., Zhou, Z., Li, C., Liu, X., Peng, Y., Shen, C., and Feng, F. Diffusionvla: Scaling robot foundation models via unified diffusion and autoregression. _arXiv preprint arXiv:None_ , 2024. 

- Wen, J., Zhu, Y., Li, J., Tang, Z., Shen, C., and Feng, F. Dexvla: Vision-language model with plug-in diffusion expert for general robot control. _arXiv preprint arXiv:2502.05855_ , 2025a. 

- Wen, J., Zhu, Y., Li, J., Zhu, M., Tang, Z., Wu, K., Xu, Z., Liu, N., Cheng, R., Shen, C., et al. Tinyvla: Towards fast, data-efficient vision-language-action models for robotic manipulation. _IEEE Robotics and Automation Letters_ , 2025b. 

- Xiang, F., Qin, Y., Mo, K., Xia, Y., Zhu, H., Liu, F., Liu, M., Jiang, H., Yuan, Y., Wang, H., et al. Sapien: A simulated part-based interactive environment. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pp. 11097–11107, 2020. 

- Ze, Y., Zhang, G., Zhang, K., Hu, C., Wang, M., and Xu, H. 3d diffusion policy: Generalizable visuomotor policy learning via simple 3d representations. _arXiv preprint arXiv:2403.03954_ , 2024. 

- Zhao, T. Z., Kumar, V., Levine, S., and Finn, C. Learning fine-grained bimanual manipulation with low-cost hardware, 2023. URL https://arxiv.org/abs/ 2304.13705. 

- Zheng, J., Li, J., Wang, Z., Liu, D., Kang, X., Feng, Y., Zheng, Y., Zou, J., Chen, Y., Zeng, J., Zhang, Y.Q., Pang, J., Liu, J., Wang, T., and Zhan, X. X-vla: Soft-prompted transformer as scalable cross-embodiment vision-language-action model, 2025. URL https:// arxiv.org/abs/2510.10274. 

- Zheng, Y., Zhang, R., Zhang, J., Ye, Y., Luo, Z., Feng, Z., and Ma, Y. Llamafactory: Unified efficient finetuning of 100+ language models. In _Proceedings of the_ 

11 

**RMBench: Memory-Dependent Robotic Manipulation Benchmark with Insights into Policy Design** 

## **A. RMBench Tasks Description** 

_Table 4._ **Task descriptions of RMBench benchmark.** 

|**_Task_**|**Description**|
|---|---|
|_Observe and Pick Up_|A reference object is placed on a shelf, and multiple objects are placed on the table.<br>The robot first observes the reference object while remaining stationary. After the<br>reference object is hidden, the robot must pick up the matching object from the table.|
|_Rearrange Blocks_|Two pads and a button are placed on the table. One block is positioned between the<br>two pads, and another block is placed on one of the pads. The robot moves the middle<br>block onto a pad, presses the button, and then moves the other block to the middle<br>position.|
|_Put Back Block_|Four pads are arranged around a central position, with one block placed on one of the<br>pads. The robot moves the block to the center, presses the button, and then returns the<br>block to its original pad.|
|_Swap Block_|Three pads and a button are placed on the table, with two blocks placed on different<br>pads. The robot uses the empty pad to swap the positions of the two blocks and then<br>presses the button.|
|_Swap T_|Two T-shaped blocks with different colors are placed on the table. The robot picks up<br>both blocks and swaps their positions and orientations.|
|_Battery Try_|Two batteries with random orientations and a dual-slot battery holder are placed on the<br>table. The robot repeatedly attempts different insertion orders, placing both batteries<br>into the holder with the correct orientations until the insertion succeeds.|
|_Blocks Ranking Try_|Three blocks of different colors are randomly arranged on the table, along with a button.<br>The robot repeatedly attempts different block arrangements and presses the button to<br>confirm until the correct ordering is achieved.|
|_Cover Blocks_|Three colored blocks (red, green, and blue) and three covers are placed on the table.<br>The robot covers the blocks from left to right, then uncovers them in red–green–blue<br>order and returns the covers to their original positions.|
|_Press Button_|Three buttons (left, middle, and right) and two single-digit number tiles are placed on<br>the table. The robot presses the left button the number of times indicated by the left<br>digit, presses the middle button the number of times indicated by the right digit, and<br>then presses the right button to confirm.|



## **B. Training Details** 

### **B.1. Planning Module** 

In the Planning Module, we fine-tune the vision–language model (Qwen3-VL-8B-Instruct) using LoRA via LLaMAFactory (Zheng et al., 2024) to enable reasoning over key memories. After fine-tuning, we deploy the model with vLLM (Kwon et al., 2023) for efficient loading and inference. The key hyperparameters used for VLM fine-tuning are summarized in Table 5. Training is conducted on 8 NVIDIA A800 GPUs and the duration of training for a single task is approximately half an hour. 

_Table 5._ **Hyperparameters for fine-tuning the Mem-0 Planning Module.** 

|**Configuration**|Finetuning Type|LoRA Rank|Batch Size|Learning Rate|Epochs|LR Scheduler|Warmup Ratio|Dtype|
|---|---|---|---|---|---|---|---|---|
|**Value**|LoRA|8|16|1_._0_×_10<sup>_−_4</sup>|25|Cosine|0.1|bf16|



12 

**RMBench: Memory-Dependent Robotic Manipulation Benchmark with Insights into Policy Design** 

### **B.2. Execution Module** 

In this section, we detail the training infrastructure, training organization strategy and hyperparameter configurations employed for Execution Module in Mem-0. 

### B.2.1. TRAINING INFRASTRUCTURE AND TIME BUDGET 

The Execution Module of Mem-0 utilizes a single-task training strategy, where the model is trained from scratch for each specific task. Training is conducted on 8 NVIDIA A800 GPUs with a global batch size of 448 over 30K iterations. The duration of training for a single task is approximately 18 hours. 

### B.2.2. TRAINING ORGANIZATION STRATEGY 

**Forward Pass Strategy** . Given the memory-centric architecture of Mem-0, we employ a specialized training methodology. Within each batch, the processes of VLM token generation and DiT-based action chunk generation are executed in parallel. Conversely, the fusion of Sliding Memory and Anchor Memory requires the temporal integrity of the data; therefore, this stage is processed serially, ensuring that all frames within an episode remain sequentially aligned. 

This approach guaranties the effective utilization of VLM tokens. Furthermore, we maintain a global data structure during training to store memory information for each episode, facilitating seamless cross-batch token utilization. 

**Dataloader Implementation** . Consequently, the dataloader was custom-designed to align with this architecture. Episodes are distributed as evenly as possible across all GPUs. Each GPU processes its assigned episodes using a specific number of workers and manages the dataloader reset independently. Due to the stochastic nature of the distribution, as training iterations progress, the frames within a global batch become temporally desynchronized. This allows the model to learn simultaneously from data that span various time steps. 

### B.2.3. HYPERPARAMETER CONFIGURATIONS 

Table 6. summarizes the key training hyperparameters. To balance the learning rate requirements of different modules, we implemented a grouped learning rate strategy alongside a cosine learning rate schedule with linear warm-up. Regarding numerical precision, the VLM and Memory Bank components operate in bfloat16, while all other modules utilize float32. Furthermore, regarding visual inputs, images are resized to 224 _×_ 224 and subjected to mild data augmentation via frame-independent ColorJitter, aimed at enhancing the model’s generalization capabilities. 

_Table 6._ **Hyperparameters for training the Mem-0 Execution Module.** 

|**Configuration**|**Value**|
|---|---|
|Batch Size|448|
|Iterations|30,000|
|Max Grad. Norm|2.5|
|LR Scheduler|Cosine|
|Warmup Ratio|0.05|
|Optimizer|AdamW|
|Momentum<br>_β_1_,_|_β_2 = 0_._9_,_0_._999|
|Weight Decay|0.005|
|Image Resize|224_×_224<br>|
|Image Aug.|ColorJitter<sup>†</sup>|
|† Jitter: (0.1, 0.1, 0.1, 0)||



|**Configuration**|**Value**|
|---|---|
|LR (Base)|1_._0_×_10<sup>_−_5</sup><br>|
|LR (VLM)|1_._0_×_10<sup>_−_5</sup><br>|
|LR (Action Head)|1_._0_×_10<sup>_−_4</sup><br>|
|LR (Classifier)|1_._0_×_10<sup>_−_4</sup>|
|Min LR (Base)|1_._0_×_10<sup>_−_6</sup><br>|
|Min LR (VLM)|1_._0_×_10<sup>_−_6</sup><br>|
|Min LR (Action Head)|5_._0_×_10<sup>_−_6</sup><br>|
|Min LR (Classifier)|5_._0_×_10<sup>_−_6</sup>|
|Workers per GPU|2|



## **C. Additional Visualizations and Analysis of Failure Cases in Mem-0** 

While Mem-0 demonstrates substantial improvements over the baselines, its architectural design still offers extensive room for further exploration. In this section, we present representative cases where Mem-0 exhibits suboptimal performance, aiming to provide valuable insights for future research. 

13 

**RMBench: Memory-Dependent Robotic Manipulation Benchmark with Insights into Policy Design** 



<!-- Start of picture text -->
0 Observe and Pick Up T 0 Swap Blocks  T<br>Observe initial scene Target object hidden Prepare to grasp Pick up target object Initial scene Move block Move block Move block Press button<br><!-- End of picture text -->

_Figure 5._ **Failure examples of Observe and Pick Up** . **(Top)** Confused by objects with similar colors and shapes. **(Middle)** Confused by identical object morphologies. **(Bottom)** General failure to identify the target, resulting in the robot grasping a mean position or unintended position. 

_Figure 6._ **Failure examples of Swap Blocks** . **(Top)** Premature termination after a single subtask. **(Middle)** Premature termination after two subtasks. **(Bottom)** Failure to terminate on time, resulting in the initiation of a redundant subtask. 



<!-- Start of picture text -->
0 Rearrange Blocks  T<br>Initial scene Move blocks Press button Release Press button (redundant) Back to origin Move blocks<br><!-- End of picture text -->

_Figure 7._ **Failure examples of Rearrange Blocks** . Mem-0 redundantly presses the button, resulting in task failure. 

### **C.1. Failures Analysis for** _M_ (1) **Tasks** 

For _M_ (1) tasks, in addition to the failures illustrated in Fig. 3, we summarize the representative errors encountered by Mem-0 below. 

**Observe and Pickup & Swap Blocks** . Fig. 5 illustrates typical failure modes in the _Observe and Pick Up_ task, where Mem-0 fails to accurately identify the target object. Similarly, Fig. 6 presents examples of misjudgments regarding the termination of the swapping sequence in the _Swap blocks_ task, resulting in the confirmation button being pressed at inappropriate timings. 

These instances reveal that the Anchor Memory, in fact, exerts a continuous influence throughout the entire task horizon. Consequently, the model must maintain constant attention to the Anchor Memory and intelligently modulate the degree of its contribution to action prediction. 

Nevertheless, our ablation studies have already demonstrated the substantial performance gains brought by the Anchor Memory within the Mem-0 architecture, with its impact being particularly pronounced in tasks such as _Rearrange Blocks_ and _Put Back Block_ . 

**Rearrange Blocks** . On the other hand, Fig. 7 illustrates failure cases in the _Rearrange Blocks_ task where Mem-0 performs excessive button presses, a behavior we attribute to the limitations of the Sliding Memory module. Quantitative results from our ablation studies show a significant performance degradation in this task when the Sliding Memory is omitted. By analyzing the video playbacks, we found that the frequency of redundant button-pressing events increases markedly without Sliding Memory, identifying it as a primary failure mode. These findings demonstrate that while the current Sliding Memory provides substantial performance gains, there remains potential for further refinement. 

**Summarize** . To address these observations, we believe that one potential avenue for enhancement involves exploring more richer representation fusion mechanisms to improve the utilization of both Anchor and Sliding Memory. Such advances would further improve the performance and stability of the model in more intricate and versatile scenarios. Additionally, increasing the visual processing capabilities of existing VLM architectures is expected to yield better results in tasks such as _Observe and Pick Up_ . 

14 

**RMBench: Memory-Dependent Robotic Manipulation Benchmark with Insights into Policy Design** 

### **C.2. Failures Analysis for** _M_ ( _n_ ) **Tasks** 

For _M_ ( _n_ ) tasks, although Mem-0 demonstrates substantial improvements over the baselines, there remains significant room for further improvement. We believe that the primary challenge to be addressed lies in the performance and robustness of the Classifier. 



<!-- Start of picture text -->
0 Cover Blocks  T<br>Initial scene Cover blocks Uncover right cover<br>Initial scene Cover blocks Uncover right cover Uncover left cover<br><!-- End of picture text -->

_Figure 8._ **Failure examples of Cover Blocks** . The Classifier fails to accurately detect the completion of the _Uncover xxx_ subtask, thereby preventing a subtask transition. As the instruction remains unchanged, the model is forced to operate under a wrong task context, leading to unintended and erratic behaviors. 

**Cover Blocks** . The current design of Classifier occasionally fails to accurately perceive the ongoing task progress, leading to an inability to discriminate whether a specific state represents the initiation or the termination of a subtask. This phenomenon is exemplified in Fig. 8 during the execution of the _Cover Blocks_ task. 

**Blocks Ranking Try** . Another primary challenge pertains to the execution of button-pressing operations. We observe that the inclusion of button-pressing actions often introduces interference into hybrid tasks that are not exclusively focused on pressing. For instance, in _Blocks Ranking Try_ , the transition between button-pressing and block-swapping is occasionally fluidly disrupted, a phenomenon illustrated in Fig. 9. 

In the _Blocks Ranking Try_ task, even a single execution error inevitably leads to overall task failure. This sensitivity is the main driver of failure for this task, as clearly substantiated by the quantitative results of our ablation studies. 



<!-- Start of picture text -->
0 Blocks Ranking Try  T<br> … … … …<br>Initial scene Press button Release Try to swap blocks Force button press Unexpected process Chaotic scene<br><!-- End of picture text -->

_Figure 9._ **Failure examples of Blocks Ranking Try** . Upon pressing the button, the system is expected to transition to the next subtask to execute the swapping of designated blocks. However, the Classifier fails to trigger this transition promptly, causing the task to stall in the Press button state. This leads to a coordination conflict between the dual arms: the right hand attempts to initiate manipulation while the left hand remains tethered to the button-pressing instruction. 



<!-- Start of picture text -->
0 Press Button  T<br>Initial scene Press button  Release Press button (incomplete) Release Move to middle button<br>Initial scene Press button  Release Still press button and release Move to middle button<br><!-- End of picture text -->

_Figure 10._ **Failure examples of Press Button** . **(Top)** Insufficient presses: the Classifier issues a false positive termination signal even when the button-press is unsuccessful. **(Bottom)** Excessive presses: the Classifier fails to recognize a successful subtask completion, leading to redundant execution of the same subtask. 

15 



<!-- Start of picture text -->
RMBench: Memory-Dependent Robotic Manipulation Benchmark with Insights into Policy Design<br>0 Battery Try  T<br> … … … …<br>Initial scene Grasp battery Battery slips away Insert battery into slot Unexpected process Chaotic scene<br>Initial scene Battery inserted into slot Switch left Switch right Switch left (bad grasping) Bad placement Bad placement<br><!-- End of picture text -->

_Figure 11._ **Failure examples of Battery Try** . **(Top)** For the horizontally oriented cylindrical battery, a suboptimal grasp pose prevents a successful lift and causes significant displacement, leading the model into unforeseen observational states. **(Bottom)** The model fails to commit to a specific manipulation strategy during battery adjustment, resulting in a mean action that leads to improper placement in the slot. 

**Press Button** . Specifically, the _Press Button_ task, as a dedicated button-pressing scenario, underscores the inherent difficulty of this operation. In the current design of Mem-0, the state of the button (pressed vs. unpressed) is reflected in the visual input only through extremely subtle cues. Consequently, the visual tokens generated by the VLM backbone lack the granularity to encapsulate such fine-grained information, creating a fundamental bottleneck for the downstream Classifier. Fig. 10 displays representative failure cases of the Classifier in the _Press Button_ task. 

**Battery Try** . Furthermore, for the _Battery Try_ task, a representative failure mode is suboptimal manipulation precision. This encompasses both insertion errors when placing the battery into the slot and challenges in determining the appropriate grasp strategy due to the extremely subtle visual cues of the slot, as illustrated in Fig. 11. 

**Summarize** . To address these observations, we posit that incorporating proprioceptive or tactile feedback could provide the Classifier with critical non-visual information. Moreover, as the Classifier is the most downstream module in Mem-0, optimizing upstream VLM token extraction and the fusion mechanisms of Anchor and Sliding Memory remains a promising avenue. Such refinements would allow the Classifier to operate on input tokens with better-conditioned distributions and enhanced informational saliency. We also anticipate that more interpretable tokens could improve the synergy between diverse subtasks, thereby further enhancing performance in complex scenarios like _Blocks Ranking Try_ . 

Nevertheless, the efficacy of the current design has been validated across various _M_ ( _n_ ) tasks, notably yielding substantial performance gains in _Cover Blocks_ compared to the baseline. We hope that the aforementioned analysis provides valuable insights for future work to further improve performance. 

16 

