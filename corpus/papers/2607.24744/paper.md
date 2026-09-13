Title: Data Pyramid for Embodied Manipulation

URL Source: https://arxiv.org/html/2607.24744

Markdown Content:
arXiv is now an independent nonprofit!
Learn more
×
Back to arXiv
Why HTML?
Report Issue
Back to Abstract
Download PDF
Abstract
1Introduction
2Real-Robot Data
3UMI Data
4Egocentric and Exocentric Data
5Simulation Data
6General Data
7Data Applications in Embodied Foundation Models
8Challenges and Future Directions
9Conclusion
References
License: CC BY 4.0
arXiv:2607.24744v1 [cs.RO] 27 Jul 2026
\project

Embodied Data Pyramid \githubAwesome Embodied Data Pyramid \affiliationlogos          1]PKU 2]NTU 3]HKUST 4]NUS 5]CUHK 6]HKU 7]Duke 8]UCB 9]GBU 10]NJU 11]SJTU \contribution[*]Equal Contributions \contribution[†]Project Leader \contribution[‡]Corresponding Author

Data Pyramid for Embodied Manipulation
Yifan Ye∗
Yankai Fu∗
Yaoxu Lv∗
Bohan Hou∗
Jun Cen∗,†
Lingdong Kong∗
Duo Zheng
Tianxing Chen
Jiaming Liu
Ziang Cao
Yunfan Lou
Wei Chow
Xian Sun
Yingshuo Wang
Kuangzhi Ge
Xiaowei Chi
Xidong Zhang
Zhibo Pang
Yiwu Zhong
Sirui Han
Zhihe Lu
Weihao Yuan
Qifeng Chen
Michael Yu Wang
Yao Mu
Ziwei Liu
Jianfei Yang
Ping Luo
Shanghang Zhang‡
[
[
[
[
[
[
[
[
[
[
[
(July 27, 2026)
Abstract

Multimodal foundation models learned to see and to speak by consuming the whole internet. Embodied agents admit no such shortcut, since they require data that couple observations with physical states and actions. These signals can be provided, to varying degrees, by multiple data sources. In this work, we organize the embodied data ecosystem as a “pyramid” spanning five complementary sources: 1real-robot data, 2UMI-style data, 3egocentric and exocentric data, 4simulation data, and 5general vision-language data. We organize the pyramid around the tension between scalability and robot alignment, and further characterize each source in terms of data quality, diversity, reusability, and physical fidelity. We then analyze recent embodied foundation models through the lens of their data recipes, examining how different sources are selected, aligned, and mixed during pretraining. For embodied brain models, vision-language-action models, and world-action models alike, we relate data composition to capabilities in perception, reasoning, planning, action generation, and world prediction. We close by discussing six open challenges: building large-scale tactile datasets, collecting failure and recovery data, developing scalable data-collection pipelines, aligning actions across embodiments, leveraging egocentric data for dexterous manipulation, and designing principled data recipes for robot learning. We hope this work paves the foundation for the design of next-generation embodied systems.

Figure 1:Overview of Organization and Scope. This work is organized into three parts. We first introduce the embodied data pyramid, which categorizes data sources into real-robot, UMI-style, egocentric-exocentric, simulation, and general data according to their scalability and robot alignment, and reviews their datasets, collection pipelines, embodiments, sensing configurations, supervision, scale, diversity, and transferability. We then examine how these heterogeneous sources support embodied foundation models. Finally, we discuss future directions in embodied data.
1Introduction

Multimodal foundation models have acquired broad perceptual, linguistic, and reasoning capabilities largely by scaling pretraining over abundant visual and linguistic data [24, 88]. Extending this paradigm to embodied foundation models requires more than the ability to process visual and linguistic inputs. Embodied agents must additionally understand physical states and dynamics, reason about how actions transform the environment, and execute appropriate behaviors in the physical world [258, 23, 174, 20]. These requirements fundamentally change the nature of supervision needed for pretraining and raise a central question:

What data should be used to train embodied foundation models with such capabilities?

The community has explored several complementary sources of supervision for embodied foundation models. These include observation-action trajectories collected from physical robots and simulation, egocentric and exocentric recordings of human interaction, general image, video, language, and vision-language data, and, more recently, UMI-style demonstrations that capture object- and end-effector-centric manipulation without requiring a robot during collection [24, 26, 359, 119, 23, 62]. Each source contributes different forms of semantic, temporal, physical, and action-related supervision. However, their respective roles, trade-offs, relationships, and integration strategies remain insufficiently systematized.

Several recent embodied foundation models, including Motus [18] and GR00T [20], have introduced hierarchical or pyramid-like views of training data to motivate the use of heterogeneous sources. These formulations are typically designed around the training recipe of a particular model and provide limited analysis of how data categories differ in collection scalability, embodiment dependence, physical fidelity, transferability, and downstream utility. Existing surveys likewise tend to focus on model architectures or specific subproblems, including Vision-Language-Action models [446], World-Action models [141], and world modeling [177]. Consequently, the embodied data ecosystem has not yet been systematically organized at the category level, nor has the use of heterogeneous data by embodied foundation models been comprehensively analyzed from a data-centric perspective.

To answer this question systematically, we therefore frame the embodied data problem around two questions:

• 

“How can heterogeneous embodied data sources be collected, organized, compared, and integrated despite their differences in scalability, robot alignment, physical fidelity, and transferability?”

• 

“How can these heterogeneous data sources be effectively selected, combined, and utilized by embodied foundation models?”

To examine the first question, we organize the existing embodied data ecosystem into a data pyramid for robotics and embodied intelligence. The pyramid is guided by two complementary design principles. The first is Scalability, which concerns how efficiently a data source can be expanded with respect to hardware dependence, human labor, environment reset, safety supervision, and marginal generation cost. The second is Robot Alignment, which concerns how directly its observations, representations, and supervision signals can support learning and execution on physical robots.

Figure 2:Evolution of Data Scale. Evolution of data scale across the embodied data pyramid. From left to right, we select a subset of representative datasets from simulation data, UMI data, robot data, ego data, and general data to visualize their chronological scale growth. Simulation, UMI, and robot datasets are measured by the number of demonstrations, ego datasets by the number of collected hours, and general datasets by the number of question-answer pairs. The inset curves summarize the scale evolution of selected datasets within the corresponding data categories.

These two principles are often in tension: data that is closely aligned with real-robot execution is typically expensive and difficult to scale, whereas highly scalable data may provide only indirect or imperfect supervision for physical interaction. Simulation, for example, can continuously generate robot-oriented trajectories once the required assets, tasks, and generation pipelines have been constructed, but its utility remains constrained by discrepancies between simulated and real physical systems [359, 144, 332, 262, 67, 202].

Beyond these two primary organizing principles, we introduce four complementary category-level dimensions to characterize the broader utility of different data sources. Quality concerns the validity, consistency, informativeness, and task relevance of the collected experience; simply increasing data volume may provide limited benefit when trajectories are repetitive, annotations are inaccurate, signals are poorly synchronized, or collection conditions are systematically biased [211]. Diversity reflects coverage across tasks, objects, scenes, viewpoints, instructions, embodiments, sensing configurations, behaviors, and outcomes, and is important for robustness under distribution shift [64, 113]. As shown in Figure 4, representative datasets exhibit substantially different distributions of interaction trajectories, observation viewpoints, task stages, and operating environments. Reusability characterizes how readily data can be transferred across tasks, environments, embodiments, sensing systems, and model families. For example, object- or end-effector-centric demonstrations can be retargeted to more dexterous embodiments [62, 303], while egocentric videos provide reusable supervision for hand-object interaction, affordance understanding, and task decomposition [118, 119, 171, 391, 325]. Finally, physical fidelity describes how faithfully a data category captures real interaction dynamics and feedback, including contact, friction, compliance, sensing noise, actuation delay, and object motion. These four dimensions complement scalability and robot alignment by explaining whether a data source is not only large and robot-relevant, but also informative, diverse, transferable, and physically meaningful.

Taken together, these six dimensions motivate our category-level organization of the embodied data ecosystem. From the apex to the base, we order the five categories as real-robot data, UMI-style data (Universal Manipulation Interface), egocentric and exocentric data, simulation data, and general data. This ordering reflects an overall transition from stronger physical grounding and more directly actionable robot supervision toward greater accessibility and scalability.

Descending the pyramid is thus a systematic trade. Real-robot data sits at the apex because its recorded actions are directly executable on the platform that produced them, a fidelity paid for in hardware, human effort, and resets [24, 347, 64, 26]. Each layer below relaxes some part of that coupling in exchange for reach: UMI-style rigs drop the robot from the collection loop while retaining end-effector supervision [62, 441, 218, 303, 224]; human recordings drop the gripper but keep real physics and everyday diversity [118, 119, 148, 171, 391, 237]; simulation restores executable actions and privileged labels but approximates the physics itself [359, 144, 332, 262, 67, 202]; and general data abandons robot grounding altogether for web-scale semantic and reasoning coverage [23, 74]. The cards below summarize each layer, and the ordering represents an overall synthesis of the six dimensions rather than a strictly monotonic progression along every individual property.

The Data Pyramid at a Glance
Two primary axes organize the pyramid. Scalability captures how efficiently a data source can be expanded, while robot alignment captures how directly it supports learning and execution on physical robots. These axes are often in tension. Quality, diversity, reusability, and physical fidelity further characterize the utility and limitations of each layer.
Type 1: Real-Robot Data: 
Real-Robot
Trajectories teleoperated or scripted on the target robot, recording observations, states, actions, and outcomes in one closed loop. Nothing is more directly executable, or more expensive: every hour needs hardware, an operator, and a reset.
Type 2: UMI-Style Data: 
UMI
Demonstrations captured with handheld grippers that record end-effector motion without a robot in the loop, freeing collection to move into everyday environments. The gripper supervision survives that move; joint-level proprioception does not, and retargeting remains.
Type 3: Egocentric / Exocentric Data: 
Egocentric
First- and third-person recordings of people acting, offering real physics, dexterous hands, and everyday variety at the scale wearable capture allows. No robot is involved, so actions must be reconstructed and retargeted across the human-robot gap.
Type 4: Simulation Data: 
Simulation
Interaction generated in physics engines, returning executable actions alongside privileged labels such as contacts, poses, and success signals, in parallel and at negligible marginal cost. What it cannot return is the physics it approximates.
Type 5: General Data: 
General
Web-scale images, video, language, and vision-language corpora carrying semantics, spatial structure, and commonsense far beyond any robot dataset. They ground perception and reasoning rather than action, and say nothing of contact or consequence.

The data pyramid provides a static, category-level organization of the embodied data ecosystem. Beyond this taxonomy, we further examine how the scale of each data category has evolved over time. As shown in Figure 2, real-robot, UMI-style, egocentric and exocentric, simulation, and general datasets have all expanded over time, with each category exhibiting a strong growth trend.

We further examine embodied foundation models from a data-centric perspective. As shown in Figure 3, we examine the data sources used during the pretraining of representative embodied foundation models. The reviewed systems indicate a transition from early training recipes dominated by real-robot trajectories toward increasingly heterogeneous mixtures of real-robot, simulation, general, egocentric, and, less frequently, UMI-style data. A more comprehensive summary of the models covered and their data compositions is provided in Table 7.

Beyond documenting these data recipes, we analyze how embodied foundation models accommodate heterogeneous datasets from two perspectives: action-space alignment and geometric alignment. The former concerns how models reconcile differences in action dimensionality, control interfaces, and action semantics across robot embodiments, whereas the latter concerns how observations and demonstrations collected from different viewpoints, coordinate systems, and embodiments are mapped into robot-compatible representations. We then examine how three representative model families draw on different layers of the data pyramid. Embodied brain models primarily use broad multimodal and embodied data to support perception, spatial and temporal reasoning, affordance understanding, memory, and high-level planning [258, 158, 74]. Vision-language-action models require robot-compatible action supervision to map observations and instructions to executable behaviors [23, 174, 20]. World action models additionally exploit action-free temporal data, action-conditioned interaction, and synthetic experience to model environment evolution and action consequences [8, 109, 3, 36, 141]. This analysis connects the composition and alignment of heterogeneous training data with their roles across different embodied foundation-model architectures.

Figure 3:Evolution of Data Utilization. Evolution of data utilization in representative embodied foundation models. Early systems primarily rely on real-robot trajectories, whereas recent systems increasingly explore co-training with real-robot, egocentric, simulation, general, and UMI-style data. The architectural landscape also expands from action-centric VLA systems toward world-model-augmented and unified world-action models. Data sources are indicated by 
 real-robot data, 
 egocentric data, 
 UMI data, 
 simulation data, and 
 general data.

Finally, we organize the key challenges and future directions for embodied data around three questions: what to collect, how to collect it, and how to use it. First, future datasets should better represent interaction signals that remain scarce, particularly failure and recovery trajectories [21, 70, 235] and tactile feedback that captures contact states, forces, material properties, and fine-grained interaction dynamics [444, 232, 387]. Second, data-collection pipelines should reduce their dependence on costly manual teleoperation, broaden task and environmental coverage, and adaptively prioritize underrepresented or informative interactions [24, 26, 62]. Third, heterogeneous data should be used more systematically by aligning action representations across embodiments, determining effective mixtures of data from different pyramid layers, and transferring human interaction experience to robot manipulation [64, 20, 303, 171, 391].

Overall, this work organizes the embodied data ecosystem through a data-pyramid taxonomy, reviews the construction and collection of each data category, and examines how heterogeneous sources are utilized by embodied foundation models.

The main contributions of this work are summarized as follows.

• 

A data-pyramid taxonomy guided by scalability and robot alignment. We organize embodied AI data into five complementary categories, ordered from apex to base: real-robot data, UMI-style data, egocentric and exocentric data, simulation data, and general data. We characterize each category in terms of quality, diversity, reusability, physical fidelity, collection scalability, and alignment with physical robot learning. To support the community, we also release and maintain an open-source repository that curates representative datasets and related resources under this taxonomy.

• 

A data-centric analysis of embodied foundation models. We examine how heterogeneous data sources support embodied vision-language models, vision-language-action models, and world action models. Our analysis connects dataset properties, including semantic coverage, temporal structure, action supervision, embodiment alignment, diversity, and physical fidelity, with model capabilities such as perception, reasoning, planning, action generation, and world prediction.

• 

A discussion of open challenges and future directions. We identify key challenges in embodied data, including quality and diversity assessment, cross-embodiment reuse, multimodal standardization, scalable real-world collection, human-to-robot and sim-to-real transfer, failure and recovery data curation, and the integration and evaluation of heterogeneous data sources. We further discuss opportunities for constructing more general, robust, reusable, and physically grounded embodied agents.

Figure 4:Visualization of action-trajectory diversity across representative datasets. Key action frames are extracted using a heuristic procedure following PerAct [314]. Under comparable initial configurations, the spatial distribution of grasping and interaction keyframes provides a proxy for the diversity of demonstrated trajectories. Human-hand trajectories are processed with an analogous heuristic to extract corresponding hand keyframes.
2Real-Robot Data
Figure 5:Overview of UMI data and real-robot data. Left: UMI data are collected with portable human-operated interfaces, emphasizing robot-free in-the-wild demonstrations, multimodal sensing, and relative end-effector trajectories. Right: real-robot data are collected on physical robot embodiments with diverse teleoperation systems, sensing modalities, and collection paradigms. Retargeting connects the two by transferring UMI demonstrations to executable robot actions for cross-embodiment deployment.
2.1Data Overview

Real-robot data occupies the apex of the embodied-intelligence data pyramid, representing the layer closest to real-world execution. It is collected through the physical interaction between a robot and the environment, typically comprising sensory observations, robot states, and control actions. In contrast to human videos or embodiment-agnostic demonstration data, it directly captures the closed-loop relationship between perception, action, and physical consequences under a specific robotic embodiment. Consequently, it is characterized by high quality, physical fidelity, diversity, and reusability, making it the most direct source of supervision for learning executable manipulation policies and grounding model predictions in real-world dynamics.

The development of real-robot datasets has evolved from isolated, task-specific collections toward increasingly general and reusable data resources, as shown in Table 1. Early efforts primarily focused on large numbers of trials for a small set of behaviors, such as robotic grasping, often collected using scripted controllers in controlled environments [275, 169]. Subsequent datasets expanded toward human-operated demonstrations covering broader task sets and more diverse scenes [308, 248, 91, 173]. More recent efforts further scale data collection across institutions, robot platforms, and application domains [64, 366, 372, 142], incorporating bimanual systems [335, 98], mobile manipulators [103], humanoids [440, 294], and dexterous hands [435, 203]. This progression reflects a broader shift from collecting data for individual policies toward constructing large-scale data ecosystems for training generalist robot models.

Table 1:Statistics and key properties of real-robot manipulation datasets. “Embod.” denotes the number of distinct embodiment types; “Calib.” denotes calibrated camera extrinsics. “Dex.” denotes dexterous-hand data. “Anno.” denotes subtask labels within episodes; “Mobile” denotes mobile manipulation data. In the “Arm” column, ‘S’, ‘D’, ‘H’ denote single-arm, dual-arm, and Humanoid embodiments.
Dataset
 	
Time
	
Arm
	
Embod.
	
Traj. / Hours
	
Tasks
	
Modality
	
Calib.
	
Dex.
	
Anno.
	
Mobile


Pinto and Gupta [275]
 	
2015
	
S
	
1
	
50K / 700
	
1
	
RGB; Proprio.; Act.; Grasp Succ.
	
✗
	
✗
	
✗
	
✗


DAML [407]
 	
2018
	
S
	
2
	
2.9K / 4.08
	
3
	
RGB; Proprio.; Act.
	
✗
	
✗
	
✗
	
✗


MIME [308]
 	
2018
	
S/D
	
1
	
8.3K / 13.7
	
20
	
RGB-D; Proprio.
	
✗
	
✗
	
✗
	
✗


QT-Opt [169]
 	
2018
	
S
	
1
	
580k / 800
	
1
	
RGB; Proprio.; Act.; Grasp Succ.
	
✗
	
✗
	
✗
	
✗


RoboTurk [248]
 	
2018
	
S
	
3
	
2.1K / 111.3
	
3
	
RGB-D; Proprio.;
	
✗
	
✗
	
✗
	
✗


RoboNet [75]
 	
2019
	
S
	
7
	
162k / 833.3
	
-
	
RGB; Proprio.; Act.
	
✗
	
✗
	
✗
	
✗


MT-Opt [170]
 	
2021
	
S
	
1
	
800k / -
	
12
	
RGB; Proprio.; Act.
	
✗
	
✗
	
✗
	
✗


BridgeData [91]
 	
2021
	
S
	
1
	
7.2k / -
	
71
	
RGB; Proprio.; Act.
	
✗
	
✗
	
✗
	
✗


BC-Z [154]
 	
2022
	
S
	
1
	
26K / 125
	
100
	
RGB; Proprio.; Act.; Lang.
	
✗
	
✗
	
✗
	
✗


RT-1 [24]
 	
2022
	
S
	
1
	
130K / -
	
700+
	
RGB; Proprio.; Act.; Lang.
	
✗
	
✗
	
✗
	
✓


FurnitureBench [134]
 	
2023
	
S
	
1
	
5.1K / 219.6
	
9
	
RGB-D; Proprio.; Act.
	
✗
	
✗
	
✗
	
✗


RH20T [95]
 	
2023
	
S
	
4
	
110K / -
	
147
	
RGB-D; Proprio.; Act.; Lang.; Tactile; Force; Audio
	
✓
	
✗
	
✗
	
✗


RoboSet [16]
 	
2023
	
S
	
1
	
98.5K / -
	
48
	
RGB-D; Proprio.; Act.; Lang.; Force; Torque
	
✗
	
✗
	
✓
	
✗


BridgeData V2 [347]
 	
2023
	
S
	
1
	
60.1K / -
	
13
	
RGB; Proprio.; Act.; Lang.
	
✗
	
✗
	
✗
	
✗


Open X-Embodiment [64]
 	
2023
	
S/D
	
22
	
2.4M / -
	
527
	
RGB-D; Proprio.; Act.; Lang.; Force
	
✗
	
✗
	
✗
	
✗


ALOHA Unleashed [439]
 	
2024
	
D
	
1
	
26.2K / -
	
5
	
RGB; Proprio.; Act.
	
✗
	
✗
	
✗
	
✗


DROID [173]
 	
2024
	
S
	
1
	
76K / 350
	
86
	
RGB; Proprio.; Act.; Lang.
	
✓
	
✗
	
✗
	
✗


EgoMimic [171]
 	
2024
	
D
	
1
	
1K / 12
	
3
	
RGB; Proprio.; Act.
	
✓
	
✗
	
✗
	
✗


FMB dataset [240]
 	
2024
	
S
	
1
	
22.6K / -
	
2
	
RGB-D; Proprio.; Act.; Force; Torque
	
✗
	
✗
	
✗
	
✗


RoboMIND [366]
 	
2024
	
S/D
	
4
	
107K / 305.5
	
479
	
RGB; Proprio.; Act.; Lang.; Fail. labels
	
✓
	
✓
	
✓
	
✗


AgiBot World Beta [26]
 	
2025
	
D
	
1
	
1M / 2976.4
	
217
	
RGB-D; Proprio.; Act.; Lang.; Force
	
✓
	
✓
	
✓
	
✓


REASSEMBLE [323]
 	
2025
	
S
	
1
	
4.6K / 13.02
	
121
	
RGB; Proprio.; Act.; Audio; Force; Torque
	
✓
	
✗
	
✓
	
✗


PH2D [283]
 	
2025
	
H
	
1
	
1.6K / -
	
-
	
RGB; Proprio.; Act.; Lang.
	
✗
	
✓
	
✗
	
✗


AIST-Bimanual [257]
 	
2025
	
D
	
1
	
10.7K / -
	
119
	
RGB; Proprio.; Act.; Lang.
	
✗
	
✗
	
✗
	
✗


ActionNet [98]
 	
2025
	
H
	
3
	
30K / 140
	
-
	
RGB-D; Proprio.; Act.; Lang.
	
✗
	
✓
	
✗
	
✗


MotionTrans [411]
 	
2025
	
S
	
1
	
1.5K / -
	
15
	
RGB; Proprio.; Act.; Lang.
	
✗
	
✓
	
✗
	
✗


Open Galaxea [162]
 	
2025
	
D
	
1
	
20.5K / 500
	
150
	
RGB-D; Proprio.; Act.; Lang.
	
✗
	
✗
	
✓
	
✓


RealSource World [292]
 	
2025
	
D
	
1
	
26.7K / -
	
35
	
RGB; Proprio.; Act.; Lang.; Force; Torque
	
✓
	
✗
	
✓
	
✗


RoboCOIN [372]
 	
2025
	
D
	
15
	
180K / -
	
421
	
RGB-D; Proprio.; Act.; Lang.
	
✗
	
✓
	
✓
	
✗


Humanoid Everyday [440]
 	
2025
	
H
	
2
	
10.3k / 27.8+
	
260
	
RGB-D; Proprio.; Lang.; Tactile; LiDAR; IMU; Odometry
	
✗
	
✓
	
✗
	
✓


RoboMIND 2.0 [142]
 	
2026
	
S/D
	
6
	
310K / 1000+
	
739
	
RGB-D; Proprio.; Act.; Lang.; Tactile; Force; Torque; Fail. labels
	
✓
	
✓
	
✓
	
✓


DECO-50 [203]
 	
2026
	
D
	
1
	
8K / 50+
	
28
	
RGB; Proprio.; Act.; Tactile
	
✗
	
✓
	
✗
	
✗


Dexora [435]
 	
2026
	
D
	
1
	
12.2K / 40.5
	
200
	
RGB; Proprio.; Act.; Lang.
	
✗
	
✓
	
✗
	
✗


MolmoAct2-BimanualYAM [97]
 	
2026
	
D
	
1
	
34.5K / 720
	
736
	
RGB; Proprio.; Act.; Lang.
	
✗
	
✗
	
✗
	
✗


MolmoAct2-SO100/101 [97]
 	
2026
	
S
	
2
	
38K / 184
	
1315
	
RGB; Proprio.; Act.; Lang.
	
✗
	
✗
	
✗
	
✗


AgiBot World 2026 [335]
 	
2026
	
D
	
1
	
20.7K / -
	
-
	
RGB-D; Proprio.; Act.; Lang.; Tactile
	
✓
	
✓
	
✓
	
✓


Unitree UnifoLM-WBT [294]
 	
2026
	
H
	
1
	
4.5K / 30.74
	
14
	
RGB; Proprio.; Act.; Lang.
	
✗
	
✓
	
✗
	
✓


Open-H-Embodiment [264]
 	
2026
	
S/D
	
20
	
125.8K / 780
	
33
	
RGB; Proprio.; Lang.
	
✗
	
✗
	
✗
	
✗


Baihu-VTouch [145]
 	
2026
	
S/D
	
3
	
- / 1000+
	
380+
	
RGB-D; Proprio.; Act.; Lang.; Tactile
	
✗
	
✓
	
✗
	
✓


LET-Base-Dataset [186]
 	
2026
	
H
	
2
	
91.5K / 1000+
	
31
	
RGB-D; Proprio.; Lang.; IMU
	
✗
	
✓
	
✓
	
✗


LET-Dex-Dataset
 	
2026
	
H
	
1
	
3K / 20
	
14
	
RGB-D; Proprio.; Lang.; Tactile; Force; Torque; IMU
	
✗
	
✓
	
✗
	
✗


Haptile [4]
 	
2026
	
S
	
1
	
1.7K / 12.5
	
38
	
RGB; Proprio.; Act.; Lang.; Tactile
	
✗
	
✗
	
✗
	
✗


OmniViTac [444]
 	
2026
	
S
	
1
	
21.9K / -
	
86
	
RGB-D; Proprio.; Act.; Lang.; Tactile
	
✗
	
✗
	
✗
	
✗


ABC-130K [5]
 	
2026
	
D
	
1
	
130.7K / 3591
	
197
	
RGB; Proprio.; Act.; Lang.; Torque
	
✗
	
✗
	
✓
	
✗
2.2Embodiment and Modalities

Real-robot datasets vary substantially in the physical embodiments through which interaction data are collected. Early datasets were predominantly built around fixed-base, single-arm manipulators equipped with parallel grippers, as exemplified by large-scale grasping and manipulation datasets such as Pinto and Gupta, QT-Opt, and RoboNet [275, 169, 75]. This configuration offers a relatively constrained action space and facilitates scalable data collection, but covers only a limited range of whole-body and coordinated manipulation behaviors. Later datasets increasingly incorporate dual-arm systems for coordinated and bimanual manipulation [438, 439, 257], mobile manipulators that jointly control the base and upper body [103, 162], and humanoid platforms involving whole-body motion and locomotion [132, 102, 440]. More recent efforts further extend data collection to dexterous hands and heterogeneous robot fleets, enabling the study of fine-grained contact-rich manipulation and cross-embodiment learning [372, 142, 435].

Another notable trend is the increasing richness of sensing modalities in real-robot datasets. Beyond the standard vision-proprioception-action interface, several datasets incorporate additional signals to capture aspects of physical interaction that cannot be reliably inferred from images alone. Tactile sensing has emerged as an important complementary modality for contact-rich and dexterous manipulation, as it provides local, surface-level information about contact location, pressure distribution, incipient slip, grasp stability, and object-hand interaction states [95, 440, 142, 203, 444, 4]. In contrast, force and torque measurements typically characterize the net interaction wrench at the wrist, gripper, or robot joints, offering a more global view of contact forces and interaction dynamics [95, 323, 240]. Audio can further provide complementary cues about collisions, friction, and task progression [95, 323]. Whole-body platforms may additionally include IMU, LiDAR, and odometry signals to represent balance, locomotion, and spatial navigation [440, 186]. These richer multimodal observations not only enable a more comprehensive characterization of robot and environment interaction, but also increase heterogeneity across datasets.

Proprioceptive and action representations also differ considerably across embodiments. Fixed-base manipulators commonly record joint positions (qpos), represented as an ordered vector of joint coordinates, or end-effector (EEF) poses, represented by the Cartesian position and orientation of the wrist or tool, together with gripper states. Correspondingly, actions are typically expressed in either joint space or Cartesian EEF space [75, 91, 134, 16, 95, 64]. Dexterous hands, due to their high-dimensional articulated structures, are typically represented through finger joint positions [366, 372, 203, 435], sometimes complemented by fingertip poses to facilitate alignment with human data [391, 101, 443]. Mobile and humanoid platforms further incorporate base velocity, torso configuration, and whole-body joint states [103, 440, 142, 294].

These differences affect not only the dimensionality of the learning problem, but also the extent to which data can be shared across platforms. Joint-space representations provide direct and precise control for articulated hands, but are strongly tied to robot morphology. In contrast, end-effector-centric representations can offer a more transferable interface for arm motion across embodiments, although they still require consistent coordinate frames and control conventions. In policy learning, proprioceptive states and actions are often normalized to maintain suitable numerical scales and facilitate learning across embodiments. Common choices include MinMax, Q01-Q99, and MeanStd normalization [377].

2.3Data Collection Paradigms
2.3.1Script

Scripted collection broadly refers to data-generation processes in which robot behavior is executed without continuous human control. Rather than manually controlling each episode, human effort is shifted toward designing the task logic, control pipeline, and execution conditions. This separation between system design and episode-level execution enables repeated and parallel data collection at relatively low marginal cost. Depending on how robot actions are produced, scripted collection can be broadly categorized into rule-based execution, trajectory playback, and autonomous policy rollouts.

Rule-Based Execution. In rule-based collection, robot behavior is generated through manually specified task logic, often implemented as action sequences, finite-state machines, or heuristic controllers. A task is typically decomposed into predefined stages, such as detecting a target, selecting an interaction pose, approaching the target, executing the grasp, and transporting the object to a designated region. Within this pipeline, inverse kinematics, motion planning, and trajectory optimization can be used to convert stage-level goals into kinematically feasible and collision-free robot motions, while visual observations or sensor thresholds determine target poses and trigger transitions between stages. Early large-scale grasping datasets adopted such structured pipelines to execute repeated grasp attempts and automatically record both successful and failed interactions [275, 75, 355]. Rule-based execution is particularly effective for tasks with clearly specified objectives, structured workspaces, and reliable reset mechanisms. However, because both the task decomposition and motion-generation process rely on predefined assumptions, the collected behaviors may exhibit limited diversity and may not generalize well to unanticipated states or complex contact interactions.

Trajectory Playback. Trajectory playback collects data by re-executing previously recorded robot motions. The source trajectories may originate from human teleoperation, kinesthetic teaching, motion planning or earlier policy rollouts. During playback, the robot can reproduce demonstrations under different object configurations or environmental conditions, and perturbations may be introduced into the initial state, trajectory, or control parameters to increase variation. This approach reduces the need for repeated human operation while preserving the overall structure of successful demonstrations. For example, RoboSet combines human-collected demonstrations with playback-based execution to expand its real-robot experience [16]. Nevertheless, playback is sensitive to discrepancies between the original and current environment, since open-loop reproduction may fail when object locations or contact conditions change substantially. Closed-loop correction and replanning are therefore often required for reliable reuse.

Autonomous Policy Rollouts. A more adaptive form of automated collection uses learned or heuristic policies to interact with the environment and generate new trajectories. Such policies may be obtained through reinforcement learning, imitation learning, or prior rounds of data collection, and can be iteratively improved as additional experience becomes available. Large-scale systems such as QT-Opt [169] and MT-Opt [170] use autonomous rollouts to collect interaction data across repeated task executions, enabling the dataset and policy to improve through a continual data-generation cycle. RoboSet [16] similarly includes heuristic policy rollouts alongside human demonstrations and trajectory playback. Compared with fixed scripts, autonomous rollouts can respond to observations and cover a broader range of states, including policy failures and recovery behaviors. However, their data distribution is strongly shaped by the capabilities and exploration strategy of the current policy, potentially reinforcing existing biases or producing large quantities of low-quality trajectories.

2.3.2Teleoperation

Teleoperation collects robot demonstrations by allowing a human operator to continuously control the robot during task execution. Relative to scripted collection, it can more naturally capture adaptive, goal-directed, and contact-rich behaviors, particularly in tasks where suitable control policies are not yet available. As illustrated in Figure 5, teleoperation systems exhibit diverse configurations.

Kinesthetic Teaching. Kinesthetic teaching provides the most direct form of human control: the operator physically guides the robot arm or end effector through the desired motion while the system records joint states, end-effector poses, and gripper commands. This approach requires little additional interface hardware and allows operators to specify precise trajectories through direct physical interaction, as demonstrated in datasets such as MIME [308]. It is particularly suitable for fixed-base manipulators and tasks requiring accurate geometric guidance. However, physical manipulation of the robot can impose substantial operator effort, and the approach is difficult to apply to high-payload systems, mobile platforms, or complex whole-body embodiments.

Leader-Follower Teleoperation. Leader-follower systems use a dedicated input mechanism whose motion is mapped to a follower robot. When the leader and follower have identical or closely matched kinematic structures, joint-level correspondence provides an intuitive and precise control interface. This paradigm has been widely adopted for bimanual manipulation, as exemplified by ALOHA and its subsequent extensions [438, 439], as well as larger bimanual datasets such as AIST-Bimanual [257] and Open Galaxea [162]. GELLO [371] further demonstrates that the leader can be implemented as a low-cost, scaled, and kinematically equivalent replica of the target arm, using 3D-printed components and inexpensive actuators while retaining direct joint-level correspondence. Leader-follower control is well suited to coordinated two-arm behaviors because it preserves temporal synchronization between both arms and reduces the complexity of Cartesian control. Its main limitation is the dependence on specialized, and often embodiment-specific, hardware, which can constrain portability across robot platforms.

Integrated Leader-Follower Teleoperation. Integrated leader-follower teleoperation combines the input device and controlled robot within the same mechanism: the operator steers a gravity-compensated arm through an end-effector handle while the system records robot states, visual observations, and interaction signals. This method provides direct visual and force feedback, supports precise contact-rich manipulation, and avoids the kinematic-mapping errors of separate leader arms. Eliminating the dedicated leader device also reduces hardware, calibration, and workspace requirements. A representative practical example is the ARX AC-One, which has recently been used as a real-world platform for bimanual manipulation research [18, 28]. The main drawback is that the operator’s hand may appear in training images but be absent during autonomous deployment, creating a visual distribution shift. However, existing evidence from human-to-robot [391, 443] transfer suggests that models can learn to accommodate such differences during pretraining.

Device-Mediated Teleoperation. A more portable alternative uses generic input devices, including VR or XR controllers, smartphones, SpaceMouse devices and joysticks. Human inputs are commonly interpreted as Cartesian end-effector poses or velocities and converted into robot joint commands through inverse kinematics or low-level controllers. RoboTurk [248], for example, uses a mobile device for six-degree-of-freedom control, while DROID [173] employs VR controllers to collect manipulation demonstrations across diverse environments. SpaceMouse and haptic devices have also been used in datasets such as FMB [240] and REASSEMBLE [323]. For mobile manipulators, foot pedals can provide an additional hands-free control channel for commanding base motion, allowing the operator to navigate the platform while keeping both hands available for arm manipulation. Because these devices are largely independent of robot morphology, they can be deployed across different manipulators with relatively limited hardware modification. Yet their indirect control interfaces may introduce latency, scaling ambiguities, and reduced precision, especially for coordinated, high-dimensional, or contact-intensive tasks.

Vision-Based Teleoperation. Vision-based systems infer human body, arm, or hand motion directly from RGB or RGB-D observations, without requiring physical markers or dedicated wearable sensors. The estimated poses are then retargeted to the robot while accounting for kinematic feasibility and joint constraints. Such approaches can support both whole-body humanoid control and dexterous-hand teleoperation. HumanPlus [102], for example, transfers monocularly estimated human motion to a humanoid robot, while DexPilot [130] and Anyteleop [281] map visually estimated human hand motion to a dexterous robot hand. By reducing the need for specialized sensing hardware, vision-based teleoperation provides a low-cost and accessible interface, although its performance remains sensitive to occlusion, fast motion, and pose-estimation errors.

Wearable and Motion-Capture Retargeting. Wearable systems capture human motion using devices such as inertial suits, optical trackers, data gloves, or arm and hand exoskeletons. These interfaces can provide higher-dimensional and more stable motion measurements than generic controllers, making them particularly suitable for bimanual, dexterous, and whole-body teleoperation. RoboMIND [366, 142] combines multiple teleoperation interfaces, including motion-capture systems, to collect data across heterogeneous robot embodiments, while RoboCOIN [372] extends wearable control to a large collection of dual-arm platforms. For dexterous manipulation, systems such as Dexora [435] use arm exoskeletons and VR-based hand control to capture coordinated arm and finger motion. Whole-body teleoperation systems further map human upper-body, locomotion, and base motion to humanoid robots [294, 440]. The required mapping depends on the sensing interface and the degree of correspondence between the human and robot embodiments. Exoskeleton-based systems often provide direct joint-level control, whereas data gloves and motion-capture devices typically require calibration and kinematic retargeting to account for differences in scale, joint ranges, and morphology.

Feedback Mechanisms. Feedback mechanisms constitute another important dimension of teleoperation-system design. Visual feedback provides the operator with observations of the environment, objects, and robot state through external or egocentric cameras. Immersive systems further employ stereoscopic displays and head-tracked active viewpoints to improve depth perception and spatial awareness [59, 86]. Force feedback transmits interaction forces or end-effector resistance from the robot to the operator through bilateral leader devices, haptic controllers, or exoskeletons, enabling more precise regulation of contact during insertion, assembly, and other force-sensitive tasks [323, 421]. Tactile feedback communicates localized contact events at the fingers or grasp interface, for example, through vibration, pressure, or fingertip stimulation, helping the operator perceive contact onset and adjust grasping behavior [86, 421, 4]. The integration of these feedback channels improves teleoperation accuracy, stability, and operator awareness, enabling more natural control and higher-quality demonstrations, particularly in contact-rich tasks.

2.3.3Human-in-the-Loop Enhancement

Human-in-the-loop enhancement is an iterative real-robot data collection paradigm in which a human supervisor intervenes during policy execution and records corrective actions at difficult or failure-prone states. Rather than repeatedly collecting complete demonstrations, it selectively augments the dataset with policy-induced states, failure cases, and recovery trajectories that are underrepresented in the original training data.

Following the dataset-aggregation principle introduced by DAgger [297], robot-learning methods have progressively improved the efficiency and scalability of this process. Intervention Weighted Regression [249] collects corrective demonstrations through remote teleoperation and increases their contribution during training. ThriftyDAgger [135] selectively requests assistance according to state novelty and estimated failure risk, while Sirius [220] continuously incorporates intervention data gathered during robot deployment. Fleet-DAgger [136] extends this collection process to multiple concurrently operating robots. In dexterous manipulation, DexCap [348] records human takeovers and residual corrections during policy rollouts. IntervenGen [137] subsequently improves data efficiency by generating additional corrective trajectories from a small number of human interventions. More recently, CR-DAgger [385] collects compliant delta-action corrections for contact-rich manipulation without fully interrupting autonomous execution. Overall, human-in-the-loop enhancement transforms errors encountered during real-robot deployment into targeted training data, improving the coverage of challenging states while reducing the need for full-trajectory demonstrations.

2.4Scale and Diversity

The scale of real-robot data has increased substantially over the past decade. Early datasets often accumulated large numbers of interactions within a narrowly defined task distribution. For example, Pinto and Gupta [275] and QT-Opt [169] collected tens or hundreds of thousands of grasping trials, but primarily focused on a single behavior. Subsequent efforts expanded both the number of trajectories and the breadth of tasks. MT-Opt [170] contains approximately 800K episodes across 12 tasks, while RT-1 [24] and RoboMIND [366] cover hundreds of manipulation skills with 130K and 107K trajectories, respectively. More recent datasets have reached million-scale collection or thousands of hours of interaction. AgiBot World Beta [26] reports one million trajectories and nearly 3,000 hours of data, while RoboMIND 2.0 [142] contains more than 310K trajectories across 739 tasks and over 1,000 hours of execution. Cross-dataset initiatives such as Open X-Embodiment [64] further aggregate more than two million trajectories from multiple institutions and robot platforms.

For policy learning, data diversity is often more consequential than the raw number of trajectories, as it determines the range of states, behaviors, and interaction conditions represented during training. A clear trend in recent real-robot datasets is therefore the expansion of task coverage, from narrowly defined behaviors such as grasping or assembly to hundreds of household, industrial, mobile-manipulation, and dexterous tasks [372, 142, 97, 435]. This task diversity is accompanied by increasing variation in objects, scene layouts, initial configurations and camera viewpoints, providing broader coverage of the state distributions encountered at deployment. At the same time, datasets increasingly span heterogeneous embodiments, including single-arm and bimanual manipulators, mobile platforms, humanoids, and dexterous hands, as well as richer sensing modalities such as RGB-D, force and torque, audio, IMU, LiDAR, and odometry [64, 372, 440, 142]. Together, diversity across tasks, environments, embodiments, and modalities provides the foundation for learning policies that are more robust and transferable beyond a narrowly defined data-collection setting.

Beyond task, embodiment, and modality diversity, trajectory-level diversity captures variations in how the same nominal task is executed across demonstrations. Such variation may arise from differences in initial and terminal states, motion paths, contact sequences, and execution speeds, thereby broadening the state-action distribution covered within each task. The Cartesian distributions of keyframes in Figure 4 provide an intuitive illustration of the resulting differences in spatial coverage and distributional structure. In the examples shown, AgiBot-World-2026 [335] exhibits several well-separated spatial clusters, suggesting coverage of distinct workspace regions and interaction heights. RoboMIND [366] shows a more compact and anisotropic distribution, whereas RoboMIND 2.0 [142] presents broader, partially connected, and vertically stratified spatial coverage.

2.5Advantages and Limitations

Real-robot data provide physically grounded supervision by directly capturing the relationship between robot observations, actions, and their physical consequences. Compared with simulation or human videos, they naturally reflect real sensing noise, control latency, contact dynamics, and hardware constraints. Moreover, the recorded actions are directly executable on the corresponding platform, making such data particularly valuable for learning reliable manipulation policies and recovery behaviors.

The main limitation is the high cost of collection. Real-robot data require physical hardware, human operation or automated collection pipelines, repeated environment resets, maintenance, and safety supervision, which restrict large-scale and parallel data acquisition. In addition, many datasets remain concentrated on specific robots, tasks, and environments. Differences in embodiment, sensor configuration, coordinate systems, and action representation further complicate data aggregation and cross-platform transfer.

Takeaway
Real-Robot
Real-robot data sits at the apex: it provides direct robot control signals together with observations of physical interactions between the robot and the real world. We believe that future real-robot data development should proceed along three directions. First, teleoperation systems should support simpler, smoother, and more intuitive control while preserving fine-grained trajectories and direct physical interaction feedback. In this respect, Integrated Leader-Follower Teleoperation provides a promising direction because of its relatively low hardware cost, ease of use, and capacity for fluent demonstration collection. Second, real-robot datasets should capture broader in-the-wild variations in scenes, objects, and environmental conditions. AIGC may complement physical collection by synthesizing embodiment-specific data or augmenting scenes and observations, thereby expanding environmental diversity without requiring proportional increases in real-world deployment. Third, data collection should extend beyond human teleoperation to include policy rollouts, which reflect how an embodied agent explores the physical world under its own induced state distribution. Human-in-the-loop interventions during these rollouts can further transform failures into recovery trajectories, providing supervision for error correction, task continuation, and improved policy robustness.
3UMI Data
3.1Data Overview

Universal Manipulation Interface (UMI) data denotes a family of real-world manipulation datasets collected through portable, robot-independent interfaces rather than directly operating a target robot [62]. A typical UMI system integrates one or more cameras, a hand-held gripper or dexterous interface, and a pose-tracking module to record visual observations, end-effector motion, gripper states, and, in some recent systems, force or tactile signals. These demonstrations can then be converted into robot-compatible trajectories through calibration, action transformation, and embodiment-specific retargeting.

Compared with human egocentric videos, UMI data preserves natural, first-person observations of real-world manipulation while additionally providing synchronized end-effector poses, gripper states, and calibrated action trajectories. These structured signals yield high-quality, physically faithful demonstrations that are readily usable for robot policy learning and, with appropriate retargeting, cross-embodiment deployment. Since the original UMI system, this paradigm has expanded from single-arm gripper manipulation [229, 441] to bimanual systems [261], dexterous hands [333, 384, 383], mobile platforms [125], multi-view perception, 3D sensing, and tactile or force-aware data collection [191, 453, 368]. As summarized in Table 2, UMI-style datasets span a growing range of embodiments, sensing modalities, tasks, and collection scales.

3.2UMI Data Collection Pipeline
3.2.1Evolution of UMI-Style Systems

The evolution of UMI-style systems first manifests in the design of the physical collection interface. The original UMI established a robot-free collection paradigm based on a portable hand-held gripper that integrates wrist-mounted cameras, pose tracking, and gripper-state sensing to capture synchronized observations and actions during natural human demonstrations [62]. Later systems improved the portability, modularity, and ease of deployment of this design. FastUMI [441], for example, simplifies the hardware and tracking pipeline to reduce setup effort and support more efficient large-scale collection, while LEGATO [303] introduces a grasping-tool interface designed to provide a consistent demonstration representation across different robot platforms. More recent systems further reconsider how users physically interact with the collection device. In contrast to conventional UMI interfaces, where the operator holds a gripper-shaped tool and controls its opening through a trigger, FreeTacMan [369] adopts a finger-worn interface that allows the operator to directly control the gripper using natural finger motion. This progression reflects a broader shift from mechanically mediated hand-held tools toward more natural, wearable, and contact-aware collection interfaces.

The data produced by UMI-style systems has also become increasingly diverse in terms of embodiment, sensing modality, and scale. Early UMI datasets primarily focused on single-arm manipulation with parallel grippers and wrist-view RGB observations. Subsequent work extended this paradigm to mobile manipulators and whole-body humanoid systems, as in UMI on Legs [125] and HuMI [261], as well as to bimanual manipulation and dexterous hands [333, 384].

The observation space has similarly expanded beyond monocular wrist-view images. MV-UMI [291] introduces additional camera viewpoints, while UMI-3D [362] incorporates explicit 3D spatial observations. ViTaMIn [218], FreeTacMan [369], exUMI [386], and ManipForce [182] further augment visual trajectories with tactile or force-related measurements, enabling the collection of contact-rich manipulation data. At the same time, improvements in collection efficiency have increased dataset scale from several hundred task-specific demonstrations to collections containing tens of thousands or even hundreds of thousands of trajectories, such as FastUMI-100K [224]. Overall, UMI-style data has evolved from single-arm visual trajectories into a broader class of large-scale, multimodal, and multi-embodiment manipulation data.

3.2.2Relative Trajectory Action Representation

Since UMI demonstrations are collected without directly operating a target robot, they do not contain embodiment-specific joint states or joint-space actions. Instead, UMI combines a wrist-mounted camera with an onboard IMU and visual–inertial SLAM to estimate the six-degree-of-freedom pose trajectory of the handheld gripper, along with its open–close state. To reduce sensitivity to the global tracking frame, future end-effector targets are represented relative to the current end-effector pose. This relative trajectory representation preserves the spatial structure of the demonstrated motion while reducing error accumulation during execution. It also provides an embodiment-agnostic action space that can be mapped to different robot platforms through calibration, inverse kinematics, and low-level control.

Table 2:Statistics and key properties of UMI datasets. These are real-world manipulation datasets collected with portable handheld interfaces that enable scalable, low-cost demonstration collection across diverse tasks and environments.
Dataset	Time	Arm Type	Tasks	End Effector	Demos	Tactile
UMI [62]	2024	Single / Bimanual	5	Gripper	2.5K	✗
ManiWAV [229]	2024	Single	5	Gripper	1.0K	✗
UMI on Legs [125]	2024	Single	2	Gripper	514	✗
FastUMI [441]	2024	Single	22	Gripper	9.0K	✗
Data Scaling Laws [211]	2024	Single	6	Gripper	24.1K	✗
LEGATO [303]	2024	Single	6	Gripper	900	✗
ViTaMIn [218]	2025	Single	8	Gripper	841	✓
DexWild [333]	2025	Single	10	Dexterous Hand	9.5K	✗
DexUMI [384]	2025	Single	5	Dexterous Hand	1.8K	Force / Torque
FreeTacMan [369]	2025	Single	50	Gripper	10K	✓
Touch in the Wild [453]	2025	Single	20	Gripper	2.7K	✓
exUMI [386]	2025	Single	8	Gripper	1.7K	✓
MV-UMI [291]	2025	Single	4	Gripper	1.4K	✗
ManipForce [182]	2025	Single	6	Gripper	597	Force / Torque
FastUMI-100K [224]	2025	Single	32	Gripper	100K	✗
ViTaMIn-B [191]	2025	Single	4	Gripper	884	✓
HuMI [261]	2026	Bimanual	5	Gripper	827	✓
RealOmni	2026	Bimanual	Diverse	Gripper	789.8K	✓
UMI-3D [362]	2026	Single	3	Gripper	4.6K	✗
TAMEn [368]	2026	Bimanual	4	Gripper	724	✓
Daimon-Infinity	2026	Bimanual	Diverse	Gripper	274.7K	✓
3.3Cross-Embodiment Deployment

Cross-embodiment deployment aims to transfer policies learned from robot-free UMI demonstrations to robots with different kinematic structures and sensing configurations. The key idea is to separate an embodiment-agnostic task representation from embodiment-specific execution. Rather than predicting joint-space commands tied to a particular robot, UMI-style policies operate in an end-effector-centric action space, typically producing relative six-degree-of-freedom trajectories and gripper commands. During deployment, the predicted relative trajectory is composed with the robot’s current end-effector pose and transformed into the robot base frame. The resulting target poses are then converted into executable joint commands through inverse kinematics, motion planning, or a low-level Cartesian controller.

This deployment strategy has been extended to robots with increasingly different morphologies. LEGATO [303] maps demonstrations into a motion-invariant task space and retargets the predicted gripper motion to different fixed-base and mobile manipulators through whole-body inverse kinematics. For dexterous manipulation, however, transferring only the wrist trajectory is insufficient because human and robot hands differ substantially in finger morphology, degrees of freedom, and visual appearance. DexUMI [384] addresses these gaps through a wearable exoskeleton that constrains human motions toward robot-feasible hand configurations, together with visual inpainting that replaces the human hand with the target robot hand in demonstration images. Other systems improve deployment consistency by using modular cameras or tactile sensors that can be shared between the collection interface and the robot [191, 368]. These approaches progressively extend UMI data from gripper-level transfer toward cross-embodiment deployment across diverse arms, mobile manipulators, and dexterous hands.

3.4Advantages and Limitations

The primary advantages of UMI data are its scalability, environmental diversity, and reduced dependence on a specific robot embodiment. In particular, its portable and robot-free collection setup enables in-the-wild data acquisition in diverse everyday environments. Human operators can collect smooth and natural manipulation demonstrations across diverse real-world settings, while end-effector-centric action representations facilitate reuse across different robot platforms. Recent extensions incorporating bimanual motion, dexterous hands, and tactile sensing further broaden the range of behaviors captured by UMI-style systems.

However, UMI data does not fully eliminate the embodiment gap. Differences in camera viewpoints, end-effector geometry, kinematic constraints, dynamics, and contact conditions may lead to discrepancies between demonstration and robot execution. Its quality also relies on accurate pose tracking and sensor synchronization, which may degrade under occlusion, rapid motion, or visually challenging environments [362]. Moreover, UMI typically lacks robot-specific proprioception and actuator dynamics. Therefore, it is best regarded as a scalable complement to embodiment-specific real-robot data rather than a complete replacement.

Takeaway
UMI
UMI-style data bridges human demonstration and robot execution through portable, robot-free interfaces, providing explicit end-effector action structure at far lower cost than teleoperation. Yet its reliability is limited by fragile visual tracking and calibration, the difficulty of validating trajectory quality without robot execution, and the lack of force feedback for contact-rich tasks.
4Egocentric and Exocentric Data
4.1Data Overview

Egocentric and Exocentric data occupy the middle layer of the embodied data pyramid, offering more authentic real-world observations and physical interactions than simulation data, but less direct alignment with robot execution than UMI data. Most egocentric datasets rely on a head- or body-worn camera that follows the wearer and keeps the ongoing activity in view. However, this moving viewpoint can leave critical aspects of the interaction unobserved: the hands may occlude contact regions, while much of the body and surrounding scene may remain outside the frame. Synchronized exocentric cameras are therefore often added to provide complementary views and a more complete record. Such data is readily scalable and offers high diversity, reusability, and physical fidelity, but its quality is often limited by the constraints of single-camera capture.

Beyond RGB video, egocentric datasets can be enriched through three complementary routes: multimodal sensing during capture, post-processing of the recorded streams, and conversion into robot-oriented representations. During capture, additional hardware provides complementary signals: H2O [181] records synchronized RGB-D views. HoloAssist [356] combines RGB and depth with head tracking, hand tracking, gaze, and audio. EgoEMG [378] records muscle activity, while FEEL [85] measures applied force and EgoTouch [449] provides tactile pressure.

Post-processing further adds semantic and geometric structure. In EPIC-KITCHENS [72, 73], narration and manual annotation produce action segments and verb-noun labels. HOT3D [11] uses calibrated motion capture to obtain hand and object poses. EgoDex [138] combines headset tracking with SLAM to recover camera and hand trajectories.

Finally, obtaining robot-oriented representations requires mapping human motion into robot-compatible action spaces: EgoMimic [171] aligns tracked human hand trajectories with robot demonstrations, whereas EgoVLA [391] uses inverse kinematics and retargeting to map human wrist and hand poses to robot actions. Figure 6 illustrates the capture infrastructure and supervision types. Table 3 summarizes representative egocentric and exocentric datasets and their key characteristics.

Table 3:Statistics and key properties of egocentric and exocentric datasets. Modalities include only Semantic, Gaze, and Interaction annotations reported as part of each dataset; Interaction includes contact, force, tactile, and hand-object annotations. “Cam. Calib.”, “Hand Pose”, and “3D” denote provided camera calibration parameters, articulated frame-level hand poses, and metric 3D supervision, respectively.
Dataset
 	
Year
	
View
	
Hours/Clips
	
Modality
	
Cam. Calib.
	
Hand Pose
	
3D


EgoHands [10]
 	
2015
	
Ego
	
-/48
	
Semantic
	
✗
	
✗
	
✗


KrishnaCam [322]
 	
2016
	
Ego
	
70.2/460
	
-
	
✗
	
✗
	
✗


FPHA [110]
 	
2017
	
Ego
	
-/1,175
	
Semantic
	
✓
	
✓
	
✓


Charades-Ego [319]
 	
2018
	
Ego+Exo
	
68.8/7,860
	
Semantic
	
✗
	
✗
	
✗


EGTEA Gaze+ [206]
 	
2018
	
Ego
	
28/86
	
Semantic; Gaze
	
✗
	
✗
	
✗


EPIC-KITCHENS v1 [72]
 	
2018
	
Ego
	
55/432
	
Semantic
	
✗
	
✗
	
✗


EPIC-KITCHENS-100 [73]
 	
2020
	
Ego
	
100/700
	
Semantic; Interaction
	
✗
	
✗
	
✗


Ego4D [118]
 	
2021
	
Ego
	
3,670/-
	
Semantic; Gaze; Interaction
	
✗
	
✗
	
✓


H2O [181]
 	
2021
	
Ego+Exo
	
-/-
	
Semantic; Interaction
	
✓
	
✓
	
✓


Assembly101 [302]
 	
2022
	
Ego+Exo
	
513/-
	
Semantic
	
✓
	
✓
	
✓


HOI4D [227]
 	
2022
	
Ego
	
-/4,000
	
Semantic; Interaction
	
✓
	
✓
	
✓


ARCTIC [93]
 	
2023
	
Ego+Exo
	
-/339
	
Interaction
	
✓
	
✓
	
✓


AssemblyHands [270]
 	
2023
	
Ego+Exo
	
-/-
	
-
	
✓
	
✓
	
✓


CaptainCook4D [272]
 	
2023
	
Ego
	
94.5/384
	
Semantic; Interaction
	
✗
	
✗
	
✓


Ego-Exo4D [119]
 	
2023
	
Ego+Exo
	
1,286/5,035
	
Semantic; Gaze
	
✓
	
✓
	
✓


ENIGMA-51 [287]
 	
2023
	
Ego
	
22/51
	
Semantic; Interaction
	
✗
	
✓
	
✗


HoloAssist [356]
 	
2023
	
Ego
	
166/2,221
	
Semantic; Gaze
	
✓
	
✓
	
✓


EgoExo-Fitness [207]
 	
2024
	
Ego+Exo
	
32/1,276
	
Semantic
	
✗
	
✗
	
✗


EgoExoLearn [148]
 	
2024
	
Ego+Exo
	
120/747
	
Semantic; Gaze
	
✗
	
✗
	
✗


EgoSurgery [104]
 	
2024
	
Ego
	
15/21
	
Semantic; Gaze
	
✗
	
✗
	
✗


HOT3D [11]
 	
2024
	
Ego
	
13.9/425
	
Gaze; Interaction
	
✓
	
✓
	
✓


IndustReal [299]
 	
2024
	
Ego
	
5.8/84
	
Semantic; Gaze
	
✗
	
✓
	
✓


EgoDex [138]
 	
2025
	
Ego
	
829/338,000
	
Semantic
	
✓
	
✓
	
✓


Egocentric-100K [1]
 	
2025
	
Ego
	
100,405/2.01M
	
-
	
✓
	
✗
	
✗


EgoLife [390]
 	
2025
	
Ego+Exo
	
266/-
	
Semantic; Gaze
	
✗
	
✗
	
✓


EgoMe [282]
 	
2025
	
Ego+Exo
	
82.8/15,804
	
Semantic; Gaze
	
✗
	
✗
	
✗


HD-EPIC [274]
 	
2025
	
Ego
	
41.3/156
	
Semantic; Gaze; Interaction
	
✓
	
✓
	
✓


IndEgo [40]
 	
2025
	
Ego+Exo
	
293.9/-
	
Semantic; Gaze
	
✓
	
✓
	
✓


OpenEgo [157]
 	
2025
	
Ego
	
1,107/-
	
Semantic; Interaction
	
✓
	
✓
	
✓


TASTE-Rob [436]
 	
2025
	
Ego
	
-/100,856
	
Semantic; Interaction
	
✗
	
✗
	
✗


ChildLens [327]
 	
2026
	
Ego
	
108.58/354
	
Semantic
	
✗
	
✗
	
✗


Ego-1K [183]
 	
2026
	
Ego
	
-/956
	
-
	
✓
	
✗
	
✗


EgoLive [205]
 	
2026
	
Ego
	
1,680/-
	
Semantic; Interaction
	
✓
	
✓
	
✓


EgoMAGIC [343]
 	
2026
	
Ego
	
-/3,355
	
Semantic; Interaction
	
✗
	
✗
	
✗


EgoVerse [278]
 	
2026
	
Ego
	
1,362/-
	
Semantic; Interaction
	
✗
	
✓
	
✓


HumanNet [81]
 	
2026
	
Ego+Exo
	
967,000/-
	
Semantic
	
✗
	
✓
	
✓


Open-AoE [209]
 	
2026
	
Ego
	
2,000/-
	
Semantic; Interaction
	
✓
	
✓
	
✓


Xperience-10M [296]
 	
2026
	
Ego
	
10,000/-
	
Semantic; Interaction
	
✓
	
✓
	
✓
Figure 6:Overview of capture infrastructure and supervision representations in egocentric and exocentric data. Wearable and external cameras, motion trackers, and auxiliary sensors record complementary raw signals, which are subsequently annotated, reconstructed, or aligned to form semantic, geometric, multimodal, and robot-oriented supervision.
4.2Capture Infrastructure

Egocentric and exocentric capture systems combine wearable and environmental devices to measure appearance, geometry, motion, attention, and physical interaction. We organize the hardware by the physical quantity it observes: visual sensors record image and depth streams, motion-tracking systems measure device or body kinematics, and auxiliary sensors expose attention, muscle activation, and contact. Across these categories, the main trade-offs concern spatial coverage, temporal resolution, mobility, intrusiveness, accuracy, and calibration requirements.

4.2.1Visual Sensing

Wearable Cameras. Monocular RGB capture uses a single color camera, typically mounted on the head or chest, to approximate the wearer’s visual stream. Such cameras are lightweight, inexpensive, and suitable for long-duration collection, as demonstrated by EPIC-KITCHENS [72, 73] and Ego4D [118]; however, they do not measure metric depth directly, and rapid head motion can introduce blur while the hands can occlude manipulated objects.

Stereo rigs instead use two synchronized cameras with a known baseline to estimate depth from binocular disparity. They avoid active illumination but require accurate calibration and can fail in textureless or occluded regions. HOT3D [11], for example, uses the onboard multi-camera configurations of Project Aria and Quest 3 to observe hands and objects from slightly different viewpoints. RGB-D cameras pair color images with per-pixel depth, providing more direct 3D measurements but with device-dependent range and sensitivity to reflective surfaces or strong ambient light. H2O [181] synchronizes egocentric and exocentric RGB-D views, while HoloAssist [356] records RGB and depth with a HoloLens 2.

Event cameras take a different approach: rather than producing conventional frames at a fixed rate, each pixel asynchronously reports changes in brightness. Their high temporal resolution reduces motion blur during fast hand motion, but produces sparse data that requires specialized processing; EgoEV-HandPose [351] explores this setting with stereo event cameras.

Exocentric Camera Systems. Exocentric systems place one or more cameras in the environment to observe the wearer from fixed third-person viewpoints. They reveal body, hand, and scene context that may be occluded or outside the wearable camera’s field of view, and synchronized views support triangulation and multi-view reconstruction. Assembly101 [302] combines wearable and external cameras for fine-grained assembly activities, while Ego-Exo4D [119] pairs Project Aria glasses with multiple surrounding GoPro cameras. The additional coverage comes at the cost of environmental setup, cross-camera calibration and synchronization, higher storage demand, and reduced collection mobility.

4.2.2Motion and Pose Tracking

Inertial Body Tracking. An inertial measurement unit (IMU) combines accelerometers and gyroscopes, and sometimes a magnetometer, to measure linear acceleration and angular velocity. IMUs are compact and remain available under visual occlusion, but orientation and especially position estimates accumulate drift when the measurements are integrated over time. Ego-Exo4D [119] records the IMUs built into Project Aria, while WEAR [22] distributes body-worn IMUs to capture outdoor activity dynamics. Full-body inertial motion-capture systems extend this idea by placing multiple synchronized IMUs across the body and fitting their measurements to an articulated skeleton. Nymeria [245] combines Project Aria and wrist-worn miniAria devices with an Xsens suit to capture head, hand, and body motion without an external camera array, although the suit increases fitting and body-specific calibration effort.

Hand-Worn Tracking Devices. Hand-worn trackers place magnetic, inertial, flex, or optical sensors on the fingers, hand, or wrist to obtain pose-related measurements such as sensor orientation, finger bending, fingertip position, and wrist motion. These measurements are converted into articulated hand pose through device calibration, sensor fusion, or kinematic fitting; they should therefore be regarded as sensor-based pose capture rather than unprocessed pose labels. FPHA [110] attaches six magnetic sensors to the fingertips and recovers articulated hand motion even when the fingers are visually occluded, although nearby metal and electromagnetic interference can distort the measurements. DexGloveHOI [178] instead synchronizes egocentric images with glove-mounted 6-DoF IMUs and motion-capture reference poses for vision-inertial 3D hand tracking. Compared with image-only estimation, instrumented wearables generally provide more continuous and accurate 3D hand motion under severe hand-object occlusion. Their scalability is nevertheless limited by hand-shape calibration, sensor displacement, magnetic interference or inertial drift, and possible interference with natural grasping.

Optical and Headset-Based Tracking. Optical motion capture uses multiple external cameras to triangulate visible markers, yielding accurate 3D trajectories in a calibrated workspace. HOT3D [11] places small markers on the hands and objects to obtain pose ground truth, trading accuracy for controlled setup and possible marker occlusion. AR/VR headsets provide a more portable alternative through device-native inside-out tracking, which fuses onboard cameras and inertial sensing to estimate the headset and hands. EgoDex [138] uses Apple Vision Pro tracking, and HoloAssist [356] uses HoloLens 2. These systems reduce external infrastructure but depend on proprietary tracking pipelines and can fail under fast motion, occlusion, or limited field of view.

4.2.3Auxiliary and Interaction Sensing

Gaze and Audio. Eye trackers use inward-facing cameras and illumination to estimate gaze direction, providing a cue to visual attention but requiring wearer-specific calibration and stable headset placement. Microphones record speech, environmental sounds, and acoustic contact events; multi-channel arrays additionally capture spatial information and support more robust speech recording, but remain sensitive to background noise and privacy constraints. Ego-Exo4D [119] records eye gaze and seven-channel audio with Project Aria, while HoloAssist [356] synchronizes gaze and audio with its visual and tracking streams.

Electromyography. Electromyography (EMG) uses skin-contact electrodes to measure the electrical activity produced by muscle activation rather than hand pose itself. EgoEMG [378] records bilateral wrist EMG and IMU at higher rates than video, exposing motor activity even when the hands are visually occluded. The signal is informative about muscle recruitment, but depends strongly on electrode placement, skin contact, participant physiology, and session-specific calibration.

Force and Tactile Sensing. Force and tactile devices measure physical interaction more directly. A force sensor reports the magnitude or direction of load at one or several locations, whereas a tactile or pressure array resolves how contact is distributed over an area. FEEL [85] places piezoresistive force sensors on the fingertips and palm, Touch and Go [388] records visual probing with a tactile sensor, and EgoTouch [449] combines continuous wearable pressure measurements with head- and wrist-mounted RGB and bimanual hand pose. These wearables provide contact signals unavailable from vision alone, but limited sensing area, sensor fitting, per-user calibration, cabling, and interference with natural grasping can restrict collection scale and fidelity.

4.3Supervision Construction

Egocentric supervision may be recorded by sensors, added by human annotators, or recovered through post-processing. We group it into four complementary categories according to the information exposed to learning: semantic supervision describes task content and structure; geometric supervision describes spatial state and motion; multimodal supervision supplies attention, dynamics, and contact cues beyond RGB; and robot-oriented supervision expresses human demonstrations in forms intended for robot policy learning. Their construction combines annotation, calibration, synchronization, reconstruction, and cross-embodiment alignment, with different costs and sources of error.

4.3.1Semantic Supervision

Action and Language Annotations. Action and language annotations identify what activity occurs and when it occurs. Common forms include narrations, object references, verb-noun actions, action classes, and temporal segments. Participant narrations are transcribed and aligned with video, while action classes and boundaries are assigned manually or with model assistance followed by human verification. EPIC-KITCHENS [72, 73], for example, turns free-form participant narrations into temporally localized verb-noun actions. Ego4D [118] combines large-scale narrations with annotations tailored to individual benchmarks, and Assembly101 [302] provides fine-grained labels for assembly actions. These annotations support recognition, retrieval, and temporal grounding, but do not specify the continuous motion or contact needed to perform an action.

Procedural and State Annotations. Procedural annotations describe how individual actions compose into a task. They include ordered steps, task progress, execution errors, and object or environment state changes, and are typically constructed by aligning videos with task scripts and verifying deviations from the expected procedure. Assembly101 [302] organizes fine-grained actions within assembly procedures, whereas CaptainCook4D [272] labels cooking steps together with errors and object states. This supervision exposes long-horizon dependencies and task outcomes that isolated action labels miss, although its vocabulary is often specific to a particular procedure or domain.

4.3.2Geometric Supervision

Camera and Scene Geometry. Camera and scene supervision establishes the spatial reference in which an interaction occurs. It includes camera intrinsics and extrinsics, depth maps, point clouds, camera poses and trajectories, scene reconstructions, and point tracks. Depth may be measured directly, as in the RGB-D observations of H2O [181]; other quantities require calibration, visual-inertial SLAM, multi-view reconstruction, or tracking. HOT3D [11] provides calibrated camera poses, while EgoDex [138] combines headset tracking with SLAM to recover the moving egocentric camera trajectory. Synchronized egocentric and external views, such as those in Ego-Exo4D [119], further support reconstruction beyond the wearable camera’s field of view. The resulting geometry is sensitive to calibration drift, motion blur, and the choice of coordinate frame.

Hand and Body Pose. Human pose supervision is released at several levels of detail. Hands may be encoded as 2D or 3D keypoints, wrist and fingertip poses, joint rotations and trajectories, or MANO pose and shape parameters, while body supervision ranges from head and upper-body trajectories to full articulated skeletons or parametric body models. Because these representations describe the output rather than how it was obtained, pose labels can be organized by three main construction routes.

(1) Annotation-assisted Geometric Reconstruction. Annotators mark 2D hand or body keypoints in one or more RGB views. With registered depth, visible keypoints can be back-projected into 3D; with synchronized and calibrated cameras, corresponding keypoints can instead be triangulated across views. AssemblyHands [270] illustrates a scalable variant: it starts from a manually annotated subset and trains a multi-view annotation model to extend accurate 3D hand poses to a much larger collection. This route retains direct human quality control, but manual labeling is costly and its 3D accuracy depends on depth quality, cross-view correspondence, and camera calibration.

(2) Model-based Pose Prediction and Parametric Fitting. Learned vision models generate pose labels directly from recorded RGB or RGB-D observations. Keypoint detectors predict 2D or 3D joints, while model-recovery networks may regress MANO pose and shape parameters or analogous body-model parameters. These initial predictions can be used as pseudo-labels or further refined with depth, silhouettes, multi-view consistency, temporal constraints, and parametric-model fitting. H2O [181], for example, uses OpenPose predictions to initialize its hand annotations and then optimizes a MANO hand model with synchronized multi-view RGB-D observations. This route scales more readily to unlabeled recordings, but the resulting supervision inherits model errors and biases caused by occlusion, motion blur, and domain shift.

(3) Sensor-based Pose Capture. The aforementioned hand-worn, optical, headset-based, and inertial systems provide pose-related measurements independently of image-only inference. Their signals are converted into articulated pose through calibration, sensor fusion, or kinematic fitting. FPHA [110] uses magnetic fingertip sensors, HOT3D [11] converts professional optical motion capture into UmeTrack and MANO hand annotations, EgoDex [138] obtains upper-body and hand trajectories from headset tracking, and Nymeria [245] combines wearable devices with an inertial suit for full-body motion. Instrumented capture generally yields more continuous and accurate 3D motion under visual occlusion, but increases calibration effort and participant burden and remains dependent on device-specific tracking models. Across all three routes, pose supervision differs in joint definitions, coordinate frames, temporal stability, and confidence, which should be reported explicitly when datasets are compared or combined.

Object and Interaction Geometry. Object-centered supervision describes how manipulated objects move relative to the hands and scene. Typical forms include object 6-DoF pose, object trajectory, hand-object relative pose, and coupled hand-object trajectories. They are obtained through CAD-model registration, multi-view reconstruction, marker-based motion capture, pose estimation, or temporal tracking. H2O [181] jointly provides hand and object poses in RGB-D sequences, HOI4D [227] annotates the spatial evolution of hand-object interactions, and HOT3D [11] releases calibrated hand and object poses. Unlike semantic object labels, these annotations expose manipulation geometry directly, although occlusion and object symmetry can make pose recovery ambiguous.

4.3.3Multimodal Supervision

Gaze and Audio. Gaze and audio associate visual observations with attention and acoustic context. Raw gaze measurements must be calibrated and projected into fixation points, attended regions, or gaze trajectories, whereas audio must be synchronized with video and optionally aligned with actions or state changes. EGTEA Gaze+ [206] provides gaze aligned with kitchen activities, and Ego-Exo4D [119] records synchronized gaze and multi-channel audio. HoloAssist [356] similarly pairs gaze and audio with instructional activities. These cues help localize relevant objects and events, but gaze calibration errors and ambient sound can weaken their semantic correspondence.

Inertial and Physiological Signals. Inertial and physiological supervision captures motion dynamics and muscle activity that are not explicit in images. IMU streams provide acceleration and angular velocity, while EMG measures muscle activation; their use requires synchronization, filtering, normalization, and segmentation at a sampling rate often much higher than video. WEAR [22] combines egocentric video with body-worn IMUs for outdoor activities, Ego-Exo4D [119] includes Project Aria inertial measurements, and EgoEMG [378] pairs bilateral EMG and IMU streams with hand motion. These signals remain available under visual occlusion, but are sensitive to sensor placement, drift, and participant physiology.

Force and Tactile Signals. Force and tactile supervision provides direct evidence of physical contact. Depending on the sensor, it may be represented as scalar or vector force, binary contact events, fingertip and palm pressure, or spatial tactile maps. FEEL [85] synchronizes fingertip and palm force measurements with egocentric observations, Touch and Go [388] records visual exploration together with tactile sensing, and EgoTouch [449] pairs wearable tactile pressure with head- and wrist-mounted RGB and bimanual hand pose. Such measurements distinguish visually similar motions with different contact outcomes, although wearable sensing does not necessarily reproduce a robot’s contact geometry or force response.

4.3.4Robot-Oriented Supervision

Robot-Compatible Action Supervision. Robot-compatible supervision maps human motion into a target action space, such as an end-effector pose, robot joint configuration, gripper command, or retargeted dexterous-hand action. It is constructed from tracked wrist, fingertip, or MANO motion through coordinate transformation, inverse kinematics, morphology-aware retargeting, and feasibility filtering. EgoVLA [391], for example, predicts wrist pose and MANO parameters and converts them to robot commands through inverse kinematics and hand retargeting. EgoScale [443] supervises human pretraining with relative wrist motion and retargeted high-DoF hand actions, while MotionTrans [411] transforms human VR motion for robot policy learning. These targets are often generated by a method rather than released as native annotations of the source dataset, and geometric plausibility does not guarantee dynamically feasible or contact-consistent execution.

Human-Robot Alignment Supervision. Human-robot alignment supervision relates demonstrations across embodiments without requiring identical raw action spaces. It may take the form of task- or scene-matched human and robot demonstrations, temporally aligned trajectories, or a shared representation to which both embodiments are mapped. EgoMimic [171] normalizes human hand and robot demonstrations for joint policy training, while EgoScale [443] uses aligned human-robot play in matched tasks and scenes as an intermediate training stage. H-RDT [19] instead combines human hand-pose pretraining with robot-specific action adapters, and METIS [101] integrates multiple human and robot sources under a consistent action space. These examples provide different strengths of correspondence rather than uniform frame-level pairs, so the type and precision of alignment should be stated explicitly.

4.4Advantages and Limitations

Egocentric and exocentric data capture natural human demonstrations at scale without requiring execution on a target robot. Wearable systems can be deployed across different participants, environments, objects, and tasks, while first-person observations emphasize hand-object interaction and long-horizon activity structure. Unlike fixed-base robot collection, human operators can move freely through the environment and manipulate objects directly with dexterous hands. This mobility enables broader spatial coverage and more diverse hand trajectories, including variations in approach direction, motion path, and interaction height. The EgoVerse example in Figure 4 illustrates this potential: its human-hand interaction keyframes span a broad three-dimensional workspace and exhibit substantial spatial variation.

However, egocentric data are not equivalent to robot experience. First-person recordings suffer from partial observability, hand-object occlusion, rapid camera motion, motion blur, and device-dependent viewpoints. Dense geometric and multimodal supervision requires additional hardware, calibration, synchronization, and reconstruction, and is not uniformly available at scale. More fundamentally, human recordings lack robot proprioception, actuator dynamics, and directly executable actions. Differences in morphology, kinematics, sensing, and contact response further prevent direct trajectory transfer. Their use in robot learning therefore requires reliable state reconstruction, embodiment-aware representations, and human-to-robot alignment or retargeting.

Takeaway
Egocentric
Egocentric and exocentric data occupy the middle layer of the embodied data pyramid, providing a scalable route to collecting physically grounded interaction trajectories in diverse in-the-wild environments. By recording human agents interacting with the physical world, these data capture natural manipulation behaviors and high-degree-of-freedom hand motions with substantial potential for reuse across robotic embodiments. Their collection, however, involves a trade-off between scalability and annotation fidelity. Device-free recordings are unobtrusive and easy to scale, but typically lack accurate hand-pose and interaction labels, requiring downstream pipelines to rely on model-based estimation or learned visual retargeting. Wearable sensing systems can provide more direct and precise measurements, but remain constrained by hardware cost, durability, user burden, and portability, particularly for large-scale outdoor collection. Beyond this acquisition trade-off, their utility for robot learning is further limited by unresolved alignment between human and robot embodiments.
5Simulation Data
Table 4:Simulation Benchmarks. Simulation benchmarks, summarized by embodiment configuration, tactile sensing, object deformability, dexterous manipulation, and mobile manipulation. In the “Embod.” column, ‘S’, ‘D’, and ‘H’ denote single-arm, dual-arm, and humanoid embodiments, respectively.
Benchmark
 	
Time
	
Embod.
	
Tactile
	
Deform.
	
Dex.
	
Mobile


Adroit [288]
 	
2017
	
S
	
✗
	
Rigid
	
✓
	
✗


Franka Kitchen [122]
 	
2019
	
S
	
✗
	
Rigid
	
✗
	
✗


Meta-World [408]
 	
2019
	
S
	
✗
	
Rigid
	
✗
	
✗


RLBench [153]
 	
2019
	
S
	
✗
	
Rigid
	
✗
	
✗


PlasticineLab [149]
 	
2021
	
S
	
✗
	
Soft
	
✗
	
✗


RoboMimic [250]
 	
2021
	
S/D
	
✗
	
Rigid
	
✗
	
✗


CALVIN [254]
 	
2021
	
S
	
✗
	
Rigid
	
✗
	
✗


Tactile Gym 2.0 [214]
 	
2022
	
S
	
✓
	
Rigid
	
✗
	
✗


Bi-DexHands [56]
 	
2022
	
D
	
✗
	
Rigid
	
✓
	
✗


BEHAVIOR-1K [189]
 	
2022
	
S
	
✗
	
Soft
	
✗
	
✓


VIMA-Bench [164]
 	
2022
	
S
	
✗
	
Rigid
	
✗
	
✗


DaXBench [48]
 	
2023
	
S
	
✗
	
Soft
	
✗
	
✗


DexArt [12]
 	
2023
	
S
	
✗
	
Rigid
	
✓
	
✗


RoboHive [180]
 	
2023
	
S
	
✗
	
Rigid
	
✓
	
✗


ARNOLD [115]
 	
2023
	
S
	
✗
	
Rigid
	
✗
	
✗


FurnitureBench [134]
 	
2023
	
S
	
✗
	
Rigid
	
✗
	
✗


ManiSkill2 [121]
 	
2023
	
S/D
	
✗
	
Soft
	
✗
	
✓


LIBERO [217]
 	
2023
	
S
	
✗
	
Rigid
	
✗
	
✗


ManiSkill-HAB [315]
 	
2024
	
S
	
✗
	
Rigid
	
✗
	
✓


TacSL [2]
 	
2024
	
S
	
✓
	
Soft
	
✗
	
✗


ManiSkill-ViTac [190]
 	
2024
	
S
	
✓
	
Soft
	
✗
	
✗


GarmentLab [233]
 	
2024
	
S/D
	
✗
	
Soft
	
✓
	
✓


HumanoidBench [305]
 	
2024
	
H
	
✗
	
Rigid
	
✓
	
✓


BiGym [61]
 	
2024
	
D
	
✗
	
Rigid
	
✗
	
✓


The Colosseum [277]
 	
2024
	
S
	
✗
	
Rigid
	
✗
	
✗


VLABench [427]
 	
2024
	
S
	
✗
	
Rigid
	
✗
	
✗


SIMPLER-Env [202]
 	
2024
	
S
	
✗
	
Rigid
	
✗
	
✗


ManiSkill3 [332]
 	
2024
	
S/H
	
✗
	
Soft
	
✓
	
✓


RoboTwin [259]
 	
2024
	
S/D
	
✗
	
Rigid
	
✗
	
✗


RoboCasa [262]
 	
2024
	
S/H
	
✗
	
Rigid
	
✗
	
✓


MemoryBench [96]
 	
2025
	
S
	
✗
	
Rigid
	
✗
	
✗


RoboCerebra [129]
 	
2025
	
S
	
✗
	
Rigid
	
✗
	
✗


GenManip [107]
 	
2025
	
S
	
✗
	
Rigid
	
✗
	
✗


RoboTwin 2.0 [49]
 	
2025
	
D
	
✗
	
Rigid
	
✗
	
✗


RoboEval [358]
 	
2025
	
D
	
✗
	
Rigid
	
✗
	
✗


REALM [301]
 	
2025
	
S
	
✗
	
Rigid
	
✗
	
✗


RobotArena
∞
 [156]
 	
2025
	
S
	
✗
	
Rigid
	
✗
	
✗


DuoBench [167]
 	
2026
	
D
	
✗
	
Rigid
	
✗
	
✗


BiCoord [273]
 	
2026
	
D
	
✗
	
Rigid
	
✗
	
✗


MIKASA-Robo-VLA [60]
 	
2026
	
S
	
✗
	
Rigid
	
✗
	
✗


RoboMemArena [185]
 	
2026
	
S
	
✗
	
Rigid
	
✗
	
✗


RoboMME [71]
 	
2026
	
S
	
✗
	
Rigid
	
✗
	
✗


RMBench [51]
 	
2026
	
D
	
✗
	
Rigid
	
✗
	
✗


UniVTAC [43]
 	
2026
	
S
	
✓
	
Soft
	
✗
	
✗


RoboCasa365 [263]
 	
2026
	
S/H
	
✗
	
Rigid
	
✗
	
✓


RoboDojo [50]
 	
2026
	
D
	
✗
	
Rigid
	
✗
	
✗
Figure 7:Overview of simulation infrastructure and data collection pipeline. The left panel summarizes the core components required for constructing simulation environments, including robotic embodiments and sensors, scenes, object assets, and simulation engines. The right panel categorizes representative trajectory collection methods into human teleoperation, trajectory replay with scene and object augmentation, rule-based execution based on expert-defined task logic and motion planning, and autonomous generation through trained policies or LLM-based planning.
5.1Data Overview

Within the embodied data pyramid, simulation data serves as a scalable and controllable complement to real-world robot data. While real-robot data is necessary for capturing embodiment-specific dynamics, sensor noise, contact effects, and deployment constraints, its collection is often limited by cost, safety, hardware availability, and annotation difficulty. Simulation alleviates part of this bottleneck by enabling repeatable interaction conditions, automatic access to privileged states and labels, and low-cost trajectory generation. Recent datasets further attempt to reduce the sim-to-real gap through digital twins [366, 349], domain randomization [49, 259, 50], and paired sim-to-real evaluation [202]. As a result, simulation has become not only a tool for benchmark construction and evaluation, but also an important data layer for training general-purpose robot policies. Nevertheless, compared with the three higher layers of the embodied data pyramid, simulation data generally remains more limited in diversity and physical fidelity.

As summarized in Tables 4 and 5, simulated robot data has expanded in both scale and scope. Early benchmarks mainly focused on constrained rigid-body manipulation, such as table-top control [408], language-conditioned manipulation [153], and large-scale grasping [92]. Later datasets introduced longer-horizon interaction [254], compositional task settings [217], and household manipulation [262]. More recent resources further extend simulation toward bimanual manipulation [49, 259, 50], humanoid manipulation [26], articulated-object interaction [121], deformable-object and fluid manipulation [341, 233, 360], and contact-rich manipulation with tactile sensing [43]. In parallel, data generation has become increasingly automated, moving from scripted tasks and teleoperation toward pipelines based on motion planning and RL experts [332], demonstration augmentation [251], digital-twin reconstruction [366, 349], and LLM-assisted task or environment synthesis [341, 53, 144, 359].

5.2Simulation Infrastructure and Asset Ecosystems

As shown in Figure 7, a simulated robot-learning ecosystem can be viewed as a composition of three coupled infrastructure components: embodiment-sensor systems, object and scene assets, and physics and rendering backends. The embodiment-sensor system defines not only the robot morphology, kinematics, actuation model, and controller interface, but also the sensing channels physically associated with the robot body, such as proprioception, wrist or head-mounted vision, force sensing, and tactile arrays. Object and scene assets provide the geometric, semantic, and affordance structure of the environment, while physics and rendering backends determine the contact dynamics, visual appearance, and execution fidelity of the generated data.

5.2.1Robotic Embodiments and Sensors

Constructing simulated manipulation data first requires importing an embodiment-sensor system into the simulator, where the robot asset is instantiated as a controllable articulated model and its sensors are configured to acquire observations during interaction. The simulator parses robot descriptions such as URDF, MJCF, SDF, or USD to specify kinematics, joint limits, inertial properties, collision and visual geometries, actuators, and sensor attachments [439, 53, 341], before placing the system in a scene with defined initial pose, joint state, objects, lighting, gravity, and solver settings.

Across simulated robotics datasets, stationary single-arm manipulators are the most common embodiment, as in Franka Kitchen [122], Meta-World [408], RLBench [153], RoboMimic [250], CALVIN [254], Simpler-Env [202], RMBench [51], UniVTAC [43] and LIBERO [217]. Other benchmarks cover dexterous hands, including Adroit [288], Bi-DexHands [56], and DexArt [12]; mobile manipulation, including BEHAVIOR-1K [189], ManiSkill2 [121], ManiSkill-HAB [315], and RoboCasa [262]; and dual-arm or bimanual systems, including RoboTwin [259], BiGym [61], RMBench [51], DuoBench [167], BiCoord [273], and RoboDojo [50]. ManiSkill3 [332], RoboTwin 2.0 [49], RoboMIND-Sim [366] and RoboMME [71] further support multiple embodiment classes, while HumanoidBench [305] targets humanoid control.

5.2.2Object and Scene Assets

The asset ecosystem not only includes robot models but also manipulable objects, materials, and scene layouts. Object-level assets determine the category coverage, geometric variation, texture realism, and affordance structure of simulated manipulation data. CAD-based and web-scale resources such as ShapeNet [39], Objaverse [78], and Objaverse-XL [77] provide broad object-shape diversity. Scanned or product-grounded resources such as Google Scanned Objects [87], Amazon Berkeley Objects [65], and OmniObject3D [375] improve geometric and texture realism by grounding assets in real objects or product data.

However, robot-learning assets must be more than visually plausible meshes: they also require simulator-compatible collision geometry, consistent scale, stable poses, mass and inertia parameters, material properties, and, when applicable, grasp affordances or articulation annotations. PartNet-Mobility [379] provides articulated objects with mobility annotations for part-level interaction. ObjectFolder 2.0 [108] introduces multisensory object representations that connect visual, acoustic, and tactile properties. PhysXNet [33] further introduces the first physics-aware 3D dataset, extending 3D representations beyond geometry and appearance to include physical properties. PhysX-Mobility [34] then provides plug-and-play simulation-ready articulated assets, while PhysXVerse [35] generalizes this paradigm to a broader and more diverse collection of simulation-ready 3D assets.

Beyond curated or reconstructed collections, generative modeling offers a scalable route to expanding the asset pool: ManiTwin [349] contributes a large-scale library of over 100K AIGC-generated rigid-body objects, each equipped with manipulation-oriented annotations for automated data synthesis. This coupling is essential, because asset scaling alone is insufficient—without operable annotations, even abundant assets cannot be converted into usable manipulation data—which positions AIGC-driven generation as a promising direction for jointly scaling asset diversity and downstream data synthesis. At the scene level, ScanNet [68], Replica [326], Matterport3D [38], HM3D [289], HSSD [172], 3D-FRONT [100], and ProcTHOR [76] provide reconstructed, synthetic, or procedurally generated indoor environments for navigation, rearrangement, and household interaction. Thus, the asset ecosystem defines not only what the agent observes, but also what it can physically contact, manipulate, and reason about.

5.2.3Physics and Rendering Backends

The physics and rendering backend determines contact dynamics, joint constraints, numerical stability, parallelization, and visual realism. MuJoCo [342] remains widely used for rigid-body control, contact-rich manipulation, and locomotion because of its efficient dynamics and stable contact modeling. MuJoCo Playground [414] further extends this line toward accelerator-based parallel training. Isaac Sim [268], Isaac Gym [247], and Isaac Lab/Orbit [256] emphasize GPU-parallel simulation, photorealistic rendering, sensor modeling, and synthetic data generation. SAPIEN/PartNet-Mobility [379] focus on articulated-object interaction and scalable manipulation under PhysX-based simulation. PyBullet [66] remains useful for lightweight prototyping and baseline environments. CoppeliaSim [295] and PyRep [152] support flexible task construction, as exemplified by RLBench [153]. Habitat 2.0 [329] and Habitat 3.0 [276] target navigation, rearrangement, and human-robot co-habitation.

Specialized engines further extend the physical regime: SoftGym [213] supports deformable-object manipulation, TDW [105] supports multimodal physical interaction, Brax [99] provides JAX-native accelerator-friendly simulation, Genesis [112] targets unified multi-physics simulation, and Newton [216] aims at differentiable physical simulation. In this sense, the simulator backend constrains the feasible data distribution: rigid-body engines favor contact-rich control and manipulation, rendering-oriented engines support visually diverse synthetic data, and specialized engines enable deformable, tactile, or differentiable robot learning.

Table 5:Large-scale simulation datasets for robot learning, summarized by embodiment configuration, data scale, task coverage, object deformability, dexterous manipulation, and mobile manipulation. In the “Arm” column, ‘S’ and ‘D’ denote single-arm and dual-arm, respectively.
Dataset
 	
Time
	
Arm
	
Embod.
	
Traj./Hours
	
Tasks
	
Deform.
	
Dex.
	
Mobile


DexGraspNet [353]
 	
2022
	
S
	
1
	
1.32M / -
	
Grasp
	
Rigid
	
✓
	
✗


MimicGen [251]
 	
2023
	
S
	
3
	
50K / -
	
16
	
Rigid
	
✗
	
✗


DexMimicGen [166]
 	
2024
	
D
	
2
	
21K / -
	
9
	
Rigid
	
✓
	
✗


DexGraspNet 2.0 [424]
 	
2024
	
S
	
1
	
427M / -
	
Grasp
	
Rigid
	
✓
	
✗


InternData-A1 [341]
 	
2025
	
S/D
	
4
	
637K / 7.4K h
	
70
	
Soft
	
✗
	
✗


InternData-M1 [53]
 	
2025
	
S
	
1
	
244K / -
	
200
	
Rigid
	
✗
	
✗


NVIDIA GR00T-X Sim [267]
 	
2025
	
S/D
	
6+
	
345K / -
	
58+
	
Rigid
	
✓
	
✓


SynGrasp-1B [80]
 	
2025
	
S
	
1
	
10M / -
	
Grasp
	
Rigid
	
✗
	
✗


Dex1B [395]
 	
2025
	
S
	
3
	
1B / -
	
2
	
Rigid
	
✓
	
✗


MolmoB0T [84]
 	
2026
	
S
	
2
	
1.7M / 5.8K h
	
8
	
Artic.
	
✗
	
✓
5.3Benchmarks and Simulation Datasets

Table 4 and Table 5 summarize two complementary components of the simulation-data ecosystem. Table 4 focuses on benchmarks and evaluation environments, characterizing their supported embodiments and interaction settings. In this table, ‘S’, ‘D’, and ‘H’ denote single-arm, dual-arm, and humanoid embodiments, respectively; dexterous-hand and mobile capabilities are reported separately in the “Dex.” and “Mobile” columns. Existing benchmarks cover a broad range of configurations, from single-arm tabletop manipulation [217, 153, 408] to bimanual coordination [50, 49], dexterous manipulation, mobile manipulation [262, 263], and humanoid control [305]. Nevertheless, rigid-object manipulation remains dominant, while tactile sensing and deformable-object interaction are supported by a comparatively limited subset of environments.

Table 5 summarizes large-scale synthetic and simulation-based data resources in terms of data volume, task coverage, embodiment diversity, and supported manipulation settings. These resources range from task-level manipulation trajectories to large-scale grasp and dexterous-hand samples. Recent datasets increasingly incorporate multiple embodiments and substantially larger data volumes, including hundreds of thousands or millions of trajectories and, for grasp-centric resources, substantially larger collections of generated samples. However, most datasets remain concentrated on rigid or articulated-object manipulation, with comparatively limited coverage of deformable interaction and mobile manipulation.

5.4Synthetic Demonstration Generation

Human-Executed Demonstrations. The most direct form of simulation data collection records trajectories produced through human control. This category includes teleoperation, VR-based control, motion-capture retargeting, and play-style interaction in simulated environments [254, 217, 61]. In this setting, the simulator mainly serves as an instrumented recording environment that can provide synchronized observations, robot states, actions, language annotations, object states, and success labels. Compared with fully scripted generation, human-executed demonstrations can better capture adaptive behaviors, recovery strategies, and temporally coherent manipulation patterns, especially in contact-rich or long-horizon tasks where suitable scripted experts are difficult to specify. However, scalability remains limited by episode-level human labor, operator expertise, and the fidelity of the teleoperation interface. For this reason, human demonstrations in simulation are often used either as direct imitation-learning data or as seed trajectories for later replay, augmentation, or synthetic expansion.

Rule-Based Execution. Rule-based execution generates simulation trajectories from manually specified or privileged task-level experts, including heuristic skill compositions [427], programmatic expert policies [164], and waypoint sequences [153]. A task is typically decomposed into stages such as selecting the target object, computing a grasp or contact pose, approaching the target, executing the interaction, and checking task completion. Inverse kinematics, task-and-motion planning, model-predictive control, collision-aware motion planning, trajectory optimization, or task-specific controllers may then convert these stage-level goals into dense joint-space or end-effector actions [121, 332]. This paradigm has been widely used in simulation benchmarks and data-generation environments where privileged states, object poses, articulation states, and collision models are available, including RLBench [153], ManiSkill2 [121], ManiSkill3 [332], VIMA-Bench [164], and VLABench [427].

Rule-based execution improves reproducibility and reduces the marginal cost of collecting additional episodes. It is particularly effective for tasks with clearly specified geometric goals, structured workspaces, reliable resets, and well-defined success conditions. Nevertheless, the resulting behavior distribution is bounded by the expressiveness of the programmatically specified expert. Conventional motion planning primarily addresses geometric execution and does not, by itself, determine task semantics, contact affordances, or recovery strategies. As a result, rule-based datasets may underrepresent ambiguous semantic decisions, non-prehensile interactions, contact-rich adaptation, and the diversity of suboptimal but recoverable behaviors encountered during real deployment.

Trajectory Playback and Demonstration Expansion. Trajectory playback and demonstration expansion start from existing trajectories and reuse them to synthesize additional simulation data. The source trajectories may come from teleoperation, kinesthetic teaching, human video retargeting, motion planning, or previous policy rollouts. In simulation, these trajectories can be replayed, spatially transformed, segmented, stitched, or retargeted under new object configurations, scene layouts, embodiments, and camera conditions. Object-centric augmentation methods such as MimicGen reuse the temporal structure of a small number of demonstrations while adapting them to new object poses and task instances [251]. This paradigm has been extended to bimanual and dexterous manipulation in DexMimicGen [166]. RoboCasa integrates related automated trajectory-generation methods as one component of a broader framework for household and kitchen manipulation [262].

This class occupies an intermediate position between human-executed data and fully scripted generation. Compared with teleoperation, it substantially reduces repeated human effort; compared with hand-designed rule-based experts, it can preserve the temporal structure, motion style, and task strategies present in human demonstrations. However, its coverage remains conditioned on the diversity and quality of the seed trajectories. When object geometry, contact conditions, or task constraints differ substantially from the original demonstrations, open-loop playback may fail, and closed-loop correction, replanning, or filtering becomes necessary. Therefore, demonstration expansion is most effective when the task admits object-centric transformations or reusable skill segments, but remains limited for tasks requiring qualitatively new strategies or long-horizon semantic decisions.

Autonomous Policy and Generative Rollouts. A more automated class generates simulation trajectories through learned policies [398], reinforcement-learning experts, LLM- or VLM-generated task programs, scene-graph solvers, compositional skill libraries, world models, or generative simulation pipelines. In autonomous policy rollouts, actions are produced by policies trained through reinforcement learning, imitation learning, curriculum learning, or previous rounds of data generation. Such rollouts are commonly collected in simulation environments for multi-task manipulation, dexterous manipulation, articulated-object manipulation, deformable-object manipulation, and whole-body control. Representative environments include Meta-World [408], DexArt [12], Bi-DexHands [56], SoftGym [213], PlasticineLab [149], DaXBench [48], and HumanoidBench [305]. Relative to fixed rule-based experts, learned policies can react to observations and may visit policy-induced states, including failures and recoveries. That said, the resulting data distribution is strongly shaped by reward design, exploration strategy, policy competence, simulator fidelity, and data-retention criteria. Low-quality policies may therefore produce large quantities of redundant or unsuccessful data, making validation and filtering essential.

Recent generative data factories further shift human effort from writing per-task experts toward designing scalable task, asset, skill, and validation pipelines. LLM-assisted systems such as GenSim [350] and GenSim2 [144] generate simulation tasks, task code, or articulated-object manipulation settings and subsequently use planning or reinforcement-learning solvers to produce demonstrations. RoboGen [359] similarly explores generative simulation for task, scene, and supervision generation. RoboTwin [259] and RoboTwin2.0 [49] emphasize scalable bimanual manipulation data generation through generative digital twins, domain randomization, and embodiment-aware execution. InternData-M1 [53], GenManip [107], and InternData-A1 [341] instantiate a more modular form of this paradigm, in which scene graphs, object states, skill specifications, waypoints, motion planning, physics validation, and rendering are decoupled to generate data for instruction following, spatial grounding, and generalist-policy training. Larger simulation platforms and data engines, including RoboVerse [113], Genie Sim 3.0 [401], ManiTwin [349], and RoboCasa365 [263], further extend this direction by integrating asset construction, scenario generation, multi-embodiment execution, rendering, and benchmarking into unified synthetic-data pipelines.

This category also includes recent efforts to expand synthetic data beyond conventional RGB and proprioceptive trajectories. GraspVLA/SynGrasp-1B [80] and Dex1B [395] generate large-scale synthetic action data for grasping and dexterous manipulation. Visuotactile simulation frameworks extend policy rollouts to contact-rich settings by synthesizing tactile observations, elastomer deformation, marker displacement fields, or other multimodal tactile signals. Representative systems include Taxim [317], TACTO [354], Tactile Gym [214], Tacchi [58], DiffTactile [318], TacEx [265], TacSL [2], TacCel [208], TacFlex [418], UniVTac [43], and Tac2Real [387]. Tac2Real [387] is representative of this trend because it uses high-throughput visuotactile simulation to support online reinforcement learning and zero-shot sim-to-real transfer for contact-rich manipulation.

Beyond physics-based simulation, DreamGen [155] introduces a related world-model-based direction in which video world models generate synthetic robot videos or neural trajectories from initial visual states and language instructions, followed by pseudo-action recovery for policy learning. These approaches reduce the need for task-specific teleoperation and manual scripting, but they also introduce additional risks, including task invalidity, semantic inconsistency, visual-action misalignment, hallucinated dynamics, planner-induced behavioral bias, and distributional mismatch with real-robot experience.

5.5World Models as Simulators

Beyond conventional physics-based simulation, recent work increasingly explores world models as learned simulators for robot learning [141]. Rather than relying on manually specified assets, dynamics, and rendering pipelines, world models learn predictive environment dynamics from collected observations and interactions. Conditioned on observation histories, task descriptions, and candidate actions, they generate future observations, latent states, rewards, or task outcomes, thereby transforming existing experience into imagined interaction trajectories. This idea builds on model-based reinforcement learning and latent imagination [124, 126, 127, 128], and has been extended to real-robot learning and interactive visual simulation by systems such as DayDreamer and UniSim [370, 392].

5.5.1World Models for Policy Training and Post-Training

World models can serve as virtual environments in which policies are optimized without repeatedly interacting with physical robots. Imagined transitions and estimated rewards support reinforcement learning, policy adaptation, and VLA post-training. Representative approaches include World-Env, VLA-RFT, DiWA, and World4RL [380, 193, 37, 165], while recent systems further explore autonomous play, progress estimation, and large-scale world-model-based policy refinement [307, 402, 381, 452, 423, 389, 336]. These methods can reduce real-world interaction costs and generate experience in parallel, but their benefits depend on whether the learned simulator preserves the causal effects of robot actions.

5.5.2World Models for Policy Evaluation

Learned simulators can also estimate policy performance before physical deployment. WorldGym and WorldEval study whether generated rollouts can support policy evaluation and checkpoint selection, while Veo-based evaluation examines nominal, out-of-distribution, and safety-related behaviors [286, 204, 111]. Unlike policy training, the goal here is to predict task progress, completion, failure, or unsafe outcomes rather than directly update the policy. Visual realism alone is therefore insufficient: useful evaluators must preserve action controllability, task semantics, temporal consistency, and the relative ranking of candidate policies.

5.5.3World Models as Synthetic Data Engines

World models further provide a scalable mechanism for synthesizing interaction data conditioned on scenes, instructions, actions, viewpoints, or embodiments. DreamGen generates robot videos and recovers pseudo-actions through latent-action or inverse-dynamics models, while GigaWorld-0 explicitly treats generative world models as data engines for embodied learning [155, 114]. Compared with conventional augmentation, these methods model temporal evolution and action consequences; compared with physics-based simulation, they reduce dependence on manually constructed assets. However, generated videos do not necessarily contain accurate executable actions, and action recovery may introduce visual-action misalignment.

Overall, world models are evolving from auxiliary predictive components into learned environments for policy optimization, evaluation, and data generation. Their scalability and visual coverage make them a promising complement to physics-based simulators, but prediction errors, model exploitation, action ambiguity, and distribution shift can produce plausible yet physically invalid trajectories. Consequently, learned simulation requires filtering, uncertainty estimation, and calibration against real-robot or physics-based interaction data.

5.6Sim-to-Real Gap

Sim-to-real transfer has long been a central problem in simulation-based robot data, as synthetic trajectories are expected to support real-robot deployment rather than only in-simulator learning [259, 263, 43]. From a data-centric perspective, this gap mainly arises from observation mismatch and interaction mismatch.

The first concerns the discrepancy between rendered sensory observations and real-robot perception. Although recent simulation datasets improve realism through 3D assets, lighting, camera models, articulated objects, and multimodal sensors [341, 84, 53], their observations can still differ from real data in texture, illumination, material appearance, depth noise, camera calibration, occlusion, tactile response, and force signals.

The second concerns the mismatch between simulated and real robot-environment interactions, which can be further separated into kinematic and dynamic gaps. Kinematic gaps arise from differences in robot morphology, joint limits, coordinate frames, action interfaces, controller frequencies, and calibration, and can often be reduced through accurate robot modeling, real-to-sim alignment, and hardware-consistent control interfaces. Dynamic gaps are harder to eliminate, because real execution involves actuator latency, compliance, torque saturation, backlash, friction, contact, deformation, object inertia, and multi-body interaction effects that are only partially captured by current simulation engines. For efficiency, many simulation pipelines further simplify contact dynamics, use coarse time steps, or approximate robot-environment state updates, producing trajectories that may be visually plausible but dynamically inconsistent with real execution [67, 361, 14]. As a result, the core question is not only how much simulation data can be generated, but whether the generated data captures the sensory variations, kinematic consistency, and irreducible dynamic uncertainty that determine real-robot deployment.

5.7Advantages and Limitations

Simulation data provide scalable supervision by generating robot observations, actions, and task outcomes in controllable virtual environments. Compared with real-robot data, they can be collected with lower marginal cost, extensive parallelization, limited safety risk, and no hardware wear. Simulators also provide privileged labels that are difficult to obtain from real systems, such as object poses, segmentation masks, contact states, physical parameters, success signals, and dense rewards. These properties make simulation useful for large-scale policy pretraining, controlled evaluation, scene and task randomization, and the synthesis of diverse interaction trajectories.

A primary limitation of simulation data is limited physical fidelity. Approximations in sensing, actuation, contact, and object dynamics can produce substantial sim-to-real discrepancies, particularly for contact-rich manipulation, deformable objects, reflective materials, liquids, and long-horizon tasks. Moreover, scale does not necessarily imply diversity: as shown by InternData-A1 in Figure 4, skill-constrained generation can produce many trajectories whose action keypoints remain concentrated within a small region, indicating substantial repetition. Consequently, simulated policies often require domain randomization, system identification, digital-twin reconstruction, or real-world fine-tuning.

Takeaway
Simulation
Simulation data forms the most scalable and cost-efficient robot-aligned layer of the pyramid. Automated generation enables repeated collection of finely segmented atomic behaviors, providing dense and controllable supervision for learning specific manipulation skills. However, real-world deployment requires policies to internalize physical regularities that cannot be fully recovered from imperfect simulator dynamics—particularly for contact, materials, and object interactions. Scene diversity is likewise hard to guarantee, as generated data remain constrained by available assets. Moreover, scaling skill and task diversity in simulation still hinges on annotation, skill scripting, or human demonstrations; automating skill discovery from video and using it to drive data synthesis will be a critical next step.
6General Data
Figure 8:Overview of General Data Categories. Overview of general data categories and representative tasks, spanning cognition, task reasoning, perception, and action, with examples of vision-language QA, OCR, video temporal reasoning, physical reasoning, planning, segmentation and localization, spatial and 3D perception, and grasping.
6.1Data Overview

Robot learning relies on more than data collected directly from physical robots. Although robot trajectories provide the most direct supervision for action generation and control, they are expensive to collect and usually cover only a limited range of objects, environments, and tasks. Modern robot foundation models therefore incorporate large amounts of general image, video, language, spatial, and reasoning data. The primary purpose of such data is not to teach a particular robot how to move, but to establish the perceptual, semantic, spatial, temporal, and planning capabilities required before physical actions can be learned efficiently. General data is characterized by massive scale, broad coverage, and high diversity, but remains weakly aligned with executable robot actions and lacks direct robot-environment interaction.

General data should be understood according to the capability they contribute to a robot model. As shown in Figure 8 and Table 6, general vision-language data provide broad semantic knowledge and commonsense reasoning; segmentation and localization data ground this knowledge in spatial regions; video data introduce temporal and procedural structure; spatial and 3D data provide geometric awareness; and planning-oriented data teach the model to organize observations and goals into executable steps. These capabilities form the cognitive foundation on top of which robot-specific trajectories and control signals are learned.

Table 6:Representative General-Data Datasets. Representative datasets grouped by their primary supervision type, with their data modality, scale, supervision targets, and annotation pipeline. VL denotes vision-language, Seg./Loc. denotes segmentation and localization, and EE denotes end effector. Annotation types are categorized as Manual for human-provided labels, Automatic for labels generated from metadata, geometry, simulation, computer-vision methods, or programmatic procedures, and Semi-auto for model-assisted annotation followed by filtering or human verification.
Task
 	
Dataset
	
Modality
	
Scale
	
Labels / Targets
	
Anno.


VL (8)
 	
LLaVA-Instruct-150K [219]
	
Image-Text
	
158K
	
Caption; VQA; instruction; dialogue
	
Semi-auto


 	
LLaVA-OneVision-SI [188]
	
Image-Text
	
3.2M
	
VQA; caption; OCR; knowledge; instruction
	
Semi-auto


 	
ShareGPT4V [46]
	
Image-Text
	
1.2M captions
	
Dense caption; image-text alignment
	
Semi-auto


 	
PixMo [79]
	
Image-Text
	
1M img.; 3.3M ann.
	
Caption; QA; count; point; document
	
Manual


 	
A-OKVQA [300]
	
Image-Text
	
24.9K QA
	
Knowledge VQA; rationale
	
Manual


 	
LLaVA-Video-178K [434]
	
Video-Text
	
178K vid.; 1.34M ann.
	
Caption; open/MC QA
	
Semi-auto


 	
RoboVQA [304]
	
Robot/Human Video-Text
	
238 h; 829.5K pairs
	
Plan; success; affordance; past/future QA
	
Semi-auto


 	
EgoTaskQA [159]
	
Video-Text
	
40K QA
	
State; causal; predictive QA
	
Automatic


Robot Video (4)
 	
RoVid-X [82]
	
Video / Depth / Flow
	
4M clips; 10K+ h
	
Verb; segment; captions; depth; flow
	
Automatic


 	
RoboNet [75]
	
Robot Video
	
162K vid.; 15M frames
	
Raw robot trajectories
	
-


 	
RH20T-P [57]
	
Robot Video
	
38K clips; 67 tasks
	
Primitive labels; temporal segments
	
Manual


 	
RobAVA [328]
	
Robot Video
	
40K videos
	
Action class; normal/anomaly attributes
	
Manual


OCR/UI (5)
 	
OCR-VQA [255]
	
Image-Text
	
207K img.; 1.0M QA
	
OCR-based VQA
	
Automatic


 	
TextOCR [321]
	
Image-Text
	
28K img.; 900K words
	
Text boxes; transcription
	
Manual


 	
DocVQA [253]
	
Document-Text
	
12.8K img.; 50K Qs
	
Document VQA
	
Manual


 	
ChartQA [252]
	
Chart-Text
	
32.7K QA
	
Chart QA; numeric reasoning
	
Semi-auto


 	
MultiUI [223]
	
UI-Text
	
7.3M
	
UI grounding; elements; actions
	
Automatic


Seg./Loc. (8)
 	
Objects365 [306]
	
Image-Box
	
638K img.; 10M boxes
	
Object boxes; categories
	
Manual


 	
SA-1B [176]
	
Image-Mask
	
11M img.; 1.1B masks
	
Promptable masks
	
Semi-auto


 	
ADE20K [448]
	
Image-Mask
	
25K img.; 150 cls.
	
Semantic masks; scene labels
	
Manual


 	
PACO-LVIS [290]
	
Image-Mask
	
641K part masks
	
Object/part masks; attributes
	
Manual


 	
RefCOCO [404]
	
Image-Text-Box
	
142K expr.
	
Referring-expression grounding
	
Manual


 	
PixMo-Points [79]
	
Image-Text-Point
	
428k img.; 2.3M expr.
	
Pointing; counting
	
Manual


 	
RoboPoint [413]
	
Image-Text-Point
	
1.43M QA
	
Object/affordance/free-space points
	
Automatic


 	
RoboAfford++ [131]
	
Image-Text-Point
	
870K img.; 2.0M QA
	
Part/affordance localization
	
Semi-auto


Temporal (6)
 	
Ego4D NLQ [118]
	
Video-Text-Time
	
227 h; 19K queries
	
Query-to-moment localization
	
Manual


 	
Charades-STA [106]
	
Video-Text-Time
	
16K moment pairs
	
Sentence-to-moment grounding
	
Semi-auto


 	
DiDeMo [133]
	
Video-Text-Time
	
10K vid.; 40K desc.
	
Text-to-moment localization
	
Manual


 	
HiREST [415]
	
Video-Text-Time
	
3.4K pairs; 8.6K steps
	
Moment; step segments; captions
	
Manual


 	
Moment-10M [279]
	
Video-Text-Time
	
10M moments
	
Segment-level grounding
	
Semi-auto


 	
COIN [331]
	
Video-Step-Time
	
11.8K vid.; 46.3K seg.
	
Procedure steps; temporal segments
	
Manual


Spatial/3D (7)
 	
ScanNet [68]
	
RGB-D / Mesh
	
1.5K scenes; 2.5M views
	
Depth; pose; mesh; 3D semantics
	
Semi-auto


 	
ScanNet++ [399]
	
RGB-D / Mesh
	
460 scenes
	
Hi-res geometry; views; semantics
	
Semi-auto


 	
ARKitScenes [13]
	
RGB-D / LiDAR
	
5K cap.; 1.7K scenes
	
Depth; reconstruction; 3D boxes
	
Semi-auto


 	
WildDet3D [147]
	
Image-3D Box
	
1.00M img.; 3.91M boxes
	
Open-vocab monocular 3D boxes
	
Semi-auto


 	
MMScan [242]
	
3D-Text
	
1.4M cap.; 3.04M QA
	
3D captions; grounding; QA
	
Semi-auto


 	
ScanQA [9]
	
3D-Text
	
41K Qs; 800 scenes
	
Free-form 3D QA
	
Manual


 	
SQA3D [246]
	
3D-Text
	
33.4K Qs; 6.8K sit.
	
Situated agent-centric 3D QA
	
Manual


Planning (4)
 	
ALFRED [313]
	
Obs.-Lang.-Action
	
25K dir.; 8K demos
	
Goal/subgoal instructions; actions
	
Manual


 	
EgoPlan-IT [55]
	
Video-Text-Plan
	
50K planning QA
	
Goal; subgoal; next action
	
Semi-auto


 	
ShareRobot [158]
	
Robot Video/Image-Text-Point
	
51.4K eps.; 1.028M QA
	
Planning; affordance; EE trajectory
	
Semi-auto


 	
LLaRP [330]
	
3D-Lang.-Action
	
114 tasks
	
Language task; high-level actions
	
Automatic


Physics/Failure (4)
 	
CLEVRER [400]
	
Video-QA
	
20K vid.; 300K+ Qs
	
Descriptive/causal/predictive/counterfactual QA
	
Automatic


 	
IntPhys [293]
	
Video-Label
	
15K videos
	
Possible vs. impossible
	
Automatic


 	
InfLevel [363]
	
Video-Label
	
5.8K real; 75.3K sim.
	
Continuity; solidity; gravity; plausibility
	
Automatic


 	
RoboFail [228]
	
Video-Text
	
100 sim.; 30 real
	
Failure type/cause; recovery
	
Manual


Grasping (4)
 	
RealVLG-11B [197]
	
Image-Text-Mask-Grasp
	
165K img.; 1.3M ann.; 11B grasps
	
Grounding; box; mask; 4-DoF grasp; contact
	
Semi-auto


 	
GraspNet-1Billion [94]
	
RGB-D-Point-Grasp
	
97K img.; 1.1B grasps
	
Scene-level 6-DoF grasps
	
Semi-auto


 	
SuctionNet-1Billion [32]
	
RGB-D-Point-Grasp
	
97K img.; 1.1B labels
	
Contact; normal; suction quality
	
Automatic


 	
DexGraspNet 2.0 [424]
	
Mesh-Hand-Grasp
	
8.3K scenes; 427M grasps
	
Hand pose; contact; grasp quality
	
Automatic
6.2Visual-Language Data

Vision-language data provide the semantic and reasoning foundation of modern robot models. Whereas robot trajectories usually cover a limited set of environments, objects, and tasks, large-scale image-text and video-text corpora expose models to diverse scenes, activities, tools, materials, and linguistic expressions. Beyond visual recognition and instruction understanding, captioning, visual question answering, and multimodal instruction data provide commonsense knowledge about object attributes, spatial relations, human intentions, functions, and likely event outcomes, which is rarely annotated explicitly in robot demonstrations but is important for task interpretation and planning.

Image-based instruction mixtures, including LLaVA-style data [219, 188], ShareGPT4V [46], and PixMo [79], combine captions, questions, detailed descriptions, and reasoning responses from benchmarks such as COCO, VQA, Visual Genome, and A-OKVQA [212, 6, 179, 300]. They teach models to identify scene content, understand relations, compare regions or views, and answer natural-language questions. Video-language data further introduce human actions, object-state changes, temporal order, and procedural structure. Egocentric or robot-view videos annotated with captions or question-answer pairs provide first-person semantic and temporal understanding, although they do not directly supervise robot actions [118, 159, 434, 139].

These datasets are collected from public visual repositories, web pages, video platforms, instructional videos, and existing benchmarks. While earlier corpora relied mainly on human annotations, recent pipelines increasingly use multimodal models to generate captions, questions, answers, rationales, and conversations at scale. Such generation improves coverage and diversity but may introduce hallucinated objects, incorrect relations, or generic language. Consequently, automatic annotations are commonly filtered through confidence estimation, consistency checks, answer verification, model agreement, or human review, and are often combined with smaller high-quality human-annotated datasets.

In robot foundation models, vision-language data are primarily used for multimodal alignment and instruction tuning, establishing broad scene understanding, commonsense reasoning, task comprehension, and question answering, including under ambiguous, insufficient, or noisy instructions [374, 160, 373]. During subsequent robot-specific training, general vision-language data may be retained to preserve semantic and linguistic capabilities, while spatial grounding, planning, grasping, and trajectory supervision connect this broad knowledge to executable behavior.

6.3Segmentation and Localization Data

Object detection, segmentation, and localization remain central to robot perception because manipulation is typically organized around objects and their functional parts. Before grasping a cup, opening a drawer, pressing a button, or placing an item, a robot must identify the relevant object, estimate its location, and determine where interaction should occur. These data therefore complement the semantic knowledge learned from vision-language corpora with explicit spatial grounding.

Classical datasets supervise object categories through bounding boxes, semantic or instance masks, and part-level regions. COCO [212], LVIS [123], Objects365 [306], ADE20K [448], and SA-1B [176] provide broad coverage of everyday objects and scenes, while part-centric resources such as PACO [290] annotate finer-grained functional components. This distinction is particularly important for manipulation: recognizing a drawer is insufficient if the policy cannot localize its handle, and using a tool may require separating its graspable region from its functional end.

Language-conditioned localization further connects visual perception with natural-language instructions. Referring-expression datasets such as RefCOCO [404] associate descriptions with target boxes or masks, while PixMo-Points [79] supervises coordinate-based pointing. Robot-oriented datasets, including RoboRefIt [236] and RoboPoint [413], extend these tasks to cluttered indoor and manipulation scenes. Affordance datasets add action relevance by identifying graspable parts, handles, buttons, placement regions, or navigable areas, shifting the prediction target from what an object is to where an interaction should occur.

These datasets are collected from web images, photographic repositories, recognition benchmarks, tabletop scenes, egocentric recordings, robot-mounted cameras, and synthetic environments. Their annotations may take the form of manually drawn boxes and masks, interactive or model-assisted segmentation, referring expressions, or point clicks. Simulation offers exact object identities, masks, coordinates, and visibility information without manual labeling, while recent pipelines increasingly use multimodal models to generate region descriptions or referring expressions, followed by geometric checks, confidence filtering, or human verification.

For robot learning, the main value of such data is that they transform semantic concepts into spatially actionable targets. A model may understand the instruction “pick up the red cup,” but a manipulation system still requires a box, mask, point, or part-level region identifying the cup in the current observation. This representation can then condition grasp generation, navigation, motion planning, or low-level control. Segmentation, grounding, pointing, and affordance localization thus provide a direct interface between general visual understanding and physical execution.

However, annotations from general images do not fully capture the difficulty of robot perception. Robotic scenes often involve heavy occlusion, repeated object categories, unusual viewpoints, motion blur, and partially visible functional parts, while automatically generated expressions may be ambiguous or refer to the wrong instance. Robust systems therefore commonly combine large-scale general-purpose segmentation data with smaller, task-relevant robot datasets and apply relation-aware annotation, filtering, or validation in real manipulation scenes.

6.43D Data

3D data provide the geometric foundation required for embodied perception and physical interaction. General vision-language data may allow a model to recognize objects and understand scene semantics, but a robot must additionally estimate where objects are located, how far away they are, how they are oriented, and how they are arranged relative to the agent and the surrounding environment. Such data therefore support reasoning about depth, distance, direction, containment, support relations, visibility, free space, and viewpoint changes.

This category includes RGB-D images, point clouds, reconstructed meshes, multi-view observations, camera trajectories, 3D object annotations, and spatial question-answer data. Indoor scene datasets such as ScanNet [68], ScanNet++ [399], ARKitScenes [13], and 3RScan [345] provide RGB-D sequences together with camera poses and reconstructed geometry. Language-augmented resources such as ScanQA [9] and SQA3D [246] further require models to answer questions about object locations, relative directions, and agent-centric observations. Other datasets generate spatial questions from object coordinates, scene graphs, or reconstructed environments, providing scalable supervision for metric and relational reasoning.

These data are commonly collected using RGB-D cameras, mobile-device depth sensors, stereo systems, LiDAR, or multi-view reconstruction pipelines. Depth maps [320, 13] and camera poses may be obtained directly from sensors, visual-inertial tracking, SLAM [260], or structure-from-motion [298]. Multiple observations are then fused into point clouds or meshes. Object identities [68, 345], semantic regions [68, 399], 3D boxes [13], and scene relations are often annotated manually [346], while geometric properties such as distance, relative direction, visibility, and containment can be computed automatically from the reconstructed scene. Synthetic environments provide an additional source of supervision because they can directly output exact depth, surface normals, camera parameters, segmentation labels, and object poses.

For robot learning, the main value of these data is to transform visual observations into a physically meaningful representation of the environment. A robot must determine not only which object is relevant, but also whether it is reachable, whether another object blocks the approach path, whether enough free space is available for manipulation, and from which viewpoint the target can be observed most clearly. Three-dimensional scene representations also allow information obtained from different views to be integrated, which is important when objects are partially occluded or temporarily outside the camera view.

Spatial and 3D supervision consequently supports navigation, collision avoidance, viewpoint selection, object search, grasp planning, and manipulation. It provides the geometric context needed to evaluate whether a semantically reasonable action is physically feasible. In a robot foundation model, these data complement visual-language understanding by grounding object and task concepts in depth, scale, orientation, and scene structure, thereby providing the geometric basis on which planning and control modules operate.

6.5Planning Data for Task Decomposition

Planning data supervise the transformation of a task goal and current observation into an ordered sequence of steps, subgoals, or next actions. Unlike low-level control data, their targets describe abstract task structure, such as locating an object, preparing the workspace, executing a manipulation step, and verifying the resulting state. They therefore provide an intermediate reasoning layer between semantic understanding and embodiment-specific execution.

Such data can be derived from human-written instructions, manuals, instructional or egocentric videos, robot demonstrations, and simulation. Video-based datasets segment demonstrations into meaningful stages and associate observations with next-step predictions, subgoal sequences, or reasoning traces, as exemplified by EgoPlan-IT and EgoCOT [55, 258]. Robot trajectories provide more physically grounded supervision through action sequences and object-state transitions, while simulation enables scalable generation and verification using explicit states, action preconditions, and success conditions.

Language and multimodal models are also increasingly used to generate plans or next-action annotations, although such outputs may contain plausible but physically invalid steps. A complementary use is to let a pretrained model supply the plan directly at deployment time rather than as offline supervision: ASCENT performs zero-shot object-goal navigation in multi-floor buildings by pairing frontier ranking with LLM-based contextual reasoning, generalizing to unseen object categories without task-specific retraining [116]. Such training-free pipelines illustrate how far general-data priors alone can carry high-level embodied decision making, while also exposing their limits, since the resulting plans remain ungrounded in the embodiment’s own dynamics. These annotations therefore require filtering through demonstrated action order, state changes, simulator execution, success conditions, or human review. Combined with robot trajectories and control data, planning supervision helps embodied models decompose long-horizon goals, track task progress, and select executable subgoals.

6.6Video and Temporal Data

Video-temporal data provide embodied models with a form of perceptual memory. Rather than observing a task as a single image, a robot receives a continuous stream of partial observations in which objects may move, change state, become occluded, or disappear from view. The model must therefore connect the current observation with past events and determine when a task-relevant action or state transition occurred. This category mainly includes temporal grounding, event localization, step segmentation, and long-video memory supervision, rather than generic action recognition.

Temporal grounding aligns a language query or task description with a specific video interval. Given a query such as when was the drawer opened?” or locate the step in which the object was placed into the container,” the model predicts the corresponding start and end timestamps. This requires searching over the observation history, retrieving the relevant event, and distinguishing it from similar actions at other times. Long-video and episodic-memory data further train models to answer what happened, when it occurred, and where an object was previously observed. Egocentric recordings are especially useful because they provide long, moving, and partially observed streams resembling the perceptual history of an embodied agent.

These datasets are typically collected from instructional, daily-activity, or egocentric videos. Human annotators may segment videos into meaningful steps, write descriptions for each interval, or select temporal boundaries in response to queries. Narration timestamps provide weaker event-language alignment, while larger-scale pipelines may use multimodal models to generate descriptions and candidate timestamps, followed by transcript alignment, consistency filtering, or human verification. Such supervision helps an embodied model retrieve past events, track object-state changes, and recognize task progress over extended interactions.

6.7Physical, Causal, and Failure Reasoning

Physical reasoning data teach models to predict how scenes may change under different actions. Datasets such as CLEVRER [400] and IntPhys [293] provide supervision for collision, object permanence, stability, future-state prediction, and counterfactual reasoning. They help models distinguish a physically plausible outcome from an impossible one.

Failure-oriented data [228] such as RoboFail [228] focus more directly on robot execution. They may label whether a task failed, identify the stage at which the failure occurred, explain its cause, or describe a possible recovery action. Such supervision can be used to train critics, progress estimators, or planning-time verifiers.

Synthetic physics datasets are generated in simulation, where object states, collisions, and counterfactual outcomes can be obtained automatically. Real-world failure data require repeated task execution under varied conditions. Labels may be provided manually, derived from task-success detectors, or generated by comparing observed state transitions with the intended goal.

6.8Grasp Data

Grasp data specify where and how a robot end effector should contact an object. The appropriate representation depends on the task. Cornell [187], Jacquard [83], and Grasp-Anything [344] provide image-plane grasp rectangles, whereas GraspNet-1Billion [94], ACRONYM [92], and Grasp-Anything-6D [266] provide three-dimensional parallel-jaw grasp candidates. SuctionNet [32] provides surface contact and suction quality, while DexGraspNet-style datasets [353, 424] represent wrist pose and multi-finger joint configurations.

Two-dimensional grasp data are efficient to construct and support top-down grasp prediction, but their outputs normally require depth and camera calibration before robot execution. In contrast, 6-DoF datasets directly specify the gripper pose in three-dimensional coordinates and may also include gripper width, collision status, and grasp quality. Language-conditioned datasets such as RefGraspNet [17] additionally connect a referring expression to a set of executable grasp poses.

Grasp annotations are generated using several methods. Small 2D datasets may use manually drawn oriented rectangles. Larger synthetic datasets sample candidate poses around object meshes and evaluate them using geometric metrics, collision checking, or physics simulation. Scene-level datasets render or capture RGB-D observations and transform valid object-level grasps into scene coordinates. Real-robot trials may then be used to validate simulation-derived quality scores.

It is important to distinguish direct grasp-pose data from auxiliary grasp resources. Contact maps, human hand-object interaction data, object pose annotations, and 3D object meshes provide useful priors, but they do not directly specify an executable robot gripper pose. They normally require a separate grasp generator, retargeting method, or optimization stage.

6.9Advantages and Limitations

General data offer a scale, diversity, and accessibility that no robot-collected source can match. Requiring neither hardware, teleoperation, nor environment resets, they can be assembled from existing web, video, and benchmark resources at low marginal cost, and they remain embodiment-agnostic, so the same corpora are reusable across platforms and tasks. Their contributions are also complementary rather than redundant: semantic and commonsense knowledge from vision-language corpora, spatial grounding from segmentation and localization data, geometric structure from 3D data, perceptual memory from video, task structure from planning data, and physical plausibility from reasoning benchmarks jointly form the cognitive foundation that makes subsequent robot-specific training more efficient.

The corresponding limitation is weak action grounding. General data capture neither proprioception, actuator dynamics, contact forces, nor the physical consequences of executed actions, and therefore cannot by themselves specify what a particular robot should do. Their visual distributions also differ from robot observations, which more often involve occlusion, unusual viewpoints, motion blur, and repeated object instances. Increasingly automated annotation pipelines add a further risk, since generated labels, referring expressions, and plans may be hallucinated, ambiguous, or physically invalid, and require verification before use. General data are consequently best treated as a broad prior over perception, semantics, and reasoning that the more robot-aligned layers must ground in physical execution.

Takeaway
General
General data supplies web-scale semantic, perceptual, and reasoning priors far beyond what physical-interaction data can provide, but with weak action and contact grounding. It is most valuable for building perception, reasoning, and planning capabilities that higher, more robot-aligned layers later ground in physical execution.
7Data Applications in Embodied Foundation Models
7.1Overview

The preceding sections introduced the collection paradigms and major forms of embodied data. We now shift from data construction to data application, examining how different types of data are used to train embodied foundation models. We first summarize the data recipes and action representations adopted by representative model architectures. We then discuss three major model families, i.e., embodied brain models, vision-language-action models (VLAs), and world action models (WAMs), focusing on how each family utilizes data with and without robot action labels. This perspective highlights the distinct roles of different data types in supporting embodied understanding, control, and dynamics modeling.

7.2Data Composition and Action Representation
Table 7:Representative VLA and WAM approaches. The models are grouped and sorted by release time, model type, institution, and data source icons. Left Half: Models from 2023 to 2025; Right Half: Models from 2026. Data sources are indicated by 
 real-robot data, 
 egocentric data, 
 UMI data, 
 simulation data, and 
 general data.
Time
 	
Method
	
Institution
	
Model
	
Data


2023.3
 	
PaLM-E [88]
	
Google
	
VLA
	
 
 


2023.7
 	
RT-2 [23]
	
DeepMind
	
VLA
	
 


2024.5
 	
Octo [339]
	
UC Berkeley
	
VLA
	


2024.6
 	
OpenVLA [174]
	
Stanford University
	
VLA
	


2024.10
 	
GR-2 [41]
	
ByteDance
	
VLA
	
 


2024.10
 	
𝜋
0
 [21]
	
Physical Intelligence
	
VLA
	


2024.10
 	
RDT-1B [225]
	
THU
	
VLA
	


2024.11
 	
CogACT [198]
	
THU
	
VLA
	


2025.1
 	
SpatialVLA [285]
	
Shanghai AI Lab
	
VLA
	


2025.1
 	
UP-VLA [425]
	
THU
	
VLA
	
 


2025.3
 	
HybridVLA [221]
	
PKU
	
VLA
	


2025.3
 	
GR00T N1 [20]
	
NVIDIA
	
VLA
	
 
 


2025.3
 	
GO-1 [26]
	
Shanghai AI Lab & AgiBot
	
VLA
	
 
 


2025.3
 	
CoT-VLA [437]
	
NVIDIA
	
VLA
	
 
 


2025.4
 	
UWM [450]
	
University of Washington
	
WAM
	


2025.4
 	
𝜋
0.5
 [150]
	
Physical Intelligence
	
VLA
	
 


2025.5
 	
UniVLA [27]
	
HKU
	
VLA
	
 


2025.6
 	
SmolVLA [316]
	
Hugging Face
	
VLA
	


2025.6
 	
GR00T N1.5 [20]
	
NVIDIA
	
VLA
	
 


2025.7
 	
DreamVLA [430]
	
SJTU
	
VLA
	
 


2025.7
 	
EgoVLA [391]
	
UC San Diego
	
VLA
	


2025.7
 	
Being-H0 [237]
	
PKU
	
VLA
	


2025.7
 	
GR-3 [42]
	
ByteDance Seed
	
VLA
	
 
 


2025.7
 	
H-RDT [19]
	
THU
	
VLA
	


2025.8
 	
GalaxeaVLA(G0) [162]
	
Galaxea
	
VLA
	


2025.8
 	
ReconVLA [324]
	
HKUST
	
VLA
	
 


2025.8
 	
MolmoAct [184]
	
AI2 & UW
	
VLA
	
 


2025.8
 	
EO-1 [284]
	
Shanghai AI Lab
	
VLA
	
 


2025.9
 	
RynnVLA-001 [163]
	
DAMO
	
VLA
	
 


2025.10
 	
X-VLA [442]
	
THU
	
VLA
	


2025.10
 	
InternVLA-M1 [53]
	
Shanghai AI Lab
	
VLA
	
 
 


2025.10
 	
VITRA [199]
	
THU
	
VLA
	


2025.11
 	
METIS [101]
	
PKU
	
VLA
	
 


2025.11
 	
iFlyBot-VLA [433]
	
iFLYTEK & LindenBot
	
VLA
	
 
 


2025.12
 	
VideoVLA [311]
	
XJTU
	
VLA
	


2025.12
 	
Motus [18]
	
THU
	
WAM
	
 
 
 


2025.12
 	
GR00T N1.6 [20]
	
NVIDIA
	
VLA
	
 
 
Time
 	
Method
	
Institution
	
Model
	
Data


2026.1
 	
InternVLA-A1 [28]
	
Shanghai AI Lab
	
VLA
	
 
 


2026.1
 	
LaST0 [230]
	
PKU
	
VLA
	


2026.1
 	
Being-H0.5 [238]
	
BeingBeyond
	
VLA
	
 
 
 


2026.1
 	
LingbotVLA [376]
	
Ant Group
	
VLA
	


2026.1
 	
LingbotVA [196]
	
Ant Group
	
WAM
	
 
 


2026.2
 	
RDT2 [226]
	
THU
	
VLA
	
 


2026.2
 	
Xiaomi-
Robotics-0 [29]
	
Xiaomi
	
VLA
	
 


2026.2
 	
ABot-M0 [393]
	
AMAP CV Lab
	
VLA
	


2026.2
 	
DM0 [403]
	
Dexmal
	
VLA
	
 
 


2026.2
 	
LDA-1B [241]
	
PKU
	
WAM
	
 
 


2026.2
 	
DreamZero [397]
	
NVIDIA
	
WAM
	


2026.2
 	
EgoScale [443]
	
NVIDIA
	
VLA
	
 


2026.3
 	
GigaWorld-Policy [394]
	
GigaAI
	
WAM
	
 
 


2026.3
 	
MolmoB0T [84]
	
AI2
	
VLA
	


2026.3
 	
UniDex [420]
	
THU
	
VLA
	


2026.4
 	
JoyAI-RA 0.1 [428]
	
JD
	
VLA
	
 
 
 


2026.4
 	
𝜋
0.7
 [151]
	
Physical Intelligence
	
VLA
	
 
 


2026.4
 	
GR00T N1.7 [20]
	
NVIDIA
	
VLA
	
 
 


2026.4
 	
Being-H0.7 [239]
	
BeingBeyond
	
WAM
	
 
 
 


2026.4
 	
MotuBrain [338]
	
Shengshu
	
WAM
	
 
 


2026.5
 	
Qwen-VLA [352]
	
Qwen
	
VLA
	
 
 
 


2026.5
 	
Wall-OSS-0.5 [406]
	
X Square Robot
	
VLA
	
 


2026.5
 	
MolmoAct2 [97]
	
UW & AI2
	
VLA
	
 
 


2026.6
 	
Wall-WM [200]
	
X Square Robot
	
WAM
	
 
 
 


2026.6
 	
LaST-HD [222]
	
PKU
	
VLA
	
 


2026.6
 	
Hy-Embodied-0.5-VLA [422]
	
Tencent Robotics X
	
VLA
	


2026.6
 	
Kairos [168]
	
ACE Robotics
	
WAM
	
 
 


2026.6
 	
Qwen-RobotManip [412]
	
Qwen
	
VLA
	
 
 
 


2026.7
 	
ABot-M0.5 [47]
	
AMAP CV Lab
	
WAM
	
 


2026.7
 	
ACE-Brain-0.5 [334]
	
ACE-Robotics
	
VLA
	
 


2026.7
 	
InternVLA-A1.5 [243]
	
Shanghai AI Lab
	
VLA
	
 
 
 


2026.7
 	
LingbotVLA 2.0 [377]
	
Ant Group
	
VLA
	
 


2026.7
 	
LingbotVA 2.0 [426]
	
Ant Group
	
WAM
	
 
 
 
 


2026.7
 	
HumanScale [244]
	
PKU
	
WAM
	


2026.7
 	
GigaWorld-Policy-0.5 [337]
	
GigaAI
	
WAM
	


2026.7
 	
Xiaomi-Robotics-1 [340]
	
Xiaomi
	
VLA
	
 
 


2026.7
 	
Xiaomi-Robotics-U0 [201]
	
Xiaomi Robotics
	
WAM
	
 
 
 
7.2.1Data Recipes for Embodied Foundation Models

As summarized in Table 7, the data recipes of embodied foundation models are evolving from single-source robot demonstrations toward increasingly heterogeneous mixtures. This evolution can be clearly observed in successive generations of the same model families. For example, the 
𝜋
 series expands from real-robot data in 
𝜋
0
 [21], to real-robot and general-purpose data in 
𝜋
0.5
 [150], and further incorporates egocentric data in 
𝜋
0.7
 [151]. Similarly, LingbotVA [196] combines real-robot, UMI, and simulation data, whereas LingbotVA 2.0 [426] extends its recipe to all five levels of the data pyramid.

Despite this growing tendency toward heterogeneous pretraining, the optimal data recipe for embodied foundation models remains an open question. Models pretrained primarily or exclusively on robot data, such as LingbotVLA [376] and DreamZero [397], can also achieve strong performance, and current evidence does not establish that incorporating more data sources is inherently preferable. Nevertheless, we argue that different data types can provide complementary capabilities: general-purpose data can support broad visual, semantic, and reasoning abilities; egocentric and UMI data can expand the coverage of scenes, objects, viewpoints, and human behaviors; and simulation data can strengthen particular skills through controllable and scalable interaction generation.

A second trend is the rapid growth of pretraining scale. Earlier models such as OpenVLA [174] and VideoVLA [311] were pretrained predominantly on robot demonstration corpora from the Open X-Embodiments ecosystem, while GR-2 [41] represented an early attempt to jointly train on a comparatively limited mixture of robot and egocentric data. By 2026, the reported scale had increased substantially. Qwen-RobotManip [412] constructs an approximately 38,100-hour multi-source corpus, including roughly 11.4K hours of open-source robot data and 24,808 hours of robot-compatible trajectories synthesized from 1,933 hours of egocentric video. Xiaomi-Robotics-1 [340] further reports pretraining on more than 100,000 hours of real-world UMI trajectories, followed by approximately 10,000 hours of cross-embodiment post-training data, including more than 7,200 hours of in-house robot trajectories and over 1,000 hours of instruction-labeled UMI data. Although these quantities are not directly comparable because of differences in modality, processing, filtering, and training stage, they collectively illustrate the progression of embodied pretraining from conventional robot datasets toward corpora containing tens or even hundreds of thousands of hours of interaction data.

A third and closely related trend is the growing importance of egocentric human data. As shown in Table 7, egocentric data has been incorporated into the heterogeneous pretraining recipes of an increasing number of models, including GR-2 [41], GR00T N1 [20], GR-3 [42], Motus [18], 
𝜋
0.7
 [151], Qwen-VLA [352], Qwen-RobotManip [412], and LingbotVA 2.0 [426].

Other studies go further by using egocentric data as the primary or exclusive source during pretraining, including EgoVLA [391], Being-H0 [237], H-RDT [19], VITRA [199], UniDex [420], and HumanScale [244]. HumanScale performs controlled pretraining with up to 5,000 hours of curated egocentric video, while EgoScale [443] scales action-labeled egocentric pretraining to 20,854 hours and reports consistent improvements across subsets ranging from 1,000 to 20,000 hours. Together, these studies suggest that egocentric data is becoming more than an auxiliary source: its scalability and rich coverage of human-object interactions make it an increasingly important pretraining substrate between general web data and embodiment-specific robot trajectories.

Beyond these major trends, recent work is also beginning to reconsider strict data-quality requirements. Whereas earlier training pipelines largely retained successful and carefully curated demonstrations, 
𝜋
0.7
 [151] suggests that lower-quality trajectories may still provide useful supervision when paired with sufficiently clear prompts. This observation remains preliminary, but indicates a possible shift from aggressive data filtering toward more explicit conditioning on trajectory quality and intent.

7.2.2Action-Space Representation in Robot Foundation Models

Robot foundation models increasingly integrate action trajectories collected from heterogeneous embodiments, yet these actions are not directly comparable across datasets. Differences arise not only from embodiment-specific control dimensionality and actuator semantics, but also from the coordinate frames and geometric conventions used to express end-effector motion. Accordingly, existing approaches address action-space heterogeneity at two complementary levels: cross-embodiment structural alignment, which determines how different control variables are mapped into a shared action interface, and geometric alignment, which determines the reference frame in which spatial actions are represented. The following discussion reviews representative strategies along these two dimensions and examines their implications for multi-embodiment pretraining and transfer.

Cross-Embodiment Action-Space Representations. Existing embodied foundation models reconcile heterogeneous robot action spaces through three principal strategies: embodiment-specific projection, fixed-dimensional zero-padded action interfaces, and semantic action slots. These strategies differ in the level at which action-space heterogeneity is resolved. The first preserves embodiment-specific action interfaces and aligns them only within the shared model representation; the second standardizes the action-vector dimensionality without enforcing consistent physical semantics; and the third explicitly organizes heterogeneous robot actions into a common set of semantically defined control dimensions.

(1) Embodiment-specific projection. This strategy preserves the native action dimensionality and control semantics of each robot while using embodiment-specific projectors, adapters, or action heads to map heterogeneous state-action representations into and out of a shared policy backbone. It therefore enables representation-level sharing without replacing robot-specific control interfaces, as adopted by GR00T N1 [20], Octo [339], and SmolVLA [316].

(2) Fixed-dimensional zero-padded action interfaces. This strategy reserves a common action-vector dimensionality across embodiments. Robots with lower-dimensional action spaces fill unused dimensions with zeros, typically together with validity masks that exclude inactive dimensions from the learning objective. It provides a uniform tensor interface for joint training but does not require corresponding dimensions to share the same physical semantics across robots [21, 352, 393].

(3) Semantic action slots. This strategy assigns fixed physical meanings to predefined dimensions or blocks in a shared action vector, such that corresponding entries preserve consistent control semantics across embodiments while unavailable components are masked or zero-padded. Qwen-RobotManip[412] provides a representative implementation by defining an 80-dimensional canonical state-action vector composed of two 29-dimensional arm blocks and 22 reserved dimensions. Each arm block allocates 7 dimensions to arm joints, 9 to the end-effector pose, 1 to the parallel gripper, and 12 to dexterous-hand joints. Different embodiments populate only the relevant subsets, while inactive fields are zero-filled and excluded from the flow-matching objective using per-dimension binary masks. Related semantic-slot designs are adopted by RDT-1B [225], Being-H0.5 [238], Green-VLA [7], UniDex [420], JoyAI-RA 0.1 [428], Wall-OSS-0.5 [406] and LingbotVLA 2.0 [377]. Compared with shape-level padding, this formulation provides stronger cross-embodiment alignment by unifying both action dimensionality and physical semantics.

Geometric Action Representations.

Beyond dimensional and semantic alignment, better cross-dataset alignment may also require considering the coordinate frames used to represent end-effector actions, since equivalent motions can acquire different numerical values under different geometric conventions. Existing approaches can be broadly grouped into three categories.

(1) Robot-centric representations. This strategy expresses end-effector poses or motion commands in a world or robot-base frame, providing a direct interface to conventional robot controllers. However, the resulting representations may remain coupled to embodiment-specific base poses, mounting configurations, and workspace conventions, as exemplified by Open X-Embodiment [64].

(2) Camera-centric representations. This strategy transforms actions into a camera coordinate frame, aiming to assign more consistent numerical representations to visually similar motions across embodiments and scenes. Camera-centric action representations are adopted by Qwen-RobotManip [412] and OC-VLA [429].

(3) Wrist-centric representations. This strategy expresses fingertip poses, hand keypoints, or local hand motions in a canonical wrist frame, aiming to separate local hand articulation from global wrist translation and rotation. Such representations are used by METIS [101] and LDA-1B [241].

Regardless of the selected coordinate frame, its implementation should clearly specify the frame origin and axes, coordinate-system handedness, tool-center point, absolute-versus-delta action mode, rotation parameterization, and physical units. However, existing studies provide limited controlled ablations across these representations, and it remains unclear whether any coordinate convention consistently offers superior transfer or control performance across datasets and embodiments.

7.3Applications for Embodied Brain

Embodied brain models generally refer to vision-language models pretrained on physical-world understanding tasks to support robot perception and high-level decision-making. Unlike VLAs and WAMs, which directly emphasize action generation or dynamics prediction, these models focus on spatiotemporal perception, physical reasoning, grounding, temporal memory, and task planning. Representative models, including RynnBrain [74], HY-Embodied [409], and Pelican-VL [431], strengthen these capabilities before or alongside downstream control.

Accordingly, their training data need not always contain low-level robot actions; the key is whether the supervision contributes to physically meaningful representations. General multimodal data can support semantic understanding, visual commonsense [140], OCR, memory, and spatial reasoning, whereas interaction data containing actions, states, or motion cues can provide affordance, grasp, trajectory, and planning supervision. Recent embodied brain models therefore combine general multimodal data with embodied grounding data [74, 158]. Based on the available supervision, we organize these data into two categories: action-free data for embodied understanding and action-labeled data for physical grounding.

7.3.1Action-free Data for Embodied Understanding

A substantial portion of the data used to train embodied brains contains no explicit action annotations. This includes general-purpose image-text, video-text, VQA, OCR, spatial and temporal QA, and instruction-following data, as well as egocentric videos re-annotated for captioning, question answering, temporal memory, or task understanding. Although lacking robot states and control commands, such data provide essential supervision for object recognition, language understanding, visual commonsense, spatial-temporal reasoning, object-function inference, and task comprehension [74, 31].

Action-free videos can also support robot-centric video pretraining through temporal prediction, masked video modeling, or future-frame generation. These objectives allow models to learn object permanence, motion patterns, hand-object interactions, state transitions, and coarse physical dynamics without requiring action labels for every segment. Representative examples include GR-1, V-JEPA 2, DreamDojo, and physical-AI video foundation models such as Cosmos Predict [365, 8, 109, 3]. The resulting visual-temporal representations can serve as backbones for embodied brains or be coupled with planning and action modules during downstream adaptation.

Nevertheless, video-only pretraining provides primarily observational rather than executable knowledge. It does not necessarily reveal which action causes a transition or whether that transition is feasible for a particular embodiment; action-conditioned trajectories, robot states, inverse-dynamics supervision, or policy adaptation are still required for executable control. The primary role of action-free data is therefore to support physical-world understanding rather than embodiment-specific action alignment. General multimodal data provide broad semantic and commonsense priors, while egocentric data expose embodied brains to first-person viewpoints, occlusions, hand-object interactions, object-state changes, and long-horizon task progress [74, 409].

7.3.2Action-Labeled Data for Physical Grounding

Action-labeled data contain explicit interaction signals or demonstrations from which such signals can be recovered. They include real-robot trajectories, handheld-gripper demonstrations, simulation rollouts, and interaction videos containing robot states, hand or gripper motion, object-state changes, or action annotations. For embodied brains, these signals are typically not used as raw control targets, but are converted into more transferable supervision, such as affordance locations, grasp poses, trajectory waypoints, action boundaries, subtask labels, and next-action annotations. RoboBrain and ShareRobot exemplify this approach by treating task planning, object affordance, and end-effector trajectories as core supervision signals [158].

Such supervision supports three closely related capabilities. First, affordance labels identify functionally meaningful interaction regions, such as handles, buttons, contact surfaces, and placement areas. They can be manually annotated or recovered through segmentation, hand or gripper tracking, contact estimation, and object-state change detection [158, 431]. Second, tracking, pose estimation, SLAM, and camera-motion recovery can convert demonstrations into 2D or 3D trajectories, sparse waypoints, and approach directions. These targets provide richer physical information than static affordances while remaining more transferable than embodiment-specific joint commands [158, 432]. Third, demonstrations can be segmented into atomic actions, subgoals, and step boundaries, enabling embodied brains to recognize task progress and predict subsequent actions [74, 158].

Different data sources provide complementary action supervision. Real-robot data offer embodiment-aligned states and control traces; handheld-gripper data provide a more scalable source of end-effector-like motion; and simulation provides automatically accessible object states, contacts, trajectories, success conditions, and failure labels. Overall, action-labeled data provide a bridge between physical understanding and executable behavior. Rather than requiring embodied brains to directly imitate low-level controls, they ground perception and reasoning in affordances, trajectories, and task structures, which can subsequently support downstream VLA and WAM training [409, 269, 432].

7.4Applications for Vision-Language-Action Models

Vision-Language-Action (VLA) models are typically built upon vision-language model backbones and extend their multimodal understanding capabilities toward downstream robot control. While embodied VLMs primarily focus on perception, grounding, spatio-temporal reasoning, and high-level task understanding, VLA models further emphasize how multimodal representations can be translated into executable behaviors. Accordingly, a VLA model is expected not only to recognize task-relevant objects, states, and affordances, but also to plan action sequences, predict low-level action chunks, and adapt its behavior to specific robot embodiments and environments. Recent systems such as OpenVLA [174], Octo [339], and 
𝜋
0.5
 [150] exemplify this paradigm by integrating large-scale vision-language pretraining with embodied data, enabling generalist manipulation policies that can follow language instructions, ground task-relevant objects, and generate actions across diverse tasks, scenes, and robot platforms.

The action-oriented nature of VLA models involves two complementary levels of capability. On the one hand, a VLA model must understand the scene and environment, including task-relevant object localization, object tracking, spatial relations, object states, affordances, and temporal context. These capabilities build on high-level vision-language understanding and provide the grounding needed to decide what should be done. On the other hand, a VLA model must translate such grounded understanding into embodiment-specific planning and execution, such as action chunk generation, and adaptation to the robot’s morphology and action space. This dual objective leads to a distinctive view of embodied training data: action-free data mainly support embodied understanding and physical grounding, while action-labeled data directly supervise policy learning and action generation.

7.4.1Action-Labeled Data

Action-labeled data constitute the primary supervision source for training VLA models, as they explicitly specify how visual-language observations should be mapped to executable robot behaviors. A common strategy is to discretize continuous robot actions into action tokens, allowing actions to be generated autoregressively in the same format as language tokens. This formulation makes it possible to reuse pretrained VLM backbones and preserve the semantic knowledge acquired from large-scale vision-language pretraining, while adapting the model to robot control through trajectory supervision [23, 174].

However, discrete action tokens may be insufficient for high-frequency, fine-grained, or dexterous manipulation, where small errors in continuous control can substantially affect execution. To address this limitation, recent VLA systems increasingly adopt continuous action heads. Diffusion-based policies [225, 364, 20] formulate action generation as a conditional denoising process over continuous action sequences, enabling expressive modeling of high-dimensional and potentially multimodal action distributions. In parallel, flow matching-based policies [150, 316, 28] learn a continuous vector field that transports noisy action samples toward executable action trajectories, offering a direct and efficient formulation for high-frequency continuous control. These approaches show that action-labeled data not only provide direct control supervision, but also shape the action representation used by VLA models, from language-like discrete tokens to expressive continuous action distributions.

Beyond robot-collected trajectories, egocentric human videos with hand poses provide a scalable source of supervision for dexterous VLA learning. Some approaches reconstruct human hand motions, retarget them to a specific robot hand, and further reduce the visual embodiment gap through hand inpainting or robot-hand rendering [280, 310, 384, 192]. Other approaches map human and robot demonstrations into a shared action or interaction space, such as wrist and fingertip poses, MANO parameters, 3D keypoints, or entity-level hand-object representations, enabling large-scale pretraining on human data followed by lightweight adaptation to robot embodiments [391, 238, 101, 443, 420]. These methods broaden the scope of action-labeled data beyond directly collected robot trajectories and offer a promising route toward scaling dexterous policy learning.

7.4.2Action-Free Data

Action-Proxy Learning from Action-Free Videos. Large-scale videos provide abundant evidence about how objects move and states change during manipulation, despite lacking the corresponding action labels. To exploit such data for VLA learning, existing methods derive action proxies from temporal observations, either as implicit latent representations or explicit geometric structures such as point trajectories and motion fields. These proxies are not assumed to be executable robot actions. Instead, they encode visual transitions or interaction dynamics that can be learned at video scale and subsequently grounded in a robot-specific action space.

Latent actions are compact representations of action-relevant state transitions learned from observation sequences without explicit action labels. LAPA [396] learns discrete latent actions from unlabeled videos and uses them as pseudo-action labels for VLA pretraining. Moto [54] represents visual motion as latent motion tokens for autoregressive motion-prior learning. UniVLA [27] introduces task-centric latent actions to suppress task-irrelevant visual dynamics. Villa-X [52] further constrains latent actions with visual and proprioceptive future prediction, while CLAP [419] and ConLA [69] use contrastive objectives to disentangle action-relevant dynamics from nuisance visual changes. These methods are scalable because they avoid explicit hand-object annotation, but the learned latent space still requires robot demonstrations, action decoders, or downstream alignment to become executable on a specific embodiment.

A complementary body of work explicitly reconstructs spatial or geometric representations of manipulation from video. Although most of these methods are not VLA models in the strict sense, they demonstrate how such representations can bridge visual observation and robot control. VideoDex [309] transfers human hand motion extracted from video to robot policies, while ViSA-Flow [44] learns visual motion representations from human demonstrations before adapting them with robot data. General Flow [410] represents manipulation as object-centric motion fields, and Track2Act [15] converts predicted point trajectories into object transformations and robot actions. VidBot [45] and Actron3D [417] further recover three-dimensional trajectories or transferable action representations from human videos. Compared with latent actions, these geometric representations are more interpretable and transferable to robot control, but depend more heavily on reliable perception and cross-embodiment alignment.

Hierarchical and Intermediate Supervision. Another way to exploit action-free data is to use it for learning hierarchical or intermediate supervision that guides downstream action generation. General visual-language data, human videos, and task demonstrations can provide high-level semantic knowledge, task decomposition, subgoal prediction, and chain-of-thought reasoning, which help VLA models decide what should be done before predicting how to act. For example, 
𝜋
0.5
 [150] uses heterogeneous data to improve semantic scene understanding and high-level task structure modeling, while CoT-style VLA models [437, 445, 447, 312] introduce explicit intermediate reasoning, such as visual goals, linguistic plans, or action-level thoughts, to support long-horizon manipulation. In addition, action-free or weakly annotated data can provide affordance-level supervision, such as predicting task-relevant interaction regions, graspable areas, or functional object parts, which serves as a spatially grounded bridge between perception and control. Recent affordance-aware VLA models [195, 405] use such intermediate representations to focus the policy on where and how to interact, thereby improving the robustness of action generation in complex scenes.

7.5Applications for World Action Models

World Action Models (WAMs) model the temporal evolution of the physical world and the effect of actions on this evolution. In contrast to embodied VLMs, which emphasize perception and reasoning, and to VLA models, which directly map multimodal observations to executable behaviors, WAMs treat world dynamics as the primary learning objective, so that a trained WAM can function as a learned simulator, a planner, a policy evaluator, or a generator of synthetic interaction data.

This paradigm originates from latent world models in model-based control, where the Dreamer family [128, 370] learns action-conditioned latent dynamics and optimizes policies through imagined rollouts in both simulated and real environments. Generative video modeling subsequently extended world modeling to open-domain visual data, as represented by UniSim [392], GAIA-1 [143], and the Genie series [25, 117]. In robotics, recent systems including GR-2 [41], IRASim [451], EnerVerse [146], Genie Envisioner [210], V-JEPA 2 [8], WorldVLA [36], and Cosmos [269] apply this paradigm to manipulation, coupling action-conditioned future prediction with action generation and policy evaluation.

Recent surveys organize this rapidly expanding design space along complementary axes: by output representation, spanning video-, occupancy-, and LiDAR-based world modeling and generation [177], and by capability level, separating models that merely predict from those that act as controllable simulators or self-improving agents, together with the physical and digital regularities each is expected to respect [63]. These organizing views are useful here because they make explicit which data a given world-model role demands: passive prediction can be learned from observation alone, whereas simulator- and evaluator-style uses additionally require action-conditioned and outcome-labeled interaction.

From a data perspective, WAM training data can be divided according to whether an action condition is available for modeling world transitions. Action-labeled data, including real-robot trajectories, UMI-style demonstrations, and simulation rollouts from the upper layers of the data pyramid, provide executable or action-like signals that specify which action induces a given transition. Action-free data, including general multimodal corpora and egocentric or exocentric videos from the lower layers, support only observation-level future prediction. In practice, most systems follow a two-stage recipe, in which large-scale action-free pretraining establishes general dynamics priors and action-conditioned post-training grounds these priors in interaction data [365, 41, 8, 269].

7.5.1Action-Labeled Data

Most WAMs are built upon diffusion-based video generation models and therefore formulate action prediction as a continuous generative problem. Given a ground-truth action sequence, noise is added according to a diffusion schedule, and the model learns to recover the clean actions through a diffusion or flow-matching objective similar to that used for video generation. Architecturally, Genie Envisioner attaches a lightweight flow-matching action expert to its pretrained world-model backbone [210, 231]. Motus adopts a Mixture-of-Transformers architecture with separate video and action experts, allowing the two modalities to interact while retaining modality-specific parameters [18, 194]. VideoVLA jointly models video and action streams within a multimodal Diffusion Transformer [311], whereas Cosmos Policy directly encodes actions as latent frames and processes them using the original video diffusion backbone without introducing a separate action architecture [175].

An important exception is WorldVLA, which is built upon an autoregressive generative model rather than a diffusion model [36]. It discretizes both visual observations and continuous robot actions into tokens and jointly predicts them through autoregressive next-token prediction with cross-entropy losses. A related design compresses the prediction target instead of the backbone: in autonomous driving, Xiaomi OneVL routes reasoning through a small set of latent tokens trained with auxiliary decoders that reconstruct textual chain-of-thought and forecast future-frame tokens, so that world dynamics are absorbed into the latent space during training while inference remains a single parallel pass [234]. Such latent formulations are attractive from a data standpoint, since the future-prediction objective can be supervised by raw future observations rather than by additional action or reasoning annotations.

The use of action labels in WAMs can therefore be broadly summarized into two paradigms: continuous action denoising with diffusion or flow-matching objectives, and discrete action-token prediction with autoregressive objectives. In both cases, action supervision aligns the model’s representation of future world evolution with executable robot control.

7.5.2Action-Free Data

Action-free data establish the basic world priors of WAMs. This category comprises general multimodal data and egocentric or exocentric videos that contain no explicit or recoverable action variables. The corresponding supervision is observation-only future prediction, such as predicting future frames, visual latents, scene states, or language descriptions of state changes. Such objectives teach the model how the world typically evolves but not which action drives a given transition, and action-free data therefore contribute broad temporal and physical priors rather than executable action-conditioned dynamics.

General Video and Multimodal Data. Web-scale video corpora constitute the largest source of dynamics supervision. UniPi [89] formulates decision making as text-conditioned video generation, UniSim [392] aggregates heterogeneous video and interaction data into an interactive simulator of the real world, and Cosmos [269] pretrains omnimodal world models on web-scale video prior to physical-AI post-training. Prediction need not occur in pixel space: V-JEPA 2 learns latent video prediction from over one million hours of Internet video [8]. These data provide generic but essential priors, including object permanence, scene geometry, physical plausibility, and the typical temporal structure of everyday tasks.

Egocentric and Exocentric Videos. Egocentric datasets such as EPIC-KITCHENS [72], Ego4D [118], and Ego-Exo4D [119] are distributionally closer to robot observations than generic web video. They expose the model to first-person interaction, hand-object occlusion, viewpoint changes, object-state transitions, and long-horizon task progress, which are precisely the observation patterns encountered by deployed manipulation systems and are difficult to obtain from other sources.

Viewed through the data pyramid, action-free data at the base provide scalable and diverse observational priors, while higher levels progressively add semantic, latent, and embodiment-specific action grounding. Trained world models in turn feed back into the pyramid by generating synthetic interaction data [155, 269] and by substituting for physical environments in policy evaluation [286, 161]. Unified systems such as Pelican-Unified [432] and Cosmos 3 [269] further demonstrate how heterogeneous supervision—including visual, language, future-generation, and action data—can be incorporated within a shared modeling framework.

Takeaway
Data application
Embodied foundation-model training recipes are shifting toward more heterogeneous data sources and substantially larger scales. Egocentric data are also playing an increasingly important role, alongside growing tolerance for noisy or lower-quality trajectories. Nevertheless, the optimal recipe remains unresolved, as the contribution of each data source has not been systematically isolated and robot-only models can still achieve strong performance. Across embodied brains (VLMs), VLAs, and WAMs, action-unlabeled data primarily preserve general capabilities or support latent-dynamics learning, whereas action-labeled data ground models in physical interaction and executable control.
8Challenges and Future Directions

This section organizes future directions for robot learning data around three questions: what to collect, how to collect it, and how to use it. First, what to collect concerns forms of interaction information that remain insufficiently represented in existing datasets. In particular, failure and recovery trajectories are needed to expose models to unsuccessful behaviors and corrective actions, while tactile data are required to characterize contact states, forces, material properties, and fine-grained interaction dynamics that cannot be fully inferred from visual observations alone.

Second, how to collect focuses on scalable and adaptive data acquisition. Future collection pipelines should reduce dependence on costly manual teleoperation, expand task and scene coverage, and allocate collection effort toward underrepresented, uncertain, or behaviorally informative regions of the data distribution.

Finally, how to use addresses the integration of heterogeneous data sources during model training. This includes aligning action representations across embodiments, determining effective data recipes for combining sources from different levels of the data pyramid, and transferring egocentric human interaction data to dexterous robotic manipulation. Together, these directions shift the focus from merely increasing dataset size toward improving the informational coverage, acquisition efficiency, and effective utilization of embodied data.

8.1Tactile Data for Contact-Rich Robot Learning

Despite the rapid expansion of large-scale robot learning datasets, most existing data resources remain centered on RGB-D observations, proprioceptive states, language instructions, and action trajectories, while tactile sensing is still not systematically incorporated as a standard modality [64, 173, 232]. This leaves a missing contact layer in the current data pyramid: visual observations can describe scene geometry and object motion, but they provide only indirect evidence of contact force, slip, local deformation, friction, and grasp stability. Although recent robot datasets such as RoboMIND 2.0 [142] and Humanoid Everyday [440] have begun to incorporate tactile signals, tactile sensing has not yet become a common or standardized component of robot data collection. Compared with RGB-D observations and action trajectories, tactile data remains less prevalent and is often constrained by sensor-specific hardware, inconsistent signal formats, limited task coverage, and weaker integration with long-horizon manipulation trajectories.

8.2Failure Data and Recovery-Centric Robot Learning

Most existing robot learning datasets remain strongly biased toward successful expert demonstrations, while failed, near-failure, and suboptimal trajectories are frequently discarded or only coarsely labeled. This success-centric data bias limits the exposure of learned policies to the diverse error states encountered during real-world deployment. A policy trained primarily on successful trajectories may learn how to complete a task under nominal conditions, but receive little supervision on how to recognize an impending failure, diagnose its cause, or recover after execution deviates from the expected trajectory. Failure data should therefore not be regarded merely as low-quality experience, but as an important source of supervision for failure awareness, causal diagnosis, and recovery behavior.

Existing efforts have begun to explore failure-centric data from several perspectives. Some works construct failure-oriented datasets or benchmarks for failure detection, diagnosis, explanation, and correction reasoning [228, 90, 235, 416, 271, 120]. Other studies explicitly augment training data with failure and recovery trajectories, enabling policies to learn corrective behaviors beyond nominal expert demonstrations [70, 215]. Meanwhile, imperfect and unsuccessful demonstrations have also been shown to contain reusable trajectory segments that can benefit policy learning when appropriately identified and filtered rather than simply discarded [367].

Nevertheless, failure data remains limited in scale and lacks systematic organization. Failures in robotic manipulation are highly heterogeneous and may arise from perception errors, inaccurate localization, inappropriate task planning, failed grasping, object slippage, excessive contact force, kinematic constraints, or accumulated errors over long-horizon execution. A binary success or failure label is insufficient to represent such temporal and causal structures. Future datasets should provide richer failure-centric annotations, including pre-failure context, failure onset, failure categories and causes, object and robot state changes, recovery actions, and final recovery outcomes. More importantly, future data collection pipelines should systematically preserve and curate failures encountered during robot execution rather than treating them as unusable trajectories. Building large-scale, diverse, and structured failure-recovery datasets may enable embodied agents not only to imitate successful behaviors, but also to understand, anticipate, and recover from their own mistakes.

8.3Scalable Data Collection Across Pyramid Layers

Egocentric data collection is expanding from first-person RGB video to diverse wearable sensors. Head-mounted AR/MR devices can record gaze, head pose, depth, and spatial context, while wrist cameras, hand-mounted IMUs, and instrumented gloves capture hand motion and hand-object interaction. Tactile patches, force sensors, and EMG further provide contact, grasp force, slip, and muscle activation signals that vision cannot directly observe [245, 171, 138, 178, 351, 378, 449]. Wearable interfaces such as DexUMI and RealDexUMI also show the potential to collect dexterous demonstrations that are more compatible with robot learning [384, 383].

However, many of these devices are not yet mature for large-scale deployment. Head and hand tracking remain sensitive to occlusion, motion blur, drift, and limited field of view, while gloves, tactile sensors, and EMG require careful fitting, calibration, and synchronization and may interfere with natural operation. Future systems should therefore become lighter, wireless, modular, and less intrusive, while supporting automatic cross-device calibration, on-device hand-object reconstruction, and standardized sensor metadata. These developments could extend the scale of emerging task-oriented ego datasets [157, 443, 205] without sacrificing the detailed motion and contact signals needed for robot learning.

8.4Cross-Embodiment State-Action Alignment

As real-robot datasets increasingly aggregate trajectories from heterogeneous platforms, a common data format alone does not ensure compatible supervision. To facilitate cross-embodiment aggregation, robot states and actions are often represented in Cartesian end-effector pose space rather than morphology-specific joint space, providing a more transferable interface across platforms. Yet end-effector poses may still be defined relative to the robot base, camera, local end-effector, or world coordinate frame, causing the same physical motion to correspond to different numerical representations across datasets [64]. Such inconsistencies can introduce conflicting supervision during joint training and substantially affect cross-embodiment transfer and data-scaling behavior [357, 412]. Effective state-action alignment therefore requires consistent geometric semantics, rather than merely a shared storage schema. Future datasets should record coordinate conventions, calibration parameters, and controller modes as first-class metadata. When reliable transformations are available, actions can be canonicalized into a shared reference frame, such as the camera, end-effector, or world frame, to support consistent aggregation across embodiments [382, 412].

8.5Egocentric Priors for Dexterous Hand Policy Learning

While egocentric human data provides an intuitive source of priors for dexterous manipulation [391, 238, 101, 30, 443], how to optimally leverage such priors for robot policy learning remains an open question. The human-to-robot gap is not merely visual, but also kinematic, morphological, and physical: human and robotic hands differ in joint topology, degrees of freedom, fingertip geometry, compliance, actuation, sensing, friction, and force limits. Consequently, retargeted motions that appear geometrically plausible may still violate robot-specific constraints or fail to produce stable contacts on real hardware. Embodiment-aware inpainting [384, 192] can reduce the visual gap by replacing human hands with robot renderings, but the synthesized interactions are often not physically grounded, potentially impairing the model’s ability to learn realistic hand-object interaction dynamics, which are crucial for dexterous manipulation [411, 383].

A promising direction is to treat egocentric data as a source of structured interaction priors rather than precise robot action labels. Human videos can provide transferable information about task intent, object affordances, grasp choices, contact sequences, tool-use strategies, and long-horizon manipulation structure, while robot data can ground these priors in embodiment-specific actions and contact dynamics. Morphology-conditioned policies, contact-centric representations, and uncertainty-aware retargeting may further help separate broadly transferable interaction knowledge from embodiment-dependent execution details.

8.6Data Recipes

As illustrated in Figure 3, the data recipes of embodied foundation models have gradually expanded across different layers of the data pyramid. Earlier models mainly combined real-robot data with general-purpose vision-language data, whereas more recent reports increasingly construct mixtures involving real-robot, simulation, egocentric, UMI-style, and general multimodal data. Rather than converging on a single composition, different models select and combine different subsets of these data sources according to their training pipelines. In particular, egocentric data has become an increasingly common component of large-scale pretraining, while general data continues to be widely used together with embodiment-specific and action-aligned data. This trend indicates a shift from relying on a dominant data source toward jointly exploiting multiple levels of the data pyramid.

However, the most effective data composition remains an open problem. The optimal proportions among general, egocentric, simulation, UMI-style, and real-robot data have not been established. Existing reports differ substantially in their source combinations, sampling ratios, and stage-wise allocation strategies. Precise, compute-matched ablations of individual datasets and data categories remain limited. The performance gains produced by the same data source across different model architectures, action representations, and training objectives are also insufficiently characterized. Future work should therefore investigate architecture-aware and stage-dependent data recipes, with systematic comparisons of fixed, curriculum-based, and adaptive mixture strategies. Such studies may help identify when each layer of the data pyramid is most useful and provide more reproducible principles for allocating data under constrained collection and training budgets.

9Conclusion

This work provides a data-centric synthesis of embodied intelligence and defines an embodied data pyramid organized primarily by scalability and robot alignment, with quality, diversity, reusability, and physical fidelity serving as complementary dimensions. From the apex to the base, the pyramid contains five data categories with distinct strengths and limitations. Real-robot data provides the most direct supervision for executable behavior and physical interaction, but remains costly and difficult to scale. UMI-style data supports portable real-world collection with explicit end-effector and gripper supervision, but requires calibration, retargeting, and embodiment-specific control. Egocentric and exocentric data captures diverse human activities, hand-object interactions, and long-horizon task structures, while being limited by the human-robot embodiment gap. Simulation data enables controllable and parallel generation of robot-oriented experience with privileged supervision, although its utility is constrained by environment coverage and the sim-to-real gap. General multimodal data offers broad semantic, perceptual, and reasoning priors at large scale, but provides limited grounding in robot actions, contacts, and physical outcomes.

Building on this taxonomy, we analyze recent embodied foundation models from the perspective of data recipes. We summarize action representations and their implications for data compatibility and cross-embodiment transfer, and examine how embodied brain models, vision-language-action models, and world-action models utilize heterogeneous data to support reasoning and planning, executable control, and predictive world modeling, respectively. Finally, we identify open challenges in data quality assessment, multimodal and action-space standardization, cross-embodiment reuse, human-to-robot and sim-to-real transfer, scalable real-world collection, failure and recovery data, and the design and evaluation of heterogeneous data mixtures. We hope this paper provides both a coherent foundation for understanding the embodied data landscape and a forward-looking roadmap for selecting, collecting, combining, and evaluating data for different capabilities, embodiments, and deployment settings.

References
bui [2025]	Builddotai/egocentric-100k · datasets at hugging face.https://huggingface.co/datasets/builddotai/Egocentric-100K, 2025.URL https://huggingface.co/datasets/builddotai/egocentric-100k.
Akinola et al. [2025]	Iretiayo Akinola, Jie Xu, Jan Carius, Dieter Fox, and Yashraj Narang.TacSL: A library for visuotactile sensor simulation and learning.IEEE Trans. Robot., 2025.URL https://arxiv.org/abs/2408.06506.
Ali et al. [2025]	Arslan Ali, Junjie Bai, Maciej Bala, Yogesh Balaji, Aaron Blakeman, Tiffany Cai, Jiaxin Cao, Tianshi Cao, Elizabeth Cha, Yu-Wei Chao, et al.World simulation with video foundation models for physical AI.arXiv preprint arXiv:2511.00062, 2025.URL https://arxiv.org/abs/2511.00062.
Alian et al. [2026]	Amirhosein Alian, Yongqiang Zhao, Shiyi Gu, Xuyang Zhang, Zhuo Chen, Christopher E Mower, Haitham Bou-Ammar, and Shan Luo.HapTile: a haptic-informed Vision-Tactile-Language-Action dataset for contact-rich imitation learning.arXiv preprint arXiv:2606.04825, 2026.URL https://arxiv.org/abs/2606.04825.
Allshire et al. [2026]	Arthur Allshire, Himanshu Gaurav Singh, Ritvik Singh, Adam Rashid, Hongsuk Choi, David McAllister, Justin Yu, Yiyuan Chen, Huang Huang, Pieter Abbeel, Xi Chen, Rocky Duan, Phillip Isola, Jitendra Malik, Fred Shentu, Guanya Shi, Philipp Wu, and Angjoo Kanazawa.Scalable behavior cloning with open data, training, and evaluation, 2026.URL https://arxiv.org/abs/2606.27375.
Antol et al. [2015]	Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C. Lawrence Zitnick, and Devi Parikh.VQA: Visual question answering.In IEEE/CVF Int. Conf. Comput. Vis., 2015.URL https://arxiv.org/abs/1505.00468.
Apanasevich et al. [2026]	I. Apanasevich, M. Artemyev, R. Babakyan, P. Fedotova, D. Grankin, E. Kupryashin, A. Misailidi, D. Nerus, et al.Green-VLA: Staged vision-language-action model for generalist robots.arXiv preprint arXiv:2602.00919, 2026.URL https://arxiv.org/abs/2602.00919.
Assran et al. [2025]	Mahmoud Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba Komeili, Matthew J. Muckley, Ammar Rizvi, et al.V-JEPA 2: self-supervised video models enable understanding, prediction and planning.arXiv preprint arXiv:2506.09985, 2025.URL https://arxiv.org/abs/2506.09985.
Azuma et al. [2022]	Daichi Azuma, Taiki Miyanishi, Shuhei Kurita, and Motoaki Kawanabe.ScanQA: 3D question answering for spatial scene understanding.In IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2022.URL https://arxiv.org/abs/2112.10482.
Bambach et al. [2015]	Sven Bambach, Stefan Lee, David J. Crandall, and Chen Yu.Lending a hand: detecting hands and recognizing activities in complex egocentric interactions.In IEEE/CVF Int. Conf. Comput. Vis., 2015.URL https://doi.org/10.1109/iccv.2015.226.
Banerjee et al. [2024]	Prithviraj Banerjee, Sindi Shkodrani, Pierre Moulon, Shreyas Hampali, Fan Zhang, Jade Fountain, Edward Miller, Selen Basol, Richard Newcombe, Robert Wang, Jakob Julian Engel, and Tomas Hodan.Introducing hot3d: an egocentric dataset for 3D hand and object tracking.arXiv preprint arXiv:2406.09598, 2024.URL https://arxiv.org/abs/2406.09598.
Bao et al. [2023]	Chen Bao, Helin Xu, Yuzhe Qin, and Xiaolong Wang.DexArt: Benchmarking generalizable dexterous manipulation with articulated objects.In IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2023.URL https://arxiv.org/abs/2305.05706.
Baruch et al. [2021]	Gilad Baruch, Zhuoyuan Chen, Afshin Dehghan, Tal Dimry, Yuri Feigin, Peter Fu, Thomas Gebauer, Brandon Joffe, Daniel Kurz, Arik Schwartz, and Elad Shulman.Arkitscenes: a diverse real-world dataset for 3D indoor scene understanding using mobile RGB-D data.In Adv. Neural Inf. Process. Syst., 2021.URL https://arxiv.org/abs/2111.08897.
Beattie et al. [2016]	Charles Beattie, Joel Z. Leibo, Denis Teplyashin, Tom Ward, Marcus Wainwright, Heinrich Küttler, Andrew Lefrancq, Simon Green, et al.DeepMind lab.arXiv preprint arXiv:1612.03801, 2016.URL https://arxiv.org/abs/1612.03801.
Bharadhwaj et al. [2024a]	Homanga Bharadhwaj, Roozbeh Mottaghi, Abhinav Gupta, and Shubham Tulsiani.Track2Act: Predicting point tracks from internet videos enables generalizable robot manipulation.In Eur. Conf. Comput. Vis., 2024a.URL https://arxiv.org/abs/2405.01527.
Bharadhwaj et al. [2024b]	Homanga Bharadhwaj, Jay Vakil, Mohit Sharma, Abhinav Gupta, Shubham Tulsiani, and Vikash Kumar.RoboAgent: generalization and efficiency in robot manipulation via semantic augmentations and action chunking.In IEEE Int. Conf. Robot. Autom., 2024b.URL https://arxiv.org/abs/2309.01918.
Bhat et al. [2026]	Vineet Bhat, Naman Patel, Prashanth Krishnamurthy, Ramesh Karri, and Farshad Khorrami.MapleGrasp: mask-guided feature pooling for language-driven efficient robotic grasping.In IEEE/CVF Winter Conf. Appl. Comput. Vis., 2026.URL https://arxiv.org/abs/2506.06535.
Bi et al. [2026a]	Hongzhe Bi, Hengkai Tan, Shenghao Xie, Zeyuan Wang, Shuhe Huang, Haitian Liu, Ruowen Zhao, Yao Feng, et al.Motus: a unified latent action world model.In IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2026a.URL https://arxiv.org/abs/2512.13030.
Bi et al. [2026b]	Hongzhe Bi, Lingxuan Wu, Tianwei Lin, Hengkai Tan, Zhizhong Su, Hang Su, and Jun Zhu.H-RDT: human manipulation enhanced bimanual robotic manipulation.In AAAI Conf. Artif. Intell., 2026b.URL https://doi.org/10.1609/aaai.v40i22.38875.
Bjorck et al. [2025]	Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, et al.GR00T n1: an open foundation model for generalist humanoid robots.arXiv preprint arXiv:2503.14734, 2025.URL https://arxiv.org/abs/2503.14734.
Black et al. [2025]	Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, et al.
𝜋
0
: a vision-language-action flow model for general robot control.In Robot. Sci. Syst., 2025.URL https://arxiv.org/abs/2410.24164.
Bock et al. [2024]	Marius Bock, Hilde Kuehne, Kristof Van Laerhoven, and Michael Moeller.WEAR: An outdoor sports dataset for wearable and egocentric activity recognition.Proc. ACM Interact. Mob. Wearable Ubiquitous Technol., 2024.URL https://arxiv.org/abs/2304.05088.
Brohan et al. [2023a]	Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, et al.RT-2: vision-language-action models transfer web knowledge to robotic control.In Conf. Robot Learn., 2023a.URL https://arxiv.org/abs/2307.15818.
Brohan et al. [2023b]	Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al.RT-1: robotics transformer for real-world control at scale.In Robot. Sci. Syst., 2023b.URL https://arxiv.org/abs/2212.06817.
Bruce et al. [2024]	Jake Bruce, Michael Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al.Genie: generative interactive environments.In Int. Conf. Mach. Learn., 2024.URL https://arxiv.org/abs/2402.15391.
Bu et al. [2025a]	Qingwen Bu, Jisong Cai, Li Chen, Xiuqi Cui, Yan Ding, Siyuan Feng, Shenyuan Gao, Xindong He, Xuan Hu, Xu Huang, et al.AgiBot world colosseo: a large-scale manipulation platform for scalable and intelligent embodied systems.In IEEE/RSJ Int. Conf. Intell. Robots Syst., 2025a.URL https://arxiv.org/abs/2503.06669.
Bu et al. [2025b]	Qingwen Bu, Yanting Yang, Jisong Cai, Shenyuan Gao, Guanghui Ren, Maoqing Yao, Ping Luo, and Hongyang Li.UniVLA: learning to act anywhere with task-centric latent actions.In Robot. Sci. Syst., 2025b.URL https://doi.org/10.15607/rss.2025.xxi.014.
Cai et al. [2026a]	Junhao Cai, Zetao Cai, Jiafei Cao, Yilun Chen, Zeyu He, Lei Jiang, Hang Li, Hengjie Li, Yang Li, Yufei Liu, et al.Internvla-a1: unifying understanding, generation and action for robotic manipulation.arXiv preprint arXiv:2601.02456, 2026a.URL https://arxiv.org/abs/2601.02456.
Cai et al. [2026b]	Rui Cai, Jun Guo, Xinze He, Piaopiao Jin, Jie Li, Bingxuan Lin, Futeng Liu, Wei Liu, et al.Xiaomi-robotics-0: An open-sourced vision-language-action model with real-time execution.arXiv preprint arXiv:2602.12684, 2026b.URL https://arxiv.org/abs/2602.12684.
Cai et al. [2025]	Xiongyi Cai, Ri-Zhao Qiu, Geng Chen, Lai Wei, Isabella Liu, Tianshu Huang, Xuxin Cheng, and Xiaolong Wang.In-n-on: scaling egocentric manipulation with in-the-wild and on-task data.arXiv preprint arXiv:2511.15704, 2025.URL https://arxiv.org/abs/2511.15704.
Cai et al. [2026c]	Zhongang Cai, Ruisi Wang, Chenyang Gu, Fanyi Pu, Junxiang Xu, Yubo Wang, Wanqi Yin, Zhitao Yang, et al.Scaling spatial intelligence with multimodal foundation models.In IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2026c.URL https://arxiv.org/abs/2511.13719.
Cao et al. [2021]	Hanwen Cao, Hao-Shu Fang, Wenhai Liu, and Cewu Lu.SuctionNet-1Billion: a large-scale benchmark for suction grasping.IEEE Robot. Autom. Lett., 2021.URL https://arxiv.org/abs/2103.12311.
Cao et al. [2025]	Ziang Cao, Zhaoxi Chen, Liang Pan, and Ziwei Liu.Physx-3D: Physical-grounded 3D asset generation.In Adv. Neural Inf. Process. Syst., 2025.URL https://arxiv.org/abs/2507.12465.
Cao et al. [2026a]	Ziang Cao, Fangzhou Hong, Zhaoxi Chen, Liang Pan, and Ziwei Liu.Physx-anything: Simulation-ready physical 3D assets from single image.In IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2026a.URL https://arxiv.org/abs/2511.13648.
Cao et al. [2026b]	Ziang Cao, Yinghao Liu, Haitian Li, Runmao Yao, Fangzhou Hong, Zhaoxi Chen, Liang Pan, and Ziwei Liu.PhysX-Omni: Unified simulation-ready physical 3D generation for rigid, deformable, and articulated objects.arXiv preprint arXiv:2605.21572, 2026b.URL https://arxiv.org/abs/2605.21572.
Cen et al. [2025]	Jun Cen, Chaohui Yu, Hangjie Yuan, Yuming Jiang, Siteng Huang, Jiayan Guo, Xin Li, Yibing Song, Hao Luo, Fan Wang, Deli Zhao, and Hao Chen.WorldVLA: towards autoregressive action world model.arXiv preprint arXiv:2506.21539, 2025.URL https://arxiv.org/abs/2506.21539.
Chandra et al. [2025]	Akshay L. Chandra, Iman Nematollahi, Chenguang Huang, Tim Welschehold, Wolfram Burgard, and Abhinav Valada.DiWA: diffusion policy adaptation with world models.In Conf. Robot Learn., 2025.URL https://arxiv.org/abs/2508.03645.
Chang et al. [2017]	Angel Chang, Angela Dai, Thomas Funkhouser, Maciej Halber, Matthias Nießner, Manolis Savva, Shuran Song, Andy Zeng, and Yinda Zhang.Matterport3D: learning from RGB-D data in indoor environments.In Int. Conf. 3D Vis., 2017.URL https://doi.org/10.1109/3dv.2017.00081.
Chang et al. [2015]	Angel X. Chang, Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qixing Huang, Zimo Li, Silvio Savarese, Manolis Savva, et al.ShapeNet: an information-rich 3D model repository.arXiv preprint arXiv:1512.03012, 2015.URL https://arxiv.org/abs/1512.03012.
Chavan et al. [2025]	Vivek Chavan, Yasmina Imgrund, Tung Dao, Sanwantri Bai, Bosong Wang, Ze Lu, Oliver Heimann, and Jörg Krüger.IndEgo: A dataset of industrial scenarios and collaborative work for egocentric assistants.In Adv. Neural Inf. Process. Syst., 2025.URL https://arxiv.org/abs/2511.19684.
Cheang et al. [2024]	Chi-Lam Cheang, Guangzeng Chen, Ya Jing, Tao Kong, Hang Li, Yifeng Li, Yuxiao Liu, Hongtao Wu, Jiafeng Xu, Yichu Yang, Hanbo Zhang, and Minzhao Zhu.GR-2: a generative video-language-action model with web-scale knowledge for robot manipulation.arXiv preprint arXiv:2410.06158, 2024.URL https://arxiv.org/abs/2410.06158.
Cheang et al. [2025]	Chilam Cheang, Sijin Chen, Zhongren Cui, Yingdong Hu, Liqun Huang, Tao Kong, Hang Li, Yifeng Li, et al.Gr-3 technical report.arXiv preprint arXiv:2507.15493, 2025.URL https://arxiv.org/abs/2507.15493.
Chen et al. [2026a]	Baijun Chen, Weijie Wan, Tianxing Chen, Xianda Guo, Congsheng Xu, Yuanyang Qi, Haojie Zhang, Longyan Wu, Tianling Xu, Zixuan Li, et al.Univtac: A unified simulation platform for visuo-tactile manipulation data generation, learning, and benchmarking.arXiv preprint arXiv:2602.10093, 2026a.
Chen et al. [2026b]	Changhe Chen, Quantao Yang, Xiaohao Xu, Nima Fazeli, and Olov Andersson.Visa-flow: accelerating robot skill learning via large-scale video semantic action flow.In IEEE Int. Conf. Robot. Autom., 2026b.URL https://arxiv.org/abs/2505.01288.
Chen et al. [2025a]	Hanzhi Chen, Boyang Sun, Anran Zhang, Marc Pollefeys, and Stefan Leutenegger.Vidbot: learning generalizable 3D actions from in-the-wild 2D human videos for zero-shot robotic manipulation.In IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2025a.URL https://doi.org/10.1109/cvpr52734.2025.02576.
Chen et al. [2024]	Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin.ShareGPT4V: improving large multi-modal models with better captions.In Eur. Conf. Comput. Vis., 2024.URL https://arxiv.org/abs/2311.12793.
Chen et al. [2026c]	Ronghan Chen, Yandan Yang, Zuojin Tang, Dongjie Huo, Tong Lin, Haoning Wu, Haoyun Liu, Yuzhi Chen, et al.ABot-M0.5: Unified mobility-and-manipulation world action model.arXiv preprint arXiv:2607.00678, 2026c.URL https://arxiv.org/abs/2607.00678.
Chen et al. [2023a]	Siwei Chen, Yiqing Xu, Cunjun Yu, Linfeng Li, Xiao Ma, Zhongwen Xu, and David Hsu.Daxbench: benchmarking deformable object manipulation with differentiable physics.In Int. Conf. Learn. Represent., 2023a.URL https://arxiv.org/abs/2210.13066.
Chen et al. [2025b]	Tianxing Chen, Zanxin Chen, Baijun Chen, Zijian Cai, Yibin Liu, Zixuan Li, Qiwei Liang, Xianliang Lin, Yiheng Ge, Zhenyu Gu, et al.Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation.arXiv preprint arXiv:2506.18088, 2025b.URL https://arxiv.org/abs/2506.18088.
Chen et al. [2026d]	Tianxing Chen, Yue Chen, Zixuan Li, Junyuan Tang, Kailun Su, Weijie Wan, Baijun Chen, Haoran Lu, Haowen Yan, Honghao Su, et al.Robodojo: A unified sim-and-real benchmark for comprehensive evaluation of generalist robot manipulation policies.arXiv preprint arXiv:2607.04434, 2026d.
Chen et al. [2026e]	Tianxing Chen, Yuran Wang, Mingleyang Li, Yan Qin, Hao Shi, Zixuan Li, Yifan Hu, Yingsheng Zhang, Kaixuan Wang, Yue Chen, et al.Rmbench: Memory-dependent robotic manipulation benchmark with insights into policy design.arXiv preprint arXiv:2603.01229, 2026e.
Chen et al. [2026f]	Xiaoyu Chen, Hangxing Wei, Pushi Zhang, Chuheng Zhang, Kaixin Wang, Yanjiang Guo, Rushuai Yang, Yucen Wang, Xinquan Xiao, Li Zhao, et al.Villa-x: enhancing latent action modeling in vision-language-action models.In Int. Conf. Learn. Represent., 2026f.URL https://arxiv.org/abs/2507.23682.
Chen et al. [2025c]	Xinyi Chen, Yilun Chen, Yanwei Fu, Ning Gao, Jiaya Jia, Weiyang Jin, Hao Li, Yao Mu, et al.InternVLA-M1: a spatially guided vision-language-action framework for generalist robot policy.arXiv preprint arXiv:2510.13778, 2025c.URL https://arxiv.org/abs/2510.13778.
Chen et al. [2025d]	Yi Chen, Yuying Ge, Weiliang Tang, Yizhuo Li, Yixiao Ge, Mingyu Ding, Ying Shan, and Xihui Liu.Moto: Latent motion token as the bridging language for learning robot manipulation from videos.In IEEE/CVF Int. Conf. Comput. Vis., 2025d.URL https://arxiv.org/abs/2412.04445.
Chen et al. [2026g]	Yi Chen, Yuying Ge, Yixiao Ge, Mingyu Ding, Bohao Li, Rui Wang, Ruifeng Xu, Ying Shan, and Xihui Liu.EgoPlan-Bench: benchmarking multimodal large language models for human-level planning.Int. J. Comput. Vis., 2026g.URL https://arxiv.org/abs/2312.06722.
Chen et al. [2022]	Yuanpei Chen, Tianhao Wu, Shengjie Wang, Xidong Feng, Jiechuang Jiang, Stephen Marcus McAleer, Yiran Geng, Hao Dong, Zongqing Lu, Song-Chun Zhu, and Yaodong Yang.Towards human-level bimanual dexterous manipulation with reinforcement learning.In Adv. Neural Inf. Process. Syst., 2022.URL https://arxiv.org/abs/2206.08686.
Chen et al. [2025e]	Zeren Chen, Zhelun Shi, Xiaoya Lu, Lehan He, Sucheng Qian, Enshen Zhou, Zhenfei Yin, Wanli Ouyang, Jing Shao, Yu Qiao, Cewu Lu, and Lu Sheng.RH20T-P: a primitive-level robotic manipulation dataset towards composable generalization agents in real-world scenarios.In IEEE/RSJ Int. Conf. Intell. Robots Syst., 2025e.URL https://arxiv.org/abs/2403.19622.
Chen et al. [2023b]	Zixi Chen, Shixin Zhang, Shan Luo, Fuchun Sun, and Bin Fang.Tacchi: A pluggable and low computational cost elastomer deformation simulator for optical tactile sensors.IEEE Robot. Autom. Lett., 2023b.URL https://arxiv.org/abs/2301.08343.
Cheng et al. [2024]	Xuxin Cheng, Jialong Li, Shiqi Yang, Ge Yang, and Xiaolong Wang.Open-television: teleoperation with immersive active visual feedback.In Conf. Robot Learn., 2024.URL https://arxiv.org/abs/2407.01512.
Cherepanov et al. [2026]	Egor Cherepanov, Nikita Kachaev, Alexey K. Kovalev, and Aleksandr I. Panov.Memory, benchmark & robots: a benchmark for solving complex tasks with reinforcement learning.In Int. Conf. Learn. Represent., 2026.URL https://doi.org/10.48550/arXiv.2502.10550.
Chernyadev et al. [2024]	Nikita Chernyadev, Nicholas Backshall, Xiao Ma, Yunfan Lu, Younggyo Seo, and Stephen James.BiGym: A demo-driven mobile bi-manual manipulation benchmark.In Conf. Robot Learn., 2024.URL https://arxiv.org/abs/2407.07788.
Chi et al. [2024]	Cheng Chi, Zhenjia Xu, Chuer Pan, Eric Cousineau, Benjamin Burchfiel, Siyuan Feng, Russ Tedrake, and Shuran Song.Universal manipulation interface: in-the-wild robot teaching without in-the-wild robots.In Robot. Sci. Syst., 2024.URL https://arxiv.org/abs/2402.10329.
Chu et al. [2026]	Meng Chu, Xuan Billy Zhang, Kevin Qinghong Lin, Lingdong Kong, Jize Zhang, Teng Tu, Weijian Ma, Ziqi Huang, et al.Agentic world modeling: Foundations, capabilities, laws, and beyond.arXiv preprint arXiv:2604.22748, 2026.URL https://arxiv.org/abs/2604.22748.
Collaboration et al. [2024]	OX-Embodiment Collaboration, Abby O’Neill, Abdul Rehman, Abhinav Gupta, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, et al.Open X-Embodiment: Robotic learning datasets and RT-X models.In IEEE Int. Conf. Robot. Autom., 2024.URL https://arxiv.org/abs/2310.08864.
Collins et al. [2022]	Jasmine Collins, Shubham Goel, Kenan Deng, Achleshwar Luthra, Leon Xu, Erhan Gundogdu, Xi Zhang, Tomas F. Yago Vicente, Thomas Dideriksen, Himanshu Arora, Matthieu Guillaumin, and Jitendra Malik.ABO: Dataset and benchmarks for real-world 3D object understanding.In IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2022.URL https://arxiv.org/abs/2110.06199.
Coumans and Bai [2016–2021]	Erwin Coumans and Yunfei Bai.PyBullet, a python module for physics simulation for games, robotics and machine learning.http://pybullet.org, 2016–2021.URL https://pybullet.org/wordpress/.
Da et al. [2025]	Longchao Da, Justin Turnau, Thirulogasankar Pranav Kutralingam, Alvaro Velasquez, Paulo Shakarian, and Hua Wei.A survey of sim-to-real methods in RL: progress, prospects and challenges with foundation models.arXiv preprint arXiv:2502.13187, 2025.URL https://arxiv.org/abs/2502.13187.
Dai et al. [2017]	Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner.ScanNet: richly-annotated 3D reconstructions of indoor scenes.In IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2017.URL https://arxiv.org/abs/1702.04405.
Dai et al. [2026a]	Weisheng Dai, Kai Lan, Jianyi Zhou, Bo Zhao, Xiu Su, Junwen Tong, Weili Guan, and Shuo Yang.Conla: contrastive latent action learning from human videos for robotic manipulation.arXiv preprint arXiv:2602.00557, 2026a.URL https://arxiv.org/abs/2602.00557.
Dai et al. [2025]	Yinpei Dai, Jayjun Lee, Nima Fazeli, and Joyce Chai.RACER: Rich language-guided failure recovery policies for imitation learning.In IEEE Int. Conf. Robot. Autom., 2025.URL https://arxiv.org/abs/2409.14674.
Dai et al. [2026b]	Yinpei Dai, Hongze Fu, Jayjun Lee, Yuejiang Liu, Haoran Zhang, Jianing Yang, Chelsea Finn, Nima Fazeli, and Joyce Chai.RoboMME: benchmarking and understanding memory for robotic generalist policies.In Int. Conf. Mach. Learn., 2026b.URL https://doi.org/10.48550/arXiv.2603.04639.
Damen et al. [2018]	Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Sanja Fidler, Antonino Furnari, Evangelos Kazakos, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, and Michael Wray.Scaling egocentric vision: the epic-kitchens dataset.In Eur. Conf. Comput. Vis., 2018.URL https://arxiv.org/abs/1804.02748.
Damen et al. [2022]	Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Antonino Furnari, Evangelos Kazakos, Jian Ma, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, and Michael Wray.Rescaling egocentric vision.Int. J. Comput. Vis., 2022.URL https://arxiv.org/abs/2006.13256.
Dang et al. [2026]	Ronghao Dang, Jiayan Guo, Bohan Hou, Sicong Leng, Kehan Li, Xin Li, Jiangpin Liu, Yunxuan Mao, et al.RynnBrain: open embodied foundation models.arXiv preprint arXiv:2602.14979, 2026.URL https://arxiv.org/abs/2602.14979.
Dasari et al. [2020]	Sudeep Dasari, Frederik Ebert, Stephen Tian, Suraj Nair, Bernadette Bucher, Karl Schmeckpeper, Siddharth Singh, Sergey Levine, and Chelsea Finn.RoboNet: large-scale multi-robot learning.In Conf. Robot Learn., 2020.URL https://arxiv.org/abs/1910.11215.
Deitke et al. [2022]	Matt Deitke, Eli VanderBilt, Alvaro Herrasti, Luca Weihs, Jordi Salvador, Kiana Ehsani, Winson Han, Eric Kolve, Ali Farhadi, Aniruddha Kembhavi, and Roozbeh Mottaghi.ProcTHOR: large-scale embodied AI using procedural generation.In Adv. Neural Inf. Process. Syst., 2022.URL https://doi.org/10.52202/068431-0433.
Deitke et al. [2023a]	Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, et al.Objaverse-xl: a universe of 10m+ 3D objects.In Adv. Neural Inf. Process. Syst., 2023a.URL https://doi.org/10.52202/075280-1554.
Deitke et al. [2023b]	Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi.Objaverse: a universe of annotated 3D objects.In IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2023b.URL https://doi.org/10.1109/cvpr52729.2023.01263.
Deitke et al. [2025]	Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, et al.Molmo and PixMo: Open weights and open data for state-of-the-art vision-language models.In IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2025.URL https://arxiv.org/abs/2409.17146.
Deng et al. [2025]	Shengliang Deng, Mi Yan, Songlin Wei, Haixin Ma, Yuxin Yang, Jiayi Chen, Zhiqi Zhang, Taoyu Yang, et al.GraspVLA: a grasping foundation model pre-trained on billion-scale synthetic action data.In Conf. Robot Learn., 2025.URL https://arxiv.org/abs/2505.03233.
Deng and Zhou [2026]	Yufan Deng and Daquan Zhou.HumanNet: Scaling human-centric video learning to one million hours.arXiv preprint arXiv:2605.06747, 2026.URL https://arxiv.org/abs/2605.06747.
Deng et al. [2026]	Yufan Deng, Zilin Pan, Hongyu Zhang, Xiaojie Li, Ruoqing Hu, Yufei Ding, Yiming Zou, Yan Zeng, and Daquan Zhou.Rethinking video generation model for the embodied world.In Int. Conf. Mach. Learn., 2026.URL https://arxiv.org/abs/2601.15282.
Depierre et al. [2018]	Amaury Depierre, Emmanuel Dellandréa, and Liming Chen.Jacquard: a large scale dataset for robotic grasp detection.In IEEE/RSJ Int. Conf. Intell. Robots Syst., 2018.URL https://doi.org/10.1109/iros.2018.8593950.
Deshpande et al. [2026]	Abhay Deshpande, Maya Guru, Rose Hendrix, Snehal Jauhri, Ainaz Eftekhar, Rohun Tripathi, Max Argus, Jordi Salvador, et al.MolmoB0T: large-scale simulation enables zero-shot manipulation.arXiv preprint arXiv:2603.16861, 2026.URL https://arxiv.org/abs/2603.16861.
Dessalene et al. [2026]	Eadom Dessalene, Botao He, Michael Maynord, Yonatan Tussa, Pavan Mantripragada, Yianni Karabati, Nirupam Roy, and Yiannis Aloimonos.Feel (force-enhanced egocentric learning): a dataset for physical action understanding.arXiv preprint arXiv:2603.15847, 2026.URL https://arxiv.org/abs/2603.15847.
Ding et al. [2025]	Runyu Ding, Yuzhe Qin, Jiyue Zhu, Chengzhe Jia, Shiqi Yang, Ruihan Yang, Xiaojuan Qi, and Xiaolong Wang.Bunny-VisionPro: Real-time bimanual dexterous teleoperation for imitation learning.In IEEE/RSJ Int. Conf. Intell. Robots Syst., 2025.URL https://arxiv.org/abs/2407.03162.
Downs et al. [2022]	Laura Downs, Anthony Francis, Nate Koenig, Brandon Kinman, Ryan Hickman, Krista Reymann, Thomas B. McHugh, and Vincent Vanhoucke.Google scanned objects: A high-quality dataset of 3D scanned household items.In IEEE Int. Conf. Robot. Autom., 2022.URL https://arxiv.org/abs/2204.11918.
Driess et al. [2023]	Danny Driess, Fei Xia, Mehdi S. M. Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, et al.PaLM-E: an embodied multimodal language model.In Int. Conf. Mach. Learn., 2023.URL https://arxiv.org/abs/2303.03378.
Du et al. [2023]	Yilun Du, Sherry Yang, Bo Dai, Hanjun Dai, Ofir Nachum, Joshua B. Tenenbaum, Dale Schuurmans, and Pieter Abbeel.Learning universal policies via text-guided video generation.In Adv. Neural Inf. Process. Syst., 2023.URL https://arxiv.org/abs/2302.00111.
Duan et al. [2025]	Jiafei Duan, Wilbert Pumacay, Nishanth Kumar, Yi Ru Wang, Shulin Tian, Wentao Yuan, Ranjay Krishna, Dieter Fox, Ajay Mandlekar, and Yijie Guo.Aha: a vision-language-model for detecting and reasoning over failures in robotic manipulation.In Int. Conf. Learn. Represent., 2025.URL https://arxiv.org/abs/2410.00371.
Ebert et al. [2022]	Frederik Ebert, Yanlai Yang, Karl Schmeckpeper, Bernadette Bucher, Georgios Georgakis, Kostas Daniilidis, Chelsea Finn, and Sergey Levine.Bridge data: Boosting generalization of robotic skills with cross-domain datasets.In Robot. Sci. Syst., 2022.URL https://arxiv.org/abs/2109.13396.
Eppner et al. [2021]	Clemens Eppner, Arsalan Mousavian, and Dieter Fox.Acronym: a large-scale grasp dataset based on simulation.In IEEE Int. Conf. Robot. Autom., 2021.URL https://arxiv.org/abs/2011.09584.
Fan et al. [2023]	Zicong Fan, Omid Taheri, Dimitrios Tzionas, Muhammed Kocabas, Manuel Kaufmann, Michael J. Black, and Otmar Hilliges.Arctic: a dataset for dexterous bimanual hand-object manipulation.In IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2023.URL https://arxiv.org/abs/2204.13662.
Fang et al. [2020]	Hao-Shu Fang, Chenxi Wang, Minghao Gou, and Cewu Lu.GraspNet-1Billion: A large-scale benchmark for general object grasping.In IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2020.URL https://doi.org/10.1109/cvpr42600.2020.01146.
Fang et al. [2024]	Hao-Shu Fang, Hongjie Fang, Zhenyu Tang, Jirong Liu, Chenxi Wang, Junbo Wang, Haoyi Zhu, and Cewu Lu.RH20T: A comprehensive robotic dataset for learning diverse skills in one-shot.In IEEE Int. Conf. Robot. Autom., 2024.URL https://arxiv.org/abs/2307.00595.
Fang et al. [2025]	Haoquan Fang, Markus Grotz, Wilbert Pumacay, Yi Ru Wang, Dieter Fox, Ranjay Krishna, and Jiafei Duan.SAM2Act: integrating visual foundation model with a memory architecture for robotic manipulation.In Int. Conf. Mach. Learn., 2025.URL https://arxiv.org/abs/2501.18564.
Fang et al. [2026]	Haoquan Fang, Jiafei Duan, Donovan Clay, Sam Wang, Shuo Liu, Weikai Huang, Xiang Fan, Wei-Chuan Tsai, Shirui Chen, Yi Ru Wang, et al.MolmoAct2: action reasoning models for real-world deployment.arXiv preprint arXiv:2605.02881, 2026.URL https://arxiv.org/abs/2605.02881.
Fourier ActionNet Team [2025]	Yao Mu Fourier ActionNet Team.ActionNet: a dataset for dexterous bimanual manipulation, 2025.URL https://action-net.org.
Freeman et al. [2021]	C. Daniel Freeman, Erik Frey, Anton Raichuk, Sertan Girgin, Igor Mordatch, and Olivier Bachem.Brax: a differentiable physics engine for large scale rigid body simulation.In Adv. Neural Inf. Process. Syst., 2021.URL https://arxiv.org/abs/2106.13281.
Fu et al. [2021]	Huan Fu, Bowen Cai, Lin Gao, Ling-Xiao Zhang, Jiaming Wang, Cao Li, Qixun Zeng, Chengyue Sun, Rongfei Jia, Binqiang Zhao, and Hao Zhang.3D-front: 3D furnished rooms with layouts and semantics.In IEEE/CVF Int. Conf. Comput. Vis., 2021.URL https://arxiv.org/abs/2011.09127.
Fu et al. [2025]	Yankai Fu, Ning Chen, Junkai Zhao, Shaozhe Shan, Guocai Yao, Pengwei Wang, Zhongyuan Wang, and Shanghang Zhang.Metis: multi-source egocentric training for integrated dexterous vision-language-action model.arXiv preprint arXiv:2511.17366, 2025.URL https://arxiv.org/abs/2511.17366.
Fu et al. [2024a]	Zipeng Fu, Qingqing Zhao, Qi Wu, Gordon Wetzstein, and Chelsea Finn.HumanPlus: Humanoid shadowing and imitation from humans.In Conf. Robot Learn., 2024a.URL https://arxiv.org/abs/2406.10454.
Fu et al. [2024b]	Zipeng Fu, Tony Z Zhao, and Chelsea Finn.Mobile ALOHA: learning bimanual mobile manipulation with low-cost whole-body teleoperation.In Conf. Robot Learn., 2024b.URL https://arxiv.org/abs/2401.02117.
Fujii et al. [2024]	Ryo Fujii, Masashi Hatano, Hideo Saito, and Hiroki Kajita.Egosurgery-phase: a dataset of surgical phase recognition from egocentric open surgery videos.In Int. Conf. Med. Image Comput. Comput.-Assist. Interv., 2024.URL https://arxiv.org/abs/2405.19644.
Gan et al. [2021]	Chuang Gan, Jeremy Schwartz, Seth Alter, Damian Mrowca, Martin Schrimpf, James Traer, Julian De Freitas, Jonas Kubilius, et al.ThreeDWorld: a platform for interactive multi-modal physical simulation.In Adv. Neural Inf. Process. Syst., 2021.URL https://arxiv.org/abs/2007.04954.
Gao et al. [2017]	Jiyang Gao, Chen Sun, Zhenheng Yang, and Ram Nevatia.Tall: temporal activity localization via language query.In IEEE/CVF Int. Conf. Comput. Vis., 2017.URL https://doi.org/10.1109/iccv.2017.563.
Gao et al. [2025]	Ning Gao, Yilun Chen, Shuai Yang, Xinyi Chen, Yang Tian, Hao Li, Haifeng Huang, Hanqing Wang, Tai Wang, and Jiangmiao Pang.GENMANIP: LLM-driven simulation for generalizable instruction-following manipulation.In IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2025.URL https://arxiv.org/abs/2506.10966.
Gao et al. [2022]	Ruohan Gao, Zilin Si, Yen-Yu Chang, Samuel Clarke, Jeannette Bohg, Li Fei-Fei, Wenzhen Yuan, and Jiajun Wu.ObjectFolder 2.0: A multisensory object dataset for Sim2Real transfer.In IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2022.URL https://arxiv.org/abs/2204.02389.
Gao et al. [2026]	Shenyuan Gao, William Liang, Kaiyuan Zheng, Ayaan Malik, Seonghyeon Ye, Sihyun Yu, Wei-Cheng Tseng, Yuzhu Dong, Kaichun Mo, Chen-Hsuan Lin, et al.DreamDojo: a generalist robot world model from large-scale human videos.arXiv preprint arXiv:2602.06949, 2026.URL https://arxiv.org/abs/2602.06949.
Garcia-Hernando et al. [2018]	Guillermo Garcia-Hernando, Shanxin Yuan, Seungryul Baek, and Tae-Kyun Kim.First-person hand action benchmark with RGB-D videos and 3D hand pose annotations.In IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2018.URL https://doi.org/10.1109/cvpr.2018.00050.
Gemini Robotics Team et al. [2025]	Gemini Robotics Team, Krzysztof Choromanski, Coline Devin, Yilun Du, Debidatta Dwibedi, Ruiqi Gao, Abhishek Jindal, Thomas Kipf, et al.Evaluating gemini robotics policies in a Veo world simulator.arXiv preprint arXiv:2512.10675, 2025.URL https://arxiv.org/abs/2512.10675.
Genesis-Embodied-AI [2024]	Genesis-Embodied-AI.Genesis: a generative and universal physics engine for robotics and beyond.https://github.com/Genesis-Embodied-AI/Genesis, 2024.URL https://genesis-embodied-ai.github.io/.
Geng et al. [2025]	Haoran Geng, Feishi Wang, Songlin Wei, Yuyang Li, Bangjun Wang, Boshi An, Charlie Tianyue Cheng, Haozhe Lou, et al.RoboVerse: towards a unified platform, dataset and benchmark for scalable and generalizable robot learning.In Robot. Sci. Syst., 2025.URL https://doi.org/10.48550/arXiv.2504.18904.
GigaWorld Team et al. [2025]	GigaWorld Team, Angen Ye, Boyuan Wang, Chaojun Ni, Guan Huang, Guosheng Zhao, Haoyun Li, Jiagang Zhu, Kerui Li, Mengyuan Xu, et al.GigaWorld-0: world models as data engine to empower embodied AI.arXiv preprint arXiv:2511.19861, 2025.URL https://arxiv.org/abs/2511.19861.
Gong et al. [2023]	Ran Gong, Jiangyong Huang, Yizhou Zhao, Haoran Geng, Xiaofeng Gao, Qingyang Wu, Wensi Ai, Ziheng Zhou, Demetri Terzopoulos, Song-Chun Zhu, Baoxiong Jia, and Siyuan Huang.ARNOLD: a benchmark for language-grounded task learning with continuous states in realistic 3D scenes.In IEEE/CVF Int. Conf. Comput. Vis., 2023.URL https://arxiv.org/abs/2304.04321.
Gong et al. [2026]	Zeying Gong, Rong Li, Tianshuai Hu, Ronghe Qiu, Lingdong Kong, Lingfeng Zhang, Guoyang Zhao, Yiyi Ding, and Junwei Liang.Stairway to success: An online floor-aware zero-shot object-goal navigation framework via LLM-driven coarse-to-fine exploration.IEEE Robot. Autom. Lett., 2026.URL https://arxiv.org/abs/2505.23019.
Google DeepMind [2025]	Google DeepMind.Genie 3: a new frontier for world models.https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/, 2025.URL https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/.
Grauman et al. [2022]	Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, et al.Ego4D: around the world in 3,000 hours of egocentric video.In IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2022.URL https://arxiv.org/abs/2110.07058.
Grauman et al. [2025]	Kristen Grauman, Andrew Westbury, Lorenzo Torresani, Kris Kitani, Jitendra Malik, Triantafyllos Afouras, Kumar Ashutosh, Vijay Baiyya, et al.Ego-Exo4D: Understanding skilled human activity from first- and third-person perspectives.Int. J. Comput. Vis., 2025.URL https://arxiv.org/abs/2311.18259.
Grislain et al. [2026]	Clemence Grislain, Hamed Rahimi, Olivier Sigaud, and Mohamed Chetouani.I-failsense: towards general robotic failure detection with vision-language models.In IEEE Int. Conf. Robot. Autom., 2026.URL https://arxiv.org/abs/2509.16072.
Gu et al. [2023]	Jiayuan Gu, Fanbo Xiang, Xuanlin Li, Zhan Ling, Xiqiang Liu, Tongzhou Mu, Yihe Tang, Stone Tao, et al.ManiSkill2: a unified benchmark for generalizable manipulation skills.In Int. Conf. Learn. Represent., 2023.URL https://arxiv.org/abs/2302.04659.
Gupta et al. [2019a]	Abhishek Gupta, Vikash Kumar, Corey Lynch, Sergey Levine, and Karol Hausman.Relay policy learning: solving long-horizon tasks via imitation and reinforcement learning.In Conf. Robot Learn., 2019a.URL http://arxiv.org/abs/1910.11956.
Gupta et al. [2019b]	Agrim Gupta, Piotr Dollar, and Ross Girshick.LVIS: a dataset for large vocabulary instance segmentation.In IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2019b.URL https://arxiv.org/abs/1908.03195.
Ha and Schmidhuber [2018]	David Ha and Jürgen Schmidhuber.Recurrent world models facilitate policy evolution.In Adv. Neural Inf. Process. Syst., 2018.URL https://worldmodels.github.io/.
Ha et al. [2024]	Huy Ha, Yihuai Gao, Zipeng Fu, Jie Tan, and Shuran Song.UMI on legs: making manipulation policies mobile with manipulation-centric whole-body controllers.In Conf. Robot Learn., 2024.URL https://arxiv.org/abs/2407.10353.
Hafner et al. [2019]	Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson.Learning latent dynamics for planning from pixels.In Int. Conf. Mach. Learn., 2019.URL https://arxiv.org/abs/1811.04551.
Hafner et al. [2020]	Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi.Dream to control: learning behaviors by latent imagination.In Int. Conf. Learn. Represent., 2020.URL https://arxiv.org/abs/1912.01603.
Hafner et al. [2025]	Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap.Mastering diverse control tasks through world models.Nature, 2025.URL https://arxiv.org/abs/2301.04104.
Han et al. [2025]	Songhao Han, Boxiang Qiu, Yue Liao, Siyuan Huang, Chen Gao, Shuicheng Yan, and Si Liu.RoboCerebra: a large-scale benchmark for long-horizon robotic manipulation evaluation.In Adv. Neural Inf. Process. Syst., 2025.URL https://doi.org/10.48550/arXiv.2506.06677.
Handa et al. [2020]	Ankur Handa, Karl Van Wyk, Wei Yang, Jacky Liang, Yu-Wei Chao, Qian Wan, Stan Birchfield, Nathan Ratliff, and Dieter Fox.DexPilot: Vision-based teleoperation of dexterous robotic hand-arm system.In IEEE Int. Conf. Robot. Autom., 2020.URL https://arxiv.org/abs/1910.03135.
Hao et al. [2025]	Xiaoshuai Hao, Yingbo Tang, Lingfeng Zhang, Yanbiao Ma, Yunfeng Diao, Ziyu Jia, Wenbo Ding, Hangjun Ye, and Long Chen.RoboAfford++: a generative AI-enhanced dataset for multimodal affordance learning in robotic manipulation and navigation.arXiv preprint arXiv:2511.12436, 2025.URL https://arxiv.org/abs/2511.12436.
He et al. [2024]	Tairan He, Zhengyi Luo, Xialin He, Wenli Xiao, Chong Zhang, Weinan Zhang, Kris Kitani, Changliu Liu, and Guanya Shi.OmniH2O: Universal and dexterous human-to-humanoid whole-body teleoperation and learning.In Conf. Robot Learn., 2024.URL https://arxiv.org/abs/2406.08858.
Hendricks et al. [2017]	Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell.Localizing moments in video with natural language.In IEEE/CVF Int. Conf. Comput. Vis., 2017.URL https://doi.org/10.1109/iccv.2017.618.
Heo et al. [2025]	Minho Heo, Youngwoon Lee, Doohyun Lee, and Joseph J Lim.FurnitureBench: Reproducible real-world benchmark for long-horizon complex manipulation.Int. J. Robot. Res., 2025.URL https://arxiv.org/abs/2305.12821.
Hoque et al. [2021]	Ryan Hoque, Ashwin Balakrishna, Ellen Novoseller, Albert Wilcox, Daniel S Brown, and Ken Goldberg.Thriftydagger: budget-aware novelty and risk gating for interactive imitation learning.In Conf. Robot Learn., 2021.URL https://arxiv.org/abs/2109.08273.
Hoque et al. [2022]	Ryan Hoque, Lawrence Yunliang Chen, Satvik Sharma, Karthik Dharmarajan, Brijen Thananjeyan, Pieter Abbeel, and Ken Goldberg.Fleet-DAgger: interactive robot fleet learning with scalable human supervision.In Conf. Robot Learn., 2022.URL https://arxiv.org/abs/2206.14349.
Hoque et al. [2024]	Ryan Hoque, Ajay Mandlekar, Caelan Garrett, Ken Goldberg, and Dieter Fox.IntervenGen: interventional data generation for robust and data-efficient robot imitation learning.In IEEE/RSJ Int. Conf. Intell. Robots Syst., 2024.URL https://arxiv.org/abs/2405.01472.
Hoque et al. [2026]	Ryan Hoque, Peide Huang, David J. Yoon, Mouli Sivapurapu, and Jian Zhang.Egodex: learning dexterous manipulation from large-scale egocentric video.In Int. Conf. Learn. Represent., 2026.URL https://arxiv.org/abs/2505.11709.
Hou et al. [2025a]	Bohan Hou, Haoqiang Lin, Xuemeng Song, Haokun Wen, Meng Liu, Yupeng Hu, and Xiangyu Zhao.Fire: enhancing mllms with fine-grained context learning for complex image retrieval.In Int. ACM SIGIR Conf. Res. Dev. Inf. Retr., 2025a.URL https://doi.org/10.1145/3726302.3729979.
Hou et al. [2026a]	Bohan Hou, Jiuning Gu, Jiayan Guo, Ronghao Dang, Sicong Leng, Xin Li, Xuemeng Song, and Jianfei Yang.Interlv-search: benchmarking interleaved multimodal agentic search.arXiv preprint arXiv:2605.07510, 2026a.URL https://arxiv.org/abs/2605.07510.
Hou et al. [2026b]	Bohan Hou, Gen Li, Jindou Jia, Tuo An, Xinying Guo, Sicong Leng, Haoran Geng, Yanjie Ze, Tatsuya Harada, Philip Torr, et al.World model for robot learning: a comprehensive survey.arXiv preprint arXiv:2605.00080, 2026b.URL https://arxiv.org/abs/2605.00080.
Hou et al. [2025b]	Chengkai Hou, Kun Wu, Jiaming Liu, Zhengping Che, Di Wu, Fei Liao, Guangrun Li, Jingyang He, Qiuxuan Feng, Zhao Jin, et al.RoboMind 2.0: a multimodal, bimanual mobile manipulation dataset for generalizable embodied intelligence.arXiv preprint arXiv:2512.24653, 2025b.URL https://arxiv.org/abs/2512.24653.
Hu et al. [2023]	Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, and Gianluca Corrado.GAIA-1: a generative world model for autonomous driving.arXiv preprint arXiv:2309.17080, 2023.URL https://arxiv.org/abs/2309.17080.
Hua et al. [2024]	Pu Hua, Minghuan Liu, Annabella Macaluso, Yunfeng Lin, Weinan Zhang, Huazhe Xu, and Lirui Wang.GenSim2: Scaling robot data generation with multi-modal and reasoning LLMs.In Conf. Robot Learn., 2024.URL https://arxiv.org/abs/2410.03645.
Hua et al. [2026]	Qianxi Hua, Xinyue Li, Zheng Yan, Yang Li, Chi Zhang, Yongyao Li, and Yufei Liu.VTouch++: a multimodal dataset with vision-based tactile enhancement for bimanual manipulation.arXiv preprint arXiv:2604.20444, 2026.URL https://arxiv.org/abs/2604.20444.
Huang et al. [2025]	Siyuan Huang, Liliang Chen, Pengfei Zhou, Shengcong Chen, Zhengkai Jiang, Yue Hu, Yue Liao, Peng Gao, Hongsheng Li, Maoqing Yao, and Guanghui Ren.EnerVerse: envisioning embodied future space for robotics manipulation.In Adv. Neural Inf. Process. Syst., 2025.URL https://doi.org/10.48550/arXiv.2501.01895.
Huang et al. [2026]	Weikai Huang, Jieyu Zhang, Sijun Li, Taoyang Jia, Jiafei Duan, Yunqian Cheng, Jaemin Cho, Matthew Wallingford, Rustin Soraki, Chris Dongjoo Kim, Shuo Liu, et al.Wilddet3d: Scaling promptable 3d detection in the wild.arXiv preprint arXiv:2604.08626, 2026.URL https://arxiv.org/abs/2604.08626.
Huang et al. [2024]	Yifei Huang, Guo Chen, Jilan Xu, Mingfang Zhang, Lijin Yang, Baoqi Pei, Hongjie Zhang, Lu Dong, Yali Wang, Limin Wang, and Yu Qiao.EgoExoLearn: A dataset for bridging asynchronous ego- and exo-centric view of procedural activities in real world.In IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2024.URL https://arxiv.org/abs/2403.16182.
Huang et al. [2021]	Zhiao Huang, Yuanming Hu, Tao Du, Siyuan Zhou, Hao Su, Joshua B. Tenenbaum, and Chuang Gan.Plasticinelab: a soft-body manipulation benchmark with differentiable physics.In Int. Conf. Learn. Represent., 2021.URL https://arxiv.org/abs/2104.03311W.
Intelligence et al. [2025]	Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al.
𝜋
0.5
: a vision-language-action model with open-world generalization.In Conf. Robot Learn., 2025.URL https://arxiv.org/abs/2504.16054.
Intelligence et al. [2026]	Physical Intelligence, Bo Ai, Ali Amin, Raichelle Aniceto, Ashwin Balakrishna, Greg Balke, Kevin Black, George Bokinsky, et al.
𝜋
0.7
: a steerable generalist robotic foundation model with emergent capabilities.arXiv preprint arXiv:2604.15483, 2026.URL https://arxiv.org/abs/2604.15483.
James et al. [2019]	Stephen James, Marc Freese, and Andrew J. Davison.PyRep: bringing v-rep to deep robot learning.arXiv preprint arXiv:1906.11176, 2019.URL https://arxiv.org/abs/1906.11176.
James et al. [2020]	Stephen James, Zicong Ma, David Rovick Arrojo, and Andrew J. Davison.Rlbench: the robot learning benchmark & learning environment.IEEE Robot. Autom. Lett., 2020.URL http://arxiv.org/abs/1909.12271.
Jang et al. [2022]	Eric Jang, Alex Irpan, Mohi Khansari, Daniel Kappler, Frederik Ebert, Corey Lynch, Sergey Levine, and Chelsea Finn.Bc-z: zero-shot task generalization with robotic imitation learning.In Conf. Robot Learn., 2022.URL https://arxiv.org/abs/2202.02005.
Jang et al. [2025]	Joel Jang, Seonghyeon Ye, Zongyu Lin, Jiannan Xiang, Johan Bjorck, Yu Fang, Fengyuan Hu, Spencer Huang, et al.Dreamgen: unlocking generalization in robot learning through video world models.In Conf. Robot Learn., 2025.URL https://arxiv.org/abs/2505.12705.
Jangir et al. [2026]	Yash Jangir, Yidi Zhang, Pang-Chi Lo, Kashu Yamazaki, Chenyu Zhang, Kuan-Hsun Tu, Tsung-Wei Ke, Lei Ke, Yonatan Bisk, and Katerina Fragkiadaki.Robotarena 
∞
: scalable robot benchmarking via real-to-sim translation.In Int. Conf. Learn. Represent., 2026.URL https://arxiv.org/abs/2510.23571.
Jawaid and Xiang [2025]	Ahad Jawaid and Yu Xiang.OpenEgo: a large-scale multimodal egocentric dataset for dexterous manipulation.arXiv preprint arXiv:2509.05513, 2025.URL https://arxiv.org/abs/2509.05513.
Ji et al. [2025]	Yuheng Ji, Huajie Tan, Jiayu Shi, Xiaoshuai Hao, Yuan Zhang, Hengyuan Zhang, Pengwei Wang, Mengdi Zhao, et al.RoboBrain: a unified brain model for robotic manipulation from abstract to concrete.In IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2025.URL https://doi.org/10.1109/cvpr52734.2025.00168.
Jia et al. [2022]	Baoxiong Jia, Ting Lei, Song-Chun Zhu, and Siyuan Huang.EgoTaskQA: Understanding human tasks in egocentric videos.In Adv. Neural Inf. Process. Syst., 2022.URL https://arxiv.org/abs/2210.03929.
Jiang et al. [2026a]	Chenxi Jiang, Chuhao Zhou, and Jianfei Yang.REI-Bench: Can embodied agents understand vague human instructions in task planning?In Int. Conf. Learn. Represent., 2026a.URL https://arxiv.org/abs/2505.10872.
Jiang et al. [2026b]	Feng Jiang, Yang Chen, Kyle Xu, Yuchen Liu, Haifeng Wang, Zhenhao Shen, Jasper Lu, Shengze Huang, Yuanfei Wang, Chen Xie, and Ruihai Wu.RoboWM-Bench: a benchmark for evaluating world models in robotic manipulation.arXiv preprint arXiv:2604.19092, 2026b.URL https://arxiv.org/abs/2604.19092.
Jiang et al. [2025a]	Tao Jiang, Tianyuan Yuan, Yicheng Liu, Chenhao Lu, Jianning Cui, Xiao Liu, Shuiqi Cheng, Jiyang Gao, Huazhe Xu, and Hang Zhao.Galaxea open-world dataset and G0 dual-system VLA model.arXiv preprint arXiv:2509.00576, 2025a.URL https://arxiv.org/abs/2509.00576.
Jiang et al. [2026c]	Yuming Jiang, Siteng Huang, Shengke Xue, Yaxi Zhao, Jun Cen, Sicong Leng, Kehan Li, Jiayan Guo, et al.RynnVLA-001: using human demonstrations to improve robot manipulation.In IEEE Int. Conf. Robot. Autom., 2026c.URL https://arxiv.org/abs/2509.15212.
Jiang et al. [2023]	Yunfan Jiang, Agrim Gupta, Zichen Zhang, Guanzhi Wang, Yongqiang Dou, Yanjun Chen, Li Fei-Fei, Anima Anandkumar, Yuke Zhu, and Linxi Fan.VIMA: General robot manipulation with multimodal prompts.In Int. Conf. Mach. Learn., 2023.URL https://arxiv.org/abs/2210.03094.
Jiang et al. [2025b]	Zhennan Jiang, Kai Liu, Yuxin Qin, Shuai Tian, Yupeng Zheng, Mingcai Zhou, Chao Yu, Haoran Li, and Dongbin Zhao.World4RL: diffusion world models for policy refinement with reinforcement learning for robotic manipulation.arXiv preprint arXiv:2509.19080, 2025b.URL https://arxiv.org/abs/2509.19080.
Jiang et al. [2025c]	Zhenyu Jiang, Yuqi Xie, Kevin Lin, Zhenjia Xu, Weikang Wan, Ajay Mandlekar, Linxi Fan, and Yuke Zhu.DexMimicGen: automated data generation for bimanual dexterous manipulation via imitation learning.In IEEE Int. Conf. Robot. Autom., 2025c.URL https://arxiv.org/abs/2410.24185.
Juelg et al. [2026]	Tobias Juelg, Seongjin Bien, Simon Hilber, Yannik Blei, Pierre Krack, Maximilian Li, Sven Parusel, Rudolf Lioutikov, Florian Walter, and Wolfram Burgard.Duobench: a reproducible benchmark for bimanual manipulation in simulation and the real world.arXiv preprint arXiv:2606.11901, 2026.URL https://arxiv.org/abs/2606.11901.
Kairos Team et al. [2026]	Kairos Team, Fei Wang, Shan You, Qiming Zhang, Tao Huang, Zuoyi Fu, Zhisheng Zheng, Yunlong Xi, Feng Lv, Xiaoming Wu, et al.Kairos: A regret-aware native world-action model stack for physical AI.arXiv preprint arXiv:2606.16533, 2026.URL https://arxiv.org/abs/2606.16533.
Kalashnikov et al. [2018]	Dmitry Kalashnikov, Alex Irpan, Peter Pastor, Julian Ibarz, Alexander Herzog, Eric Jang, Deirdre Quillen, Ethan Holly, Mrinal Kalakrishnan, Vincent Vanhoucke, et al.QT-Opt: Scalable deep reinforcement learning for vision-based robotic manipulation.In Conf. Robot Learn., 2018.URL https://arxiv.org/abs/1806.10293.
Kalashnikov et al. [2021]	Dmitry Kalashnikov, Jacob Varley, Yevgen Chebotar, Benjamin Swanson, Rico Jonschkowski, Chelsea Finn, Sergey Levine, and Karol Hausman.Mt-opt: continuous multi-task robotic reinforcement learning at scale.arXiv preprint arXiv:2104.08212, 2021.URL https://arxiv.org/abs/2104.08212.
Kareer et al. [2025]	Simar Kareer, Dhruv Patel, Ryan Punamiya, Pranay Mathur, Shuo Cheng, Chen Wang, Judy Hoffman, and Danfei Xu.EgoMimic: Scaling imitation learning via egocentric video.In IEEE Int. Conf. Robot. Autom., 2025.URL https://arxiv.org/abs/2410.24221.
Khanna et al. [2024]	Mukul Khanna, Yongsen Mao, Hanxiao Jiang, Sanjay Haresh, Brennan Shacklett, Dhruv Batra, Alexander Clegg, Eric Undersander, Angel X. Chang, and Manolis Savva.Habitat synthetic scenes dataset (hssd-200): an analysis of 3D scene scale and realism tradeoffs for objectgoal navigation.In IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2024.URL https://doi.org/10.1109/cvpr52733.2024.01550.
Khazatsky et al. [2024]	Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, et al.DROID: a large-scale in-the-wild robot manipulation dataset.In Robot. Sci. Syst., 2024.URL https://arxiv.org/abs/2403.12945.
Kim et al. [2024]	Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al.OpenVLA: an open-source vision-language-action model.In Conf. Robot Learn., 2024.URL https://arxiv.org/abs/2406.09246.
Kim et al. [2026]	Moo Jin Kim, Yihuai Gao, Tsung-Yi Lin, Yen-Chen Lin, Yunhao Ge, Grace Lam, Percy Liang, Shuran Song, Ming-Yu Liu, Chelsea Finn, and Jinwei Gu.Cosmos policy: Fine-tuning video models for visuomotor control and planning.In Int. Conf. Learn. Represent., 2026.URL https://arxiv.org/abs/2601.16163.
Kirillov et al. [2023]	Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollar, and Ross Girshick.Segment anything.In IEEE/CVF Int. Conf. Comput. Vis., 2023.URL https://arxiv.org/abs/2304.02643.
Kong et al. [2025]	Lingdong Kong, Yu Yang, Jianbiao Mei, Youquan Liu, Ao Liang, Dekai Zhu, Dongyue Lu, Wei Yin, et al.3D and 4D world modeling: A survey.arXiv preprint arXiv:2509.07996, 2025.URL https://arxiv.org/abs/2509.07996.
Kou et al. [2026]	Ziyi Kou, Ankit Kumar, Mia Huang, Taylor Niehues, Vatsal Mehta, Ergys Ristani, and Li Guan.Avi-ht: adaptive vision-imu fusion for 3D hand tracking.arXiv preprint arXiv:2605.21714, 2026.URL https://arxiv.org/abs/2605.21714.
Krishna et al. [2017]	Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A. Shamma, Michael S. Bernstein, and Li Fei-Fei.Visual Genome: Connecting language and vision using crowdsourced dense image annotations.Int. J. Comput. Vis., 2017.URL https://arxiv.org/abs/1602.07332.
Kumar et al. [2023]	Vikash Kumar, Rutav Shah, Gaoyue Zhou, Vincent Moens, Vittorio Caggiano, Jay Vakil, Abhishek Gupta, and Aravind Rajeswaran.RoboHive: a unified framework for robot learning.In Adv. Neural Inf. Process. Syst., 2023.URL https://arxiv.org/abs/2310.06828.
Kwon et al. [2021]	Taein Kwon, Bugra Tekin, Jan Stuhmer, Federica Bogo, and Marc Pollefeys.H2O: Two hands manipulating objects for first person interaction recognition.In IEEE/CVF Int. Conf. Comput. Vis., 2021.URL https://arxiv.org/abs/2104.11181.
Lee et al. [2026a]	Geonhyup Lee, Yeongjin Lee, Kangmin Kim, Seongju Lee, Sangjun Noh, Seunghyeok Back, and Kyoobin Lee.ManipForce: force-guided policy learning with frequency-aware representation for contact-rich manipulation.In IEEE Int. Conf. Robot. Autom., 2026a.URL https://arxiv.org/abs/2509.19047.
Lee et al. [2026b]	Jae Yong Lee, Daniel Scharstein, Akash Bapat, Hao Hu, Andrew Fu, Haoru Zhao, Paul Sammut, Xiang Li, et al.Ego-1K – a large-scale multiview video dataset for egocentric vision.In IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2026b.URL https://arxiv.org/abs/2603.13741.
Lee et al. [2025]	Jason Lee, Jiafei Duan, Haoquan Fang, Yuquan Deng, Shuo Liu, Boyang Li, Bohan Fang, Jieyu Zhang, Yi Ru Wang, Sangho Lee, et al.MolmoAct: Action reasoning models that can reason in space.arXiv preprint arXiv:2508.07917, 2025.URL https://arxiv.org/abs/2508.07917.
Lei et al. [2026]	Huashuo Lei, Wenxuan Song, Huarui Zhang, Jieyuan Pei, Jiayi Chen, Haodong Yan, Han Zhao, Pengxiang Ding, et al.RoboMemArena: a comprehensive and challenging robotic memory benchmark.arXiv preprint arXiv:2605.10921, 2026.URL https://arxiv.org/abs/2605.10921.
Leju Robotics [2026]	Leju Robotics.Let-base-dataset: let basic operation dataset.https://www.modelscope.cn/datasets/lejurobot/LET-Base-Dataset, 2026.URL https://huggingface.co/datasets/LejuRobotics/LET-Base-Dataset.
Lenz et al. [2015]	Ian Lenz, Honglak Lee, and Ashutosh Saxena.Deep learning for detecting robotic grasps.Int. J. Robot. Res., 2015.URL https://doi.org/10.1177/0278364914549607.
Li et al. [2024a]	Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li.LLaVA-OneVision: easy visual task transfer.Trans. Mach. Learn. Res., 2024a.URL https://arxiv.org/abs/2408.03326.
Li et al. [2022]	Chengshu Li, Ruohan Zhang, Josiah Wong, Cem Gokmen, Sanjana Srivastava, Roberto Martín-Martín, Chen Wang, Gabrael Levine, et al.BEHAVIOR-1K: a human-centered, embodied AI benchmark with 1,000 everyday activities and realistic simulation.In Conf. Robot Learn., 2022.URL https://arxiv.org/abs/2403.09227.
Li et al. [2024b]	Chuanyu Li, Renjun Dang, Xiang Li, Zhiyuan Wu, Jing Xu, Hamidreza Kasaei, Roberto Calandra, Nathan Lepora, Shan Luo, Hao Su, and Rui Chen.ManiSkill-vitac 2025: challenge on manipulation skill learning with vision and tactile sensing.arXiv preprint arXiv:2411.12503, 2024b.URL https://arxiv.org/abs/2411.12503.
Li et al. [2025a]	Chuanyu Li, Chaoyi Liu, Daotan Wang, Shuyu Zhang, Lusong Li, Zecui Zeng, Fangchen Liu, Jing Xu, and Rui Chen.ViTaMIn-b: a reliable and efficient visuo-tactile bimanual manipulation interface.arXiv preprint arXiv:2511.05858, 2025a.URL https://arxiv.org/abs/2511.05858.
Li et al. [2025b]	Guangrun Li, Yaoxu Lyu, Zhuoyang Liu, Chengkai Hou, Jieyu Zhang, and Shanghang Zhang.H2r: a human-to-robot data augmentation for robot pre-training from videos.arXiv preprint arXiv:2505.11920, 2025b.URL https://arxiv.org/abs/2505.11920.
Li et al. [2025c]	Hengtao Li, Pengxiang Ding, Runze Suo, Yihao Wang, Zirui Ge, Dongyuan Zang, Kexian Yu, Mingyang Sun, Hongyin Zhang, Donglin Wang, et al.VLA-RFT: vision-language-action reinforcement fine-tuning with verified rewards in world simulators.arXiv preprint arXiv:2510.00406, 2025c.URL https://arxiv.org/abs/2510.00406.
Li et al. [2026a]	Jiajun Li, Tiecheng Guo, Yifan Ye, Rongyu Zhang, Xiaowei Chi, Qianpu Sun, Ying Li, Yunfan Lou, Yan Huang, Zhihe Lu, et al.Efficient-WAM: A 1B-parameter world-action model with low-cost future imagination.arXiv preprint arXiv:2606.10040, 2026a.
Li et al. [2025d]	Jinming Li, Yichen Zhu, Zhibin Tang, Junjie Wen, Minjie Zhu, Xiaoyu Liu, Chengmeng Li, Ran Cheng, Yaxin Peng, Yan Peng, et al.Coa-VLA: improving vision-language-action models via visual-text chain-of-affordance.In IEEE/CVF Int. Conf. Comput. Vis., 2025d.URL https://arxiv.org/abs/2412.20451.
Li et al. [2026b]	Lin Li, Qihang Zhang, Yiming Luo, Shuai Yang, Ruilin Wang, Fei Han, Mingrui Yu, Zelin Gao, Nan Xue, Xing Zhu, Yujun Shen, and Yinghao Xu.Causal world modeling for robot control.In Robot. Sci. Syst., 2026b.URL https://arxiv.org/abs/2601.21998.
Li et al. [2026c]	Linfei Li, Lin Zhang, and Ying Shen.RealVLG-R1: A large-scale real-world visual-language grounding benchmark for robotic perception and manipulation.In IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2026c.URL https://arxiv.org/abs/2603.14880.
Li et al. [2024c]	Qixiu Li, Yaobo Liang, Zeyu Wang, Lin Luo, Xi Chen, Mozheng Liao, Fangyun Wei, Yu Deng, et al.CogACT: A foundational vision-language-action model for synergizing cognition and action in robotic manipulation.arXiv preprint arXiv:2411.19650, 2024c.URL https://arxiv.org/abs/2411.19650.
Li et al. [2025e]	Qixiu Li, Yu Deng, Yaobo Liang, Lin Luo, Lei Zhou, Chengtang Yao, Lingqi Zeng, Zhiyuan Feng, et al.Scalable vision-language-action model pretraining for robotic manipulation with real-life human activity videos.In IEEE Int. Conf. Robot. Autom., 2025e.URL https://arxiv.org/abs/2510.21571.
Li et al. [2026d]	Shalfun Li, Victor Yao, Charles Yang, Truth Qu, Regis Cheng, Ryan Yu, Howard Lu, Newton Von, et al.WALL-WM: Carving world action modeling at the event joints.arXiv preprint arXiv:2606.01955, 2026d.URL https://arxiv.org/abs/2606.01955.
Li et al. [2026e]	Xinghang Li, Jun Guo, Qiwei Li, Long Qian, Hang Lai, Yueze Wang, Hongyu Yan, Jiahang Cao, Xi Chen, Jingen Qu, et al.Xiaomi-Robotics-U0: Unified embodied synthesis with world foundation model.arXiv preprint arXiv:2607.11643, 2026e.URL https://arxiv.org/abs/2607.11643.
Li et al. [2024d]	Xuanlin Li, Kyle Hsu, Jiayuan Gu, Karl Pertsch, Oier Mees, Homer Rich Walke, Chuyuan Fu, Ishikaa Lunawat, et al.Evaluating real-world robot manipulation policies in simulation.In Conf. Robot Learn., 2024d.URL https://arxiv.org/abs/2405.05941.
Li et al. [2026f]	Xukun Li, Yu Sun, Lei Zhang, Bosheng Huang, Yibo Peng, Yuan Meng, Haojun Jiang, Shaoxuan Xie, Guocai Yao, Alois Knoll, et al.DECO: decoupled multimodal diffusion transformer for bimanual dexterous manipulation with a plugin tactile adapter.In Int. Conf. Mach. Learn., 2026f.URL https://arxiv.org/abs/2602.05513.
Li et al. [2025f]	Yaxuan Li, Yichen Zhu, Junjie Wen, Chaomin Shen, and Yi Xu.WorldEval: World model as real-world robot policies evaluator.arXiv preprint arXiv:2505.19017, 2025f.URL https://arxiv.org/abs/2505.19017.
Li et al. [2026g]	Yihang Li, Xuelong Wei, Jingzhou Luo, Yingjing Xiao, Yibo Bai, Guangyuan Zhou, Teng Zou, Chenguang Gui, et al.Egolive: a large-scale egocentric dataset from real-world human tasks.arXiv preprint arXiv:2604.23570, 2026g.URL https://arxiv.org/abs/2604.23570.
Li et al. [2023]	Yin Li, Miao Liu, and James M. Rehg.In the eye of the beholder: Gaze and actions in first person video.IEEE Trans. Pattern Anal. Mach. Intell., 2023.URL https://www.computer.org/csdl/journal/tp/2023/06/09325929/1qpv4fOGLEA.
Li et al. [2024e]	Yuan-Ming Li, Wei-Jin Huang, An-Lan Wang, Ling-An Zeng, Jing-Ke Meng, and Wei-Shi Zheng.EgoExo-Fitness: towards egocentric and exocentric full-body action understanding.In Eur. Conf. Comput. Vis., 2024e.URL https://arxiv.org/abs/2406.08877.
Li et al. [2025g]	Yuyang Li, Wenxin Du, Chang Yu, Puhao Li, Zihang Zhao, Tengyu Liu, Chenfanfu Jiang, Yixin Zhu, and Siyuan Huang.Taccel: scaling up vision-based tactile robotics via high-performance GPU simulation.In Adv. Neural Inf. Process. Syst., 2025g.URL https://doi.org/10.48550/arXiv.2504.12908.
Li et al. [2026h]	Zishuo Li, Bowen Yang, Changtao Miao, Kai Zhu, Hao Chen, Qingze Guan, Zhengxing Wu, Wanke Zhan, Yang Sun, Zhiyi Huang, et al.Open-aoe: An open egocentric manipulation dataset and toolchain for embodied learning.arXiv preprint arXiv:2607.14183, 2026h.URL https://arxiv.org/abs/2607.14183.
Liao et al. [2026]	Yue Liao, Pengfei Zhou, Siyuan Huang, Donglin Yang, Shengcong Chen, Yuxin Jiang, Yue Hu, Jingbin Cai, et al.Genie envisioner: a unified world foundation platform for robotic manipulation.In Int. Conf. Learn. Represent., 2026.URL https://arxiv.org/abs/2508.05635.
Lin et al. [2025a]	Fanqi Lin, Yingdong Hu, Pingyue Sheng, Chuan Wen, Jiacheng You, and Yang Gao.Data scaling laws in imitation learning for robotic manipulation.In Int. Conf. Learn. Represent., 2025a.URL https://arxiv.org/abs/2410.18647.
Lin et al. [2014]	Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick.Microsoft COCO: common objects in context.In Eur. Conf. Comput. Vis., 2014.URL https://arxiv.org/abs/1405.0312.
Lin et al. [2020]	Xingyu Lin, Yufei Wang, Jake Olkin, and David Held.SoftGym: benchmarking deep reinforcement learning for deformable object manipulation.In Conf. Robot Learn., 2020.URL https://arxiv.org/abs/2011.07215.
Lin et al. [2022]	Yijiong Lin, John Lloyd, Alex Church, and Nathan F. Lepora.Tactile Gym 2.0: Sim-to-real deep reinforcement learning for comparing low-cost high-resolution robot touch.IEEE Robot. Autom. Lett., 2022.URL https://arxiv.org/abs/2207.10763.
Lin et al. [2025b]	Zijun Lin, Jiafei Duan, Haoquan Fang, Dieter Fox, Ranjay Krishna, Cheston Tan, and Bihan Wen.Failsafe: reasoning and recovery from failures in vision-language-action models.arXiv preprint arXiv:2510.01642, 2025b.URL https://arxiv.org/abs/2510.01642.
Linux Foundation and NVIDIA and Google DeepMind and Disney Research [2025]	Linux Foundation and NVIDIA and Google DeepMind and Disney Research.Newton: an open-source, GPU-accelerated physics engine for robotics.https://github.com/newton-physics/newton, 2025.URL https://developer.nvidia.com/blog/announcing-newton-an-open-source-physics-engine-for-robotics-simulation/.
Liu et al. [2023a]	Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone.LIBERO: benchmarking knowledge transfer for lifelong robot learning.In Adv. Neural Inf. Process. Syst., 2023a.URL https://arxiv.org/abs/2306.03310.
Liu et al. [2025a]	Fangchen Liu, Chuanyu Li, Yihua Qin, Ankit Shaw, Jing Xu, Pieter Abbeel, and Rui Chen.ViTaMIn: learning contact-rich tasks through robot-free visuo-tactile manipulation interface.arXiv preprint arXiv:2504.06156, 2025a.URL https://arxiv.org/abs/2504.06156.
Liu et al. [2023b]	Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee.Visual instruction tuning.In Adv. Neural Inf. Process. Syst., 2023b.URL https://doi.org/10.52202/075280-1516.
Liu et al. [2025b]	Huihan Liu, Soroush Nasiriany, Lance Zhang, Zhiyao Bao, and Yuke Zhu.Robot learning on the job: human-in-the-loop autonomy and learning during deployment.Int. J. Robot. Res., 2025b.URL https://arxiv.org/abs/2211.08416.
Liu et al. [2026a]	Jiaming Liu, Hao Chen, Pengju An, Zhuoyang Liu, Renrui Zhang, Chenyang Gu, Xiaoqi Li, Ziyu Guo, et al.HybridVLA: collaborative diffusion and autoregression in a unified vision-language-action model.In Int. Conf. Learn. Represent., 2026a.URL https://arxiv.org/abs/2503.10631.
Liu et al. [2026b]	Jiaming Liu, Yinxi Wang, Chenyang Gu, Siyuan Qian, Xiangju Mi, Hao Chen, Jiawei Chen, Qingpo Wuwu, et al.LaST-HD: Learning latent physical reasoning from scalable human data for robot manipulation.arXiv preprint arXiv:2606.23685, 2026b.URL https://arxiv.org/abs/2606.23685.
Liu et al. [2025c]	Junpeng Liu, Tianyue Ou, Yifan Song, Yuxiao Qu, Wai Lam, Chenyan Xiong, Wenhu Chen, Graham Neubig, and Xiang Yue.Harnessing webpage UIs for text-rich visual understanding.In Int. Conf. Learn. Represent., 2025c.URL https://arxiv.org/abs/2410.13824.
Liu et al. [2025d]	Kehui Liu, Zhongjie Jia, Yang Li, Zhaxizhuoma, Pengan Chen, Song Liu, Xin Liu, Pingrui Zhang, Haoming Song, Xinyi Ye, et al.FastUMI-100k: advancing data-driven robotic manipulation with a large-scale UMI-style dataset.arXiv preprint arXiv:2510.08022, 2025d.URL https://arxiv.org/abs/2510.08022.
Liu et al. [2025e]	Songming Liu, Lingxuan Wu, Bangguo Li, Hengkai Tan, Huayu Chen, Zhengyi Wang, Ke Xu, Hang Su, and Jun Zhu.RDT-1b: a diffusion foundation model for bimanual manipulation.In Int. Conf. Learn. Represent., 2025e.URL https://arxiv.org/abs/2410.07864.
Liu et al. [2026c]	Songming Liu, Bangguo Li, Kai Ma, Lingxuan Wu, Hengkai Tan, Xiao Ouyang, Hang Su, and Jun Zhu.RDT2: Exploring the scaling limit of UMI data towards zero-shot cross-embodiment generalization.In ICML, 2026c.URL https://arxiv.org/abs/2602.03310.
Liu et al. [2022]	Yunze Liu, Yun Liu, Che Jiang, Kangbo Lyu, Weikang Wan, Hao Shen, Boqiang Liang, Zhoujie Fu, He Wang, and Li Yi.HOI4D: a 4D egocentric dataset for category-level human-object interaction.In IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2022.URL https://arxiv.org/abs/2203.01577.
Liu et al. [2023c]	Zeyi Liu, Arpit Bahety, and Shuran Song.Reflect: summarizing robot experiences for failure explanation and correction.In Conf. Robot Learn., 2023c.URL https://arxiv.org/abs/2306.15724.
Liu et al. [2024]	Zeyi Liu, Cheng Chi, Eric Cousineau, Naveen Kuppuswamy, Benjamin Burchfiel, and Shuran Song.ManiWAV: learning robot manipulation from in-the-wild audio-visual data.In Conf. Robot Learn., 2024.URL https://arxiv.org/abs/2406.19464.
Liu et al. [2026d]	Zhuoyang Liu, Jiaming Liu, Hao Chen, Jiale Yu, Ziyu Guo, Chengkai Hou, Chenyang Gu, Xiangju Mi, et al.LaST0: Latent spatio-temporal chain-of-thought for robotic vision-language-action model.In Int. Conf. Mach. Learn., 2026d.URL https://arxiv.org/abs/2601.05248.
Lou et al. [2026a]	Yunfan Lou, Xiaowei Chi, Xiaojie Zhang, Zezhong Qian, Chengxuan Li, Rongyu Zhang, Yaoxu Lyu, Guoyu Song, Chuyao Fu, Haoxuan Xu, Pengwei Wang, and Shanghang Zhang.Predicting what matters: Robust generalist robot policy learning via future semantic mask.In Int. Conf. Mach. Learn., 2026a.URL https://arxiv.org/abs/2604.19683.
Lou et al. [2026b]	Yunfan Lou, Yifan Ye, Yankai Fu, Jun Cen, Xiaowei Chi, Yaoxu Lyu, Peidong Jia, Sirui Han, Zhihe Lu, and Shanghang Zhang.Dream-Tac: A unified tactile world action model for contact-rich robot manipulation, 2026b.URL https://arxiv.org/abs/2606.08737.
Lu et al. [2024]	Haoran Lu, Ruihai Wu, Yitong Li, Sijie Li, Ziyu Zhu, Chuanruo Ning, Yan Shen, Longzan Luo, Yuanpei Chen, and Hao Dong.GarmentLab: A unified simulation and benchmark for garment manipulation.In Adv. Neural Inf. Process. Syst., 2024.URL https://arxiv.org/abs/2411.01200.
Lu et al. [2026]	Jinghui Lu, Jiayi Guan, Zhijian Huang, Jinlong Li, Guang Li, Lingdong Kong, Yingyan Li, Han Wang, et al.Xiaomi OneVL: One-step latent reasoning and planning with vision-language explanation.arXiv preprint arXiv:2604.18486, 2026.URL https://arxiv.org/abs/2604.18486.
Lu et al. [2025]	Weifeng Lu, Minghao Ye, Zewei Ye, Ruihan Tao, Shuo Yang, and Bo Zhao.RoboFAC: a comprehensive framework for robotic failure analysis and correction.arXiv preprint arXiv:2505.12224, 2025.URL https://arxiv.org/abs/2505.12224.
Lu et al. [2023]	Yuhao Lu, Yixuan Fan, Beixing Deng, Fangfu Liu, Yali Li, and Shengjin Wang.VL-Grasp: A 6-DoF interactive grasp policy for language-oriented objects in cluttered indoor scenes.In IEEE/RSJ Int. Conf. Intell. Robots Syst., 2023.URL https://arxiv.org/abs/2308.00640.
Luo et al. [2026a]	Hao Luo, Yicheng Feng, Wanpeng Zhang, Sipeng Zheng, Ye Wang, Haoqi Yuan, Jiazheng Liu, Chaoyi Xu, Qin Jin, and Zongqing Lu.Being-h0: Vision-language-action pretraining from large-scale human videos.In Int. Conf. Mach. Learn., 2026a.URL https://arxiv.org/abs/2507.15597.
Luo et al. [2026b]	Hao Luo, Ye Wang, Wanpeng Zhang, Sipeng Zheng, Ziheng Xi, Chaoyi Xu, Haiweng Xu, Haoqi Yuan, Chi Zhang, Yiqing Wang, Yicheng Feng, and Zongqing Lu.Being-h0.5: Scaling human-centric robot learning for cross-embodiment generalization.arXiv preprint arXiv:2601.12993, 2026b.URL https://arxiv.org/abs/2601.12993.
Luo et al. [2026c]	Hao Luo, Wanpeng Zhang, Yicheng Feng, Sipeng Zheng, Haiweng Xu, Chaoyi Xu, Ziheng Xi, Yuhui Fu, and Zongqing Lu.Being-h0.7: A latent world-action model from egocentric videos.arXiv preprint arXiv:2605.00078, 2026c.URL https://arxiv.org/abs/2605.00078.
Luo et al. [2024]	Jianlan Luo, Charles Xu, Fangchen Liu, Liam Tan, Zipeng Lin, Jeffrey Wu, Pieter Abbeel, and Sergey Levine.FMB: A functional manipulation benchmark for generalizable robotic learning.Int. J. Robot. Res., 2024.URL https://arxiv.org/abs/2401.08553.
Lyu et al. [2026]	Jiangran Lyu, Kai Liu, Xuheng Zhang, Haoran Liao, Yusen Feng, Wenxuan Zhu, Tingrui Shen, Jiayi Chen, et al.LDA-1B: scaling latent dynamics action model via universal embodied data ingestion.In Robot. Sci. Syst., 2026.URL https://arxiv.org/abs/2602.12215.
Lyu et al. [2024]	Ruiyuan Lyu, Jingli Lin, Tai Wang, Shuai Yang, Xiaohan Mao, Yilun Chen, Runsen Xu, Haifeng Huang, Chenming Zhu, Dahua Lin, and Jiangmiao Pang.MMScan: a multi-modal 3D scene dataset with hierarchical grounded language annotations.In Adv. Neural Inf. Process. Syst., 2024.URL https://arxiv.org/abs/2406.09401.
Ma et al. [2026a]	Haoxiang Ma, Junhao Cai, Xiaoxu Xu, Hao Li, Yuyin Yang, Yang Tian, Jiafei Cao, Hongrui Zhu, et al.InternVLA-A1.5: Unifying understanding, latent foresight, and action for compositional generalization.arXiv preprint arXiv:2607.04988, 2026a.URL https://arxiv.org/abs/2607.04988.
Ma et al. [2026b]	Juncheng Ma, Jianxin Bi, Yufan Deng, Xuanran Zhai, Kewei Zhang, Ye Huang, Bo Liang, Shukai Gong, et al.HumanScale: Egocentric human video can outperform r eal-robot data for embodied pretraining.arXiv preprint arXiv:2606.20521, 2026b.URL https://arxiv.org/abs/2606.20521.
Ma et al. [2024]	Lingni Ma, Yuting Ye, Fangzhou Hong, Vladimir Guzov, Yifeng Jiang, Rowan Postyeni, Luis Pesqueira, Alexander Gamino, et al.Nymeria: A massive collection of multimodal egocentric daily motion in the wild.In Eur. Conf. Comput. Vis., 2024.URL https://arxiv.org/abs/2406.09905.
Ma et al. [2023]	Xiaojian Ma, Silong Yong, Zilong Zheng, Qing Li, Yitao Liang, Song-Chun Zhu, and Siyuan Huang.SQA3D: situated question answering in 3D scenes.In Int. Conf. Learn. Represent., 2023.URL https://arxiv.org/abs/2210.07474.
Makoviychuk et al. [2021]	Viktor Makoviychuk, Lukasz Wawrzyniak, Yunrong Guo, Michelle Lu, Kier Storey, Miles Macklin, David Hoeller, Nikita Rudin, Arthur Allshire, Ankur Handa, and Gavriel State.Isaac gym: high performance GPU-based physics simulation for robot learning.In Adv. Neural Inf. Process. Syst., 2021.URL https://arxiv.org/abs/2108.10470.
Mandlekar et al. [2018]	Ajay Mandlekar, Yuke Zhu, Animesh Garg, Jonathan Booher, Max Spero, Albert Tung, Julian Gao, John Emmons, Anchit Gupta, Emre Orbay, et al.RoboTurk: a crowdsourcing platform for robotic skill learning through imitation.In Conf. Robot Learn., 2018.URL https://arxiv.org/abs/1811.02790.
Mandlekar et al. [2020]	Ajay Mandlekar, Danfei Xu, Roberto Martín-Martín, Yuke Zhu, Li Fei-Fei, and Silvio Savarese.Human-in-the-loop imitation learning using remote teleoperation.arXiv preprint arXiv:2012.06733, 2020.URL https://arxiv.org/abs/2012.06733.
Mandlekar et al. [2021]	Ajay Mandlekar, Danfei Xu, Josiah Wong, Soroush Nasiriany, Chen Wang, Rohun Kulkarni, Li Fei-Fei, Silvio Savarese, Yuke Zhu, and Roberto Martín-Martín.What matters in learning from offline human demonstrations for robot manipulation.In Conf. Robot Learn., 2021.URL https://arxiv.org/abs/2108.03298.
Mandlekar et al. [2023]	Ajay Mandlekar, Soroush Nasiriany, Bowen Wen, Iretiayo Akinola, Yashraj Narang, Linxi Fan, Yuke Zhu, and Dieter Fox.MimicGen: a data generation system for scalable robot learning using human demonstrations.In Conf. Robot Learn., 2023.URL https://arxiv.org/abs/2310.17596.
Masry et al. [2022]	Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque.ChartQA: A benchmark for question answering about charts with visual and logical reasoning.In Annu. Meet. Assoc. Comput. Linguist., 2022.URL https://arxiv.org/abs/2203.10244.
Mathew et al. [2021]	Minesh Mathew, Dimosthenis Karatzas, and C. V. Jawahar.DocVQA: a dataset for VQA on document images.In IEEE/CVF Winter Conf. Appl. Comput. Vis., 2021.URL https://arxiv.org/abs/2007.00398.
Mees et al. [2022]	Oier Mees, Lukas Hermann, Erick Rosete-Beas, and Wolfram Burgard.Calvin: a benchmark for Language-Conditioned policy learning for long-horizon robot manipulation tasks.IEEE Robot. Autom. Lett., 2022.URL https://arxiv.org/abs/2112.03227.
Mishra et al. [2019]	Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty.OCR-VQA: Visual question answering by reading text in images.In Int. Conf. Doc. Anal. Recognit., 2019.URL https://doi.org/10.1109/icdar.2019.00156.
Mittal et al. [2023]	Mayank Mittal, Calvin Yu, Qinxi Yu, Jingzhou Liu, Nikita Rudin, David Hoeller, Jia Lin Yuan, Ritvik Singh, et al.Orbit: a unified simulation framework for interactive robot learning environments.IEEE Robot. Autom. Lett., 2023.URL https://arxiv.org/abs/2301.04195.
Motoda et al. [2025]	Tomohiro Motoda, Masaki Murooka, Ryoichi Nakajo, Muhammad A. Muttaqien, Koshi Makihara, Hanbit Oh, Keisuke Shirai, Floris Erich, Ryo Hanai, and Yukiyasu Domae.AIST-bimanual manipulation, 2025.URL https://aistairc.github.io/aist_bimanip_site.
Mu et al. [2023]	Yao Mu, Qinglong Zhang, Mengkang Hu, Wenhai Wang, Mingyu Ding, Jun Jin, Bin Wang, Jifeng Dai, Yu Qiao, and Ping Luo.EmbodiedGPT: vision-language pre-training via embodied chain of thought.In Adv. Neural Inf. Process. Syst., 2023.URL https://arxiv.org/abs/2305.15021.
Mu et al. [2024]	Yao Mu, Tianxing Chen, Shijia Peng, Zanxin Chen, Zeyu Gao, Yude Zou, Lunkai Lin, Zhiqiang Xie, and Ping Luo.RoboTwin: Dual-arm robot benchmark with generative digital twins.arXiv preprint arXiv:2409.02920, 2024.URL https://arxiv.org/abs/2409.02920.
Mur-Artal et al. [2015]	Raul Mur-Artal, J. M. M. Montiel, and Juan D. Tardos.ORB-SLAM: a versatile and accurate monocular slam system.IEEE Trans. Robot., 2015.URL https://doi.org/10.1109/tro.2015.2463671.
Nai et al. [2026]	Ruiqian Nai, Boyuan Zheng, Junming Zhao, Haodong Zhu, Sicong Dai, Zunhao Chen, Yihang Hu, Yingdong Hu, Tong Zhang, Chuan Wen, et al.Humanoid manipulation interface: humanoid whole-body manipulation from robot-free demonstrations.arXiv preprint arXiv:2602.06643, 2026.URL https://arxiv.org/abs/2602.06643.
Nasiriany et al. [2024]	Soroush Nasiriany, Abhiram Maddukuri, Lance Zhang, Adeet Parikh, Aaron Lo, Abhishek Joshi, Ajay Mandlekar, and Yuke Zhu.RoboCasa: Large-scale simulation of everyday tasks for generalist robots.In Robot. Sci. Syst., 2024.URL https://arxiv.org/abs/2406.02523.
Nasiriany et al. [2026]	Soroush Nasiriany, Sepehr Nasiriany, Abhiram Maddukuri, and Yuke Zhu.RoboCasa365: a large-scale simulation framework for training and benchmarking generalist robots.In Int. Conf. Learn. Represent., 2026.URL https://doi.org/10.48550/arXiv.2603.04356.
Nelson et al. [2026]	Nigel Nelson, Juo-Tung Chen, Jesse Haworth, Xinhao Chen, Lukas Zbinden, Dianye Huang, Alaa Eldin Abdelaal, Alberto Arezzo, Ayberk Acar, Farshid Alambeigi, et al.Open-H-Embodiment: a large-scale dataset for enabling foundation models in medical robotics.arXiv preprint arXiv:2604.21017, 2026.URL https://arxiv.org/abs/2604.21017.
Nguyen et al. [2024a]	Duc Huy Nguyen, Tim Schneider, Guillaume Duret, Alap Kshirsagar, Boris Belousov, and Jan Peters.Tacex: gelsight tactile simulation in Isaac sim – combining soft-body and visuotactile simulators.arXiv preprint arXiv:2411.04776, 2024a.URL https://arxiv.org/abs/2411.04776.
Nguyen et al. [2024b]	Toan Nguyen, Minh Nhat Vu, Baoru Huang, An Vuong, Quan Vuong, Ngan Le, Thieu Vo, and Anh Nguyen.Language-driven 6-DoF grasp detection using negative prompt guidance.In Eur. Conf. Comput. Vis., 2024b.URL https://arxiv.org/abs/2407.13842.
NVIDIA [2025a]	NVIDIA.Physicalai-robotics-GR00T-x-embodiment-sim.Hugging Face dataset, 2025a.URL https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim.Synthetic cross-embodiment trajectory datasets used for GR00T N1 post-training.
NVIDIA [2025b]	NVIDIA.NVIDIA Isaac sim: robotics simulation and synthetic data generation.https://developer.nvidia.com/isaac/sim, 2025b.URL https://developer.nvidia.com/isaac/sim.
NVIDIA et al. [2026]	NVIDIA, Aditi, Niket Agarwal, Arslan Ali, Jon Allen, Martin Antolini, Adeline Aubame, Alisson Azzolini, Junjie Bai, Maciej Bala, Yogesh Balaji, et al.Cosmos 3: omnimodal world models for physical AI.arXiv preprint arXiv:2606.02800, 2026.URL https://arxiv.org/abs/2606.02800.
Ohkawa et al. [2023]	Takehiko Ohkawa, Kun He, Fadime Sener, Tomas Hodan, Luan Tran, and Cem Keskin.Assemblyhands: towards egocentric activity understanding via 3D hand pose estimation.In IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2023.URL https://arxiv.org/abs/2304.12301.
Pacaud et al. [2025]	Paul Pacaud, Ricardo Garcia, Shizhe Chen, and Cordelia Schmid.Scaling cross-environment failure reasoning data for vision-language robotic manipulation.arXiv preprint arXiv:2512.01946, 2025.URL https://arxiv.org/abs/2512.01946.
Peddi et al. [2024]	Rohith Peddi, Shivvrat Arya, Bharath Challa, Likhitha Pallapothula, Akshay Vyas, Bhavya Gouripeddi, Jikai Wang, Qifan Zhang, et al.CaptainCook4D: A dataset for understanding errors in procedural activities.In Adv. Neural Inf. Process. Syst., 2024.URL https://arxiv.org/abs/2312.14556.
Peng et al. [2026]	Xingyu Peng, Chen Gao, Liankai Jin, Annan Li, and Si Liu.BiCoord: a bimanual manipulation benchmark towards long-horizon spatial-temporal coordination.arXiv preprint arXiv:2604.05831, 2026.URL https://arxiv.org/abs/2604.05831.
Perrett et al. [2025]	Toby Perrett, Ahmad Darkhalil, Saptarshi Sinha, Omar Emara, Sam Pollard, Kranti Parida, Kaiting Liu, Prajwal Gatti, et al.HD-EPIC: A highly-detailed egocentric video dataset.In IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2025.URL https://arxiv.org/abs/2502.04144.
Pinto and Gupta [2016]	Lerrel Pinto and Abhinav Gupta.Supersizing self-supervision: learning to grasp from 50k tries and 700 robot hours.In IEEE Int. Conf. Robot. Autom., 2016.URL https://arxiv.org/abs/1509.06825.
Puig et al. [2024]	Xavier Puig, Eric Undersander, Andrew Szot, Mikael Dallaire Cote, Tsung-Yen Yang, Ruslan Partsey, Ruta Desai, Alexander William Clegg, et al.Habitat 3.0: a co-Habitat for humans, avatars and robots.In Int. Conf. Learn. Represent., 2024.URL https://arxiv.org/abs/2310.13724.
Pumacay et al. [2024]	Wilbert Pumacay, Ishika Singh, Jiafei Duan, Ranjay Krishna, Jesse Thomason, and Dieter Fox.THE COLOSSEUM: A benchmark for evaluating generalization for robotic manipulation.In Robot. Sci. Syst., 2024.URL https://arxiv.org/abs/2402.08191.
Punamiya et al. [2026]	Ryan Punamiya, Simar Kareer, Zeyi Liu, Josh Citron, Ri-Zhao Qiu, Xiongyi Cai, Alexey Gavryushin, Jiaqi Chen, et al.Egoverse: an egocentric human dataset for robot learning from around the world.arXiv preprint arXiv:2604.07607, 2026.URL https://arxiv.org/abs/2604.07607.
Qian et al. [2024]	Long Qian, Juncheng Li, Yu Wu, Yaobo Ye, Hao Fei, Tat-Seng Chua, Yueting Zhuang, and Siliang Tang.Momentor: advancing video large language model with fine-grained temporal reasoning.In Int. Conf. Mach. Learn., 2024.URL https://arxiv.org/abs/2402.11435.
Qin et al. [2022]	Yuzhe Qin, Yueh-Hua Wu, Shaowei Liu, Hanwen Jiang, Ruihan Yang, Yang Fu, and Xiaolong Wang.DexMV: imitation learning for dexterous manipulation from human videos.In Eur. Conf. Comput. Vis., 2022.URL https://arxiv.org/abs/2108.05877.
Qin et al. [2023]	Yuzhe Qin, Wei Yang, Binghao Huang, Karl Van Wyk, Hao Su, Xiaolong Wang, Yu-Wei Chao, and Dieter Fox.AnyTeleop: a general vision-based dexterous robot arm-hand teleoperation system.In Robot. Sci. Syst., 2023.URL https://arxiv.org/abs/2307.04577.
Qiu et al. [2025a]	Heqian Qiu, Zhaofeng Shi, Lanxiao Wang, Huiyu Xiong, Xiang Li, and Hongliang Li.Egome: a new dataset and challenge for following me via egocentric view in real world.arXiv preprint arXiv:2501.19061, 2025a.URL https://arxiv.org/abs/2501.19061.
Qiu et al. [2025b]	Ri-Zhao Qiu, Shiqi Yang, Xuxin Cheng, Chaitanya Chawla, Jialong Li, Tairan He, Ge Yan, David J Yoon, Ryan Hoque, Lars Paulsen, et al.Humanoid policy˜ human policy.In Conf. Robot Learn., 2025b.URL https://arxiv.org/abs/2503.13441.
Qu et al. [2025a]	Delin Qu, Haoming Song, Qizhi Chen, Zhaoqing Chen, Xianqiang Gao, Dong Wang, , Xinyi Ye, Qi Lv, Modi Shi, Guanghui Ren, Maoqing Yao, Bin Zhao, et al.EO-1: An open unified embodied foundation model for general robot control.arXiv preprint arXiv:2508.21112, 2025a.URL https://arxiv.org/abs/2508.21112.
Qu et al. [2025b]	Delin Qu, Haoming Song, Qizhi Chen, Yuanqi Yao, Xinyi Ye, Yan Ding, Zhigang Wang, JiaYuan Gu, Bin Zhao, Dong Wang, and Xuelong Li.SpatialVLA: exploring spatial representations for visual-language-action model.In Robot. Sci. Syst., 2025b.URL https://arxiv.org/abs/2501.15830.
Quevedo et al. [2025]	Julian Quevedo, Ansh Kumar Sharma, Yixiang Sun, Varad Suryavanshi, Percy Liang, and Sherry Yang.WorldGym: world model as an environment for policy evaluation.arXiv preprint arXiv:2506.00613, 2025.URL https://arxiv.org/abs/2506.00613.
Ragusa et al. [2024]	Francesco Ragusa, Rosario Leonardi, Michele Mazzamuto, Claudia Bonanno, Rosario Scavo, Antonino Furnari, and Giovanni Maria Farinella.Enigma-51: towards a fine-grained understanding of human-object interactions in industrial scenarios.In IEEE/CVF Winter Conf. Appl. Comput. Vis., 2024.URL https://arxiv.org/abs/2309.14809.
Rajeswaran et al. [2018]	Aravind Rajeswaran, Vikash Kumar, Abhishek Gupta, Giulia Vezzani, John Schulman, Emanuel Todorov, and Sergey Levine.Learning complex dexterous manipulation with deep reinforcement learning and demonstrations.In Robot. Sci. Syst., 2018.URL http://arxiv.org/abs/1709.10087.
Ramakrishnan et al. [2021]	Santhosh K. Ramakrishnan, Aaron Gokaslan, Erik Wijmans, Oleksandr Maksymets, Alex Clegg, John Turner, Eric Undersander, Wojciech Galuba, et al.Habitat-matterport 3D dataset (hm3d): 1000 large-scale 3D environments for embodied AI.In Adv. Neural Inf. Process. Syst., 2021.URL https://arxiv.org/abs/2109.08238.
Ramanathan et al. [2023]	Vignesh Ramanathan, Anmol Kalia, Vladan Petrovic, Yi Wen, Baixue Zheng, Baishan Guo, Rui Wang, Aaron Marquez, Rama Kovvuri, Abhishek Kadian, Amir Mousavi, Yiwen Song, Abhimanyu Dubey, and Dhruv Mahajan.PACo: parts and attributes of common objects.In IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2023.URL https://arxiv.org/abs/2301.01795.
Rayyan et al. [2025]	Omar Rayyan, John Abanes, Mahmoud Hafez, Anthony Tzes, and Fares Abu-Dakka.MV-UMI: a scalable multi-view interface for cross-embodiment learning.arXiv preprint arXiv:2509.18757, 2025.URL https://doi.org/10.48550/arXiv.2509.18757.
RealSource [2025]	RealSource.Realsource world: a large-scale real-world dual-arm manipulation dataset.https://huggingface.co/datasets/RealSourceData/RealSource-World, 2025.URL https://huggingface.co/datasets/RealSourceData/RealSource-World.
Riochet et al. [2018]	Ronan Riochet, Mario Ynocente Castro, Mathieu Bernard, Adam Lerer, Rob Fergus, Véronique Izard, and Emmanuel Dupoux.IntPhys: a framework and benchmark for visual intuitive physics reasoning.IEEE Trans. Pattern Anal. Mach. Intell., 2018.URL https://arxiv.org/abs/1803.07616.
Robotics [2026]	Unitree Robotics.UniFoLM-WBT-Dataset.https://huggingface.co/collections/unitreerobotics/unifolm-wbt-dataset, 2026.
Rohmer et al. [2013]	Eric Rohmer, Surya P. N. Singh, and Marc Freese.V-rep: a versatile and scalable robot simulation framework.In IEEE/RSJ Int. Conf. Intell. Robots Syst., 2013.URL https://doi.org/10.1109/iros.2013.6696520.
Ropedia [2026]	Ropedia.Xperience-10M: A large-scale egocentric multimodal dataset with structured 3d/4d annotations, 2026.Dataset.
Ross et al. [2011]	Stéphane Ross, Geoffrey Gordon, and Drew Bagnell.A reduction of imitation learning and structured prediction to no-regret online learning.In Int. Conf. Artif. Intell. Stat., 2011.URL https://arxiv.org/abs/1011.0686.
Schonberger and Frahm [2016]	Johannes L. Schonberger and Jan-Michael Frahm.Structure-from-motion revisited.In IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2016.URL https://doi.org/10.1109/cvpr.2016.445.
Schoonbeek et al. [2024]	Tim J. Schoonbeek, Tim Houben, Hans Onvlee, Peter H. N. de With, and Fons van der Sommen.IndustReal: A dataset for procedure step recognition handling execution errors in egocentric videos in an industrial-like setting.In IEEE/CVF Winter Conf. Appl. Comput. Vis., 2024.URL https://arxiv.org/abs/2310.17323.
Schwenk et al. [2022]	Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi.A-OKVQA: A benchmark for visual question answering using world knowledge.In Eur. Conf. Comput. Vis., 2022.URL https://arxiv.org/abs/2206.01718.
Sedlacek et al. [2026]	Martin Sedlacek, Pavlo Yefanov, Georgy Ponimatkin, Jai Bardhan, Simon Pilc, Mederic Fourmy, Evangelos Kazakos, Cees G. M. Snoek, Josef Sivic, and Vladimir Petrik.REALM: A real-to-sim validated benchmark for generalization in robotic manipulation.IEEE Robot. Autom. Lett., 2026.URL https://arxiv.org/abs/2512.19562.
Sener et al. [2022]	Fadime Sener, Dibyadip Chatterjee, Daniel Shelepov, Kun He, Dipika Singhania, Robert Wang, and Angela Yao.Assembly101: A large-scale multi-view video dataset for understanding procedural activities.In IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2022.URL https://arxiv.org/abs/2203.14712.
Seo et al. [2025]	Mingyo Seo, H. Andy Park, Shenli Yuan, Yuke Zhu, and Luis Sentis.LEGATO: cross-embodiment imitation using a grasping tool.IEEE Robot. Autom. Lett., 2025.URL https://arxiv.org/abs/2411.03682.
Sermanet et al. [2024]	Pierre Sermanet, Tianli Ding, Jeffrey Zhao, Fei Xia, Debidatta Dwibedi, Keerthana Gopalakrishnan, Christine Chan, Gabriel Dulac-Arnold, et al.RoboVQA: multimodal long-horizon reasoning for robotics.In IEEE Int. Conf. Robot. Autom., 2024.URL https://arxiv.org/abs/2311.00899.
Sferrazza et al. [2024]	Carmelo Sferrazza, Dun-Ming Huang, Xingyu Lin, Youngwoon Lee, and Pieter Abbeel.HumanoidBench: Simulated humanoid benchmark for whole-body locomotion and manipulation.In Robot. Sci. Syst., 2024.URL https://arxiv.org/abs/2403.10506.
Shao et al. [2019]	Shuai Shao, Zeming Li, Tianyuan Zhang, Chao Peng, Gang Yu, Xiangyu Zhang, Jing Li, and Jian Sun.Objects365: A large-scale, high-quality dataset for object detection.In IEEE/CVF Int. Conf. Comput. Vis., 2019.URL https://doi.org/10.1109/iccv.2019.00852.
Sharma et al. [2026]	Ansh Kumar Sharma, Yixiang Sun, Ninghao Lu, Yunzhe Zhang, Jiarao Liu, and Sherry Yang.World-Gymnast: training robots with reinforcement learning in a world model.arXiv preprint arXiv:2602.02454, 2026.URL https://arxiv.org/abs/2602.02454.
Sharma et al. [2018]	Pratyusha Sharma, Lekha Mohan, Lerrel Pinto, and Abhinav Gupta.Multiple interactions made easy (mime): large scale demonstrations data for imitation.In Conf. Robot Learn., 2018.URL https://arxiv.org/abs/1810.07121.
Shaw et al. [2023]	Kenneth Shaw, Shikhar Bahl, and Deepak Pathak.VideoDex: learning dexterity from internet videos.In Conf. Robot Learn., 2023.URL https://arxiv.org/abs/2212.04498.
Shaw et al. [2024]	Kenneth Shaw, Shikhar Bahl, Aravind Sivakumar, Aditya Kannan, and Deepak Pathak.Learning dexterity from human hand motion in internet videos.Int. J. Robot. Res., 2024.URL https://doi.org/10.1177/02783649241227559.
Shen et al. [2025]	Yichao Shen, Fangyun Wei, Zhiying Du, Yaobo Liang, Yan Lu, Jiaolong Yang, Nanning Zheng, and Baining Guo.VideoVLA: Video generators can be generalizable robot manipulators.In Adv. Neural Inf. Process. Syst., 2025.URL https://arxiv.org/abs/2512.06963.
Shou et al. [2026]	Quanxin Shou, Fangqi Zhu, Shawn Chen, Puxin Yan, Zhengyang Yan, Yikun Miao, Xiaoyi Pang, Zicong Hong, Ruikai Shi, Hao Huang, et al.Halo: a unified vision-language-action model for embodied multimodal chain-of-thought reasoning.In Int. Conf. Mach. Learn., 2026.URL https://doi.org/10.48550/arXiv.2602.21157.
Shridhar et al. [2020]	Mohit Shridhar, Jesse Thomason, Daniel Gordon, Yonatan Bisk, Winson Han, Roozbeh Mottaghi, Luke Zettlemoyer, and Dieter Fox.ALFRED: a benchmark for interpreting grounded instructions for everyday tasks.In IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2020.URL https://arxiv.org/abs/1912.01734.
Shridhar et al. [2022]	Mohit Shridhar, Lucas Manuelli, and Dieter Fox.Perceiver-actor: A multi-task transformer for robotic manipulation.In Conf. Robot Learn., 2022.URL https://arxiv.org/abs/2209.05451.
Shukla et al. [2025]	Arth Shukla, Stone Tao, and Hao Su.ManiSkill-HAB: A benchmark for low-level manipulation in home rearrangement tasks.In Int. Conf. Learn. Represent., 2025.URL https://arxiv.org/abs/2412.13211.
Shukor et al. [2025]	Mustafa Shukor, Dana Aubakirova, Francesco Capuano, Pepijn Kooijmans, Steven Palma, Adil Zouitine, Michel Aractingi, Caroline Pascal, Martino Russi, Andres Marafioti, et al.SmolVLA: a vision-language-action model for affordable and efficient robotics.arXiv preprint arXiv:2506.01844, 2025.URL https://arxiv.org/abs/2506.01844.
Si and Yuan [2022]	Zilin Si and Wenzhen Yuan.Taxim: an example-based simulation model for gelsight tactile sensors.IEEE Robot. Autom. Lett., 2022.URL https://arxiv.org/abs/2109.04027.
Si et al. [2024]	Zilin Si, Gu Zhang, Qingwei Ben, Branden Romero, Zhou Xian, Chao Liu, and Chuang Gan.DIFFTACTILE: A physics-based differentiable tactile simulator for contact-rich robotic manipulation.In Int. Conf. Learn. Represent., 2024.URL https://arxiv.org/abs/2403.08716.
Sigurdsson et al. [2018]	Gunnar A. Sigurdsson, Abhinav Gupta, Cordelia Schmid, Ali Farhadi, and Karteek Alahari.Charades-ego: a large-scale dataset of paired third and first person videos.arXiv preprint arXiv:1804.09626, 2018.URL https://arxiv.org/abs/1804.09626.
Silberman et al. [2012]	Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus.Indoor segmentation and support inference from rgbd images.In Eur. Conf. Comput. Vis., 2012.URL https://doi.org/10.1007/978-3-642-33715-4_54.
Singh et al. [2021]	Amanpreet Singh, Guan Pang, Mandy Toh, Jing Huang, Wojciech Galuba, and Tal Hassner.TextOCR: towards large-scale end-to-end reasoning for arbitrary-shaped scene text.In IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2021.URL https://arxiv.org/abs/2105.05486.
Singh et al. [2016]	Suriya Singh, Chetan Arora, and C. V. Jawahar.First person action recognition using deep learned descriptors.In IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2016.URL https://doi.org/10.1109/cvpr.2016.287.
Sliwowski et al. [2025]	Daniel Sliwowski, Shail Jadav, Sergej Stanovcic, Jedrzej Orbik, Johannes Heidersberger, and Dongheui Lee.Reassemble: A multimodal dataset for contact-rich robotic assembly and disassembly.In Robot. Sci. Syst., 2025.URL https://arxiv.org/abs/2502.05086.
Song et al. [2026]	Wenxuan Song, Ziyang Zhou, Han Zhao, Jiayi Chen, Pengxiang Ding, Haodong Yan, Yuxin Huang, Feilong Tang, Donglin Wang, and Haoang Li.ReconVLA: Reconstructive vision-language-action model as effective robot perceiver.In AAAI Conf. Artif. Intell., 2026.URL https://ojs.aaai.org/index.php/AAAI/article/view/38921.
Srirama et al. [2024]	Mohan Kumar Srirama, Sudeep Dasari, Shikhar Bahl, and Abhinav Gupta.HRP: human affordances for robotic pre-training.In Robot. Sci. Syst., 2024.URL https://doi.org/10.15607/rss.2024.xx.068.
Straub et al. [2019]	Julian Straub, Thomas Whelan, Lingni Ma, Yufan Chen, Erik Wijmans, Simon Green, Jakob J. Engel, Raul Mur-Artal, et al.The replica dataset: a digital replica of indoor spaces.arXiv preprint arXiv:1906.05797, 2019.URL https://arxiv.org/abs/1906.05797.
Suffo et al. [2026]	Nele-Pauline Suffo, Pierre-Etienne Martin, Anas Suffo, Daniel Haun, and Manuel Bohn.Childlens: an egocentric video dataset for activity analysis in children.Behavior Research Methods, 2026.URL https://doi.org/10.3758/s13428-026-02982-6.
Sun et al. [2025]	Baoli Sun, Ning Wang, Xinzhu Ma, Anqi Zou, Yihang Lu, Chuixuan Fan, Zhihui Wang, Kun Lu, and Zhiyong Wang.RobAVA: a large-scale dataset and baseline towards video-based robotic arm action understanding.In IEEE/CVF Int. Conf. Comput. Vis., 2025.URL https://doi.org/10.1109/iccv51701.2025.01298.
Szot et al. [2021]	Andrew Szot, Alex Clegg, Eric Undersander, Erik Wijmans, Yili Zhao, John Turner, Noah Maestre, Mustafa Mukadam, et al.Habitat 2.0: training home assistants to rearrange their Habitat.In Adv. Neural Inf. Process. Syst., 2021.URL https://arxiv.org/abs/2106.14405.
Szot et al. [2024]	Andrew Szot, Max Schwarzer, Harsh Agrawal, Bogdan Mazoure, Walter Talbott, Katherine Metcalf, Natalie Mackraz, Devon Hjelm, and Alexander Toshev.Large language models as generalizable policies for embodied tasks.In Int. Conf. Learn. Represent., 2024.URL https://arxiv.org/abs/2310.17722.
Tang et al. [2019]	Yansong Tang, Dajun Ding, Yongming Rao, Yu Zheng, Danyang Zhang, Lili Zhao, Jiwen Lu, and Jie Zhou.Coin: a large-scale dataset for comprehensive instructional video analysis.In IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2019.URL https://doi.org/10.1109/cvpr.2019.00130.
Tao et al. [2025a]	Stone Tao, Fanbo Xiang, Arth Shukla, Yuzhe Qin, Xander Hinrichsen, Xiaodi Yuan, Chen Bao, Xinsong Lin, et al.ManiSkill3: GPU parallelized robotics simulation and rendering for generalizable embodied AI.In Robot. Sci. Syst., 2025a.URL https://arxiv.org/abs/2410.00425.
Tao et al. [2025b]	Tony Tao, Mohan Kumar Srirama, Jason Jingzhou Liu, Kenneth Shaw, and Deepak Pathak.DexWild: dexterous human interactions for in-the-wild robot policies.In Robot. Sci. Syst., 2025b.URL https://arxiv.org/abs/2505.07813.
Team et al. [2026a]	ACE-Brain Team, Ziyang Gong, Haoming Gu, Zehang Luo, Tianyi Zhang, Tao Tao, Yixiao Chi, Zhe Liu, et al.ACE-Brain-0.5: A unified embodied foundational model for physical agentic AI.arXiv preprint arXiv:2607.04426, 2026a.URL https://arxiv.org/abs/2607.04426.
Team [2026]	AgiBot World Team.AgiBot world 2026.https://huggingface.co/datasets/agibot-world/AgiBotWorld2026, 2026.URL https://huggingface.co/datasets/agibot-world/AgiBotWorld2026.
Team et al. [2026b]	GigaBrain Team, Boyuan Wang, Bohan Li, Chaojun Ni, Guan Huang, Guosheng Zhao, Hao Li, Jie Li, Jindi Lv, Jingyu Liu, et al.GigaBrain-0.5 M*: a VLA that learns from world model-based reinforcement learning.arXiv preprint arXiv:2602.12099, 2026b.URL https://arxiv.org/abs/2602.12099.
Team et al. [2026c]	GigaWorld Team, Angen Ye, Angyuan Ma, Boyuan Wang, Chaojun Ni, Fangzheng Ye, Guan Huang, Guo Li, et al.GigaWorld-Policy-0.5: A faster and stronger WAM empowered by AutoResearch.arXiv preprint arXiv:2607.13960, 2026c.URL https://arxiv.org/abs/2607.13960.
Team et al. [2026d]	Motubrain Team, Chendong Xiang, Fan Bao, Haitian Liu, Hengkai Tan, Hongzhe Bi, James Li, Jiabao Liu, et al.Motubrain: An advanced world action model for robot control.arXiv preprint arXiv:2604.27792, 2026d.URL https://arxiv.org/abs/2604.27792.
Team et al. [2024]	Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, et al.Octo: an open-source generalist robot policy.In Robot. Sci. Syst., 2024.URL https://doi.org/10.15607/rss.2024.xx.090.
Team et al. [2026e]	Xiaomi Robotics Team, Jun Guo, Piaopiao Jin, Jason Li, Peiyan Li, Yingyan Li, Futeng Liu, Wanli Peng, Optimus Qin, Yifei Su, et al.Xiaomi-Robotics-1: Scaling vision-language-action models with over 100k hours of real-world trajectories.arXiv preprint arXiv:2607.15330, 2026e.URL https://arxiv.org/abs/2607.15330.
Tian et al. [2026]	Yang Tian, Yuyin Yang, Yiman Xie, Zetao Cai, Xu Shi, Ning Gao, Hangxu Liu, Xuekun Jiang, et al.InternData-A1: pioneering high-fidelity synthetic data for pre-training generalist policy.In IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2026.URL https://doi.org/10.48550/arXiv.2511.16651.
Todorov et al. [2012]	Emanuel Todorov, Tom Erez, and Yuval Tassa.MuJoCo: a physics engine for model-based control.In IEEE/RSJ Int. Conf. Intell. Robots Syst., 2012.URL https://doi.org/10.1109/IROS.2012.6386109.
VanVoorst et al. [2026]	Brian VanVoorst, Nicholas Walczak, Christopher Gilleo, Charles Meissner, Fabio Felix, Iran Roman, Bea Steers, Claudio Silva, Yuhan Shen, Zijia Lu, Shih-Po Lee, and Ehsan Elhamifar.Egomagic- an egocentric video field medicine dataset for training perception algorithms.arXiv preprint arXiv:2604.22036, 2026.URL https://arxiv.org/abs/2604.22036.
Vuong et al. [2024]	An Dinh Vuong, Minh Nhat Vu, Hieu Le, Baoru Huang, Binh Huynh, Thieu Vo, Andreas Kugi, and Anh Nguyen.Grasp-anything: large-scale grasp dataset from foundation models.In IEEE Int. Conf. Robot. Autom., 2024.URL https://doi.org/10.1109/icra57147.2024.10611277.
Wald et al. [2019]	Johanna Wald, Armen Avetisyan, Nassir Navab, Federico Tombari, and Matthias NieSSner.RIO: 3D object instance re-localization in changing indoor environments.In IEEE/CVF Int. Conf. Comput. Vis., 2019.URL https://arxiv.org/abs/1908.06109.
Wald et al. [2020]	Johanna Wald, Helisa Dhamo, Nassir Navab, and Federico Tombari.Learning 3D semantic scene graphs from 3D indoor reconstructions.In IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2020.URL https://doi.org/10.1109/cvpr42600.2020.00402.
Walke et al. [2023]	Homer Rich Walke, Kevin Black, Tony Z Zhao, Quan Vuong, Chongyi Zheng, Philippe Hansen-Estruch, Andre Wang He, Vivek Myers, Moo Jin Kim, Max Du, et al.BridgeData v2: a dataset for robot learning at scale.In Conf. Robot Learn., 2023.URL https://arxiv.org/abs/2308.12952.
Wang et al. [2024a]	Chen Wang, Haochen Shi, Weizhuo Wang, Ruohan Zhang, Li Fei-Fei, and C Karen Liu.DexCap: scalable and portable mocap data collection system for dexterous manipulation.In Robot. Sci. Syst., 2024a.URL https://arxiv.org/abs/2403.07788.
Wang et al. [2026a]	Kaixuan Wang, Tianxing Chen, Jiawei Liu, Honghao Su, Shaolong Zhu, Minxuan Wang, Zixuan Li, Yue Chen, et al.ManiTwin: Scaling data-generation-ready digital object dataset to 100k.arXiv preprint arXiv:2603.16866, 2026a.URL https://arxiv.org/abs/2603.16866.
Wang et al. [2024b]	Lirui Wang, Yiyang Ling, Zhecheng Yuan, Mohit Shridhar, Chen Bao, Yuzhe Qin, Bailin Wang, Huazhe Xu, and Xiaolong Wang.GenSim: generating robotic simulation tasks via large language models.In Int. Conf. Learn. Represent., 2024b.URL https://arxiv.org/abs/2310.01361.
Wang et al. [2026b]	Luming Wang, Hao Shi, Jiajun Zhai, Kailun Yang, and Kaiwei Wang.EgoEV-HandPose: Egocentric 3D hand pose estimation and gesture recognition with stereo event cameras.arXiv preprint arXiv:2605.12297, 2026b.URL https://arxiv.org/abs/2605.12297.
Wang et al. [2026c]	Qiuyue Wang, Mingsheng Li, Jian Guan, Jinhui Ye, Sicheng Xie, Yitao Liu, Junhao Chen, Zhixuan Liang, et al.Qwen-VLA: Unifying vision-language-action modeling across tasks, environments, and robot embodiments.arXiv preprint arXiv:2605.30280, 2026c.URL https://arxiv.org/abs/2605.30280.
Wang et al. [2023a]	Ruicheng Wang, Jialiang Zhang, Jiayi Chen, Yinzhen Xu, Puhao Li, Tengyu Liu, and He Wang.DexGraspNet: a large-scale robotic dexterous grasp dataset for general objects based on simulation.In IEEE Int. Conf. Robot. Autom., 2023a.URL https://arxiv.org/abs/2210.02697.
Wang et al. [2022]	Shaoxiong Wang, Mike Lambeta, Po-Wei Chou, and Roberto Calandra.Tacto: a fast, flexible, and open-source simulator for high-resolution vision-based tactile sensors.IEEE Robot. Autom. Lett., 2022.URL https://doi.org/10.1109/LRA.2022.3146945.
Wang et al. [2025a]	Wenhao Wang, Kehe Ye, Xinyu Zhou, Tianxing Chen, Cao Min, Qiaoming Zhu, Xiaokang Yang, Ping Luo, Yongjian Shen, Yang Yang, et al.Fieldgen: From teleoperated pre-manipulation trajectories to field-guided data generation.arXiv preprint arXiv:2510.20774, 2025a.
Wang et al. [2023b]	Xin Wang, Taein Kwon, Mahdi Rad, Bowen Pan, Ishani Chakraborty, Sean Andrist, Dan Bohus, Ashley Feniello, Bugra Tekin, Felipe Vieira Frujeri, Neel Joshi, and Marc Pollefeys.Holoassist: an egocentric human interaction dataset for interactive AI assistants in the real world.In IEEE/CVF Int. Conf. Comput. Vis., 2023b.URL https://arxiv.org/abs/2309.17024.
Wang et al. [2026d]	Ye Wang, Sipeng Zheng, Hao Luo, Wanpeng Zhang, Haoqi Yuan, Chaoyi Xu, Haiweng Xu, Yicheng Feng, Mingyang Yu, Zhiyu Kang, et al.Rethinking visual-language-action model scaling: alignment, mixture, and regularization.arXiv preprint arXiv:2602.09722, 2026d.URL https://arxiv.org/abs/2602.09722.
Wang et al. [2025b]	Yi Ru Wang, Carter Ung, Christopher Tan, Grant Tannert, Jiafei Duan, Josephine Li, Anh Le, Rishabh Oswal, Markus Grotz, Wilbert Pumacay, Yuquan Deng, Ranjay Krishna, Dieter Fox, and Siddhartha Srinivasa.RoboEval: where robotic manipulation meets structured and scalable evaluation.arXiv preprint arXiv:2507.00435, 2025b.URL https://arxiv.org/abs/2507.00435.
Wang et al. [2024c]	Yufei Wang, Zhou Xian, Feng Chen, Tsun-Hsuan Wang, Yian Wang, Katerina Fragkiadaki, Zackory Erickson, David Held, and Chuang Gan.RoboGen: towards unleashing infinite data for automated robot learning via generative simulation.In Int. Conf. Mach. Learn., 2024c.URL https://arxiv.org/abs/2311.01455.
Wang et al. [2025c]	Yuran Wang, Ruihai Wu, Yue Chen, Jiarui Wang, Jiaqi Liang, Ziyu Zhu, Haoran Geng, Jitendra Malik, Pieter Abbeel, and Hao Dong.Dexgarmentlab: Dexterous garment manipulation environment with generalizable policy.In Adv. Neural Inf. Process. Syst., 2025c.URL https://arxiv.org/abs/2505.11032.
Wang et al. [2024d]	Zihan Wang, Xiangyang Li, Jiahao Yang, Yeqi Liu, and Shuqiang Jiang.Sim-to-real transfer via 3D feature fields for vision-and-language navigation.In Conf. Robot Learn., 2024d.URL https://doi.org/10.48550/arXiv.2406.09798.
Wang [2026]	Ziming Wang.UMI-3D: extending universal manipulation interface from vision-limited to 3D spatial perception.arXiv preprint arXiv:2604.14089, 2026.URL https://arxiv.org/abs/2604.14089.
Weihs et al. [2022]	Luca Weihs, Amanda Rose Yuile, Renée Baillargeon, Cynthia Fisher, Gary Marcus, Roozbeh Mottaghi, and Aniruddha Kembhavi.Benchmarking progress to infant-level physical reasoning in AI.Trans. Mach. Learn. Res., 2022.URL https://openreview.net/forum?id=9NjqD9i48M.
Wen et al. [2025]	Junjie Wen, Yichen Zhu, Minjie Zhu, Zhibin Tang, Jinming Li, Zhongyi Zhou, Xiaoyu Liu, Chaomin Shen, Yaxin Peng, and Feifei Feng.Diffusion-VLA: scaling robot foundation models via unified diffusion and autoregression.In Int. Conf. Mach. Learn., 2025.URL https://arxiv.org/abs/2412.03293.
Wu et al. [2024a]	Hongtao Wu, Ya Jing, Chi-Lam Cheang, Guangzeng Chen, Jiafeng Xu, Xinghang Li, Minghuan Liu, Hang Li, and Tao Kong.Unleashing large-scale video generative pre-training for visual robot manipulation.In Int. Conf. Learn. Represent., 2024a.URL https://arxiv.org/abs/2312.13139.
Wu et al. [2025a]	Kun Wu, Chengkai Hou, Jiaming Liu, Zhengping Che, Xiaozhu Ju, Zhuqin Yang, Meng Li, Yinuo Zhao, et al.RoboMIND: benchmark on multi-embodiment intelligence normative data for robot manipulation.In Robot. Sci. Syst., 2025a.URL https://doi.org/10.48550/arXiv.2412.13877.
Wu et al. [2025b]	Kun Wu, Ning Liu, Zhen Zhao, Di Qiu, Jinming Li, Zhengping Che, Zhiyuan Xu, Qinru Qiu, and Jian Tang.Learning from imperfect demonstrations with self-supervision for robotic manipulation.In IEEE Int. Conf. Robot. Autom., 2025b.URL https://doi.org/10.1109/icra55743.2025.11127918.
Wu et al. [2026a]	Longyan Wu, Jieji Ren, Chenghang Jiang, Junxi Zhou, Shijia Peng, Ran Huang, Guoying Gu, Li Chen, and Hongyang Li.TAMEn: tactile-aware manipulation engine for closed-loop data collection in contact-rich tasks.arXiv preprint arXiv:2604.07335, 2026a.URL https://arxiv.org/abs/2604.07335.
Wu et al. [2026b]	Longyan Wu, Checheng Yu, Jieji Ren, Li Chen, Yufei Jiang, Ran Huang, Guoying Gu, and Hongyang Li.FreeTacMan: robot-free visuo-tactile data collection system for contact-rich manipulation.In IEEE Int. Conf. Robot. Autom., 2026b.URL https://arxiv.org/abs/2506.01941.
Wu et al. [2023a]	Philipp Wu, Alejandro Escontrela, Danijar Hafner, Pieter Abbeel, and Ken Goldberg.Daydreamer: world models for physical robot learning.In Conf. Robot Learn., 2023a.URL https://arxiv.org/abs/2206.14176.
Wu et al. [2024b]	Philipp Wu, Yide Shentu, Zhongke Yi, Xingyu Lin, and Pieter Abbeel.GELLO: a general, low-cost, and intuitive teleoperation framework for robot manipulators.In IEEE/RSJ Int. Conf. Intell. Robots Syst., 2024b.URL https://arxiv.org/abs/2309.13037.
Wu et al. [2025c]	Shihan Wu, Xuecheng Liu, Shaoxuan Xie, Pengwei Wang, Xinghang Li, Bowen Yang, Zhe Li, Kai Zhu, Hongyu Wu, Yiheng Liu, et al.RoboCoin: an open-sourced bimanual robotic data collection for integrated manipulation.arXiv preprint arXiv:2511.17441, 2025c.URL https://arxiv.org/abs/2511.17441.
Wu et al. [2026c]	Tao Wu, Chuhao Zhou, Haozhi Cao, Yen Heng Wong, Lin Gu, and Jianfei Yang.NoisyEQA: Benchmarking embodied question answering with imperfect queries from non-expert users.Robot Learning, 2026c.URL https://arxiv.org/abs/2412.10726.
Wu et al. [2026d]	Tao Wu, Chuhao Zhou, Guangyu Zhao, Haozhi Cao, Yewen Pu, and Jianfei Yang.When robots should say “i don’t know”: Benchmarking abstention in embodied question answering.In IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2026d.URL https://arxiv.org/abs/2512.04597.
Wu et al. [2023b]	Tong Wu, Jiarui Zhang, Xiao Fu, Yuxin Wang, Jiawei Ren, Liang Pan, Wayne Wu, Lei Yang, Jiaqi Wang, Chen Qian, Dahua Lin, and Ziwei Liu.OmniObject3D: Large-vocabulary 3D object dataset for realistic perception, reconstruction and generation.In IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2023b.URL https://arxiv.org/abs/2301.07525.
Wu et al. [2026e]	Wei Wu, Fan Lu, Yunnan Wang, Shuai Yang, Shi Liu, Fangjing Wang, Qian Zhu, He Sun, et al.A pragmatic VLA foundation model.arXiv preprint arXiv:2601.18692, 2026e.URL https://arxiv.org/abs/2601.18692.
Wu et al. [2026f]	Wei Wu, Fangjing Wang, Fan Lu, He Sun, Shi Liu, Yunnan Wang, Yibin Yan, Yong Wang, et al.From foundation to application: Improving VLA models in practice.arXiv preprint arXiv:2607.06403, 2026f.URL https://arxiv.org/abs/2607.06403.
Xi et al. [2026]	Ziheng Xi, Jiayi Yu, Yitao Wang, Yanbo Duan, Jianjiang Feng, and Jie Zhou.EgoEMG: A multimodal egocentric dataset with bilateral EMG and vision for hand pose estimation.arXiv preprint arXiv:2605.05712, 2026.URL https://arxiv.org/abs/2605.05712.
Xiang et al. [2020]	Fanbo Xiang, Yuzhe Qin, Kaichun Mo, Yikuan Xia, Hao Zhu, Fangchen Liu, Minghua Liu, Hanxiao Jiang, et al.Sapien: a simulated part-based interactive environment.In IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2020.URL https://arxiv.org/abs/2003.08515.
Xiao et al. [2025]	Junjin Xiao, Yandan Yang, Xinyuan Chang, Ronghan Chen, Feng Xiong, Mu Xu, Wei-Shi Zheng, and Qing Zhang.World-Env: leveraging world model as a virtual environment for VLA post-training.arXiv preprint arXiv:2509.24948, 2025.URL https://arxiv.org/abs/2509.24948.
Xiao et al. [2026]	Junjin Xiao, Yandan Yang, Xinyuan Chang, Ronghan Chen, Feng Xiong, Mu Xu, Wei-Shi Zheng, and Qing Zhang.RehearseVLA: simulated post-training for VLAs with a physically consistent world model.In IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2026.URL https://arxiv.org/abs/2509.24948v4.
Xie et al. [2025]	Sicheng Xie, Lingchen Meng, Zijie Diao, Haidong Cao, Zhiying Du, Shuyuan Tu, Jiaqi Leng, Qiuyue Wang, Mingsheng Li, Shuai Bai, et al.Unify robot actions in camera frame.arXiv preprint arXiv:2511.17001, 2025.URL https://arxiv.org/abs/2511.17001.
Xu et al. [2026]	Chaoyi Xu, Yixuan Jiang, Jiahui Huan, Yuhui Fu, Haoyu Zhou, Weitian Yuan, Jiayi Yu, Wanpeng Zhang, Haoqi Yuan, and Zongqing Lu.RealDexUMI: a wearable universal manipulation interface for dexterous robot learning.arXiv preprint arXiv:2606.06033, 2026.URL https://arxiv.org/abs/2606.06033.
Xu et al. [2025a]	Mengda Xu, Han Zhang, Yifan Hou, Zhenjia Xu, Linxi Fan, Manuela Veloso, and Shuran Song.DexUMI: using human hand as the universal manipulation interface for dexterous manipulation.In Conf. Robot Learn., 2025a.URL https://arxiv.org/abs/2505.21864.
Xu et al. [2025b]	Xiaomeng Xu, Yifan Hou, Chendong Xin, Zeyi Liu, and Shuran Song.Compliant residual DAgger: improving real-world contact-rich manipulation with human corrections.In Adv. Neural Inf. Process. Syst., 2025b.URL https://arxiv.org/abs/2506.16685.
Xu et al. [2025c]	Yue Xu, Litao Wei, Pengyu An, Qingyu Zhang, and Yong-Lu Li.Exumi: extensible robot teaching system with action-aware task-agnostic tactile representation.In Conf. Robot Learn., 2025c.URL https://doi.org/10.48550/arXiv.2509.14688.
Yan et al. [2026]	Ningyu Yan, Shuai Wang, Xing Shen, Hui Wang, Hanqing Wang, Yang Xiang, and Jiangmiao Pang.Tac2Real: reliable and GPU visuotactile simulation for online reinforcement learning and zero-shot real-world deployment.arXiv preprint arXiv:2603.28475, 2026.URL https://arxiv.org/abs/2603.28475.
Yang et al. [2022]	Fengyu Yang, Chenyang Ma, Jiacheng Zhang, Jing Zhu, Wenzhen Yuan, and Andrew Owens.Touch and go: learning from human-collected vision and touch.In Adv. Neural Inf. Process. Syst., 2022.URL https://arxiv.org/abs/2211.12498.
Yang et al. [2026a]	Jiazhi Yang, Kunyang Lin, Jinwei Li, Wencong Zhang, Tianwei Lin, Longyan Wu, Zhizhong Su, Hao Zhao, Ya-Qin Zhang, Li Chen, et al.RISE: self-improving robot policy with compositional world model.arXiv preprint arXiv:2602.11075, 2026a.URL https://arxiv.org/abs/2602.11075.
Yang et al. [2025a]	Jingkang Yang, Shuai Liu, Hongming Guo, Yuhao Dong, Xiamengwei Zhang, Sicheng Zhang, Pengyun Wang, Zitang Zhou, et al.EgoLife: Towards egocentric life assistant.In IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2025a.URL https://arxiv.org/abs/2503.03803.
Yang et al. [2025b]	Ruihan Yang, Qinxi Yu, Yecheng Wu, Rui Yan, Borui Li, An-Chieh Cheng, Xueyan Zou, Yunhao Fang, Xuxin Cheng, Ri-Zhao Qiu, et al.EgoVLA: learning vision-language-action models from egocentric human videos.arXiv preprint arXiv:2507.12440, 2025b.URL https://arxiv.org/abs/2507.12440.
Yang et al. [2024]	Sherry Yang, Yilun Du, Kamyar Ghasemipour, Jonathan Tompson, Leslie Kaelbling, Dale Schuurmans, and Pieter Abbeel.Learning interactive real-world simulators.In Int. Conf. Learn. Represent., 2024.URL https://openreview.net/forum?id=sFyTZEqmUY.
Yang et al. [2026b]	Yandan Yang, Shuang Zeng, Tong Lin, Xinyuan Chang, Dekang Qi, Junjin Xiao, Haoyun Liu, Ronghan Chen, et al.ABot-M0: VLA foundation model for robotic manipulation with action manifold learning.arXiv preprint arXiv:2602.11236, 2026b.URL https://arxiv.org/abs/2602.11236.
Ye et al. [2026a]	Angen Ye, Boyuan Wang, Chaojun Ni, Guan Huang, Guosheng Zhao, Hao Li, Hengtao Li, Jie Li, et al.GigaWorld-Policy: An efficient action-centered world–action model.arXiv preprint arXiv:2603.17240, 2026a.URL https://arxiv.org/abs/2603.17240.
Ye et al. [2025a]	Jianglong Ye, Keyi Wang, Chengjing Yuan, Ruihan Yang, Yiquan Li, Jiyue Zhu, Yuzhe Qin, Xueyan Zou, and Xiaolong Wang.Dex1b: learning with 1b demonstrations for dexterous manipulation.In Robot. Sci. Syst., 2025a.URL https://doi.org/10.48550/arXiv.2506.17198.
Ye et al. [2025b]	Seonghyeon Ye, Joel Jang, Byeongguk Jeon, Se June Joo, Jianwei Yang, Baolin Peng, Ajay Mandlekar, Reuben Tan, Yu-Wei Chao, Bill Yuchen Lin, et al.Latent action pretraining from videos.In Int. Conf. Learn. Represent., 2025b.URL https://arxiv.org/abs/2410.11758.
Ye et al. [2026b]	Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, Shenyuan Gao, Sihyun Yu, George Kurian, Suneel Indupuru, You Liang Tan, et al.World action models are zero-shot policies.arXiv preprint arXiv:2602.15922, 2026b.URL https://arxiv.org/abs/2602.15922.
Ye et al. [2026c]	Yifan Ye, Jun Cen, Jing Chen, and Zhihe Lu.Self-evolved imitation learning in simulated world.IEEE Robotics and Automation Letters, 11(6):6967–6974, 2026c.10.1109/LRA.2026.3683594.URL https://doi.org/10.1109/LRA.2026.3683594.
Yeshwanth et al. [2023]	Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai.ScanNet++: a high-fidelity dataset of 3D indoor scenes.In IEEE/CVF Int. Conf. Comput. Vis., 2023.URL https://doi.org/10.1109/iccv51070.2023.00008.
Yi et al. [2020]	Kexin Yi, Chuang Gan, Yunzhu Li, Pushmeet Kohli, Jiajun Wu, Antonio Torralba, and Joshua B. Tenenbaum.CLEVRER: collision events for video representation and reasoning.In Int. Conf. Learn. Represent., 2020.URL https://arxiv.org/abs/1910.01442.
Yin et al. [2026a]	Chenghao Yin, Da Huang, Di Yang, Jichao Wang, Nanshu Zhao, Chen Xu, Wenjun Sun, Linjie Hou, et al.Genie sim 3.0 : a high-fidelity comprehensive simulation platform for humanoid robot.arXiv preprint arXiv:2601.02078, 2026a.URL https://arxiv.org/abs/2601.02078.
Yin et al. [2026b]	Tenny Yin, Zhiting Mei, Zhonghe Zheng, Miyu Yamane, David Wang, Jade Sceats, Samuel M. Bateman, Lihan Zha, Apurva Badithela, Ola Shorinwa, et al.PlayWorld: learning robot world models from autonomous play.arXiv preprint arXiv:2603.09030, 2026b.URL https://arxiv.org/abs/2603.09030.
Yu et al. [2026a]	En Yu, Haoran Lv, Jianjian Sun, Kangheng Lin, Ruitao Zhang, Yukang Shi, Yuyang Chen, Ze Chen, et al.DM0: An embodied-native vision-language-action model towards physical AI.arXiv preprint arXiv:2602.14974, 2026a.URL https://arxiv.org/abs/2602.14974.
Yu et al. [2016]	Licheng Yu, Patrick Poirson, Shan Yang, Alexander C. Berg, and Tamara L. Berg.Modeling context in referring expressions.In Eur. Conf. Comput. Vis., 2016.URL https://arxiv.org/abs/1608.00272.
Yu et al. [2026b]	Qize Yu, Jiadi You, Yuran Wang, Jiaqi Liang, Bowen Ping, Yang Tian, Yue Chen, Minghong Cai, Zeying Gong, Ruihai Wu, et al.AffordanceVLA: a vision-language-action model empowering action generation through affordance-aware understanding.arXiv preprint arXiv:2606.06155, 2026b.URL https://arxiv.org/abs/2606.06155.
Yu et al. [2026c]	Ryan Yu, Pushi Zhang, Starrick Liu, Brae Liu, Miracle Kang, Shalfun Li, Lights Shi, Ellie Ma, et al.Wall-OSS-0.5 technical report.arXiv preprint arXiv:2605.30877, 2026c.URL https://arxiv.org/abs/2605.30877.
Yu et al. [2018]	Tianhe Yu, Chelsea Finn, Annie Xie, Sudeep Dasari, Tianhao Zhang, Pieter Abbeel, and Sergey Levine.One-shot imitation from observing humans via domain-adaptive meta-learning.In Robot. Sci. Syst., 2018.URL https://arxiv.org/abs/1802.01557.
Yu et al. [2019]	Tianhe Yu, Deirdre Quillen, Zhanpeng He, Ryan Julian, Avnish Narayan, Hayden Shively, Adithya Bellathur, Karol Hausman, Chelsea Finn, and Sergey Levine.Meta-world: a benchmark and evaluation for multi-task and meta reinforcement learning.In Conf. Robot Learn., 2019.URL http://arxiv.org/abs/1910.10897.
Yu et al. [2026d]	Xumin Yu, Zuyan Liu, Ziyi Wang, He Zhang, Yongming Rao, Fangfu Liu, Yani Zhang, Ruowen Zhao, et al.HY-Embodied-0.5: embodied foundation models for real-world agents.arXiv preprint arXiv:2604.07430, 2026d.URL https://arxiv.org/abs/2604.07430.
Yuan et al. [2024a]	Chengbo Yuan, Chuan Wen, Tong Zhang, and Yang Gao.General flow as foundation affordance for scalable robot learning.In Conf. Robot Learn., 2024a.URL https://arxiv.org/abs/2401.11439.
Yuan et al. [2025]	Chengbo Yuan, Rui Zhou, Mengzhen Liu, Yingdong Hu, Shengjie Wang, Li Yi, Chuan Wen, Shanghang Zhang, and Yang Gao.MotionTrans: human vr data enable motion-level learning for robotic manipulation policies.arXiv preprint arXiv:2509.17759, 2025.URL https://arxiv.org/abs/2509.17759.
Yuan et al. [2026]	Haoqi Yuan, Zhixuan Liang, Anzhe Chen, Ye Wang, Haoyang Li, Pei Lin, Yiyang Huang, Zixing Lei, Tong Zhang, Jiazhao Zhang, et al.Qwen-robotmanip technical report: alignment unlocks scale for robotic manipulation foundation models.arXiv preprint arXiv:2606.17846, 2026.URL https://arxiv.org/abs/2606.17846.
Yuan et al. [2024b]	Wentao Yuan, Jiafei Duan, Valts Blukis, Wilbert Pumacay, Ranjay Krishna, Adithyavairavan Murali, Arsalan Mousavian, and Dieter Fox.RoboPoint: a vision-language model for spatial affordance prediction in robotics.In Conf. Robot Learn., 2024b.URL https://arxiv.org/abs/2406.10721.
Zakka et al. [2025]	Kevin Zakka, Baruch Tabanpour, Qiayuan Liao, Mustafa Haiderbhai, Samuel Holt, Jing Yuan Luo, Arthur Allshire, Erik Frey, et al.MuJoCo playground.In Robot. Sci. Syst., 2025.URL https://doi.org/10.48550/arXiv.2502.08844.Demo paper.
Zala et al. [2023]	Abhay Zala, Jaemin Cho, Satwik Kottur, Xilun Chen, Barlas Oguz, Yashar Mehdad, and Mohit Bansal.Hierarchical video-moment retrieval and step-captioning.In IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2023.URL https://arxiv.org/abs/2303.16406.
Zeng et al. [2026]	Xianchao Zeng, Xinyu Zhou, Youcheng Li, Jiayou Shi, Tianle Li, Liangming Chen, Lei Ren, and Yong-Lu Li.Diagnose, correct, and learn from manipulation failures via visual symbols.In IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2026.URL https://arxiv.org/abs/2512.02787.
Zhang et al. [2026a]	Anran Zhang, Hanzhi Chen, Yannick Burkhardt, Yao Zhong, Johannes Betz, Helen Oleynikova, and Stefan Leutenegger.Actron3d: learning actionable neural functions from videos for transferable robotic manipulation.In IEEE Int. Conf. Robot. Autom., 2026a.URL https://arxiv.org/abs/2510.12971.
Zhang et al. [2025a]	Chaofan Zhang, Shaowei Cui, Jingyi Hu, Tianyu Jiang, Tiandong Zhang, Rui Wang, and Shuo Wang.TacFlex: multimode tactile imprints simulation for visuotactile sensors with coating patterns.IEEE Trans. Robot., 2025a.URL https://doi.org/10.1109/TRO.2025.3576970.
Zhang et al. [2026b]	Chubin Zhang, Jianan Wang, Zifeng Gao, Yue Su, Tianru Dai, Cai Zhou, Jiwen Lu, and Yansong Tang.Clap: contrastive latent action pretraining for learning vision-language-action models from human videos.arXiv preprint arXiv:2601.04061, 2026b.URL https://arxiv.org/abs/2601.04061.
Zhang et al. [2026c]	Gu Zhang, Qicheng Xu, Haozhe Zhang, Jianhan Ma, Long He, Yiming Bao, et al.UniDex: a robot foundation suite for universal dexterous hand control from egocentric human videos.In IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2026c.URL https://arxiv.org/abs/arXiv.2603.22264.
Zhang et al. [2025b]	Han Zhang, Songbo Hu, Zhecheng Yuan, and Huazhe Xu.DOGlove: Dexterous manipulation with a low-cost open-source haptic force feedback glove.In Robot. Sci. Syst., 2025b.URL https://arxiv.org/abs/2502.07730.
Zhang et al. [2026d]	He Zhang, Lingzhu Xiang, Haitao Lin, Zeyu Huang, Minghui Wang, Dingyan Zhong, Yubo Dong, Yihao Wu, et al.Hy-Embodied-0.5-VLA: From vision-language-action models to a real-world robot learning stack.arXiv preprint arXiv:2606.14409, 2026d.URL https://arxiv.org/abs/2606.14409.
Zhang et al. [2025c]	Jiahui Zhang, Ze Huang, Chun Gu, Zipei Ma, and Li Zhang.Reinforcing action policies by prophesying.arXiv preprint arXiv:2511.20633, 2025c.URL https://arxiv.org/abs/2511.20633.
Zhang et al. [2024]	Jialiang Zhang, Haoran Liu, Danshi Li, Xinqiang Yu, Haoran Geng, Yufei Ding, Jiayi Chen, and He Wang.DexGraspNet 2.0: Learning generative dexterous grasping in large-scale synthetic cluttered scenes.In Conf. Robot Learn., 2024.URL https://arxiv.org/abs/2410.23004.
Zhang et al. [2025d]	Jianke Zhang, Yanjiang Guo, Yucheng Hu, Xiaoyu Chen, Xiang Zhu, and Jianyu Chen.Up-vla: A unified understanding and prediction model for embodied agent.In Proceedings of the 42nd International Conference on Machine Learning, 2025d.
Zhang et al. [2026e]	Qihang Zhang, Lin Li, Luyao Zhang, Shuai Yang, Yiming Luo, Shuaiting Li, Ruilin Wang, Junke Wang, et al.Native video-action pretraining for generalizable robot control.arXiv preprint arXiv:2607.08639, 2026e.URL https://arxiv.org/abs/2607.08639.
Zhang et al. [2025e]	Shiduo Zhang, Zhe Xu, Peiju Liu, Xiaopeng Yu, Yuan Li, Qinghui Gao, Zhaoye Fei, Zhangyue Yin, Zuxuan Wu, Yu-Gang Jiang, and Xipeng Qiu.VLABench: a large-scale benchmark for Language-Conditioned robotics manipulation with long-horizon reasoning tasks.In IEEE/CVF Int. Conf. Comput. Vis., 2025e.URL https://arxiv.org/abs/2412.18194.
Zhang et al. [2026f]	Tianle Zhang, Zhihao Yuan, Dafeng Chi, Peidong Liu, Dongwei Li, Kejun Hu, Likui Zhang, Junnan Nie, et al.JoyAI-RA 0.1: A foundation model for robotic autonomy.arXiv preprint arXiv:2604.20100, 2026f.URL https://arxiv.org/abs/2604.20100.
Zhang et al. [2026g]	Tianyi Zhang, Haonan Duan, Haoran Hao, Yu Qiao, Jifeng Dai, and Zhi Hou.Grounding actions in camera space: Observation-centric vision-language-action policy.In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 18782–18790, 2026g.
Zhang et al. [2025f]	Wenyao Zhang, Hongsi Liu, Zekun Qi, Yunnan Wang, Xinqiang Yu, Jiazhao Zhang, Runpei Dong, Jiawei He, et al.DreamVLA: a vision-language-action model dreamed with comprehensive world knowledge.In Adv. Neural Inf. Process. Syst., 2025f.URL https://arxiv.org/abs/2507.04447.
Zhang et al. [2025g]	Yi Zhang, Che Liu, Xiancong Ren, Hanchu Ni, Shuai Zhang, Zeyuan Ding, Jiayu Hu, Hanzhe Shan, et al.Pelican-VL 1.0: a foundation brain model for embodied intelligence.arXiv preprint arXiv:2511.00108, 2025g.URL https://arxiv.org/abs/2511.00108.
Zhang et al. [2026h]	Yi Zhang, Yinda Chen, Che Liu, Zeyuan Ding, Jin Xu, Shilong Zou, Junwei Liao, Jiayu Hu, et al.Pelican-Unify 1.0: a unified embodied intelligence model for understanding, reasoning, imagination and action.arXiv preprint arXiv:2605.15153, 2026h.URL https://arxiv.org/abs/2605.15153.
Zhang et al. [2025h]	Yuan Zhang, Chenyu Xue, Wenjie Xu, Chao Ji, Jiajia Wu, and Jia Pan.iFlyBot-VLA technical report.arXiv preprint arXiv:2511.01914, 2025h.URL https://arxiv.org/abs/2511.01914.
Zhang et al. [2025i]	Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li.LLaVA-Video: video instruction tuning with synthetic data.Trans. Mach. Learn. Res., 2025i.URL https://arxiv.org/abs/2410.02713.
Zhang et al. [2026i]	Zongzheng Zhang, Jingrui Pang, Zhuo Yang, Kun Li, Minwen Liao, Saining Zhang, Guoxuan Chi, Jinbang Guo, Huan-ang Gao, Modi Shi, et al.Dexora: open-source VLA for High-DoF bimanual dexterity.In IEEE Int. Conf. Robot. Autom., 2026i.URL https://arxiv.org/abs/2605.18722.
Zhao et al. [2025a]	Hongxiang Zhao, Xingchen Liu, Mutian Xu, Yiming Hao, Weikai Chen, and Xiaoguang Han.TASTE-Rob: Advancing video generation of task-oriented hand-object interaction for generalizable robotic manipulation.In IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2025a.URL https://arxiv.org/abs/2503.11423.
Zhao et al. [2025b]	Qingqing Zhao, Yao Lu, Moo Jin Kim, Zipeng Fu, Zhuoyang Zhang, Yecheng Wu, Zhaoshuo Li, Qianli Ma, Song Han, Chelsea Finn, et al.CoT-VLA: visual chain-of-thought reasoning for vision-language-action models.In IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2025b.URL https://doi.org/10.1109/cvpr52734.2025.00166.
Zhao et al. [2023]	Tony Z Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn.Learning fine-grained bimanual manipulation with low-cost hardware.In Robot. Sci. Syst., 2023.URL https://arxiv.org/abs/2304.13705.
Zhao et al. [2024]	Tony Z Zhao, Jonathan Tompson, Danny Driess, Pete Florence, Kamyar Ghasemipour, Chelsea Finn, and Ayzaan Wahid.ALOHA unleashed: a simple recipe for robot dexterity.In Conf. Robot Learn., 2024.URL https://arxiv.org/abs/2410.13126.
Zhao et al. [2026]	Zhenyu Zhao, Hongyi Jing, Xiawei Liu, Jiageng Mao, Abha Jha, Hanwen Yang, Rong Xue, Sergey Zakharov, Vitor Guizilini, and Yue Wang.Humanoid everyday: a comprehensive robotic dataset for open-world humanoid manipulation.In IEEE Int. Conf. Robot. Autom., 2026.URL https://arxiv.org/abs/2510.08807.
Zhaxizhuoma et al. [2025]	Zhaxizhuoma, Kehui Liu, Chuyue Guan, Zhongjie Jia, Ziniu Wu, Xin Liu, Tianyu Wang, Shuai Liang, et al.FastUMI: A scalable and hardware-independent universal manipulation interface with dataset.In Conf. Robot Learn., 2025.URL https://arxiv.org/abs/2409.19499.
Zheng et al. [2026a]	Jinliang Zheng, Jianxiong Li, Zhihao Wang, Dongxiu Liu, Xirui Kang, Yuchun Feng, Yinan Zheng, Jiayin Zou, et al.X-VLA: soft-prompted transformer as scalable cross-embodiment vision-language-action model.In Int. Conf. Learn. Represent., 2026a.URL https://arxiv.org/abs/2510.10274.
Zheng et al. [2026b]	Ruijie Zheng, Dantong Niu, Yuqi Xie, Jing Wang, Mengda Xu, Yunfan Jiang, Fernando Castañeda, Fengyuan Hu, You Liang Tan, Letian Fu, et al.EgoScale: scaling dexterous manipulation with diverse egocentric human data.arXiv preprint arXiv:2602.16710, 2026b.URL https://arxiv.org/abs/2602.16710.
Zheng et al. [2026c]	Yuhang Zheng, Songen Gu, Weize Li, Yupeng Zheng, Yujie Zang, Shuai Tian, Xiang Li, Ce Hao, Chen Gao, Si Liu, et al.OmniVTA: Visuo-tactile world modeling for contact-rich robotic manipulation.arXiv preprint arXiv:2603.19201, 2026c.URL https://arxiv.org/abs/2603.19201.
Zhong et al. [2026a]	Linqing Zhong, Yi Liu, Yifei Wei, Ziyu Xiong, Maoqing Yao, Si Liu, and Guanghui Ren.Acot-VLA: action chain-of-thought for vision-language-action models.In IEEE/CVF Conf. Comput. Vis. Pattern Recog., 2026a.URL https://doi.org/10.48550/arXiv.2601.11404.
Zhong et al. [2025]	Yifan Zhong, Fengshuo Bai, Shaofei Cai, et al.A survey on vision-language-action models: An action tokenization perspective.arXiv preprint arXiv:2507.01925, 2025.URL https://arxiv.org/abs/2507.01925.
Zhong et al. [2026b]	Zhide Zhong, Junfeng Li, Junjie He, Haodong Yan, Xin Gong, Guanyi Zhao, Yingjie Cai, Jiantao Gao, Xu Yan, Bingbing Liu, et al.Dualcot-VLA: visual-linguistic chain of thought via parallel reasoning for vision-language-action models.arXiv preprint arXiv:2603.22280, 2026b.URL https://arxiv.org/abs/2603.22280.
Zhou et al. [2019]	Bolei Zhou, Hang Zhao, Xavier Puig, Tete Xiao, Sanja Fidler, Adela Barriuso, and Antonio Torralba.Semantic understanding of scenes through the ADE20K dataset.Int. J. Comput. Vis., 2019.URL https://arxiv.org/abs/1608.05442.
Zhou et al. [2026]	Jianyi Zhou, Ziteng Gao, Feiyang Hong, Zirui Liu, Guannan Zhang, Weisheng Dai, Ruichen Zhen, Chuqiao Lyu, et al.Touchanything: a dataset and framework for bimanual tactile estimation from egocentric video.arXiv preprint arXiv:2605.13083, 2026.URL https://arxiv.org/abs/2605.13083.
Zhu et al. [2025a]	Chuning Zhu, Raymond Yu, Siyuan Feng, Benjamin Burchfiel, Paarth Shah, and Abhishek Gupta.Unified world models: coupling video and action diffusion for pretraining on large robotic datasets.In Robot. Sci. Syst., 2025a.URL https://arxiv.org/abs/2504.02792.
Zhu et al. [2025b]	Fangqi Zhu, Hongtao Wu, Song Guo, Yuxiao Liu, Chilam Cheang, and Tao Kong.IRASim: A fine-grained world model for robot manipulation.In IEEE/CVF Int. Conf. Comput. Vis., 2025b.URL https://arxiv.org/abs/2406.14540.
Zhu et al. [2026]	Fangqi Zhu, Zhengyang Yan, Zicong Hong, Quanxin Shou, Xiao Ma, and Song Guo.WMPO: world model-based policy optimization for vision-language-action models.In Int. Conf. Learn. Represent., 2026.URL https://arxiv.org/abs/2511.09515.
Zhu et al. [2025c]	Xinyue Zhu, Binghao Huang, and Yunzhu Li.Touch in the wild: learning fine-grained manipulation with a portable visuo-tactile gripper.In Adv. Neural Inf. Process. Syst., 2025c.URL https://arxiv.org/abs/2507.15062.
Experimental support, please view the build logs for errors. Generated by L A T E xml  .
Instructions for reporting errors

We are continuing to improve HTML versions of papers, and your feedback helps enhance accessibility and mobile support. To report errors in the HTML that will help us improve conversion and rendering, choose any of the methods listed below:

Click the "Report Issue" button, located in the page header.

Tip: You can select the relevant text first, to include it in your report.

Our team has already identified the following issues. We appreciate your time reviewing and reporting rendering errors we may not have found yet. Your efforts will help us improve the HTML versions for all readers, because disability should not be a barrier to accessing research. Thank you for your continued support in championing open access for all.

Have a free development cycle? Help support accessibility at arXiv! Our collaborators at LaTeXML maintain a list of packages that need conversion, and welcome developer contributions.

We gratefully acknowledge support from our major funders, member institutions, and all contributors.
About
·
Help
·
Contact
·
Subscribe
·
Copyright
·
Privacy
·
Accessibility
·
Operational Status
(opens in new tab)
Major funding support from

