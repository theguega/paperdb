Title: A Survey on Vision-Language-Action Models for Embodied AI

URL Source: https://arxiv.org/html/2405.14093

Markdown Content:
Back to arXiv

This is experimental HTML to improve accessibility. We invite you to report rendering errors. 
Use Alt+Y to toggle on accessible reporting links and Alt+Shift+Y to toggle off.
Learn more about this project and help improve conversions.

Why HTML?
Report Issue
Back to Abstract
Download PDF
 Abstract
IIntroduction
IIBackground
IIIVision-Language-Action Models
IVTask Planners
VDatasets and Benchmarks
VIChallenges and Future Directions
VIIConclusion
 References

HTML conversions sometimes display errors due to content that did not convert correctly from the source. This paper uses the following packages that are not yet supported by the HTML conversion tool. Feedback on these issues are not necessary; they are known and are being worked on.

failed: arydshln.sty
failed: forest.sty
failed: mdframed.sty
failed: forest.sty
failed: scalerel.sty

Authors: achieve the best HTML results from your LaTeX submissions by following these best practices.

License: arXiv.org perpetual non-exclusive license
arXiv:2405.14093v5 [cs.RO] 31 Aug 2025
A Survey on Vision-Language-Action Models for Embodied AI
Yueen Ma, Zixing Song, Yuzheng Zhuang, Jianye Hao, Irwin King
Y. Ma, Z. Song, and I. King are with the Department of Computer Science and Engineering, The Chinese University of Hong Kong, Hong Kong, China (Email: {yema21, zxsong, king}@cse.cuhk.edu.hk) Y. Zhuang and J. Hao are with Huawei Noah’s Ark Lab, Shenzhen, China (Email: {zhuangyuzheng, haojianye}@huawei.com)
Abstract

Embodied AI is widely recognized as a key element of artificial general intelligence because it involves controlling embodied agents to perform tasks in the physical world. Building on the success of large language models and vision-language models, a new category of multimodal models—referred to as vision-language-action models (VLAs)—has emerged to address language-conditioned robotic tasks in embodied AI by leveraging their distinct ability to generate actions. In recent years, a myriad of VLAs have been developed, making it imperative to capture the rapidly evolving landscape through a comprehensive survey. To this end, we present the first survey on VLAs for embodied AI. This work provides a detailed taxonomy of VLAs, organized into three major lines of research. The first line focuses on individual components of VLAs. The second line is dedicated to developing control policies adept at predicting low-level actions. The third line comprises high-level task planners capable of decomposing long-horizon tasks into a sequence of subtasks, thereby guiding VLAs to follow more general user instructions. Furthermore, we provide an extensive summary of relevant resources, including datasets, simulators, and benchmarks. Finally, we discuss the challenges faced by VLAs and outline promising future directions in embodied AI. We have created a project associated with this survey, which is available at https://github.com/yueen-ma/Awesome-VLA.

Index Terms: Vision-Language-Action Model, Embodied AI, Robotics, Multi-modality, Large Language Model, World Model
IIntroduction

Vision-language-action models (VLAs) are a class of multimodal models within the field of embodied AI, designed to process information from vision, language, and action modalities. Unlike conversational AI such as ChatGPT [1], embodied AI requires controlling physical embodiments that interact with the environment and robotics is the most prominent domain of embodied AI. In language-conditioned robotic tasks, the policy must possess the capability to understand language instructions, visually perceive the environment, and generate appropriate actions, necessitating the multimodal capabilities of VLAs. The term was recently coined by RT-2 [2]. Compared to earlier deep reinforcement learning approaches, VLAs offer superior versatility, dexterity, and generalizability in complex environments. As a result, they are well-suited not only for controlled settings like factories but also for everyday tasks in household environments.

Early developments in deep learning primarily consist of unimodal models. In computer vision (CV), models like AlexNet [3] showcased the potential of artifical neural networks. Recurrent neural networks laid the groundwork for numerous natural language processing (NLP) models but have seen a transition in recent years, with Transformers [4] taking precedence. The Deep Q-network [5] demonstrated that ANNs can successfully tackle reinforcement learning problems. Leveraging advancements of unimodal models across diverse machine learning fields, multimodal models now address a wide range of tasks, such as visual question answering, image captioning, and text-to-video generation.

Conventional robot policies based on reinforcement learning have largely focused on addressing a limited set of tasks within controlled environments, such as item grasping [6]. However, there is a growing need for more versatile multi-task policies, akin to recent trends in large language models (LLMs) and vision-language models (VLMs). Developing a multi-task policy is challenging, as it requires learning a broader set of skills and adapting to diverse environments. Furthermore, specifying tasks via language instructions offers a more intuitive user-robot interface, which necessitates the development of language-conditioned robot policies.

Figure 1:The general architecture of vision-language-action models. Three representative methods for action prediction are shown. Related components are presented in dashed boxes.

Built upon the success of large VLMs, vision-language-action models have demonstrated their potential in addressing these challenges, as illustrated in Figure 1. Similar to VLMs, VLAs utilize vision foundation models as vision encoders to obtain pretrained visual representations of the current environmental state, such as object class, pose, and geometry. VLAs encode instructions using the token embeddings of their LLMs and employ various strategies to align vision and language embeddings, including approaches like BLIP-2 [7] and LLaVA [8]. By finetuning on robot data, the LLM can function as a decoder to predict actions and perform language-conditioned robotic tasks. These cross-disciplinary innovations drive the advancement of VLAs in embodied AI—a critical building block for artificial general intelligence (AGI).

(a)Concepts
 
(b)Timelines
Figure 2:(a) A Venn diagram that outlines the main concepts in embodied AI discussed in this paper. (b) Timelines that trace the evolution from unimodal models to vision-language-action models.

VLAs are closely related to three lines of work, as depicted by the timelines in Figure 2b and the taxonomy in Figure 3. Some approaches focus on individual components of VLAs (§III-A), such as pretrained visual representations, dynamics learning, world models, and reasoning. Meanwhile, a substantial body of research is dedicated to low-level control policies (§III-B). In this category, language instructions and visual perceptions are fed into the control policy, which then generates low-level actions—such as translation and rotation—thereby rendering VLAs an ideal choice for control policies. In contrast, another category of models serves as high-level task planners responsible for task decomposition (§IV). These models break down long-horizon tasks into a sequence of subtasks that, in turn, guide VLAs toward the overall goal, as illustrated in Figure 4. Most current robotic systems adopt such a hierarchical framework [9, 10, 11] because the high-level task planner can leverage models with high capacity while the low-level control policy can focus on speed and precision, similar to hierarchical reinforcement learning.

To provide a more comprehensive overview of current progress in embodied AI, we propose a generalized definition of “VLA,” as illustrated by the Venn diagram in Figure 2a. We define a VLA as any model capable of processing multimodal inputs from vision and language to produce robot actions that accomplish embodied tasks, typically following the architecture in Figure 1. The original concept of VLA referred to a model that adapts VLMs to robotic tasks [2]. Analogous to the distinction between LLMs and more general language models, we designate the original VLAs as “Large VLAs” (LVLAs) because they are based on LLMs or large VLMs.

Related Work. To the best of our knowledge, this survey is the first to review the recent progress of VLA models, a rapidly emerging research area. Previous surveys have investigated other facets of embodied AI. [12] comprehensively summarized foundation models in robotics up to 2023, while [13] focused on LLMs in robotics. [14] examined more recent vision, language, and robotic foundation models for general-purpose robots. [15] concentrates on real-world robot applications. In contrast, our work emphasizes VLA models, thereby complementing and extending the existing literature on embodied AI.

Contributions. To the best of our knowledge, this paper is the first comprehensive survey of emerging vision-language-action (VLA) models in the field of embodied AI.

• 

Comprehensive Review. We present a thorough review of emerging VLA models in embodied AI, covering various aspects including components, architectures, training objectives, robotic tasks, etc.

• 

Taxonomy. We introduce a taxonomy of VLAs based on the hierarchical framework of current robotic systems, comprising two hierarchies: a low-level control policy and a high-level task planner. Control policies execute low-level actions based on specified language commands and the perceived environment. Task planners provide guidance to control policies by decomposing long-horizon tasks into subtasks. We also discuss various essential components of VLAs.

• 

Resources. We summarize the necessary resources for training and evaluating VLAs, including recent datasets and benchmarks in real-world or simulated environments. We also discuss various approaches to address current challenges, such as data scarcity and inconsistency.

• 

Future Directions. We outline current challenges and promising future opportunities in the field, such as safety, foundation models, and real-world deployment.

IIBackground
\forestset

rect/.append style=rectangle, rounded corners=2pt, dir tree switch/.style args=at #1for tree= fit=rectangle, , where level=0 calign=child, calign child=3.2, xshift=-1.2cm, , where level=#1 for tree= folder, grow’=0, , delay=child anchor=north, , before typesetting nodes= for tree= content/.wrap value=##1, , if=isodd(n_children(”!r”)) for nodewalk/.wrap pgfmath arg=fake=r,n=##1calign with current edgeint((n_children(”!r”)+1)/2), , , , {forest} dir tree switch=at 1, for tree= if level=1align=center, edge path= [draw, \forestoptionedge] (!u.parent anchor) – +(0,-20pt) -— (.child anchor)\forestoptionedge label; , align=@C25mm@ , , rect, draw, l sep=0.5cm, s sep=3pt, align=center, edge+=thick, rounded corners=5pt, rounded corners=2pt, , for level=0s sep=25pt, [VLA §III, fill=gray!45, parent [Components of VLA §III-A, for tree=component [Reinforcement Learning §III-A1, component_sub] [Pretrained Visual Repr. §III-A2, component_sub [Video Repr. §III-A3, component_sub2] ] [Dynamics Learning §III-A4, component_sub] [World Models §III-A5, component_sub [LLM-induced §III-A6, component_sub2] [Visual §III-A7, component_sub2] ] [Reasoning §III-A8, component_sub] ] [Control Policies §III-B, for tree=policy [Non-Transformer §III-B1, policy_sub] [Transformer-based §III-B2, policy_sub [Multimodal Instructions §III-B3, policy_sub2] [3D Vision §III-B4, policy_sub2, name=3d [3D Vision + Diffusion §III-B6, policy_sub, text width=4cm, name=3d_diff] ] [Diffusion §III-B5, policy_sub2, name=diff] [Motion Planning §III-B7, policy_sub2] [Point-based Action §III-B8, policy_sub2] [Large VLA §III-B9, policy_sub2] ] ] [Task Planners §IV, for tree=planner [Monolithic §IV-A, planner_type [End-to-end §IV-A1, planner_sub [3D Vision §IV-A2, planner_sub, text width=3cm] ] [Grounded §IV-A3, planner_sub] ] [Modular §IV-B, planner_type [Language-based §IV-B1, planner_sub] [Code-based §IV-B2, planner_sub] ] ] [Datasets & Benchmarks §V, for tree=data [Real-world §V-A, data_sub] [Simulation §V-B, data_sub [Simulators, data_sub2] ] [Auto Data Collection §V-C, data_sub] [Human Data §V-D, data_sub] [Task Planning Bench. §V-E, data_sub] [Embodied QA §V-F, data_sub] ] ] \draw[thick, rounded corners=5pt](
(
𝑑
𝑖
𝑓
𝑓
.
𝑛
𝑜
𝑟
𝑡
ℎ
)
−
(
63.5
𝑝
𝑡
,
0
)
) – (
(
𝑑
𝑖
𝑓
𝑓
.
𝑛
𝑜
𝑟
𝑡
ℎ
)
−
(
63.5
𝑝
𝑡
,
−
12
𝑝
𝑡
)
) – (3d_diff.west);

Figure 3:The taxonomy of VLA models. The organization of this survey follows this taxonomy.

Embodied AI is a unique form of artificial intelligence that actively interacts with the physical environment. This sets it apart from other AI models, such as conversational AI, which primarily handles textual conversations (ChatGPT [1]), or generative AI models that focus on tasks like text-to-video generation (Sora [16]). Embodied AI encompasses a broad spectrum of embodiments, including smart appliances, smart glasses, autonomous vehicles, and more. Among them all, robots stand out as one of the most prominent embodiments.

Robot learning is also usually framed as reinforcement learning problems, represented as a Markov Decision Process (MDP) consisting of states (
𝑠
), actions (
𝑎
), and rewards (
𝑟
). A MDP trajectory is denoted as 
𝜏
=
(
𝑠
1
,
𝑎
1
,
𝑟
1
,
…
,
𝑠
𝑇
,
𝑎
𝑇
,
𝑟
𝑇
)
. In certain scenarios, robotic tasks may also be viewed as partially-observable Markov Decision Processes (POMDPs) due to incomplete observations. The primary objective of reinforcement learning is to train a policy capable of generating optimal actions for the current state 
𝜋
​
(
𝑎
𝑡
|
𝑠
≤
𝑡
,
𝑎
<
𝑡
)
. Various methods such as TD learning, policy gradients, etc., can be employed to achieve this. However, in cases where defining the reward function proves challenging, imitation learning is utilized to directly model the action distribution within trajectories devoid of rewards 
𝜏
=
(
𝑠
1
,
𝑎
1
,
…
,
𝑠
𝑇
,
𝑎
𝑇
)
. Furthermore, many multi-task robot models employ language as instructions 
𝑝
 to determine which task or skill to execute, leading to the development of language-conditioned robot policies 
𝜋
​
(
𝑎
𝑡
|
𝑝
,
𝑠
≤
𝑡
,
𝑎
<
𝑡
)
.

IIIVision-Language-Action Models
III-AComponents of VLA

A body of work focuses on individual components of VLA models, drawing on successes in CV, NLP, and RL. We introduce them through the lens of embodied AI.

III-A1Reinforcement Learning

Reinforcement learning (RL) laid the foundation for embodied AI and continues to help advance VLAs. Deep Q-Network (DQN) first demonstrated learning policies directly from high-dimensional pixel inputs, underscoring the need for greater model capacity in end-to-end RL. RL trajectories—sequences of states, actions, and rewards—naturally align with sequence modeling problems, making them well-suited to transformer architectures. Pioneering efforts in this direction include Decision Transformer (DT) [17] and Trajectory Transformer (TT) [18]. Gato [19] further extended this paradigm to a multimodal, multitask, multi-embodiment setting. These methods in RL directly inspired modern VLAs [20, 21], which also process multimodal sequences using transformer-based architectures.

Furthermore, a synergy has emerged between RL and LLMs, benefiting embodied AI in multiple ways. Reinforcement learning from human feedback (RLHF) aligns LLMs with human preferences and has also been applied to robot learning. SEED [22] utilizes RLHF alongside skill-based RL to address the sparse reward issue and to improve safety via human evaluation. Conversely, LLMs also enable novel RL methods: Reflexion [23] proposes a novel verbal reinforcement learning framework that replaces weight updates in RL models with linguistic feedback, which is naturally applicable to embodied decision making where planning also occurs in language form. Eureka [24] shows that LLMs can design reward functions for embodied agents that outperform expert human-engineered ones. As LLMs evolve, this synergy accelerates progress in RL and embodied AI.

III-A2Pretrained Visual Representations

The effectiveness of the vision encoder directly influences the performance of VLAs, as it provides crucial information regarding the current state 
𝑠
𝑡
, such as object categories, positions, and affordance. Consequently, numerous methods are devoted to improving the quality of pretrained visual representations (PVRs). Their technical details are compared in Table I.

CLIP [25] has gained widespread adoption as a vision encoder in robotic models. The training objective of CLIP is to identify the correct text-image pair among all possible combinations in a given batch. CLIP undergoes training on the WIT dataset, comprising 400 million image-text pairs. This large-scale training allows CLIP to develop a rich understanding of the relationship between visual and textual information.

R3M [26] proposes two main pretraining objectives: time contrastive learning and video-language alignment. The objective of time contrastive learning is to minimize the distance between video frames that are temporally close while simultaneously increasing the separation between frames that are temporally distant. This objective aims to create PVRs that capture the temporal relationship within the video sequence. On the other hand, video-language alignment is to learn whether a video corresponds to a language instruction. This objective enriches the semantic relevance embedded in the PVRs. VIP [27] also capitalizes on video temporal relationship.

MVP [28] adopts masked autoencoder (MAE) from computer vision to robotic datasets. The MAE involves masking out a portion of input patches to a ViT model and training it to reconstruct the corrupted patches. This approach closely resembles the masked language modeling technique used in BERT [29] and falls within the purview of self-supervised training. RPT [30] undergoes pretraining with a focus not only on reconstructing visual inputs but also on robotic actions and proprioceptive states.

Table I:Pretrained visual representations. V: Vision. L: Language. CL: contrastive learning. TFM: Transformer. Sim/Real: simulated/real-world. Mani/Navi: manipulation/navigation. For simplicity, we only show the main part of the objective equation, omitting elements such as temperature, auxiliary loss, etc. 
𝒮
​
(
⋅
)
 denotes a similarity measure.
Model
 	
Network
	
Type
	
Objective
	
Notations
	
Robotic Tasks


CLIP [25]
 	
ViT-B
	
VL-CL
	
∑
𝑖
=
1
𝑁
−
log
⁡
exp
⁡
(
𝒮
​
(
𝑥
𝑖
,
𝑦
𝑖
)
)
∑
𝑗
=
1
𝑁
exp
⁡
(
𝒮
​
(
𝑥
𝑖
,
𝑦
𝑗
)
)
	
(
𝑥
𝑖
,
𝑦
𝑖
)
: image-text pair
	
(Used by CLIPort [31], EmbCLIP [32], and CoW [33])


R3M [26]
 	
ResNet-50
	
Time-CL
	
∑
𝑖
=
1
𝑇
−
log
⁡
exp
⁡
(
𝒮
​
(
𝑥
𝑖
,
𝑥
𝑗
)
)
exp
⁡
(
𝒮
​
(
𝑥
𝑖
,
𝑥
𝑗
)
)
+
exp
⁡
(
𝒮
​
(
𝑥
𝑖
,
𝑥
𝑘
)
)
	
𝑥
𝑖
: 
𝑖
-th video frame, 
𝑖
<
𝑗
<
𝑘
	
Sim-Mani: Meta-World, Franka Kitchen, Adroit


MVP [28]
 	
ViT-B/L
	
MAE
	
∑
𝑖
=
1
𝑁
−
log
⁡
𝑃
​
(
𝑥
𝑖
∣
𝑥
≠
𝑖
)
	
𝑥
𝑖
: image patch
	
Real-Mani (xArm7): pick, reach block, push cube, close fridge


VIP [27]
 	
ResNet-50
	
Time-CL
	
∑
𝑖
=
1
𝑇
−
log
⁡
exp
⁡
(
𝒮
​
(
𝑥
0
,
𝑥
𝑗
)
)
exp
⁡
(
𝒮
​
(
𝑥
0
,
𝑥
𝑗
)
)
+
exp
⁡
(
𝒮
​
(
𝑥
𝑖
,
𝑥
𝑗
)
)
	
𝑥
𝑖
: 
𝑖
-th video frame, 
0
<
𝑖
<
𝑗
	
Real-Mani (Franka): pick and place, close drawer, push bottle, fold towel


VC-1 [34]
 	
ViT-L
	
MAE, CL
	
(A combination of previous works, including CLIP, R3M, MVP, and VIP.)
	
 
	
Sim-Mani: Meta-World, Adroit, DMC, TriFinger, Habitat 2.0; Sim-Navi: Habitat


Voltron [35]
 	
TFM
	
MAE, Lang-Gen
	
∑
𝑖
=
1
𝑁
−
log
⁡
𝑃
​
(
𝑥
𝑖
∣
𝑥
≠
𝑖
,
𝑦
)
;

∑
𝑖
=
1
𝑁
′
−
log
⁡
𝑃
​
(
𝑦
𝑖
∣
𝑥
,
𝑦
<
𝑖
)
	
𝑥
𝑖
: image patch;

𝑦
: language instruction
	
Sim-Mani: Franka Kitchen;
Real-Mani (Franka): custom study desk


RPT [30]
 	
ViT
	
MAE
	
∑
𝑖
=
1
𝑁
−
log
⁡
𝑃
​
(
𝑥
𝑖
∣
𝑥
≠
𝑖
,
𝑦
,
𝑧
,
…
)
	
𝑥
,
𝑦
,
𝑧
: three distinct modalities
	
Real-Mani (Franka): stack, pick, pick from bin


DINOv2 [36]
 	
ResNet, ViT
	
Self-distillation
	
∑
𝑥
∑
𝑥
′
≠
𝑥
𝐻
​
(
𝑃
𝑡
​
(
𝑥
)
,
𝑃
𝑥
​
(
𝑥
′
)
)
	
𝑥
,
𝑥
′
: image views; 
𝐻
​
(
)
: cross-entropy; 
𝑃
𝑡
,
𝑃
𝑠
: teacher, student
	
(Used by OpenVLA [37], ReKep [38])


I-JEPA [39]
 	
ViT
	
JEPA
	
1
𝑀
​
∑
𝑖
=
1
𝑀
∑
𝑗
‖
𝑥
𝑗
​
(
𝑖
)
−
𝑦
𝑗
​
(
𝑖
)
‖
2
2
	
𝑥
𝑗
​
(
𝑖
)
: 
𝑗
-th masked patch embedding of block 
𝑖
; 
𝑦
𝑗
​
(
𝑖
)
: unmasked patch embedding
	
 


Theia [40]
 	
ViT-T/S/B
	
Distillation
	
(Distillation of vision foundation models: ViT, CLIP, SAM, DINOv2, Depth-Anything.)
	
 
	
Sim-Mani&Navi: CortexBench (VC-1); Real-Mani: pick, place, open door/drawer

Voltron [35] introduces a pretraining objective by incorporating language conditioning and language generation into the MAE objective. Employing an encoder-decoder Transformer structure, the pretraining alternates between language-conditioned masked image reconstruction and language generation from masked images. This enhances the alignment between language and vision modalities.

VC-1 [34] conducts an in-depth examination of prior PVRs and introduces an enhanced PVR model by systematically exploring optimal ViT configurations across diverse datasets. Additionally, they perform a comprehensive comparative analysis of their model against previous methods on various manipulation and navigation datasets, shedding light on the critical factors that contribute to the improvement of PVR. Another study [41] also compares previous PVRs obtained under supervised learning or self-supervised learning.

DINOv2 [36] proposes a new self-supervised training paradigm for PVRs that achieves performance beyond that of MAE. It employs a self-distillation framework in which the teacher and student networks receive different views of the same image and match their encoded representations. The student network is updated using SGD, while the teacher network is maintained as the EMA of the student network.

I-JEPA [39] is motivated by the joint-embedding predictive architectures proposed by [42]. It constructs a “primitive” internal world model by comparing the embeddings of patches. Unlike DINO, which uses cropped images, I-JEPA employs masked patches. Moreover, it differs from MAE because it is a non-generative approach.

Theia [40] proposes distilling various vision foundation models into a single model. By fusing their information from segmentation, depth, semantics, etc., it outperforms previous PVRs while requiring less data and a smaller model size.

III-A3Video Representations

Videos are simple sequences of images and can be represented by concatenating the usual PVRs of each frame. However, their multi-view nature enables a variety of unique representation techniques beyond those mentioned above, such as time contrastive learning and MAE. NeRF can be extracted from videos and contains rich 3D information for robot learning, as exemplified by F3RM [43] and 3D-LLM [44]. The recent 3D Gaussian Splatting (3D-GS) method has been gaining popularity in computer vision as well as in embodied AI. 3D Gaussians can be deformed through physics-based simulation for generative dynamics, as proposed in PhysGaussian [45], or they can serve as 3D representations for VLMs, as enabled by UniGS [46]. Additionally, many videos contain audio, which can provide important cues for robot policies [47].

III-A4Dynamics Learning

Dynamics learning encompasses objectives aimed at endowing the model 
𝑓
​
(
⋅
)
 with an understanding of forward or inverse dynamics. Forward dynamics involves predicting the subsequent state resulting from a given action, whereas inverse dynamics entails determining the action required to transition from a previous state to a known subsequent state:

	
Forward dynamics: 
	
𝑠
^
𝑡
+
1
←
𝑓
fwd
​
(
𝑠
𝑡
,
𝑎
𝑡
)
,


Inverse dynamics: 
	
𝑎
^
𝑡
←
𝑓
inv
​
(
𝑠
𝑡
,
𝑠
𝑡
+
1
)
.
		
(1)

Some approaches also frame these objectives as reordering problems for shuffled state sequences. Some forward dynamics methods closely resemble the image or video prediction pretraining used in PVRs. We compare them in Table II.

Table II:Dynamics learning methods for VLAs. 
𝑓
​
(
⋅
)
 is the dynamic model. Fwd, inv: forward & inverse dynamics.
Model
 	
Vision Encoder
	
Type
	
Objective
	
Notations
	
Robotic Tasks


Vi-PRoM [48]
 	
ResNet
	
Temporal dynamics
	
∑
𝑖
=
1
𝑇
CE
⁡
(
𝑖
,
𝑓
​
(
𝑥
𝑖
∣
𝑥
′
)
)
	
𝑓
​
(
⋅
)
: predicts frame index in raw seq. 
𝑥
 given shuffled seq. 
𝑥
′
	
Sim-Mani: Meta-World, Franka Kitchen; Real-Mani: open & close drawer/door


MaskDP [49]
 	
ViT (MAE)
	
MAE (implicit fwd & inv dynamics)
	
∑
𝑖
=
1
𝑇
−
log
⁡
𝑃
​
(
𝑥
𝑖
∣
𝑥
≠
𝑖
,
𝑦
)
	
𝑥
𝑖
: state or action token; 
𝑦
: the other modality
	
Sim: DeepMind Control Suite


MIDAS [50]
 	
ViT, Mask R-CNN
	
Inverse dynamics
	
∑
𝑖
=
1
𝑇
MSE
⁡
(
𝑎
𝑡
,
𝑓
inv
​
(
𝑠
𝑡
,
𝑠
𝑡
+
1
)
)
	
𝑠
𝑡
, 
𝑎
𝑡
: state, action
	
Sim-Mani: VIMA-Bench


SMART [51]
 	
CNN
	
Forward & inverse dynamics
	
∑
𝑖
=
1
𝑇
(
MSE
(
𝑠
𝑡
+
1
,
𝑓
fwd
(
𝑠
𝑡
,
𝑎
𝑡
)
)

    
+
MSE
(
𝑎
𝑡
,
𝑓
inv
(
𝑠
𝑡
,
𝑠
𝑡
+
1
)
)
)
	
𝑠
𝑡
, 
𝑎
𝑡
: state, action
	
Sim: DeepMind Control Suite


PACT [52]
 	
ResNet-18, PointNet
	
Forward dynamics
	
∑
𝑖
=
1
𝑇
MSE
⁡
(
𝑠
𝑡
+
1
,
𝑓
fwd
​
(
𝑠
𝑡
,
𝑎
𝑡
)
)
	
𝑠
𝑡
, 
𝑎
𝑡
: state, action
	
Sim-Navi: Habitat; Real-Navi (MuSHR vehicle)


VPT [53]
 	
ResNet
	
Inverse dynamics
	
∑
𝑖
=
1
𝑇
MSE
⁡
(
𝑎
𝑡
,
𝑓
inv
​
(
𝑠
𝑡
,
𝑠
𝑡
+
1
)
)
	
𝑠
𝑡
, 
𝑎
𝑡
: state, action
	
Sim: Minecraft


GR-1 [54]
 	
ViT (MAE)
	
Forward dynamics
	
∑
𝑖
=
1
𝑇
MSE
⁡
(
𝑠
𝑡
+
1
,
𝑓
fwd
​
(
𝑠
𝑡
,
𝑎
𝑡
)
)
	
𝑠
𝑡
, 
𝑎
𝑡
: state, action
	
Sim-Mani: CALVIN

Vi-PRoM [48] presents three distinct pretraining objectives. The first involves a contrastive self-supervised learning objective designed to distinguish between different videos. The remaining two objectives are centered around supervised learning tasks: temporal dynamics learning, aimed at recovering shuffled video frames, and image classification employing pseudo labels. Through a comprehensive comparison with preceding pretraining methods, Vi-PRoM demonstrates its effectiveness for behavior cloning and PPO.

MIDAS [50] introduces an inverse dynamics prediction task as part of its pretraining. The objective is to train the model to predict actions from observations, formulated as a motion-following task. This approach enhances the model’s understanding of the transition dynamics of the environment.

SMART [51] presents a pretraining scheme encompassing three distinct objectives: forward dynamics prediction, inverse dynamics prediction, and randomly masked hindsight control. The forward dynamics prediction task involves predicting the next latent state, while the inverse dynamics prediction task entails predicting the last action. In the case of hindsight control, the entire control sequence is provided as input, with some actions masked, and the model is trained to recover these masked actions. The incorporation of the first two dynamics prediction tasks facilitates capturing local and short-term dynamics, while the third task is designed to capture global and long-term temporal dependencies.

MaskDP [49] features the masked decision prediction task, wherein both state and action tokens are masked for reconstruction. This masked modeling task is specifically crafted to equip the model with an understanding of both forward and inverse dynamics. Notably, in contrast to preceding masked modeling approaches like BERT or MAE, MaskDP is applied directly to downstream tasks

PACT [52] introduces a pretraining objective aimed at modeling state-action transitions. It receives sequences of states and actions as input, and predicts each state and action token autoregressively. This pretrained model serves as a dynamics model, which can then be finetuned for various downstream tasks such as localization, mapping, and navigation.

VPT [53] proposes a video pretraining method that harnesses unlabeled internet data to pretrain a foundational model for the game of Minecraft. The approach begins by training an inverse dynamics model using a limited amount of labeled data, which is then utilized to label internet videos. Subsequently, this newly auto-labeled data is employed to train the VPT foundation model through behavior cloning. This methodology follows semi-supervised imitation learning. As a result of this process, the model demonstrates human-level performance across a multitude of tasks.

GR-1 [54] introduces video prediction pretraining tailored for a GPT-style model. The ability to anticipate future frames aligns with forward dynamics learning, contributing to more accurate action prediction.

III-A5World Models

A world model 
𝑃
​
(
⋅
)
 encodes commonsense knowledge about the world and predicts the future state for a given action [42]:

	
𝑠
^
𝑡
+
1
∼
𝑃
​
(
𝑠
^
𝑡
+
1
|
𝑠
𝑡
,
𝑎
𝑡
)
.
		
(2)

It enables model-based control and planning for embodied agents, as they can search for an optimal action sequence in imaginary space before executing any real actions. Although forward dynamics learning also attempts to predict the next state, it is usually treated as a pretraining task or an auxiliary loss to enhance the RL-Transformer-based action decoder for the primary robotics tasks, rather than serving as a standalone module. Additionally, new embodied experiences can be sampled from visual world models that explicitly generate images or videos of future states.

Dreamer [55] employs three primary modules to construct a latent dynamics model: a representation model, responsible for encoding images into latent states; a transition model, which captures transitions between latent states; and a reward model, predicting the reward associated with a given state. Under the actor-critic framework, Dreamer utilizes an action model and a value model to learn behavior through imagination by propagating analytic gradients through the learned dynamics. Building upon this foundation, DreamerV2 [56] introduces a discrete latent state space along with an improved objective. DreamerV3 [57] extends its focus to a broader range of domains with fixed hyperparameters. DayDreamer [58] applies this method to physical robots performing real-world tasks.

IRIS [59] employs a GPT-like autoregressive Transformer as the foundation of its world model, with a VQ-VAE serving as the vision encoder. Subsequently, a policy is trained using imagined trajectories, which are unrolled from a real observation by the world model. TWM [60] also investigates the application of Transformers in building world models.

III-A6LLM-induced World Models

LLMs encompass a wealth of commonsense knowledge about the world, prompting many approaches to leverage that knowledge for improving VLAs.

DECKARD [61] prompts LLM to generate abstract world models (AWMs) represented as directed acyclic graphs, specifically tailored for the task of item crafting in Minecraft. DECKARD iterates between two phases: in the Dream phase, it samples a subgoal guided by the AWM; in the Wake phase, DECKARD executes the subgoal and updates the AWM through interactions with the game. This guided approach enables DECKARD to achieve markedly faster item crafting compared to baselines lacking such guidance.

LLM-DM [62] uses an LLM to construct world models in planning domain definition language (PDDL)—a capability that LLM+P [63] did not achieve, as its PDDL world model was hand-crafted. The LLM also acts as an interface, mediating between the generated PDDL model and corrective feedback from syntactic validators and human experts. Finally, the PDDL world model serves as a symbolic simulator, assisting the LLM planner in generating plans.

RAP [64] repurposes an LLM to act as both a policy that predicts actions and a world model that provides the state transition distribution. Unlike previous chain-of-thought prompting methods, RAP incorporates Monte Carlo Tree Search (MCTS) to enable structured planning, allowing the LLM to build a reasoning tree incrementally. This reasoning strategy helps RAP find a high-reward path that balances exploration and exploitation. Tree-Planner [65] improves efficiency by prompting the LLM only once to generate diverse paths within an action tree.

LLM-MCTS [66] builds upon RAP but extends the problem setting to POMDPs. As a world model, the LLM generates the initial belief of the current state; as a policy, it serves as a heuristic to guide action selection. By leveraging its commonsense knowledge, the LLM reduces the search space of MCTS, thereby improving search efficiency.

III-A7Visual World Models

Unlike LLM-induced world models, which are in textual form, visual world models generate images, videos, or 3D scenes of future states—aligning more closely with the physical world. They can further be utilized to generate new trajectories. This direction has been gaining increasing attention since Sora demonstrated world-simulation capabilities [67], as investigated by a dedicated survey [68].

Genie [69] introduces a new class of generative models, termed Generative Interactive Environments. It consists of three main components: a spatiotemporal video tokenizer, an autoregressive dynamics model, and a latent action model. After being trained on unlabeled videos in an unsupervised manner, Genie allows users to interact with the generative environment on a frame-by-frame basis. Consequently, Genie establishes a foundation world model.

3D-VLA [70] proposes a 3D world model capable of goal generation. It processes visual inputs, such as images, depth map, and point clouds, and then generates a goal state—either as an image or a point cloud—using diffusion models in response to the user’s query. The generated goal state can subsequently be used to guide robot control.

UniSim [71] builds a generative model based on real-world interaction videos. It is capable of simulating visual outcomes for both high-level and low-level actions, which can then be leveraged as new experiences for training embodied agents. E2WM [72] even treats existing simulators as world models to collect embodied experiences via MCTS.

III-A8Reasoning

Reasoning has become a key capability of LLMs, as demonstrated by chain-of-thought (CoT) methods [73, 74]. In embodied AI, researchers are exploring how to leverage CoT reasoning to refine the decision-making process.

Reasoning is naturally compatible with high-level task planning, as both often occur in the language domain. ThinkBot [75] applies CoT to recover missing action descriptions in sparse human instructions, thereby enhancing instruction coherence and boosting success rates in difficult tasks. ReAct [76] interleaves reasoning traces and actions, where CoT can help create action plans, inject commonsense knowledge, and handle exceptions, ultimately improving embodied decision-making. RAT [77] integrates CoT with retrieval-augmented generation (RAG) to mitigate hallucinations and thus improve embodied planning. Tree-Planner [65] employs a tree-of-thoughts approach for task planning.

Another paradigm involves equipping low-level control policies with reasoning capabilities, as pioneered by ECoT [78]. This method trains OpenVLA [37] to conduct embodied CoT reasoning about plans, sub-tasks, motions, and visual features, before predicting low-level actions. By relying on this multi-step reasoning rather than the “muscle memory” of VLAs, it improves success rates on challenging generalization tasks without requiring additional robot data.

Strengths and Limitations.

PVRs

Although time contrastive learning with videos and text-guided pretraining methods, such as CLIP, provide image-level information, they lack the pixel-level details offered by MAE-based self-supervised learning methods. Pixel-level information contains rich details—including segmentation masks, object positions, and depth estimation—that are generally more useful for robot manipulation tasks requiring high precision, as demonstrated by VC-1’s comparison [34]. DINOv2 learns both pixel- and image-level features by combining masked image modeling with a momentum encoder and multi-crop augmentation, with benefits on downstream robotic tasks evidenced by OpenVLA [37]. I-JEPA focuses on patches in the representation space and, as a result, captures low-level image features more effectively than view-invariance methods such as DINO. Theia distills various off-the-shelf vision foundation models into a single model that surpasses isolated individual models, as demonstrated by comprehensive evaluations against most existing PVRs in robot learning [40].

Forward & inverse dynamics

Forward dynamics learning is generally more challenging than inverse dynamics learning because predicting future states is more complex than predicting past actions. Consequently, the difficulty of forward dynamics often leads to greater performance improvements, as demonstrated in SMART. However, inverse dynamics models can be used to generate action labels for datasets that contain only states, such as raw robot manipulation videos.

World Models & Reasoning

Although both world models and reasoning methods can be applied to low-level control policies and high-level task planners, current approaches remain distinct. World models are predominantly used to interact with control policies because they excel at generating the immediate next state given low-level actions. In contrast, CoT-based reasoning methods focus on task planning since they express thought chains in text, making them well-suited for refining text-based task plans.

III-BLow-level Control Policies

Through the integration of an action decoder with perception modules, such as vision encoders and language encoders, a VLA model 
𝜋
𝜃
 with parameters 
𝜃
 is formed as a control policy to execute language instructions 
𝑝
:

	
𝑎
^
𝑡
∼
𝜋
𝜃
​
(
𝑎
^
𝑡
|
𝑝
,
𝑠
≤
𝑡
,
𝑎
<
𝑡
)
.
		
(3)

It can also be referred to as a low-level policy, low-level controller, or action primitive. The diversity among VLAs arises from individual modules and overall architectures. The general architecture is shown in Figure 1. This section explores various approaches to designing low-level control policies. Table III summarizes their technical details.

Figure 4:Illustration of a hierarchical robot policy. The high-level task planner decomposes the user instruction into subtasks, which are then executed step by step by the low-level control policy.
III-B1Non-Transformer Control Policies

Prior to the adoption of Transformer models, early control policies for language-conditioned robotic tasks varied significantly in architecture.

CLIPort [31] integrates CLIP with the Transporter network, creating a two-stream architecture. The CLIP vision encoder extracts “semantic” information from the RGB image, while the Transporter network extracts “spatial” information from the RGB-D image. The CLIP sentence encoder encodes the language instruction and guides the output 
𝑆
​
𝐸
​
(
2
)
 action, consisting of paired pick and place end-effector poses. It represents an early demonstration of language-conditioned pick-and-place capabilities.

BC-Z [79] processes two types of task instructions: a language instruction or a human demonstration video. The environment is presented to the model in the form of an RGB image. Then the instruction embedding and the image embedding are combined through the FiLM layer, culminating in the generation of actions. This conditional policy is asserted to exhibit zero-shot task generalization to unseen tasks.

MCIL [80] represents a pioneering robot policy that integrates free-form natural language conditioning. This is in contrast to earlier approaches that typically rely on conditions in the form of task ID or goal image. MCIL introduces the capability to leverage unlabeled and unstructured demonstration data. This is achieved by training the policy to follow either image or language goals, with a small fraction of the training dataset consisting of paired image and language goals.

HULC [81] introduces several techniques aimed at enhancing robot learning architectures. These include a hierarchical decomposition of robot learning, a multimodal Transformer, and discrete latent plans. The Transformer learns high-level behaviors, hierarchically dividing low-level local policies and the global plan. Additionally, HULC incorporates a visuo-lingual semantic alignment loss based on contrastive learning to align VL modalities. HULC++ [82] further integrates a self-supervised affordance model. This model guides HULC to the actionable region specified by a language instruction, enabling it to fulfill tasks within this designated area.

UniPi [83] treats the decision-making problem as a text-conditioned video generation problem. To predict actions, UniPi generates a video based on a given text instruction, and actions are extracted from the frames of the video through inverse dynamics. This innovative policy-as-video formulation offers several advantages, including enhanced generalization across diverse robot tasks and the potential for knowledge transfer from internet videos to real robots.

Table III:VLAs that serve as low-level control policies. 
◆
 indicates large VLAs. Closely related (non-VLA models) are included in brackets. The remaining are generalized VLAs. BC: behavior cloning (cont/disc: continuous/discrete action). Xattn: cross-attention. Concat: concatenation. Quant.: quantization. 
𝑝
/
𝑠
: prompt/state vision encoder. [SC]: self-collect data.
Model
 	
Vision
Encoder
	
Language Encoder / VLM
	
Action Decoder/Head
	
Architecture
	
Training Objectives
	
Training Datasets
	
Environments, Embodiments, Tasks, and Skills


CLIPort [31]
 	
CLIP-ResNet50, Transporter-ResNet
	
CLIP-GPT
	
LingUNet
	
Hadamard
	
BC (
𝑆
​
𝐸
​
(
2
)
)
	
[SC]
	
Sim: Ravens


MCIL [80]
 	
Custom-CNN
	
USE
	
RNN
	
LMP
	
MCIL
	
Play-LMP, [SC]
	
Sim: 3D Playroom environment


HULC [81, 82]
 	
MCIL CNN
	
Sentence-BERT
	
RNN
	
LMP
	
MCIL
	
CALVIN data
	
Sim: CALVIN


Language costs [84]
 	
CLIP-ViT, UNet
	
CLIP-GPT
	
STORM
	
Concat
	
MLE (cost map)
	
[SC]
	
Sim & Real (Franka): pick, place


Interactive Language [85]
 	
ResNet
	
CLIP-GPT
	
TFM
	
Xattn
	
BC (cont)
	
[SC: Language-Table]
	
Sim & Real (xArm6): rearrangement


Hiveformer [86]
 	
UNet
	
CLIP-GPT
	
TFM, UNet
	
Concat
	
BC (cont)
	
RLBench data
	
Sim: RLBench


PerAct [87]
 	
3D CNN
	
CLIP-GPT
	
PerceiverIO
	
Concat
	
3D affordance
	
RLBench data, [SC]
	
Sim: RLBench; Real (Franka): pick, place, stack, open drawer, sweep, insert peg, etc.


Act3D [88]
 	
CLIP-ResNet50
	
CLIP-GPT
	
TFM
	
Xattn
	
3D affordance (voxel)
	
RLBench data, [SC]
	
Sim: RLBench; Real (Franka): reach target, wipe, stack, etc.


RVT [89],
RVT-2 [90]
 	
CLIP-ResNet50
	
CLIP-GPT
	
TFM
	
Concat
	
2D affordance (project to 3D)
	
RLBench data, [SC]
	
Sim: RLBench; Real (Franka): stack, press, place


RoboPoint [91]
 	
ViT-L/14
	
Vicuna-V1.5 13B
	
 
	
Concat
	
2D affordance (project to 3D)
	
[SC]
	
Real (Franka): pick, place


Gato [19]
 	
ViT
	
Sent.Piece
	
TFM
	
Concat
	
BC (cont & disc)
	
[SC]
	
Sim & Real (Sawyer): RGB-stacking


(RoboCat) [92]
 	
VQ-GAN (
𝑝
,
𝑠
)
	
 
	
TFM
	
Quant.
	
BC, observation prediction
	
Self-improvement
	
Sim & Real (Sawyer, Franka, KUKA): stacking, building, lifting, insertion, removal


VIMA [93]
 	
ViT, Mask R-CNN
	
T5
	
TFM
	
Xattn
	
BC (
𝑆
​
𝐸
​
(
2
)
)
	
[SC:VIMA-Data]
	
Sim (Ravens): VIMA-Bench


BC-Z [79]
 	
ResNet18 (
𝑝
,
𝑠
)
	
USE
	
MLP
	
FiLM
	
BC (cont)
	
[SC]
	
Real (EDR): pick-place/wipe/drag, grasp, push


RT-1 [94]
 	
EfficientNet
	
USE
	
TFM
	
FiLM
	
BC (disc)
	
[SC: Fractal]
	
Real (EDR): pick-place, move, knock


MOO [93]
 	
OWL-ViT (
𝑝
), EfficientNet (
𝑠
)
	
USE
	
TFM
	
FiLM
	
BC (disc)
	
[SC]
	
Real (EDR): pick, move near, knock, place upright, place into


Q-Transformer [95]
 	
EfficientNet
	
USE
	
TFM
	
FiLM
	
TD error
	
Fractal, Auto-collect
	
Sim: pick; Real (EDR): pick, place, open/close drawer, move near


(RT-Trajectory) [96]
 	
EfficientNet
	
 
	
TFM
	
 
	
BC (disc)
	
[SC]
	
Real (EDR): pick, place, fold towel, swivel chair, etc.


(ACT) [97]
 	
ResNet18
	
 
	
CVAE-TFM
	
 
	
BC (cont, action chunking)
	
[SC] with ALOHA
	
Sim: transfer cube, bimanual insertion; Real (ViperX, WidowX): slot battery, open cup, etc


RoboAgent / MT-ACT [98]
 	
CNN
	
 
	
CVAE-TFM
	
FiLM
	
BC (cont, action chunking)
	
RoboSet
	
Real (Franka): pick, place, open/close drawers, pour, push, drag, etc.


RoboFlamingo [99]
 	
CLIP-ViT-L/14
	
LLaMA, MPT, GPT-NeoX
	
LSTM, TFM
	
Xattn
	
BC (cont)
	
CALVIN data
	
Sim: CALVIN


RoboUniView [100]
 	
UVFormer
	(Model design follows RoboFlamingo)	
BC (cont)
	
CALVIN data
	
Sim: CALVIN


DeeR-VLA [101]
 	(Model design follows RoboFlamingo++)	
BC (cont)
	
CALVIN data
	
Sim: CALVIN


Instruct2Act [102]
 	
CLIP, SAM
	
ChatGPT
	
Robot APIs
	
Tool-Use
	
 
	
 
	
Sim (Ravens): VIMA-Bench


VoxPoser [103]
 	
ViLD, MDETR, OWL-ViT, SAM
	
GPT-4
	
MPC
	
Tool-Use
	
 
	
 
	
Sim: Sapien; Real (Franka): move&avoid, set table, close drawer, open bottle, sweep trash


UniPi [83]
 	
Imagen video
	
T5-XXL
	
CNN
	
Xattn
	
Inverse dynamics
	
[SC:PF], Ravens, BridgeV1
	
Sim: Painting Factory (PF), Ravens


(Diffusion Policy) [104]
 	
ResNet18
	
 
	
UNet, TFM
	
 
	
DDPM
	
[SC]
	
Sim: Robomimic, Franka Kitchen, etc; Real (UR5, Franka): push-T, flip mug, pour sauce


(DP3) [105]
 	
DP3 Encoder
	(Model design follows Diffusion Policy)	
DDIM
	
[SC]
	
Sim: (72 tasks); Real (Franka, Allegro hand): roll-up, dumpling, drill, pour


SUDD [106]
 	
ResNet18
	
CLIP-GPT
	
UNet, TFM
	
Concat
	
DDPM
	
Lang-guided data generation
	
Sim: MuJoCo; Real (UR5e): pick, place


Octo [107]
 	
CNN
	
T5-base
	
TFM
	
Concat
	
DDPM
	
OXE
	
Real: BridgeV2, CMU Baking, Stanford Coffee, Berkeley Peg Insert, etc.


3D Diffuser Actor [108]
 	(Model design follows Act3D)	
DDPM
	
RLBench, CALVIN, [SC]
	
Sim: RLBench, CALVIN; Real (Franka): put, open, close, stack, etc.


MDT [109]
 	
CLIP-ViT-B/16 (
𝑝
), Voltron/ResNet18 (
𝑠
)
	
CLIP-GPT
	
DiT
	
Concat
	
DDIM
	
CALVIN, LIBERO, [SC]
	
Sim: CALVIN, LIBERO; Real (Franka): pick, push, open, close, etc.


RDT-1B [110]
 	
SigLIP
	
T5-XXL
	
DiT
	
Xattn
	
DDPM
	
(Aggregated Datasets)
	
Real (ALOHA dual-arm): wash, pour, fold, etc.


RT-2 [2] 
◆
 	
ViT-4B, ViT-22B
	
PaLI-X, PaLM-E
	
Symbol-tuning
	
Concat
	
BC (disc), Co-fine-tuning
	
Fractal, VQA
	
Sim: Language-Table; Real: RT-1 evaluation tasks


RT-H [111] 
◆
 	(Model design follows RT-2)	
BC (disc)
	
Diverse+Kitchen
	
Real: Diverse+Kitchen eval tasks


RT-X [112] 
◆
 	(Models from RT-1 and RT-2)	
BC (disc)
	
[SC: OXE]
	
Real: BridgeV2, RT-1 evaluation tasks, etc.


OpenVLA [37] 
◆
 	
DINOv2, SigLIP
	
Prismatic-7B
	
Symbol-tuning
	
Concat
	
BC (disc)
	
OXE, DROID
	
Real: BridgeV2, RT-1 evaluation tasks, Franka-Tabletop, DROID, etc.


OpenVLA-OFT [113] 
◆
 	(Improves OpenVLA with OFT recipe)	
BC (cont, parallel decode w/ chunk.)
	
LIBERO, [SC]
	
Sim: LIBERO; Real (ALOHA setup): fold, scoop, put


TraceVLA [114] 
◆
 	(Model design follows OpenVLA, adding visual trace prompting)	
BC (disc)
	
BridgeV2, Fractal, [SC]
	
Sim: SimplerEnv; Real (WidowX): pick, push, fold, swipe


𝜋
0
 [115] 
◆
 	
SigLIP
	
PaliGemma
	
Action expert
	
Concat
	
Flow matching
	
OXE, [SC: 
𝜋
-cross-embod.]
	
Real (general robot control): sweep, open, pack, etc.


RoboMamba [116] 
◆
 	
CLIP-ViT
	
Mamba
	
MLP
	
Concat
	
BC (cont)
	
[SC]
	
Sim: Sapien; Real (Franka): open door/box, staple


SpatialVLA [117] 
◆
 	
SigLIP, Ego3D Position Encoding
	
PaliGemma 2
	
MLP
	
Concat
	
BC (Adaptive Action Grids)
	
OXE, BridgeV2, Fractal, RH20T
	
Sim: SimplerEnv, LIBERO; Real (WidowX): pick, place, close, push


TinyVLA [118] 
◆
 	
ViT
	
Pythia
	
Diffusion Policy
	
Concat
	
DDPM
	
OXE, [SC]
	
Sim: MetaWorld; Real (Franka, UR5): place, stack, flip mug, close drawer, open box


CogACT [119] 
◆
 	
DINOv2, SigLIP
	
LLaMA 2
	
DiT
	
Concat
	
DDIM
	
OXE, [SC]
	
Sim: SimplerEnv; Real (Realman, Franka): pick, place, stack, open/close oven


DexVLA [20] 
◆
 	
ViT, ResNet-50
	
Qwen2-VL 2B, DistilBERT,
	
ScaleDP
	
Concat
	
DDPM
	
[SC]
	
Real (Franka, UR5e, AgileX): pick, fold shirt, bus table, pour


HybridVLA [120] 
◆
 	
DINOv2, SigLIP, CLIP
	
LLaMA 2, Phi-2
	
MLP
	
Concat
	
DDIM
	
OXE, DROID
	
Sim: RLBench; Real (Franka, AgileX): pick-place, pour, open drawer, fold, etc.


LAPA [121] 
◆
 	
C-ViViT, VQ-GAN
	
LWM-Chat-1M
	
MLP
	
Concat
	
LAPA
	
OXE, BridgeV2, Something V2
	
Sim: Language-Table, SimplerEnv; Real (Franka): pick, cover with towel, knock over


WorldVLA [122] 
◆
 	
VQ-GAN
	
Chameleon
	
TFM
	
Quant.
	
BC (disc, chunk.), world model
	
LIBERO, OpenVLA data
	
Sim: LIBERO


UniVLA [123] 
◆
 	
MoVQ
	
Emu3
	
TFM
	
Quant.
	
BC (disc, chunk.), world model
	
(Sim benchmark data)
	
Sim: CALVIN, LIBERO, SimplerEnv
III-B2Transformer-based Control Policies

Since the introduction of Transformers, control policies have converged to similar Transformer-based architectures.

Interactive Language [85] presents a robotic system wherein the low-level control policy can be guided in real-time by human instructions conveyed through language, enabling the completion of long-horizon rearrangement tasks. The efficacy of such language-based guidance is primarily attributed to the utilization of a meticulously collected dataset containing diverse language instructions, which surpasses previous datasets by an order of magnitude in scale.

Hiveformer [86] places significant emphasis on leveraging multi-view scene observations and maintaining the full observation history for a language-conditioned policy. This approach represents an advancement over previous systems such as CLIPort and BC-Z that only use the current observation. Notably, Hiveformer stands out as one of the early adopters of Transformer architecture as its policy backbone.

Gato [19] proposes a model that can play Atari games, caption images, and stack blocks, all with a single set of model parameters. This achievement is facilitated by a unified tokenization scheme, harmonizing the input and output across diverse tasks and domains. Consequently, Gato enables the simultaneous training of different tasks. Representing a significant milestone, Gato exemplifies the potential of constructing a “multi-modal, multi-task, multi-embodiment generalist agent”.

RoboCat [92] proposes a self-improvement process designed to enable an agent to rapidly adapt to new tasks with as few as 100 demonstrations. This self-improvement process iteratively finetunes the model and self-generates new data with the finetuned model. Built upon the Gato model, RoboCat incorporates the VQ-GAN image encoder [124]. During training, RoboCat predicts not only the next action but also future observations. The effectiveness of the self-improvement process is demonstrated through comprehensive experiments conducted in both simulated and real-world environments under multi-task, multi-embodiment settings.

RT-1 [94], developed by the same team as BC-Z, shares similarities with BC-Z but introduces some key distinctions. Notably, RT-1 employs a vision encoder based on the more efficient EfficientNet [125], departing from BC-Z’s use of ResNet. However, RT-1 does not use video as a task instruction. Additionally, RT-1 replaces the MLP action decoder in BC-Z with a Transformer decoder, producing discretized actions. This modification enables RT-1 to attend to past images, enhancing its performance compared to BC-Z.

Q-Transformer [95] extends RT-1 by introducing autoregressive Q-functions. In contrast to RT-1, which learns expert trajectories through imitation learning, Q-Transformer adopts Q-learning methods. Alongside the TD error objective of Q-learning, a conservative regularizer is incorporated to ensure that the maximum value action remains in-distribution. This approach allows Q-Transformer to leverage not only successful demonstrations but also failed trajectories for learning.

RT-Trajectory [96] adopts trajectory sketches as policy conditions instead of relying on language conditions or goal conditions. These trajectory sketches consist of curves that delineate the intended trajectory for the robot end-effector to follow. They can be manually specified through a graphical user interface, extracted from human demonstration videos, or generated by foundation models. RT-Trajectory’s policy is built upon the backbone of RT-1 and trained to control the robot arm to accurately follow the trajectory sketches. This approach facilitates generalization to novel objects, tasks, and skills, as trajectories from various tasks are transferable.

ACT [97] builds a conditional VAE policy with action chunking, requiring the policy to predict a sequence of actions rather than a single one. During inference, the action sequence is averaged using a method called temporal ensembling. RoboAgent [98] extends this approach with its MT-ACT model, demonstrating that action chunking improves temporal consistency. Additionally, RoboAgent introduces a semantic augmentation method that leverages inpainting to augment existing demonstrations.

RoboFlamingo [99] adapts the existing Flamingo VLM to a robot policy by attaching an LSTM-based policy head to the VLM. This demonstrates that pretrained VLMs can be effectively transferred to language-conditioned robotic manipulation tasks.

(a)FiLM
(b)Cross-attention
(c)Concatenation
(d)Quantization
(e)Tool Use
Figure 5:Representative architectures of VLA models. 
⊙
: Hadamard product. 
⊕
: Concatenation.
III-B3Control Policies for Multimodal Instructions

Multimodal instruction enables new ways to specify tasks, such as through demonstrations, by naming novel objects, or by pointing with a finger or mouse click.

VIMA [126] places a significant emphasis on multimodal prompts and the generalization capabilities of models. By incorporating multimodal prompts, more specific and intricate tasks can be formulated compared to traditional pure text prompts. VIMA introduces four main types of tasks: object manipulation, visual goal reaching, novel concept grounding, one-shot video imitation, visual constraint satisfaction, visual reasoning. These tasks are often challenging or even infeasible to express using only language prompts. VIMA-Bench has been developed to evaluate across four generalizability levels: placement, combinatorial, novel object, novel task.

MOO [93] extends RT-1 to handle multimodal prompts. Leveraging the backbone of RT-1, it incorporates OWL-ViT to encode images within the prompt. By expanding the RT-1 dataset with new objects and additional prompt images, MOO enhances the generalization capabilities of RT-1. This extension also facilitates new methods of specifying target objects, such as pointing with a finger, and clicking on graphical user interfaces.

III-B4Control Policies with 3D Vision

In our 3D world, it is reasonable to assume that 3D vision provides richer information than 2D images.

Point clouds are a common representation for 3D visual inputs due to their straightforward derivation from RGB-D inputs, as demonstrated by both DP3 [105] and 3D Diffuser Actor [108]. Voxels have likewise been extensively studied. VER [127] proposes voxelizing multi-view images into 3D cells in a coarse-to-fine manner, enhancing performance in vision-language navigation tasks. PerAct [87] facilitates efficient task learning with only a few demonstrations by leveraging voxel representations in both the observation and action spaces. The input comprises voxel maps reconstructed from multi-view RGB-D images, while the output corresponds to the best voxel for guiding the gripper’s movement. RoboUniView [100] improves performance by injecting 3D information from multi-perspective images through a novel UVFormer vision encoder, which is pretrained on a 3D occupancy task.

In contrast, Act3D [88] introduces a continuous resolution 3D feature field, with adaptive resolutions based on current task, addressing the computational cost of voxelization. RVT, RVT-2 [89, 90] propose re-rendering images from virtual views of the scene point cloud and using these images as inputs, rather than directly relying on 3D inputs.

III-B5Diffusion-based Control Policies

Diffusion-based action generation leverages the success of diffusion models in the field of computer vision.

Diffusion Policy [104] formulates a robot policy as a DDPM [128]. This approach incorporates a variety of techniques, including receding horizon control, visual conditioning, and the time-series diffusion Transformer. The effectiveness of this diffusion-based visuomotor policy is underscored by its proficiency in multimodal action distributions, high-dimensional action spaces, and training stability.

SUDD [106] presents a framework where an LLM guides data generation, and subsequently, the filtered dataset is distilled into a visuo-linguo-motor policy. This framework achieves language-guided data generation by composing an LLM with a suite of primitive robot utilities, such as grasp samplers and motion planners. It then extends Diffusion Policy by incorporating language-based conditioning for multi-task learning and facilitates the distillation of the filtered dataset.

Octo [107] introduces a Transformer-based diffusion policy characterized by a modular open-framework design, allowing for flexible connections from different task definition encoders, observation encoders, and action decoders to the Octo Transformer. Being among the first to utilize the OXE dataset [112], Octo demonstrates positive transfer and generalizability across diverse robots and tasks.

MDT [109] adapts the newly introduced DiT model [129] from computer vision to the action prediction head. DiT, originally proposed as a Transformer-based diffusion model, replaces the classical U-Net architecture for video generation. Coupled with two auxiliary objectives—masked generative foresight and contrastive latent alignment—MDT demonstrates better performance than the U-Net-based diffusion model, SUDD.

RDT-1B [110] is a diffusion-based foundation model for bimanual manipulation, also built on DiT. It addresses data scarcity by introducing a unified action format across various robots, enabling pretraining on heterogeneous multi-robot datasets with over 6K trajectories. As a result, RDT scales up to 1.2B parameters and demonstrates zero-shot generalization.

III-B6Diffusion-based Control Policies with 3D Vision

Several works have proposed combining 3D Vision with diffusion-based policies. DP3 [105] introduces 3D inputs to a diffusion policy, resulting in improved performance. Similarly, 3D Diffuser Actor [108] shares the core idea of DP3 but differs in model architecture by combining Act3D with Diffusion Policy. 3D-MoE [130] explores an efficient mixture-of-experts architecture with DiT-based action diffusion using a rectified flow scheduler.

III-B7Control Policies for Motion Planning

Motion planning involves decomposing movement tasks into discrete waypoints while satisfying constraints such as obstacle avoidance and kinematic limits.

Language costs [84] presents a novel approach to robot correction using natural language for human-in-the-loop robot control systems. This method leverages predicted cost maps generated from human instructions, which are then utilized by the motion planner to compute the optimal action. This framework enables users to correct goals, specify preferences, or recover from errors through intuitive language commands.

VoxPoser [103] employs LLM and VLM to create two 3D voxel maps that represent affordance and constraint. It leverages the programming capability of LLMs, and the perception capability of VLMs. LLM translates language instructions into executable code, invoking VLM to obtain object coordinates. Based on the composed affordance and constraint maps, VoxPoser employs model predictive control to generate a feasible trajectory for the robot arm’s end-effector. Notably, VoxPoser does not require any training, as it directly connects LLM and VLM for motion planning.

RoboTAP [131] breaks down demonstrations into stages marked by the opening and closing of the gripper. In each stage, RoboTAP uses the TAPIR algorithm to detect active points that track the relevant object from the source to the target pose. The path can then be used by visual servoing to control the robot. A motion plan is created by stringing together these stages, enabling few-shot visual imitation.

III-B8Control Policies with Point-based Action

Recent research has explored leveraging the capabilities of VLMs to select or predict point-based actions—a cost-effective alternative to building full VLAs.

PIVOT [132] casts robotics tasks as visual question answering, leveraging VLMs to select the best robot action from a set of visual proposals. Visual proposals are annotated in the form of keypoints on images. The VLM is iteratively prompted to refine them until the best option is identified.

RoboPoint [91] finetunes VLM using the task of spatial affordance prediction, which is to point at where to act on the image. These affordance points on 2D images are subsequently projected into 3D space using depth maps, forming the predicted robot action.

ReKep [38] is a constraint function that maps 3D keypoints in the scene to a numerical cost. Robotics manipulation tasks can be represented as a sequence of ReKep constraints, which are produced with large vision models and VLMs. Consequently, robot actions can be obtained by solving the constrained optimization problem.

III-B9Large VLA

Large VLA is equivalent to the original VLA definition proposed by RT-2 [2], as illustrated in Figure 2a. This terminology is analogous to the distinction between LLMs and general language models, or between large VLMs and general VLMs.

RT-2 [2] endeavors to harness the capabilities of large multimodal models in robotics tasks, drawing inspiration from models like PaLI-X and PaLM-E. The approach introduces co-fine-tuning, aiming to fit the model to both Internet-scale visual question answering (VQA) data and robot data. This training scheme enhances the model’s generalizability and brings about emergent capabilities.

RT-H [111] introduces an action hierarchy that includes an intermediate prediction layer of language motions, situated between language instructions and low-level actions (translation and rotation). This additional layer facilitates improved data sharing across different tasks. For example, both the language instructions “pick” and “pour” may involve the language motion “move the arm up”. Moreover, this action hierarchy enables users to specify corrections to recover from failures, which the model can then learn from.

RT-X [112] builds upon the previous RT-1 and RT-2 models. These models are re-trained using the newly introduced open-source large dataset, named Open X-Embodiment (OXE), which is orders of magnitude larger than previous datasets. The resulting models, RT-1-X and RT-2-X, both outperform their original versions.

OpenVLA [37] was later developed as an open-source counterpart to RT-2-X. They additionally explored efficient fine-tuning methods, including LoRA and model quantization. OpenVLA-OFT [113] applies an Optimized Fine-Tuning (OFT) recipe for improved efficiency and performance. TraceVLA [114] finetunes OpenVLA to enable visual trace prompting, enhancing spatial-temporal awareness.

𝜋
0
 [115] proposes a flow-matching architecture for transforming VLMs into VLAs. By incorporating an additional action expert based on the mixture-of-experts framework, it effectively inherits the Internet-scale knowledge in the VLM while extending its capabilities to address robotic tasks.

RoboMamba [116] replaces the costly Transformer with a Mamba state space model that has linear inference complexity, achieving efficient robotic reasoning and action capabilities.

SpatialVLA [117] introduces Ego3D Position Encoding for injecting 3D information and represents actions with Adaptive Action Grids for enhanced generalizability and robustness.

LAPA [121] devises the first unsupervised pretraining method for VLAs based on latent actions [69]. This approach employs a three-stage process to learn from internet-scale unlabeled videos. First, a VQ-VAE framework is pretrained to extract quantized latent actions between image frames. Next, a VLA model is pretrained to predict these latent actions. Finally, the model is finetuned using only a small robotic dataset to map the latent actions to actual robot actions.

The integration of large VLAs with diffusion has also been gaining popularity. TinyVLA [118] leverages the Diffusion Policy, while CogACT [119] utilizes a DiT action diffusion module. DexVLA [20] proposes embodied curriculum learning to progressively train a diffusion action expert, incorporating sub-step reasoning for decomposing long-horizon tasks. HybridVLA [120] integrates diffusion with the autoregression paradigm to fully leverage VLMs’ reasoning capabilities.

Visual autoregressive modeling (VAR) [133] with quantized visual tokens demonstrates improved performance over diffusion models in image generation. This suggests that the three modalities of VLAs can be unified under the autoregressive paradigm [134]. WorldVLA [122] and UniVLA [123] advance this direction by integrating VLAs with world models. It quantizes multimodal data into discrete tokens, forming a shared vocabulary of quantized multimodal tokens. Consequently, all modalities can be modeled autoregressively, enabling not only action and text generation but also image generation, thereby constituting a world model.

A recent trend in LLMs is equipping them with tool-use capabilities by generating code that calls tools via APIs [135]. Instruct2Act [102] follows this paradigm by integrating vision and action tools, enabling LLMs to perform robotic tasks.

Strengths and Limitations.

Architectures

Representative VLA architectures are illustrated in Figure 5. FiLM is used in RT-1 and thus its follow-up models inherit this mechanism. While cross-attention may offer superior performance with smaller model sizes, concatenation is simpler to implement and can achieve comparable results with larger models [126]. Quantization unifies multimodal tokens into a shared vocabulary, thus enabling integration with world models. The tool-use paradigm of LLMs can also be applied to robotic tasks.

Action types and their training objectives

Most low-level control policies predict actions for the end-effector pose while abstracting away the motion planning module that produces more fine-grained motions. While this abstraction facilitates better generalization to different embodiments, it also imposes limitations on dexterity.

The behavior cloning (BC) objective is used in imitation-learning, with different variants for different action types. The BC objective for continuous action [99] can be written as:

	
ℒ
Cont
=
∑
𝑡
MSE
⁡
(
𝑎
𝑡
,
𝑎
^
𝑡
)
,
		
(4)

where 
MSE
⁡
(
⋅
)
 stands for mean squared error. 
𝑎
𝑡
 is the action annotation from expert demonstrations.

Discrete action is achieved by dividing the action value range into a number of bins [94]. Its BC objective is:

	
ℒ
Disc
=
∑
𝑡
CE
⁡
(
𝑎
𝑡
,
𝑎
^
𝑡
)
,
		
(5)

where 
CE
⁡
(
⋅
)
 stands for cross-entropy loss.

CLIPort [31] and VIMA [126] use 
𝑆
​
𝐸
​
(
2
)
 action and its BC objective can be expressed as follows:

	
ℒ
𝑆
​
𝐸
​
(
2
)
	
=
CE
⁡
(
𝑎
pick
,
𝑎
^
pick
)
+
CE
⁡
(
𝑎
place
,
𝑎
^
place
)
.
		
(6)

The DDPM objective in diffusion-based control policies [104] is represented as:

	
ℒ
DDPM
=
MSE
⁡
(
𝜀
𝑘
,
𝜀
𝜃
​
(
𝑎
𝑡
+
𝜀
𝑘
,
𝑘
)
)
,
		
(7)

where 
𝜀
𝑘
 is a random noise for iteration 
𝑘
; 
𝜀
𝜃
 is the noise prediction network, i.e., the VLA model.

While discrete action has demonstrated superior performance in RT-1 [94], Octo [107] argues that it leads to early grasping issues. 
𝑆
​
𝐸
​
(
2
)
 actions require the model to predict only the pick and place end-effector poses, which is sufficient for many tabletop manipulation tasks. However, more complex tasks—such as “pouring water into a cup”—may require additional DoFs, thereby necessitating 
𝑆
​
𝐸
​
(
3
)
 actions [38]. Although point-based actions can be coarse-grained, they are more easily obtained from VLMs in a zero-shot manner [38].

Table IV:High-level task planners.
Model
 	
Vision
Model
	
Language Model / VLM
	
Low-level Control Policy
	
Architecture
	
Require Training
	
Environments, Embodiments, Tasks, and Skills


PaLM-E [11]
 	
ViT, OSRT
	
PaLM
	
Interactive Language, RT-1
	
Concat
	
✓
	
Sim: TAMP; Sim&Real-Mani: Language-Table; Real: SayCan setup


EmbodiedGPT [136]
 	
EVA ViT-G/14
	
LLaMA-7B
	
MLP
	
Xattn
	
✓
	
Sim-Mani: Franka Kitchen, Meta-World


LEO [137]
 	
ConvNeXt, PointNet++
	
Vicuna-7B
	
MLP
	
Concat
	
✓
	
Sim-Mani: CLIPort task subset; Sim-Navi: Habitat-web


3D-LLM [44]
 	
CLIP, 3D-CLR, ConceptFusion
	
Flamingo, BLIP-2
	
DD-PPO
	
Concat
	
✓
	
Sim-Navi: Habitat


ShapeLLM [138]
 	
ReCon++
	
LLaMA
	
 
	
Concat
	
✓
	
3D MM-Vet Benchmark


SayCan [10]
 	
 
	
PaLM, FLAN
	
BC-Z, MT-Opt
	
 
	
✗
	
Real-Navi&Mani (EDR): office kitchen


Translated 
⟨
LM
⟩
 [139]
 	
 
	
GPT-3, Codex
	
 
	
 
	
✗
	
Sim: VirtualHome


(SL)3 [140]
 	
 
	
T5-small
	
seq2seq
	
 
	
✓
	
Sim-Navi: ALFRED


Inner Monologue [9]
 	
MDETR, CLIP
	
InstructGPT, PaLM
	
CLIPort, heuristics, SayCan policies
	
Language
	
✗
	
Sim&Real-Mani: Tabletop Rearrangement; Real-Navi&Mani: Kitchen Mobile Mani.


LLM-Planner [141]
 	
HLSM object detector
	
GPT-3
	
HLSM
	
Language
	
✓
	
Sim-Navi: ALFRED


LID [142]
 	
CNN
	
GPT-2
	
 
	
Lang., Embed
	
✓
	
Sim: VirtualHome


SMs [143]
 	
CLIP, ViLD
	
GPT-3, RoBERTa
	
CLIPort
	
Lang., Code
	
✗
	
Sim-Mani: pick, place


ProgPrompt [144]
 	
ViLD
	
GPT-3
	
API
	
Code
	
✗
	
Sim: VirtualHome


ChatGPT for Robotics [145]
 	
YOLOv8
	
ChatGPT
	
API
	
Code
	
✗
	
Real-Navi: drone flight; Sim-Navi: AirSim, Habitat


CaP [146]
 	
ViLD, MDETR
	
GPT-3, Codex
	
API
	
Code
	
✗
	
Sim-Mani & Sim-Navi: pick, place, etc.


DEPS [147]
 	
CLIP
	
ChatGPT
	
MC-Controller, Steve-1
	
Code
	
✗
	
Sim: Minecraft


ConceptGraphs [148]
 	
SAM, CLIP, LLaVA
	
GPT-4
	
API
	
Code (JSON)
	
✗
	
Sim: AI2-THOR; Real (Spot Arm): pick-place
RT series

RT-1 [94] inspired a series of “Robotic Transformer” models. The Transformer backbone surpasses previous RNN backbones by harnessing the higher capacity of Transformers to absorb larger robot datasets. Preceding RT-1 was BC-Z, which solely utilized MLP layers for action prediction. Subsequent to RT-1, several works emerged, each introducing new capabilities. MOO adapted RT-1 to accommodate multimodal prompts. RT-Trajectory enabled RT-1 to process trajectory sketches as prompts. Q-Transformer utilized Q-learning to train RT-1. RT-2, based on ViT and LLM, introduced a completely different architecture from RT-1. RT-X retrained RT-1 and RT-2 with a significantly larger dataset, resulting in improved performance. Based on RT-2, RT-H [111] introduces action hierarchies for better data sharing.

LVLA vs generalized VLA

While LVLAs can greatly enhance instruction-following abilities because they can better parse user intentions, concerns arise regarding their training cost and deployment speed. Slow inference speed, in particular, can significantly impact performance in dynamic environments, as changes in the environment may occur during inference. Therefore, several methods have been proposed to improve efficiency. TinyVLA [118] focused on inference speed and data efficiency through smaller VLM and diffusion head for robot action. DeeR-VLA [101] proposes only partially activating the model with dynamic inference with early-exit.

Scaling Law

Similar to LLMs, scaling laws have also been observed in robotics [149], revealing the importance of model size, data quality, and diversity in environments and objects. Further research in this area can guide the development of VLAs with improved in-the-wild generalization capabilities.

IVTask Planners

A high-level task planner 
𝜋
𝜙
 aims to divide a complex task 
ℓ
 into a sequence of subtasks 
𝑝
𝑖
 (i.e., task plan), each serving as an instruction to low-level control policies 
𝜋
𝜃
:

	
[
𝑝
1
,
𝑝
2
,
…
,
𝑝
𝑁
]
∼
𝜋
𝜙
​
(
ℓ
,
𝑠
𝑡
)
.
		
(8)

This process is sometimes referred to as task or subgoal decomposition and is closely related to task and motion planning (TAMP) and embodied decision making. When equipped with task planners, VLAs can complete more complex, long-horizon tasks, as illustrated in Figure 4. Details are summarized in Table IV. Ideally, task plans should also incorporate optimal scheduling for these subtasks.

IV-AMonolithic Task Planners

A single LLM or multimodal LLM (MLLM) can typically generate task plans by employing a tailored framework or through finetuning on embodied datasets. We refer to these as monolithic models.

IV-A1End-to-end Task Planners

Similar to LVLAs, task planners can be implemented as end-to-end multimodal LLMs, leveraging their Internet-scale knowledge for task planning.

PaLM-E [11] integrates ViT and PaLM to create a large embodied multimodal language model capable of performing high-level embodied reasoning tasks. Based on perceived images and high-level language instructions, PaLM-E generates a text plan that serves as instructions for low-level robotic policies. In a mobile manipulation environment, they map the generated plan to executable low-level instructions with SayCan [10]. As the low-level policy executes actions, PaLM-E can also replan based on changes in the environment. With PaLM as its backbone, PaLM-E can handle both normal visual question answering (VQA) tasks, along with the additional embodied VQA tasks.

EmbodiedGPT [136] introduces the embodied-former, which outputs task-relevant instance-level features. This is achieved by incorporating information from vision encoder embeddings and embodied planning information provided by LLM. The instance feature serves to inform the low-level policy about the immediate next action to take.

IV-A2End-to-end Task Planners with 3D Vision

Some task planners also explore the use of 3D vision. Because the majority of current MLLMs deal with images as visual inputs, they require architectural changes to incorporate 3D visual inputs, and therefore they are usually end-to-end models.

LEO [137] uses a two-stage training strategy to integrate a point cloud encoder with an LLM: the first stage focuses on 3D vision-language alignment, followed by the second stage, which involves 3D vision-language-action instruction tuning. LEO performs well not only in 3D question-answering tasks but also in manipulation, navigation, and task planning.

3D-LLM [44] injects 3D information into LLMs and empowers them to perform 3D tasks, such as 3D-assisted dialog and navigation, using features from point cloud, gradSLAM, and neural voxel field. MultiPLY [150] is an object-centric embodied LLM that incorporates even more modalities, including audio, tactile, and thermal.

ShapeLLM [138] is built on the novel 3D vision encoder ReCon++, which distills knowledge from multi-view image and text teachers, along with a point cloud MAE. By integrating ReCon++ with LLaMA, ShapeLLM improves embodied interaction performance on their newly proposed 3D benchmark, 3D MM-Vet.

IV-A3Grounded Task Planners

Grounded task planning involves generating high-level actions while considering whether they can be executed by low-level control policies.

SayCan [10] is a framework designed to integrate high-level LLM planners with low-level control policies. In this framework, the LLM planner accepts users’ high-level instruction and “says” what the most probable next low-level skill is, a concept referred to as task-grounding. The low-level policy provides the value function as the affordance function, determining the possibility that the policy “can” complete the skill, known as world-grounding. By considering both LLM’s plan and affordance, the framework selects the optimal skill for the current state.

Translated 
⟨
LM
⟩
 [139] employs a two-step process to translate high-level instructions into executable actions. Initially, a pretrained causal LLM is utilized for plan generation, breaking down the high-level instruction into the next action expressed in free-form language phrases. Then, as these phrases may not directly map to VirtualHome actions, a pretrained masked LLM is then employed for action translation. This step involves calculating the similarity between the generated action phrases and VirtualHome action. The translated action is appended to the plan, and the updated plan is read by the LLM to generate the next action phrase. The two-step process is repeated until a complete plan is formed. A “Re-prompting” strategy [151] is further proposed to generate corrective actions when the agent encounters precondition errors.

(SL)3 [140] is a learning algorithm that alternates between three steps: segmentation, labeling, and parameter update. In the segmentation step, high-level subtasks are aligned with low-level actions, subtask descriptions are then inferred in the labeling step, and finally, the network parameters are updated. This approach enables a hierarchical policy to discover reusable skills with sparse natural language annotations.

IV-BModular Task Planners

Finetuning end-to-end models on embodied data can be expensive, and some approaches adopt a modular design by assembling off-the-shelf LLMs and VLMs into task planners. These approaches can also be viewed as following the tool-use architecture [135].

(a)Language-based
(b)Code-based
Figure 6:Different approaches to connect LLM to multi-modal modules in modular task planners.
IV-B1Language-based Task Planners

Language-based approaches use natural language descriptions as the medium for exchanging multimodal information, as illustrated in Figure 6a.

Inner Monologue [9] sits between high-level command and low-level policy to enable closed-loop control planning. It employs LLM to generate language instructions for low-level control policies and dynamically updates these instructions based on feedback received from the control policies. The feedback encompasses various sources: success feedback, object and scene feedback, and human feedback. As the feedback is communicated to the LLM in textual format, no additional training is required for the LLM. A similar approach is used in ReAct [76].

LLM-Planner [141] introduces a novel approach to constructing a hierarchical policy comprising a high-level planner and a low-level planner. The high-level planner harnesses the capabilities of LLM to generate natural language plans, while the low-level planner translates each subgoal within the plan into primitive actions. While sharing similarities with previous methods in its overall architecture, LLM-Planner distinguishes itself by incorporating a re-planning mechanism, aiding the robot to “get unstuck”.

LID [142] introduces a novel data collection procedure termed Active Data Gathering (ADG). A key aspect of ADG is hindsight relabeling, which reassigns labels to unsuccessful trajectories, effectively maximizing the utilization of data irrespective of their success. By converting all environmental inputs into textual descriptions, their language-model-based policy demonstrates enhanced combinatorial generalization.

Socratic Models (SMs) [143] presents a unique framework wherein diverse pretrained models are effectively composed without the need for finetuning. The framework is based on the key component, named multimodal-informed prompting, facilitating information exchange among models with varied multimodal capabilities. The idea is to utilize multimodal models to convert non-language inputs into language descriptions, effectively unifying different modalities within the language space. Beyond excelling in conventional multimodal tasks, SMs showcases its versatility in robot perception and planning. In addition to natural language plans, task plans can also be represented in the form of pseudocode.

IV-B2Code-based Task Planners

Code-based task planners leverage the coding ability of LLMs to generate task plans in the form of a program. Object detectors, VLMs, and control policies can be invoked via APIs, as shown in Figure 6b.

ProgPrompt [144] introduces a novel task-planning approach by prompting LLMs with program-like specifications detailing available actions and objects. This enables LLMs to generate high-level plans for household tasks in a few-shot manner. Environmental feedback can be incorporated through assertions within the program.

ChatGPT for Robotics [145] take advantage of the programming ability of ChatGPT to facilitate “user on the loop” control, a departure from the conventional “engineer in the loop” methodology. The procedure includes several steps: firstly, a list of APIs is defined, such as an object-detection API, a grasp API, a move API; secondly, a prompt is then constructed for ChatGPT, specifying the environment, API functionality, task goal, etc.; thirdly, iteratively prompting ChatGPT to write code with the defined APIs that can execute the task, provided the access to simulation and user feedback for evaluating the code quality and safety; finally, executing the ChatGPT generated code. In this procedure, ChatGPT serves as a high-level task planner and actions are generated through function calls to corresponding low-level APIs.

Code as policies (CaP) [146] also leverages the code-writing capability of LLMs. It employs GPT-3 or Codex to generate policy code, which, in turn, invokes perception modules and control APIs. CaP exhibits proficiency on spatial-geometric reasoning, generalization to new instructions, and parameterization for low-level control primitives. By leveraging the multimodal capabilities of GPT-4V, COME-robot [152] eliminates the need for perception APIs in CaP. This also opens up possibilities for open-ended reasoning and adaptive planning within a closed-loop framework, enabling capabilities such as failure recovery and free-form instruction following.

DEPS [147] stands for “Describe, Explain, Plan and Select”. This approach employs an LLM to generate plans and explain failures based on feedback descriptions collected from the environment—a process referred to as “self-explanation”, aiding in re-planning. Additionally, DEPS introduces a trainable goal selector to choose among parallel candidate sub-goals based on how easily they can be achieved, a crucial aspect often overlooked by other high-level task planners.

ConceptGraphs [148] introduces a method to convert observation sequences into open-vocabulary 3D scene graphs. Objects are extracted from RGB images using 2D segmentation models, and VLMs are employed to caption objects and establish inter-object relations, resulting in the formation of the 3D scene graph. This graph can then be translated into a text description (JSON), offering rich semantic and spatial relationships between entities to LLMs for task planning.

Strengths and Limitations.

Monolithic task planners that utilize grounded task planning focus on generating executable plans. End-to-end models share an architecture similar to most LVLAs and can be finetuned on specialized embodied data to achieve better performance. However, the training costs of such large models can be a concern. In contrast, modular task planners are more readily deployable because they leverage off-the-shelf LLMs and VLMs. Language-based task planners offer the advantage of seamless integration of LLMs and VLMs, as they are designed to operate in the natural language space. However, they often require extra steps to align the generated task plans with language instructions that are admissible to low-level control policies. Conversely, while code-based task planners may require manually wrapping VLMs and control policies in APIs and preparing clear documentation in advance, they enable code debugging and provide greater controllability. Nevertheless, their performance can be constrained by the programming capabilities of existing models.

VDatasets and Benchmarks
V-AReal-world Robot Datasets & Benchmarks

Embodied AI faces significant data scarcity issues because real-world robot data is not as readily available as language data. Collecting real-world robot datasets poses multiple challenges. Firstly, it is impeded by the cost and time required to procure robotic equipment, set up environments, and gather expert data through dedicated policies or human teleoperation. Secondly, the diverse types and configurations of robots introduce inconsistencies in sensory data, control modes, gripper types, etc. Lastly, accurately capturing object 6D poses and reproducing setups remains elusive. We summarize recent robot datasets in Table V. In addition, real-world benchmarks are further complicated by the need for human evaluation.

V-BSimulators, Simulated Robot Datasets & Benchmarks

Many researchers resort to simulated environments to circumvent real-world obstacles and scale the data collection process. We compare simulators and simulated benchmarks in Table VI. Nevertheless, this strategy presents its own challenges, chief among which is the sim-to-real gap. This discrepancy arises when models trained on simulated data exhibit poor performance during real-world deployment. The causes of this gap are multifaceted, encompassing unrealistic rendering quality, inaccuracies in physics simulations, and domain shifts characterized by object properties and robot motion planners. For instance, simulating non-rigid objects such as deformable objects or liquids presents significant difficulties. Moreover, importing new objects into simulators requires considerable effort, often involving techniques such as 3D scanning and mesh editing. Despite these hurdles, simulated environments provide automated evaluation metrics that aid researchers in consistently evaluating robotic models. Many benchmarks are based on simulators because they can precisely reproduce the experimental setup and yield fair comparisons of different models. Another technique, known as real-to-sim, can improve simulation fidelity, recreate failure cases, or facilitate digital twins.

Table V:Embodied Datasets. Per RT-X, skills correspond to verbs, and tasks are different combinations of verbs and objects. 
∗
 Dataset collected in simulation rather than the real world. Some datasets are continually updated, and we include only the original version. Table adapted from [153, 154].
Dataset
 	
Skills
	
Tasks
	
Scenes
	
Episodes
	
Collection
	
Obs.
	
Instruction
	
Robots


MIME [155]
 	
12
	
20
	
1
	
8.3K
	
Human teleop.
	
RGBD
	
Demo
	
Baxter


∗
RoboTurk [156]
 	
2
	
 
	
1
	
2.1K
	
Human teleop.
	
RGB
	
 
	
Sawyer


RoboNet [157]
 	
 
	
 
	
10
	
162K
	
Script
	
RGB
	
Goal image
	
(7 robots)


MT-Opt [158]
 	
2
	
12
	
1
	
800K
	
Script
	
RGB
	
Lang
	
(7 robots)


BC-Z [79]
 	
3
	
100
	
1
	
25.9K
	
Human teleop.
	
RGB
	
Lang, demo
	
EDR


Fractal [94]
 	
12
	
700+
	
2
	
130K
	
Human teleop.
	
RGB
	
Lang
	
EDR


MOO [93]
 	
5
	
 
	
 
	
59.1K
	
Human teleop.
	
RGB
	
Multimodal
	
EDR


∗
VIMA [126]
 	
17
	
 
	
1
	
650K
	
Script
	
RGB
	
Multimodal
	
UR5


RoboSet [159]
 	
12
	
38
	
11
	
98.5K
	
Human, script
	
RGBD
	
Lang
	
Franka


BridgeV2 [160]
 	
13
	
 
	
24
	
60.1K
	
Human, script
	
RGBD
	
Lang
	
WidowX


RH20T [153]
 	
42
	
147
	
7
	
110K+
	
Human teleop.
	
RGBD
	
Lang
	
(4 robots)


DROID [154]
 	
86
	
 
	
564
	
76K
	
Human teleop.
	
RGBD
	
Lang
	
Franka


OXE [112]
 	
527
	
160K
	
311
	
1M+
	
Aggregate data
	
RGBD
	
Lang
	
(22 robots)
Table VI:Simulators and simulated benchmarks. Control: continuous control tasks. D, S, A, N: depth, segmentation, audio, normal. Force: simulated contact force between end-effector and item. PD: pre-defined. Table adapted from [161, 162].
Name
 	
Scenes
/Rooms
	
Objects
/Cat
	
UI
	
Physics Engine
	
Task
	
Observation
	
Action
	
Agent
	
Description
	
Related


Gibson [163]
 	
572/-
	
 
	
 
	
Pybullet
	
Navi
	
RGB, D, N, S
	
 
	
 
	
Navi only
	
 


iGibson [164, 165, 162]
 	
15/108
	
152/5
	
Mouse, VR
	
Pybullet
	
Navi, Mani
	
RGB, D, S, N, Flow, LiDAR
	
Force
	
TurtleBot v2, LoCoBot, etc.
	
VR, Continuous Extended States. Versions: 0.5, 1.0, 2.0
	
Benchmarks: BEHAVIOR-100 [161], BEHAVIOR-1K [166]


SAPIEN [167]
 	
 
	
2346/-
	
Code
	
PhysX
	
Navi, Mani
	
RGB, D, S
	
Force
	
Franka
	
Articulation, Ray Tracing
	
VoxPoser. Benchmark: SIMPLER [168]


AI2-THOR [169]
 	
-/120
	
118/118
	
Mouse
	
Unity
	
Navi, Mani
	
RGB, D, S, A
	
Force, PD
	
ManipulaTHOR, LoCoBot, etc.
	
Object States, Task Planning. Versions: [170, 171]
	
Benchmarks: ALFRED, RoPOR [172]


VirtualHome [173]
 	
7/-
	
-/509
	
Lang
	
Unity
	
Navi, Mani
	
RGB, D, S
	
Force, PD
	
Human
	
Object States, Task Planning
	
LID, Translated〈LM〉, ProgPrompt


TDW [174]
 	
15/120
	
112/50
	
VR
	
Unity, Flex
	
Navi, Mani
	
RGB, D, S, A
	
Force
	
Fetch, Sawyer, Baxter
	
Audio, Fluids
	
 


RLBench [175]
 	
1/-
	
28/28
	
Code
	
Bullet
	
Mani
	
RGB, D, S
	
Force
	
Franka
	
Tiered Task Difficulty
	
Hiveformer, PerAct


Meta-World [176]
 	
1/-
	
80/7
	
Code
	
MuJoCo
	
Mani
	
Pose
	
Force
	
Sawyer
	
Meta-RL
	
R3M, VC-1, Vi-PRoM, EmbodiedGPT


CALVIN [177]
 	
4/-
	
7/5
	
 
	
Pybullet
	
Mani
	
RGB, D
	
Force
	
Franka
	
Long-horizon Lang-cond tasks
	
GR-1, HULC, RoboFlamingo


Franka Kitchen [178]
 	
1/-
	
10/6
	
VR
	
MuJoCo
	
Mani
	
Pose
	
Force
	
Franka
	
Extended by R3M with RGB
	
R3M, Voltron, Vi-PRoM, Diffusion Policy, EmbodiedGPT


Habitat [179, 180]
 	(Matterport + Gibson)	
Mouse
	
Bullet
	
Navi
	
RGB, D, S, A
	
Force
	
Fetch, Franka, AlienGO
	
Fast, Navi only. Versions: 1.0, 2.0, Rearrangement [181]
	
VC-1, PACT; Benchmark: OVMM [182]


ALFRED [183]
 	
-/120
	
84/84
	
 
	
Unity
	
Navi, Mani
	
RGB, D, S
	
PD
	
Human
	
Diverse long-horizon tasks
	
(SL)3, LLM-Planner


DMC [184]
 	
1/-
	
4/4
	
Code
	
MuJoCo
	
Control
	
RGB, D
	
Force
	
 
	
Continuous RL
	
VC-1, SMART


OpenAI Gym [185]
 	
1/-
	
4/4
	
Code
	
MuJoCo
	
Control
	
RGB
	
Force
	
 
	
Single agent RL environments
	
 


Genesis [186]
 	
(Rigid, deformable, liquid, etc.)
	
Code
	
(Propri-etary)
	
Navi, Mani
	
RGB, D, S, N
	
Force
	
Franka, Unitree, etc.
	
High-speed comprehensive physics simulation
	
 
V-CAutomated Dataset Collection

Several approaches advocate for automated dataset collection. RoboGen [187] employs a generative simulation paradigm that proposes interesting skills, simulates corresponding environments, and selects optimal learning approaches to train policies for acquiring those skills. AutoRT [188] functions as a robot orchestrator driven by LLMs, generating tasks, filtering them by affordance, and utilizing either autonomous policies or human teleoperators to collect and evaluate data. DIAL [189] focuses on augmenting language instructions in existing datasets using VLMs. RoboPoint [91] generates scenes procedurally with randomized 3D layouts, objects, and camera viewpoints.

V-DHuman Datasets

An alternative strategy to address data scarcity in real-world settings is to leverage human data. Human behavior offers plentiful guidance for robot policies due to its dexterity and diversity. However, this strategy also comes with inherent drawbacks. Capturing and transferring human hand/body motions to robot embodiments is inherently difficult. Moreover, the inconsistency in human data poses a hurdle, as some data may be egocentric while others are captured from third-person perspectives. Additionally, filtering human data to extract useful information can be labor-intensive. These obstacles underscore the complexities involved in incorporating human data into robot learning processes. UMI [190] proposes a method to mitigate these issues using hand-held grippers. For a more comprehensive comparison of human datasets, we refer interested readers to [191].

V-ETask Planning Benchmarks

EgoPlan-Bench [192] focuses on benchmarking real-world task planning with human annotations. PlanBench [193, 194] comprehensively assesses various aspects of task planning ability, such as cost optimality, plan verification, and replanning. LoTa-Bench [195] directly evaluates task planning by executing the generated plans in simulators and calculating success rates. Embodied Agent Interface (EAI) [196] argues that this approach fails to pinpoint issues in LLMs. By formalizing the input-output of LLM-based modules for decision-making tasks, EAI enables more fine-grained metrics beyond success rates.

V-FEmbodied Question Answering Benchmarks

Embodied question answering (EQA) benchmarks, as summarized in Table VII, do not directly evaluate robotic tasks like manipulation and navigation, but they are aimed at other relevant abilities for embodied AI, such as spatial reasoning, physics understanding, as well as world knowledge. EQA is akin to previous visual question answering benchmarks, but differs in that the agent can actively explore the environment before providing an answer. EmbodiedQA [197] and IQUAD [198] were among the first works to introduce this type of benchmarks. MT-EQA [199] focuses on complex questions involving multiple targets. MP3D-EQA [200] converts previous RGB inputs to point clouds, testing 3D perception capabilities.

Active exploration requires access to a simulator, limiting the types of data that can be used, such as real-world videos. EgoVQA [201] shifts the focus of VQA to egocentric videos. EgoTaskQA [202] emphasizes spatial, temporal, and causal relationship reasoning. EQA-MX [203] investigates multimodal expressions (MX), including regular verbal utterances and nonverbal gestures like eye gaze and pointing. OpenEQA [204] evaluates seven main categories, including functional reasoning and world knowledge.

Table VII:Embodied question answering benchmarks. Explore: active exploration.
Benchmark
 	
QA Pairs
	
Scenes
	
Source
	
Answer Type
	
Collection
	
Explore
	
Metrics


EQA [197]
 	
5K
	
750 envs
	
House3D simulator
	
Answer set (172 answers)
	
Template
	
✓
	
Accuracy


IQUAD [198]
 	
75K
	
30 rooms
	
AI2-THOR
	
Multiple choice
	
Template
	
✓
	
Accuracy


MT-EQA [199]
 	
19.3K
	
588 envs
	
House3D simulator
	
Binary answer
	
Template
	
✓
	
Accuracy


MP3D-EQA [200]
 	
1,136
	
83 envs
	
MatterPort3D
	
Answer set (53 answers)
	
Template
	
✓
	
Accuracy


EgoVQA [201]
 	
600
	
16 videos
	
IU Multi-view
	
Multiple choice (1 out of 5)
	
Human annotators
	
✗
	
Accuracy


EgoTaskQA [202]
 	
40K
	
2K videos
	
LEMMA dataset
	
Open answer, binary verification
	
Human annotators & template
	
✗
	
Accuracy


EQA-MX [203]
 	
8.2M
	
750K images
	
CAESAR simulator
	
Answer set
	
Question templates, answer set
	
✗
	
Accuracy


OpenEQA [204]
 	
557 + 1079
	
180 envs
	
HM3D + ScanNet
	
Open answer
	
Human annotators
	
✗
	
LLM Score
VIChallenges and Future Directions
Safety first

Safety is paramount in robotics, as robots interact directly with the physical world. Ensuring the safety of robotic systems requires the integration of real-world commonsense. This involves the incorporation of robust safety guardrails, risk assessment frameworks, and human-robot interaction protocols. RLHF and “evaluation without execution” can also significantly lower safety risks [22]. Interpretability and expandability of VLA decision-making processes are also crucial for enhancing robot safety through error diagnosis and troubleshooting.

Datasets & Benchmarks

In addition to the issues discussed in §V, comprehensive benchmarks that cover a wide range of skills, objects, embodiments, and environments remain to be developed. Moreover, metrics beyond the success rate are needed for a fine-grained diagnosis of issues in VLA models, as highlighted by EAI [196] for LLMs.

Foundation Models & Generalization

VLA foundation models or robotic foundation models (RFM) for embodied AI remains an open research topic primarily due to the diversity in embodiments, environments, and tasks. Many have made significant progress [110, 115], but still lacks generalization capability on par with LLMs in NLP. Attaining such a level of generalization is very challenging, as it requires the development of many core AGI capabilities.

Multimodality

VLAs inherit many challenges associated with multimodal models, such as obtaining useful embeddings and aligning different modalities. Current approaches, like ImageBind [205] and LanguageBind [206], align different modalities to the image or language embedding space, respectively. Within a unified embedding space, MLLMs can accommodate diverse modalities and become more general. However, whether focusing on embeddings alone is sufficient remains under debate. Although beyond the scope of VLAs, other modalities—such as audio [47], haptics [207], and gaze [208]—have proven useful for certain embodied AI applications. For instance, modeling human gaze data using an additional gaze network [209] or incorporating it via an auxiliary loss [210] has been shown to enhance the performance of the primary policy network. These approaches demonstrate that RL policies can benefit from human-like visual attention, as human gaze often reveals the most salient locations in the environment [211]. While incorporating additional modalities is often advantageous, it inevitably introduces extra complexity into the model design.

Framework for Long-Horizon Tasks

The hierarchical framework is currently the most practical approach for long-horizon tasks. However, it increases system complexity and potential points of failure. Frequent task execution failures can trigger re-planning, which can cause significant latency. Moreover, monolithic task planners share similar architectures with LVLAs, so employing two large models can be redundant and may hinder scalability. Additionally, although modular task planners typically do not require training, they are not plug-and-play: language-based models may generate subtasks that control policies cannot execute, whereas code-based models require modules to be pre-wrapped in APIs manually. Therefore, developing a unified framework that directly translates long-horizon tasks into low-level control signals in an end-to-end fashion is worth exploring.

Real-Time Responsiveness

Unlike conversational AI, many robotic applications require real-time decision-making to respond to dynamic environments. If inference time cannot keep pace with environmental changes, the model may generate obsolete actions repeatedly. However, current VLA models—especially LVLAs and task planners—face a trade-off between speed and capacity. Novel mechanisms are thus needed to strike an optimal balance.

Multi-agent Systems

Cooperative multi-agent systems offer benefits such as distributed perception and collaborative fault recovery. However, they also face challenges, including effective communication, coordinated dispatching, and fleet heterogeneity. In certain scenarios, individual agents may have conflicting goals, which further increases complexity.

Ethical and Societal Implications

Robotics has always raised various ethical, societal, and legal concerns. These include risks related to privacy, job displacement, decision-making bias, and the impact on social norms and human relationships.

Applications

Most current VLAs focus on household or industrial settings, but a wider range of applications is possible, such as virtual assistants, autonomous vehicles, and agricultural robots. Various embodiments may also call for specialized VLAs, including dexterous hands, drones, quadruped robots, and humanoid robots. One particularly important field is healthcare, encompassing surgical robots [212], care robots [213]. Healthcare demands higher safety and privacy standards and may necessitate novel techniques such as human-in-the-loop (HITL) control and federated learning. Moreover, due to the significant domain gap, specialized vision models might be needed for medical images.

VIIConclusion

Vision-language-action models hold immense promise for enabling embodied agents to interact with the physical world and fulfill users’ instructions. This paper is the first survey to review large VLAs alongside generalized VLAs. Our taxonomy provides a high-level overview of three main lines of research: key components, control policies, and task planners. We meticulously analyze and compare their technical details, including model architectures, training strategies, and individual modules. Additionally, we highlight essential resources for training and evaluating VLAs, such as datasets, simulators, and benchmarks. We hope this survey captures the rapidly evolving landscape of embodied AI and inspires future research.

References
[1]
↑
	OpenAI. (2023) Introducing chatgpt. [Online]. Available: https://openai.com/blog/chatgpt
[2]
↑
	A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, X. Chen, K. Choromanski, T. Ding, D. Driess, A. Dubey, C. Finn, P. Florence, C. Fu, M. G. Arenas, K. Gopalakrishnan, K. Han, K. Hausman, A. Herzog, J. Hsu, B. Ichter, A. Irpan, N. J. Joshi, R. Julian, D. Kalashnikov, Y. Kuang, I. Leal, L. Lee, T. E. Lee, S. Levine, Y. Lu, H. Michalewski, I. Mordatch, K. Pertsch, K. Rao, K. Reymann, M. S. Ryoo, G. Salazar, P. Sanketi, P. Sermanet, J. Singh, A. Singh, R. Soricut, H. T. Tran, V. Vanhoucke, Q. Vuong, A. Wahid, S. Welker, P. Wohlhart, J. Wu, F. Xia, T. Xiao, P. Xu, S. Xu, T. Yu, and B. Zitkovich, “RT-2: vision-language-action models transfer web knowledge to robotic control,” CoRR, vol. abs/2307.15818, 2023.
[3]
↑
	A. Krizhevsky, I. Sutskever, and G. E. Hinton, “Imagenet classification with deep convolutional neural networks,” in NIPS, 2012, pp. 1106–1114.
[4]
↑
	A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, “Attention is all you need,” in NIPS, 2017, pp. 5998–6008.
[5]
↑
	V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. A. Riedmiller, A. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D. Wierstra, S. Legg, and D. Hassabis, “Human-level control through deep reinforcement learning,” Nat., vol. 518, no. 7540, pp. 529–533, 2015.
[6]
↑
	S. Levine, P. Pastor, A. Krizhevsky, and D. Quillen, “Learning hand-eye coordination for robotic grasping with large-scale data collection,” in ISER, ser. Springer Proceedings in Advanced Robotics, vol. 1. Springer, 2016, pp. 173–184.
[7]
↑
	J. Li, D. Li, S. Savarese, and S. C. H. Hoi, “BLIP-2: bootstrapping language-image pre-training with frozen image encoders and large language models,” in ICML, vol. 202. PMLR, 2023, pp. 19 730–19 742.
[8]
↑
	H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” CoRR, vol. abs/2304.08485, 2023.
[9]
↑
	W. Huang, F. Xia, T. Xiao, H. Chan, J. Liang, P. Florence, A. Zeng, J. Tompson, I. Mordatch, Y. Chebotar, P. Sermanet, T. Jackson, N. Brown, L. Luu, S. Levine, K. Hausman, and B. Ichter, “Inner monologue: Embodied reasoning through planning with language models,” in CoRL, vol. 205. PMLR, 2022, pp. 1769–1782.
[10]
↑
	B. Ichter, A. Brohan, Y. Chebotar, C. Finn, K. Hausman, A. Herzog, D. Ho, J. Ibarz, A. Irpan, E. Jang, R. Julian, D. Kalashnikov, S. Levine, Y. Lu, C. Parada, K. Rao, P. Sermanet, A. Toshev, V. Vanhoucke, F. Xia, T. Xiao, P. Xu, M. Yan, N. Brown, M. Ahn, O. Cortes, N. Sievers, C. Tan, S. Xu, D. Reyes, J. Rettinghouse, J. Quiambao, P. Pastor, L. Luu, K. Lee, Y. Kuang, S. Jesmonth, N. J. Joshi, K. Jeffrey, R. J. Ruano, J. Hsu, K. Gopalakrishnan, B. David, A. Zeng, and C. K. Fu, “Do as I can, not as I say: Grounding language in robotic affordances,” in CoRL, vol. 205. PMLR, 2022, pp. 287–318.
[11]
↑
	D. Driess, F. Xia, M. S. M. Sajjadi, C. Lynch, A. Chowdhery, B. Ichter, A. Wahid, J. Tompson, Q. Vuong, T. Yu, W. Huang, Y. Chebotar, P. Sermanet, D. Duckworth, S. Levine, V. Vanhoucke, K. Hausman, M. Toussaint, K. Greff, A. Zeng, I. Mordatch, and P. Florence, “Palm-e: An embodied multimodal language model,” in ICML, vol. 202. PMLR, 2023, pp. 8469–8488.
[12]
↑
	R. Firoozi, J. Tucker, S. Tian, A. Majumdar, J. Sun, W. Liu, Y. Zhu, S. Song, A. Kapoor, K. Hausman, B. Ichter, D. Driess, J. Wu, C. Lu, and M. Schwager, “Foundation models in robotics: Applications, challenges, and the future,” CoRR, vol. abs/2312.07843, 2023.
[13]
↑
	J. Wang, Z. Wu, Y. Li, H. Jiang, P. Shu, E. Shi, H. Hu, C. Ma, Y. Liu, X. Wang, Y. Yao, X. Liu, H. Zhao, Z. Liu, H. Dai, L. Zhao, B. Ge, X. Li, T. Liu, and S. Zhang, “Large language models for robotics: Opportunities, challenges, and perspectives,” CoRR, vol. abs/2401.04334, 2024.
[14]
↑
	Y. Hu, Q. Xie, V. Jain, J. Francis, J. Patrikar, N. V. Keetha, S. Kim, Y. Xie, T. Zhang, S. Zhao, Y. Q. Chong, C. Wang, K. P. Sycara, M. Johnson-Roberson, D. Batra, X. Wang, S. A. Scherer, Z. Kira, F. Xia, and Y. Bisk, “Toward general-purpose robots via foundation models: A survey and meta-analysis,” CoRR, vol. abs/2312.08782, 2023.
[15]
↑
	K. Kawaharazuka, T. Matsushima, A. Gambardella, J. Guo, C. Paxton, and A. Zeng, “Real-world robot applications of foundation models: a review,” Adv. Robotics, vol. 38, no. 18, pp. 1232–1254, 2024.
[16]
↑
	T. Brooks, B. Peebles, C. Holmes, W. DePue, Y. Guo, L. Jing, D. Schnurr, J. Taylor, T. Luhman, E. Luhman, C. Ng, R. Wang, and A. Ramesh, “Video generation models as world simulators,” OpenAI, 2024. [Online]. Available: https://openai.com/research/video-generation-models-as-world-simulators
[17]
↑
	L. Chen, K. Lu, A. Rajeswaran, K. Lee, A. Grover, M. Laskin, P. Abbeel, A. Srinivas, and I. Mordatch, “Decision transformer: Reinforcement learning via sequence modeling,” in NeurIPS, 2021, pp. 15 084–15 097.
[18]
↑
	M. Janner, Q. Li, and S. Levine, “Offline reinforcement learning as one big sequence modeling problem,” in NeurIPS, 2021, pp. 1273–1286.
[19]
↑
	S. E. Reed, K. Zolna, E. Parisotto, S. G. Colmenarejo, A. Novikov, G. Barth-Maron, M. Gimenez, Y. Sulsky, J. Kay, J. T. Springenberg, T. Eccles, J. Bruce, A. Razavi, A. Edwards, N. Heess, Y. Chen, R. Hadsell, O. Vinyals, M. Bordbar, and N. de Freitas, “A generalist agent,” Trans. Mach. Learn. Res., vol. 2022, 2022.
[20]
↑
	J. Wen, Y. Zhu, J. Li, Z. Tang, C. Shen, and F. Feng, “Dexvla: Vision-language model with plug-in diffusion expert for general robot control,” CoRR, vol. abs/2502.05855, 2025.
[21]
↑
	Y. Ma, D. Chi, S. Wu, Y. Liu, Y. Zhuang, and I. King, “Actra: Optimized transformer architecture for vision-language-action models in robot learning,” CoRR, vol. abs/2408.01147, 2024.
[22]
↑
	A. Hiranaka, M. Hwang, S. Lee, C. Wang, L. Fei-Fei, J. Wu, and R. Zhang, “Primitive skill-based robot learning from human evaluative feedback,” in IROS, 2023, pp. 7817–7824.
[23]
↑
	N. Shinn, F. Cassano, A. Gopinath, K. Narasimhan, and S. Yao, “Reflexion: language agents with verbal reinforcement learning,” in NeurIPS, 2023.
[24]
↑
	Y. J. Ma, W. Liang, G. Wang, D. Huang, O. Bastani, D. Jayaraman, Y. Zhu, L. Fan, and A. Anandkumar, “Eureka: Human-level reward design via coding large language models,” in ICLR. OpenReview.net, 2024.
[25]
↑
	A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, “Learning transferable visual models from natural language supervision,” in ICML, vol. 139. PMLR, 2021, pp. 8748–8763.
[26]
↑
	S. Nair, A. Rajeswaran, V. Kumar, C. Finn, and A. Gupta, “R3M: A universal visual representation for robot manipulation,” in CoRL, vol. 205. PMLR, 2022, pp. 892–909.
[27]
↑
	Y. J. Ma, S. Sodhani, D. Jayaraman, O. Bastani, V. Kumar, and A. Zhang, “VIP: towards universal visual reward and representation via value-implicit pre-training,” in ICLR, 2023.
[28]
↑
	I. Radosavovic, T. Xiao, S. James, P. Abbeel, J. Malik, and T. Darrell, “Real-world robot learning with masked visual pre-training,” in CoRL, vol. 205. PMLR, 2022, pp. 416–426.
[29]
↑
	J. Devlin, M. Chang, K. Lee, and K. Toutanova, “BERT: pre-training of deep bidirectional transformers for language understanding,” in NAACL-HLT (1). Association for Computational Linguistics, 2019, pp. 4171–4186.
[30]
↑
	I. Radosavovic, B. Shi, L. Fu, K. Goldberg, T. Darrell, and J. Malik, “Robot learning with sensorimotor pre-training,” in CoRL, vol. 229. PMLR, 2023, pp. 683–693.
[31]
↑
	M. Shridhar, L. Manuelli, and D. Fox, “Cliport: What and where pathways for robotic manipulation,” in CoRL, vol. 164. PMLR, 2021, pp. 894–906.
[32]
↑
	A. Khandelwal, L. Weihs, R. Mottaghi, and A. Kembhavi, “Simple but effective: CLIP embeddings for embodied AI,” in CVPR. IEEE, 2022, pp. 14 809–14 818.
[33]
↑
	S. Y. Gadre, M. Wortsman, G. Ilharco, L. Schmidt, and S. Song, “Cows on pasture: Baselines and benchmarks for language-driven zero-shot object navigation,” in CVPR. IEEE, 2023, pp. 23 171–23 181.
[34]
↑
	A. Majumdar, K. Yadav, S. Arnaud, Y. J. Ma, C. Chen, S. Silwal, A. Jain, V. Berges, T. Wu, J. Vakil, P. Abbeel, J. Malik, D. Batra, Y. Lin, O. Maksymets, A. Rajeswaran, and F. Meier, “Where are we in the search for an artificial visual cortex for embodied intelligence?” in NeurIPS, 2023.
[35]
↑
	S. Karamcheti, S. Nair, A. S. Chen, T. Kollar, C. Finn, D. Sadigh, and P. Liang, “Language-driven representation learning for robotics,” in RSS, 2023.
[36]
↑
	M. Oquab, T. Darcet, T. Moutakanni, H. V. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, M. Assran, N. Ballas, W. Galuba, R. Howes, P. Huang, S. Li, I. Misra, M. Rabbat, V. Sharma, G. Synnaeve, H. Xu, H. Jégou, J. Mairal, P. Labatut, A. Joulin, and P. Bojanowski, “Dinov2: Learning robust visual features without supervision,” Trans. Mach. Learn. Res., vol. 2024, 2024.
[37]
↑
	M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. P. Foster, G. Lam, P. Sanketi, Q. Vuong, T. Kollar, B. Burchfiel, R. Tedrake, D. Sadigh, S. Levine, P. Liang, and C. Finn, “Openvla: An open-source vision-language-action model,” CoRR, vol. abs/2406.09246, 2024.
[38]
↑
	W. Huang, C. Wang, Y. Li, R. Zhang, and L. Fei-Fei, “Rekep: Spatio-temporal reasoning of relational keypoint constraints for robotic manipulation,” CoRR, vol. abs/2409.01652, 2024.
[39]
↑
	M. Assran, Q. Duval, I. Misra, P. Bojanowski, P. Vincent, M. G. Rabbat, Y. LeCun, and N. Ballas, “Self-supervised learning from images with a joint-embedding predictive architecture,” in CVPR. IEEE, 2023, pp. 15 619–15 629.
[40]
↑
	J. Shang, K. Schmeckpeper, B. B. May, M. V. Minniti, T. Kelestemur, D. Watkins, and L. Herlant, “Theia: Distilling diverse vision foundation models for robot learning,” in CoRL, ser. Proceedings of Machine Learning Research, vol. 270. PMLR, 2024, pp. 724–748.
[41]
↑
	S. Parisi, A. Rajeswaran, S. Purushwalkam, and A. Gupta, “The unsurprising effectiveness of pre-trained vision models for control,” in ICML, vol. 162. PMLR, 2022, pp. 17 359–17 371.
[42]
↑
	Y. LeCun and Courant, “A path towards autonomous machine intelligence version 0.9.2, 2022-06-27,” 2022. [Online]. Available: https://api.semanticscholar.org/CorpusID:251881108
[43]
↑
	W. Shen, G. Yang, A. Yu, J. Wong, L. P. Kaelbling, and P. Isola, “Distilled feature fields enable few-shot language-guided manipulation,” in CoRL, vol. 229. PMLR, 2023, pp. 405–424.
[44]
↑
	Y. Hong, H. Zhen, P. Chen, S. Zheng, Y. Du, Z. Chen, and C. Gan, “3d-llm: Injecting the 3d world into large language models,” in NeurIPS, 2023.
[45]
↑
	T. Xie, Z. Zong, Y. Qiu, X. Li, Y. Feng, Y. Yang, and C. Jiang, “Physgaussian: Physics-integrated 3d gaussians for generative dynamics,” in CVPR. IEEE, 2024, pp. 4389–4398.
[46]
↑
	H. Li, Y. Zhou, T. Tang, J. Song, Y. Zeng, M. Kampffmeyer, H. Xu, and X. Liang, “Unigs: Unified language-image-3d pretraining with gaussian splatting,” in ICLR, 2025.
[47]
↑
	A. Thankaraj and L. Pinto, “That sounds right: Auditory self-supervision for dynamic robot manipulation,” in CoRL, vol. 229. PMLR, 2023, pp. 1036–1049.
[48]
↑
	Y. Jing, X. Zhu, X. Liu, Q. Sima, T. Yang, Y. Feng, and T. Kong, “Exploring visual pre-training for robot manipulation: Datasets, models and methods,” in IROS, 2023, pp. 11 390–11 395.
[49]
↑
	F. Liu, H. Liu, A. Grover, and P. Abbeel, “Masked autoencoding for scalable and generalizable decision making,” in NeurIPS, 2022.
[50]
↑
	J. Li, Q. Gao, M. Johnston, X. Gao, X. He, H. Shi, S. Shakiah, R. Ghanadan, and W. Y. Wang, “Mastering robot manipulation with multimodal prompts through pretraining and multi-task fine-tuning,” in ICML, 2024.
[51]
↑
	Y. Sun, S. Ma, R. Madaan, R. Bonatti, F. Huang, and A. Kapoor, “SMART: self-supervised multi-task pretraining with control transformers,” in ICLR, 2023.
[52]
↑
	R. Bonatti, S. Vemprala, S. Ma, F. Frujeri, S. Chen, and A. Kapoor, “PACT: perception-action causal transformer for autoregressive robotics pre-training,” in IROS, 2023, pp. 3621–3627.
[53]
↑
	B. Baker, I. Akkaya, P. Zhokhov, J. Huizinga, J. Tang, A. Ecoffet, B. Houghton, R. Sampedro, and J. Clune, “Video pretraining (VPT): learning to act by watching unlabeled online videos,” in NeurIPS, 2022.
[54]
↑
	H. Wu, Y. Jing, C. Cheang, G. Chen, J. Xu, X. Li, M. Liu, H. Li, and T. Kong, “Unleashing large-scale video generative pre-training for visual robot manipulation,” in ICLR, 2024.
[55]
↑
	D. Hafner, T. P. Lillicrap, J. Ba, and M. Norouzi, “Dream to control: Learning behaviors by latent imagination,” in ICLR, 2020.
[56]
↑
	D. Hafner, T. P. Lillicrap, M. Norouzi, and J. Ba, “Mastering atari with discrete world models,” in ICLR, 2021.
[57]
↑
	D. Hafner, J. Pasukonis, J. Ba, and T. P. Lillicrap, “Mastering diverse domains through world models,” CoRR, vol. abs/2301.04104, 2023.
[58]
↑
	P. Wu, A. Escontrela, D. Hafner, P. Abbeel, and K. Goldberg, “Daydreamer: World models for physical robot learning,” in CoRL, vol. 205. PMLR, 2022, pp. 2226–2240.
[59]
↑
	V. Micheli, E. Alonso, and F. Fleuret, “Transformers are sample-efficient world models,” in ICLR, 2023.
[60]
↑
	J. Robine, M. Höftmann, T. Uelwer, and S. Harmeling, “Transformer-based world models are happy with 100k interactions,” in ICLR, 2023.
[61]
↑
	K. Nottingham, P. Ammanabrolu, A. Suhr, Y. Choi, H. Hajishirzi, S. Singh, and R. Fox, “Do embodied agents dream of pixelated sheep: Embodied decision making using language guided world modelling,” in ICML, vol. 202. PMLR, 2023, pp. 26 311–26 325.
[62]
↑
	L. Guan, K. Valmeekam, S. Sreedharan, and S. Kambhampati, “Leveraging pre-trained large language models to construct and utilize world models for model-based task planning,” in NeurIPS, 2023.
[63]
↑
	B. Liu, Y. Jiang, X. Zhang, Q. Liu, S. Zhang, J. Biswas, and P. Stone, “LLM+P: empowering large language models with optimal planning proficiency,” CoRR, vol. abs/2304.11477, 2023.
[64]
↑
	S. Hao, Y. Gu, H. Ma, J. J. Hong, Z. Wang, D. Z. Wang, and Z. Hu, “Reasoning with language model is planning with world model,” in EMNLP, 2023, pp. 8154–8173.
[65]
↑
	M. Hu, Y. Mu, X. Yu, M. Ding, S. Wu, W. Shao, Q. Chen, B. Wang, Y. Qiao, and P. Luo, “Tree-planner: Efficient close-loop task planning with large language models,” in ICLR, 2024.
[66]
↑
	Z. Zhao, W. S. Lee, and D. Hsu, “Large language models as commonsense knowledge for large-scale task planning,” in NeurIPS, 2023.
[67]
↑
	OpenAI. (2024) Video generation models as world simulators. [Online]. Available: https://openai.com/index/video-generation-models-as-world-simulators/
[68]
↑
	Z. Zhu, X. Wang, W. Zhao, C. Min, N. Deng, M. Dou, Y. Wang, B. Shi, K. Wang, C. Zhang, Y. You, Z. Zhang, D. Zhao, L. Xiao, J. Zhao, J. Lu, and G. Huang, “Is sora a world simulator? A comprehensive survey on general world models and beyond,” CoRR, vol. abs/2405.03520, 2024.
[69]
↑
	J. Bruce, M. D. Dennis, A. Edwards, J. Parker-Holder, Y. Shi, E. Hughes, M. Lai, A. Mavalankar, R. Steigerwald, C. Apps, Y. Aytar, S. Bechtle, F. M. P. Behbahani, S. C. Y. Chan, N. Heess, L. Gonzalez, S. Osindero, S. Ozair, S. E. Reed, J. Zhang, K. Zolna, J. Clune, N. de Freitas, S. Singh, and T. Rocktäschel, “Genie: Generative interactive environments,” in ICML, 2024.
[70]
↑
	H. Zhen, X. Qiu, P. Chen, J. Yang, X. Yan, Y. Du, Y. Hong, and C. Gan, “3d-vla: A 3d vision-language-action generative world model,” in ICML, 2024.
[71]
↑
	S. Yang, Y. Du, S. K. S. Ghasemipour, J. Tompson, L. P. Kaelbling, D. Schuurmans, and P. Abbeel, “Learning interactive real-world simulators,” in ICLR, 2024.
[72]
↑
	J. Xiang, T. Tao, Y. Gu, T. Shu, Z. Wang, Z. Yang, and Z. Hu, “Language models meet world models: Embodied experiences enhance language models,” in NeurIPS, 2023.
[73]
↑
	T. Kojima, S. S. Gu, M. Reid, Y. Matsuo, and Y. Iwasawa, “Large language models are zero-shot reasoners,” in NeurIPS, 2022.
[74]
↑
	J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. H. Chi, Q. V. Le, and D. Zhou, “Chain-of-thought prompting elicits reasoning in large language models,” in NeurIPS, 2022.
[75]
↑
	G. Lu, Z. Wang, C. Liu, J. Lu, and Y. Tang, “Thinkbot: Embodied instruction following with thought chain reasoning,” CoRR, vol. abs/2312.07062, 2023.
[76]
↑
	S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. R. Narasimhan, and Y. Cao, “React: Synergizing reasoning and acting in language models,” in ICLR, 2023.
[77]
↑
	Z. Wang, A. Liu, H. Lin, J. Li, X. Ma, and Y. Liang, “RAT: retrieval augmented thoughts elicit context-aware reasoning in long-horizon generation,” CoRR, vol. abs/2403.05313, 2024.
[78]
↑
	M. Zawalski, W. Chen, K. Pertsch, O. Mees, C. Finn, and S. Levine, “Robotic control via embodied chain-of-thought reasoning,” CoRR, vol. abs/2407.08693, 2024.
[79]
↑
	E. Jang, A. Irpan, M. Khansari, D. Kappler, F. Ebert, C. Lynch, S. Levine, and C. Finn, “BC-Z: zero-shot task generalization with robotic imitation learning,” in CoRL, vol. 164. PMLR, 2021, pp. 991–1002.
[80]
↑
	C. Lynch and P. Sermanet, “Language conditioned imitation learning over unstructured data,” in RSS, 2021.
[81]
↑
	O. Mees, L. Hermann, and W. Burgard, “What matters in language conditioned robotic imitation learning over unstructured data,” IEEE Robotics Autom. Lett., vol. 7, no. 4, pp. 11 205–11 212, 2022.
[82]
↑
	O. Mees, J. Borja-Diaz, and W. Burgard, “Grounding language with visual affordances over unstructured data,” in ICRA. IEEE, 2023, pp. 11 576–11 582.
[83]
↑
	Y. Du, S. Yang, B. Dai, H. Dai, O. Nachum, J. Tenenbaum, D. Schuurmans, and P. Abbeel, “Learning universal policies via text-guided video generation,” in NeurIPS, 2023.
[84]
↑
	P. Sharma, B. Sundaralingam, V. Blukis, C. Paxton, T. Hermans, A. Torralba, J. Andreas, and D. Fox, “Correcting robot plans with natural language feedback,” in RSS, 2022.
[85]
↑
	C. Lynch, A. Wahid, J. Tompson, T. Ding, J. Betker, R. Baruch, T. Armstrong, and P. Florence, “Interactive language: Talking to robots in real time,” CoRR, vol. abs/2210.06407, 2022.
[86]
↑
	P. Guhur, S. Chen, R. G. Pinel, M. Tapaswi, I. Laptev, and C. Schmid, “Instruction-driven history-aware policies for robotic manipulations,” in CoRL, vol. 205. PMLR, 2022, pp. 175–187.
[87]
↑
	M. Shridhar, L. Manuelli, and D. Fox, “Perceiver-actor: A multi-task transformer for robotic manipulation,” in CoRL, vol. 205. PMLR, 2022, pp. 785–799.
[88]
↑
	T. Gervet, Z. Xian, N. Gkanatsios, and K. Fragkiadaki, “Act3d: 3d feature field transformers for multi-task robotic manipulation,” in CoRL, vol. 229. PMLR, 2023, pp. 3949–3965.
[89]
↑
	A. Goyal, J. Xu, Y. Guo, V. Blukis, Y. Chao, and D. Fox, “RVT: robotic view transformer for 3d object manipulation,” in CoRL, vol. 229. PMLR, 2023, pp. 694–710.
[90]
↑
	A. Goyal, V. Blukis, J. Xu, Y. Guo, Y. Chao, and D. Fox, “RVT-2: learning precise manipulation from few demonstrations,” CoRR, vol. abs/2406.08545, 2024.
[91]
↑
	W. Yuan, J. Duan, V. Blukis, W. Pumacay, R. Krishna, A. Murali, A. Mousavian, and D. Fox, “Robopoint: A vision-language model for spatial affordance prediction for robotics,” CoRR, vol. abs/2406.10721, 2024.
[92]
↑
	K. Bousmalis, G. Vezzani, D. Rao, C. Devin, A. X. Lee, M. Bauzá, T. Davchev, Y. Zhou, A. Gupta, A. Raju, A. Laurens, C. Fantacci, V. Dalibard, M. Zambelli, M. F. Martins, R. Pevceviciute, M. Blokzijl, M. Denil, N. Batchelor, T. Lampe, E. Parisotto, K. Zolna, S. E. Reed, S. G. Colmenarejo, J. Scholz, A. Abdolmaleki, O. Groth, J. Regli, O. Sushkov, T. Rothörl, J. E. Chen, Y. Aytar, D. Barker, J. Ortiz, M. A. Riedmiller, J. T. Springenberg, R. Hadsell, F. Nori, and N. Heess, “Robocat: A self-improving foundation agent for robotic manipulation,” CoRR, vol. abs/2306.11706, 2023.
[93]
↑
	A. Stone, T. Xiao, Y. Lu, K. Gopalakrishnan, K. Lee, Q. Vuong, P. Wohlhart, B. Zitkovich, F. Xia, C. Finn, and K. Hausman, “Open-world object manipulation using pre-trained vision-language models,” CoRR, vol. abs/2303.00905, 2023.
[94]
↑
	A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog, J. Hsu, J. Ibarz, B. Ichter, A. Irpan, T. Jackson, S. Jesmonth, N. J. Joshi, R. Julian, D. Kalashnikov, Y. Kuang, I. Leal, K. Lee, S. Levine, Y. Lu, U. Malla, D. Manjunath, I. Mordatch, O. Nachum, C. Parada, J. Peralta, E. Perez, K. Pertsch, J. Quiambao, K. Rao, M. S. Ryoo, G. Salazar, P. R. Sanketi, K. Sayed, J. Singh, S. Sontakke, A. Stone, C. Tan, H. T. Tran, V. Vanhoucke, S. Vega, Q. Vuong, F. Xia, T. Xiao, P. Xu, S. Xu, T. Yu, and B. Zitkovich, “RT-1: robotics transformer for real-world control at scale,” in RSS, 2023.
[95]
↑
	Y. Chebotar, Q. Vuong, A. Irpan, K. Hausman, F. Xia, Y. Lu, A. Kumar, T. Yu, A. Herzog, K. Pertsch, K. Gopalakrishnan, J. Ibarz, O. Nachum, S. Sontakke, G. Salazar, H. T. Tran, J. Peralta, C. Tan, D. Manjunath, J. Singh, B. Zitkovich, T. Jackson, K. Rao, C. Finn, and S. Levine, “Q-transformer: Scalable offline reinforcement learning via autoregressive q-functions,” CoRR, vol. abs/2309.10150, 2023.
[96]
↑
	J. Gu, S. Kirmani, P. Wohlhart, Y. Lu, M. G. Arenas, K. Rao, W. Yu, C. Fu, K. Gopalakrishnan, Z. Xu, P. Sundaresan, P. Xu, H. Su, K. Hausman, C. Finn, Q. Vuong, and T. Xiao, “Rt-trajectory: Robotic task generalization via hindsight trajectory sketches,” CoRR, vol. abs/2311.01977, 2023.
[97]
↑
	T. Z. Zhao, V. Kumar, S. Levine, and C. Finn, “Learning fine-grained bimanual manipulation with low-cost hardware,” in RSS, 2023.
[98]
↑
	H. Bharadhwaj, J. Vakil, M. Sharma, A. Gupta, S. Tulsiani, and V. Kumar, “Roboagent: Generalization and efficiency in robot manipulation via semantic augmentations and action chunking,” in ICRA. IEEE, 2024, pp. 4788–4795.
[99]
↑
	X. Li, M. Liu, H. Zhang, C. Yu, J. Xu, H. Wu, C. Cheang, Y. Jing, W. Zhang, H. Liu, H. Li, and T. Kong, “Vision-language foundation models as effective robot imitators,” CoRR, vol. abs/2311.01378, 2023.
[100]
↑
	F. Liu, F. Yan, L. Zheng, C. Feng, Y. Huang, and L. Ma, “Robouniview: Visual-language model with unified view representation for robotic manipulaiton,” CoRR, vol. abs/2406.18977, 2024.
[101]
↑
	Y. Yue, Y. Wang, B. Kang, Y. Han, S. Wang, S. Song, J. Feng, and G. Huang, “Deer-vla: Dynamic inference of multimodal large language models for efficient robot execution,” in NeurIPS, 2024.
[102]
↑
	S. Huang, Z. Jiang, H. Dong, Y. Qiao, P. Gao, and H. Li, “Instruct2act: Mapping multi-modality instructions to robotic actions with large language model,” CoRR, vol. abs/2305.11176, 2023.
[103]
↑
	W. Huang, C. Wang, R. Zhang, Y. Li, J. Wu, and L. Fei-Fei, “Voxposer: Composable 3d value maps for robotic manipulation with language models,” CoRR, vol. abs/2307.05973, 2023.
[104]
↑
	C. Chi, S. Feng, Y. Du, Z. Xu, E. Cousineau, B. Burchfiel, and S. Song, “Diffusion policy: Visuomotor policy learning via action diffusion,” in RSS, 2023.
[105]
↑
	Y. Ze, G. Zhang, K. Zhang, C. Hu, M. Wang, and H. Xu, “3d diffusion policy: Generalizable visuomotor policy learning via simple 3d representations,” in RSS, 2024.
[106]
↑
	H. Ha, P. Florence, and S. Song, “Scaling up and distilling down: Language-guided robot skill acquisition,” in CoRL, vol. 229. PMLR, 2023, pp. 3766–3777.
[107]
↑
	D. Ghosh, H. R. Walke, K. Pertsch, K. Black, O. Mees, S. Dasari, J. Hejna, T. Kreiman, C. Xu, J. Luo, Y. L. Tan, L. Y. Chen, Q. Vuong, T. Xiao, P. R. Sanketi, D. Sadigh, C. Finn, and S. Levine, “Octo: An open-source generalist robot policy,” in RSS, 2024.
[108]
↑
	T. Ke, N. Gkanatsios, and K. Fragkiadaki, “3d diffuser actor: Policy diffusion with 3d scene representations,” CoRR, vol. abs/2402.10885, 2024.
[109]
↑
	M. Reuss, Ö. E. Yagmurlu, F. Wenzel, and R. Lioutikov, “Multimodal diffusion transformer: Learning versatile behavior from multimodal goals,” CoRR, vol. abs/2407.05996, 2024.
[110]
↑
	S. Liu, L. Wu, B. Li, H. Tan, H. Chen, Z. Wang, K. Xu, H. Su, and J. Zhu, “Rdt-1b: a diffusion foundation model for bimanual manipulation,” arXiv preprint arXiv:2410.07864, 2024.
[111]
↑
	S. Belkhale, T. Ding, T. Xiao, P. Sermanet, Q. Vuong, J. Tompson, Y. Chebotar, D. Dwibedi, and D. Sadigh, “RT-H: action hierarchies using language,” CoRR, vol. abs/2403.01823, 2024.
[112]
↑
	O. X. Collaboration, A. Padalkar, A. Pooley, A. Jain, A. Bewley, A. Herzog, A. Irpan, A. Khazatsky, A. Raj, A. Singh, A. Brohan, A. Raffin, A. Wahid, B. Burgess-Limerick, B. Kim, B. Schölkopf, B. Ichter, C. Lu, C. Xu, C. Finn, C. Xu, C. Chi, C. Huang, C. Chan, C. Pan, C. Fu, C. Devin, D. Driess, D. Pathak, D. Shah, D. Büchler, D. Kalashnikov, D. Sadigh, E. Johns, F. Ceola, F. Xia, F. Stulp, G. Zhou, G. S. Sukhatme, G. Salhotra, G. Yan, G. Schiavi, G. Kahn, H. Su, H. Fang, H. Shi, H. B. Amor, H. I. Christensen, H. Furuta, H. Walke, H. Fang, I. Mordatch, I. Radosavovic, and et al., “Open x-embodiment: Robotic learning datasets and RT-X models,” CoRR, vol. abs/2310.08864, 2023.
[113]
↑
	M. J. Kim, C. Finn, and P. Liang, “Fine-tuning vision-language-action models: Optimizing speed and success,” CoRR, vol. abs/2502.19645, 2025.
[114]
↑
	R. Zheng, Y. Liang, S. Huang, J. Gao, H. D. III, A. Kolobov, F. Huang, and J. Yang, “Tracevla: Visual trace prompting enhances spatial-temporal awareness for generalist robotic policies,” in ICLR. OpenReview.net, 2025.
[115]
↑
	K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, S. Jakubczak, T. Jones, L. Ke, S. Levine, A. Li-Bell, M. Mothukuri, S. Nair, K. Pertsch, L. X. Shi, J. Tanner, Q. Vuong, A. Walling, H. Wang, and U. Zhilinsky, “
𝜋
0
: A vision-language-action flow model for general robot control,” CoRR, vol. abs/2410.24164, 2024.
[116]
↑
	J. Liu, M. Liu, Z. Wang, P. An, X. Li, K. Zhou, S. Yang, R. Zhang, Y. Guo, and S. Zhang, “Robomamba: Efficient vision-language-action model for robotic reasoning and manipulation,” in NeurIPS, 2024.
[117]
↑
	D. Qu, H. Song, Q. Chen, Y. Yao, X. Ye, Y. Ding, Z. Wang, J. Gu, B. Zhao, D. Wang, and X. Li, “Spatialvla: Exploring spatial representations for visual-language-action model,” CoRR, vol. abs/2501.15830, 2025.
[118]
↑
	J. Wen, Y. Zhu, J. Li, M. Zhu, K. Wu, Z. Xu, N. Liu, R. Cheng, C. Shen, Y. Peng, F. Feng, and J. Tang, “Tinyvla: Towards fast, data-efficient vision-language-action models for robotic manipulation,” CoRR, vol. abs/2409.12514, 2024.
[119]
↑
	Q. Li, Y. Liang, Z. Wang, L. Luo, X. Chen, M. Liao, F. Wei, Y. Deng, S. Xu, Y. Zhang, X. Wang, B. Liu, J. Fu, J. Bao, D. Chen, Y. Shi, J. Yang, and B. Guo, “Cogact: A foundational vision-language-action model for synergizing cognition and action in robotic manipulation,” CoRR, vol. abs/2411.19650, 2024.
[120]
↑
	J. Liu, H. Chen, P. An, Z. Liu, R. Zhang, C. Gu, X. Li, Z. Guo, S. Chen, M. Liu, C. Hou, M. Zhao, K. alex Zhou, P. Heng, and S. Zhang, “Hybridvla: Collaborative diffusion and autoregression in a unified vision-language-action model,” CoRR, vol. abs/2503.10631, 2025.
[121]
↑
	S. Ye, J. Jang, B. Jeon, S. J. Joo, J. Yang, B. Peng, A. Mandlekar, R. Tan, Y. Chao, B. Y. Lin, L. Liden, K. Lee, J. Gao, L. Zettlemoyer, D. Fox, and M. Seo, “Latent action pretraining from videos,” in ICLR. OpenReview.net, 2025.
[122]
↑
	J. Cen, C. Yu, H. Yuan, Y. Jiang, S. Huang, J. Guo, X. Li, Y. Song, H. Luo, F. Wang, D. Zhao, and H. Chen, “Worldvla: Towards autoregressive action world model,” CoRR, vol. abs/2506.21539, 2025.
[123]
↑
	Y. Wang, X. Li, W. Wang, J. Zhang, Y. Li, Y. Chen, X. Wang, and Z. Zhang, “Unified vision-language-action model,” CoRR, vol. abs/2506.19850, 2025.
[124]
↑
	P. Esser, R. Rombach, and B. Ommer, “Taming transformers for high-resolution image synthesis,” in CVPR. Computer Vision Foundation / IEEE, 2021, pp. 12 873–12 883.
[125]
↑
	M. Tan and Q. V. Le, “Efficientnet: Rethinking model scaling for convolutional neural networks,” in ICML, vol. 97. PMLR, 2019, pp. 6105–6114.
[126]
↑
	Y. Jiang, A. Gupta, Z. Zhang, G. Wang, Y. Dou, Y. Chen, L. Fei-Fei, A. Anandkumar, Y. Zhu, and L. Fan, “VIMA: general robot manipulation with multimodal prompts,” CoRR, vol. abs/2210.03094, 2022.
[127]
↑
	R. Liu, W. Wang, and Y. Yang, “Volumetric environment representation for vision-language navigation,” in CVPR. IEEE, 2024, pp. 16 317–16 328.
[128]
↑
	J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” in NeurIPS, 2020.
[129]
↑
	W. Peebles and S. Xie, “Scalable diffusion models with transformers,” in ICCV. IEEE, 2023, pp. 4172–4182.
[130]
↑
	Y. Ma and I. King, “3d-moe: A mixture-of-experts multi-modal LLM for 3d vision and pose diffusion via rectified flow,” CoRR, vol. abs/2501.16698, 2025.
[131]
↑
	M. Vecerík, C. Doersch, Y. Yang, T. Davchev, Y. Aytar, G. Zhou, R. Hadsell, L. Agapito, and J. Scholz, “Robotap: Tracking arbitrary points for few-shot visual imitation,” in ICRA. IEEE, 2024, pp. 5397–5403.
[132]
↑
	S. Nasiriany, F. Xia, W. Yu, T. Xiao, J. Liang, I. Dasgupta, A. Xie, D. Driess, A. Wahid, Z. Xu, Q. Vuong, T. Zhang, T. E. Lee, K. Lee, P. Xu, S. Kirmani, Y. Zhu, A. Zeng, K. Hausman, N. Heess, C. Finn, S. Levine, and B. Ichter, “PIVOT: iterative visual prompting elicits actionable knowledge for vlms,” in ICML, 2024.
[133]
↑
	K. Tian, Y. Jiang, Z. Yuan, B. Peng, and L. Wang, “Visual autoregressive modeling: Scalable image generation via next-scale prediction,” in NeurIPS, 2024.
[134]
↑
	X. Zhuang, Y. Xie, Y. Deng, L. Liang, J. Ru, Y. Yin, and Y. Zou, “VARGPT: unified understanding and generation in a visual autoregressive multimodal large language model,” CoRR, vol. abs/2501.12327, 2025.
[135]
↑
	C. Qu, S. Dai, X. Wei, H. Cai, S. Wang, D. Yin, J. Xu, and J. Wen, “Tool learning with large language models: a survey,” Frontiers Comput. Sci., vol. 19, no. 8, p. 198343, 2025.
[136]
↑
	Y. Mu, Q. Zhang, M. Hu, W. Wang, M. Ding, J. Jin, B. Wang, J. Dai, Y. Qiao, and P. Luo, “Embodiedgpt: Vision-language pre-training via embodied chain of thought,” CoRR, vol. abs/2305.15021, 2023.
[137]
↑
	J. Huang, S. Yong, X. Ma, X. Linghu, P. Li, Y. Wang, Q. Li, S. Zhu, B. Jia, and S. Huang, “An embodied generalist agent in 3d world,” in ICML, 2024.
[138]
↑
	Z. Qi, R. Dong, S. Zhang, H. Geng, C. Han, Z. Ge, L. Yi, and K. Ma, “Shapellm: Universal 3d object understanding for embodied interaction,” in ECCV (43), ser. Lecture Notes in Computer Science, vol. 15101. Springer, 2024, pp. 214–238.
[139]
↑
	W. Huang, P. Abbeel, D. Pathak, and I. Mordatch, “Language models as zero-shot planners: Extracting actionable knowledge for embodied agents,” in ICML, vol. 162. PMLR, 2022, pp. 9118–9147.
[140]
↑
	P. Sharma, A. Torralba, and J. Andreas, “Skill induction and planning with latent language,” in ACL (1), 2022, pp. 1713–1726.
[141]
↑
	C. H. Song, B. M. Sadler, J. Wu, W. Chao, C. Washington, and Y. Su, “Llm-planner: Few-shot grounded planning for embodied agents with large language models,” in ICCV. IEEE, 2023, pp. 2986–2997.
[142]
↑
	S. Li, X. Puig, C. Paxton, Y. Du, C. Wang, L. Fan, T. Chen, D. Huang, E. Akyürek, A. Anandkumar, J. Andreas, I. Mordatch, A. Torralba, and Y. Zhu, “Pre-trained language models for interactive decision-making,” in NeurIPS, 2022.
[143]
↑
	A. Zeng, M. Attarian, B. Ichter, K. M. Choromanski, A. Wong, S. Welker, F. Tombari, A. Purohit, M. S. Ryoo, V. Sindhwani, J. Lee, V. Vanhoucke, and P. Florence, “Socratic models: Composing zero-shot multimodal reasoning with language,” in ICLR, 2023.
[144]
↑
	I. Singh, V. Blukis, A. Mousavian, A. Goyal, D. Xu, J. Tremblay, D. Fox, J. Thomason, and A. Garg, “Progprompt: Generating situated robot task plans using large language models,” in ICRA. IEEE, 2023, pp. 11 523–11 530.
[145]
↑
	S. Vemprala, R. Bonatti, A. Bucker, and A. Kapoor, “Chatgpt for robotics: Design principles and model abilities,” CoRR, vol. abs/2306.17582, 2023.
[146]
↑
	J. Liang, W. Huang, F. Xia, P. Xu, K. Hausman, B. Ichter, P. Florence, and A. Zeng, “Code as policies: Language model programs for embodied control,” in ICRA. IEEE, 2023, pp. 9493–9500.
[147]
↑
	Z. Wang, S. Cai, A. Liu, X. Ma, and Y. Liang, “Describe, explain, plan and select: Interactive planning with large language models enables open-world multi-task agents,” CoRR, vol. abs/2302.01560, 2023.
[148]
↑
	Q. Gu, A. Kuwajerwala, S. Morin, K. M. Jatavallabhula, B. Sen, A. Agarwal, C. Rivera, W. Paul, K. Ellis, R. Chellappa, C. Gan, C. M. de Melo, J. B. Tenenbaum, A. Torralba, F. Shkurti, and L. Paull, “Conceptgraphs: Open-vocabulary 3d scene graphs for perception and planning,” CoRR, vol. abs/2309.16650, 2023.
[149]
↑
	F. Lin, Y. Hu, P. Sheng, C. Wen, J. You, and Y. Gao, “Data scaling laws in imitation learning for robotic manipulation,” in ICLR. OpenReview.net, 2025.
[150]
↑
	Y. Hong, Z. Zheng, P. Chen, Y. Wang, J. Li, and C. Gan, “Multiply: A multisensory object-centric embodied large language model in 3d world,” in CVPR. IEEE, 2024, pp. 26 396–26 406.
[151]
↑
	S. S. Raman, V. Cohen, E. Rosen, I. Idrees, D. Paulius, and S. Tellex, “Planning with large language models via corrective re-prompting,” CoRR, vol. abs/2211.09935, 2022.
[152]
↑
	P. Zhi, Z. Zhang, M. Han, Z. Zhang, Z. Li, Z. Jiao, B. Jia, and S. Huang, “Closed-loop open-vocabulary mobile manipulation with GPT-4V,” CoRR, vol. abs/2404.10220, 2024.
[153]
↑
	H. Fang, H. Fang, Z. Tang, J. Liu, C. Wang, J. Wang, H. Zhu, and C. Lu, “RH20T: A comprehensive robotic dataset for learning diverse skills in one-shot,” in ICRA. IEEE, 2024, pp. 653–660.
[154]
↑
	A. Khazatsky, K. Pertsch, S. Nair, A. Balakrishna, S. Dasari, S. Karamcheti, S. Nasiriany, M. K. Srirama, L. Y. Chen, K. Ellis, P. D. Fagan, J. Hejna, M. Itkina, M. Lepert, Y. J. Ma, P. T. Miller, J. Wu, S. Belkhale, S. Dass, H. Ha, A. Jain, A. Lee, Y. Lee, M. Memmel, S. Park, I. Radosavovic, K. Wang, A. Zhan, K. Black, C. Chi, K. B. Hatch, S. Lin, J. Lu, J. Mercat, A. Rehman, P. R. Sanketi, A. Sharma, C. Simpson, Q. Vuong, H. R. Walke, B. Wulfe, T. Xiao, J. H. Yang, A. Yavary, T. Z. Zhao, C. Agia, R. Baijal, M. G. Castro, D. Chen, Q. Chen, T. Chung, J. Drake, E. P. Foster, and et al., “DROID: A large-scale in-the-wild robot manipulation dataset,” CoRR, vol. abs/2403.12945, 2024.
[155]
↑
	P. Sharma, L. Mohan, L. Pinto, and A. Gupta, “Multiple interactions made easy (MIME): large scale demonstrations data for imitation,” in CoRL, vol. 87. PMLR, 2018, pp. 906–915.
[156]
↑
	A. Mandlekar, Y. Zhu, A. Garg, J. Booher, M. Spero, A. Tung, J. Gao, J. Emmons, A. Gupta, E. Orbay, S. Savarese, and L. Fei-Fei, “ROBOTURK: A crowdsourcing platform for robotic skill learning through imitation,” in CoRL, vol. 87. PMLR, 2018, pp. 879–893.
[157]
↑
	S. Dasari, F. Ebert, S. Tian, S. Nair, B. Bucher, K. Schmeckpeper, S. Singh, S. Levine, and C. Finn, “Robonet: Large-scale multi-robot learning,” in CoRL, vol. 100. PMLR, 2019, pp. 885–897.
[158]
↑
	D. Kalashnikov, J. Varley, Y. Chebotar, B. Swanson, R. Jonschkowski, C. Finn, S. Levine, and K. Hausman, “Mt-opt: Continuous multi-task robotic reinforcement learning at scale,” CoRR, vol. abs/2104.08212, 2021.
[159]
↑
	V. Kumar, R. M. Shah, G. Zhou, V. Moens, V. Caggiano, A. Gupta, and A. Rajeswaran, “Robohive: A unified framework for robot learning,” in NeurIPS, 2023.
[160]
↑
	F. Ebert, Y. Yang, K. Schmeckpeper, B. Bucher, G. Georgakis, K. Daniilidis, C. Finn, and S. Levine, “Bridge data: Boosting generalization of robotic skills with cross-domain datasets,” in RSS, 2022.
[161]
↑
	S. Srivastava, C. Li, M. Lingelbach, R. Martín-Martín, F. Xia, K. E. Vainio, Z. Lian, C. Gokmen, S. Buch, C. K. Liu, S. Savarese, H. Gweon, J. Wu, and L. Fei-Fei, “BEHAVIOR: benchmark for everyday household activities in virtual, interactive, and ecological environments,” in CoRL, vol. 164. PMLR, 2021, pp. 477–490.
[162]
↑
	C. Li, F. Xia, R. Martín-Martín, M. Lingelbach, S. Srivastava, B. Shen, K. E. Vainio, C. Gokmen, G. Dharan, T. Jain, A. Kurenkov, C. K. Liu, H. Gweon, J. Wu, L. Fei-Fei, and S. Savarese, “igibson 2.0: Object-centric simulation for robot learning of everyday household tasks,” in CoRL, vol. 164. PMLR, 2021, pp. 455–465.
[163]
↑
	F. Xia, A. R. Zamir, Z. He, A. Sax, J. Malik, and S. Savarese, “Gibson env: Real-world perception for embodied agents,” in CVPR. Computer Vision Foundation / IEEE Computer Society, 2018, pp. 9068–9079.
[164]
↑
	F. Xia, W. B. Shen, C. Li, P. Kasimbeg, M. Tchapmi, A. Toshev, R. Martín-Martín, and S. Savarese, “Interactive gibson benchmark: A benchmark for interactive navigation in cluttered environments,” IEEE Robotics Autom. Lett., vol. 5, no. 2, pp. 713–720, 2020.
[165]
↑
	B. Shen, F. Xia, C. Li, R. Martín-Martín, L. Fan, G. Wang, C. Pérez-D’Arpino, S. Buch, S. Srivastava, L. Tchapmi, M. Tchapmi, K. Vainio, J. Wong, L. Fei-Fei, and S. Savarese, “igibson 1.0: A simulation environment for interactive tasks in large realistic scenes,” in IROS. IEEE, 2021, pp. 7520–7527.
[166]
↑
	C. Li, R. Zhang, J. Wong, C. Gokmen, S. Srivastava, R. Martín-Martín, C. Wang, G. Levine, W. Ai, B. Martinez, H. Yin, M. Lingelbach, M. Hwang, A. Hiranaka, S. Garlanka, A. Aydin, S. Lee, J. Sun, M. Anvari, M. Sharma, D. Bansal, S. Hunter, K. Kim, A. Lou, C. R. Matthews, I. Villa-Renteria, J. H. Tang, C. Tang, F. Xia, Y. Li, S. Savarese, H. Gweon, C. K. Liu, J. Wu, and L. Fei-Fei, “BEHAVIOR-1K: A human-centered, embodied AI benchmark with 1, 000 everyday activities and realistic simulation,” CoRR, vol. abs/2403.09227, 2024.
[167]
↑
	F. Xiang, Y. Qin, K. Mo, Y. Xia, H. Zhu, F. Liu, M. Liu, H. Jiang, Y. Yuan, H. Wang, L. Yi, A. X. Chang, L. J. Guibas, and H. Su, “SAPIEN: A simulated part-based interactive environment,” in CVPR. Computer Vision Foundation / IEEE, 2020, pp. 11 094–11 104.
[168]
↑
	X. Li, K. Hsu, J. Gu, K. Pertsch, O. Mees, H. R. Walke, C. Fu, I. Lunawat, I. Sieh, S. Kirmani, S. Levine, J. Wu, C. Finn, H. Su, Q. Vuong, and T. Xiao, “Evaluating real-world robot manipulation policies in simulation,” CoRR, vol. abs/2405.05941, 2024.
[169]
↑
	E. Kolve, R. Mottaghi, D. Gordon, Y. Zhu, A. Gupta, and A. Farhadi, “AI2-THOR: an interactive 3d environment for visual AI,” CoRR, vol. abs/1712.05474, 2017.
[170]
↑
	K. Ehsani, W. Han, A. Herrasti, E. VanderBilt, L. Weihs, E. Kolve, A. Kembhavi, and R. Mottaghi, “Manipulathor: A framework for visual object manipulation,” in CVPR. Computer Vision Foundation / IEEE, 2021, pp. 4497–4506.
[171]
↑
	L. Weihs, M. Deitke, A. Kembhavi, and R. Mottaghi, “Visual room rearrangement,” in CVPR. Computer Vision Foundation / IEEE, 2021, pp. 5922–5931.
[172]
↑
	K. Mirakhor, S. Ghosh, D. Das, and B. Bhowmick, “Task planning for visual room rearrangement under partial observability,” in ICLR, 2024.
[173]
↑
	X. Puig, K. Ra, M. Boben, J. Li, T. Wang, S. Fidler, and A. Torralba, “Virtualhome: Simulating household activities via programs,” in 2018 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2018, Salt Lake City, UT, USA, June 18-22, 2018. Computer Vision Foundation / IEEE Computer Society, 2018, pp. 8494–8502.
[174]
↑
	C. Gan, J. Schwartz, S. Alter, D. Mrowca, M. Schrimpf, J. Traer, J. D. Freitas, J. Kubilius, A. Bhandwaldar, N. Haber, M. Sano, K. Kim, E. Wang, M. Lingelbach, A. Curtis, K. T. Feigelis, D. Bear, D. Gutfreund, D. D. Cox, A. Torralba, J. J. DiCarlo, J. Tenenbaum, J. H. McDermott, and D. Yamins, “Threedworld: A platform for interactive multi-modal physical simulation,” in NeurIPS Datasets and Benchmarks, 2021.
[175]
↑
	S. James, Z. Ma, D. R. Arrojo, and A. J. Davison, “Rlbench: The robot learning benchmark & learning environment,” IEEE Robotics Autom. Lett., vol. 5, no. 2, pp. 3019–3026, 2020.
[176]
↑
	T. Yu, D. Quillen, Z. He, R. Julian, K. Hausman, C. Finn, and S. Levine, “Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning,” in CoRL, vol. 100. PMLR, 2019, pp. 1094–1100.
[177]
↑
	O. Mees, L. Hermann, E. Rosete-Beas, and W. Burgard, “CALVIN: A benchmark for language-conditioned policy learning for long-horizon robot manipulation tasks,” IEEE Robotics Autom. Lett., vol. 7, no. 3, pp. 7327–7334, 2022.
[178]
↑
	A. Gupta, V. Kumar, C. Lynch, S. Levine, and K. Hausman, “Relay policy learning: Solving long-horizon tasks via imitation and reinforcement learning,” in CoRL, vol. 100. PMLR, 2019, pp. 1025–1037.
[179]
↑
	M. Savva, J. Malik, D. Parikh, D. Batra, A. Kadian, O. Maksymets, Y. Zhao, E. Wijmans, B. Jain, J. Straub, J. Liu, and V. Koltun, “Habitat: A platform for embodied AI research,” in ICCV. IEEE, 2019, pp. 9338–9346.
[180]
↑
	A. Szot, A. Clegg, E. Undersander, E. Wijmans, Y. Zhao, J. M. Turner, N. Maestre, M. Mukadam, D. S. Chaplot, O. Maksymets, A. Gokaslan, V. Vondrus, S. Dharur, F. Meier, W. Galuba, A. X. Chang, Z. Kira, V. Koltun, J. Malik, M. Savva, and D. Batra, “Habitat 2.0: Training home assistants to rearrange their habitat,” in NeurIPS, 2021, pp. 251–266.
[181]
↑
	D. Batra, A. X. Chang, S. Chernova, A. J. Davison, J. Deng, V. Koltun, S. Levine, J. Malik, I. Mordatch, R. Mottaghi, M. Savva, and H. Su, “Rearrangement: A challenge for embodied AI,” CoRR, vol. abs/2011.01975, 2020.
[182]
↑
	S. Yenamandra, A. Ramachandran, K. Yadav, A. S. Wang, M. Khanna, T. Gervet, T. Yang, V. Jain, A. Clegg, J. M. Turner, Z. Kira, M. Savva, A. X. Chang, D. S. Chaplot, D. Batra, R. Mottaghi, Y. Bisk, and C. Paxton, “Homerobot: Open-vocabulary mobile manipulation,” in CoRL, vol. 229. PMLR, 2023, pp. 1975–2011.
[183]
↑
	M. Shridhar, J. Thomason, D. Gordon, Y. Bisk, W. Han, R. Mottaghi, L. Zettlemoyer, and D. Fox, “ALFRED: A benchmark for interpreting grounded instructions for everyday tasks,” in CVPR. Computer Vision Foundation / IEEE, 2020, pp. 10 737–10 746.
[184]
↑
	Y. Tassa, Y. Doron, A. Muldal, T. Erez, Y. Li, D. de Las Casas, D. Budden, A. Abdolmaleki, J. Merel, A. Lefrancq, T. P. Lillicrap, and M. A. Riedmiller, “Deepmind control suite,” CoRR, vol. abs/1801.00690, 2018.
[185]
↑
	G. Brockman, V. Cheung, L. Pettersson, J. Schneider, J. Schulman, J. Tang, and W. Zaremba, “Openai gym,” CoRR, vol. abs/1606.01540, 2016.
[186]
↑
	G. Authors, “Genesis: A universal and generative physics engine for robotics and beyond,” December 2024. [Online]. Available: https://github.com/Genesis-Embodied-AI/Genesis
[187]
↑
	Y. Wang, Z. Xian, F. Chen, T. Wang, Y. Wang, K. Fragkiadaki, Z. Erickson, D. Held, and C. Gan, “Robogen: Towards unleashing infinite data for automated robot learning via generative simulation,” in ICML, 2024.
[188]
↑
	M. Ahn, D. Dwibedi, C. Finn, M. G. Arenas, K. Gopalakrishnan, K. Hausman, B. Ichter, A. Irpan, N. J. Joshi, R. Julian, S. Kirmani, I. Leal, T. E. Lee, S. Levine, Y. Lu, S. Maddineni, K. Rao, D. Sadigh, P. Sanketi, P. Sermanet, Q. Vuong, S. Welker, F. Xia, T. Xiao, P. Xu, S. Xu, and Z. Xu, “Autort: Embodied foundation models for large scale orchestration of robotic agents,” CoRR, vol. abs/2401.12963, 2024.
[189]
↑
	T. Xiao, H. Chan, P. Sermanet, A. Wahid, A. Brohan, K. Hausman, S. Levine, and J. Tompson, “Robotic skill acquisition via instruction augmentation with vision-language models,” in RSS, 2023.
[190]
↑
	C. Chi, Z. Xu, C. Pan, E. Cousineau, B. Burchfiel, S. Feng, R. Tedrake, and S. Song, “Universal manipulation interface: In-the-wild robot teaching without in-the-wild robots,” in RSS, 2024.
[191]
↑
	G. Moon, S. Saito, W. Xu, R. Joshi, J. Buffalini, H. Bellan, N. Rosen, J. Richardson, M. Mize, P. de Bree, T. Simon, B. Peng, S. Garg, K. McPhail, and T. Shiratori, “A dataset of relighted 3d interacting hands,” in NeurIPS, 2023.
[192]
↑
	Y. Chen, Y. Ge, Y. Ge, M. Ding, B. Li, R. Wang, R. Xu, Y. Shan, and X. Liu, “Egoplan-bench: Benchmarking egocentric embodied planning with multimodal large language models,” CoRR, vol. abs/2312.06722, 2023.
[193]
↑
	K. Valmeekam, M. Marquez, A. O. Hernandez, S. Sreedharan, and S. Kambhampati, “Planbench: An extensible benchmark for evaluating large language models on planning and reasoning about change,” in NeurIPS, 2023.
[194]
↑
	K. Valmeekam, M. Marquez, S. Sreedharan, and S. Kambhampati, “On the planning abilities of large language models - A critical investigation,” in NeurIPS, 2023.
[195]
↑
	J. Choi, Y. Yoon, H. Ong, J. Kim, and M. Jang, “Lota-bench: Benchmarking language-oriented task planners for embodied agents,” CoRR, vol. abs/2402.08178, 2024.
[196]
↑
	M. Li, S. Zhao, Q. Wang, K. Wang, Y. Zhou, S. Srivastava, C. Gokmen, T. Lee, L. E. Li, R. Zhang, W. Liu, P. Liang, L. Fei-Fei, J. Mao, and J. Wu, “Embodied agent interface: Benchmarking llms for embodied decision making,” in NeurIPS, 2024.
[197]
↑
	A. Das, S. Datta, G. Gkioxari, S. Lee, D. Parikh, and D. Batra, “Embodied question answering,” in CVPR. Computer Vision Foundation / IEEE Computer Society, 2018, pp. 1–10.
[198]
↑
	D. Gordon, A. Kembhavi, M. Rastegari, J. Redmon, D. Fox, and A. Farhadi, “IQA: visual question answering in interactive environments,” in CVPR. Computer Vision Foundation / IEEE Computer Society, 2018, pp. 4089–4098.
[199]
↑
	L. Yu, X. Chen, G. Gkioxari, M. Bansal, T. L. Berg, and D. Batra, “Multi-target embodied question answering,” in CVPR. Computer Vision Foundation / IEEE, 2019, pp. 6309–6318.
[200]
↑
	E. Wijmans, S. Datta, O. Maksymets, A. Das, G. Gkioxari, S. Lee, I. Essa, D. Parikh, and D. Batra, “Embodied question answering in photorealistic environments with point cloud perception,” in CVPR. Computer Vision Foundation / IEEE, 2019, pp. 6659–6668.
[201]
↑
	C. Fan, “Egovqa - an egocentric video question answering benchmark dataset,” in ICCV Workshops. IEEE, 2019, pp. 4359–4366.
[202]
↑
	B. Jia, T. Lei, S. Zhu, and S. Huang, “Egotaskqa: Understanding human tasks in egocentric videos,” in NeurIPS, 2022.
[203]
↑
	M. M. Islam, A. Gladstone, R. Islam, and T. Iqbal, “EQA-MX: embodied question answering using multimodal expression,” in ICLR, 2024.
[204]
↑
	A. Majumdar, A. Ajay, X. Zhang, P. Putta, S. Yenamandra, M. Henaff, S. Silwal, P. Mcvay, O. Maksymets, S. Arnaud, K. Yadav, Q. Li, B. Newman, M. Sharma, V. Berges, S. Zhang, P. Agrawal, Y. Bisk, D. Batra, M. Kalakrishnan, F. Meier, C. Paxton, S. Sax, and A. Rajeswaran, “OpenEQA: Embodied Question Answering in the Era of Foundation Models,” in CVPR, 2024.
[205]
↑
	R. Girdhar, A. El-Nouby, Z. Liu, M. Singh, K. V. Alwala, A. Joulin, and I. Misra, “Imagebind one embedding space to bind them all,” in CVPR. IEEE, 2023, pp. 15 180–15 190.
[206]
↑
	B. Zhu, B. Lin, M. Ning, Y. Yan, J. Cui, H. Wang, Y. Pang, W. Jiang, J. Zhang, Z. Li, C. Zhang, Z. Li, W. Liu, and L. Yuan, “Languagebind: Extending video-language pretraining to n-modality by language-based semantic alignment,” in ICLR, 2024.
[207]
↑
	I. Güzey, B. Evans, S. Chintala, and L. Pinto, “Dexterity from touch: Self-supervised pre-training of tactile representations with robotic play,” in CoRL, vol. 229. PMLR, 2023, pp. 3142–3166.
[208]
↑
	R. Zhang, A. Saran, B. Liu, Y. Zhu, S. Guo, S. Niekum, D. Ballard, and M. Hayhoe, “Human gaze assisted artificial intelligence: A review,” in IJCAI-20, 7 2020, pp. 4951–4958, survey track.
[209]
↑
	R. Zhang, Z. Liu, L. Zhang, J. A. Whritner, K. S. Muller, M. M. Hayhoe, and D. H. Ballard, “AGIL: learning attention from human for visuomotor tasks,” in ECCV (11), vol. 11215. Springer, 2018, pp. 692–707.
[210]
↑
	A. Saran, R. Zhang, E. S. Short, and S. Niekum, “Efficiently guiding imitation learning agents with human gaze,” in AAMAS. ACM, 2021, pp. 1109–1117.
[211]
↑
	S. Guo, R. Zhang, B. Liu, Y. Zhu, D. H. Ballard, M. M. Hayhoe, and P. Stone, “Machine versus human attention in deep reinforcement learning tasks,” in NeurIPS, 2021, pp. 25 370–25 385.
[212]
↑
	H. M. Le, T. N. Do, and S. J. Phee, “A survey on actuators-driven surgical robots,” Sensors and Actuators A-physical, vol. 247, pp. 323–354, 2016.
[213]
↑
	M. S. Laursen, J. S. Pedersen, S. A. Just, T. R. Savarimuthu, B. Blomholt, J. K. H. Andersen, and P. J. Vinholt, “Factors facilitating the acceptance of diagnostic robots in healthcare: A survey,” in ICHI. IEEE, 2022, pp. 442–448.
[214]
↑
	K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in CVPR. IEEE Computer Society, 2016, pp. 770–778.
[215]
↑
	A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An image is worth 16x16 words: Transformers for image recognition at scale,” in ICLR, 2021.
[216]
↑
	A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W. Lo, P. Dollár, and R. B. Girshick, “Segment anything,” CoRR, vol. abs/2304.02643, 2023.
[217]
↑
	S. Hochreiter and J. Schmidhuber, “Long short-term memory,” Neural Comput., vol. 9, no. 8, pp. 1735–1780, 1997.
[218]
↑
	K. Cho, B. van Merrienboer, Ç. Gülçehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio, “Learning phrase representations using RNN encoder-decoder for statistical machine translation,” in EMNLP. ACL, 2014, pp. 1724–1734.
[219]
↑
	D. Silver, A. Huang, C. J. Maddison, A. Guez, L. Sifre, G. van den Driessche, J. Schrittwieser, I. Antonoglou, V. Panneershelvam, M. Lanctot, S. Dieleman, D. Grewe, J. Nham, N. Kalchbrenner, I. Sutskever, T. P. Lillicrap, M. Leach, K. Kavukcuoglu, T. Graepel, and D. Hassabis, “Mastering the game of go with deep neural networks and tree search,” Nat., vol. 529, no. 7587, pp. 484–489, 2016.
[220]
↑
	J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” CoRR, vol. abs/1707.06347, 2017.
[221]
↑
	OpenAI, M. Andrychowicz, B. Baker, M. Chociej, R. Józefowicz, B. McGrew, J. Pachocki, A. Petron, M. Plappert, G. Powell, A. Ray, J. Schneider, S. Sidor, J. Tobin, P. Welinder, L. Weng, and W. Zaremba, “Learning dexterous in-hand manipulation,” CoRR, vol. abs/1808.00177, 2018.
[222]
↑
	R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn, “Direct preference optimization: Your language model is secretly a reward model,” in NeurIPS, 2023.
[223]
↑
	X. Chen, H. Fang, T. Lin, R. Vedantam, S. Gupta, P. Dollár, and C. L. Zitnick, “Microsoft COCO captions: Data collection and evaluation server,” CoRR, vol. abs/1504.00325, 2015.
[224]
↑
	S. Antol, A. Agrawal, J. Lu, M. Mitchell, D. Batra, C. L. Zitnick, and D. Parikh, “VQA: visual question answering,” in ICCV. IEEE Computer Society, 2015, pp. 2425–2433.
[225]
↑
	L. Yu, P. Poirson, S. Yang, A. C. Berg, and T. L. Berg, “Modeling context in referring expressions,” in ECCV (2), ser. Lecture Notes in Computer Science, vol. 9906. Springer, 2016, pp. 69–85.
[226]
↑
	O. Vinyals, A. Toshev, S. Bengio, and D. Erhan, “Show and tell: A neural image caption generator,” in CVPR. IEEE Computer Society, 2015, pp. 3156–3164.
[227]
↑
	A. Radford, K. Narasimhan, T. Salimans, and I. Sutskever, “Improving language understanding by generative pre-training,” OpenAI blog, 2018.
[228]
↑
	J. Lu, D. Batra, D. Parikh, and S. Lee, “Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks,” in NeurIPS, 2019, pp. 13–23.
[229]
↑
	J. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds, R. Ring, E. Rutherford, S. Cabi, T. Han, Z. Gong, S. Samangooei, M. Monteiro, J. L. Menick, S. Borgeaud, A. Brock, A. Nematzadeh, S. Sharifzadeh, M. Binkowski, R. Barreira, O. Vinyals, A. Zisserman, and K. Simonyan, “Flamingo: a visual language model for few-shot learning,” in NeurIPS, 2022.
[230]
↑
	A. Khan, A. Sohail, U. Zahoora, and A. S. Qureshi, “A survey of the recent architectures of deep convolutional neural networks,” Artif. Intell. Rev., vol. 53, no. 8, pp. 5455–5516, 2020.
[231]
↑
	S. H. Khan, M. Naseer, M. Hayat, S. W. Zamir, F. S. Khan, and M. Shah, “Transformers in vision: A survey,” ACM Comput. Surv., vol. 54, no. 10s, pp. 200:1–200:41, 2022.
[232]
↑
	Y. LeCun, B. E. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. E. Hubbard, and L. D. Jackel, “Backpropagation applied to handwritten zip code recognition,” Neural Comput., vol. 1, no. 4, pp. 541–551, 1989.
[233]
↑
	K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” in ICLR, 2015.
[234]
↑
	C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. E. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich, “Going deeper with convolutions,” in CVPR. IEEE Computer Society, 2015, pp. 1–9.
[235]
↑
	C. Szegedy, S. Ioffe, V. Vanhoucke, and A. A. Alemi, “Inception-v4, inception-resnet and the impact of residual connections on learning,” in AAAI. AAAI Press, 2017, pp. 4278–4284.
[236]
↑
	S. Xie, R. B. Girshick, P. Dollár, Z. Tu, and K. He, “Aggregated residual transformations for deep neural networks,” in CVPR. IEEE Computer Society, 2017, pp. 5987–5995.
[237]
↑
	J. Hu, L. Shen, and G. Sun, “Squeeze-and-excitation networks,” in CVPR. Computer Vision Foundation / IEEE Computer Society, 2018, pp. 7132–7141.
[238]
↑
	R. B. Girshick, J. Donahue, T. Darrell, and J. Malik, “Rich feature hierarchies for accurate object detection and semantic segmentation,” in CVPR. IEEE Computer Society, 2014, pp. 580–587.
[239]
↑
	R. B. Girshick, “Fast R-CNN,” in ICCV. IEEE Computer Society, 2015, pp. 1440–1448.
[240]
↑
	S. Ren, K. He, R. B. Girshick, and J. Sun, “Faster R-CNN: towards real-time object detection with region proposal networks,” in NIPS, 2015, pp. 91–99.
[241]
↑
	K. He, G. Gkioxari, P. Dollár, and R. B. Girshick, “Mask R-CNN,” in ICCV. IEEE Computer Society, 2017, pp. 2980–2988.
[242]
↑
	J. Redmon, S. K. Divvala, R. B. Girshick, and A. Farhadi, “You only look once: Unified, real-time object detection,” in CVPR. IEEE Computer Society, 2016, pp. 779–788.
[243]
↑
	T. Lin, P. Dollár, R. B. Girshick, K. He, B. Hariharan, and S. J. Belongie, “Feature pyramid networks for object detection,” in CVPR. IEEE Computer Society, 2017, pp. 936–944.
[244]
↑
	T. Lin, P. Goyal, R. B. Girshick, K. He, and P. Dollár, “Focal loss for dense object detection,” in ICCV. IEEE Computer Society, 2017, pp. 2999–3007.
[245]
↑
	P. Anderson, X. He, C. Buehler, D. Teney, M. Johnson, S. Gould, and L. Zhang, “Bottom-up and top-down attention for image captioning and visual question answering,” in CVPR. Computer Vision Foundation / IEEE Computer Society, 2018, pp. 6077–6086.
[246]
↑
	J. Long, E. Shelhamer, and T. Darrell, “Fully convolutional networks for semantic segmentation,” in CVPR. IEEE Computer Society, 2015, pp. 3431–3440.
[247]
↑
	V. Badrinarayanan, A. Kendall, and R. Cipolla, “Segnet: A deep convolutional encoder-decoder architecture for image segmentation,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 39, no. 12, pp. 2481–2495, 2017.
[248]
↑
	O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” in MICCAI (3), ser. Lecture Notes in Computer Science, vol. 9351. Springer, 2015, pp. 234–241.
[249]
↑
	N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and S. Zagoruyko, “End-to-end object detection with transformers,” in ECCV (1), ser. Lecture Notes in Computer Science, vol. 12346. Springer, 2020, pp. 213–229.
[250]
↑
	R. Strudel, R. G. Pinel, I. Laptev, and C. Schmid, “Segmenter: Transformer for semantic segmentation,” in ICCV. IEEE, 2021, pp. 7242–7252.
[251]
↑
	A. Ioannidou, E. Chatzilari, S. Nikolopoulos, and I. Kompatsiaris, “Deep learning advances in computer vision with 3d data: A survey,” ACM Comput. Surv., vol. 50, no. 2, pp. 20:1–20:38, 2017.
[252]
↑
	E. Ahmed, A. Saint, A. E. R. Shabayek, K. Cherenkova, R. Das, G. Gusev, D. Aouada, and B. E. Ottersten, “Deep learning advances on different 3d data representations: A survey,” CoRR, vol. abs/1808.01462, 2018.
[253]
↑
	Y. Guo, H. Wang, Q. Hu, H. Liu, L. Liu, and M. Bennamoun, “Deep learning for 3d point clouds: A survey,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 43, no. 12, pp. 4338–4364, 2021.
[254]
↑
	C. R. Qi, H. Su, M. Nießner, A. Dai, M. Yan, and L. J. Guibas, “Volumetric and multi-view cnns for object classification on 3d data,” in CVPR. IEEE Computer Society, 2016, pp. 5648–5656.
[255]
↑
	Y. Feng, Y. Feng, H. You, X. Zhao, and Y. Gao, “Meshnet: Mesh neural network for 3d shape representation,” in AAAI. AAAI Press, 2019, pp. 8279–8286.
[256]
↑
	D. W. Otter, J. R. Medina, and J. K. Kalita, “A survey of the usages of deep learning for natural language processing,” IEEE Trans. Neural Networks Learn. Syst., vol. 32, no. 2, pp. 604–624, 2021.
[257]
↑
	P. Liu, W. Yuan, J. Fu, Z. Jiang, H. Hayashi, and G. Neubig, “Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing,” ACM Comput. Surv., vol. 55, no. 9, pp. 195:1–195:35, 2023.
[258]
↑
	Y. Bengio, R. Ducharme, and P. Vincent, “A neural probabilistic language model,” in NIPS. MIT Press, 2000, pp. 932–938.
[259]
↑
	T. Mikolov, I. Sutskever, K. Chen, G. S. Corrado, and J. Dean, “Distributed representations of words and phrases and their compositionality,” in NIPS, 2013, pp. 3111–3119.
[260]
↑
	T. Mikolov, K. Chen, G. Corrado, and J. Dean, “Efficient estimation of word representations in vector space,” in ICLR (Workshop Poster), 2013.
[261]
↑
	J. Pennington, R. Socher, and C. D. Manning, “Glove: Global vectors for word representation,” in EMNLP. ACL, 2014, pp. 1532–1543.
[262]
↑
	J. L. Elman, “Finding structure in time,” Cogn. Sci., vol. 14, no. 2, pp. 179–211, 1990.
[263]
↑
	D. Bahdanau, K. Cho, and Y. Bengio, “Neural machine translation by jointly learning to align and translate,” in ICLR, 2015.
[264]
↑
	Z. Huang, W. Xu, and K. Yu, “Bidirectional LSTM-CRF models for sequence tagging,” CoRR, vol. abs/1508.01991, 2015.
[265]
↑
	M. E. Peters, M. Neumann, M. Iyyer, M. Gardner, C. Clark, K. Lee, and L. Zettlemoyer, “Deep contextualized word representations,” in NAACL-HLT. Association for Computational Linguistics, 2018, pp. 2227–2237.
[266]
↑
	J. Howard and S. Ruder, “Universal language model fine-tuning for text classification,” in ACL (1). Association for Computational Linguistics, 2018, pp. 328–339.
[267]
↑
	Y. Kim, “Convolutional neural networks for sentence classification,” in EMNLP. ACL, 2014, pp. 1746–1751.
[268]
↑
	X. Zhang, J. J. Zhao, and Y. LeCun, “Character-level convolutional networks for text classification,” in NIPS, 2015, pp. 649–657.
[269]
↑
	X. Ma and E. H. Hovy, “End-to-end sequence labeling via bi-directional lstm-cnns-crf,” in ACL (1). The Association for Computer Linguistics, 2016.
[270]
↑
	A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever et al., “Language models are unsupervised multitask learners,” OpenAI blog, vol. 1, no. 8, p. 9, 2019.
[271]
↑
	T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell et al., “Language models are few-shot learners,” arXiv preprint arXiv:2005.14165, 2020.
[272]
↑
	OpenAI, “GPT-4 technical report,” CoRR, vol. abs/2303.08774, 2023.
[273]
↑
	Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov, “Roberta: A robustly optimized BERT pretraining approach,” CoRR, vol. abs/1907.11692, 2019.
[274]
↑
	Z. Lan, M. Chen, S. Goodman, K. Gimpel, P. Sharma, and R. Soricut, “ALBERT: A lite BERT for self-supervised learning of language representations,” in ICLR, 2020.
[275]
↑
	K. Clark, M. Luong, Q. V. Le, and C. D. Manning, “ELECTRA: pre-training text encoders as discriminators rather than generators,” in ICLR, 2020.
[276]
↑
	Z. Yang, Z. Dai, Y. Yang, J. G. Carbonell, R. Salakhutdinov, and Q. V. Le, “Xlnet: Generalized autoregressive pretraining for language understanding,” in NeurIPS, 2019, pp. 5754–5764.
[277]
↑
	S. Zhang, S. Roller, N. Goyal, M. Artetxe, M. Chen, S. Chen, C. Dewan, M. T. Diab, X. Li, X. V. Lin, T. Mihaylov, M. Ott, S. Shleifer, K. Shuster, D. Simig, P. S. Koura, A. Sridhar, T. Wang, and L. Zettlemoyer, “OPT: open pre-trained transformer language models,” CoRR, vol. abs/2205.01068, 2022.
[278]
↑
	M. Lewis, Y. Liu, N. Goyal, M. Ghazvininejad, A. Mohamed, O. Levy, V. Stoyanov, and L. Zettlemoyer, “BART: denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension,” in ACL. Association for Computational Linguistics, 2020, pp. 7871–7880.
[279]
↑
	C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu, “Exploring the limits of transfer learning with a unified text-to-text transformer,” J. Mach. Learn. Res., vol. 21, pp. 140:1–140:67, 2020.
[280]
↑
	A. Chowdhery, S. Narang, J. Devlin, M. Bosma, G. Mishra, A. Roberts, P. Barham, H. W. Chung, C. Sutton, S. Gehrmann, P. Schuh, K. Shi, S. Tsvyashchenko, J. Maynez, A. Rao, P. Barnes, Y. Tay, N. Shazeer, V. Prabhakaran, E. Reif, N. Du, B. Hutchinson, R. Pope, J. Bradbury, J. Austin, M. Isard, G. Gur-Ari, P. Yin, T. Duke, A. Levskaya, S. Ghemawat, S. Dev, H. Michalewski, X. Garcia, V. Misra, K. Robinson, L. Fedus, D. Zhou, D. Ippolito, D. Luan, H. Lim, B. Zoph, A. Spiridonov, R. Sepassi, D. Dohan, S. Agrawal, M. Omernick, A. M. Dai, T. S. Pillai, M. Pellat, A. Lewkowycz, E. Moreira, R. Child, O. Polozov, K. Lee, Z. Zhou, X. Wang, B. Saeta, M. Diaz, O. Firat, M. Catasta, J. Wei, K. Meier-Hellstern, D. Eck, J. Dean, S. Petrov, and N. Fiedel, “Palm: Scaling language modeling with pathways,” J. Mach. Learn. Res., vol. 24, pp. 240:1–240:113, 2023.
[281]
↑
	R. Anil, A. M. Dai, O. Firat, M. Johnson, D. Lepikhin, A. Passos, S. Shakeri, E. Taropa, P. Bailey, Z. Chen, E. Chu, J. H. Clark, L. E. Shafey, Y. Huang, K. Meier-Hellstern, G. Mishra, E. Moreira, M. Omernick, K. Robinson, S. Ruder, Y. Tay, K. Xiao, Y. Xu, Y. Zhang, G. H. Ábrego, J. Ahn, J. Austin, P. Barham, J. A. Botha, J. Bradbury, S. Brahma, K. Brooks, M. Catasta, Y. Cheng, C. Cherry, C. A. Choquette-Choo, A. Chowdhery, C. Crepy, S. Dave, M. Dehghani, S. Dev, J. Devlin, M. Díaz, N. Du, E. Dyer, V. Feinberg, F. Feng, V. Fienber, M. Freitag, X. Garcia, S. Gehrmann, L. Gonzalez, and et al., “Palm 2 technical report,” CoRR, vol. abs/2305.10403, 2023.
[282]
↑
	H. Touvron, T. Lavril, G. Izacard, X. Martinet, M. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, A. Rodriguez, A. Joulin, E. Grave, and G. Lample, “Llama: Open and efficient foundation language models,” CoRR, vol. abs/2302.13971, 2023.
[283]
↑
	H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, D. Bikel, L. Blecher, C. Canton-Ferrer, M. Chen, G. Cucurull, D. Esiobu, J. Fernandes, J. Fu, W. Fu, B. Fuller, C. Gao, V. Goswami, N. Goyal, A. Hartshorn, S. Hosseini, R. Hou, H. Inan, M. Kardas, V. Kerkez, M. Khabsa, I. Kloumann, A. Korenev, P. S. Koura, M. Lachaux, T. Lavril, J. Lee, D. Liskovich, Y. Lu, Y. Mao, X. Martinet, T. Mihaylov, P. Mishra, I. Molybog, Y. Nie, A. Poulton, J. Reizenstein, R. Rungta, K. Saladi, A. Schelten, R. Silva, E. M. Smith, R. Subramanian, X. E. Tan, B. Tang, R. Taylor, A. Williams, J. X. Kuan, P. Xu, Z. Yan, I. Zarov, Y. Zhang, A. Fan, M. Kambadur, S. Narang, A. Rodriguez, R. Stojnic, S. Edunov, and T. Scialom, “Llama 2: Open foundation and fine-tuned chat models,” CoRR, vol. abs/2307.09288, 2023.
[284]
↑
	Baidu. (2023) Introducing ernie 3.5: Baidu’s knowledge-enhanced foundation model takes a giant leap forward. [Online]. Available: http://research.baidu.com/Blog/index-view?id=185
[285]
↑
	L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. L. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, J. Schulman, J. Hilton, F. Kelton, L. Miller, M. Simens, A. Askell, P. Welinder, P. F. Christiano, J. Leike, and R. Lowe, “Training language models to follow instructions with human feedback,” in NeurIPS, 2022.
[286]
↑
	J. Wei, M. Bosma, V. Y. Zhao, K. Guu, A. W. Yu, B. Lester, N. Du, A. M. Dai, and Q. V. Le, “Finetuned language models are zero-shot learners,” in ICLR, 2022.
[287]
↑
	H. W. Chung, L. Hou, S. Longpre, B. Zoph, Y. Tay, W. Fedus, E. Li, X. Wang, M. Dehghani, S. Brahma, A. Webson, S. S. Gu, Z. Dai, M. Suzgun, X. Chen, A. Chowdhery, S. Narang, G. Mishra, A. Yu, V. Y. Zhao, Y. Huang, A. M. Dai, H. Yu, S. Petrov, E. H. Chi, J. Dean, J. Devlin, A. Roberts, D. Zhou, Q. V. Le, and J. Wei, “Scaling instruction-finetuned language models,” CoRR, vol. abs/2210.11416, 2022.
[288]
↑
	R. Taori, I. Gulrajani, T. Zhang, Y. Dubois, X. Li, C. Guestrin, P. Liang, and T. B. Hashimoto, “Stanford alpaca: An instruction-following llama model,” https://github.com/tatsu-lab/stanford_alpaca, 2023.
[289]
↑
	W.-L. Chiang, Z. Li, Z. Lin, Y. Sheng, Z. Wu, H. Zhang, L. Zheng, S. Zhuang, Y. Zhuang, J. E. Gonzalez, I. Stoica, and E. P. Xing, “Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality,” March 2023. [Online]. Available: https://lmsys.org/blog/2023-03-30-vicuna/
[290]
↑
	Y. Li, “Deep reinforcement learning: An overview,” CoRR, vol. abs/1701.07274, 2017.
[291]
↑
	K. Arulkumaran, M. P. Deisenroth, M. Brundage, and A. A. Bharath, “A brief survey of deep reinforcement learning,” CoRR, vol. abs/1708.05866, 2017.
[292]
↑
	W. Li, H. Luo, Z. Lin, C. Zhang, Z. Lu, and D. Ye, “A survey on transformers in reinforcement learning,” CoRR, vol. abs/2301.03044, 2023.
[293]
↑
	H. van Hasselt, A. Guez, and D. Silver, “Deep reinforcement learning with double q-learning,” in AAAI. AAAI Press, 2016, pp. 2094–2100.
[294]
↑
	M. Andrychowicz, D. Crow, A. Ray, J. Schneider, R. Fong, P. Welinder, B. McGrew, J. Tobin, P. Abbeel, and W. Zaremba, “Hindsight experience replay,” in NIPS, 2017, pp. 5048–5058.
[295]
↑
	S. Fujimoto, D. Meger, and D. Precup, “Off-policy deep reinforcement learning without exploration,” in ICML, vol. 97. PMLR, 2019, pp. 2052–2062.
[296]
↑
	A. Kumar, J. Fu, M. Soh, G. Tucker, and S. Levine, “Stabilizing off-policy q-learning via bootstrapping error reduction,” in NeurIPS, 2019, pp. 11 761–11 771.
[297]
↑
	A. Kumar, A. Zhou, G. Tucker, and S. Levine, “Conservative q-learning for offline reinforcement learning,” in NeurIPS, 2020.
[298]
↑
	S. Levine and V. Koltun, “Guided policy search,” in ICML (3), ser. JMLR Workshop and Conference Proceedings, vol. 28. JMLR.org, 2013, pp. 1–9.
[299]
↑
	D. Silver, G. Lever, N. Heess, T. Degris, D. Wierstra, and M. A. Riedmiller, “Deterministic policy gradient algorithms,” in ICML, ser. JMLR Workshop and Conference Proceedings, vol. 32. JMLR.org, 2014, pp. 387–395.
[300]
↑
	T. P. Lillicrap, J. J. Hunt, A. Pritzel, N. Heess, T. Erez, Y. Tassa, D. Silver, and D. Wierstra, “Continuous control with deep reinforcement learning,” in ICLR (Poster), 2016.
[301]
↑
	V. Mnih, A. P. Badia, M. Mirza, A. Graves, T. P. Lillicrap, T. Harley, D. Silver, and K. Kavukcuoglu, “Asynchronous methods for deep reinforcement learning,” in ICML, ser. JMLR Workshop and Conference Proceedings, vol. 48. JMLR.org, 2016, pp. 1928–1937.
[302]
↑
	S. Gu, T. P. Lillicrap, I. Sutskever, and S. Levine, “Continuous deep q-learning with model-based acceleration,” in ICML, ser. JMLR Workshop and Conference Proceedings, vol. 48. JMLR.org, 2016, pp. 2829–2838.
[303]
↑
	T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, “Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor,” in ICML, vol. 80. PMLR, 2018, pp. 1856–1865.
[304]
↑
	J. Schulman, S. Levine, P. Abbeel, M. I. Jordan, and P. Moritz, “Trust region policy optimization,” in ICML, ser. JMLR Workshop and Conference Proceedings, vol. 37. JMLR.org, 2015, pp. 1889–1897.
[305]
↑
	J. Schulman, P. Moritz, S. Levine, M. I. Jordan, and P. Abbeel, “High-dimensional continuous control using generalized advantage estimation,” in ICLR (Poster), 2016.
[306]
↑
	J. Ho and S. Ermon, “Generative adversarial imitation learning,” in NIPS, 2016, pp. 4565–4573.
[307]
↑
	L. Pinto, J. Davidson, R. Sukthankar, and A. Gupta, “Robust adversarial reinforcement learning,” in ICML, vol. 70. PMLR, 2017, pp. 2817–2826.
[308]
↑
	P. F. Christiano, J. Leike, T. B. Brown, M. Martic, S. Legg, and D. Amodei, “Deep reinforcement learning from human preferences,” in NIPS, 2017, pp. 4299–4307.
[309]
↑
	A. S. Vezhnevets, S. Osindero, T. Schaul, N. Heess, M. Jaderberg, D. Silver, and K. Kavukcuoglu, “Feudal networks for hierarchical reinforcement learning,” in ICML, vol. 70. PMLR, 2017, pp. 3540–3549.
[310]
↑
	S. Levine, C. Finn, T. Darrell, and P. Abbeel, “End-to-end training of deep visuomotor policies,” J. Mach. Learn. Res., vol. 17, pp. 39:1–39:40, 2016.
[311]
↑
	D. Kalashnikov, A. Irpan, P. Pastor, J. Ibarz, A. Herzog, E. Jang, D. Quillen, E. Holly, M. Kalakrishnan, V. Vanhoucke, and S. Levine, “Qt-opt: Scalable deep reinforcement learning for vision-based robotic manipulation,” CoRR, vol. abs/1806.10293, 2018.
[312]
↑
	OpenAI, I. Akkaya, M. Andrychowicz, M. Chociej, M. Litwin, B. McGrew, A. Petron, A. Paino, M. Plappert, G. Powell, R. Ribas, J. Schneider, N. Tezak, J. Tworek, P. Welinder, L. Weng, Q. Yuan, W. Zaremba, and L. Zhang, “Solving rubik’s cube with a robot hand,” CoRR, vol. abs/1910.07113, 2019.
[313]
↑
	F. Scarselli, M. Gori, A. C. Tsoi, M. Hagenbuchner, and G. Monfardini, “The graph neural network model,” IEEE Trans. Neural Networks, vol. 20, no. 1, pp. 61–80, 2009.
[314]
↑
	Z. Wu, S. Pan, F. Chen, G. Long, C. Zhang, and P. S. Yu, “A comprehensive survey on graph neural networks,” IEEE Trans. Neural Networks Learn. Syst., vol. 32, no. 1, pp. 4–24, 2021.
[315]
↑
	J. Bruna, W. Zaremba, A. Szlam, and Y. LeCun, “Spectral networks and locally connected networks on graphs,” in ICLR, 2014.
[316]
↑
	M. Defferrard, X. Bresson, and P. Vandergheynst, “Convolutional neural networks on graphs with fast localized spectral filtering,” in NIPS, 2016, pp. 3837–3845.
[317]
↑
	T. N. Kipf and M. Welling, “Semi-supervised classification with graph convolutional networks,” in ICLR (Poster), 2017.
[318]
↑
	A. Micheli, “Neural network for graphs: A contextual constructive approach,” IEEE Trans. Neural Networks, vol. 20, no. 3, pp. 498–511, 2009.
[319]
↑
	J. Gilmer, S. S. Schoenholz, P. F. Riley, O. Vinyals, and G. E. Dahl, “Neural message passing for quantum chemistry,” in ICML, vol. 70. PMLR, 2017, pp. 1263–1272.
[320]
↑
	K. Xu, W. Hu, J. Leskovec, and S. Jegelka, “How powerful are graph neural networks?” in ICLR, 2019.
[321]
↑
	W. L. Hamilton, Z. Ying, and J. Leskovec, “Inductive representation learning on large graphs,” in NIPS, 2017, pp. 1024–1034.
[322]
↑
	P. Velickovic, G. Cucurull, A. Casanova, A. Romero, P. Liò, and Y. Bengio, “Graph attention networks,” in ICLR (Poster), 2018.
[323]
↑
	S. Cao, W. Lu, and Q. Xu, “Deep neural networks for learning graph representations,” in AAAI. AAAI Press, 2016, pp. 1145–1152.
[324]
↑
	T. N. Kipf and M. Welling, “Variational graph auto-encoders,” CoRR, vol. abs/1611.07308, 2016.
[325]
↑
	Y. Seo, M. Defferrard, P. Vandergheynst, and X. Bresson, “Structured sequence modeling with graph convolutional recurrent networks,” in ICONIP (1), ser. Lecture Notes in Computer Science, vol. 11301. Springer, 2018, pp. 362–373.
[326]
↑
	W. Du, H. Zhang, Y. Du, Q. Meng, W. Chen, N. Zheng, B. Shao, and T. Liu, “SE(3) equivariant graph neural networks with complete local frames,” in ICML, vol. 162. PMLR, 2022, pp. 5583–5608.
[327]
↑
	V. G. Satorras, E. Hoogeboom, and M. Welling, “E(n) equivariant graph neural networks,” in ICML, vol. 139. PMLR, 2021, pp. 9323–9332.
[328]
↑
	Y. Rong, Y. Bian, T. Xu, W. Xie, Y. Wei, W. Huang, and J. Huang, “Self-supervised graph transformer on large-scale molecular data,” in NeurIPS, 2020.
[329]
↑
	F. Fuchs, D. E. Worrall, V. Fischer, and M. Welling, “Se(3)-transformers: 3d roto-translation equivariant attention networks,” in NeurIPS, 2020.
[330]
↑
	R. Krishna, Y. Zhu, O. Groth, J. Johnson, K. Hata, J. Kravitz, S. Chen, Y. Kalantidis, L. Li, D. A. Shamma, M. S. Bernstein, and L. Fei-Fei, “Visual genome: Connecting language and vision using crowdsourced dense image annotations,” Int. J. Comput. Vis., vol. 123, no. 1, pp. 32–73, 2017.
[331]
↑
	L. Wu, Y. Chen, K. Shen, X. Guo, H. Gao, S. Li, J. Pei, and B. Long, “Graph neural networks for natural language processing: A survey,” Found. Trends Mach. Learn., vol. 16, no. 2, pp. 119–328, 2023.
[332]
↑
	D. Ghosal, N. Majumder, S. Poria, N. Chhaya, and A. F. Gelbukh, “Dialoguegcn: A graph convolutional neural network for emotion recognition in conversation,” in EMNLP/IJCNLP (1). Association for Computational Linguistics, 2019, pp. 154–164.
[333]
↑
	W. Zhong, J. Xu, D. Tang, Z. Xu, N. Duan, M. Zhou, J. Wang, and J. Yin, “Reasoning over semantic-level graph for fact checking,” in ACL. Association for Computational Linguistics, 2020, pp. 6170–6180.
[334]
↑
	L. Wang, Y. Li, Ö. Aslan, and O. Vinyals, “Wikigraphs: A wikipedia text - knowledge graph paired dataset,” CoRR, vol. abs/2107.09556, 2021.
[335]
↑
	J. Tang, J. Zhang, L. Yao, J. Li, L. Zhang, and Z. Su, “Arnetminer: extraction and mining of academic social networks,” in KDD. ACM, 2008, pp. 990–998.
[336]
↑
	H. Tan and M. Bansal, “LXMERT: learning cross-modality encoder representations from transformers,” in EMNLP/IJCNLP (1). Association for Computational Linguistics, 2019, pp. 5099–5110.
[337]
↑
	L. H. Li, M. Yatskar, D. Yin, C. Hsieh, and K. Chang, “Visualbert: A simple and performant baseline for vision and language,” CoRR, vol. abs/1908.03557, 2019.
[338]
↑
	W. Su, X. Zhu, Y. Cao, B. Li, L. Lu, F. Wei, and J. Dai, “VL-BERT: pre-training of generic visual-linguistic representations,” in ICLR, 2020.
[339]
↑
	Y. Chen, L. Li, L. Yu, A. E. Kholy, F. Ahmed, Z. Gan, Y. Cheng, and J. Liu, “UNITER: universal image-text representation learning,” in ECCV (30), ser. Lecture Notes in Computer Science, vol. 12375. Springer, 2020, pp. 104–120.
[340]
↑
	W. Kim, B. Son, and I. Kim, “Vilt: Vision-and-language transformer without convolution or region supervision,” in ICML, vol. 139. PMLR, 2021, pp. 5583–5594.
[341]
↑
	Z. Wang, J. Yu, A. W. Yu, Z. Dai, Y. Tsvetkov, and Y. Cao, “Simvlm: Simple visual language model pretraining with weak supervision,” in ICLR, 2022.
[342]
↑
	Z. Dai, H. Liu, Q. V. Le, and M. Tan, “Coatnet: Marrying convolution and attention for all data sizes,” in NeurIPS, 2021, pp. 3965–3977.
[343]
↑
	J. Wang, Z. Yang, X. Hu, L. Li, K. Lin, Z. Gan, Z. Liu, C. Liu, and L. Wang, “GIT: A generative image-to-text transformer for vision and language,” Trans. Mach. Learn. Res., vol. 2022, 2022.
[344]
↑
	L. Yuan, D. Chen, Y. Chen, N. Codella, X. Dai, J. Gao, H. Hu, X. Huang, B. Li, C. Li, C. Liu, M. Liu, Z. Liu, Y. Lu, Y. Shi, L. Wang, J. Wang, B. Xiao, Z. Xiao, J. Yang, M. Zeng, L. Zhou, and P. Zhang, “Florence: A new foundation model for computer vision,” CoRR, vol. abs/2111.11432, 2021.
[345]
↑
	W. Wang, H. Bao, L. Dong, J. Bjorck, Z. Peng, Q. Liu, K. Aggarwal, O. K. Mohammed, S. Singhal, S. Som, and F. Wei, “Image as a foreign language: BEIT pretraining for vision and vision-language tasks,” in CVPR. IEEE, 2023, pp. 19 175–19 186.
[346]
↑
	L. Yao, R. Huang, L. Hou, G. Lu, M. Niu, H. Xu, X. Liang, Z. Li, X. Jiang, and C. Xu, “FILIP: fine-grained interactive language-image pre-training,” in ICLR, 2022.
[347]
↑
	C. Jia, Y. Yang, Y. Xia, Y. Chen, Z. Parekh, H. Pham, Q. V. Le, Y. Sung, Z. Li, and T. Duerig, “Scaling up visual and vision-language representation learning with noisy text supervision,” in ICML, vol. 139. PMLR, 2021, pp. 4904–4916.
[348]
↑
	Q. Xie, M. Luong, E. H. Hovy, and Q. V. Le, “Self-training with noisy student improves imagenet classification,” in CVPR. Computer Vision Foundation / IEEE, 2020, pp. 10 684–10 695.
[349]
↑
	J. Li, R. R. Selvaraju, A. Gotmare, S. R. Joty, C. Xiong, and S. C. Hoi, “Align before fuse: Vision and language representation learning with momentum distillation,” in NeurIPS, 2021, pp. 9694–9705.
[350]
↑
	A. Singh, R. Hu, V. Goswami, G. Couairon, W. Galuba, M. Rohrbach, and D. Kiela, “FLAVA: A foundational language and vision alignment model,” in CVPR. IEEE, 2022, pp. 15 617–15 629.
[351]
↑
	Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, “Swin transformer: Hierarchical vision transformer using shifted windows,” in ICCV. IEEE, 2021, pp. 9992–10 002.
[352]
↑
	H. Wu, B. Xiao, N. Codella, M. Liu, X. Dai, L. Yuan, and L. Zhang, “Cvt: Introducing convolutions to vision transformers,” in ICCV. IEEE, 2021, pp. 22–31.
[353]
↑
	A. Brock, S. De, S. L. Smith, and K. Simonyan, “High-performance large-scale image recognition without normalization,” in ICML, vol. 139. PMLR, 2021, pp. 1059–1071.
[354]
↑
	J. Hoffmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, D. de Las Casas, L. A. Hendricks, J. Welbl, A. Clark, T. Hennigan, E. Noland, K. Millican, G. van den Driessche, B. Damoc, A. Guy, S. Osindero, K. Simonyan, E. Elsen, J. W. Rae, O. Vinyals, and L. Sifre, “Training compute-optimal large language models,” CoRR, vol. abs/2203.15556, 2022.
[355]
↑
	Y. Fang, W. Wang, B. Xie, Q. Sun, L. Wu, X. Wang, T. Huang, X. Wang, and Y. Cao, “EVA: exploring the limits of masked visual representation learning at scale,” in CVPR. IEEE, 2023, pp. 19 358–19 369.
[356]
↑
	X. Chen, X. Wang, S. Changpinyo, A. J. Piergiovanni, P. Padlewski, D. Salz, S. Goodman, A. Grycner, B. Mustafa, L. Beyer, A. Kolesnikov, J. Puigcerver, N. Ding, K. Rong, H. Akbari, G. Mishra, L. Xue, A. V. Thapliyal, J. Bradbury, and W. Kuo, “Pali: A jointly-scaled multilingual language-image model,” in ICLR, 2023.
[357]
↑
	L. Xue, N. Constant, A. Roberts, M. Kale, R. Al-Rfou, A. Siddhant, A. Barua, and C. Raffel, “mt5: A massively multilingual pre-trained text-to-text transformer,” in NAACL-HLT. Association for Computational Linguistics, 2021, pp. 483–498.
[358]
↑
	X. Chen, J. Djolonga, P. Padlewski, B. Mustafa, S. Changpinyo, J. Wu, C. R. Ruiz, S. Goodman, X. Wang, Y. Tay, S. Shakeri, M. Dehghani, D. Salz, M. Lucic, M. Tschannen, A. Nagrani, H. Hu, M. Joshi, B. Pang, C. Montgomery, P. Pietrzyk, M. Ritter, A. J. Piergiovanni, M. Minderer, F. Pavetic, A. Waters, G. Li, I. Alabdulmohsin, L. Beyer, J. Amelot, K. Lee, A. P. Steiner, Y. Li, D. Keysers, A. Arnab, Y. Xu, K. Rong, A. Kolesnikov, M. Seyedhosseini, A. Angelova, X. Zhai, N. Houlsby, and R. Soricut, “Pali-x: On scaling up a multilingual vision and language model,” CoRR, vol. abs/2305.18565, 2023.
[359]
↑
	M. Dehghani, J. Djolonga, B. Mustafa, P. Padlewski, J. Heek, J. Gilmer, A. Steiner, M. Caron, R. Geirhos, I. Alabdulmohsin, R. Jenatton, L. Beyer, M. Tschannen, A. Arnab, X. Wang, C. Riquelme, M. Minderer, J. Puigcerver, U. Evci, M. Kumar, S. van Steenkiste, G. F. Elsayed, A. Mahendran, F. Yu, A. Oliver, F. Huot, J. Bastings, M. P. Collier, A. A. Gritsenko, V. Birodkar, C. Vasconcelos, Y. Tay, T. Mensink, A. Kolesnikov, F. Pavetic, D. Tran, T. Kipf, M. Lucic, X. Zhai, D. Keysers, J. Harmsen, and N. Houlsby, “Scaling vision transformers to 22 billion parameters,” CoRR, vol. abs/2302.05442, 2023.
[360]
↑
	Y. Tay, M. Dehghani, V. Q. Tran, X. Garcia, J. Wei, X. Wang, H. W. Chung, D. Bahri, T. Schuster, H. S. Zheng, D. Zhou, N. Houlsby, and D. Metzler, “UL2: unifying language learning paradigms,” in ICLR, 2023.
[361]
↑
	R. Zhang, J. Han, A. Zhou, X. Hu, S. Yan, P. Lu, H. Li, P. Gao, and Y. Qiao, “Llama-adapter: Efficient fine-tuning of language models with zero-init attention,” CoRR, vol. abs/2303.16199, 2023.
[362]
↑
	P. Gao, J. Han, R. Zhang, Z. Lin, S. Geng, A. Zhou, W. Zhang, P. Lu, C. He, X. Yue, H. Li, and Y. Qiao, “Llama-adapter V2: parameter-efficient visual instruction model,” CoRR, vol. abs/2304.15010, 2023.
[363]
↑
	S. Huang, L. Dong, W. Wang, Y. Hao, S. Singhal, S. Ma, T. Lv, L. Cui, O. K. Mohammed, B. Patra, Q. Liu, K. Aggarwal, Z. Chi, J. Bjorck, V. Chaudhary, S. Som, X. Song, and F. Wei, “Language is not all you need: Aligning perception with language models,” CoRR, vol. abs/2302.14045, 2023.
[364]
↑
	Z. Peng, W. Wang, L. Dong, Y. Hao, S. Huang, S. Ma, and F. Wei, “Kosmos-2: Grounding multimodal large language models to the world,” CoRR, vol. abs/2306.14824, 2023.
[365]
↑
	H. Wang, S. Ma, S. Huang, L. Dong, W. Wang, Z. Peng, Y. Wu, P. Bajaj, S. Singhal, A. Benhaim, B. Patra, Z. Liu, V. Chaudhary, X. Song, and F. Wei, “Foundation transformers,” CoRR, vol. abs/2210.06423, 2022.
[366]
↑
	W. Dai, J. Li, D. Li, A. M. H. Tiong, J. Zhao, W. Wang, B. Li, P. Fung, and S. C. H. Hoi, “Instructblip: Towards general-purpose vision-language models with instruction tuning,” CoRR, vol. abs/2305.06500, 2023.
[367]
↑
	D. Zhu, J. Chen, X. Shen, X. Li, and M. Elhoseiny, “Minigpt-4: Enhancing vision-language understanding with advanced large language models,” CoRR, vol. abs/2304.10592, 2023.
[368]
↑
	H. Zhang, X. Li, and L. Bing, “Video-llama: An instruction-tuned audio-visual language model for video understanding,” CoRR, vol. abs/2306.02858, 2023.
[369]
↑
	Y. Su, T. Lan, H. Li, J. Xu, Y. Wang, and D. Cai, “Pandagpt: One model to instruction-follow them all,” CoRR, vol. abs/2305.16355, 2023.
[370]
↑
	K. Li, Y. He, Y. Wang, Y. Li, W. Wang, P. Luo, Y. Wang, L. Wang, and Y. Qiao, “Videochat: Chat-centric video understanding,” CoRR, vol. abs/2305.06355, 2023.
[371]
↑
	S. contributors. (2023) Stablelm: Stability ai language models. [Online]. Available: https://github.com/stability-AI/stableLM
[372]
↑
	L. Zhao, E. Yu, Z. Ge, J. Yang, H. Wei, H. Zhou, J. Sun, Y. Peng, R. Dong, C. Han, and X. Zhang, “Chatspot: Bootstrapping multimodal llms via precise referring instruction tuning,” CoRR, vol. abs/2307.09474, 2023.
[373]
↑
	Q. Ye, H. Xu, G. Xu, J. Ye, M. Yan, Y. Zhou, J. Wang, A. Hu, P. Shi, Y. Shi, C. Li, Y. Xu, H. Chen, J. Tian, Q. Qi, J. Zhang, and F. Huang, “mplug-owl: Modularization empowers large language models with multimodality,” CoRR, vol. abs/2304.14178, 2023.
[374]
↑
	Q. Ye, H. Xu, J. Ye, M. Yan, A. Hu, H. Liu, Q. Qian, J. Zhang, F. Huang, and J. Zhou, “mplug-owl2: Revolutionizing multi-modal large language model with modality collaboration,” CoRR, vol. abs/2311.04257, 2023.
[375]
↑
	C. Xu, D. Guo, N. Duan, and J. J. McAuley, “Baize: An open-source chat model with parameter-efficient tuning on self-chat data,” in EMNLP. Association for Computational Linguistics, 2023, pp. 6268–6278.
[376]
↑
	C. Wu, S. Yin, W. Qi, X. Wang, Z. Tang, and N. Duan, “Visual chatgpt: Talking, drawing and editing with visual foundation models,” CoRR, vol. abs/2303.04671, 2023.
[377]
↑
	F. Chen, M. Han, H. Zhao, Q. Zhang, J. Shi, S. Xu, and B. Xu, “X-LLM: bootstrapping advanced large language models by treating multi-modalities as foreign languages,” CoRR, vol. abs/2305.04160, 2023.
[378]
↑
	X. Zhai, A. Kolesnikov, N. Houlsby, and L. Beyer, “Scaling vision transformers,” in CVPR. IEEE, 2022, pp. 1204–1213.
[379]
↑
	Z. Du, Y. Qian, X. Liu, M. Ding, J. Qiu, Z. Yang, and J. Tang, “GLM: general language model pretraining with autoregressive blank infilling,” in ACL (1). Association for Computational Linguistics, 2022, pp. 320–335.
[380]
↑
	F. Chen, D. Zhang, M. Han, X. Chen, J. Shi, S. Xu, and B. Xu, “VLP: A survey on vision-language pre-training,” Int. J. Autom. Comput., vol. 20, no. 1, pp. 38–56, 2023.
[381]
↑
	P. Xu, X. Zhu, and D. A. Clifton, “Multimodal learning with transformers: A survey,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 45, no. 10, pp. 12 113–12 132, 2023.
[382]
↑
	X. Wang, G. Chen, G. Qian, P. Gao, X. Wei, Y. Wang, Y. Tian, and W. Gao, “Large-scale multi-modal pre-trained models: A comprehensive survey,” Mach. Intell. Res., vol. 20, no. 4, pp. 447–482, 2023.
[383]
↑
	J. Zhang, J. Huang, S. Jin, and S. Lu, “Vision-language models for vision tasks: A survey,” CoRR, vol. abs/2304.00685, 2023.
[384]
↑
	C. Sun, A. Myers, C. Vondrick, K. Murphy, and C. Schmid, “Videobert: A joint model for video and language representation learning,” in ICCV. IEEE, 2019, pp. 7463–7472.
[385]
↑
	H. Bao, W. Wang, L. Dong, Q. Liu, O. K. Mohammed, K. Aggarwal, S. Som, S. Piao, and F. Wei, “Vlmo: Unified vision-language pre-training with mixture-of-modality-experts,” in NeurIPS, 2022.
[386]
↑
	X. Zhai, X. Wang, B. Mustafa, A. Steiner, D. Keysers, A. Kolesnikov, and L. Beyer, “Lit: Zero-shot transfer with locked-image text tuning,” in CVPR. IEEE, 2022, pp. 18 102–18 112.
[387]
↑
	M. Tsimpoukelli, J. Menick, S. Cabi, S. M. A. Eslami, O. Vinyals, and F. Hill, “Multimodal few-shot learning with frozen language models,” in NeurIPS, 2021, pp. 200–212.
[388]
↑
	J. Yu, Z. Wang, V. Vasudevan, L. Yeung, M. Seyedhosseini, and Y. Wu, “Coca: Contrastive captioners are image-text foundation models,” Trans. Mach. Learn. Res., vol. 2022, 2022.
[389]
↑
	P. Wang, A. Yang, R. Men, J. Lin, S. Bai, Z. Li, J. Ma, C. Zhou, J. Zhou, and H. Yang, “OFA: unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework,” in ICML, vol. 162. PMLR, 2022, pp. 23 318–23 340.
[390]
↑
	E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “Lora: Low-rank adaptation of large language models,” in ICLR, 2022.
[391]
↑
	C. Zheng, T. Vuong, J. Cai, and D. Phung, “Movq: Modulating quantized vectors for high-fidelity image generation,” in NeurIPS, 2022.
[392]
↑
	X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer, “Sigmoid loss for language image pre-training,” in ICCV. IEEE, 2023, pp. 11 941–11 952.
[393]
↑
	X. Gu, T. Lin, W. Kuo, and Y. Cui, “Open-vocabulary object detection via vision and language knowledge distillation,” in ICLR, 2022.
[394]
↑
	A. Kamath, M. Singh, Y. LeCun, G. Synnaeve, I. Misra, and N. Carion, “MDETR - modulated detection for end-to-end multi-modal understanding,” in ICCV. IEEE, 2021, pp. 1760–1770.
[395]
↑
	V. Blukis, C. Paxton, D. Fox, A. Garg, and Y. Artzi, “A persistent spatial semantic representation for high-level natural language instruction execution,” in CoRL, vol. 164. PMLR, 2021, pp. 706–717.
[396]
↑
	K. He, X. Chen, S. Xie, Y. Li, P. Dollár, and R. B. Girshick, “Masked autoencoders are scalable vision learners,” in CVPR. IEEE, 2022, pp. 15 979–15 988.
[397]
↑
	M. Minderer, A. A. Gritsenko, A. Stone, M. Neumann, D. Weissenborn, A. Dosovitskiy, A. Mahendran, A. Arnab, M. Dehghani, Z. Shen, X. Wang, X. Zhai, T. Kipf, and N. Houlsby, “Simple open-vocabulary object detection with vision transformers,” CoRR, vol. abs/2205.06230, 2022.
[398]
↑
	Z. Liu, H. Mao, C. Wu, C. Feichtenhofer, T. Darrell, and S. Xie, “A convnet for the 2020s,” in CVPR. IEEE, 2022, pp. 11 966–11 976.
[399]
↑
	J. Ho, W. Chan, C. Saharia, J. Whang, R. Gao, A. A. Gritsenko, D. P. Kingma, B. Poole, M. Norouzi, D. J. Fleet, and T. Salimans, “Imagen video: High definition video generation with diffusion models,” CoRR, vol. abs/2210.02303, 2022.
[400]
↑
	R. Villegas, M. Babaeizadeh, P. Kindermans, H. Moraldo, H. Zhang, M. T. Saffar, S. Castro, J. Kunze, and D. Erhan, “Phenaki: Variable length video generation from open domain textual descriptions,” in ICLR. OpenReview.net, 2023.
[401]
↑
	M. S. M. Sajjadi, D. Duckworth, A. Mahendran, S. van Steenkiste, F. Pavetic, M. Lucic, L. J. Guibas, K. Greff, and T. Kipf, “Object scene representation transformer,” in NeurIPS, 2022.
[402]
↑
	C. R. Qi, H. Su, K. Mo, and L. J. Guibas, “Pointnet: Deep learning on point sets for 3d classification and segmentation,” in CVPR. IEEE Computer Society, 2017, pp. 77–85.
[403]
↑
	C. R. Qi, L. Yi, H. Su, and L. J. Guibas, “Pointnet++: Deep hierarchical feature learning on point sets in a metric space,” in NIPS, 2017, pp. 5099–5108.
[404]
↑
	Y. Hong, C. Lin, Y. Du, Z. Chen, J. B. Tenenbaum, and C. Gan, “3d concept learning and reasoning from multi-view images,” in CVPR. IEEE, 2023, pp. 9202–9212.
[405]
↑
	K. M. Jatavallabhula, A. Kuwajerwala, Q. Gu, M. Omama, G. Iyer, S. Saryazdi, T. Chen, A. Maalouf, S. Li, N. V. Keetha, A. Tewari, J. B. Tenenbaum, C. M. de Melo, K. M. Krishna, L. Paull, F. Shkurti, and A. Torralba, “Conceptfusion: Open-set multimodal 3d mapping,” in RSS, 2023.
[406]
↑
	N. Reimers and I. Gurevych, “Sentence-bert: Sentence embeddings using siamese bert-networks,” in EMNLP/IJCNLP (1). Association for Computational Linguistics, 2019, pp. 3980–3990.
[407]
↑
	V. Sanh, L. Debut, J. Chaumond, and T. Wolf, “Distilbert, a distilled version of BERT: smaller, faster, cheaper and lighter,” CoRR, vol. abs/1910.01108, 2019.
[408]
↑
	M. N. Team. (2023) Introducing mpt-7b: A new standard for open-source, commercially usable llms. Accessed: 2023-05-05. [Online]. Available: www.mosaicml.com/blog/mpt-7b
[409]
↑
	A. Awadalla, I. Gao, J. Gardner, J. Hessel, Y. Hanafy, W. Zhu, K. Marathe, Y. Bitton, S. Y. Gadre, S. Sagawa, J. Jitsev, S. Kornblith, P. W. Koh, G. Ilharco, M. Wortsman, and L. Schmidt, “Openflamingo: An open-source framework for training large autoregressive vision-language models,” CoRR, vol. abs/2308.01390, 2023.
[410]
↑
	S. Biderman, H. Schoelkopf, Q. G. Anthony, H. Bradley, K. O’Brien, E. Hallahan, M. A. Khan, S. Purohit, U. S. Prashanth, E. Raff, A. Skowron, L. Sutawika, and O. van der Wal, “Pythia: A suite for analyzing large language models across training and scaling,” in ICML, vol. 202. PMLR, 2023, pp. 2397–2430.
[411]
↑
	M. Javaheripi, S. Bubeck, M. Abdin, J. Aneja, S. Bubeck, C. C. T. Mendes, W. Chen, A. Del Giorno, R. Eldan, S. Gopi et al., “Phi-2: The surprising power of small language models,” Microsoft Research Blog, 2023.
[412]
↑
	S. Black, S. Biderman, E. Hallahan, Q. Anthony, L. Gao, L. Golding, H. He, C. Leahy, K. McDonell, J. Phang, M. Pieler, U. S. Prashanth, S. Purohit, L. Reynolds, J. Tow, B. Wang, and S. Weinbach, “Gpt-neox-20b: An open-source autoregressive language model,” CoRR, vol. abs/2204.06745, 2022.
[413]
↑
	M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. de Oliveira Pinto, J. Kaplan, H. Edwards, Y. Burda, N. Joseph, G. Brockman, A. Ray, R. Puri, G. Krueger, M. Petrov, H. Khlaaf, G. Sastry, P. Mishkin, B. Chan, S. Gray, N. Ryder, M. Pavlov, A. Power, L. Kaiser, M. Bavarian, C. Winter, P. Tillet, F. P. Such, D. Cummings, M. Plappert, F. Chantzis, E. Barnes, A. Herbert-Voss, W. H. Guss, A. Nichol, A. Paino, N. Tezak, J. Tang, I. Babuschkin, S. Balaji, S. Jain, W. Saunders, C. Hesse, A. N. Carr, J. Leike, J. Achiam, V. Misra, E. Morikawa, A. Radford, M. Knight, M. Brundage, M. Murati, K. Mayer, P. Welinder, B. McGrew, D. Amodei, S. McCandlish, I. Sutskever, and W. Zaremba, “Evaluating large language models trained on code,” CoRR, vol. abs/2107.03374, 2021.
[414]
↑
	S. Karamcheti, S. Nair, A. Balakrishna, P. Liang, T. Kollar, and D. Sadigh, “Prismatic vlms: Investigating the design space of visually-conditioned language models,” in ICML, 2024.
[415]
↑
	L. Beyer, A. Steiner, A. S. Pinto, A. Kolesnikov, X. Wang, D. Salz, M. Neumann, I. Alabdulmohsin, M. Tschannen, E. Bugliarello, T. Unterthiner, D. Keysers, S. Koppula, F. Liu, A. Grycner, A. A. Gritsenko, N. Houlsby, M. Kumar, K. Rong, J. Eisenschlos, R. Kabra, M. Bauer, M. Bosnjak, X. Chen, M. Minderer, P. Voigtlaender, I. Bica, I. Balazevic, J. Puigcerver, P. Papalampidi, O. J. Hénaff, X. Xiong, R. Soricut, J. Harmsen, and X. Zhai, “Paligemma: A versatile 3b VLM for transfer,” CoRR, vol. abs/2407.07726, 2024.
[416]
↑
	A. Steiner, A. S. Pinto, M. Tschannen, D. Keysers, X. Wang, Y. Bitton, A. A. Gritsenko, M. Minderer, A. Sherbondy, S. Long, S. Qin, R. R. Ingle, E. Bugliarello, S. Kazemzadeh, T. Mesnard, I. Alabdulmohsin, L. Beyer, and X. Zhai, “Paligemma 2: A family of versatile vlms for transfer,” CoRR, vol. abs/2412.03555, 2024.
[417]
↑
	Chameleon-Team, “Chameleon: Mixed-modal early-fusion foundation models,” CoRR, vol. abs/2405.09818, 2024.
[418]
↑
	X. Wang, X. Zhang, Z. Luo, Q. Sun, Y. Cui, J. Wang, F. Zhang, Y. Wang, Z. Li, Q. Yu, Y. Zhao, Y. Ao, X. Min, T. Li, B. Wu, B. Zhao, B. Zhang, L. Wang, G. Liu, Z. He, X. Yang, J. Liu, Y. Lin, T. Huang, and Z. Wang, “Emu3: Next-token prediction is all you need,” CoRR, vol. abs/2409.18869, 2024.
[419]
↑
	H. Liu, W. Yan, M. Zaharia, and P. Abbeel, “World model on million-length video and language with blockwise ringattention,” in ICLR. OpenReview.net, 2025.
[420]
↑
	D. K. Misra, A. Bennett, V. Blukis, E. Niklasson, M. Shatkhin, and Y. Artzi, “Mapping instructions to actions in 3d environments with visual goal prediction,” in EMNLP, 2018, pp. 2667–2678.
[421]
↑
	C. Lynch, M. Khansari, T. Xiao, V. Kumar, J. Tompson, S. Levine, and P. Sermanet, “Learning latent plans from play,” in CoRL, vol. 100. PMLR, 2019, pp. 1113–1132.
[422]
↑
	M. Bhardwaj, B. Sundaralingam, A. Mousavian, N. D. Ratliff, D. Fox, F. Ramos, and B. Boots, “STORM: an integrated framework for fast joint-space model-predictive control for reactive manipulation,” in CoRL, vol. 164. PMLR, 2021, pp. 750–759.
[423]
↑
	A. Jaegle, S. Borgeaud, J. Alayrac, C. Doersch, C. Ionescu, D. Ding, S. Koppula, D. Zoran, A. Brock, E. Shelhamer, O. J. Hénaff, M. M. Botvinick, A. Zisserman, O. Vinyals, and J. Carreira, “Perceiver IO: A general architecture for structured inputs & outputs,” in ICLR, 2022.
[424]
↑
	E. Wijmans, A. Kadian, A. Morcos, S. Lee, I. Essa, D. Parikh, M. Savva, and D. Batra, “DD-PPO: learning near-perfect pointgoal navigators from 2.5 billion frames,” in ICLR, 2020.
[425]
↑
	M. Zhu, Y. Zhu, J. Li, J. Wen, Z. Xu, N. Liu, R. Cheng, C. Shen, Y. Peng, F. Feng, and J. Tang, “Scaling diffusion policy in transformer to 1 billion parameters for robotic manipulation,” CoRR, vol. abs/2409.14411, 2024.
[426]
↑
	J. W. Wei, L. Hou, A. K. Lampinen, X. Chen, D. Huang, Y. Tay, X. Chen, Y. Lu, D. Zhou, T. Ma, and Q. V. Le, “Symbol tuning improves in-context learning in language models,” CoRR, vol. abs/2305.08298, 2023.
[427]
↑
	Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole, “Score-based generative modeling through stochastic differential equations,” in ICLR, 2021.
[428]
↑
	A. Zeng, P. Florence, J. Tompson, S. Welker, J. Chien, M. Attarian, T. Armstrong, I. Krasin, D. Duong, V. Sindhwani, and J. Lee, “Transporter networks: Rearranging the visual world for robotic manipulation,” in CoRL, vol. 155. PMLR, 2020, pp. 726–747.
[429]
↑
	R. Goyal, S. E. Kahou, V. Michalski, J. Materzynska, S. Westphal, H. Kim, V. Haenel, I. Fründ, P. Yianilos, M. Mueller-Freitag, F. Hoppe, C. Thurau, I. Bax, and R. Memisevic, “The ”something something” video database for learning and evaluating visual common sense,” in ICCV. IEEE Computer Society, 2017, pp. 5843–5851.
[430]
↑
	X. Lin, J. So, S. Mahalingam, F. Liu, and P. Abbeel, “Spawnnet: Learning generalizable visuomotor skills from pre-trained networks,” CoRR, vol. abs/2307.03567, 2023.
[431]
↑
	S. P. Arunachalam, I. Güzey, S. Chintala, and L. Pinto, “Holo-dex: Teaching dexterity with immersive mixed reality,” in ICRA. IEEE, 2023, pp. 5962–5969.
[432]
↑
	N. M. M. Shafiullah, A. Rai, H. Etukuru, Y. Liu, I. Misra, S. Chintala, and L. Pinto, “On bringing robots home,” CoRR, vol. abs/2311.16098, 2023.
[433]
↑
	Y. Seo, D. Hafner, H. Liu, F. Liu, S. James, K. Lee, and P. Abbeel, “Masked world models for visual control,” in CoRL, vol. 205. PMLR, 2022, pp. 1332–1344.
[434]
↑
	R. Mendonca, S. Bahl, and D. Pathak, “Structured world models from human videos,” in RSS, 2023.
[435]
↑
	M. Pan, X. Zhu, Y. Wang, and X. Yang, “Iso-dream: Isolating and leveraging noncontrollable visual dynamics in world models,” in NeurIPS, 2022.
[436]
↑
	Z. Dai, Z. Yang, Y. Yang, J. G. Carbonell, Q. V. Le, and R. Salakhutdinov, “Transformer-xl: Attentive language models beyond a fixed-length context,” in ACL (1). Association for Computational Linguistics, 2019, pp. 2978–2988.
[437]
↑
	A. Avetisyan, C. Xie, H. Howard-Jenkins, T. Yang, S. Aroudj, S. Patra, F. Zhang, D. P. Frost, L. Holland, C. Orme, J. Engel, E. Miller, R. A. Newcombe, and V. Balntas, “Scenescript: Reconstructing scenes with an autoregressive structured language model,” CoRR, vol. abs/2403.13064, 2024.
[438]
↑
	N. M. Shafiullah, Z. J. Cui, A. Altanzaya, and L. Pinto, “Behavior transformers: Cloning $k$ modes with one stone,” in NeurIPS, 2022.
[439]
↑
	Z. J. Cui, Y. Wang, N. M. M. Shafiullah, and L. Pinto, “From play to policy: Conditional behavior generation from uncurated robot data,” in ICLR, 2023.
[440]
↑
	S. Lee, Y. Wang, H. Etukuru, H. J. Kim, N. M. M. Shafiullah, and L. Pinto, “Behavior generation with latent actions,” in ICML, 2024.
[441]
↑
	X. Li, C. Mata, J. Park, K. Kahatapitiya, Y. S. Jang, J. Shang, K. Ranasinghe, R. D. Burgert, M. Cai, Y. J. Lee, and M. S. Ryoo, “Llara: Supercharging robot learning data for vision-language policy,” in ICLR. OpenReview.net, 2025.
[442]
↑
	K. Pertsch, K. Stachowicz, B. Ichter, D. Driess, S. Nair, Q. Vuong, O. Mees, C. Finn, and S. Levine, “FAST: efficient action tokenization for vision-language-action models,” CoRR, vol. abs/2501.09747, 2025.
[443]
↑
	P. Intelligence, K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, M. Y. Galliker, D. Ghosh, L. Groom, K. Hausman, B. Ichter, S. Jakubczak, T. Jones, L. Ke, D. LeBlanc, S. Levine, A. Li-Bell, M. Mothukuri, S. Nair, K. Pertsch, A. Z. Ren, L. X. Shi, L. M. Smith, J. T. Springenberg, K. Stachowicz, J. Tanner, Q. Vuong, H. Walke, A. Walling, H. Wang, L. Yu, and U. Zhilinsky, “
𝜋
0.5
: a vision-language-action model with open-world generalization,” CoRR, vol. abs/2504.16054, 2025.
[444]
↑
	Z. Xu, H. L. Chiang, Z. Fu, M. G. Jacob, T. Zhang, T. E. Lee, W. Yu, C. Schenck, D. Rendleman, D. Shah, F. Xia, J. Hsu, J. Hoech, P. Florence, S. Kirmani, S. Singh, V. Sindhwani, C. Parada, C. Finn, P. Xu, S. Levine, and J. Tan, “Mobility VLA: multimodal instruction navigation with long-context vlms and topological graphs,” in CoRL, ser. Proceedings of Machine Learning Research, vol. 270. PMLR, 2024, pp. 3866–3887.
[445]
↑
	J. Bjorck, F. Castañeda, N. Cherniadev, X. Da, R. Ding, Linxi, Y. Fang, D. Fox, F. Hu, S. Huang, J. Jang, Z. Jiang, J. Kautz, K. Kundalia, L. Lao, Z. Li, Z. Lin, K. Lin, G. Liu, E. LLontop, L. Magne, A. Mandlekar, A. Narayan, S. Nasiriany, S. Reed, Y. L. Tan, G. Wang, Z. Wang, J. Wang, Q. Wang, J. Xiang, Y. Xie, Y. Xu, Z. Xu, S. Ye, Z. Yu, A. Zhang, H. Zhang, Y. Zhao, R. Zheng, and Y. Zhu, “GR00T N1: an open foundation model for generalist humanoid robots,” CoRR, vol. abs/2503.14734, 2025.
[446]
↑
	P. Ding, J. Ma, X. Tong, B. Zou, X. Luo, Y. Fan, T. Wang, H. Lu, P. Mo, J. Liu, Y. Wang, H. Zhou, W. Feng, J. Liu, S. Huang, and D. Wang, “Humanoid-vla: Towards universal humanoid control with visual integration,” CoRR, vol. abs/2502.14795, 2025.
[447]
↑
	P. Ding, H. Zhao, W. Zhang, W. Song, M. Zhang, S. Huang, N. Yang, and D. Wang, “QUAR-VLA: vision-language-action model for quadruped robots,” in ECCV (5), ser. Lecture Notes in Computer Science, vol. 15063. Springer, 2024, pp. 352–367.
[448]
↑
	X. Tong, P. Ding, D. Wang, W. Zhang, C. Cui, M. Sun, Y. Fan, H. Zhao, H. Zhang, Y. Dang, S. Huang, and S. Lyu, “Quart-online: Latency-free large multimodal language model for quadruped robot learning,” CoRR, vol. abs/2412.15576, 2024.
[449]
↑
	H. Zhao, W. Song, D. Wang, X. Tong, P. Ding, X. Cheng, and Z. Ge, “More: Unlocking scalability in reinforcement learning for quadruped vision-language-action models,” CoRR, vol. abs/2503.08007, 2025.
[450]
↑
	P. Chen, P. Bu, Y. Wang, X. Wang, Z. Wang, J. Guo, Y. Zhao, Q. Zhu, J. Song, S. Yang, J. Wang, and B. Zheng, “Combatvla: An efficient vision-language-action model for combat tasks in 3d action role-playing games,” CoRR, vol. abs/2503.09527, 2025.
[451]
↑
	M. Li, Z. Wang, K. He, X. Ma, and Y. Liang, “JARVIS-VLA: post-training large-scale vision language models to play visual games with keyboards and mouse,” in ACL (Findings). Association for Computational Linguistics, 2025, pp. 17 878–17 899.
-ABackground
-A1Unimodal Models

Vision-language-action models integrate three modalities, often relying on existing unimodal models. The transition from convolutional neural networks (e.g., ResNet [214]) to visual Transformers (e.g., ViT [215], SAM [216]) in computer vision has facilitated the development of vision foundation models (VFMs). In natural language processing, the evolution from recurrent neural networks (e.g., LSTM [217], GRU [218]) to Transformers [4] initially led to the pretrain-finetune paradigm (e.g., BERT [29], ChatGPT [1]), followed by the recent success of prompt tuning driven by large language models. Reinforcement learning (e.g., DQN [5], AlphaGo [219], PPO [220], Dactyl [221]) has also witnessed a shift towards employing Transformers to model the Markov Decision Process as autoregressive sequential data. DPO [222] directly trains LLMs on human preferences, simplifying RLHF.

-A2Vision-Language Models

Vision-language tasks, encompassing image captioning [223], visual question answering [224], visual grounding [225], require the fusion of computer vision and natural language processing models. Early efforts, such as Show and Tell [226], leveraged the success of early CNNs and RNNs. The advent of high-capacity language models, including BERT [29] and GPT [227], ushered in a new era of Transformer-based VLMs. One of the pioneering self-supervised pretraining methods is ViLBERT [228], while CLIP [25] popularized contrastive pretraining approaches for multimodal alignment. The recent emergence of large language models has driven the development of multimodal LLMs (MLLMs) or large multimodal models (LMMs), which achieve state-of-the-art performance on multimodal instruction-following tasks. Representative MLLMs include Flamingo [229], BLIP-2 [7], and LLaVA [8]. VLMs share a close relationship with VLAs, as the multimodal architectures of VLMs can be readily adopted for VLAs. Additionally, VLMs can function as high-level task planners if they possess sufficient reasoning capabilities.

-BBackground (Extended Version): Unimodal Models

Vision-language-action models involve three modalities, and consequently, many VLAs depend on existing unimodal models for processing inputs from different modalities. Therefore, it is crucial to summarize representative developments in unimodal models, as they often serve as integral components in VLAs. Specifically, for the vision modality, we collect models designed for image classification, object detection, and image segmentation, as these tasks are particularly relevant for robotic learning. Natural language processing models play a crucial role in enabling VLAs to understand language instructions or generate language responses. Reinforcement learning is a foundational component for obtaining optimal policies, facilitating the generation of appropriate actions in a given environment and condition. A brief time of the development of unimodal models is depicted in Figure 7. Additionally, Figure 8 highlights the progressive increase in model size within these fields.

-B1Computer Vision
Figure 7:A brief timeline of pivotal unimodal models leading to the development of vision-language-action models, organized by their publication years. Details can be found in Appendix:-B.

Computer vision witnessed the inception of modern neural networks. In robotics, object classification models can be used to inform a policy about which objects are of interest, and models for object detection or image segmentation can help precisely locate objects. Therefore, we mainly summarize approaches for these tasks, but numerous excellent surveys on visual models, ranging from convolutional neural networks (CNNs) [230] to Transformers [231], offer more detailed insights. Interested readers are directed to these surveys for a more comprehensive introduction. Here, we will briefly touch upon some of the key developments in the field of computer vision.

Convolutional neural network

Early developments in computer vision (CV) were primarily focused on the image classification task. LeNet [232] was among the first convolutional neural networks, designed for identifying handwritten digits in zip codes. In 2012, AlexNet [3] emerged as a breakthrough by winning the ImageNet challenge, showcasing the potential of neural networks. VGG [233] demonstrated the benefits of increasing the depth of CNNs. GoogLeNet [234], also known as Inception-V1, introduced the concept of blocks. ResNet [214] introduced skip connections or residual connections. Inception-ResNet [235], as the name suggests, combines residual connects and inception blocks. ResNeXt [236] explored the concept of split, transform, and merge. SENet [237] introduces the squeeze-and-excitation blocks, utilizing a type of attention mechanism. EfficientNet [125] studied the width, depth, and resolution of CNN models with “compound scaling”, highlighting the trade-off between efficiency and performance.

Alongside image classification, object detection became an integral component in many applications. Building upon the success of image classification backbone networks, a series of works optimized region-based methods: R-CNN [238], Fast R-CNN [239], Faster R-CNN [240], and Mask R-CNN [241]. Grid-based methods like YOLO [242] are also widely adopted. Bottom-up, top-down is also a popular strategy, employed by FPN [243], RetinaNet [244], BUTD [245], etc. In scenarios requiring more detailed and precise object detection, image segmentation aims to determine the exact outline of objects. Many popular models adopt an “encoder-decoder” architecture, where the encoder understands both the global and local context of the image, and the decoder produces a segmentation map based on this context information. Representative works following this idea include FCN [246], SegNet [247], Mask R-CNN [241], and U-Net [248].

Vision Transformer

Convolutional neural networks (CNNs) have historically been the foundation of computer vision models. However, the landscape shifted with the introduction of the Transformer architecture in the seminal work by [4]. This paradigm shift was initiated by ViT [215]. It revolutionizes image processing by breaking down images into 16-by-16 pixel patches, treating each as a token akin to those in NLP; leveraging a BERT-like model, ViT encodes these patches and has exhibited superior performance over many traditional CNN models in image classification tasks.

The transformative power of the Transformer extends beyond classification. DETR [249] employs an encoder-decoder Transformer architecture to tackle object detection. The encoder processes the input image, and its output embeddings are fed into the decoder through cross-attention. Notably, DETR introduces learnable object queries to the decoder, facilitating the extraction of crucial object-wise information from the encoder’s output. Venturing into image segmentation, Segmenter [250] was the first to utilize Transformer on this task. The Segment Anything model (SAM) [216] achieves remarkable milestones in promptable segmentation, zero-shot performance, and versatile architecture, further underlining the transformative impact of Vision Transformer models in various computer vision domains.

Vision in 3D

Aside from the most common RGB data, other types of visual inputs are widely used [251, 252]. In robotics, depth maps are useful since they provide essential 3D information that is not explicitly stored in RGB images. Depth maps can be captured with Microsoft Kinect 1 or Intel RealSense 2 or recovered from pure RGB images. Point clouds [253] are also popular visual input types due to the widespread adoption of LiDARs and 3D scanners; depth maps can be easily converted to point clouds. Volumetric data [254], such as voxels or octrees, is usually more information-rich than depth maps and is suitable for representing rigid objects. Despite the widespread use of 3D meshes as the default data format in computer graphics, their irregular nature poses challenges for neural networks [255].

Figure 8:The growing scale of unimodal models over the years. 
∗
GPT-4’s model size is estimated since it has not been officially disclosed.
-B2Natural Language Processing

Natural Language Processing (NLP) plays a pivotal role in VLA, serving as a vital component for understanding user instructions, or even generating appropriate textual responses. The recent surge in NLP owes much to the success of Transformer models [4]. In the landscape of contemporary NLP, there is a noticeable shift towards implicit learning of language syntax and semantics, a departure from the previous paradigms. To provide context, this subsection will commence with a concise overview of fundamental yet enduring concepts before delving into the noteworthy advancements in contemporary NLP. For an in-depth exploration of the progress in the NLP domain, readers are directed to comprehensive surveys by [256] and [257], which meticulously reviews the trajectory of advancements in NLP.

Early developments

The field of NLP, which was more frequently referred to as Computational Linguistics (CL) in the early days, tries to solve various tasks regarding to natural language. In CL, natural languages used to be processed in a hierarchical way: word, syntax, and semantics. Firstly, on the word level, many aspects need to be accounted for, including morphology, lexicology, phonology, etc. This leads to problems like tokenization, lemmatization, stemming, semantic relations, word sense disambiguation, and Zipf’s Law problem. Then in terms of syntax, natural language, in contrast to formal language, has much less restricted grammar and thus is more challenging to parse: in the Chomsky hierarchy, natural language is generally considered to follow context-sensitive grammar while programming languages are covered under context-free grammar. Syntactic parsing includes tasks such as part-of-speech tagging, constituency parsing, dependency parsing, and named entity recognition. Finally, to understand the semantics of a sentence in written language or an utterance in spoken language, the following tasks were studied: semantic role labeling, frame semantic parsing, abstract meaning representation, logical form parsing, etc.

Recurrent neural network & convolutional neural network

In the initial stages of NLP, rudimentary models relied on simple feed-forward neural networks to tackle various tasks [258]. After the introduction of word embeddings like word2vec [259, 260] and GloVe [261], NLP techniques embraced recurrent neural networks (RNNs) [262], such as LSTM [217], GRU [218], RNNsearch [263], LSTM-CRF [264], etc. Examples of representative RNNs in NLP include ELMo [265] and ULMFiT [266]. While RNNs played a significant role, alternative models utilizing convolutional neural networks also emerged. WordCNN [267] employed CNNs at the word level, with word2vec features serving as input to the CNNs. Another approach, CharCNN [268], focused on modeling language at the character level with CNN. Subsequent research [269] highlighted that character-level CNNs excel at capturing word morphology, and their combination with a word-level LSTM backbone can significantly enhance performance.

Transformer & large language model

The groundbreaking Transformer model, introduced by [4], revolutionized natural language processing through the introduction of the self-attention mechanism, inspiring a cascade of subsequent works. BERT [29] leverages the Transformer encoder stack, excelling in natural language understanding. On the other hand, the GPT family [227, 270, 271, 272] is built upon Transformer decoder blocks, showcasing prowess in natural language generation tasks. A line of work strives to refine the original BERT, including RoBERTa [273], ALBERT [274], ELECTRA [275]. Simultaneously, a parallel line of research following the GPT paradigm has given rise to models like XLNet [276], OPT [277]. BART [278], an encoder-decoder Transformer, distinguishes itself through pretraining using the denoising sequence-to-sequence task. Meanwhile, T5 [279] introduces modifications to the original Transformer, maintaining the encoder-decoder architecture. T5 unifies various NLP tasks through a shared text-to-text format, exhibiting enhanced performance in transfer learning. These diverse models collectively showcase the versatility and ongoing evolution within the NLP landscape.

Over the past few years, there has been a remarkable expansion in the size of language models, driven by the scalability of the Transformer architecture. This trend has given rise to a series of large language models (LLMs) that have demonstrated breakthrough performance and capabilities not achievable with smaller models. A landmark model in this evolution is ChatGPT [1], which has sparked considerable interest and inspired a series of works in this domain, such as GPT-4 [272], PaLM [280], PaLM-2 [281], LLaMA [282], LLaMA 2 [283], ERNIE 3.5 [284]. Notably, LLaMA stands out as one of the few open-source LLMs, fostering interesting developments. The introduction of “instruction-tuning” has allowed efficient fine-tuning of a pretrained LLM to become an instruction-following model. This technique is popularized by InstructGPT [285] and FLAN [286, 287]. Recent advancements in instruction-following models include Alpaca [288], employing self-instruction, and Vicuna [289], leveraging conversations from ShareGPT. As LLMs grow in scale and power, there is a shift away from the need for fine-tuning on downstream tasks. With appropriate prompts, LLMs can produce accurate outputs without task-specific training, a paradigm known as prompt engineering. This approach differs from the traditional pretrain-finetune paradigm. DPO [222] introduces a method to train LLMs directly on human preferences without requiring explicit reward modeling or reinforcement learning, simplifying upon previous RLHF methods.

-B3Reinforcement Learning

Reinforcement learning (RL) seeks to acquire a policy capable of taking optimal actions based on observations of the environment. Numerous vision-language-action models are constructed based on paradigms such as imitation learning or Temporal Difference (TD) learning within RL. Many challenges faced in the development of robotic policies can be effectively addressed through insights gained from the field of RL. Consequently, delving into RL methods presents a valuable avenue for enhancing robotic learning. For a deeper exploration of RL methods, readers can refer to more comprehensive reviews provided by [290], [291], and [292].

Deep reinforcement learning

The advent of deep reinforcement learning can be attributed to the success of pioneering models, Deep Q-Network [5] and AlphaGo [219]. Deep learning, with its ability to provide low-dimensional representations, proved instrumental in overcoming traditional computational and memory complexity challenges in reinforcement learning. In recent years, a multitude of value-function based approaches has surfaced. Double DQN (D-DQN) [293] addresses the action value overestimation issue of DQN. Hindsight experience replay (HER) [294] focuses on the sparse reward issue. Batch-Constrained deep Q-learning (BCQ) [295] presents an approach aimed at enhancing off-policy learning by constraining the action space. BEAR [296] endeavors to alleviate instability arising from bootstrapping errors in off-policy RL. [297] introduces conservative Q-learning (CQL) to address the overestimation of values by standard off-policy RL methods.

Another paradigm within reinforcement learning is policy search, encompassing methods such as policy gradient and actor-critic techniques. These approaches aim to combat persistent challenges, including instability, slow convergence, and data inefficiency. Guided policy search (GPS) [298] learns a policy with importance sampling guided by another controller toward a local optimum. Deterministic policy gradient (DPG) [299], deep deterministic policy gradient (DDPG) [300], asynchronous advantage actor-critic (A3C) [301] improve efficiency without compromising stability. Normalized advantage functions (NAF) [302] is a continuous variant of Q-learning, allowing for Q-learning with experience replay. Soft actor-critic (Soft AC) [303] take advantage of maximum entropy RL framework to lower sample complexity and improve convergence properties. Trust region policy optimization (TRPO) [304] and proximal policy optimization (PPO) [220] utilize trust region methods to stabilize policy gradients, with PPO additionally incorporating a truncated generalized advantage estimation (GAE) [305].

Beyond these, various other RL methodologies exist, such as imitation learning and hierarchical reinforcement learning. Generative adversarial imitation learning (GAIL) [306] is an imitation learning method that uses a generative adversarial framework to discriminate expert trajectories against generated trajectories. Robust adversarial reinforcement learning (RARL) [307] incorporates adversarial agents to enhance generalization. RLHF [308] utilizes human preferences without access to the reward function. FeUdal Networks (FuN) [309] introduce a hierarchical reinforcement learning architecture featuring a Manager module and a Worker module.

Robotics

In the field of RL, robotics stands out as one of the most prevalent and impactful applications. A noteworthy contribution to this field is E2E-DVP [310]. This pioneering model represents one of the first end-to-end solutions for robotic control. Its neural network is designed to take raw images as input and generate motor torques as output. [6] curated a substantial real-world dataset and developed a CNN that predicts grasps based on monocular input images. Building upon these foundations, QT-Opt [311] further expands the dataset and model scale, introducing closed-loop control capabilities to enhance robotic control systems. Then Dreamer [55] addresses long-horizon tasks. OpenAI also developed a dexterous robot hand that can solve the Rubik’s cube [221, 312].

-B4Graph

Graph is ubiquitous in many scenarios, such as social network, molecule structure, 3D object meshes, etc. Even images and text can be modeled as 2D grid and linear graph (path graph), respectively. To process graph-structured data, recurrent graph neural networks [313] were first introduced, which were later optimized by convolutional graph neural networks. The review of graph neural networks (GNN) [314] can be referred to for more in-depth details.

Convolutional graph neural networks can be generally divided into two categories: spectral-based and spatial-based. Spectral-based convolutional GNNs draw inspiration from graph signal processing, which provides theoretical support for the design of the networks. However, spatial-based convolutional GNNs have advantage in terms of their efficiency and flexibility. Spectral CNN [315] is one of the first convolutional GNNs but it is not robust to changes in graph structure and has high computational cost. ChebNet [316] and GCN [317] significantly reduced the cost: ChebNet used approximation based on Chebyshev polynomials and GCN is its first-order approximation. Neural Network for Graphs (NN4G) [318] is the first spatial network. MPNN [319] introduced a general framework of spatial-based networks under which most existing GNNs can be covered. But the drawback of MPNN is that it does not embed graph structure information, which is later solved by GIN [320]. GraphSage [321] improves efficiency by sampling a fixed number of neighbors. Graph Attention Network (GAT) [322] incorporates the attention mechanism.

Besides recurrent and convolutional graph neural networks, there are graph autoencoders [323, 324] and spatial-temporal graph neural networks [325]. Equivariant message passing networks are recently introduced to handle 3D graphs, including E(n)- and SE(3)- Equivariant GNN [326, 327]. Graph Transformer models make use of the power of transformers to processing graph data. There are already over 20 such graph Transformer models, such as GROVER [328] and SE(3)-Transformers [329].

Graph and vision

Graph structures also exist in some computer vision tasks. Scene graph [330] can be used to express object relationships in most visual inputs. In addition to detecting objects in an image, scene graph generation necessitates the understanding of the relationships between detected objects. For example, a model needs to detect a person and a cup, and then understand that the person is holding the cup, which is the relationship between the two objects. Knowledge graphs often contain visual illustrations, such as WikiData. Those graph can be helpful in downstream computer vision tasks.

Graph and language

Graph structures are ubiquitous in language data [331]. Word-level graphs include dependency graph, constituency graph, AMR graph, etc. Word-level means each node of those graphs corresponds to a word in the original text. There graphs can be used to explicitly represent the syntax or semantics of the raw sentence. Sentence-level graphs can be useful in dialog tracking [332], fact checking [333], etc. Document-level graphs include knowledge graphs [334], citation graphs [335], etc. They can be used in document-level tasks, such as document retrieval, document clustering, etc. Different types language graphs are often processed using aforementioned GNNs to facilitate downstream tasks.

-CBackground (Extended Version): Vision-Language Models
Table VIII:Vision-language models. In the “Objective” column: “MLM”: masked language modeling. “MVM”: masked vision modeling, reconstructing masked image regions. “VLM”: binary classification of whether vision and language inputs are a match. “LM”: autoregressive language modeling. “VLCL”: vision-language contrastive learning. We only include representative multi-modal datasets due to limited space. “MM” includes multi-modal tasks such as visual question answering, image captioning, and vision-language retrieval. “Vision” represents computer vision tasks, like image classification. “Language” represents natural language processing tasks.
 	
	Vision Encoder	Language Encoder	
	
	
	


 	
Model
	
Name
	
Params
	
Name
	
Params
	
VL-Fusion
	
Objectives
	
Datasets
	
Tasks


Self-supervised
 	
ViLBERT [228]
	
Faster R-CNN [240, 245]
	
44M
	
Dual-stream BERT-base [29]
	
221M
	
Dual-stream
	
MLM, MVM, VLM
	
COCO, VG
	
MM


 	
LXMERT [336]
	
Faster R-CNN [240, 245]
	
44M
	
Dual-stream BERT-base [29]
	
183M
	
Dual-stream
	
MLM, MVM, VLM, VQA
	
COCO, VG, VQA, GQA, VGQA
	
MM


 	
VisualBERT [337]
	
Faster R-CNN [240, 245]
	
60M
	
BERT-base [29]
	
110M
	
Single-stream
	
MLM, VLM
	
COCO
	
MM


 	
VL-BERT [338]
	
Faster R-CNN [240, 245]
	
44M
	
BERT-base [29]
	
110M
	
Single-stream
	
MLM, MVM
	
CC
	
MM


 	
UNITER [339]
	
Faster R-CNN [240, 245]
	
44M
	
BERT-base/
BERT-large [29]
	
86M/
303M
	
Single-stream
	
MLM, VLM, MVM, WRA
	
COCO, VG, SBU, CC
	
MM


 	
ViLT [340]
	
Linear projection [215]
	
2.4M
	
BERT-base [29]
	
85M
	
Single-stream
	
MLM, VLM
	
COCO, VG, SBU, CC
	
MM


 	
SimVLM [341]
	
ViT/CoAtNet-huge (shared encoder) [342]
	
632M
	
Shared encoder
	
632M
	
Single-stream
	
PrefixLM
	
ALIGN dataset
	
MM


 	
GIT [343]
	
Florence [344]
	
637M
	
Transformer [4]
	
60M
	
Single-stream
	
LM
	
COCO, VG, SBU, CC, etc.
	
MM


 	
BEiT-3 [345]
	
V-FFN
(+Shared Attn)
	
692M (+317M)
	
L-FFN
(+Shared Attn)
	
692M (+317M)
	
Modality experts
	
MLM, VLM
	
COCO, VG, SBU, CC
	
MM, Vision


Contrastive
 	
CLIP [25]
	
ViT [215]
	
428M
	
GPT-2 [270]
	
63M
	
Two-tower
	
VLCL
	
WTI
	
Vision


 	
FILIP [346]
	
ViT-L/14 [215]
	
428M
	
GPT [270]
	
117M
	
Two-tower
	
VLCL
	
FILIP300M (Self-collect)
	
MM, Vision


 	
ALIGN [347]
	
EfficientNet-L2 [348]
	
480M
	
BERT-large [29]
	
336M
	
Two-tower
	
VLCL
	
ALIGN dataset (Self-collect)
	
MM, Vision


 	
ALBEF [349]
	
ViT-B/16 [215]
	
87M
	
BERT-base [29]
	
85M
	
Dual-stream
	
MLM, VLM, VLCL
	
COCO, VG, CC, SBU
	
MM


 	
FLAVA [350]
	
ViT-B/16 [215]
	
87M
	
RoBERTa-base [273]
	
125M
	
Dual-stream
	
MLM, MVM, VLM, VLCL
	
COCO, VG, CC, SBU, etc. (PMD)
	
MM, Vision, Language


 	
Florence [344]
	
Hierarchical Vision Transformers [351, 352]
	
637M
	
RoBERTa [273]
	
125M
	
Two-tower
	
VLCL
	
FLD-900M (Self-collect)
	
Vision


Large Multi-modal Model
 	
Flamingo [229]
	
NFNet-F6 [353]
	
438M
	
Chinchilla [354]
	
70B
	
Dual-stream
	
LM
	
M3W, ALIGN dataset, LTIP, VTP
	
MM


 	
BLIP-2 [7]
	
CLIP ViT-L/14 [25], EVA ViT-G/14 [355]
+ Q-Former
	
428M,
1B
	
OPT [277]
Flan-T5 [287]
	
6.7B
3B/11B
	
Single-stream
	
BLIP, LM
	
COCO, VG, CC, SBU, LAION
	
MM


 	
PaLI [356]
	
ViT-e [356]
	
4B
	
mT5-XXL [357]
	
13B
	
Single-stream
	
Mixed
	
WebLI, etc
	
MM


 	
PaLI-X [358]
	
ViT-22B [359]
	
22B
	
UL2 [360]
	
32B
	
Single-stream
	
Mixed
	
WebLI, etc
	
MM


 	
LLaMA-Adapter [361]
	
CLIP ViT-B/16 [25]
	
87M
	
LLaMA [282]
	
7B
	
Single-stream
	
LM
	
Self-instruct
	
Instruction-following


 	
LLaMA-Adapter-V2 [362]
	
CLIP ViT-L/14 [25]
	
428M
	
LLaMA [282]
	
7B
	
Single-stream
	
LM
	
GPT-4-LLM, COCO, ShareGPT
	
Instruction-following


 	
Kosmos-1 [363],
Kosmos-2 [364]
	
CLIP ViT-L/14 [25]
	
428M
	
Magneto [365]
	
1.3B
	
Single-stream
	
LM
	
LAION, COYO, CC; Unnatural Instructions, FLANv2
	
Instruction-following (Kosmos-2 w/ grounding, referring)


 	
InstructBLIP [366]
	
EVA ViT-G/14 [355]
	
1B
	
Flan-T5 [287]
Vicuna [289]
	
3B/11B
7B/13B
	
Single-stream
	
BLIP, LM
	
COCO, VQA, LLaVA-Instruct-150K, etc. (26 datasets)
	
Instruction-following


 	
LLaVA [8]
	
CLIP ViT-L/14 [25]
	
428M
	
LLaMA [282]
	
13B
	
Single-stream
	
LM
	
CC, (FT: GPT-assisted Visual Instruction Data Generation)
	
Instruction-following


 	
MiniGPT-4 [367]
	
EVA ViT-G/14 [355]
+ Q-Former [7]
	
1B
	
Vicuna [289]
LLaMA2 [283]
	
7B/13B
7B
	
Single-stream
	
LM
	
CC, SBU, LAION (FT: SC)
	
Instruction-following


 	
Video-LLaMA [368]
	
EVA ViT-G/14 [355]
+ Q-Former [7]
	
1B
	
LLaMA [282]
	
7B/13B
	
Single-stream
	
BLIP, LM
	
CC595k
	
Instruction-following


 	
PandaGPT [369]
	
ImageBind ViT-H [205]
	
632M
	
Vicuna [289]
	
13B
	
Single-stream
	
LM
	
(FT: LLaVA data, MiniGPT-4 data)
	
Instruction-following


 	
VideoChat [370]
	
EVA ViT-G/14 [355]
+ Q-Former [7]
	
1B
	
StableVicuna [371]
	
13B
	
Single-stream
	
LM
	
COCO, VG, CC, SBU (FT: SC, MiniGPT-4, LLaVA data)
	
Instruction-following


 	
ChatSpot [372]
	
CLIP ViT-L/14 [25]
	
428M
	
Vicuna [289]
	
7B
	
Single-stream
	
LM
	
MGVLID, RegionChat
	
Instruction-following, Vision


 	
mPLUG-Owl [373], mPLUG-Owl2 [374]
	
CLIP ViT-L/14 [25]
+ Visual Abstractor
	
428M
	
LLaMA [282]
	
7B
	
Single-stream
	
LM
	
LAION, COYO, CC, COCO (FT: Alpaca, Vicuna, Baize [375] data)
	
Instruction-following


 	
Visual ChatGPT [376]
	
(22 different models)
	
-
	
ChatGPT [1]
	
-
	
Prompt Manager
	
-
	Add image understanding and generation to ChatGPT

 	
X-LLM [377]
	
ViT-G [378] + Q-Former [7] + Adapter
	
1.8B
	
ChatGLM [379]
	
6B
	
Single-stream
	
Three-stage training
	
MiniGPT-4 data, AISHELL-2, ActivityNet, VSDial-CN (SC)
	
Instruction-following

Comprehensive surveys on VLMs exist, covering early BERT-based VLMs [380, 381] (Section -C1), as well as more recent VLMs with contrastive pretraining [382, 383] (Section -C2). Given the rapid evolution of this field and the emergence of new VLMs based on large language models, commonly known as large multi-modal models (LMMs), we also compile the latest developments of LMMs (Section -C3). To compare the most representative VLMs, we include their specifications in Table VIII.

-C1Self-supervised pretraining

The evolution of the Transformer architecture to accommodate various modalities has given rise to robust multi-modal Transformer models. Initial VLMs based on BERT can be broadly categorized into two types: single-stream and multi-stream [380]. Single-stream models employ a single stack of Transformer blocks to process both visual and linguistic inputs, whereas multi-stream models utilize a separate Transformer stack for each modality, with Transformer cross-attention layers exchanging multi-modal information. To enhance alignment among modalities, these models incorporate various pretraining tasks aimed at absorbing knowledge from out-of-domain data. ViLBERT [228] stands as the pioneer in this line of work, featuring a multi-stream Transformer architecture. Text input undergoes standard processing in the language Transformer; image input is first processed using Faster R-CNN, and the output embeddings of all objects are then passed into the vision stream. The two Transformer outputs—language embeddings and vision embeddings—are combined using a novel co-attention transformer layer. VL-BERT [338] adopts a single-stream multi-modal Transformer, simply concatenating vision and language tokens into a single input sequence. VideoBERT [384] adapts mulit-modal Transformer models to video inputs. UNITER [339] proposes the word-region alignment loss to explicitly align word with image regions. ViLT [340] uses ViT-style [215] image patch projection to embed images, deviating from previous region or grid features.

SimVLM [341] opts for a streamlined approach, relying solely on a single prefix language modeling objective to reduce training costs. VLMo [385] and BEiT-3 [345] both introduce mixture-of-modality-experts Transformers to effectively handle multi-modal inputs.

-C2Contrastive pretraining

Vision-language pretraining in the initial series of BERT-based VLMs has evolved, with refinements such as curating larger-scale pretraining datasets, leveraging multi-modal contrastive learning, and exploring specialized multi-modal architectures. CLIP [25] is one of the earliest attempts in vision-language contrastive pretraining. By contrastive pretraining on a large-scale image-text pair dataset, CLIP exhibits the capability to be transferred to downstream tasks in a zero-shot fashion. In the same line of work, other few-/zero-shot learners have emerged. FILIP [346] concentrates on finer-grained multi-modal interactions with a token-wise contrastive objective. ALIGN [347] focuses on learning from noisy datasets collected without filtering and post-processing. “Locked-image Tuning” (LiT) [386] posits that only training the text model while freezing the image model yields the best results on new tasks. In Frozen [387], the pretrained language model is frozen and a vision encoder is trained to produce image embeddings as a part of language model prompts, exemplifying an instance of prompt tuning.

Unlike the two-tower frameworks of CLIP, FILIP, and ALIGN, which solely train unimodal encoders (image encoder and text encoder), ALBEF [349] additionally incorporates training a multimodal encoder on top of the unimodal encoders, with FLAVA [350] sharing a similar idea. In contrast to contrastive pretraining methods, CoCa [388] seeks to amalgamate the strengths of CLIP’s constrastive learning and SimVLM’s generative objective. Florence [344] generalizes representations from coarse, scene-level to fine, object-level, expands from images to videos, and encompasses modalities beyond RGB channels. OFA [389] draws inspiration from T5 [279] and proposes unifying diverse unimodal and multi-modal tasks under a sequence-to-sequence learning framework.

-C3Large multi-modal model

Large language models (LLMs) encapsulate extensive knowledge, and efforts have been made to transfer this knowledge to multi-modal tasks. Given the resource-intensive nature of fine-tuning entire LLMs on multi-modal tasks due to their large size, various techniques have been explored to effectively connect frozen LLMs with vision encoders, enabling the combined model to acquire multi-modal capabilities. Flamingo [229] connects pretrained vision encoder, NFNet [353], and a large language model, Chinchilla [354], by inserting trainable gated cross-attention layers while keeping the rest of the model frozen. BLIP-2 [7] introduces Q-Former, bootstrapping vision-language representation learning first from a frozen CLIP ViT [25] and then from a frozen LLM, OPT [277] or Flan-T5 [287]. PaLI [356] and PaLI-X [358] investigate the advantages of jointly scaling up the vision and language components using large-scale multilingual image-text data.

Similar to developments in NLP, instruction-following has become a crucial aspect of VLMs, prompting the exploration of various multi-modal instruction-tuning methods. LLaMA-Adapter [361, 362] employs a parameter-efficient finetuning (PEFT) technique, enabling LLaMA [282] to process visual inputs. Kosmos-1 [363] introduces a less restrictive input format that accommodates interleaved image and text. Its Magneto LLM [365] serves as a “general-purpose interface” for docking with perception modules [25]. Kosmos-2 [364] adds additional grounding and referring capabilities. InstructBLIP [366] achieves instruction-following using an instruction-aware Q-Former based on BLIP-2’s Q-Former [7]. Comparable to Kosmos-2, ChatSpot [372] excels at following precise referring instructions, utilizing CLIP ViT [25] and Vicuna [289]. X-LLM [377] converts multi-modality data into LLM inputs using X2L interfaces and treats them as foreign languages, where the X2L interface is inspired by the Q-Former from BLIP-2 [7]. mPLUG-Owl [373, 374] introduces a two-stage training paradigm that establishes a connection between a pretrained LLM with visual encoder and visual abstractor, thereby endowing LLMs with multi-modality abilities. Visual ChatGPT [376] proposes a prompt manager that manages the interaction between ChatGPT and 22 visual foundation models, with the goal of equipping ChatGPT with the capability to understand and generate images.

Rather than employing intricate mechanisms to connect components for different modalities, both LLaVA [8] and MiniGPT-4 [367] propose connecting vision encoders with LLMs through a single linear layer. LLaVA adopts a two-stage instruction-tuning approach, pretraining the CLIP ViT vision encoder [25] in the first stage and finetuning the linear layer and the LLaMA LLM [282] in the second stage. In contrast, MiniGPT-4 freezes both the vision encoder (BLIP-2’s ViT + Q-Former [7]) and the Vicuna LLM [289], only training the linear layer. Following MiniGPT-4, Video-LLaMA [368] handles videos by incorporating two branches for video and audio, each comprising a video/audio encoder and a BLIP-2-style Q-Former [7]. PandaGPT [369] leverages ImageBind [205] to encode vision/text/audio/depth/thermal/IMU data, feeding them to the Vicuna model [289] also through a linear layer. PandaGPT diverges from MiniGPT-4 by using LoRA [390] to train Vicuna alongside the linear layer.

-DSelf-attention

Transformer [4] is built upon the novel Multi-head Self-Attention mechanism:

	
MultiHead
​
(
𝑄
,
𝐾
,
𝑉
)
	
=
Concat
​
(
head
1
,
…
,
head
ℎ
)
​
𝑊
𝑂
	
	
head
𝑖
	
=
Attention
​
(
𝑄
​
𝑊
𝑖
𝑄
,
𝐾
​
𝑊
𝑖
𝐾
,
𝑉
​
𝑊
𝑖
𝑉
)
	
	
Attention
​
(
𝑄
,
𝐾
,
𝑉
)
	
=
softmax
​
(
𝑄
​
𝐾
𝑇
𝑑
𝑘
)
​
𝑉
	

where the trainable weights are 
𝑊
𝑖
𝑄
∈
ℝ
𝑑
model
×
𝑑
𝑘
, 
𝑊
𝑖
𝐾
∈
ℝ
𝑑
model
×
𝑑
𝑘
, 
𝑊
𝑖
𝑉
∈
ℝ
𝑑
model
×
𝑑
𝑣
 and 
𝑊
𝑂
∈
ℝ
ℎ
​
𝑑
𝑣
×
𝑑
model
; 
𝑑
model
, 
𝑑
𝑘
, 
𝑑
𝑣
 are hyperparameters and 
ℎ
 is the number of self-attention heads. Because it is permutation equivariant, positional encodings are injected into the token embeddings.

Two representative lines of Transformers were BERT [29] and GPT [227, 270, 271]. BERT [29] is a deep bidirectional Transformer, which is a stack of Transformer encoder layers:

	
𝑋
	
=
MultiHead
​
(
𝐸
𝑜
​
𝑢
​
𝑡
𝑙
−
1
,
𝐸
𝑜
​
𝑢
​
𝑡
𝑙
−
1
,
𝐸
𝑜
​
𝑢
​
𝑡
𝑙
−
1
)
	
	
𝑋
′
	
=
LayerNorm
​
(
𝑋
+
𝐸
𝑜
​
𝑢
​
𝑡
𝑙
−
1
)
	
	
𝐸
𝑜
​
𝑢
​
𝑡
𝑙
	
=
LayerNorm
​
(
FFN
​
(
𝑋
′
)
+
𝑋
′
)
	

where 
𝐸
𝑜
​
𝑢
​
𝑡
𝑙
 are the encoder output at the 
𝑙
𝑡
​
ℎ
 layer. In BERT pre-training, masked language modeling (MLM) was proposed. It is a self-supervised setting where the model needs to predict the tokens that are masked out (with a probability of 15%) from the remaining tokens.

GPT models [227, 270, 271] are a stack of Transformer decoders:

	
𝑋
	
=
MaskedMultiHead
​
(
𝐷
𝑜
​
𝑢
​
𝑡
𝑙
−
1
,
𝐷
𝑜
​
𝑢
​
𝑡
𝑙
−
1
,
𝐷
𝑜
​
𝑢
​
𝑡
𝑙
−
1
)
	
	
𝑋
′
	
=
LayerNorm
​
(
𝑋
+
𝐷
𝑜
​
𝑢
​
𝑡
𝑙
−
1
)
	
	
𝐷
𝑜
​
𝑢
​
𝑡
𝑙
	
=
LayerNorm
​
(
FFN
​
(
𝑋
′
)
+
𝑋
′
)
	

where 
𝐷
𝑜
​
𝑢
​
𝑡
𝑙
 are the decoder output at the 
𝑙
𝑡
​
ℎ
 layer.

-ESupplementary Related Work

Due to page limitations, we were unable to provide citations for all the mentioned models in the main text. Here is a list of supplementary related works:


Computer Vision:

• 

Mask R-CNN [241]

• 

YOLOv8 [242]

• 

EfficientNet [125]

• 

UNet [248]

• 

VQ-GAN [124]

• 

MoVQ [391]

• 

SigLIP [392]

• 

ViLD [393]

• 

MDETR [394]

• 

HLSM object detector [395]

• 

ResNet [214]

• 

ViT, ViT-B, ViT-L [215, 28]

• 

MAE [396]

• 

ViT-22B [359]

• 

EVA ViT-G/14 [355]

• 

OWL-ViT [397]

• 

ConvNeXt [398]

• 

SAM [216]

• 

Imagen video [399]

• 

C-ViViT [400]

• 

OSRT [401]

• 

PointNet [402]

• 

PointNet++ [403]

• 

3D-CLR [404]

• 

ConceptFusion [405]

Natural Language Processing, LLM, VLM:

• 

Long Short-Term Memory (LSTM) [217]

• 

Transformer [4]

• 

Sentence-BERT / Sent.-BERT [406]

• 

DistilBERT [407]

• 

T5-small, T5-base, T5-XXL [279]

• 

LLaMA [282]

• 

LLaMA 2 [283]

• 

Vicuna-V1.5 13B [289]

• 

MPT [408]

• 

FLAN [286]

• 

Flamingo [229, 409]

• 

Pythia [410]

• 

Phi-2 [411]

• 

GPT-NeoX [412]

• 

Codex [413]

• 

InstructGPT [369]

• 

ChatGPT [1]

• 

GPT-2 [270]

• 

GPT-3 [271]

• 

GPT-4 [272]

• 

PaLI-X [358]

• 

Prismatic-7B VLM [414]

• 

PaliGemma [415]

• 

PaliGemma 2 [416]

• 

Chameleon [417]

• 

Emu3 [418]

• 

LWM-Chat-1M [419]

• 

LLaMA-Adapter [361]

Action Modules:

• 

LingUNet [420]

• 

Latent Motor Plans (LMP) [421]

• 

STORM [422]

• 

PerceiverIO [423]

• 

DD-PPO [424]

• 

ScaleDP [425]

• 

Symbol-tuning [426]

• 

DiT [129]

Others:

• 

DDPM [128]

• 

DDIM [427]

• 

Ravens [428]

• 

RLBench [175]

• 

Something V2: Something-something V2 [429]

Abbreviations:

• 

TFM: transformer

• 

MCIL: multicontext imitation learning

• 

MLE: maximum likelihood estimation

• 

MPC: model predictive control

• 

CVAE: conditional variational autoencoder

• 

EDR: Everyday Robots

-FAdditional VLA-Related Work

We include additional VLA-related work that could not be included in the main manuscript due to the page limit.

-F1Components of VLA: Pretraining

Value-Implicit Pre-training (VIP) [27] capitalizes on the temporal sequences present in videos, distinguishing itself from R3M [26] by attracting both the initial and target frames while simultaneously repelling successive frames. This objective aims to capture long-range temporal relationships and uphold local smoothness. While their own experiments demonstrate superior performance compared to R3M in specific tasks, subsequent research presents conflicting findings through more comprehensive evaluations [34].

SpawnNet [430] utilizes a two-stream architecture incorporating adapter layers to fuse features from both a pretrained vision encoder and features learned from scratch. This innovative approach eliminates the necessity of training the pretrained vision encoders while surpassing the performance of parameter-efficient finetuning (PEFT) methods, as evidenced by experimental results in robot manipulation tasks.

Holo-Dex [431] proposes learning low-dimensional representations for visual policies using self-supervised learning (SSL). In Dobb
⋅
E [432], SSL is also explored for 3D visual inputs, through a method termed “Home Pretrained Representations,” using data collected from 22 homes in NYC. The authors extend SSL-pretrained representations beyond visual inputs. AuRL [47] introduces learning dynamic behaviors from audio, an often-overlooked yet important information source. Additionally, visual inputs were found to be insufficient for multi-fingered robotic hands, leading to T-Dex [207], a tactile-based dexterous policy.

-F2Components of VLA: World Models

Masked World Model (MWM) [433] innovates by modifying the vision encoder of DreamerV2 to a hybrid composition of convolutional neural network and vision Transformer. In training this novel Convolutional-ViT vision encoder, MWM draws inspiration from the approach proposed in MAE [396]. Through the incorporation of an auxiliary reward prediction loss, the resulting learned latent dynamics model exhibits a notable performance improvement across various visual robotic tasks.

SWIM [434] advocates for the use of human videos in training a world model due to the availability of large-scale human-centric data. However, acknowledging the substantial gap between human and robot data, SWIM addresses this disparity by grounding actions in visual affordance maps. This approach involves inferring target poses based on the affordance maps, facilitating an effective transfer of knowledge from human data to enhance robot control.

Iso-Dream [435] introduces two key enhancements to the Dreamer framework. The first enhancement focuses on optimizing inverse dynamics by decoupling controllable and noncontrollable dynamics. The second enhancement involves optimizing agent behavior based on the decoupled latent imaginations. This approach is particularly advantageous, as the noncontrollable state transition branch can be rolled out independently of actions, yielding benefits for long-horizon decision-making processes.

Transformer-based World Model (TWM) [60] shares the same motivation as Dreamer but adopts a different approach by constructing a world model based on the Transformer-XL architecture [436]. This Transformer-based world model trains a model-free agent in latent imagination by generating new trajectories. Additionally, TWM suggests a modification to the balanced KL divergence loss from DreamerV2 and introduces a novel thresholded entropy loss tailored for the advantage actor-critic framework.

SceneScript [437] introduces a set of structured language commands that can define an entire scene, specifying the layout and objects. This language-based scene representation distinguishes itself from previous methods—such as meshes, voxel grids, point clouds, or radiance fields—by generating scenes in an autoregressive, token-based fashion. It achieves competitive results against state-of-the-art methods in layout estimation and object detection.

-F3Components of VLA: Imitation Learning

BeT [438], an imitation learning variant of the Trajectory Transformer, addresses the noisy and multimodal nature of non-expert human demonstration data by using k-means-based action discretization coupled with continuous action correction. C-BeT [439] extends BeT by adding a target frame or demonstration as the goal specification. VQ-BeT [440] further improves BeT by incorporating vector quantization instead of k-means, enabling better modeling of long-range actions.

-F4Latest Developments of VLA

LLaRA [441] introduces a data-centric method that uses a Vision-Language Model to automatically relabel existing robot demonstration videos with rich, descriptive language, significantly boosting the performance of downstream vision-language policies.

𝜋
0
 undergoes further enhancements, including FAST [442] and 
𝜋
0.5
. FAST is a novel robot action tokenization method designed to prioritize dexterous skills. 
𝜋
0.5
 [443] pushes the boundaries of in-the-wild generalization by incorporating co-training with diverse data sources.

Mobility VLA [444] proposes a hierarchical policy that integrates the commonsense reasoning capabilities of VLMs with a topological-graph-based low-level navigation policy.

While many VLAs focus primarily on robotic arms, recent advancements have extended their application to more complex embodiments, including humanoid and quadruped robots.

Humanoid Robot:

• 

GR00T N1 [445]

• 

Humanoid-VLA [446]

Quadruped Robot:

• 

QUAR-VLA [447]

• 

QUART-Online [448]

• 

MoRE [449]

Additionally, some VLAs are designed to function as virtual assistants capable of taking actions within operating systems, video games, and other environments. Notable examples include:

• 

CombatVLA [450]

• 

JARVIS-VLA [451]

Report Issue
Report Issue for Selection
Generated by L A T E xml 
Instructions for reporting errors

We are continuing to improve HTML versions of papers, and your feedback helps enhance accessibility and mobile support. To report errors in the HTML that will help us improve conversion and rendering, choose any of the methods listed below:

Click the "Report Issue" button.
Open a report feedback form via keyboard, use "Ctrl + ?".
Make a text selection and click the "Report Issue for Selection" button near your cursor.
You can use Alt+Y to toggle on and Alt+Shift+Y to toggle off accessible reporting links at each section.

Our team has already identified the following issues. We appreciate your time reviewing and reporting rendering errors we may not have found yet. Your efforts will help us improve the HTML versions for all readers, because disability should not be a barrier to accessing research. Thank you for your continued support in championing open access for all.

Have a free development cycle? Help support accessibility at arXiv! Our collaborators at LaTeXML maintain a list of packages that need conversion, and welcome developer contributions.

