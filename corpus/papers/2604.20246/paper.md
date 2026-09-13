Title: Cortex 2.0: Grounding World Models in Real-World Industrial Deployment

URL Source: https://arxiv.org/html/2604.20246

Published Time: Thu, 23 Apr 2026 00:32:10 GMT

Markdown Content:
Adriana Aida Walida Amer Katarina Bankovic Dhruv Behl Fabian Busch Annie Bhalla Minh Duong Florian Gienger Rohan Godse Denis Grachev Ralf Gulde Elisa Hagensieker Junpeng Hu Shivam Joshi Tobias Knoblauch Likith Kumar Damien LaRocque Keerthana Lokesh Omar Moured Khiem Nguyen Christian Preyss Ranjith Sriganesan Vikram Singh Carsten Sponner Anh Tong Dominik Tuscher Marc Tuscher Pavan Upputuri Sereact GmbH Authors listed in alphabetical order.

###### Abstract

Industrial robotic manipulation demands reliable long-horizon execution across embodiments, tasks, and changing object distributions. While Vision-Language-Action models have demonstrated strong generalization, they remain fundamentally reactive. By optimizing the next action given the current observation without evaluating potential futures, they are brittle to the compounding failure modes of long-horizon tasks. Cortex 2.0 shifts from reactive control to plan-and-act by generating candidate future trajectories in visual latent space, scoring them for expected success and efficiency, then committing only to the highest-scoring candidate. We evaluate Cortex 2.0 on a single-arm and dual-arm manipulation platform across four tasks of increasing complexity: pick and place items, item and trash sorting, screw sorting, and shoebox unpacking. Cortex 2.0 consistently outperforms state-of-the-art Vision-Language-Action baselines, achieving the best results across all tasks. The system remains reliable in unstructured environments characterized by heavy clutter, frequent occlusions, and contact-rich manipulation, where reactive policies fail. These results demonstrate that our world-model-based planning can operate reliably in complex industrial environments.

![Image 1: Refer to caption](https://arxiv.org/html/2604.20246v1/graphics/intro_arch_upper.jpg)

Figure 1: Overview of Cortex 2.0

## 1 Introduction

Reliable robotic manipulation at industrial scale requires more than generalization. Actions are irreversible, failures compound over long horizons, and a single unrecovered error can disrupt an entire production workflow. While recent Vision–Language–Action (VLA) models[[6](https://arxiv.org/html/2604.20246#bib.bib1 "π0: A Vision-Language-Action Flow Model for General Robot Control"), [21](https://arxiv.org/html/2604.20246#bib.bib2 "π0.5: A Vision-Language-Action Model with Open-World Generalization"), [7](https://arxiv.org/html/2604.20246#bib.bib3 "RT-1: Robotics Transformer for Real-World Control at Scale")] have demonstrated strong generalization across tasks and embodiments, they remain reactive by design: each action is selected conditioned on the current observation, without explicit reasoning about future outcomes.

Cortex 2.0 extends our preceding VLA model, Cortex, with a world-model-based planning module. At each decision step, a world model generates a set of candidate future trajectories in visual latent space. The Process-Reward Operator (PRO), our dense reward module, scores each candidate for task progress, risk likelihood, and completion likelihood, and the policy commits to the highest-scoring trajectory. Before action, the system evaluates potential futures rather than committing to the first available action.

##### Background and related progress.

Large-scale robot demonstration data and cross-embodiment corpora have driven great progress in manipulation policies[[42](https://arxiv.org/html/2604.20246#bib.bib4 "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control"), [14](https://arxiv.org/html/2604.20246#bib.bib5 "PaLM-E: An Embodied Multimodal Language Model"), [34](https://arxiv.org/html/2604.20246#bib.bib6 "Open X-Embodiment: Robotic Learning Datasets and RT-X Models"), [15](https://arxiv.org/html/2604.20246#bib.bib7 "Bridge Data: Boosting Generalization of Robotic Skills with Cross-Domain Datasets")]. Sereact’s operating fleet has accumulated large-scale manipulation data across warehouse deployments, providing training data that reflects real industrial conditions including edge cases and failure modes that are difficult to reproduce in controlled data collection. Alongside data scaling, world models have emerged as a promising approach: UniSim[[39](https://arxiv.org/html/2604.20246#bib.bib8 "Learning Interactive Real-World Simulators")] and Cosmos[[1](https://arxiv.org/html/2604.20246#bib.bib9 "Cosmos World Foundation Model Platform for Physical AI")] have shown that models trained on internet-scale video acquire broad physical priors transferable to robotic settings. Cortex 2.0 builds on this direction, grounding world model training in deployment data collected continuously from live operations.

##### Industrial setting.

Warehouse manipulation involves frequent occlusions from totes and packaging, reflective and translucent surfaces that challenge RGB-based perception, and rapid object distribution shift across shifts and sites. In tasks such as returns handling, failure modes including gradual slip, jams, and collisions emerge only after several steps. World-model-based planning addresses this: by scoring candidate futures before execution, the system can identify and avoid problematic branches prior to commitment.

Our main contributions are:

1.   1.
World-model augmented VLA: we integrate a visual-latent world model into Cortex, enabling k-step lookahead planning.

2.   2.
PRO scoring module: we introduce a multi-criteria scoring function that evaluates candidate rollouts for task progress, completion likelihood, and risk likelihood. It thereby derives an advantage signal that conditions the action heads.

3.   3.
Cross-embodiment planning: because planning operates in visual space, the same planning loop transfers across single-arm, dual-arm, and humanoid embodiments.

4.   4.
Benchmark evaluation: we benchmark Cortex 2.0 on four real-world tasks of increasing complexity, achieving highest success rates across all tasks with zero human interventions.

## 2 Related Works

### 2.1 Vision–Language–Action Models

Since RT-1[[7](https://arxiv.org/html/2604.20246#bib.bib3 "RT-1: Robotics Transformer for Real-World Control at Scale")] and RT-2[[42](https://arxiv.org/html/2604.20246#bib.bib4 "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control")], large sequence models have served as generalist robot policies by mapping camera observations and language prompts to actions. RT-2 demonstrated that pretraining on internet-scale vision–language data transfers semantic knowledge to robotic control, establishing the VLA template that subsequent work has built upon. Generalist policies such as Octo[[32](https://arxiv.org/html/2604.20246#bib.bib10 "Octo: An Open-Source Generalist Robot Policy")] and OpenVLA[[24](https://arxiv.org/html/2604.20246#bib.bib11 "OpenVLA: An Open-Source Vision-Language-Action Model")] scaled this template to cross-embodiment pretraining; OpenVLA-OFT[[23](https://arxiv.org/html/2604.20246#bib.bib12 "Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success")] further showed that parallel decoding and action chunking substantially improve inference speed without sacrificing task performance. PaLM-E[[14](https://arxiv.org/html/2604.20246#bib.bib5 "PaLM-E: An Embodied Multimodal Language Model")] further showed that embodied multimodal language models can ground high-level reasoning in physical scenes.

Recent systems increasingly adopt hierarchical designs that predict mid-level subtask tokens before executing fine-grained control[[6](https://arxiv.org/html/2604.20246#bib.bib1 "π0: A Vision-Language-Action Flow Model for General Robot Control"), [21](https://arxiv.org/html/2604.20246#bib.bib2 "π0.5: A Vision-Language-Action Model with Open-World Generalization"), [25](https://arxiv.org/html/2604.20246#bib.bib13 "MolmoAct: Action Reasoning Models That Can Reason in Space"), [31](https://arxiv.org/html/2604.20246#bib.bib14 "F1: A Vision-Language-Action Model Bridging Understanding and Generation to Actions")]. $\pi_{0}$[[6](https://arxiv.org/html/2604.20246#bib.bib1 "π0: A Vision-Language-Action Flow Model for General Robot Control")] instantiated a flow-matching action expert on top of a VLM backbone and demonstrated strong dexterous manipulation across diverse platforms. $\pi_{0.5}$[[21](https://arxiv.org/html/2604.20246#bib.bib2 "π0.5: A Vision-Language-Action Model with Open-World Generalization")] extended this by co-training discrete action tokens with web and language data, improving generalization to unseen environments. FAST[[35](https://arxiv.org/html/2604.20246#bib.bib15 "FAST: Efficient Action Tokenization for Vision-Language-Action Models")] introduced efficient action tokenization for high-frequency autoregressive control, and GR00T N1[[5](https://arxiv.org/html/2604.20246#bib.bib16 "GR00T N1: An Open Foundation Model for Generalist Humanoid Robots")] demonstrated hierarchical dual-system VLA architectures for humanoid platforms. The preceding Cortex system [[37](https://arxiv.org/html/2604.20246#bib.bib17 "Cortex: Bridging Vision, Language, and Action with Discrete Plans and Tokens")] introduced a three-level VLA design for industrial settings, combining the Sereact Lens VLM for subtask prediction and pixel-level grounding with flow-matching action heads, and demonstrated strong performance on warehouse pick-and-place and returns handling tasks. Cortex 2.0 extends this by augmenting the reactive policy with world-model-based future predictions, moving from pattern-matched responses to informed plan selection.

### 2.2 Flow Matching for Robot Control

Diffusion policies[[12](https://arxiv.org/html/2604.20246#bib.bib18 "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"), [20](https://arxiv.org/html/2604.20246#bib.bib19 "Denoising Diffusion Probabilistic Models")] model action generation as a denoising process, generating expressive multimodal policies for manipulation. Flow matching[[28](https://arxiv.org/html/2604.20246#bib.bib20 "Flow Matching for Generative Modeling"), [3](https://arxiv.org/html/2604.20246#bib.bib21 "Building Normalizing Flows with Stochastic Interpolants"), [2](https://arxiv.org/html/2604.20246#bib.bib22 "Stochastic Interpolants: A Unifying Framework for Flows and Diffusions")] improves on this by learning straight-line interpolations between noise and data, reducing the number of inference steps required. $\pi_{0}$[[6](https://arxiv.org/html/2604.20246#bib.bib1 "π0: A Vision-Language-Action Flow Model for General Robot Control")] was the first large-scale VLA to adopt flow matching for action generation, with $\pi_{0.5}$[[21](https://arxiv.org/html/2604.20246#bib.bib2 "π0.5: A Vision-Language-Action Model with Open-World Generalization")] and RDT-2[[29](https://arxiv.org/html/2604.20246#bib.bib23 "RDT2: Exploring the Scaling Limit of UMI Data Towards Zero-Shot Cross-Embodiment Generalization")] further validating its advantages in latency and trajectory quality in diverse bimanual tasks.

### 2.3 World Models for Robotics

World models are predictive models of environment dynamics with a long history in model-based RL. Early work[[17](https://arxiv.org/html/2604.20246#bib.bib24 "World Models"), [16](https://arxiv.org/html/2604.20246#bib.bib25 "Deep Visual Foresight for Planning Robot Motion")] formalized world models for policy learning; Dreamer[[18](https://arxiv.org/html/2604.20246#bib.bib26 "Mastering Atari with Discrete World Models"), [19](https://arxiv.org/html/2604.20246#bib.bib27 "Mastering Diverse Domains Through World Models")] established that latent imagination can match model-free approaches on visual control tasks, though such approaches use the world model as a training-time rollout generator, which risks compounding model errors.

At internet scale, UniSim[[39](https://arxiv.org/html/2604.20246#bib.bib8 "Learning Interactive Real-World Simulators")] and Cosmos[[1](https://arxiv.org/html/2604.20246#bib.bib9 "Cosmos World Foundation Model Platform for Physical AI")] demonstrated that world models pretrained on large-scale video acquire broad physical priors transferable to robotic settings. Several concurrent works have explored using such models at inference time: IRASim[[41](https://arxiv.org/html/2604.20246#bib.bib28 "IRASim: A Fine-Grained World Model for Robot Manipulation")] and GPC[[36](https://arxiv.org/html/2604.20246#bib.bib29 "Strengthening Generative Robot Policies Through Predictive World Modeling")] showed that scoring candidate rollouts before execution improves task success over reactive policies, while GR-2[[11](https://arxiv.org/html/2604.20246#bib.bib30 "GR-2: A Generative Video-Language-Action Model with Web-Scale Knowledge for Robot Manipulation")] and V-JEPA 2[[4](https://arxiv.org/html/2604.20246#bib.bib31 "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning")] validated that joint pretraining on internet video and robot data supports strong physical reasoning with limited robot-specific supervision. Li et al.[[27](https://arxiv.org/html/2604.20246#bib.bib32 "Robotic World Model: A Neural Network Simulator for Robust Policy Optimization in the Real World")] further demonstrated this direction on deployment data. Cortex 2.0 builds on these findings by grounding world model training in continuously collected operational data and scoring imagined rollouts via PRO before any action is executed.

Cortex 2.0 follows this direction: the world model is pretrained on internet-scale video and fine-tuned on deployment recordings at $30 ​ \text{Hz}$. Central to this design is that imagined futures remain grounded in the same representational space as real observations, so that PRO’s scoring function, learned on real executed trajectories, transfers directly to imagined futures.

### 2.4 Force Feedback and Multimodal Sensing

RGB-only policies are inherently limited in scenarios involving contact, deformation, and occlusion. Force and tactile sensing provide complementary signals that are invisible to cameras: contact forces reveal grasp stability, torque profiles encode object compliance, and vacuum pressure margins indicate suction reliability. Early work demonstrated that force–torque signals enable more robust grasping under uncertainty[[10](https://arxiv.org/html/2604.20246#bib.bib33 "More Than a Feeling: Learning to Grasp and Regrasp Using Vision and Touch")], and subsequent systems have shown that fusing vision and touch improves performance on contact-rich tasks[[26](https://arxiv.org/html/2604.20246#bib.bib34 "Making Sense of Vision and Touch: Learning Multimodal Representations for Contact-Rich Tasks")].

In Cortex 2.0, force feedback is an optional input used only when the robot supports it. When available, the robot state $r_{t}$ and force feedback $f_{t}$ are added to the multimodal observation $o_{t}$ (Eq.[1](https://arxiv.org/html/2604.20246#S3.E1 "Equation 1 ‣ 3.1 Overview ‣ 3 Methodology ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment")) alongside RGB and task instruction embeddings. This design allows Cortex 2.0 to deploy across platforms with heterogeneous sensing configurations: contact-rich tasks such as screw sorting or kitting benefit from force-aware observations. Suction-based tasks benefit from vacuum pressure signals that reveal grasp reliability not visible to cameras.

### 2.5 Datasets for Robot Learning

Scale and diversity are crucial for generalization and cross-embodiment transfer. Bridge and BridgeData V2 study cross-domain mixtures for manipulation transfer [[15](https://arxiv.org/html/2604.20246#bib.bib7 "Bridge Data: Boosting Generalization of Robotic Skills with Cross-Domain Datasets"), [38](https://arxiv.org/html/2604.20246#bib.bib35 "BridgeData V2: A Dataset for Robot Learning at Scale")]. Open X-Embodiment unifies demonstrations across many labs and robot embodiments to enable training of generalist policies [[34](https://arxiv.org/html/2604.20246#bib.bib6 "Open X-Embodiment: Robotic Learning Datasets and RT-X Models")]. DROID and Agibot World further enlarge the distribution toward in-the-wild and large-scale manipulation [[22](https://arxiv.org/html/2604.20246#bib.bib36 "DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset"), [8](https://arxiv.org/html/2604.20246#bib.bib37 "AgiBot World Colosseo: A Large-Scale Manipulation Platform for Scalable and Intelligent Embodied Systems")]. Domain-specific corpora such as Stanford Kuka and Berkeley cable routing provide long-horizon and contact-rich tasks [[26](https://arxiv.org/html/2604.20246#bib.bib34 "Making Sense of Vision and Touch: Learning Multimodal Representations for Contact-Rich Tasks"), [30](https://arxiv.org/html/2604.20246#bib.bib38 "Multistage Cable Routing Through Hierarchical Imitation Learning")]. Despite breadth, few public datasets capture industrial warehouse conditions; many production datasets remain proprietary.

##### Positioning.

Our approach builds on the VLA model with flow-matching action heads. We augment our previous reactive policy with a visual-latent world model that generates $k$ candidate futures at inference time, scored by our Process-Reward Operator (PRO) for task progress, risk, and completion likelihood before any action is executed. In addition to public datasets, training incorporates a proprietary dataset of deployment recordings collected continuously across our deployments.

## 3 Methodology

### 3.1 Overview

Cortex 2.0 is Sereact’s general-purpose vision–language–action (VLA) model, validated in industrial applications such as pick and place, returns handling, and kitting. It unifies perception, planning, reasoning, and control in a four-level hierarchical design: a high-level VLM observes and encodes the scene; a world model generates candidate futures; PRO evaluates and ranks them; and flow-based action heads commit to the highest-scoring trajectory (Figure[2](https://arxiv.org/html/2604.20246#S3.F2 "Figure 2 ‣ 3.1 Overview ‣ 3 Methodology ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment")).

![Image 2: Refer to caption](https://arxiv.org/html/2604.20246v1/graphics/cortex2.0-2.jpg)

Figure 2: Cortex 2.0 Architecture

We consider a partially observed control problem with discrete time $t$ = 1,…,$T$. At each step, the robot receives multimodal observations

$o_{t} = \left(\right. I_{t}^{r ​ g ​ b} , r_{t} , f_{t} , l_{t} \left.\right) ,$(1)

where $I_{t}^{r ​ g ​ b}$ is the RGB image from wrist cameras, $r_{t}$ is the robot state, $f_{t}$ the force feedback of the end-effector, and $l_{t}$ the embedding of the task instruction. The observation is encoded into a visual latent $z_{t} = f_{\text{enc}} ​ \left(\right. o_{t} \left.\right)$, where $f_{\text{enc}}$ is the VLM visual encoder.

A high-level VLM produces structured task context $s_{t}$ from $z_{t}$. The world model generates $k$ future candidate trajectories over planning horizon $H_{\text{wm}}$, which the PRO module scores and ranks. The selected plan $\tau^{*}$ and its binarized advantage indicator $I_{t} \in \left{\right. 0 , 1 \left.\right}$ condition the policy $\pi_{\theta}$, which produces an action chunk over execution horizon $H_{\text{act}}$:

$\pi_{\theta} ​ \left(\right. a_{t : t + H_{\text{act}} - 1} \mid z_{t} , s_{t} , \tau^{*} , I_{t} \left.\right) .$(2)

Formally, at each decision step $t$ the system solves:

$\tau^{*} = arg ⁡ \underset{\tau_{j} \in \left{\right. \tau_{1} , \ldots , \tau_{k} \left.\right}}{max} ⁡ S_{j} ​ \left(\right. z_{t} , s_{t} \left.\right) ,$(3)

where $S_{j} \equiv S ​ \left(\right. \tau_{j} \left.\right)$ is the PRO score of rollout $\tau_{j}$. The advantage of the selected rollout relative to the average candidate score is defined as:

$\Delta^{*} = S ​ \left(\right. \tau^{*} \left.\right) - \frac{1}{k} ​ \sum_{j = 1}^{k} S ​ \left(\right. \tau_{j} \left.\right) ,$(4)

and binarized via a task-dependent threshold $\epsilon ​ \left(\right. s_{t} \left.\right)$ into an indicator $I_{t} = 𝟏 ​ \left[\right. \Delta^{*} > \epsilon ​ \left(\right. s_{t} \left.\right) \left]\right.$, which conditions the VLA policy.

Given demonstration trajectories $\mathcal{D} = \left{\right. \left(\right. o_{1 : T} , a_{1 : T} \left.\right) \left.\right}$, training jointly optimizes:

$\mathcal{L}_{\text{total}} ​ \left(\right. \theta \left.\right) = \mathcal{L}_{\text{FM}} ​ \left(\right. \theta \left.\right) + \lambda_{\text{wm}} ​ \mathcal{L}_{\text{WM}} ,$(5)

where $\mathcal{L}_{\text{FM}}$ is the flow-matching action loss and $\mathcal{L}_{\text{WM}}$ the world model loss. PRO is pretrained separately on industrial deployment data and kept frozen during this stage.

Cortex 2.0 is trained on open-source multimodal datasets, Sereact’s teleoperation and deployment data, and synthetic data, enabling the model to generalize across embodiments and task families under real-world conditions.

### 3.2 Cortex 2.0 Architecture

#### 3.2.1 High-Level VLM

The high-level VLM encodes the current observation into a structured task context that mediates between perception, planning, and control. Given latent $z_{t}$, it produces:

$s_{t} = f_{hl - VLM} ​ \left(\right. z_{t} \left.\right) ,$(6)

where $s_{t}$ is a learned task-conditioned embedding that encodes a subgoal-level decomposition of the task and grounding variables aligning language with scene entities and spatial constraints, including objects, spatial relations, and contact priors. This representation steers the world model toward task-relevant futures and the execution stack toward realizing the selected future as action.

#### 3.2.2 PRO: Process-Reward Operator

PRO is our dense reward model operating over executed trajectories from real deployment data. Cortex 2.0 now lifts PRO into the planning loop: PRO heads operate on the visual latents $z_{t + 1 : t + H_{\text{wm}}}^{\left(\right. j \left.\right)}$ predicted by the world model (Section[3.2.3](https://arxiv.org/html/2604.20246#S3.SS2.SSS3 "3.2.3 World Model ‣ 3.2 Cortex 2.0 Architecture ‣ 3 Methodology ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment")), scoring imagined rollouts before any action is executed.

For each candidate rollout $j$, PRO consumes the sequence of predicted latents

$\left(\hat{\tau}\right)_{j} = \left(\right. z_{t + 1}^{\left(\right. j \left.\right)} , z_{t + 2}^{\left(\right. j \left.\right)} , \ldots , z_{t + H_{\text{wm}}}^{\left(\right. j \left.\right)} \left.\right) .$(7)

A temporal model processes this latent sequence to produce a rollout-level representation $p_{j}$ that captures both the predicted final state and the quality of the trajectory leading to it. From $p_{j}$, three prediction heads operate.

##### Progress.

The progress head estimates how much closer the predicted future brings the system to successful task completion:

$\Delta ​ p^{\left(\right. j \left.\right)} = V_{\phi} ​ \left(\right. z_{t + H_{\text{wm}}}^{\left(\right. j \left.\right)} \left.\right) - V_{\phi} ​ \left(\right. z_{t} \left.\right) ,$(8)

where $V_{\phi}$ is a value function learned over visual latents from executed trajectories with known outcomes.

##### Risk.

The risk head predicts the probability of a failure event occurring along the imagined trajectory:

$\rho^{\left(\right. j \left.\right)} = P_{\phi} ​ \left(\right. \text{fail} = 1 \mid \left(\hat{\tau}\right)_{j} \left.\right) ,$(9)

penalizing rollouts that pass through latent states associated with high-speed contact, compression, edge impacts, or surface scraping, even if the item ultimately reaches the goal.

##### Termination.

The termination head predicts the likelihood that the imagined trajectory leads to successful task completion:

$d^{\left(\right. j \left.\right)} = P_{\phi} ​ \left(\right. \text{success} = 1 \mid \left(\hat{\tau}\right)_{j} \left.\right) .$(10)

PRO aggregates these three signals into a composite rollout score:

$S_{j} = \Delta ​ p^{\left(\right. j \left.\right)} - \lambda ​ \rho^{\left(\right. j \left.\right)} + \beta ​ d^{\left(\right. j \left.\right)} ,$(11)

where $\lambda$ controls risk sensitivity and $\beta$ weights completion likelihood. Figure[3](https://arxiv.org/html/2604.20246#S3.F3 "Figure 3 ‣ Termination. ‣ 3.2.2 PRO: Process-Reward Operator ‣ 3.2 Cortex 2.0 Architecture ‣ 3 Methodology ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment") illustrates the full scoring and selection process across $k$ candidate rollouts.

As illustrated in Figure[3](https://arxiv.org/html/2604.20246#S3.F3 "Figure 3 ‣ Termination. ‣ 3.2.2 PRO: Process-Reward Operator ‣ 3.2 Cortex 2.0 Architecture ‣ 3 Methodology ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"), the best rollout is then selected as:

$z_{*}^{\text{token}} = arg ⁡ \underset{j}{max} ⁡ S_{j} .$(12)

The advantage $\Delta^{*}$ (Eq.[4](https://arxiv.org/html/2604.20246#S3.E4 "Equation 4 ‣ 3.1 Overview ‣ 3 Methodology ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment")) is binarized via threshold $\epsilon ​ \left(\right. s_{t} \left.\right)$ into indicator $I_{t} = 𝟏 ​ \left[\right. \Delta^{*} > \epsilon ​ \left(\right. s_{t} \left.\right) \left]\right.$, which is passed to the VLA policy. At inference, $I_{t}$ is fixed to $1$, biasing the VLA to generate actions that best realize the selected future.

![Image 3: Refer to caption](https://arxiv.org/html/2604.20246v1/graphics/rollout_pink.jpg)

Figure 3: PRO scores $k$ candidate rollouts via the composite score $S_{j}$ (Eq.[11](https://arxiv.org/html/2604.20246#S3.E11 "Equation 11 ‣ Termination. ‣ 3.2.2 PRO: Process-Reward Operator ‣ 3.2 Cortex 2.0 Architecture ‣ 3 Methodology ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment")). The loss landscape shows all candidate trajectories (top); PRO selects the highest-scoring rollout $\tau^{*}$ (bottom). 

The PRO heads are trained on real executed trajectories from deployment data, where ground-truth outcomes are available, and applied at inference time to imagined latents from the world model. PRO is pretrained and kept frozen during world model and policy training. This transfer is enabled by the world model producing latents in the same visual latent space as real observations.

#### 3.2.3 World Model

The world model $f_{\varphi}$ learns predictive dynamics in visual latent space via flow matching[[28](https://arxiv.org/html/2604.20246#bib.bib20 "Flow Matching for Generative Modeling")]. Conditioned on the current latent $z_{t}$ and task context $s_{t}$, it generates $k$ candidate future latent sequences, each initialized from a distinct noise realization $\xi^{\left(\right. j \left.\right)} sim \mathcal{N} ​ \left(\right. 0 , I \left.\right)$, which PRO then scores to select the highest-quality trajectory.

##### Training.

For each ground-truth future latent $z_{t + h}$ at step $h \in \left{\right. 1 , \ldots , H_{\text{wm}} \left.\right}$, conditioned on $z_{t}$ and $s_{t}$, we sample flow time $\sigma sim Beta ​ \left(\right. \alpha , \beta \left.\right)$ with $\beta \gg \alpha$, and noise $\xi^{\left(\right. h \left.\right)} sim \mathcal{N} ​ \left(\right. 0 , I \left.\right)$ independently per step, forming the interpolation:

$\left(\overset{\sim}{z}\right)_{\sigma}^{\left(\right. h \left.\right)} = \sigma ​ z_{t + h} + \left(\right. 1 - \sigma \left.\right) ​ \xi^{\left(\right. h \left.\right)} , v^{\left(\right. h \left.\right)} = z_{t + h} - \xi^{\left(\right. h \left.\right)} .$(13)

Choosing $\beta \gg \alpha$ biases sampling toward higher noise levels, which allows reliable trajectory scoring with fewer ODE integration steps at inference. Since PRO operates on motion and physical plausibility rather than rendering fidelity, coarse latent reconstruction are sufficient to distinguish good from bad trajectories. The world model is trained with the flow-matching objective:

$\mathcal{L}_{\text{WM}} ​ \left(\right. \varphi \left.\right) = \mathbb{E}_{h , \sigma , \xi^{\left(\right. h \left.\right)}} ​ \left(\parallel g_{\varphi} ​ \left(\right. \left(\overset{\sim}{z}\right)_{\sigma}^{\left(\right. h \left.\right)} , \sigma , z_{t} , s_{t} \left.\right) - v^{\left(\right. h \left.\right)} \parallel\right)_{2}^{2} .$(14)

##### Inference.

For each candidate $j = 1 , \ldots , k$, future latents are generated. A noise realization $\xi^{\left(\right. j \left.\right)} sim \mathcal{N} ​ \left(\right. 0 , I \left.\right)$ is sampled and the ODE is integrated from $\sigma = 0$ to $\sigma = 1$:

$\left(\overset{\sim}{z}\right)_{\sigma + \Delta ​ \sigma}^{\left(\right. j \left.\right)} = \left(\overset{\sim}{z}\right)_{\sigma}^{\left(\right. j \left.\right)} + \Delta ​ \sigma \cdot g_{\varphi} ​ \left(\right. \left(\overset{\sim}{z}\right)_{\sigma}^{\left(\right. j \left.\right)} , \sigma , z_{t} , s_{t} \left.\right) ,$(15)

yielding the full candidate sequence $z_{t + 1 : t + H_{\text{wm}}}^{\left(\right. j \left.\right)} = \left(\overset{\sim}{z}\right)_{\sigma = 1}^{\left(\right. j \left.\right)}$, which is passed to PRO. Figure[4](https://arxiv.org/html/2604.20246#S3.F4 "Figure 4 ‣ Inference. ‣ 3.2.3 World Model ‣ 3.2 Cortex 2.0 Architecture ‣ 3 Methodology ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment") illustrates the generation of 30 future candidate sequences.

![Image 4: Refer to caption](https://arxiv.org/html/2604.20246v1/graphics/matrix2.jpg)

Figure 4: World Model Rollout Generation. Each column of the $6 \times 5$ grid shows one of $k = 6$ candidate trajectories rolled out over horizon $H_{\text{wm}} = 5$. The rollouts share the same current latent $z_{t}$ and task context $s_{t}$, but each is generated from an independent noise realization $\xi^{\left(\right. j \left.\right)} sim \mathcal{N} ​ \left(\right. 0 , I \left.\right)$. PRO scores each rollout and selects the highest-ranked candidate $\tau^{*}$ for execution.

The world model is pretrained on internet-scale video data and fine-tuned on deployment recordings at $30 ​ Hz$. The number of ODE integration steps controls the accuracy of the numerical integration, trading inference speed against trajectory fidelity.

### 3.3 VLA Policy

The VLA policy receives the task context $s_{t}$, the current visual latent $z_{t}$, the selected world model rollout $z_{*}^{\text{token}}$, and the advantage signal $I_{t} \in \left{\right. 0 , 1 \left.\right}$ from PRO, and generates a continuous action chunk. The policy is implemented as a 2B-VLM with a flow-matching action head.

The selected rollout token and advantage indicator are projected into the VLM feature space and concatenated with the task context:

$c_{t} = \left[\right. s_{t} ; z_{t} ; W_{z} ​ z_{*}^{\text{token}} ; W_{I} ​ I_{t} \left]\right. ,$(16)

where $W_{z} \in \mathbb{R}^{d \times d_{z}}$ and $W_{I} \in \mathbb{R}^{d \times 1}$ are learned projections. This conditioning is processed through the full depth of the VLM transformer.

##### Flow-Matching Action Head.

Given a ground-truth action chunk $x \in \mathbb{R}^{H_{\text{act}} \times C}$[[40](https://arxiv.org/html/2604.20246#bib.bib39 "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware")] and conditioning $c_{t}$, we sample $\nu sim Beta ​ \left(\right. \alpha , \beta \left.\right)$ with $\alpha = 1.0$, $\beta = 1.5$ (rescaled via $\nu \leftarrow 0.001 + 0.999 ​ \nu$) and noise $\epsilon sim \mathcal{N} ​ \left(\right. 0 , I \left.\right)$. The interpolation and target velocity are:

$x_{\nu} = \nu ​ x + \left(\right. 1 - \nu \left.\right) ​ \epsilon , u_{\nu} = x - \epsilon .$(17)

The action head predicts velocity $v_{\theta} ​ \left(\right. x_{\nu} , \nu , c_{t} \left.\right)$, trained with:

$\mathcal{L}_{FM} ​ \left(\right. \theta \left.\right) = \mathbb{E}_{\left(\right. x , c_{t} \left.\right) sim \mathcal{D}} ​ \mathbb{E}_{\nu sim \rho} ​ \mathbb{E}_{\epsilon sim \mathcal{N} ​ \left(\right. 0 , I \left.\right)} ​ \left(\parallel v_{\theta} ​ \left(\right. x_{\nu} , \nu , c_{t} \left.\right) - u_{\nu} \parallel\right)_{2}^{2} .$(18)

At inference, actions are generated by integrating the ODE forward from noise:

$x_{\nu + \Delta ​ \nu} = x_{\nu} + \Delta ​ \nu \cdot v_{\theta} ​ \left(\right. x_{\nu} , \nu , c_{t} \left.\right) , x_{\nu = 0} sim \mathcal{N} ​ \left(\right. 0 , I \left.\right) , \nu : 0 \rightarrow 1 .$(19)

##### Action Mapping.

We formulate actions as future states, which are more consistent across embodiments than raw control commands, easing cross-platform transfer. However, a single shared output head cannot account for differences in kinematics, control interfaces, and workspace constraints across platforms. We therefore introduce Action Mapping Module, a lightweight adapter that minimizes the embodiment gap and enables deployability across diverse robot platforms.

Architecturally, we initialize it from the last five layers of the action heads. It consumes concatenated embeddings from the low-level VLM and action heads, and outputs robot-specific commands. Each deployed robot uses its own adapter; optional MLPs encode joint/workspace limits to respect physical constraints.

### 3.4 Training

Training proceeds in two phases. First, the Process-Reward Operator (PRO) is pretrained in isolation on real executed trajectories from industrial deployment data, where ground-truth progress, risk, and termination signals are available from operational telemetry. The PRO supervision terms $\mathcal{L}_{\text{progress}}$, $\mathcal{L}_{\text{risk}}$, and $\mathcal{L}_{\text{term}}$ (Eqs.[8](https://arxiv.org/html/2604.20246#S3.E8 "Equation 8 ‣ Progress. ‣ 3.2.2 PRO: Process-Reward Operator ‣ 3.2 Cortex 2.0 Architecture ‣ 3 Methodology ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment")–[10](https://arxiv.org/html/2604.20246#S3.E10 "Equation 10 ‣ Termination. ‣ 3.2.2 PRO: Process-Reward Operator ‣ 3.2 Cortex 2.0 Architecture ‣ 3 Methodology ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment")) are optimized during this stage only and do not enter the main training objective. PRO therefore learns directly from deployment telemetry, independently of policy updates.

Once PRO produces stable signals, its parameters are frozen and it serves as a fixed scoring module in the planning loop. The world model and action heads are then trained jointly on the composite objective in Eq.[5](https://arxiv.org/html/2604.20246#S3.E5 "Equation 5 ‣ 3.1 Overview ‣ 3 Methodology ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment") with PRO providing advantage supervision $I_{t}$ to the action heads without receiving gradient updates.

We follow the knowledge insulation scheme[[13](https://arxiv.org/html/2604.20246#bib.bib40 "Knowledge Insulating Vision-Language-Action Models: Train Fast, Run Fast, Generalize Better")]: in stage one, gradients from the world model and action heads are blocked from the pretrained VLM backbone; in stage two, all components except the frozen PRO are jointly optimized end-to-end.

The planning budget $k$ and horizons $H_{\text{wm}}$, $H_{\text{act}}$ are inference-time hyperparameters and do not affect training.

### 3.5 Cross-Embodiment Design

Across single-arm pick-and-place, dual-arm item sorting, screw sorting, and shoebox unpacking, Cortex 2.0 runs the same planning loop: generate $k$ visual-latent rollouts, score them, and commit to the best trajectory. Because planning operates in visual space, it generalizes across tasks and robot embodiments without modification. Embodiment-specific adaptation is handled entirely by the action heads (Section[3.3](https://arxiv.org/html/2604.20246#S3.SS3 "3.3 VLA Policy ‣ 3 Methodology ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment")).

## 4 Dataset Composition

Cortex 2.0 is trained on a heterogeneous corpus combining proprietary deployment data, targeted teleoperation demonstrations, open-source cross-embodiment datasets, and synthetic simulation data. Table[1](https://arxiv.org/html/2604.20246#S4.T1 "Table 1 ‣ 4 Dataset Composition ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment") summarizes the composition.

Table 1: Dataset composition. F/T = force–torque sensor; Proprio = proprioceptive state (joint positions, velocities, torques).

Source Episodes Hours Task families Sensors
Deployment$>$10M$>$25k Pick, pack, unpack, sort, returns, kitting, assembly, navigation RGB, F/T, vacuum, proprio
Teleoperation$sim$40k$sim$400 Pick, pack, unpack, sort, returns, kitting, navigation RGB, F/T, proprio
Open-source$sim$970k$sim$2k Cross-embodiment diverse RGB, proprio
Synthetic$sim$20k—Pick, sort, table-top rearrangement RGB, proprio

### 4.1 Real-World Deployment Data

Our fleet has accumulated over 500 million manipulation interactions across warehouse deployments, collected continuously at 30 Hz from operational robotic systems (Figure[5](https://arxiv.org/html/2604.20246#S4.F5 "Figure 5 ‣ 4.3 In-House Teleoperation Data ‣ 4 Dataset Composition ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment")). All sensory observations including RGB images from wrist cameras, proprioceptive signals, and operational telemetry including force–torque, vacuum pressure, and contact signals, are recorded simultaneously. Telemetry directly supervises PRO; visual observations train the world model. Cortex 2.0 is trained on a curated subset of 10 million interactions sampled to preserve task diversity and coverage of failure modes. As fleet size grows and training subset scales, Cortex 2.0 benefits from increased diversity of states and execution contexts, leading to compounding improvements in reward quality and downstream policy performance.

### 4.2 Open-Source and Synthetic Data

To broaden embodiment and task diversity, we incorporate large-scale publicly available robot datasets (Figure[6](https://arxiv.org/html/2604.20246#S4.F6 "Figure 6 ‣ 4.3 In-House Teleoperation Data ‣ 4 Dataset Composition ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment")) including components of Open X-Embodiment [[34](https://arxiv.org/html/2604.20246#bib.bib6 "Open X-Embodiment: Robotic Learning Datasets and RT-X Models")], BridgeData V2 [[38](https://arxiv.org/html/2604.20246#bib.bib35 "BridgeData V2: A Dataset for Robot Learning at Scale")], and DROID [[22](https://arxiv.org/html/2604.20246#bib.bib36 "DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset")]. Synthetic data generated in simulation using the RoboCasa framework [[33](https://arxiv.org/html/2604.20246#bib.bib42 "RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots")] augments environmental diversity and trajectory variability across a broad set of manipulation tasks[7](https://arxiv.org/html/2604.20246#S4.F7 "Figure 7 ‣ 4.3 In-House Teleoperation Data ‣ 4 Dataset Composition ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"). Together, these components form a unified corpus that bridges real and simulated data, supporting both fine-grained control learning and high-level reasoning across embodiments.

### 4.3 In-House Teleoperation Data

A core component of our training corpus is in-house data collection across a wide range of warehouse operation tasks under both single- and dual-arm configurations. We collect through complementary channels. Egocentric recordings, captured from head-mounted cameras worn during normal workflow, yield data at high scale with minimal overhead. Full-body human suite recordings capture whole-body motion via inertial and optical motion capture, essential for tasks requiring coordinated reach, bimanual handling and posture-dependent manipulation. Handheld gripper demonstrations use an instrumented end-effector, producing trajectories whose sensory and kinematic distribution closely matches the deployed robot. This channel is particularly valuable for contact-rich tasks such as screw sorting, kitting, and returns handling. Teleoperation allows operators to remotely control the target robot in real time, preserving exact embodiment dynamics and enabling fast episode collection for long-horizon and deformable-object tasks. We record synchronized visual observations, gripper force feedback, and full robot state. Our teleoperation framework maps hand motions to robot TCP trajectories via inverse kinematics, achieving latencies at the 10,ms level and sub-centimeter replay precision.

![Image 5: Refer to caption](https://arxiv.org/html/2604.20246v1/graphics/extracted_data/production_data/r3.png)![Image 6: Refer to caption](https://arxiv.org/html/2604.20246v1/graphics/extracted_data/production_data/r19.png)![Image 7: Refer to caption](https://arxiv.org/html/2604.20246v1/graphics/extracted_data/production_data/r20.png)
![Image 8: Refer to caption](https://arxiv.org/html/2604.20246v1/graphics/extracted_data/production_data/r1.png)![Image 9: Refer to caption](https://arxiv.org/html/2604.20246v1/graphics/extracted_data/production_data/r15.png)![Image 10: Refer to caption](https://arxiv.org/html/2604.20246v1/graphics/extracted_data/production_data/r18.png)
![Image 11: Refer to caption](https://arxiv.org/html/2604.20246v1/graphics/extracted_data/production_data/r11.png)![Image 12: Refer to caption](https://arxiv.org/html/2604.20246v1/graphics/extracted_data/production_data/r2.png)![Image 13: Refer to caption](https://arxiv.org/html/2604.20246v1/graphics/extracted_data/production_data/r14.png)

Figure 5: Real Production Robotic Data.

![Image 14: Refer to caption](https://arxiv.org/html/2604.20246v1/graphics/extracted_data/libero/img6.png)![Image 15: Refer to caption](https://arxiv.org/html/2604.20246v1/graphics/extracted_data/libero/img7.png)![Image 16: Refer to caption](https://arxiv.org/html/2604.20246v1/graphics/extracted_data/libero/img8.png)![Image 17: Refer to caption](https://arxiv.org/html/2604.20246v1/graphics/extracted_data/libero/img9.png)
![Image 18: Refer to caption](https://arxiv.org/html/2604.20246v1/graphics/extracted_data/libero/img16.png)![Image 19: Refer to caption](https://arxiv.org/html/2604.20246v1/graphics/extracted_data/libero/img17.png)![Image 20: Refer to caption](https://arxiv.org/html/2604.20246v1/graphics/extracted_data/libero/img18.png)![Image 21: Refer to caption](https://arxiv.org/html/2604.20246v1/graphics/extracted_data/libero/img19.png)
![Image 22: Refer to caption](https://arxiv.org/html/2604.20246v1/graphics/extracted_data/libero/img1_second_view.png)![Image 23: Refer to caption](https://arxiv.org/html/2604.20246v1/graphics/extracted_data/libero/img2_first_view.png)![Image 24: Refer to caption](https://arxiv.org/html/2604.20246v1/graphics/extracted_data/libero/img2_second_view.png)![Image 25: Refer to caption](https://arxiv.org/html/2604.20246v1/graphics/extracted_data/libero/img3_second_view.png)

Figure 6: Open Source Datasets.

![Image 26: Refer to caption](https://arxiv.org/html/2604.20246v1/graphics/extracted_data/humanoid/Screenshot_from_2025-11-09_17-26-52.png)

![Image 27: Refer to caption](https://arxiv.org/html/2604.20246v1/graphics/extracted_data/humanoid/Screenshot_from_2025-11-09_17-28-09.png)

![Image 28: Refer to caption](https://arxiv.org/html/2604.20246v1/graphics/extracted_data/humanoid/Screenshot_from_2025-11-09_17-30-33.png)

![Image 29: Refer to caption](https://arxiv.org/html/2604.20246v1/graphics/extracted_data/humanoid/Screenshot_from_2025-11-09_17-30-49.png)

(a)Pick and Place - Single Arm

![Image 30: Refer to caption](https://arxiv.org/html/2604.20246v1/graphics/extracted_data/humanoid/Screenshot_from_2025-11-09_16-22-04.png)

![Image 31: Refer to caption](https://arxiv.org/html/2604.20246v1/graphics/extracted_data/humanoid/Screenshot_from_2025-11-09_16-23-56.png)

![Image 32: Refer to caption](https://arxiv.org/html/2604.20246v1/graphics/extracted_data/humanoid/Screenshot_from_2025-11-09_16-27-13.png)

![Image 33: Refer to caption](https://arxiv.org/html/2604.20246v1/graphics/extracted_data/humanoid/Screenshot_from_2025-11-09_16-28-23.png)

(b)Sorting and Packaging

![Image 34: Refer to caption](https://arxiv.org/html/2604.20246v1/graphics/extracted_data/humanoid/Screenshot_from_2025-11-07_19-29-19.png)

![Image 35: Refer to caption](https://arxiv.org/html/2604.20246v1/graphics/extracted_data/humanoid/Screenshot_from_2025-11-07_19-31-17.png)

![Image 36: Refer to caption](https://arxiv.org/html/2604.20246v1/graphics/extracted_data/humanoid/Screenshot_from_2025-11-07_19-35-13.png)

![Image 37: Refer to caption](https://arxiv.org/html/2604.20246v1/graphics/extracted_data/humanoid/Screenshot_from_2025-11-07_19-38-01.png)

(c)Pick and Packaging

![Image 38: Refer to caption](https://arxiv.org/html/2604.20246v1/graphics/extracted_data/humanoid/Screenshot_from_2025-11-09_16-41-13.png)

![Image 39: Refer to caption](https://arxiv.org/html/2604.20246v1/graphics/extracted_data/humanoid/Screenshot_from_2025-11-09_16-42-24.png)

![Image 40: Refer to caption](https://arxiv.org/html/2604.20246v1/graphics/extracted_data/humanoid/Screenshot_from_2025-11-09_16-42-38.png)

![Image 41: Refer to caption](https://arxiv.org/html/2604.20246v1/graphics/extracted_data/humanoid/Screenshot_from_2025-11-09_16-44-04.png)

(d)Transport

Figure 7: Synthetic Isaac Sim data – Dual Robotic Arms.

### 4.4 World Model Pretraining Data

The world model is pretrained on all visual data prior to fine-tuning on deployment recordings. This approach enables general physical priors to be acquired from diverse video sources before being adapted to deployment-specific manipulation dynamics. Deployment data collected from live operations feeds back into training continuously, creating a closed-loop improvement cycle where better world models enable more accurate planning, which improves execution quality, which in turn generates cleaner training signal.

## 5 Experiments

### 5.1 Experimental Setup

We evaluate Cortex 2.0 against state-of-the-art open-source visuomotor policies on a single-arm and a dual-arm manipulation platform, each equipped with a Universal Robot arm and a parallel gripper. Visual observations come from wrist-mounted cameras on each end-effector. Actions are executed at 30 Hz.

#### 5.1.1 Baselines

We compare against three policies: $\pi_{0.5}$[[21](https://arxiv.org/html/2604.20246#bib.bib2 "π0.5: A Vision-Language-Action Model with Open-World Generalization")] operating in absolute joint space; Diffusion Policy[[12](https://arxiv.org/html/2604.20246#bib.bib18 "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion")] using relative end-effector actions; and RDT-2[[29](https://arxiv.org/html/2604.20246#bib.bib23 "RDT2: Exploring the Scaling Limit of UMI Data Towards Zero-Shot Cross-Embodiment Generalization")], a diffusion transformer for bimanual manipulation. All models are trained with equivalent computational budgets of 200 GPU hours to ensure fair comparison. The $\pi_{0.5}$[[21](https://arxiv.org/html/2604.20246#bib.bib2 "π0.5: A Vision-Language-Action Model with Open-World Generalization")] policy is trained from the pretrained checkpoint from LeRobot [[9](https://arxiv.org/html/2604.20246#bib.bib43 "LeRobot: State-of-the-Art Machine Learning for Real-World Robotics in PyTorch")]. Diffusion Policy, due to its smaller capacity, is trained from scratch for 100,000 steps, with the ResNet-18 backbone initialized from pretrained ImageNet weights. For RDT-2 we use the official implementation[[29](https://arxiv.org/html/2604.20246#bib.bib23 "RDT2: Exploring the Scaling Limit of UMI Data Towards Zero-Shot Cross-Embodiment Generalization")]. Cortex 2.0, Diffusion Policy, and RDT-2 operate in Cartesian space with relative end-effector actions, predicting translational displacements and 6D rotation representations. $\pi_{0.5}$ uses absolute joint-space observations and actions, directly predicting target joint configurations. Table[2](https://arxiv.org/html/2604.20246#S5.T2 "Table 2 ‣ 5.1.1 Baselines ‣ 5.1 Experimental Setup ‣ 5 Experiments ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment") summarizes the model configurations. For both our model and baselines, we use a action chunk of size 12. The execution of the actions is at 30 Hz for all the embodiment.

Table 2: Baseline model configurations. All models trained with equivalent computational budgets.

Model Compute Obs. Space Action Space
Cortex 2.0 (ours)200 GPU-hr Cartesian Rel. end-effector
Diffusion Policy[[12](https://arxiv.org/html/2604.20246#bib.bib18 "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion")]200 GPU-hr Cartesian Rel. end-effector
$\pi_{0.5}$[[21](https://arxiv.org/html/2604.20246#bib.bib2 "π0.5: A Vision-Language-Action Model with Open-World Generalization")]200 GPU-hr Absolute joint Absolute joint
RDT-2[[29](https://arxiv.org/html/2604.20246#bib.bib23 "RDT2: Exploring the Scaling Limit of UMI Data Towards Zero-Shot Cross-Embodiment Generalization")]200 GPU-hr Cartesian Rel. end-effector

#### 5.1.2 Evaluation Protocol

Across all experiments we track unrecoverable states where human intervention is required. These are recorded under three conditions:

1.   1.
Safety-critical collisions: the robot collides with the environment, itself, or the other arm; execution is halted and the system is re-homed before continuing.

2.   2.
Persistent control deadlocks: the policy enters repeated or oscillatory motion without measurable task progress that does not resolve through continued execution.

3.   3.
Unrecoverable scene states: robot actions change the scene such that the policy can no longer recover (e.g., severe object displacement, entanglement, or clutter accumulation).

When an unrecoverable state occurs, we apply the minimum intervention needed to resume execution from the last recoverable state rather than resetting the task from scratch. This metric captures both safety interruptions and practical autonomy breakdowns under a realistic deployment protocol.

#### 5.1.3 Training Data

Although Cortex 2.0 transfers efficiently to new tasks due to extensive pretraining on real-world deployment data, we collected targeted demonstrations for each benchmark task to enable fair comparison with baselines. Table[3](https://arxiv.org/html/2604.20246#S5.T3 "Table 3 ‣ 5.1.3 Training Data ‣ 5.1 Experimental Setup ‣ 5 Experiments ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment") summarizes data volumes.

Table 3: Demonstration data collected per benchmark task.

Task Episodes Hours Arms Robot
Pick-and-Place 560 0.5 1 single-arm
Sorting Items and Trash 8,700 21.0 2 dual-arm
Sorting Screws 3,100 8.2 2 dual-arm
Shoebox 2,900 8.1 2 dual-arm

#### 5.1.4 Planning Budget

A key design parameter in Cortex 2.0 is $k$, the number of imagined future rollouts sampled and scored by PRO before committing to an action. Success rate increases with $k$ while inference time per step increases linearly, capturing the central trade-off between foresight quality and computational cost (Figure[8](https://arxiv.org/html/2604.20246#S5.F8 "Figure 8 ‣ 5.1.4 Planning Budget ‣ 5.1 Experimental Setup ‣ 5 Experiments ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment")).

![Image 42: Refer to caption](https://arxiv.org/html/2604.20246v1/graphics/k_rolloutplot.png)

Figure 8: Cortex 2.0 Performance against Number of Rollouts $k$: With increasing number of rollouts $k$, the performance increases from 0.962 with 1 rollout to 0.996 at 30 rollouts. At the same time the time per step increases from 310 ms at a single rollout to 9200 ms for 30 rollouts.

For all task evaluations below we fix a low-latency setting of $k = 2$. The budget can be adjusted per task: higher $k$ for costly failure modes such as packing, where errors compound, and lower $k$ when recovery is cheap such as regrasping. Beyond $k$, rollout quality can be independently controlled via the number of denoising steps in the flow-matching world model, providing a second axis for the compute–quality trade-off.

### 5.2 Task Definitions

##### Single-Arm Pick-and-Place.

As illustrated in Figure[9](https://arxiv.org/html/2604.20246#S5.F9 "Figure 9 ‣ Sorting Items and Trash. ‣ 5.2 Task Definitions ‣ 5 Experiments ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"), the robot grasps an item from a source bin and places it in a target bin. We fine-tune on 160 episodes. Trials receive 1.0 for success and fractional credit for partial completion; we report the mean over 16 trials. Although the simplest task in our benchmark suite, it still exposes characteristic failure modes of reactive policies: cluttered bins or difficult object poses lead to collisions, regrasp deadlocks, and releases that leave the item unrecoverable.

##### Sorting Items and Trash.

The dual-arm robot is presented with a cardboard box containing 10–15 randomly placed items and trash. The objective is to sort contents by placing trash into a left bin and items into a right bin (Figure[10](https://arxiv.org/html/2604.20246#S5.F10 "Figure 10 ‣ Sorting Items and Trash. ‣ 5.2 Task Definitions ‣ 5 Experiments ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment")). This evaluates category discrimination in clutter, grasp planning, and reliable pick-and-place across diverse object types. A trial is considered successful only when all items are correctly sorted into their respective bins. Any operation requiring human intervention is counted as a failure. The cluttered source bin forces the robot to navigate tight spaces, where imprecise approach or extraction motions frequently result in collisions with the bin walls. Category mistakes such as placing trash into the items bin are common in the presence of visually ambiguous objects.

Figure 9: Subtasks of Single-Arm Pick-and-Place. The robot detects best item to pick, transitions to the left bin and releases the object before moving to its home position.

Figure 10: Subtasks Sorting Items and Trash. The robot must detect which items are trash and which ones are good, then place trash into right bin and good items into left bin.

##### Sorting Screws.

Fine-grained manipulation of small metal screws and tooling scattered on a tabletop is shown in Figure[11](https://arxiv.org/html/2604.20246#S5.F11 "Figure 11 ‣ Sorting Screws. ‣ 5.2 Task Definitions ‣ 5 Experiments ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"). The robot picks each screw and deposits it into the correct compartment of a multi-section toolbox. The challenge lies in precisely localizing and grasping small, reflective objects that may roll or shift during manipulation under challenging lighting conditions. A trial is considered successful only when all screws are correctly placed into their respective compartments. This task is particularly hard for visuomotor policies: screws are small, reflective, and can be occluded by one another. Sub-millimeter misalignments during approach shift the screw rather than grasping it. Human interventions are required when a dropped screw rolls outside the reachable workspace and must be physically returned, when the policy repeatedly attempts to grasp an unreachable or occluded screw without recovering, and when a screw is deposited into the wrong compartment and must be manually corrected.

Figure 11: Subtasks of Sorting Screws. All screws are sorted into the according compartments.

##### Shoebox Unpacking.

The robot executes a four-step sequence: (1)open a closed shoebox lid; (2)remove packing paper and place it in the left bin; (3)extract one shoe and place it in the right bin; (4)extract the remaining shoe and place it in the right bin (Figure[12](https://arxiv.org/html/2604.20246#S5.F12 "Figure 12 ‣ Shoebox Unpacking. ‣ 5.2 Task Definitions ‣ 5 Experiments ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment")). This tests long-horizon execution, deformable object handling, and articulated container manipulation. Success requires completion of all steps; partial completion is not counted. The robot must first identify the correct side of the shoebox to open; pressing or pulling on the wrong face can punch a hole through the cardboard. Once open, every subtask reshapes the scene: as the lid is opened or the paper is removed, new information about the next step emerges. Interventions are triggered when the robot fails to open the correct side, when a shoe slips from an unstable grasp and falls outside the bin, or when the second shoe is never located and the policy stops making progress.

Figure 12: Subtasks of Shoebox Unpacking. The shoebox is opened, packaging paper is removed and both shoes are placed into the right bin.

### 5.3 Results

#### 5.3.1 Experiment 1: Single-Arm Pick-and-Place

The pick-and-place task evaluates precision manipulation and low-data adaptation across 16 trials, with item position and orientation randomized between trials. Scoring is continuous: trials receive 1.0 for full success and fractional credit for partial completion.

Cortex 2.0 achieves the highest mean score across all methods, demonstrating strong precision under the low-data regime of only 160 fine-tuning episodes. $\pi_{0.5}$ achieves the second-highest score but exhibits greater variance across trials, reflecting inconsistent approach trajectories when adapting from its pretraining distribution to the single-arm setting. Diffusion Policy attains a moderate mean score; its primary failure mode are placement misalignment at the final stage and premature object drops during approach. RDT-2 achieves the lowest mean score, with frequent grasp failures.

Table 4: Single-arm pick-and-place results. Mean score over 16 trials; 1.0 = full success, fractional credit for partial completion.

Model Success Rate Avg. Completion Time Human Interventions
Cortex 2.0 (ours)0.98 20s 0
$\pi_{0.5}$[[21](https://arxiv.org/html/2604.20246#bib.bib2 "π0.5: A Vision-Language-Action Model with Open-World Generalization")]0.7 49s 2
Diffusion Policy[[12](https://arxiv.org/html/2604.20246#bib.bib18 "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion")]0.56 53s 4
RDT-2[[29](https://arxiv.org/html/2604.20246#bib.bib23 "RDT2: Exploring the Scaling Limit of UMI Data Towards Zero-Shot Cross-Embodiment Generalization")]0.4 63s 7

#### 5.3.2 Experiment 2: Sorting Items and Trash

The sorting task evaluates repeated pick-and-place in a cluttered environment across 10 rollouts per policy. Each rollout begins with 10–15 objects randomly placed in the source box, with object types, quantities, positions, and orientations randomized between rollouts.

Cortex 2.0 achieves the highest per-operation success rate and the shortest average task duration, completing all rollouts without any human intervention. In contrast, all baseline policies require human intervention in every rollout to complete the task. $\pi_{0.5}$ achieves higher success than the remaining baselines but fails to complete the full task within the 15-minute execution limit in all runs; its dominant failure mode is repeated local replanning around failed grasp attempts. Diffusion Policy attains moderate success but with greater instability driven by grasp failures and object drops. RDT-2 achieves significantly lower success rates than all other methods. Both Diffusion Policy and RDT-2 also reach the 15-minute execution limit in all runs without completing the full task.

Cortex 2.0 is qualitatively different from all baselines in this experiment: it reliably completes the task autonomously, whereas all baseline methods depend on repeated human intervention to finish execution.

Table 5: Sorting items and trash results. Success reported per individual sorting operation.

Model Per-Op. Success Task Completion Human Interventions
Cortex 2.0 (ours)0.95 700s 0
$\pi_{0.5}$[[21](https://arxiv.org/html/2604.20246#bib.bib2 "π0.5: A Vision-Language-Action Model with Open-World Generalization")]0.61—∗53
Diffusion Policy[[12](https://arxiv.org/html/2604.20246#bib.bib18 "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion")]0.47—∗59
RDT-2[[29](https://arxiv.org/html/2604.20246#bib.bib23 "RDT2: Exploring the Scaling Limit of UMI Data Towards Zero-Shot Cross-Embodiment Generalization")]0.18—∗95

∗ Task not completed within the execution limit.

#### 5.3.3 Experiment 3: Sorting Screws

The screw sorting task evaluates fine-grained manipulation across 10 rollouts per policy, with screw positions and orientations randomized between rollouts. Performance is reported as per-operation success rate aggregated across all screws and all rollouts.

Cortex 2.0 substantially outperforms all baseline methods, achieving near-perfect per-operation success while completing the task in the shortest average time and without entering any unrecoverable states. Among the baselines, this task exhibits the largest performance gap relative to Cortex 2.0. $\pi_{0.5}$ attains moderate success and remains the strongest baseline but exhibits limited precision on small objects of varying shapes. Diffusion Policy achieves lower success rates than $\pi_{0.5}$ but demonstrates comparatively greater stability. RDT-2 fails almost entirely, with zero successful placements and multiple unrecoverable states per rollout.

These results highlight the direct benefit of PRO-based lookahead in precision-critical scenarios: by filtering rollouts that pass through latent states associated with unstable contact, the system avoids the small errors that would otherwise shift object pose and compound task difficulty in subsequent steps.

Table 6: Screw sorting results. Success reported per individual screw placement.

Model Per-Op. Success Avg. Completion Time Human Interventions
Cortex 2.0 (ours)0.98 180 s 0
$\pi_{0.5}$[[21](https://arxiv.org/html/2604.20246#bib.bib2 "π0.5: A Vision-Language-Action Model with Open-World Generalization")]0.4—∗24
Diffusion Policy[[12](https://arxiv.org/html/2604.20246#bib.bib18 "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion")]0.2—∗16
RDT-2[[29](https://arxiv.org/html/2604.20246#bib.bib23 "RDT2: Exploring the Scaling Limit of UMI Data Towards Zero-Shot Cross-Embodiment Generalization")]0.0—∗50

∗ Task not completed within the execution limit.

#### 5.3.4 Experiment 4: Shoebox Unpacking

The shoebox task evaluates multi-step manipulation capabilities across 10 rollouts per policy, with box orientation, paper configuration, and shoe placement randomized between rollouts.

Cortex 2.0 achieves the highest holistic success rate while completing the task significantly faster than all baseline methods, and does so without requiring human intervention across any rollout. This indicates strong temporal consistency and the ability to maintain task context across the full four-step sequence. Notably, this benchmark is trained with significantly fewer demonstrations than the sorting tasks, emphasizing Cortex 2.0’s ability to transfer and scale to complex task structure in a data-limited setting.

$\pi_{0.5}$ achieves relatively high success at the subtask level but fails to complete the full task more frequently than Cortex 2.0, requires substantially longer execution times, and frequently enters unrecoverable states in later stages. It often progresses through the early steps but struggles to adapt to scene changes introduced by prior actions. Diffusion Policy exhibits significantly lower success rates, failing primarily during shoe extraction due to unstable grasping and limited recovery from failed picks. RDT-2 never completes the full task sequence and frequently enters unrecoverable states, preventing meaningful progress beyond the early stages.

Table 7: Shoebox unpacking results. Full task success requires completion of all four sequential steps.

Model Success Rate Avg. Completion Time Human Interventions
Cortex 2.0 (ours)0.96 58s 0
$\pi_{0.5}$[[21](https://arxiv.org/html/2604.20246#bib.bib2 "π0.5: A Vision-Language-Action Model with Open-World Generalization")]0.6 103s 5
Diffusion Policy[[12](https://arxiv.org/html/2604.20246#bib.bib18 "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion")]0.12 52s 9
RDT-2[[29](https://arxiv.org/html/2604.20246#bib.bib23 "RDT2: Exploring the Scaling Limit of UMI Data Towards Zero-Shot Cross-Embodiment Generalization")]0.0 62s 10

### 5.4 Toward In-Context Learning

Making the policy video-aware and training a world model jointly with policy learning constitutes a step toward in-context learning for robotics. Given a short sequence of demonstrations in the form of a video, the robot should be able to execute analogous steps without retraining. Today’s large language models exhibit in-context learning capabilities across many applications: agents can be conditioned to execute tasks through language alone. We are working toward analogous capabilities for robots, where visual demonstrations serve as task specifications. Formally, this corresponds to a policy $\pi_{\theta} ​ \left(\right. a \mid o , \tau^{\text{demo}} \left.\right)$ conditioned on a video demonstration $\tau^{\text{demo}}$ alongside the current observation $o$. The world model provides a natural substrate for this: by learning rich visual dynamics, it develops representations that support analogical reasoning over demonstration sequences, unlocking a new dimension of generalization for physical AI.

### 5.5 Summary and Discussion

Across all four benchmarks, Cortex 2.0 consistently achieves higher success rates and shorter execution times than all evaluated baseline policies, and is the only method to complete tasks without human intervention.

$\pi_{0.5}$ generally attains the strongest performance among the baselines at the subtask level but requires substantially longer execution times and frequently fails to complete tasks end-to-end, particularly in long-horizon settings. Diffusion Policy exhibits lower overall success, while RDT-2 consistently fails to complete complex tasks. All three baseline policies enter unrecoverable states more frequently than Cortex 2.0, reducing overall reliability and increasing dependence on human intervention.

![Image 43: Refer to caption](https://arxiv.org/html/2604.20246v1/graphics/sr.png)

![Image 44: Refer to caption](https://arxiv.org/html/2604.20246v1/graphics/comptime.png)

![Image 45: Refer to caption](https://arxiv.org/html/2604.20246v1/graphics/inter.png)

Figure 13: Performance comparison across all four benchmark tasks. Left: success rate (higher is better). Centre: average completion time in seconds (lower is better; baselines are capped at the 1,500 s execution limit for the item sorting tasks and at 300 s for the screw sorting task). Right: total human interventions per task (lower is better).

The results support two key claims of Cortex 2.0. First, pretraining on real-world deployment data enables strong generalization: with limited fine-tuning data, Cortex 2.0 achieves success rates beyond 90% on all tasks, enabling a strong baseline model at deployment that reaches 99% after continued operation on in-domain data. Second, world-model-based planning provides qualitative robustness benefits beyond success rate: by filtering bad branches before execution, the system avoids the characteristic failure mode of reactive baselines, in which a missed grasp triggers repeated retries and eventually escalates into deadlock. This directly translates into zero human interventions across all benchmarks, a metric that more accurately reflects the industrial operational cost than the success rate alone.

## 6 Conclusion

Deploying generalist robot policies in real production environments remains a difficult challenge: reactive systems fail under long-horizon task sequences, compound errors are costly to recover from, and the diversity of objects, layouts, and embodiments cannot be fully covered by any finite dataset. Our work with Cortex 2.0 shows that vision–language–action models can overcome many of these barriers when augmented with physical foresight. The central contribution of Cortex 2.0 is the integration of a world model into the manipulation policy loop, shifting from try-and-see control to plan-and-try. By generating candidate futures in visual latent space and scoring them via PRO before any action is executed, the system filters out bad branches before they become unrecoverable states. Across warehouse pick-and-place, item sorting, and shoebox handling evaluations, Cortex 2.0 consistently outperformed all baselines while requiring zero human interventions. The planning budget $k$ provides a practical lever to trade foresight quality against latency, enabling the same system to allocate more computation to high-stakes decisions such as packing, and less to cheap-recovery situations such as regrasping.

Training in visual latent space proved critical for two reasons. First, it enables cross-embodiment transfer: the same PRO scoring function and world model operate across single-arm, dual-arm, and other robot platforms without modification, because visual representations encode transferable physical regularities that are independent of specific kinematics. Second, it makes the data strategy scalable: deployment cameras provide a natural multiplier on training signal, and the continuous feedback loop from live operations ensures the world model learns from the full complexity of real-world conditions rather than simplified lab setups.

We are continuously expanding the deployment database with new operational data from our robot fleet, expanding coverage to new task families, object categories, and embodiments, and rolling out the policy to new industrial partners and workflows. Each deployment adds training signal that feeds back into the world model and PRO, compounding improvements in planning quality over time. The gap between a research system and a production system is not closed at training time but closed through expanded deployment, iteration, and scale.

Future work. We plan to scale world model training substantially, both in compute and data. Allocating more training time and leveraging a larger portion of our collected deployment data can improve the fidelity of predicted rollouts. Since the VLA policy is conditioned on the selected future latent, sharper predictions provide a more informative signal to the policy and directly improve the generated action chunk. Current evaluations cover a subset of embodiments and task families, and the planning horizon $H_{w ​ m}$ and budget $k$ are fixed per task. We therefore plan to strengthen online adaptation and uncertainty-aware dynamic budget allocation in PRO, tighten the coupling between video tokenization and control for longer-horizon foresight, and validate in-context learning from video demonstrations on unseen task families at test time. We believe Cortex 2.0 is a step toward dependable, general-purpose robot intelligence that plans before it acts, adapts continuously from deployment, and scales robustly to the messy, continuously changing conditions of real production environments.

## References

*   [1]N. Agarwal, A. Ali, M. Bala, Y. Balaji, E. Barker, T. Cai, P. Chattopadhyay, Y. Chen, Y. Cui, Y. Ding, et al. (2025)Cosmos World Foundation Model Platform for Physical AI. arXiv preprint arXiv:2501.03575. Cited by: [§1](https://arxiv.org/html/2604.20246#S1.SS0.SSS0.Px1.p1.1 "Background and related progress. ‣ 1 Introduction ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"), [§2.3](https://arxiv.org/html/2604.20246#S2.SS3.p2.1 "2.3 World Models for Robotics ‣ 2 Related Works ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"). 
*   [2] (2025)Stochastic Interpolants: A Unifying Framework for Flows and Diffusions. Journal of Machine Learning Research 26 (209),  pp.1–80. Cited by: [§2.2](https://arxiv.org/html/2604.20246#S2.SS2.p1.2 "2.2 Flow Matching for Robot Control ‣ 2 Related Works ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"). 
*   [3]M. S. Albergo and E. Vanden-Eijnden (2023)Building Normalizing Flows with Stochastic Interpolants. In International Conference on Learning Representations, Cited by: [§2.2](https://arxiv.org/html/2604.20246#S2.SS2.p1.2 "2.2 Flow Matching for Robot Control ‣ 2 Related Works ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"). 
*   [4]M. Assran, A. Bardes, D. Fan, Q. Garrido, R. Howes, M. Muckley, A. Rizvi, C. Roberts, K. Sinha, A. Zholus, et al. (2025)V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning. arXiv preprint arXiv:2506.09985. Cited by: [§2.3](https://arxiv.org/html/2604.20246#S2.SS3.p2.1 "2.3 World Models for Robotics ‣ 2 Related Works ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"). 
*   [5]J. Bjorck, F. Castañeda, N. Cherniadev, X. Da, R. Ding, L. Fan, Y. Fang, D. Fox, F. Hu, S. Huang, et al. (2025)GR00T N1: An Open Foundation Model for Generalist Humanoid Robots. arXiv preprint arXiv:2503.14734. Cited by: [§2.1](https://arxiv.org/html/2604.20246#S2.SS1.p2.2 "2.1 Vision–Language–Action Models ‣ 2 Related Works ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"). 
*   [6]K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, et al. (2024)$\pi_{0}$: A Vision-Language-Action Flow Model for General Robot Control. arXiv preprint arXiv:2410.24164. Cited by: [§1](https://arxiv.org/html/2604.20246#S1.p1.1 "1 Introduction ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"), [§2.1](https://arxiv.org/html/2604.20246#S2.SS1.p2.2 "2.1 Vision–Language–Action Models ‣ 2 Related Works ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"), [§2.2](https://arxiv.org/html/2604.20246#S2.SS2.p1.2 "2.2 Flow Matching for Robot Control ‣ 2 Related Works ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"). 
*   [7]A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog, J. Hsu, et al. (2022)RT-1: Robotics Transformer for Real-World Control at Scale. arXiv preprint arXiv:2212.06817. Cited by: [§1](https://arxiv.org/html/2604.20246#S1.p1.1 "1 Introduction ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"), [§2.1](https://arxiv.org/html/2604.20246#S2.SS1.p1.1 "2.1 Vision–Language–Action Models ‣ 2 Related Works ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"). 
*   [8]Q. Bu, J. Cai, L. Chen, X. Cui, Y. Ding, S. Feng, S. Gao, X. He, X. Huang, S. Jiang, et al. (2025)AgiBot World Colosseo: A Large-Scale Manipulation Platform for Scalable and Intelligent Embodied Systems. arXiv preprint arXiv:2503.06669. Cited by: [§2.5](https://arxiv.org/html/2604.20246#S2.SS5.p1.1 "2.5 Datasets for Robot Learning ‣ 2 Related Works ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"). 
*   [9]R. Cadene, S. Alibert, A. Soare, Q. Gallouedec, A. Zouitine, S. Palma, P. Kooijmans, M. Aractingi, M. Shukor, D. Aubakirova, M. Russi, F. Capuano, C. Pascale, J. Choghari, J. Moss, and T. Wolf (2024)LeRobot: State-of-the-Art Machine Learning for Real-World Robotics in PyTorch. Note: [https://github.com/huggingface/lerobot](https://github.com/huggingface/lerobot)Cited by: [§5.1.1](https://arxiv.org/html/2604.20246#S5.SS1.SSS1.p1.3 "5.1.1 Baselines ‣ 5.1 Experimental Setup ‣ 5 Experiments ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"). 
*   [10]R. Calandra, A. Owens, D. Jayaraman, J. Lin, W. Yuan, J. Malik, E. H. Adelson, and S. Levine (2018)More Than a Feeling: Learning to Grasp and Regrasp Using Vision and Touch. IEEE Robotics and Automation Letters. Cited by: [§2.4](https://arxiv.org/html/2604.20246#S2.SS4.p1.1 "2.4 Force Feedback and Multimodal Sensing ‣ 2 Related Works ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"). 
*   [11]C. Cheang, G. Chen, Y. Jing, T. Kong, H. Li, Y. Li, Y. Liu, H. Wu, J. Xu, Y. Yang, et al. (2024)GR-2: A Generative Video-Language-Action Model with Web-Scale Knowledge for Robot Manipulation. arXiv preprint arXiv:2410.06158. Cited by: [§2.3](https://arxiv.org/html/2604.20246#S2.SS3.p2.1 "2.3 World Models for Robotics ‣ 2 Related Works ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"). 
*   [12]C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song (2025)Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. The International Journal of Robotics Research 44 (10-11),  pp.1684–1704. Cited by: [§2.2](https://arxiv.org/html/2604.20246#S2.SS2.p1.2 "2.2 Flow Matching for Robot Control ‣ 2 Related Works ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"), [§5.1.1](https://arxiv.org/html/2604.20246#S5.SS1.SSS1.p1.3 "5.1.1 Baselines ‣ 5.1 Experimental Setup ‣ 5 Experiments ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"), [Table 2](https://arxiv.org/html/2604.20246#S5.T2.1.4.1 "In 5.1.1 Baselines ‣ 5.1 Experimental Setup ‣ 5 Experiments ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"), [Table 4](https://arxiv.org/html/2604.20246#S5.T4.1.4.1 "In 5.3.1 Experiment 1: Single-Arm Pick-and-Place ‣ 5.3 Results ‣ 5 Experiments ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"), [Table 5](https://arxiv.org/html/2604.20246#S5.T5.3.3.2 "In 5.3.2 Experiment 2: Sorting Items and Trash ‣ 5.3 Results ‣ 5 Experiments ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"), [Table 6](https://arxiv.org/html/2604.20246#S5.T6.3.3.2 "In 5.3.3 Experiment 3: Sorting Screws ‣ 5.3 Results ‣ 5 Experiments ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"), [Table 7](https://arxiv.org/html/2604.20246#S5.T7.1.4.1 "In 5.3.4 Experiment 4: Shoebox Unpacking ‣ 5.3 Results ‣ 5 Experiments ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"). 
*   [13]D. Driess, J. T. Springenberg, B. Ichter, L. Yu, A. Li-Bell, K. Pertsch, A. Z. Ren, H. Walke, Q. Vuong, L. X. Shi, et al. (2025)Knowledge Insulating Vision-Language-Action Models: Train Fast, Run Fast, Generalize Better. arXiv preprint arXiv:2505.23705. Cited by: [§3.4](https://arxiv.org/html/2604.20246#S3.SS4.p3.1 "3.4 Training ‣ 3 Methodology ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"). 
*   [14]D. Driess, F. Xia, M. S. M. Sajjadi, C. Lynch, A. Chowdhery, B. Ichter, A. Wahid, J. Tompson, Q. Vuong, T. Yu, et al. (2023)PaLM-E: An Embodied Multimodal Language Model. arXiv preprint arXiv:2303.03378. Cited by: [§1](https://arxiv.org/html/2604.20246#S1.SS0.SSS0.Px1.p1.1 "Background and related progress. ‣ 1 Introduction ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"), [§2.1](https://arxiv.org/html/2604.20246#S2.SS1.p1.1 "2.1 Vision–Language–Action Models ‣ 2 Related Works ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"). 
*   [15]F. Ebert, Y. Yang, K. Schmeckpeper, B. Bucher, G. Georgakis, K. Daniilidis, C. Finn, and S. Levine (2021)Bridge Data: Boosting Generalization of Robotic Skills with Cross-Domain Datasets. arXiv preprint arXiv:2109.13396. Cited by: [§1](https://arxiv.org/html/2604.20246#S1.SS0.SSS0.Px1.p1.1 "Background and related progress. ‣ 1 Introduction ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"), [§2.5](https://arxiv.org/html/2604.20246#S2.SS5.p1.1 "2.5 Datasets for Robot Learning ‣ 2 Related Works ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"). 
*   [16]C. Finn and S. Levine (2017)Deep Visual Foresight for Planning Robot Motion. In 2017 IEEE International Conference on Robotics and Automation (ICRA),  pp.2786–2793. Cited by: [§2.3](https://arxiv.org/html/2604.20246#S2.SS3.p1.1 "2.3 World Models for Robotics ‣ 2 Related Works ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"). 
*   [17]D. Ha and J. Schmidhuber (2018)World Models. arXiv preprint arXiv:1803.10122 2 (3),  pp.440. Cited by: [§2.3](https://arxiv.org/html/2604.20246#S2.SS3.p1.1 "2.3 World Models for Robotics ‣ 2 Related Works ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"). 
*   [18]D. Hafner, T. Lillicrap, M. Norouzi, and J. Ba (2020)Mastering Atari with Discrete World Models. arXiv preprint arXiv:2010.02193. Cited by: [§2.3](https://arxiv.org/html/2604.20246#S2.SS3.p1.1 "2.3 World Models for Robotics ‣ 2 Related Works ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"). 
*   [19]D. Hafner, J. Pasukonis, J. Ba, and T. Lillicrap (2023)Mastering Diverse Domains Through World Models. arXiv preprint arXiv:2301.04104. Cited by: [§2.3](https://arxiv.org/html/2604.20246#S2.SS3.p1.1 "2.3 World Models for Robotics ‣ 2 Related Works ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"). 
*   [20]J. Ho, A. Jain, and P. Abbeel (2020)Denoising Diffusion Probabilistic Models. Advances in Neural Information Processing Systems 33,  pp.6840–6851. Cited by: [§2.2](https://arxiv.org/html/2604.20246#S2.SS2.p1.2 "2.2 Flow Matching for Robot Control ‣ 2 Related Works ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"). 
*   [21]P. Intelligence, K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, et al. (2025)$\pi_{0.5}$: A Vision-Language-Action Model with Open-World Generalization. arXiv preprint arXiv:2504.16054. Cited by: [§1](https://arxiv.org/html/2604.20246#S1.p1.1 "1 Introduction ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"), [§2.1](https://arxiv.org/html/2604.20246#S2.SS1.p2.2 "2.1 Vision–Language–Action Models ‣ 2 Related Works ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"), [§2.2](https://arxiv.org/html/2604.20246#S2.SS2.p1.2 "2.2 Flow Matching for Robot Control ‣ 2 Related Works ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"), [§5.1.1](https://arxiv.org/html/2604.20246#S5.SS1.SSS1.p1.3 "5.1.1 Baselines ‣ 5.1 Experimental Setup ‣ 5 Experiments ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"), [Table 2](https://arxiv.org/html/2604.20246#S5.T2.1.1.1 "In 5.1.1 Baselines ‣ 5.1 Experimental Setup ‣ 5 Experiments ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"), [Table 4](https://arxiv.org/html/2604.20246#S5.T4.1.1.1 "In 5.3.1 Experiment 1: Single-Arm Pick-and-Place ‣ 5.3 Results ‣ 5 Experiments ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"), [Table 5](https://arxiv.org/html/2604.20246#S5.T5.1.1.1 "In 5.3.2 Experiment 2: Sorting Items and Trash ‣ 5.3 Results ‣ 5 Experiments ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"), [Table 6](https://arxiv.org/html/2604.20246#S5.T6.1.1.1 "In 5.3.3 Experiment 3: Sorting Screws ‣ 5.3 Results ‣ 5 Experiments ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"), [Table 7](https://arxiv.org/html/2604.20246#S5.T7.1.1.1 "In 5.3.4 Experiment 4: Shoebox Unpacking ‣ 5.3 Results ‣ 5 Experiments ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"). 
*   [22]A. Khazatsky, K. Pertsch, and … (2024)DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset. External Links: 2403.12945 Cited by: [§2.5](https://arxiv.org/html/2604.20246#S2.SS5.p1.1 "2.5 Datasets for Robot Learning ‣ 2 Related Works ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"), [§4.2](https://arxiv.org/html/2604.20246#S4.SS2.p1.1 "4.2 Open-Source and Synthetic Data ‣ 4 Dataset Composition ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"). 
*   [23]M. J. Kim, C. Finn, and P. Liang (2025)Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success. arXiv preprint arXiv:2502.19645. Cited by: [§2.1](https://arxiv.org/html/2604.20246#S2.SS1.p1.1 "2.1 Vision–Language–Action Models ‣ 2 Related Works ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"). 
*   [24]M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, et al. (2024)OpenVLA: An Open-Source Vision-Language-Action Model. arXiv preprint arXiv:2406.09246. Cited by: [§2.1](https://arxiv.org/html/2604.20246#S2.SS1.p1.1 "2.1 Vision–Language–Action Models ‣ 2 Related Works ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"). 
*   [25]J. Lee, J. Duan, H. Fang, Y. Deng, S. Liu, B. Li, B. Fang, J. Zhang, Y. R. Wang, S. Lee, et al. (2025)MolmoAct: Action Reasoning Models That Can Reason in Space. arXiv preprint arXiv:2508.07917. Cited by: [§2.1](https://arxiv.org/html/2604.20246#S2.SS1.p2.2 "2.1 Vision–Language–Action Models ‣ 2 Related Works ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"). 
*   [26]M. A. Lee, Y. Zhu, P. Zachares, M. Tan, K. Srinivasan, S. Savarese, L. Fei-Fei, A. Garg, and J. Bohg (2020-06)Making Sense of Vision and Touch: Learning Multimodal Representations for Contact-Rich Tasks. IEEE Transactions on Robotics 36 (3),  pp.582–596. External Links: ISSN 1941-0468, [Document](https://dx.doi.org/10.1109/TRO.2019.2959445), [Link](http://dx.doi.org/10.1109/TRO.2019.2959445)Cited by: [§2.4](https://arxiv.org/html/2604.20246#S2.SS4.p1.1 "2.4 Force Feedback and Multimodal Sensing ‣ 2 Related Works ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"), [§2.5](https://arxiv.org/html/2604.20246#S2.SS5.p1.1 "2.5 Datasets for Robot Learning ‣ 2 Related Works ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"). 
*   [27]C. Li et al. (2025)Robotic World Model: A Neural Network Simulator for Robust Policy Optimization in the Real World. arXiv preprint arXiv:2501.10100. Cited by: [§2.3](https://arxiv.org/html/2604.20246#S2.SS3.p2.1 "2.3 World Models for Robotics ‣ 2 Related Works ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"). 
*   [28]Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le (2022)Flow Matching for Generative Modeling. arXiv preprint arXiv:2210.02747. Cited by: [§2.2](https://arxiv.org/html/2604.20246#S2.SS2.p1.2 "2.2 Flow Matching for Robot Control ‣ 2 Related Works ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"), [§3.2.3](https://arxiv.org/html/2604.20246#S3.SS2.SSS3.p1.5 "3.2.3 World Model ‣ 3.2 Cortex 2.0 Architecture ‣ 3 Methodology ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"). 
*   [29]S. Liu, B. Li, K. Ma, L. Wu, H. Tan, X. Ouyang, H. Su, and J. Zhu (2026)RDT2: Exploring the Scaling Limit of UMI Data Towards Zero-Shot Cross-Embodiment Generalization. arXiv preprint arXiv:2602.03310. Cited by: [§2.2](https://arxiv.org/html/2604.20246#S2.SS2.p1.2 "2.2 Flow Matching for Robot Control ‣ 2 Related Works ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"), [§5.1.1](https://arxiv.org/html/2604.20246#S5.SS1.SSS1.p1.3 "5.1.1 Baselines ‣ 5.1 Experimental Setup ‣ 5 Experiments ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"), [Table 2](https://arxiv.org/html/2604.20246#S5.T2.1.5.1 "In 5.1.1 Baselines ‣ 5.1 Experimental Setup ‣ 5 Experiments ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"), [Table 4](https://arxiv.org/html/2604.20246#S5.T4.1.5.1 "In 5.3.1 Experiment 1: Single-Arm Pick-and-Place ‣ 5.3 Results ‣ 5 Experiments ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"), [Table 5](https://arxiv.org/html/2604.20246#S5.T5.4.4.2 "In 5.3.2 Experiment 2: Sorting Items and Trash ‣ 5.3 Results ‣ 5 Experiments ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"), [Table 6](https://arxiv.org/html/2604.20246#S5.T6.4.4.2 "In 5.3.3 Experiment 3: Sorting Screws ‣ 5.3 Results ‣ 5 Experiments ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"), [Table 7](https://arxiv.org/html/2604.20246#S5.T7.1.5.1 "In 5.3.4 Experiment 4: Shoebox Unpacking ‣ 5.3 Results ‣ 5 Experiments ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"). 
*   [30]J. Luo, C. Xu, X. Geng, G. Feng, K. Fang, L. Tan, S. Schaal, and S. Levine (2024)Multistage Cable Routing Through Hierarchical Imitation Learning. IEEE Transactions on Robotics 40,  pp.1476–1491. External Links: ISSN 1941-0468, [Document](https://dx.doi.org/10.1109/TRO.2024.3353075), [Link](http://dx.doi.org/10.1109/TRO.2024.3353075)Cited by: [§2.5](https://arxiv.org/html/2604.20246#S2.SS5.p1.1 "2.5 Datasets for Robot Learning ‣ 2 Related Works ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"). 
*   [31]Q. Lv, W. Kong, H. Li, J. Zeng, Z. Qiu, D. Qu, H. Song, Q. Chen, X. Deng, and J. Pang (2025)F1: A Vision-Language-Action Model Bridging Understanding and Generation to Actions. arXiv preprint arXiv:2509.06951. Cited by: [§2.1](https://arxiv.org/html/2604.20246#S2.SS1.p2.2 "2.1 Vision–Language–Action Models ‣ 2 Related Works ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"). 
*   [32]O. Mees, D. Ghosh, K. Pertsch, K. Black, H. R. Walke, S. Dasari, J. Hejna, T. Kreiman, C. Xu, J. Luo, et al. (2024)Octo: An Open-Source Generalist Robot Policy. In First Workshop on Vision-Language Models for Navigation and Manipulation at ICRA 2024, Cited by: [§2.1](https://arxiv.org/html/2604.20246#S2.SS1.p1.1 "2.1 Vision–Language–Action Models ‣ 2 Related Works ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"). 
*   [33]S. Nasiriany, A. Maddukuri, L. Zhang, A. Parikh, A. Lo, A. Joshi, A. Mandlekar, and Y. Zhu (2024)RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots. In Robotics: Science and Systems (RSS), Cited by: [§4.2](https://arxiv.org/html/2604.20246#S4.SS2.p1.1 "4.2 Open-Source and Synthetic Data ‣ 4 Dataset Composition ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"). 
*   [34]A. O’Neill, A. Rehman, A. Maddukuri, A. Gupta, A. Padalkar, A. Lee, A. Pooley, A. Gupta, A. Mandlekar, A. Jain, et al. (2024)Open X-Embodiment: Robotic Learning Datasets and RT-X Models. In 2024 IEEE International Conference on Robotics and Automation (ICRA),  pp.6892–6903. Cited by: [§1](https://arxiv.org/html/2604.20246#S1.SS0.SSS0.Px1.p1.1 "Background and related progress. ‣ 1 Introduction ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"), [§2.5](https://arxiv.org/html/2604.20246#S2.SS5.p1.1 "2.5 Datasets for Robot Learning ‣ 2 Related Works ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"), [§4.2](https://arxiv.org/html/2604.20246#S4.SS2.p1.1 "4.2 Open-Source and Synthetic Data ‣ 4 Dataset Composition ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"). 
*   [35]K. Pertsch, K. Stachowicz, B. Ichter, D. Driess, S. Nair, Q. Vuong, O. Mees, C. Finn, and S. Levine (2025)FAST: Efficient Action Tokenization for Vision-Language-Action Models. arXiv preprint arXiv:2501.09747. Cited by: [§2.1](https://arxiv.org/html/2604.20246#S2.SS1.p2.2 "2.1 Vision–Language–Action Models ‣ 2 Related Works ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"). 
*   [36]H. Qi et al. (2025)Strengthening Generative Robot Policies Through Predictive World Modeling. arXiv preprint arXiv:2502.00622. Cited by: [§2.3](https://arxiv.org/html/2604.20246#S2.SS3.p2.1 "2.3 World Models for Robotics ‣ 2 Related Works ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"). 
*   [37]Sereact (2025-09)Cortex: Bridging Vision, Language, and Action with Discrete Plans and Tokens. Note: Sereact Technical Blog External Links: [Link](https://sereact.ai/posts/cortex-bridging-vision-language-and-action-with-discrete-plans-and-tokens)Cited by: [§2.1](https://arxiv.org/html/2604.20246#S2.SS1.p2.2 "2.1 Vision–Language–Action Models ‣ 2 Related Works ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"). 
*   [38]H. R. Walke, K. Black, T. Z. Zhao, Q. Vuong, C. Zheng, P. Hansen-Estruch, A. W. He, V. Myers, M. J. Kim, M. Du, et al. (2023)BridgeData V2: A Dataset for Robot Learning at Scale. In Conference on Robot Learning,  pp.1723–1736. Cited by: [§2.5](https://arxiv.org/html/2604.20246#S2.SS5.p1.1 "2.5 Datasets for Robot Learning ‣ 2 Related Works ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"), [§4.2](https://arxiv.org/html/2604.20246#S4.SS2.p1.1 "4.2 Open-Source and Synthetic Data ‣ 4 Dataset Composition ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"). 
*   [39]M. Yang, Y. Du, K. Ghasemipour, J. Tompson, D. Schuurmans, and P. Abbeel (2023)Learning Interactive Real-World Simulators. arXiv preprint arXiv:2310.06114 1 (2),  pp.6. Cited by: [§1](https://arxiv.org/html/2604.20246#S1.SS0.SSS0.Px1.p1.1 "Background and related progress. ‣ 1 Introduction ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"), [§2.3](https://arxiv.org/html/2604.20246#S2.SS3.p2.1 "2.3 World Models for Robotics ‣ 2 Related Works ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"). 
*   [40]T. Z. Zhao, V. Kumar, S. Levine, and C. Finn (2023)Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware. In Robotics: Science and Systems, Cited by: [§3.3](https://arxiv.org/html/2604.20246#S3.SS3.SSS0.Px1.p1.7 "Flow-Matching Action Head. ‣ 3.3 VLA Policy ‣ 3 Methodology ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"). 
*   [41]F. Zhu, H. Wu, S. Guo, Y. Liu, C. Cheang, and T. Kong (2025)IRASim: A Fine-Grained World Model for Robot Manipulation. In Proceedings of the IEEE/CVF International Conference on Computer Vision,  pp.9834–9844. Cited by: [§2.3](https://arxiv.org/html/2604.20246#S2.SS3.p2.1 "2.3 World Models for Robotics ‣ 2 Related Works ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"). 
*   [42]B. Zitkovich, T. Yu, S. Xu, P. Xu, T. Xiao, F. Xia, J. Wu, P. Wohlhart, S. Welker, A. Wahid, et al. (2023)RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. In Conference on Robot Learning,  pp.2165–2183. Cited by: [§1](https://arxiv.org/html/2604.20246#S1.SS0.SSS0.Px1.p1.1 "Background and related progress. ‣ 1 Introduction ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment"), [§2.1](https://arxiv.org/html/2604.20246#S2.SS1.p1.1 "2.1 Vision–Language–Action Models ‣ 2 Related Works ‣ Cortex 2.0: Grounding World Models in Real-World Industrial Deployment").

