Title: Do World Action Models Generalize Better than VLAs? A Robustness Study

URL Source: https://arxiv.org/html/2603.22078

Published Time: Thu, 02 Apr 2026 00:19:50 GMT

Markdown Content:
# Do World Action Models Generalize Better than VLAs? A Robustness Study

##### Report GitHub Issue

×

Title: 
Content selection saved. Describe the issue below:

Description: 

Submit without GitHub Submit in GitHub

[![Image 1: arXiv logo](https://arxiv.org/static/browse/0.3.4/images/arxiv-logo-one-color-white.svg)Back to arXiv](https://arxiv.org/)

[Why HTML?](https://info.arxiv.org/about/accessible_HTML.html)[Report Issue](https://arxiv.org/html/2603.22078# "Report an Issue")[Back to Abstract](https://arxiv.org/abs/2603.22078v2 "Back to abstract page")[Download PDF](https://arxiv.org/pdf/2603.22078v2 "Download PDF")[](javascript:toggleNavTOC(); "Toggle navigation")[](javascript:toggleReadingMode(); "Disable reading mode, show header and footer")[](javascript:toggleColorScheme(); "Toggle dark/light mode")
1.   [Abstract](https://arxiv.org/html/2603.22078#abstract1 "In Do World Action Models Generalize Better than VLAs? A Robustness Study")
2.   [1 Introduction](https://arxiv.org/html/2603.22078#S1 "In Do World Action Models Generalize Better than VLAs? A Robustness Study")
3.   [2 Related Works](https://arxiv.org/html/2603.22078#S2 "In Do World Action Models Generalize Better than VLAs? A Robustness Study")
    1.   [2.1 Vision Language Action (VLAs)](https://arxiv.org/html/2603.22078#S2.SS1 "In 2 Related Works ‣ Do World Action Models Generalize Better than VLAs? A Robustness Study")
    2.   [2.2 World Model in Robotics](https://arxiv.org/html/2603.22078#S2.SS2 "In 2 Related Works ‣ Do World Action Models Generalize Better than VLAs? A Robustness Study")
        1.   [2.2.1 World Models as Learned Simulators](https://arxiv.org/html/2603.22078#S2.SS2.SSS1 "In 2.2 World Model in Robotics ‣ 2 Related Works ‣ Do World Action Models Generalize Better than VLAs? A Robustness Study")
        2.   [2.2.2 World Models as Auxiliary Tasks](https://arxiv.org/html/2603.22078#S2.SS2.SSS2 "In 2.2 World Model in Robotics ‣ 2 Related Works ‣ Do World Action Models Generalize Better than VLAs? A Robustness Study")
        3.   [2.2.3 World Action Models (WAMs)](https://arxiv.org/html/2603.22078#S2.SS2.SSS3 "In 2.2 World Model in Robotics ‣ 2 Related Works ‣ Do World Action Models Generalize Better than VLAs? A Robustness Study")

[License: CC BY-NC-SA 4.0](https://info.arxiv.org/help/license/index.html#licenses-available)

 arXiv:2603.22078v2 [cs.RO] 01 Apr 2026

# Do World Action Models Generalize Better than VLAs? A Robustness Study

 Zhanguang Zhang 1 Corresponding to {zhanguang.zhang, yingxue.zhang}@huawei.com Zhiyuan Li 1,2 work done during internship at Huawei Canada Behnam Rahmati 1 Rui Heng Yang 1 Yintao Ma 1 Amir Rasouli 1

 Sajjad Pakdamansavoji 1 Yangzheng Wu 1 Lingfeng Zhang 1 Tongtong Cao 1 Feng Wen 1

 Xinyu Wang 1 Xingyue Quan 1 and Yingxue Zhang 1∗

1 Huawei Technologies 2 University of Toronto 

###### Abstract

Robot action planning in the real world is challenging as it requires not only understanding the current state of the environment but also predicting how it will evolve in response to actions. Vision-language-action (VLA), which repurpose large-scale vision-language models for robot action generation using action experts, have achieved notable success across a variety of robotic tasks. Nevertheless, their performance remains constrained by the scope of their training data, exhibiting limited generalization to unseen scenarios and vulnerability to diverse contextual perturbations. More recently, world models have been revisited as an alternative to VLAs. These models, referred to as world action models (WAMs), are built upon world models that are trained on large corpora of video data to predict future visual states. With minor adaptations, their latent representation can be decoded into robot actions. It has been suggested that their explicit dynamic prediction capacity, combined with spatiotemporal priors acquired from web-scale video pre-training, enables WAMs to generalize more effectively than VLAs. In this paper, we conduct a comparative study of prominent state-of-the-art VLA policies and recently released WAMs. We evaluate their performance on the LIBERO-Plus and RoboTwin 2.0-Plus benchmarks under various visual and language perturbations. Our results show that WAMs achieve strong robustness, with LingBot-VA reaching 74.2% success rate on RoboTwin 2.0-Plus and Cosmos-Policy achieving 82.2% on LIBERO-Plus. While VLAs such as π 0.5\pi_{0.5} can achieve comparable robustness on certain tasks, they typically require extensive training with diverse robotic datasets and varied learning objectives. Hybrid approaches that partially incorporate video-based dynamic learning exhibit intermediate robustness, highlighting the importance of how video priors are integrated. Overall, our findings shed light on critical insights into the strengths of WAMs relative to VLAs, as well as the remaining challenges for real-world deployment.

## 1 Introduction

Robot motion planning in real-world environments—whether for navigation, manipulation, or locomotion—remains highly challenging. A key difficulty arises from the diversity and uncertainty of real-world settings, which makes it hard for robot policies to anticipate the consequences of their actions. Recently, vision–language–action (VLA) policies(pi05; openvla2025) have emerged as a new paradigm compared to traditional motion planning approaches(elbanhawi2014sampling). By leveraging foundation models trained on large-scale vision and language data, VLA policies have demonstrated strong performance across a wide range of robotic tasks, including navigation(xu2024mobility), manipulation(zheng2026xvla), and locomotion(jiang2026wholebodyvla).

Despite their strong performance, VLAs exhibit several notable limitations, including limited generalization ability(ma2026generalvla) and a lack of robustness to distractions and cluttered environments(rasouli2025distracted). In essence, they often lack a fundamental understanding of the physical world that is necessary for consistent planning across diverse environments. To address this issue, recent work has begun incorporating world models into robotic policies. Traditionally, world models have been used primarily as simulators for model training and evaluation(cutler2015real; ha2018_worldmodels; hafner2020_dreamer). More recently, they have been increasingly integrated into robot policies—including VLAs—in a variety of ways, such as auxiliary training objectives(Chen_2025_ICCV; cen2025worldvla), planning modules(yin2025womap; gao2025adaworld), or guidance mechanisms for flow-based policies(du2025dynaguide; huang2025ladiwm; chen2026_hwm). Building on this trend, new approaches go a step further by proposing the use of world models directly as control policies(kim2026cosmos; goswami2025osviwm; liao2025genie; li2026causal; ye2026dreamzero), where the model’s latent representations are decoded into actions.

Given the many similarities between foundation models and world models, there is ongoing debate about the primary advantages of world models and whether their explicit use in planning is necessary, since foundation-based robot policies may already implicitly model the world dynamics. To investigate this question, we conduct a comparative study of state-of-the-art VLA policies and world action models (WAMs), aiming to highlight their differences under a range of contextual perturbations. More specifically, we leverage two enhanced manipulation benchmarks to evaluate the robustness of policy models: LIBERO-Plus(fei25libero-plus), which introduces seven types of perturbations to single-arm manipulation tasks, and RoboTwin 2.0-Plus, an in-house benchmark that follows a similar perturbation protocol in the two-arm Aloha-Agilex setup of RoboTwin 2.0(chen2025robotwin).

Our findings reveal that WAMs generally demonstrate strong robustness to noise, lighting and layout perturbations in both single-arm and bimanual settings. This robustness is believed to be at least partially attributable to the spatiotemporal priors inherited from their world model backbones. While classic VLAs like π 0.5\pi_{0.5}(pi05), as well as hybrid approaches like MOTUS(bi2025motus) and VLA-JEPA(sun2026vla), can achieve comparable or even superior robustness, they often require carefully curated and diverse datasets and/or explicit dynamic prediction objectives during the embodied pre-training phase. In contrast, the simplicity of the embodied pre-training phase represents a key advantage of WAMs over classic VLAs. However, the high inference overhead of WAMs remains a major challenge that limits their deployment in real-world robotic systems, with a single inference step being at least 4.8 times slower than π 0.5\pi_{0.5}. Recent GigaWorld-Policy(ye2026gigaworld) and Fast-WAM(yuan2026fast) mitigate this issue by generating actions only at inference time, while jointly predicting future visual state or conditioning state prediction on actions during training. However, they slightly underperform compared to the state-of-the-art WAM that adopts inverse dynamic model (IDM) prediction scheme. Moreover, their inference latency remains substantial compared to the π\pi-series, posing challenges for smooth real-world deployment. Further research is needed to more efficiently exploit the dynamic priors of world-model backbones and improve both training and inference efficiency.

## 2 Related Works

### 2.1 Vision Language Action (VLAs)

Recent progress in robot foundation models has focused on integrating perception, language understanding, and control within unified multimodal architectures. A central direction in this line of work is the development of vision–language–action (VLA) models that connect high-level semantic reasoning with low-level robot execution. Early work such as PaLM-E(driess2023palm) demonstrates that large language models can be adapted to embodied settings by representing visual observations and robot states as additional tokens in a transformer. This formulation shows that knowledge learned from internet-scale data can support downstream robotic manipulation and long-horizon tasks when paired with action prediction. Building on this paradigm, a series of works explored large-scale end-to-end VLA policies trained on robot demonstrations. RT-1 and RT-2(brohan2023rt1; zitkovich2023rt2) introduced transformer policies that map visual observations and language instructions directly to robot actions, showing that large multi-task datasets enable generalization across manipulation behaviors. RT-H(belkhale2024rth) further explored hierarchical policy structures that improve data efficiency and long-horizon execution. In parallel, the π 0\pi_{0} series(black2025pi0) studied scaling behavior in VLA policies, emphasizing the role of action representation and dataset diversity in enabling robust multi-task generalization.

A second line of work emphasizes data-centric scaling and cross-embodiment generalization. Octo(ghosh2024octo), trained on the Open X-Embodiment dataset(o2024open), demonstrated that a single policy can learn from heterogeneous datasets spanning multiple robots and tasks. OpenVLA(openvla2025) further advanced this direction by providing an open-source VLA implementation built on pretrained multimodal encoders, highlighting the importance of reproducible pipelines and shared datasets for scaling robot foundation models. More recent research has focused on improving reasoning and adaptation within VLA architectures. CoT-VLA(zhao2025cotvla) introduces visual chain-of-thought reasoning by predicting intermediate visual goals before generating actions, improving long-horizon task performance. ChatVLA-2(zhou2025chatvla2) explores integrating open-world reasoning capabilities from pretrained vision–language models into robotic policies. Meanwhile, X-VLA(zheng2026xvla) investigates scalable cross-embodiment learning through embodiment-specific soft prompts, and SimpleVLA-RL(li2026simplevlarl) demonstrates that reinforcement learning applied after imitation pre-training can significantly improve robustness and long-horizon execution.

### 2.2 World Model in Robotics

World models learn the internal representation of the world and can predict future visual states conditioned on actions. Although the concept has long been studied(craik1967nature; sandage1988observational; ha2018world), it has recently experienced a surge in interest and has been applied across a wide range of applications, including image retrieval(Yuanmin_2025_CVPR), autonomous driving(Hassan_2025_CVPR; Zhao_2025_CVPR), medical imaging(Yue_2025_CVPR; yang2025medical), face generation(zheng2025learning), and robotics locomotion(hao2025neural; wang2025disentangled), navigation(bar2025navigation; yao2025navmorph), object manipulation(zhen2025learning).

In the context of robotics, world models have been used in several ways, including as simulators for training and evaluation(shang2025roboscape), as auxiliary modules to boost planning policies(yin2025womap), or as policies themselves after certain adaptations(goswami2025osvi). Below we provide a brief review of each of these categories.

#### 2.2.1 World Models as Learned Simulators

In robotics and embodied AI, world models are most operationally useful when treated as learned simulators: action-conditioned generative models that produce counterfactual futures for decision-making. Rather than focusing on perceptual reconstruction or open-ended generation, this view treats the model as a computational substrate for control, where imagined rollouts must remain action-grounded, decision-relevant, and stable under closed-loop distribution shift(ha2018world; hafner2019_planet; hafner2020_dreamer). Importantly, simulation does not need to occur in pixel space. Latent dynamics models—such as recurrent state-space models and predictive state representations—enable compact rollouts as long as task-relevant structure is preserved(littman2001_psr). This shift from reconstruction fidelity toward control-aligned prediction exposes a key tension: improvements in likelihood or visual fidelity do not necessarily translate to better planning or policy learning, especially under long-horizon drift or objective mismatch(lambert2020_objective_mismatch; soh2026_action_hallucination).

Learned simulators are used in two main ways. First, they support explicit planning, where policies optimize actions by querying the model at test time. Latent planners such as PlaNet, PETS, and MBPO perform model predictive control with varying strategies to mitigate model error(hafner2019_planet; chua2018_pets; janner2019_mbpo). More recently, V-JEPA 2-AC enables planning from image goals in latent space after large-scale pre-training on video data(assran2025vjepa2). H-WM further explores hierarchical planning by coupling symbolic task-level prediction with visual state prediction, using intermediate subgoals to guide long-horizon robotic task and motion planning(chen2026h). Video-based and foundation-scale simulators extend this idea to high-dimensional observations, enabling goal-directed planning from images or large-scale video pre-training(finn2017_visualforesight; ebert2018_visualforesight). NVIDIA presents Cosmos-Predict2.5 as a multimodal world-generation model designed to support robotics planning and large-scale synthetic rollouts(nvidia2025_cosmos; ali2025_cosmospredict25). Second, world models enable imagination-based policy optimization, where policies are trained directly on simulated trajectories, as in World Models, Dreamer, and its successors(ha2018_worldmodels; hafner2020_dreamer; hafner2023_dreamerv3). Hybrid approaches such as TD-MPC combine short-horizon model rollouts with value bootstrapping to limit compounding error while retaining sample efficiency(hansen2022_tdmpc; hansen2024_tdmpc2). Across these paradigms, the simulator’s value lies less in perceptual realism and more in producing predictions that are reliable for control.

In visual navigation, world models are increasingly used to support tasks such as image-goal navigation. The Navigation World Model (NWM)(bar2025navigation) enables goal-conditioned navigation by using a Model Predictive Control (MPC)(kouvaritakis2016model) framework to simulate and evaluate candidate trajectories. Building on this idea, a lightweight one-step world model(shen2026efficient) employs a 3D U-Net with spatial–temporal attention to predict future observations, improving robustness and efficiency in multi-modal goal navigation tasks. More recently, MindJourney(yang2025mindjourney) leverages world models for test-time scaling to enhance a vision–language model’s understanding of 3D dynamics.

Table 1: Summary of world action models (WAMs). MOT: mixture-of-transformers, indicating that separate transformer backbones are used for video and action streams, with cross-modal interactions facilitated via attention at each layer. Note that the released LingBot-VA model adopts a unified transformer for both modalities, instead of the mixture-of-transformer architecture described in the paper. Pretrain Free: the method does not require task-agnostic embodied pre-training. Causal Pred.: action prediction is conditioned on the generated visual state, or vice versa. AR Gen.: auto-regressive generation.

| Model | #Params | Backbone | MOT | Pretrain Free | Causal Pred. | AR Gen. |
| --- | --- | --- | --- | --- | --- | --- |
| VPP (hu2025video) | 1.5B | Stable Video Diffusion (SVD) | ✗ | ✗ | ✓ | ✗ |
| GE-Act (liao2025genie) | 2.2B | LTX-Video-2B | ✓ | ✗ | ✗ | ✓ |
| Cosmos-Policy (kim2026cosmos) | 2B | Cosmos-Predict2-2B | ✗ | ✓ | ✗ | ✗ |
| LingBot-VA (li2026causal) | 5.3B | Wan2.2-5B | ✓? | ✗ | ✓ | ✓ |
| DreamZero (ye2026dreamzero) | 14B | Wan2.1-14B | ✗ | ✗ | ✗ | ✓ |
| GigaWorld-Policy (ye2026gigaworld) | >5B | Wan2.2-5B | ✗ | ✗ | ✓ | ✗ |
| Fast-WAM (yuan2026fast) | 6B | Wan2.2-5B | ✓ | ✓ | ✗ | ✗ |

#### 2.2.2 World Models as Auxiliary Tasks

In this scheme, world knowledge is acquired by augmenting the policy with predictive capacities. This is achieved by introducing auxiliary prediction tasks(Chen_2025_ICCV; cen2025worldvla; zhang2025dreamvla; sun2026vla; wang2026unified) or by co-training the policy with a predictive model(wang2026unified; li2025unified). In Chen_2025_ICCV, a GPT-like architecture is proposed in which video content is converted into latent motion tokens that are incorporated with the text and the initial image observation. The model is first pre-trained to predict future video tokens and then finetuned with an action head to generate motions. WorldVLA(cen2025worldvla) enhances the VLA policy by adding image prediction component to generate future scenes given the observation and current action. DreamVLA(zhang2025dreamvla) augments the policy with three auxiliary predictive tasks: semantic, depth and dynamic prediction. In sun2026vla, the authors employ a teacher–student training framework in which a predictive video encoder generates dynamic targets from future visual states, while the VLM backbone produces latent action representations using only the current observation. These two latent representations are then mapped and combined to form future states, which are learned through an alignment loss. The authors of wang2026unified achieve multitasking by using interleaved representations that combine discrete vision, language, and action tokens as input. Unified World Model(zhu2025unified) integrates action and video diffusion processes within a unified transformer architecture, with independent diffusion timesteps governing each modality. Following this scheme, the model can learn a forward dynamics, an inverse dynamics, and a video generator. MOTUS(bi2025motus) introduces a Mixture-of-Transformers (MoT) architecture that unifies video and action generation, enabling training across five tasks with diverse data types. The model in li2025unified learns a policy by decoupling video-action decoding. Via a masking operation in video and action space, the model is induced to forward and inverse dynamics, as well as planning capability. For the navigation task. UniWM dong2025unified integrates prediction and control within a unified multimodal framework, first predicting actions and then autoregressively reconstructing future visual observations, with the reconstruction loss serving as an auxiliary supervision for action prediction.

#### 2.2.3 World Action Models (WAMs)

Despite the success of the VLAs aforementioned, they are mostly finetuned from VLM backbone models pretrained with an auto-regressive next-token prediction objective. While this language-centric pre-training enables models to capture high-level visual semantics, it often neglects the perception and prediction of fine-grained world dynamics that are essential for precise robotic control. With recent advances in action-conditioned video generation, a growing body of research has begun to explore adapting video generation models for policy learning(bi2025motus; hu2025video; kim2026cosmos; li2026causal; ye2026dreamzero; ye2026gigaworld; yuan2026fast). Video Prediction Policy (VPP) by hu2025video is among the earliest efforts to repurpose a video generation backbone for robot action generation. Specifically, a video diffusion model is first pretrained on text-guided video prediction task, and then further adapted to generate robot actions with a diffusion policy head conditioned on visual features encoded by the video model. Empirical results show that the video pre-training stage is crucial for the observed performance improvements. Building on this paradigm, mimic-video(pai2025mimicvideo) leverages a language-conditioned video generation model (i.e., Cosmos-Predict2-2B(agarwal2025cosmos), while retaining the same two-stage training scheme. It introduces a flow matching-based action decoder that is trained from scratch and serves as an inverse dynamic model (IDM). GE-Act(liao2025genie), built on the Genie Envisioner (GE) platform, follows a similar strategy. Leveraging a pretrained video diffusion model backbone (i.e., LTX-Video-2B), it introduces a lightweight, flow-matching action decoder which maps video model encoded latent features to robot action trajectories. Despite the success of mimic-video and GE-Act, training an additional action decoder from scratch can disrupt the latent space learned by the video model and incurs extra training cost. Instead, Cosmos-policy(kim2026cosmos) minimally adapts the diffusion process of the video generation model Cosmos-Predict2, encoding the robot state, future image and value estimates directly as latent frames. With these lightweight architectural modifications, the model is finetuned under joint training objectives for policy, world model and value prediction. The resulting model supports both direct policy generation and model-based planning through its predicted future state and value estimates. LingBot-VA(li2026causal) and DreamZero(ye2026dreamzero) enhance causal reasoning capacity of video-based policy model by unifying future visual state prediction and action inference within an interleaved sequence and autoregressively generating future predictions conditioned on previous step’s outputs. This formulation enables efficient KV-cache memory integration while enforcing causal consistency—both of which are crucial for long-horizon robotic tasks. While using different video generation backbones, both methods address the challenge of slow inference of video models for robot control. To improve real-time performance, they accelerate inference through techniques such as asynchronous inference pipeline and partial video denoising, among other optimizations.

GigaWorld-Policy(ye2026gigaworld) and Fast-WAM(yuan2026fast) aim to mitigate the slow inference time of WAMs by treating video generation as optional at test time. GigaWorld-Policy conditions future visual state prediction on actions, whereas Fast-WAM jointly generate both modalities. Both designs enable test-time action prediction without explicitly generating visual state, as required in other IDM-based approaches (e.g., LingBot-VA), thereby significantly reducing the inference latency. However, the reported inference latencies of GigaWorld-Policy (360 ms) and Fast-WAM (190 ms) remain substantially higher than that of π 0\pi_{0}(black2025pi0) on a consumer-level device (73 ms), leaving considerable room for improvement. Moreover, their performance on RoboTwin 2.0 slightly lags behind the state-of-the-art LingBot-VA, raising questions about the influence of causal design on overall performance. Beyond inference speed, Fast-WAM also demonstrates great data efficiency in both simulation and real-world tasks by requiring only task-specific finetuning, without costly embodied pre-training on large-scale task-agnostic data.

The key characteristics of recent WAMs are summarized in [Table˜1](https://arxiv.org/html/2603.22078#S2.T1 "In 2.2.1 World Models as Learned Simulators ‣ 2.2 World Model in Robotics ‣ 2 Related Works ‣ Do World Action Models Generalize Better than VLAs? A Robustness Study"). We only consider the methods that leverage a pretrained world model backbone for robot action generation, with minimal or no architecture modifications. Consequently, MOTUS is excluded from this list: although it adopts a pretrained Wan2.2-5B for video generation, it relies on an additional VLM for action generation rather than the world model backbone itself. Typically, lightweight modifications are introduced to the video backbones to encode robot joint state and produce robot actions. While some approaches (e.g., Cosmos-Policy, GigaWorld-Policy) employ a unified transformer backbone for both video and action streams, others such as GE-Act and Fast-WAM adopt mixture-of-transformers (MOT) architectures, using a smaller, dedicated transformer for action modeling. Regarding training objectives, WAM models are typically trained to predict future visual state and actions, and in most cases require embodied pre-training on large-scale robot data. The design of causal attention also varies across methods. For instance, LingBot-VA conditions action generation on predicted visual state, whereas Cosmos-Policy and DreamZero jointly denoise both modalities. In contrast, GigaWorld-Policy conditions video generation on actions. Autoregressive generation is employed in GE-Act, LingBot-VA and DreamZero to condition prediction on historical context, improving temporal consistency and inference efficiency.

Table 2: Training data of VLAs and WAMs across different training stages.Training stages: Embodied Pre-training, Embodied Post-training, Task-specific Finetuning. PT: filtered data for embodied post-training.

| Model | Semantic Understanding & Spatial Grounding | General Video | Robot Data | Task-specific Data |
| --- |
| VLAs |
| \arrayrulecolor lightgray π 0\pi_{0} | - | - | Cross-embodiment (>10k h) | Trajectories (5–100 h) |
| OpenVLA-OFT | - | - | Cross-embodiment (970k) | Trajectories (20–300) |
| X-VLA | - | - | Cross-embodiment (288k) | Trajectories (50) |
| π 0.5\pi_{0.5} | High-level planning | - | Mobile manipulation (400 h) | Trajectories (1–20 h) |
|  | Web data-VQA |  | Multi-env. tabletop |  |
|  | Web data-captioning |  | Cross-embodiment |  |
|  | Web data-grounding |  | Mobile manipulation (PT) |  |
|  | High-level planning (PT) |  | Multi-env. tabletop (PT) |  |
|  | Verbal instruction |  |  |  |
|  | Web data (PT) |  |  |  |
| \arrayrulecolor black VLA + WM |
| \arrayrulecolor lightgray VLA-JEPA | - | Human Ego (220k) | Single-embodiment (76k) | Trajectories (100) |
| MOTUS | - | Human Ego (231k) | Cross-embodiment (781k) | Trajectories (100) |
|  |  |  | Cross-embodiment (781k) |  |
|  |  |  | Task-agnostic data (1k) |  |
| \arrayrulecolor black WAMs |
| \arrayrulecolor lightgray Cosmos-Policy | - | - | - | Trajectories (185) |
| DreamZero | - | - | Single-embodiment (500 h) | Trajectories (12–40 h) |
| GE-Act | - | - | Single-embodiment (3k h) | Trajectories (1 h) |
| LingBot-VA | - | - | Cross-embodiment (16k h) | Trajectories (50) |
| GigaWorld-Policy | - | Human Ego (4.5k h) | Cross-embodiment (6.7k h) | Trajectories (50) |
| Fast-WAM | - | - | - | Trajectories (60 h) |

 Experimental support, please [view the build logs](https://arxiv.org/html/2603.22078v2/__stdout.txt) for errors. Generated by [L A T E xml![Image 2: [LOGO]](blob:http://localhost/70e087b9e50c3aa663763c3075b0d6c5)](https://math.nist.gov/~BMiller/LaTeXML/). 

## Instructions for reporting errors

We are continuing to improve HTML versions of papers, and your feedback helps enhance accessibility and mobile support. To report errors in the HTML that will help us improve conversion and rendering, choose any of the methods listed below:

*   Click the "Report Issue" () button, located in the page header.

**Tip:** You can select the relevant text first, to include it in your report.

Our team has already identified [the following issues](https://github.com/arXiv/html_feedback/issues). We appreciate your time reviewing and reporting rendering errors we may not have found yet. Your efforts will help us improve the HTML versions for all readers, because disability should not be a barrier to accessing research. Thank you for your continued support in championing open access for all.

Have a free development cycle? Help support accessibility at arXiv! Our collaborators at LaTeXML maintain a [list of packages that need conversion](https://github.com/brucemiller/LaTeXML/wiki/Porting-LaTeX-packages-for-LaTeXML), and welcome [developer contributions](https://github.com/brucemiller/LaTeXML/issues).

BETA

[](javascript:toggleReadingMode(); "Disable reading mode, show header and footer")

