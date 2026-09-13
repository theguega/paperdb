# **SG-VLA: Learning Spatially-Grounded Vision-Language-Action Models for Mobile Manipulation** 

Ruisen Tu<sup>1</sup> , Arth Shukla<sup>1</sup> , Sohyun Yoo<sup>1</sup> , Xuanlin Li<sup>1</sup> , Junxi Li<sup>1</sup> , Jianwen Xie<sup>2</sup> Hao Su<sup>1</sup> , Zhuowen Tu<sup>1</sup> 

1UC San Diego 2Lambda, Inc. 

## **Abstract** 

## **1. Introduction** 

_Vision-Language-Action (VLA) models have demonstrated promising capabilities for robotic control, yet their performance in complex household environments remains far from optimal. Mobile manipulation tasks require robots to simultaneously reason about global scene layout, finegrained object geometry, and high-dimensional continuous actions, making standard imitation learning insufficient for robust control. In this work, we introduce a framework for learning spatially-grounded VLA models that strengthens both perception and representation learning through auxiliary task co-training and multi-modal input enhancement. Our method addresses the challenge of controlling a 13dimensional action space involving coordinated base motion, arm articulation, and gripper actuation. To enrich spatial understanding, the model incorporates multi-view RGB observations, depth cues, and short temporal history, providing complementary perspectives of both global scene structure and local manipulation context. To further improve representation quality, we co-train a suite of auxiliary decoders that reconstruct interpretable intermediate signals—including global robot position, joint configurations, grasp affordances, target-object relative pose, and segmentation masks—from shared visual-language features. These auxiliary objectives supply dense supervision that encourages the backbone to develop spatially grounded, manipulation-aware latent representations. Through extensive evaluation on home rearrangement tasks, our approach achieves consistent improvements across picking, placing, opening, and closing operations, substantially outperforming direct imitation learning applied to the same architecture. We hypothesize that the combination of spatially informative inputs and structured auxiliary supervision enhances both scene reasoning and low-level control accuracy. Overall, our findings suggest that spatial grounding through auxiliary and multi-modal learning provides a strong direction for scaling VLA models toward generalpurpose domestic robots._ 

Vision-Language-Action (VLA) models, which leverage the powerful representations of pre-trained VisionLanguage Models (VLMs) [15, 32], have emerged as a leading paradigm for translating natural language commands into robotic actions [4, 12, 17, 25]. These models have demonstrated remarkable success in learning a wide array of skills, setting new benchmarks for generalization and semantic understanding in robotic control [3, 18, 22]. 

However, much of this recent success has been concentrated in constrained, table-top environments involving single or dual-arm manipulators [16, 17]. While valuable, these settings do not capture the complexities of real-world domestic scenarios. Household tasks require robots to navigate through spaces, interact with articulated objects like doors and drawers, and reason about their own state relative to a dynamic, unstructured environment. This leap from static to mobile manipulation introduces significant challenges [10, 29], including partial observability, highdimensional continuous action spaces that fuse navigation and manipulation, and a greater need for robust spatial and scene-level reasoning. 

Despite the growing interest in applying VLA models to mobile manipulation, this remains a relatively nascent area of research. Current approaches to household robotics have primarily relied on modular systems that decompose tasks into separate navigation and manipulation components [7, 21, 33], or end-to-end learning methods that struggle with the complexity of unified control [6, 10]. The application of VLA models to mobile manipulation represents an emerging paradigm that seeks to harness the rich semantic understanding and reasoning capabilities of large-scale vision-language pre-training for these challenging scenarios. However, existing VLA architectures, when directly applied to mobile manipulation tasks, exhibit substantial performance limitations. We observe that standard VLA models trained through direct imitation learning achieve only modest success rates on household tasks, suggesting that current approaches fail to adequately leverage 

1 







<!-- Start of picture text -->
(a) Head camera view (b) Hand camera view<br><!-- End of picture text -->

Figure 1. **Multi-view segmentation data used for auxiliary task training.** Each group shows (left) the original RGB observation, (middle) segmentation masks for all objects in the scene with different colors representing different object instances, and (right) the processed binary mask highlighting only the target object of interest. (a) Head camera view provides a global perspective of the manipulation scene. (b) Hand camera view offers a close-up perspective focused on the manipulation area. The target object masks are derived from the full segmentation by identifying the target object ID and creating binary masks to focus the model’s attention on the relevant manipulation target. 

the sophisticated reasoning abilities inherent in pre-trained VLMs. 

## **2. Related Work** 

### **2.1. Vision-Language-Action models** 

In this work, we investigate how to bridge this gap and explore the applicability of VLA models to complex mobile manipulation tasks in household settings. We hypothesize that the standard approach of direct imitation learning—predicting a high-dimensional action vector from a visual-language representation—provides insufficient supervisory signal for the model to learn the rich, multifaceted understanding required for these tasks. To address this, we introduce **SG-VLA** , a framework that enhances the learning process through two primary strategies: enriching the model’s sensory input and providing denser learning signals through auxiliary co-training. 

Our work makes the following contributions: 

- We propose an efficient co-training strategy that leverages a shared visual-language backbone to simultaneously predict actions and a suite of auxiliary tasks, which act as a powerful form of explicit supervision, forcing the model to learn more interpretable and spatially aware representations from its latent features. 

- We systematically explore how different input modalities, specifically multi-view imagery and depth information, can provide richer spatial context to improve VLA performance in mobile manipulation. 

- We validate our approach on the challenging ManiSkillHAB benchmark [28], showing that SG-VLA achieves average success rates of 73% compared to 60% for direct imitation learning on home rearrangement tasks, demonstrating the effectiveness of our proposed enhancements on a common VLA architecture. 

These findings suggest that enriching both the inputs and the training objectives is a critical and promising direction for scaling VLA models to the complexities of real-world domestic robotics. 

Vision-Language-Action (VLA) models have emerged as a dominant paradigm for robotic control, leveraging largescale pretrained models to translate multimodal prompts into actions. Pioneering works such as [5, 20, 25–27, 35, 36] demonstrated that end-to-end training on large datasets [8, 9] could produce highly capable policies. Some models discretize the continuous action space [2, 4, 16], enabling the VLM backbone [15] to directly predict action tokens in an autoregressive fashion. Others append a specialized action expert [3, 17, 31], which generates continuous actions from the VLM’s latent features. Despite their success, the capabilities of these VLA frameworks are predominantly demonstrated on tabletop manipulation tasks, leaving the challenges of mobile manipulation largely unaddressed. To address this gap, our work investigates how VLA representations can be enhanced specifically for mobile manipulation. We show that by augmenting the learning process with a suite of auxiliary objectives and enriching the model’s perception with multi-modal inputs, we can significantly improve performance on complex household tasks [28, 29]. 

### **2.2. Mobile Manipulation** 

Solving mobile manipulation requires the tight synergy of locomotion and arm control to carry out human instructions. One dominant strategy employs a hierarchical system where a Vision-Language Model (VLM) [1] serves as a high-level planner [30], decomposing instructions into simpler subtasks. These subtasks are then handled by separate, specialized policies for navigation and manipulation [7, 10, 11, 13, 14, 23]. This modularity, however, isolates the foundation model’s rich, pre-trained knowledge from the final motor control, limiting its direct influence on physical interaction. In contrast, our approach utilizes 

2 



<!-- Start of picture text -->
Input VLM Backbone Auxiliary Module<br>Observation<br>Global Position<br>Decoder GT Global Position: [x, y, z]<br>Grasp Success  GT Grasp Label: True/False<br>Decoder<br>Attention<br>Head RGB Hand RGB Layers GT Obj Pose: [3D pos, quaternion]<br>Object Position<br>Decoder<br>GT Joint Pose: 12D joint config<br>Joint Pose Decoder<br>Head Depth Hand Depth GT Segmentation:<br>Segmentation<br>Language Instruction Latent  Decoder<br>Representation<br>"Pick up the apple" Tra i ning only<br>Qwen2.5VL-0.5B Action ∆X, ∆z, ∆q, ∆G<br>De-tokenizer 13 Dimension Robot Action<br>Loss<br>Reconstruction<br><!-- End of picture text -->

Figure 2. **SG-VLA Architecture Overview.** The model processes multi-modal inputs, including RGB images and normalized depth maps from head and hand cameras, alongside a natural language instruction. During training time, the latent representation from LLM is then passed to the decoders for auxiliary task predictions. Finally, the model predicts a 13-dimensional action vector, generated either directly by the LLM backbone or through a specialized Flow Matching action expert depending on task type. The action vector consists of the following: ∆ _X_ , a 3D vector representing the **base’s pose (position+orientation)** ; ∆ _z_ , a 1D scalar representing the change in the **torso’s height** ; ∆ _q_ , a 7D vector representing the change in the **arm’s joint angles** ; ∆ _G_ , a 2D vector representing the change in the **gripper’s state** (one for each finger). 

a VLA-native framework that deeply embeds the reasoning power of foundation models directly into the action generation loop, enabling a more potent application of web-scale priors for tackling complex household tasks. 

## **3. Dataset** 

We adopt the data generation pipeline from ManiSkillHAB [28], a benchmark for low-level manipulation in home rearrangement tasks, and extend it to generate demonstration data enriched with auxiliary task annotations. The dataset comprises simulation-based trajectories across three long-horizon household tasks: TidyHouse, PrepareGroceries, and SetTable, totaling 44K episodes with 1.4M transitions. 

**Task Composition.** Each long-horizon task is decomposed into 4 fundamental manipulation subtasks: **Pick** , **Place** , **Open** , and **Close** , with varying difficulty levels. TidyHouse involves pick-and-place operations with medium to hard difficulty objects and receptacles. PrepareGroceries focuses on hard-difficulty pick-and-place scenarios requiring precise manipulation. SetTable includes all 4 subtasks, spanning easy to medium difficulty levels. Pick and place tasks require fine-grained manipulation control for accurate grasping and positioning, whereas open and close tasks emphasize broader locomotion and approach strategies. The diversity in task complexity and object types provides comprehensive coverage of household manipula- 

#### tion scenarios. 

**Auxiliary Task Data Distribution.** We strategically distribute auxiliary task training based on task relevance and data availability. For auxiliary tasks involving **global position, grasp state, and joint position (qpos) prediction** , we utilize the complete SetTable dataset (8K episodes, 240K transitions), which provides comprehensive robot state information across diverse manipulation scenarios. For **segmentation masks and object pose estimation** , we leverage pick-and-place data from all three tasks (40K episodes, 1.2M transitions), as these auxiliary tasks are only meaningful in scenarios involving target object manipulation and require clear object identification. Figure 1 shows an example of the segmentation data. This task-specific data allocation ensures that each auxiliary decoder receives relevant and sufficient training examples while maximizing the utilization of our diverse demonstration dataset. 

## **4. Model** 

### **4.1. Overall Architecture** 

SG-VLA is a lightweight 1.3B parameter model consisting of a pre-trained Prismatic VLM [15] backbone and an optional action expert, designed to efficiently handle multimodal inputs for robotic control tasks. The VLM backbone comprises three key components: (1) a **visual encoder** that fuses complementary features from DINOv2 [24] (improving spatial understanding) and SigLIP [34] (providing rich 

3 

semantic representations) following dual-encoder approach in [16], enabling the model to process both RGB and depth inputs from multiple camera viewpoints; (2) a **large language model backbone** (Qwen2.5-0.5B [32]) that serves as the central reasoning component, processing textual instructions and integrating multi-modal information for decision making; and (3) a **trainable projector** that maps high-dimensional vision features to the language embedding space, allowing seamless fusion of visual and linguistic representations within the transformer architecture. 

Additionally, we incorporate an optional 100M parameters **Flow Matching** [19] **action expert** following _π_ 0 [3], which generates continuous, high-precision actions conditioned on the rich latent representations extracted from the VLM backbone. This dual-prediction architecture provides flexibility in action generation: the model can either directly predict discrete action tokens, or leverage the action expert to generate continuous actions. The choice between these prediction modes can be adapted based on specific task requirements. The complete architecture, illustrating the flow from multi-modal inputs through the VLM backbone to auxiliary decoders and action prediction, is shown in Figure 2. 

### **4.2. Input Modalities** 

To enhance the spatial understanding and situational awareness of our VLA model, we systematically investigate the impact of different input modalities beyond standard singleview RGB observations. Specifically, we explore three key modality enhancements: (1) **Multi-view RGB inputs** that incorporate both head-mounted and hand-mounted camera observations, providing complementary perspectives of the manipulation scene compared to relying solely on head observations; (2) **Depth information** from head and hand cameras and normalized through Eq. 1, which offers explicit geometric understanding of object distances and spatial relationships in the environment; and (3) **Temporal context** through historical observations from the past 4 timesteps, enabling the model to leverage sequential information for better action planning. We conduct comprehensive ablation studies to evaluate the individual and combined effects of these modality additions on task performance. Our experiments reveal that multi-view RGB observations combined with their corresponding depth images yields the best performance, significantly improving the model’s ability to understand complex spatial relationships and object interactions in household manipulation tasks. The detailed ablation results and analysis of each modality’s contribution are presented in Section 6. 



### **4.3. Decoders** 

We design auxiliary decoders that operate on the VLM backbone’s latent representations to enable multi-task learning. The decoders reconstruct different aspects of the current state, providing complementary supervision signals for various aspects of robotic manipulation. Structure details are listed in Table 1 in supplementary material. 

|**Decoder**|**Type**|**Key Components**|
|---|---|---|
|Global Pose|MLP|Proj (896_→_512), 3-layer MLP, Avg<br>Pooling|
|Grasp Succ.|MLP|Proj (896_→_512), 3-layer MLP, Avg<br>Pooling|
|Object Pose|MLP|Proj (896_→_512), 3-layer MLP, Quat<br>Norm|
|Joint Pose|Transf.|12 Mask Tokens, 2-layer Trans-<br>former, Sine-Cos Pos|
|Mask|CNN|4-stage<br>Transpose<br>Conv,<br>Batch-<br>Norm, GELU|



Table 1. Detailed decoder specifications for the auxiliary training objectives used in SG-VLA. 

**MLP-based Decoders.** We implement three regression and classification decoders using similar MLP architectures with progressive dimensionality reduction. The **Global Position Decoder** predicts the robot’s 2D coordinates (x, y), the **Grasp Success Decoder** performs binary classification to determine grasping state, and the **Object Pose Decoder** predicts 7-dimensional object poses (3D position + quaternion orientation). We adopt MLPs for these tasks because their targets are relatively low-dimensional vectors that can be effectively reconstructed from the compact latent representation produced by the VLM backbone, making lightweight MLPs a natural and efficient architectural choice. Their respective loss functions are defined in Eq. 2 and Eq. 3, where **p** ˆ and **p** are predicted and ground truth global positions, _y_ and _y_ ˆ are ground truth and predicted grasp labels, **t** represents 3D position, and **q** represents quaternion orientation. 

**Transformer-based Joint Pose Decoder.** For the 12dimensional joint configuration prediction, we employ a Transformer architecture with learnable mask tokens that attend to VLM features through multi-head self-attention. Unlike the previous low-dimensional targets, joint angles form a higher-dimensional vector with strong interdependencies across joints. Self-attention allows each predicted joint angle to reference others through flexible, global interactions, capturing the underlying kinematic structure of the robot. This design enables the model to leverage both spatial understanding from VLM features and relational dependencies among the joints. The decoder uses fixed sinecosine positional encodings and MSE loss for joint angle 

4 



<!-- Start of picture text -->
Stage 1 Decoders  a Stage 2 Decoders  a Stage 3<br>A A Legend<br>B B Forward<br>... ... Stop gradient<br>Enable gradient<br>Visual  E Visual  E Visual  Trainable<br>Encoder  Encoder  Encoder<br>Frozen<br>+ Discretized + Discretized +<br>LLM  Action Tokens LLM  Action Tokens LLM<br>Backbone Backbone Backbone<br>Action<br>... ... Head<br><!-- End of picture text -->

Figure 3. **Multi-stage Training Scheme.** Stage 1: Decoder adaptation phase where auxiliary decoders are trained while gradient flow to the VLM backbone is blocked, allowing decoders to learn from fixed VLM representations. Stage 2: Joint refinement phase with full gradient flow enabled, co-training all auxiliary decoders with the VLM backbone. Stage 3: Action head training phase where the VLM backbone is frozen to train the flow matching action head in isolation. 

regression as shown in Eq. 4, where **J**<sup>ˆ</sup> and **J** are predicted and ground truth 12-dimensional joint configurations. 

**CNN-based Mask Decoder.** The segmentation decoder generates 128×128 binary masks for target objects using an efficient CNN upsampling pathway. We choose a CNNbased architecture because segmentation inherently requires spatially structured predictions, and convolutional decoders remain the most effective and computationally efficient approach for dense image reconstruction tasks. The model is trained using binary cross-entropy loss as shown in Eq. 4, where **M**<sup>ˆ</sup> and **M** are predicted and ground truth segmentation masks. 



## **5. Training Scheme** 

We adopt a progressive multi-stage training approach to effectively integrate auxiliary decoders and flow-matching action head with the pre-trained VLM backbone while preserving the model’s foundational capabilities. 

### **5.1. Preliminary Experiment** 

Our preliminary experiments revealed that directly cotraining randomly initialized auxiliary decoders with the VLM backbone led to performance degradation. This phenomenon occurs because the untrained decoders generate large, noisy gradients that interfere with the learned representations in the pre-trained VLM. 





**Multi-task Loss Function.** The total training loss combines the main action prediction loss with weighted auxiliary losses: 



where the _λ_ s are task-specific weighting coefficients tuned to balance contributions across diverse auxiliary objectives. Each decoder operates independently on shared VLM representations, enabling simultaneous learning of complementary manipulation aspects. 

### **5.2. Two-Stage Progressive Training** 

To address this challenge, we implement a two-stage training strategy that balances auxiliary task learning with preservation of VLM representations: 

**Stage 1: Decoder Adaptation.** We freeze the gradient flow from auxiliary decoders to the VLM backbone, allowing only the discrete action token prediction path from the VLM to update the backbone parameters. During this phase, the randomly initialized decoders learn to interpret and utilize the fixed latent representations from the VLM without disrupting the backbone’s pre-trained knowledge, while the VLM continues learning to predict discrete action tokens for robot control. This stage enables the auxiliary decoders to adapt to the VLM’s representational space and establish reasonable baseline performance on their respective tasks, while maintaining the VLM’s core action prediction capabilities through discrete token generation. 

5 



Figure 4. **Sample execution trajectories for six household manipulation tasks in ManiSkill-HAB evaluation** . Each row shows a temporal sequence for a task performed by SG-VLA. The large frames show the global environment, while the vertical insets display head depth, head RGB, hand depth, and hand RGB observations from top to bottom. 

**Stage 2: Joint Refinement.** After the decoders have stabilized, we enable full gradient flow, allowing auxiliary task losses to backpropagate through the entire network. In this phase, the auxiliary objectives guide the refinement of VLM representations to better capture manipulation-relevant features such as spatial relationships, object properties, and robot state information. The combined supervision from multiple tasks encourages the backbone to learn more comprehensive and robust representations that benefit the primary action prediction objective. This progressive approach ensures that auxiliary tasks enhance rather than hinder the model’s learning process, as demonstrated by our ablation studies in Section 6. 

### **5.3. Training the Action Head** 

Beyond the two-stage progressive training for auxiliary decoders, we introduce an additional training phase specifically for the optional flow matching action head. Our initial experiments revealed that training the flow matching action head alongside the VLM backbone, leads to optimization difficulties. Despite preventing the flow matching gradients from affecting the VLM parameters, the denoising loss still fails to converge effectively when trained concurrently with other objectives. 

**Stage 3: Isolated Action Head Training.** To address this challenge, we implement a third training stage where we completely freeze all parameters of the pre-trained VLM 

6 

backbone and train only the flow matching action head. In this stage, the action head learns to generate continuous 13dimensional actions by denoising from the rich, frozen latent representations provided by the VLM backbone. This complete isolation allows the flow matching objective to converge properly without any interference from concurrent training processes, enabling the action head to develop robust action generation capabilities. 

## **6. Experiments** 

### **6.1. Implementation Details** 

Given that VLA models for mobile manipulation remains a nascent research direction, we acknowledge that established baselines for these complex household tasks are relatively limited. Our experimental setup therefore focuses on demonstrating the effectiveness of our proposed auxiliary training strategies against a straightforward baseline implementation of direct imitation learning on the same architecture. 

All models are implemented in PyTorch and trained on 8 NVIDIA A100 GPUs. We use an Adam optimizer with constant learning rate 2 _e_<sup>_−_5</sup> . Global batch size is set to 512. For the weights of losses in Eq. 5, we set _λpos_ = 1 _._ 0, _λgrasp_ = 5 _._ 0, _λqpos_ = 1 _._ 0, _λobj_ = 1 _._ 0, _λseg_ = 1 _._ 0. The models with segmentation decoders are train on pick and place subtasks for 5 epochs, while the others are train on all 6 subtasks for 10 epochs. We evaluate all models using the evaluation pipeline from ManiSkill-HAB [28]. For each task, we run 30 episodes and calculate the mean success rate. Sample visualization episodes can be found in Figure 4 in supplementary saterial. 

### **6.2. Input Modalities** 

Table 2 demonstrates the significant impact of enhanced input modalities on VLA model performance across different household manipulation tasks. All the models in this table are train on **SetTable** dataset for 10 epochs. The results reveal several key insights about the importance of multimodal sensory information for robotic control. 

**Baseline.** The original OpenVLA model ( _∼_ 7B parameters) [16] with single-view RGB input achieves extremely poor performance across all tasks (0.04 average success rate) after trained with same number of epochs as our models, highlighting the limitations of standard vision-language models when applied directly to complex household manipulation scenarios. This baseline establishes the critical need for enhanced sensory inputs in domestic robotics applications. 

**Multi-view and Depth Benefits.** The addition of multiview observations and depth information to the OpenVLA baseline demonstrates substantial improvements across all task categories. OpenVLA with enhanced modalities 

(multi-view + depth) achieves a dramatic 8 _×_ improvement in average success rate (from 0.04 to 0.32), with particularly notable gains in drawer manipulation tasks where success rates increase from near-zero to 0.30-0.67. This improvement validates the importance of richer sensory information for spatial reasoning in household manipulation. 

**Modality Ablation Analysis.** Performance is further enhanced when replacing the LLM backbone with the more efficient Qwen2.5-0.5B [2]. Despite being smaller than the original OpenVLA model, the Qwen-based model achieves superior results. The model with multi-view and depth inputs reaches 0.60 average success rate, nearly doubling the performance of the enhanced OpenVLA variant. The systematic ablation study on the Qwen-based model reveals the individual contributions of each modality enhancement. Multi-view inputs alone provide substantial gains over single-view baselines, achieving 0.52 average success rate. Adding depth information further improves performance to 0.60, representing a 15% relative improvement and highlighting the value of explicit geometric information for manipulation tasks. However, incorporating temporal history (past 4 actions) degrades performance to 0.49. This counter-intuitive result may indicate that the model struggles to effectively integrate temporal information, or that the evaluated tasks are sufficiently reactive that historical context provides limited additional information. 

### **6.3. Auxiliary Tasks and Progressive Training** 

Table 3 provides strong empirical evidence for the necessity of our progressive training approach when incorporating auxiliary decoders. All the models in this table are trained on **SetTable** subset for 10 epochs, with 3 epochs for stage 1 and 7 epochs for stage 2. The contrast between naive co-training (top section) and progressive training (bottom section) demonstrates the importance of proper training methodology for multi-task learning in VLA models. Table 4 demonstrates the significant impact of incorporating segmentation masks and object position prediction as auxiliary tasks. Models in this table are trained on all **Pick** and **Place** data from **SetTable** , **PrepareGroceries** and **TidyHouse** tasks for 6 epochs, with 2 for stage 1 and 4 for stage 2. 

**Failure of Naive Co-training.** When all auxiliary decoders are trained simultaneously with the VLM backbone from initialization (“SG-VLA + all” without progressive training in Table 3), performance degrades significantly across most tasks, dropping from 0.60 to 0.51 average success rate. This 15% performance degradation confirms our hypothesis that randomly initialized auxiliary decoders generate disruptive gradients that interfere with the pre-trained VLM representations. 

**Progressive Training.** The progressive training approach not only recovers the baseline performance but sub- 

7 

||Pick|Place|O|pen|Cl|ose||
|---|---|---|---|---|---|---|---|
|Method|All Obj._↑_|All Obj._↑_|Fridge_↑_|Drawer_↑_|Fridge_↑_|Drawer_↑_|Avg._↑_|
|OpenVLA [16]|0.00|0.19|0.02|0.00|0.00|0.04|0.04|
|+ Multiview|0.06|0.35|0.14|0.38|0.00|0.53|0.24|
|+ Multiview + Depth|0.12|0.41|0.43|0.30|0.00|0.67|0.32|
|Base VLM + Multiview|0.06|0.53|0.60|0.30|0.63|0.93|0.52|
|+ Multiview + Depth|**0.16**|**0.56**|**0.67**|0.36|**0.83**|**1.00**|**0.60**|
|+ Multiview + Depth + History|0.00|0.47|0.57|**0.40**|0.47|**1.00**|0.49|



Table 2. Performance comparison between models taking different modalities of inputs. The Base VLM [2] uses DINOv2 + SigLIP as dual visual encoder and Qwen2.5-0.5B [32] as LLM backbone. Reported numbers are success rates for each task. The best results are **bolded** . 

|||Pick|Place|Op|en|Cl|ose||
|---|---|---|---|---|---|---|---|---|
|Progressive|Method|All Obj._↑_|All Obj._↑_|Fridge_↑_|Drawer_↑_|Fridge_↑_|Drawer_↑_|Avg.S.R._↑_|
|No|SG-VLA|0.16|0.56|0.67|0.36|0.83|1.00|0.60|
||+ all|0.03|0.50|0.60|0.23|0.67|1.00|0.51|
|Yes|+ is<br>~~g~~rasped|**0.30**|0.53|0.83|0.57|0.80|0.93|0.66|
||+ qpos|0.23|0.67|0.87|0.70|0.90|0.90|0.71|
||+ global pos|0.07|0.27|**0.90**|0.70|**0.97**|**1.00**|0.65|
||+ all|0.13|**0.70**|0.87|**0.77**|0.90|**1.00**|**0.73**|



Table 3. Ablation study on co-training with each auxiliary task. SG-VLA represents the best model (Base VLM + Multiview + Depth) in Table2. All the models are train on SetTable subset. “+all” here includes “is ~~g~~ rasped + qpos + global pos.” The best results are **bolded** and the second-best results are underlined. 

stantially improves it. Training all auxiliary tasks with progressive methodology (“+ all” with progressive training in Table 3) achieves 0.73 average success rate, representing a 22% improvement over the best input modality configuration alone. This validates our two-stage training strategy where decoders first adapt to VLM representations before jointly refining the backbone. 

**Individual Auxiliary Task Analysis.** The ablation study results in Table 3 and 4 reveals varying contributions from different auxiliary tasks. 

- **Joint position (qpos):** Reconstructing qpos of current state provides the most substantial and consistent improvements across all task categories (0.71 average), particularly excelling in manipulation tasks like place (0.67) and drawer operations (0.70 and 0.90). This suggests that explicit joint awareness significantly enhances the model’s understanding of manipulation dynamics. 

- **Grasp label (is** **~~g~~ rasped):** Grasp label prediction offers moderate but reliable improvements (0.66 average), with notable gains in pick tasks (0.30). This indicates that predicting grasp label help model gain better manipulation state awareness. 

- **Global position (global pos):** Global position reconstruction shows more task-specific benefits, dramatically improving drawer and fridge opening tasks (0.90 and 

- 1.00) while struggling with pick-and-place operations. This indicates its particular value for navigation-heavy scenarios. 

- **Segmentation+object position (seg+obj** **~~p~~ os):** Unlike previous auxiliary tasks that showed selective improvements, segmentation and object position reconstruction provide consistent benefits across all manipulation scenarios. **Place tasks show the most substantial gains** , with improvements ranging from 55-81% across different environments. This pattern aligns with the intuitive importance of precise object localization and scene understanding for successful placement operations. **Pick tasks demonstrate more modest but consistent and notable improvements** . The relatively smaller improvements in pick tasks may reflect the continued challenge of grasp planning, which requires additional skills beyond object detection and localization. 

**Synergistic Effects.** The combined auxiliary training (“+ all”) achieves performance that generally matches or exceeds the best individual auxiliary task across most categories, with the overall average (0.73) representing nearoptimal performance. This suggests that the auxiliary tasks provide complementary rather than redundant information, with each contributing unique aspects of spatial and manipulation understanding to the overall model capability. 

8 

||SetT|able|Prepare|Grocery|Tidy|House||
|---|---|---|---|---|---|---|---|
|Method|Pick_↑_|Place_↑_|Pick_↑_|Place_↑_|Pick_↑_|Place_↑_|Avg. S.R._↑_|
|SG-VLA|0.16|0.56|0.10|0.33|0.07|0.40|0.27|
|+ seg + obj<br>~~p~~os|**0.26**|**0.78**|**0.13**|**0.60**|**0.33**|**0.73**|**0.47**|



Table 4. Performance comparison between models trained with and without Segmentation and Object Position reconstruction task. The best results are **bolded** . 

||Pick|Place|O|pen|Cl|ose||
|---|---|---|---|---|---|---|---|
|Method|All Obj._↑_|All Obj._↑_|Fridge_↑_|Drawer_↑_|Fridge_↑_|Drawer_↑_|Avg._↑_|
|SG-VLA|0.13|0.70|**0.87**|**0.77**|**0.90**|**1.00**|**0.73**|
|SG-VLA + action head|**0.27**|**0.80**|0.76|0.60|0.76|0.97|0.69|



Table 5. Performance comparison between SG-VLA trained with and without flow matching action head. The best results are **bolded** . 

### **6.4. Action Head** 

The action head is trained to predict action chunk of size 8, among which the first 2 actions are executed. Number of denoising step set to 10. Table 5 reveals the mixed effects of incorporating the flow matching action head with the overall average performance decreases from 0.73 to 0.69, indicating that the action head’s benefits are selective and come with trade-offs. 

**Task-Specific Performance Patterns.** The flow matching action head shows a clear dichotomy in its effectiveness across different manipulation primitives. **Pick** tasks benefit substantially from continuous action generation, with success rates more than doubling from 0.13 to 0.27. This improvement suggests that the fine-grained control afforded by continuous actions is particularly valuable for precise grasping motions, where small variations in approach angle, grip force, or contact points can significantly impact success. **Place** tasks also show moderate improvements (0.70 to 0.80), indicating that continuous control helps with the precise positioning required for object placement. Conversely, the action head causes notable performance drops in **Open** tasks across both fridge and drawer categories. Fridge opening decreases from 0.87 to 0.76, while drawer opening drops from 0.77 to 0.60. These results suggest that the flow matching action head excels at generating finegrained manipulation actions but struggles with mobilityoriented control, where discrete action tokens may be more suitable for coordinated base movement and navigation. 

**Implications for Task-Adaptive Control.** The mixed results strongly support our design choice to implement task-adaptive action generation, where the model can flexibly choose between discrete tokens and continuous actions based on task requirements. This flexibility allows SGVLA to leverage the precision of continuous control for manipulation-heavy tasks while maintaining the decisive- 

ness of discrete actions for mechanism operation tasks. 

## **7. Conclusion and Discussion** 

This work demonstrates that VLA models can be effectively adapted for mobile manipulation through auxiliary task co-training and enhanced input modalities. SG-VLA achieves substantial improvements by incorporating multiview depth inputs and auxiliary decoders, with progressive training proving essential for multi-task learning. These results establish that scaling VLA models beyond tabletop scenarios requires architectural and training enhancements. 

Our experiments show that auxiliary supervision encourages spatially grounded and interpretable representations, leading to better scene understanding and control precision. Together, these findings indicate that combining structured auxiliary learning and rich perception is key to extending VLA models to real-world domestic robotics. Future work could focus on exploring additional modality and auxiliary objectives in real world. 

## **8. Acknowledgment** 

This work is supported by NSF award IIS-2127544 and NSF award IIS-2433768. We thank Lambda Inc. for their compute resource help on this project. 

## **References** 

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. _arXiv preprint arXiv:2303.08774_ , 2023. 2 

- [2] Suneel Belkhale and Dorsa Sadigh. Minivla: A better vla with a smaller footprint. _arXiv preprint arXiv:2410.11195_ , 2024. 2, 7, 8 

9 

- [3] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. _π_ 0: A vision-languageaction flow model for general robot control. _arXiv preprint arXiv:2410.24164_ , 2024. 1, 2, 4 

- [4] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In _Conference on Robot Learning (CoRL)_ , 2023. 1, 2 

- [5] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. In _Robotics: Science and Systems (RSS)_ , 2023. 2 

- [6] Sixiang Chen, Jiaming Liu, Siyuan Qian, Han Jiang, Zhuoyang Liu, Chenyang Gu, Xiaoqi Li, Chengkai Hou, Pengwei Wang, Zhongyuan Wang, Renrui Zhang, and Shanghang Zhang. Ac-dit: Adaptive coordination diffusion transformer for mobile manipulation. In _Advances in Neural Information Processing Systems (NeurIPS)_ , 2025. 1 

- [7] An-Chieh Cheng, Yandong Ji, Zhaojing Yang, Zaitian Gongye, Xueyan Zou, Jan Kautz, Erdem Bıyık, Hongxu Yin, Sifei Liu, and Xiaolong Wang. Navila: Legged robot visionlanguage-action model for navigation. In _Robotics: Science and Systems (RSS)_ , 2025. 1, 2 

- [8] Open X-Embodiment Collaboration et al. Open x- embodiment: Robotic learning datasets and rt-x models. In _IEEE International Conference on Robotics and Automation (ICRA)_ , 2024. 2 

- [9] Shengliang Deng, Mi Yan, Songlin Wei, Haixin Ma, Yuxin Yang, Jiayi Chen, Zhiqi Zhang, Taoyu Yang, Xuheng Zhang, Wenhao Zhang, et al. Graspvla: a grasping foundation model pre-trained on billion-scale synthetic action data. _arXiv preprint arXiv:2505.03233_ , 2025. 2 

- [10] Zipeng Fu, Tony Z Zhao, and Chelsea Finn. Mobile aloha: Learning bimanual mobile manipulation with lowcost whole-body teleoperation. In _Conference on Robot Learning (CoRL)_ , 2024. 1, 2 

- [11] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In _International Conference on Machine Learning (ICML)_ , 2018. 2 

- [12] Physical Intelligence et al. _π_ 0 _._ 5: a vision-language-action model with open-world generalization. _arXiv preprint arXiv:2504.16054_ , 2025. 1 

- [13] Yunfan Jiang, Ruohan Zhang, Josiah Wong, Chen Wang, Yanjie Ze, Hang Yin, Cem Gokmen, Shuran Song, Jiajun Wu, and Li Fei-Fei. Behavior robot suite: Streamlining realworld whole-body manipulation for everyday household activities. _arXiv preprint arXiv:2503.05652_ , 2025. 2 

- [14] Shirin Joshi, Sulabh Kumra, and Ferat Sahin. Robotic grasping using deep reinforcement learning. _arXiv preprint arXiv:2007.04499_ , 2020. 2 

- [15] Siddharth Karamcheti, Suraj Nair, Ashwin Balakrishna, Percy Liang, Thomas Kollar, and Dorsa Sadigh. Prismatic 

   - vlms: Investigating the design space of visually-conditioned language models. In _International Conference on Machine Learning (ICML)_ , 2024. 1, 2, 3 

- [16] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An opensource vision-language-action model. In _International Conference on Machine Learning (ICML)_ , 2024. 1, 2, 4, 7, 8 

- [17] Qixiu Li, Yaobo Liang, Zeyu Wang, Lin Luo, Xi Chen, Mozheng Liao, Fangyun Wei, Yu Deng, Sicheng Xu, Yizhong Zhang, et al. Cogact: A foundational visionlanguage-action model for synergizing cognition and action in robotic manipulation. _arXiv preprint arXiv:2411.19650_ , 2024. 1, 2 

- [18] Xuanlin Li, Kyle Hsu, Jiayuan Gu, Karl Pertsch, Oier Mees, Homer Rich Walke, Chuyuan Fu, Ishikaa Lunawat, Isabel Sieh, Sean Kirmani, et al. Simplerenv: Simulated manipulation policy evaluation environments for real robot setups. _arXiv preprint arXiv:2405.05941_ , 2024. 1 

- [19] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In _International Conference on Machine Learning (ICML)_ , 2023. 4 

- [20] Songming Liu, Lingxuan Wu, Bangguo Li, Hengkai Tan, Huayu Chen, Zhengyi Wang, Ke Xu, Hang Su, and Jun Zhu. Rdt-1b: a diffusion foundation model for bimanual manipulation. _arXiv preprint arXiv:2410.07864_ , 2024. 2 

- [21] Chenhao Lu, Xuxin Cheng, Jialong Li, Shiqi Yang, Mazeyu Ji, Chengjing Yuan, Ge Yang, Sha Yi, and Xiaolong Wang. Mobile-television: Predictive motion priors for humanoid whole-body control. In _IEEE International Conference on Robotics and Automation (ICRA)_ , 2025. 1 

- [22] Oier Mees, Lukas Hermann, Erick Rosete-Beas, and Wolfram Burgard. Calvin: A benchmark for languageconditioned policy learning for long-horizon robot manipulation tasks. _IEEE Robotics and Automation Letters (RA-L)_ , 7(3):7327–7334, 2022. 1 

- [23] OpenAI, Marcin Andrychowicz, Bowen Baker, Maciek Chociej, Rafal Jozefowicz, Bob McGrew, Jakub Pachocki, Arthur Petron, Matthias Plappert, Glenn Powell, et al. Learning dexterous in-hand manipulation. _The International Journal of Robotics Research_ , 38(12-13):1420–1445, 2019. 2 

- [24] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. _arXiv preprint arXiv:2304.07193_ , 2023. 3 

- [25] Delin Qu, Haoming Song, Qizhi Chen, Yuanqi Yao, Xinyi Ye, Yan Ding, Zhigang Wang, JiaYuan Gu, Bin Zhao, Dong Wang, et al. Spatialvla: Exploring spatial representations for visual-language-action model. _arXiv preprint arXiv:2501.15830_ , 2025. 1, 2 

- [26] Scott Reed, Konrad Zolna, Emilio Parisotto, Sergio Gomez Colmenarejo, Alexander Novikov, Gabriel Barth-Maron, Mai Gimenez, Yury Sulsky, Jackie Kay, Jost Tobias Springenberg, et al. A generalist agent. In _Advances in Neural Information Processing Systems (NeurIPS)_ , 2022. 

10 

- [27] Mohit Shridhar, Lucas Manuelli, and Dieter Fox. Cliport: What and where pathways for robotic manipulation. In _Conference on Robot Learning (CoRL)_ , 2021. 2 

- [28] Arth Shukla, Stone Tao, and Hao Su. Maniskill-hab: A benchmark for low-level manipulation in home rearrangement tasks. In _International Conference on Learning Representations (ICLR)_ , 2025. 2, 3, 7 

- [29] Andrew Szot, Alex Clegg, Eric Undersander, Erik Wijmans, Yili Zhao, John Turner, Noah Maestre, Mustafa Mukadam, Devendra Chaplot, Oleksandr Maksymets, et al. Habitat 2.0: Training home assistants to rearrange their habitat. In _Advances in Neural Information Processing Systems (NeurIPS)_ , 2021. 1, 2 

- [30] Naoki Wake, Atsushi Kanehira, Kazuhiro Sasabuchi, Jun Takamatsu, and Katsushi Ikeuchi. Gpt-4v(ision) for robotics: Multimodal task planning from human demonstration. _IEEE Robotics and Automation Letters (RA-L)_ , 9(11):10567– 10574, 2024. 2 

- [31] Yating Wang, Haoyi Zhu, Mingyu Liu, Jiange Yang, HaoShu Fang, and Tong He. Vq-vla: Improving visionlanguage-action models via scaling vector-quantized action tokenizers. In _International Conference on Computer Vision (ICCV)_ , 2025. 2 

- [32] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2.5 technical report. _arXiv preprint arXiv:2412.15115_ , 2024. 1, 4, 8 

- [33] Denis Yarats, Rob Fergus, Alessandro Lazaric, and Lerrel Pinto. Mastering visual continuous control: Improved data-augmented reinforcement learning. _arXiv preprint arXiv:2107.09645_ , 2021. 1 

- [34] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In _International Conference on Computer Vision (ICCV)_ , 2023. 3 

- [35] Jiazhao Zhang, Kunyu Wang, Rongtao Xu, Gengze Zhou, Yicong Hong, Xiaomeng Fang, Qi Wu, Zhizheng Zhang, and He Wang. Navid: Video-based vlm plans the next step for vision-and-language navigation. _arXiv preprint arXiv:2402.15852_ , 2024. 2 

- [36] Wenyao Zhang, Hongsi Liu, Zekun Qi, Yunan Wang, Xinqiang Yu, Jiazhao Zhang, Runpei Dong, Jiawei He, He Wang, Zhizheng Zhang, et al. Dreamvla: A vision-languageaction model dreamed with comprehensive world knowledge. _arXiv preprint arXiv:2507.04447_ , 2025. 2 

11 

