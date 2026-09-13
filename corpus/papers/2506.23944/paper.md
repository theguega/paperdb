# **Adapt Your Body: Mitigating Proprioception Shifts in Imitation Learning** 

**Fuhang Kuang**<sup>1</sup> **Jiacheng You**<sup>1</sup> **Yingdong Hu**<sup>1</sup><sup>_,_2</sup><sup>_,_3</sup> **Tong Zhang**<sup>1</sup><sup>_,_2</sup><sup>_,_3</sup> **Chuan Wen**<sup>1</sup><sup>_,_2</sup><sup>_,_3</sup> **Yang Gao**<sup>1</sup><sup>_,_2</sup><sup>_,_3</sup> 

Project page: https://proprioception-shift.github.io 

**Abstract:** Imitation learning models for robotic tasks typically rely on multimodal inputs, such as RGB images, language, and proprioceptive states. While proprioception is intuitively important for decision-making and obstacle avoidance, simply incorporating all proprioceptive states leads to a surprising degradation in imitation learning performance. In this work, we identify the underlying issue as the proprioception shift problem, where the distributions of proprioceptive states diverge significantly between training and deployment. To address this challenge, we propose a domain adaptation framework that bridges the gap by utilizing rollout data collected during deployment. Using Wasserstein distance, we quantify the discrepancy between expert and rollout proprioceptive states and minimize this gap by adding noise to both sets of states, proportional to the Wasserstein distance. This strategy enhances robustness against proprioception shifts by aligning the training and deployment distributions. Experiments on robotic manipulation tasks demonstrate the efficacy of our method, enabling the imitation policy to leverage proprioception while mitigating its adverse effects. Our approach outperforms the naive solution which discards proprioception, and other baselines designed to address distributional shifts. 

**Keywords:** Imitation Learning, Proprioception States, Distributional Shift 

## **1 Introduction** 

Imitation Learning (IL) proves to be a simple yet effective paradigm for learning policy from expert demonstrations, and has gained huge success and wide use in robot learning [1, 2, 3, 4]. Current imitation-based robotic models typically rely on multi-modal inputs, such as RGB images [3, 4], language [5, 6, 7, 8, 9], and proprioceptive states [9, 10, 11, 3]. Proprioception, conveying the robot’s internal configuration (e.g., joint angles, velocities), is intuitively crucial for decision-making and obstacle avoidance. However, we observe a counter-intuitive phenomenon that simply incorporating all proprioceptive states can lead to a surprising degradation in imitation learning performance (see Figure 2), which was also mentioned by previous work [2, 9, 11, 10]. However, previous works often resort to heuristic strategies like selectively including or entirely discarding proprioceptive information based on empirical outcomes. Such approaches risk neglecting valuable state information and lack a principled foundation. This work seeks to diagnose the underlying reasons for this performance degradation and propose a systematic method to effectively integrate proprioception. 

Exploring why adding proprioception can degrade performance, we first examine how the encountered proprioceptive states differ between training and deployment. Figure 3 illustrates a key finding: the distribution of proprioception states visited by the learned policy during rollouts often diverges significantly from the distribution seen in the expert demonstrations. This divergence exemplifies the well-known distributional shift problem common in imitation learning [12, 13], where compounding errors naturally lead the agent away from the training data distribution. Furthermore, we observe that policies trained with proprioception sometimes achieve lower prediction errors on both training and validation datasets, suggesting the model finds a shortcut between proprioceptive states 

1Tsinghua University, 2Shanghai Qi Zhi Institute, 3Shanghai Artificial Intelligence Laboratory 



Figure 1: Strategies for incorporating proprioceptive observations ( _op_ ) in robotic imitation learning. (a) NADA (Ours): A two-pass approach where proprioception shift ( _WT_ ) between initial training/rollout data guides optimized noise ( _σ_<sup>_∗_</sup> ) injection into the training set for robust policy learning. Compared Baselines: (b) Full Observations: Using all sensory inputs directly. (c) Pure RGB: Discarding proprioception entirely. (d) Dropout Proprioception: Randomly zeroing out proprioceptive inputs during training. Our method provides a systematic framework that not only mitigates the proprioception shift but also preserves valuable information in proprioceptive observations. 



Figure 2: Performance change by including proprioception states as observations. In most tasks (6 out of 9), proprioception states lead to performance drop. 



Figure 3: Distributional shift<sup>2</sup> caused by robot proprioceptive observations in task Square[2]. BC-Full: with proprioceptive observations; BC-RGB: without proprioceptive observations; NADA: our method that reduces distributional shift with proprioceptive observations. 

and target actions. This pattern is indicative of shortcut learning, where the policy exploits strong but potentially non-causal correlations between proprioceptive states and expert actions within the training data, rather than learning the underlying task logic. Such reliance on spurious correlations is a form of causal confusion [14], which is a problematic effect of distributional shift. Based on the above findings, we term this issue the **proprioception shift** problem, where the divergence between training and deployment distributions of proprioceptive states severely hinders policy performance. 

> 2The x-axis lists different proprioception states (from left to right: end effector position, end effector rotation, end effector angular velocity, end effector linear velocity, gripper position, and gripper velocity); The 

2 

To tackle this issue, we frame it as a domain adaptation problem between the trainset and rollout proprioceptive observation distributions. Our goal is to reduce the discrepancy between these two distributions, thus improving the model’s robustness to proprioceptive observations. We propose a two passes training procedure to reduce the distributional shift (see Figure 1). In the first pass, we train an initial policy with original training data, and collect rollout data from the initial policy. In the second pass, we introduce Gaussian noise into both the trainset and rollout observations, optimizing the noise level according to the Wasserstein distance that quantifies the distributional shift between the two observation sets. The optimized noise level is then used for augmentation during the second training pass. This approach effectively expands the coverage of both distributions, increasing their overlap and decreasing the Wasserstein distance between them, mitigating the adverse effects of distributional shift on the performance of imitation learning models. 

Our main contributions are summarized as follows: 

- We formally identify and analyze the **proprioception shift** problem in multi-modal imitation learning. We provide empirical evidence showing that the distributional shift is a key factor contributing to the performance-drop when including proprioception states. 

- We propose a novel domain adaptation method specifically designed to mitigate the proprioception shift. Our approach leverages the Wasserstein distance not just to measure, but to actively guide a principled noise injection strategy during training. This systematically aligns the proprioceptive distributions between expert demonstrations and policy rollouts, enhancing robustness. 

## **2 Related Work** 

**Robot learning with proprioception states.** Proprioception states are a type of easily-obtained modality that can be fed as observations in imitation learning [9, 10, 11, 3]. However, previous works [2, 9, 11, 10] have noticed that proprioception states can lead to performance drop in imitation learning tasks. To explain the reason, [2, 11] assumed that policies tends to overfit to the proprioception, and [9] hypothesized that there exists causal confusion [14] between proprioception and target actions. In our findings, proprioception has both strong correlation with the target actions and high distributional shift, thus leading to performance drop. 

**Shortcut learning and causal confusion in imitation learning.** An improper strong correlation between observations and target actions can lead to shortcut learning, also known as causal confusion [14] in imitation learning. This is identified to be an effect of distributional shift problem [14]. Previous works [14, 15, 16, 17] mainly focused on the causal confusion problem caused by history observations and irrelevant visual information. [15] aimed to reduce the shortcut information leaked by observation history by adversarial training, and [16] proposed to leverage key inputs without shortcut information to prime the policy. However, few works have focused on the shortcut problem of proprioception states. In this work, we addresses the shortcut learning from proprioceptive states by injecting noise, thereby encouraging the policy to learn more robust and generalizable relationships between proprioception and actions, rather than relying on spurious correlations. 

## **3 Method** 

### **3.1 Problem Formulation** 

In imitation learning, the training data is in the format of observation-action pairs _D_ = _{_ ( _oi, ai_ )<sup>_N_</sup> _i_ =1<sup>_}_.</sup> We assume observations in _D_ follow distribution _O_ . And there is a policy _π_ trained on training data to optimize the following loss: 



y-axis is the Wasserstein distance (see Section 3.2.1 for distributional shift measurement) between training and rollout proprioceptive observations. 

3 

where _L_ denotes the loss function. During rolling out, observations are generated by action output of deployed policy _π_ and environment transition. Due to policy inaccuracies, errors accumulate over time, causing rollout observations to deviate from _O_ and follow a shifted distribution _O_<sup>_′_</sup> . This discrepancy, termed distributional shift, often leads the learned policy to unseen observations and hence degrades policy performance. In this work, our objective is to reduce the distributional shift associated with proprioceptive observations, motivated by the performance drop it induces. 

In this work, proprioceptive states (observations) refer to measurements from a robot’s internal sensors that describe its own physical state, including but not limited to: end effector pose/velocity, joint angles/velocities, gripper position/velocity. 

### **3.2 Noise-Augmented Distribution Alignment (NADA)** 

Our experiments reveal that proprioceptive states induce severe distributional shift. Existing mitigations usually adopt heuristic approaches, such as discarding velocity terms from observations. These hand-crafted strategies can partially reduce distributional shift but may discard too much critical information. Instead, we propose a systematic method to directly reduce the distributional distance between _O_ and _O_<sup>_′_</sup> while preserving observation fidelity. 

### **3.2.1 Time-conditioned Wasserstein Distance as a Distributional Metric** 

We quantify distributional shift using the Wasserstein distance, a metric from optimal transport theory. The Wasserstein distance is well-suited for this metric due to its ability to compare distributions with non-overlapping supports and its interpretability as the minimal “cost” of transforming one distribution into another. Formally, the Wasserstein distance between _O_ and _O_<sup>_′_</sup> is defined as: 



where Γ( _O, O_<sup>_′_</sup> ) denotes the set of joint distributions with marginals _O_ and _O_<sup>_′_</sup> . 

However, the distributional shift usually occurs and increases over time, so simply computing Wasserstein distance between _O_ and _O_<sup>_′_</sup> may not fully capture the distributional shift. To address this, we introduce a time-conditioned Wasserstein distance that considers the temporal evolution of observations: 



where _t ∈_ [0 _,_ 1] is the time step within a complete trajectory normalized to the range [0 _,_ 1], and _Ot_ and _Ot_<sup>_′_are the marginal distributions of corresponding observations at time</sup><sup>_t_.This formulation</sup> captures the cumulative effect of distributional shift over time, providing a more comprehensive measure of the distributional shift. In practice the integral is hard to compute, so we approximate it by segmenting the trajectory into _K_ equal intervals and computing the average Wasserstein distance within all intervals. 

### **3.2.2 Noise-Augmented Distribution Alignment (NADA)** 

Given the metric measuring the distance between distributions, the remaining task is to design a principled and effective method reducing the distance between _O_ and _O_<sup>_′_</sup> . 

Before we proceed, we need to take a closer look at the distribution itself. In practice, only the empirical distribution version of _O_ is accessible, which has a probability density function in the form of a sum of Dirac delta functions: 



This PDF can be approximated by a sum of normal distribution with a sufficiently small standard deviation _σ >_ 0, namely 



4 

We observe that a larger _σ_ for both _O_ and _O_<sup>_′_</sup> usually results in a smaller _WT_ ( _O, O_<sup>_′_</sup> ). This inspires us to propose a noise injection strategy that broadens the support of both distributions. Concretely, we replace any observation (both in training and inference) _o_ with its noisy version _o_ + _ϵ, ϵ ∼N_ (0 _, σ_<sup>2</sup> **I** ). 

This observation also naturally induces a principled way to tuning the intensity of the Gaussian noise. We first collect observation trajectories from training data as _O_ and rollout data generated by a trained policy as _O_<sup>_′_</sup> , and then choose the _σ_<sup>_∗_</sup> minimizing the time-conditioned Wasserstein distance: 



However, a large _σ_ can destroy useful information in observations. It’s undesirable to significantly increase _σ_ for a marginal decrement of _WT_ ( _Oσ, Oσ_<sup>_′_). We opt to repeatedly increase</sup><sup>_σ_at a fixed step-</sup> size _δ_ until _WT_ ( _Oσ, Oσ_<sup>_′_)converges,i.e.thedecrementof</sup><sup>_WT_(</sup><sup>_Oσ, O_</sup> _σ_<sup>_′_)issmallerthanapre-set</sup> threshold. 

### **3.2.3 Two Passes Training** 

As mentioned in the previous section, we need to collect rollout data generated by a trained policy to determine the noise level before the training can start. However, we do not have a trained policy in hand before the training completes. 

Theoretically, assuming convergence, this chicken-and-egg problem can be solved by an iterative training strategy starting from a noise level of _σ_ = 0 and using the trained policy in the current iteration to determine the noise level in the next iteration. 

However, we observe that the improvement beyond the first iteration is marginal. Thus, we opt to use a simpler two passes strategy, described in Algorithm 1 

**Algorithm 1** Two Passes Training 

**Require:** Training data _D_ = _{_ ( _oi, ai_ ) _}_<sup>_N_</sup> _i_ =1<sup>, corresponding training observations</sup><sup>_O_=</sup><sup>_{oi}N_</sup> _i_ =1<sup>and</sup> initial state distribution _p_ 0 

- 1: Initialize policy _π_ 0 by training on _D_ with full observations 2: Roll out _π_ 0, get rollout observations _O_<sup>_′_</sup> = _{o_<sup>_′_</sup> _j_<sup>_}M_</sup> _j_ =1<sup>and task success rate</sup><sup>_S_</sup> 

- 3: Determine noise level _σ_<sup>_∗_</sup> with Equation 6 

- 4: Modify _D_ by _O ← O_ + _N_ (0 _, σ_<sup>_∗_2</sup> **I** ) 

- 5: Train policy _π_ 1 on _D_ with full observations 

- 6: **return** the best of _π_ 0 and _π_ 1 

## **4 Experiment** 

### **4.1 Simulation Experiment Setup** 

**Benchmarks** . We use tasks from Robomimic [2] and MimicGen [18] tasks as our benchmark. Both projects offer a broad set of demonstration datasets together with Robosuite [19] simulation environments suitable for imitation learning and evaluation. We evaluate our method on 9 tasks from them: NutAssemblySquare (Square) and PickPlaceCan (Can) from Robomimic, and Coffee, HammerCleanup, MugCleanup, Stack, StackThree, Threading and ThreePieceAssembly from MimicGen. Each Robomimic task contains 200 demostration trajectories (PH dataset), and each MimicGen task contains 1000 demonstration trajectories ( **D0** dataset). The action space is the delta end effector pose and the gripper state. We train all the methods over 3 seeds and evaluate the last five checkpoints by rolling out 50 trajectories. We report the mean and standard deviation (over seeds) of success rate of each task. 

**Observation Space.** We include two 84 _×_ 84 _×_ 3 RGB images respectively from wrist camera and a third-view camera as basic observations. And for proprioception states, we include the following terms: 7-dim end-effector pose, 6-dim end-effector velocity, 1-dim gripper position, 1-dim gripper velocity, 7-dim joint positions and 7-dim joint velocities. 

5 

|**Task**|BC-Full|BC-RGB|PrimeNet|Dropout|Ours|
|---|---|---|---|---|---|
|Square|3.0_±_3.0|52.8_±_4.0|4.7_±_1.9|44.8_±_5.6|**56.7**_±_**4.3**|
|Can|51.3_±_6.8|90.3_±_3.4|67.6_±_7.1|90.8_±_1.6|**93.3**_±_**1.8**|
|Coffee|93.6_±_4.0|91.1_±_4.8|**95.2**_±_**3.0**|89.6_±_1.4|93.1_±_2.8|
|Hammer Cleanup|94.7_±_4.5|99.0_±_1.0|96.7_±_1.5|**99.1**_±_**0.8**|97.9_±_0.9|
|Mug Cleanup|43.3_±_7.6|31.4_±_3.9|**44.7**_±_**7.4**|38.5_±_3.2|43.3_±_7.6|
|Stack|81.5_±_8.4|87.4_±_2.9|84.6_±_8.5|87.7_±_1.5|**90.5**_±_**3.6**|
|Stack Three|38.4_±_12.2|53.7_±_2.6|51.0_±_5.6|52.7_±_2.7|**61.4**_±_**2.0**|
|Threading|73.5_±_7.8|84.5_±_1.9|81.5_±_3.5|**92.1**_±_**0.5**|88.3_±_3.3|
|Three Assembly|33.9_±_5.7|32.5_±_2.7|31.7_±_6.0|35.1_±_2.4|**38.4**_±_**4.9**|



Table 1: Mean success rates (% _±_ std) on 9 tasks. The standard deviation is over 3 seeds. 

**Implementation Details.** We use simple BC [20] algorithm to examine our method. The observation encoder consists of a ResNet-18 visual encoder for RGB images and a 2-layer MLP for proprioception states. The features are concatenated and fed into a MLP action head for action prediction. We train the model with a batch size of 64 for 75k steps using the Adam optimizer with a learning rate of 2 _×_ 10<sup>_−_4</sup> and a cosine annealing learning rate scheduler. For our method, we independently add Gaussian noise to each type of proprioception states. We sampled 150 trajectories each from the training data and rollout data to calculate the Wasserstein distance. The number of intervals for computing time-conditioned Wasserstein distance (mentioned in Section 3.2.1) is set to 10. In NADA, the incremental step-size is set to _δ_ = 0 _._ 05, and the threshold for convergence condition is set to 0 _._ 001. In two passes training (Algorithm 1), in most tasks we return the second pass policy _π_ 1, except in the task MugCleanup we return the first pass policy _π_ 0 since we found that any noise injection in proprioception states would result in severe performance drop. When evaluating, we do not add noise to proprioception states to avoid outliner values in input. 

**Baselines.** We compare against four representative approaches: **BC-Full** : Standard Behavior Cloning with full proprioception and RGB images as observations. **BC-RGB** : BC using only RGB images (ablation for proprioception impact). **PrimeNet** [16]: Architecture with shortcut prevention via action prediction heads. **Random Dropout** [21]: Randomly mask proprioception input to zeros with probability 80% during training to prevent shortcut learning. Proved to be effective in reducing shortcut in imitation learning. 

### **4.2 Experiment Results** 

Table 1 presents the mean success rates (over 3 seeds) for our proposed method compared against 4 baselines across 9 distinct robotic manipulation tasks. Our proposed method achieves the highest average success rate in a clear majority of the tasks (6 out of 9). BC-Full can yield bad performance especially on tasks with fewer demonstrations (Square and Can), in which tasks we also observe severe proprioception shifts. In comparision, our method provides substantial improvements. This supports our hypothesis that naive inclusion of proprioception can be detrimental due to distributional shift, and that our method effectively addresses this issue. BC-RGB, which discards proprioception entirely, can also yield relatively bad performance (especially on task MugCleanup and StackThree), showing the value of proprioception our method unlocks. PrimeNet and Dropout aim to mitigate shortcut learning, but the results of them generally fall between BC-Full and BC-RGB. They also yields bad performance on tasks with strong proprioception shift, such as Sqaure. 

In summary, the experimental results validate our approach. Our method consistently enables the effective use of proprioception, and tackle proprioception shift and shortcut learning across a diverse set of manipulation tasks. 

6 



Figure 4: Real-world experiment results. Our method outperforms BC-Full and BC-RGB in most tasks. 

### **4.3 Real World Experiment** 

We conduct real-world experiments to further examine our method. We learn policies for a 7- DOF Franka Emika Panda robot arm to perform 2 tasks: Pick-and-place and Open-locker. The dataset contains 100 demonstration trajectories for each task, collected with GELLO teleoperation system [22]. The action space is absolute end effector pose and gripper state. The observation space is the same as in simulation setup, with RGB images collected by two RealSense cameras and proprioception states read from Franka robot arm sensors. To compensate for observation limit, we stack the most recent two frames as RGB input. The model predicts action chunk of size 16 and perform temporal ensemble in inference [3]. We compare the performance of BC-Full, BCRGB and our method. The environment setup and evaluation results are shown in Figure 4. Our method outperforms BC-Full and BC-RGB in both the two tasks, demonstrating its effectiveness in real-world scenarios. 

### **4.4 Analysis** 

We structure our analysis around three key questions: 

### **Q1: Does our method mitigate harmful shorcut learning while preserving useful information?** 

We compare the validation loss of BC-Full, BC-RGB and our method (see Table 2). Compared to BC-RGB, BC-Full has lower validation loss, indicating the shortcut between proprioception states and target actions. In our method, the validation loss is higher than BC-Full, but still lower than BCRGB. Meanwhile, our method achieves better or comparable performance to BC-Full and BC-RGB in all tasks. This indicates that our method can on the one hand mitigate the harmful shortcut and on the other hand preserving useful information in proprioception states. 

### **Q2: Does our method truly mitigate proprioception distributional shift?** 

We use time-conditioned Wasserstein distance between rollout and training proprioceptive observations to measure the distributional shift, shown in Table 3. Our method reduces the distributional 

7 

|**Task/ Valid Loss (**_×_10<sup>_−_2</sup>**)**|BC-Full|BC-RGB|Ours|
|---|---|---|---|
|Square|2.94|4.13|3.01|
|Can|2.55|4.85|3.49|
|Coffee|0.60|0.80|0.60|
|Hammer Cleanup|0.41|0.62|0.47|
|Mug Cleanup|0.64|1.30|0.92|
|Stack|1.44|4.58|2.13|
|Stack Three|1.45|4.58|2.37|
|Threading|0.47|0.67|0.52|
|Three Assembly|0.88|1.44|1.17|



Table 2: Average validation loss of the last checkpoint over 3 seeds. 

|**Task**|BC-Full|Ours|**Task**|Oracle|NADA(Ours)|
|---|---|---|---|---|---|
|Square|1.05|0.23|Square|57.6_±_1.0|56.7_±_4.3|
|Can|0.13|0.12|Can|94.3_±_1.0|93.3_±_1.8|
|Coffee|0.13|0.15|Coffee|94.7_±_2.6|93.1_±_2.8|
|Hammer Cleanup|0.11|0.11|Hammer Cleanup|99.5_±_0.6|97.9_±_0.9|
|Mug Cleanup|0.28|0.28|Mug Cleanup|43.3_±_7.6|43.3_±_7.6|
|Stack|0.24|0.18|Stack|93.7_±_1.2|90.5_±_3.6|
|Stack Three|0.31|0.27|Stack Three|67.3_±_4.8|61.4_±_2.0|
|Threading|0.14|0.12|Threading|89.5_±_3.4|88.3_±_3.3|
|Three Assembly|0.23|0.26|Three Assembly|40.7_±_8.4|38.4_±_4.9|



Table 3: Wasserstein distance between Table 4: Task success rate between oracle noise and policy rollout and dataset (normalized) our method. The standard deviation is over 3 seeds. proprioception observations. The result is averaged over proprioception terms. 

shift on proprioception observations in most tasks, compared to BC-Full that directly incorporates all proprioceptive observations. 

### **Q3: Does our method provide nearly optimal noise?** 

We use grid search to select the “oracle” noise level that maximizes the task success rate. We search from _σ_ = 0 (BC-Full) to _σ_ = 2 _._ 0 with incremental step 0 _._ 2 to select the noise level with the highest task success rate. We compare the oracle results with our method in Table 4. In most tasks, our method achieves comparable performance to the oracle, indicating that our method can effectively adapt the noise level to the task. 

## **5 Conclusion** 

We have investigated the performance drop problem caused by proprioception states in imitation learning, identifying distributional shift as the root cause. By framing this issue as a domain adaptation challenge, we proposed a systematic method that first leverages Wasserstein distance to measure the discrepancy between training and rollout proprioceptive observations and then apply noise injection to both distributions to minimize the distance. Our approach effectively reduces the distributional shift while preserving critical information. Our experiments demonstrated that our method improves performance in imitation learning tasks with proprioception states, showcasing its effectiveness in addressing the challenges posed by proprioception shift. Future work may explore further enhancements to this framework and its application to more modalities. 

8 

## **6 Limitations** 

Although our method mitigates the distributional shift problem caused by proprioception states, it still has some limitations. First, our method requires two passes training, which increases the training cost compared to standard behavioral cloning. Second, our proposed metric (time-conditioned Wasserstein distance) for distributional shift does not always correlate perfectly with task performance (see Tables 1 and 3). That is, better performance does not always correspond to a lower Wasserstein distance, possibly because factors other than distributional shift also affect performance. Future work could explore alternative metrics that better reflect the relationship between distributional shift and task success rate. 

### **Acknowledgments** 

This work is supported by the National Key R&D Program of China (2022ZD0161700), National Natural Science Foundation of China (62176135), Shanghai Qi Zhi Institute Innovation Program SQZ202306 and the Tsinghua University Dushi Program. 

9 

## **References** 

- [1] M. Zare, P. M. Kebria, A. Khosravi, and S. Nahavandi. A survey of imitation learning: Algorithms, recent developments, and challenges, 2023. URL `https://arxiv.org/abs/2309. 02473` . 

- [2] A. Mandlekar, D. Xu, J. Wong, S. Nasiriany, C. Wang, R. Kulkarni, L. Fei-Fei, S. Savarese, Y. Zhu, and R. Mart´ın-Mart´ın. What matters in learning from offline human demonstrations for robot manipulation. In _Conference on Robot Learning (CoRL)_ , 2021. 

- [3] T. Z. Zhao, V. Kumar, S. Levine, and C. Finn. Learning fine-grained bimanual manipulation with low-cost hardware, 2023. URL `https://arxiv.org/abs/2304.13705` . 

- [4] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song. Diffusion policy: Visuomotor policy learning via action diffusion, 2024. URL `https://arxiv.org/ abs/2303.04137` . 

- [5] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog, J. Hsu, J. Ibarz, B. Ichter, A. Irpan, T. Jackson, S. Jesmonth, N. J. Joshi, R. Julian, D. Kalashnikov, Y. Kuang, I. Leal, K.-H. Lee, S. Levine, Y. Lu, U. Malla, D. Manjunath, I. Mordatch, O. Nachum, C. Parada, J. Peralta, E. Perez, K. Pertsch, J. Quiambao, K. Rao, M. Ryoo, G. Salazar, P. Sanketi, K. Sayed, J. Singh, S. Sontakke, A. Stone, C. Tan, H. Tran, V. Vanhoucke, S. Vega, Q. Vuong, F. Xia, T. Xiao, P. Xu, S. Xu, T. Yu, and B. Zitkovich. Rt-1: Robotics transformer for real-world control at scale, 2023. URL `https://arxiv.org/abs/2212.06817` . 

- [6] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, X. Chen, K. Choromanski, T. Ding, D. Driess, A. Dubey, C. Finn, P. Florence, C. Fu, M. G. Arenas, K. Gopalakrishnan, K. Han, K. Hausman, A. Herzog, J. Hsu, B. Ichter, A. Irpan, N. Joshi, R. Julian, D. Kalashnikov, Y. Kuang, I. Leal, L. Lee, T.-W. E. Lee, S. Levine, Y. Lu, H. Michalewski, I. Mordatch, K. Pertsch, K. Rao, K. Reymann, M. Ryoo, G. Salazar, P. Sanketi, P. Sermanet, J. Singh, A. Singh, R. Soricut, H. Tran, V. Vanhoucke, Q. Vuong, A. Wahid, S. Welker, P. Wohlhart, J. Wu, F. Xia, T. Xiao, P. Xu, S. Xu, T. Yu, and B. Zitkovich. Rt-2: Vision-language-action models transfer web knowledge to robotic control, 2023. URL `https://arxiv.org/abs/2307.15818` . 

- [7] K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, S. Jakubczak, T. Jones, L. Ke, S. Levine, A. Li-Bell, M. Mothukuri, S. Nair, K. Pertsch, L. X. Shi, J. Tanner, Q. Vuong, A. Walling, H. Wang, and U. Zhilinsky. _π_ 0: A vision-language-action flow model for general robot control, 2024. URL `https://arxiv. org/abs/2410.24164` . 

- [8] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, Q. Vuong, T. Kollar, B. Burchfiel, R. Tedrake, D. Sadigh, S. Levine, P. Liang, and C. Finn. Openvla: An open-source vision-language-action model, 2024. URL `https://arxiv.org/abs/2406.09246` . 

- [9] O. M. Team, D. Ghosh, H. Walke, K. Pertsch, K. Black, O. Mees, S. Dasari, J. Hejna, T. Kreiman, C. Xu, J. Luo, Y. L. Tan, L. Y. Chen, P. Sanketi, Q. Vuong, T. Xiao, D. Sadigh, C. Finn, and S. Levine. Octo: An open-source generalist robot policy, 2024. URL `https: //arxiv.org/abs/2405.12213` . 

- [10] S. Haldar, Z. Peng, and L. Pinto. Baku: An efficient transformer for multi-task policy learning, 2024. URL `https://arxiv.org/abs/2406.07539` . 

- [11] X. Lin, J. So, S. Mahalingam, F. Liu, and P. Abbeel. Spawnnet: Learning generalizable visuomotor skills from pre-trained networks, 2023. URL `https://arxiv.org/abs/2307.03567` . 

10 

- [12] S. Ross and D. Bagnell. Efficient reductions for imitation learning. In Y. W. Teh and M. Titterington, editors, _Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics_ , volume 9 of _Proceedings of Machine Learning Research_ , pages 661–668, Chia Laguna Resort, Sardinia, Italy, 13–15 May 2010. PMLR. URL `https: //proceedings.mlr.press/v9/ross10a.html` . 

- [13] H. D. III, J. Langford, and D. Marcu. Search-based structured prediction, 2009. URL `https: //arxiv.org/abs/0907.0786` . 

- [14] P. de Haan, D. Jayaraman, and S. Levine. Causal confusion in imitation learning, 2019. URL `https://arxiv.org/abs/1905.11979` . 

- [15] C. Wen, J. Lin, T. Darrell, D. Jayaraman, and Y. Gao. Fighting copycat agents in behavioral cloning from observation histories, 2020. URL `https://arxiv.org/abs/2010.14876` . 

- [16] C. Wen, J. Qian, J. Lin, J. Teng, D. Jayaraman, and Y. Gao. Fighting fire with fire: Avoiding dnn shortcuts through priming. _ICML_ , 2022. 

- [17] S. Seo, H. Hwang, H. Yang, and K.-E. Kim. Regularized behavior cloning for blocking the leakage of past action information. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, _Advances in Neural Information Processing Systems_ , volume 36, pages 2128–2153. Curran Associates, Inc., 2023. URL `https://proceedings.neurips.cc/paper_files/paper/2023/file/ 06b71ad997f7e3e4b2e2f2ea12e5a759-Paper-Conference.pdf` . 

- [18] A. Mandlekar, S. Nasiriany, B. Wen, I. Akinola, Y. Narang, L. Fan, Y. Zhu, and D. Fox. Mimicgen: A data generation system for scalable robot learning using human demonstrations, 2023. URL `https://arxiv.org/abs/2310.17596` . 

- [19] Y. Zhu, J. Wong, A. Mandlekar, R. Mart´ın-Mart´ın, A. Joshi, K. Lin, S. Nasiriany, and Y. Zhu. robosuite: A modular simulation framework and benchmark for robot learning. In _arXiv preprint arXiv:2009.12293_ , 2020. 

- [20] D. A. Pomerleau. Alvinn: An autonomous land vehicle in a neural network. In D. Touretzky, editor, _Advances in Neural Information Processing Systems_ , volume 1. Morgan-Kaufmann, 1988. URL `https://proceedings.neurips.cc/paper_files/paper/1988/file/ 812b4ba287f5ee0bc9d43bbf5bbe87fb-Paper.pdf` . 

- [21] M. Bansal, A. Krizhevsky, and A. Ogale. Chauffeurnet: Learning to drive by imitating the best and synthesizing the worst, 2018. URL `https://arxiv.org/abs/1812.03079` . 

- [22] P. Wu, Y. Shentu, Z. Yi, X. Lin, and P. Abbeel. Gello: A general, low-cost, and intuitive teleoperation framework for robot manipulators, 2024. URL `https://arxiv.org/abs/2309. 13037` . 

11 

## **Appendix** 

|**A **|**Real World Tasks and Evaluations**|**12**|
|---|---|---|
|**B**|**Baseline Implementation**|**12**|
|**C **|**Effects of Each Proprioception State**|**12**|
|**D **|**Grid Searching Noise Level**|**13**|



## **A Real World Tasks and Evaluations** 

Figure 4 shows the tasks setup for real world robot experiments. We collect 100 expert demonstrations for each task, and deploy the policy for 50 rollout trajectories. Both the training and rollout data are used to compute the optimized noise level for NADA. The detailed description of each task is as follows: 

`Pick-and-place` . The robot is required to pick up a plastic banana and place it in a plate. Both the banana and the plate are randomly placed in the workspace, during both demonstrations collection and evaluation. 

`Open-locker` . The robot is required to use its gripper to open the door of an unlocked locker to at least 70 degrees. The locker is randomly placed in the workspace, and the door’s initial opening angle is randomly set between 0 and 15 degrees, during both demonstrations collection and evaluation. 

## **B Baseline Implementation** 

**PrimeNet** [16]. An outline of the PrimeNet architecture is shown in Figure 5. We set the raw inputs to be the RGB images and proprioception states, and the key inputs to be only the RGB images. The priming variable _ζ_ is a coarse action prediction, to prime the main module to output a refine action prediction _y_ ˆ. The priming module and the main module share similar architectures, with an encoder followed by a two-layer MLP action head. The priming variable is concatenated to the encoder output of the main module, before the action head. During backpropagation, the gradient from the main module to the priming variable is stopped. During evaluation, the refine predicted action is used for the control. 

**Random Dropout.** We implement random dropout by randomly masking the proprioception states to zeros with probability 0 _._ 8, before feeding them into the policy. The dropout is independent for each type of proprioception state. During evaluation, we use the full proprioception states. 

## **C Effects of Each Proprioception State** 

We also thoroughly evaluate the effects of each proprioception state on the performance of the policy. The results are shown in Table 5. In most tasks where including all proprioception states harms the performance, it is the inclusion of velocities that mainly degrades the performance. We also observe that the inclusion of velocities leads to lower validation loss compared to those policies without velocities observation. We hypothesize that velocities have stronger correlation with target actions since we use delta end-effector pose as action space; and velocities are more sensitive to control errors, which makes the policy accumulate more errors during rolling out, resulting in more severe distributional shift. 

In the meanwhile, half of the 6 tasks achieve the best success rate when including proprioception states, especially the task “MugCleanup”, which implies the importance of proprioception states in 

12 



Figure 5: The architecture of PrimeNet. 

|**Proprio./Tasks**|Square|Can|Mug|StackThree|Threading|Assembly|
|---|---|---|---|---|---|---|
|none|**52**_._**8**_±_**4**_._**0**|**90**_._**3**_±_**3**_._**4**|31_._4_±_3_._9|**53**_._**7**_±_**2**_._**6**|84_._5_±_1_._9|32_._5_±_2_._7|
|ee pose|37_._3_±_2_._9|87_._2_±_3_._8|35_._7_±_5_._2|52_._1_±_6_._1|83_._7_±_4_._2|29_._7_±_3_._8|
|ee vel|5_._6_±_1_._2|66_._6_±_6_._7|18_._5_±_2_._6|27_._0_±_3_._1|**87**_._**1**_±_**3**_._**6**|22_._6_±_5_._5|
|ee pose+vel|3_._8_±_1_._6|66_._8_±_5_._0|33_._9_±_5_._0|41_._6_±_8_._2|82_._7_±_4_._1|33_._4_±_8_._5|
|joint pos|40_._6_±_3_._9|83_._4_±_2_._2|38_._1_±_4_._4|52_._0_±_4_._6|84_._6_±_4_._5|35_._2_±_5_._7|
|joint vel|5_._8_±_3_._6|66_._7_±_6_._0|29_._9_±_9_._4|16_._2_±_2_._7|82_._5_±_2_._1|27_._8_±_3_._5|
|joint pos+vel|5_._0_±_1_._3|65_._1_±_5_._0|38_._7_±_3_._9|17_._2_±_2_._5|76_._4_±_5_._8|**37**_._**1**_±_**7**_._**2**|
|all|3_._0_±_3_._0|51_._3_±_6_._8|**43**_._**3**_±_**7**_._**6**|38_._4_±_12_._2|73_._5_±_7_._8|33_._9_±_5_._7|



Table 5: Ablation study of each proprioception state on the performance of the policy over 6 tasks. The results are averaged over 3 seeds. “none” means no proprioception states are used, which is exactly BC-RGB; “ee” means end-effector, which also includes gripper; “all” means all proprioception states are used, which is exactly BC-Full. 

this task. Those information reminds us that proprioception states can both be beneficial and harmful to the performance of the policy, depending on the task and the proprioception type. This calls for an adaptive method to leverage the proprioception states, which is exactly the motivation of our work and partially solved by NADA. 

## **D Grid Searching Noise Level** 

We list the results of grid searching the noise level in Table 6. We can see that different tasks have different patterns in success rate with respect to the noise level. Some tasks have a clear peak at certain noise level, such as Square( _σ_ = 0 _._ 6), “Can”( _σ_ = 0 _._ 2), “StackThree”( _σ_ = 1 _._ 2), and “Stack”( _σ_ = 1 _._ 0), which implies in these tasks there exists a trade-off between the distributional shift and useful information in proprioception states. Some tasks such as “Coffee”, “ThreePieceAssembly” and “HammerCleanup” show noisy results when noisy level increases, which implies that the policies are not sensitive to the proprioception states. This is also consistent with the results in Table 5, where the policies trained with proprioception states have similar performance as those trained without proprioception states. Lastly the task “MugCleanup” shows a clear decreasing trend in success rate when adding any level of noise, which implies that the proprioception states are important for this task. 

Therefore, since the optimal noise level is task-dependent, grid search is very unrealistic given its high computational cost. Meanwhile our method can automatically adapt to the noise level of each task, achieving comparable performance. Hence our method is general and efficient. 

13 

|**Std./Tasks**|Square|Can|Coffee|Hammer|Mug|
|---|---|---|---|---|---|
|0_._0|3_._0_±_3_._0|51_._3_±_6_._8|93_._6_±_4_._0|94_._7_±_4_._5|43_._3_±_7_._6|
|0_._2|21_._3_±_4_._4|94_._3_±_1_._0|94_._7_±_2_._6|96_._0_±_1_._1|27_._2_±_3_._8|
|0_._4|50_._5_±_3_._5|92_._9_±_2_._2|90_._9_±_4_._8|98_._4_±_1_._7|30_._8_±_9_._4|
|0_._6|57_._6_±_1_._0|92_._1_±_2_._6|91_._6_±_2_._4|99_._3_±_0_._6|24_._3_±_5_._7|
|0_._8|57_._1_±_7_._7|90_._1_±_5_._3|89_._7_±_4_._6|98_._5_±_0_._8|23_._9_±_1_._5|
|1_._0|54_._4_±_5_._4|89_._5_±_1_._3|93_._1_±_3_._1|99_._2_±_1_._4|25_._1_±_0_._6|
|1_._2|52_._4_±_5_._4|89_._9_±_1_._6|90_._5_±_2_._1|98_._3_±_1_._0|27_._2_±_6_._9|
|1_._4|51_._6_±_3_._9|88_._9_±_2_._2|87_._2_±_3_._1|99_._5_±_0_._6|24_._7_±_2_._0|
|1_._6|54_._5_±_7_._6|89_._3_±_3_._6|82_._7_±_7_._0|98_._9_±_0_._8|25_._3_±_4_._2|
|1_._8|54_._8_±_5_._2|90_._0_±_2_._6|89_._9_±_4_._6|98_._7_±_0_._2|26_._9_±_2_._6|
|2_._0|53_._1_±_4_._3|91_._3_±_1_._8|88_._1_±_4_._8|98_._3_±_0_._9|30_._4_±_4_._0|
|none|52_._8_±_4_._0|90_._3_±_3_._4|91_._1_±_4_._8|99_._0_±_1_._0|31_._4_±_3_._9|
|**Std./Tasks**|Stack|StackThree|Threading|Assembly||
|0_._0|81_._5_±_8_._4|38_._4_±_12_._2|73_._5_±_7_._8|33_._9_±_5_._7||
|0_._2|79_._3_±_3_._4|43_._1_±_1_._2|69_._3_±_6_._5|33_._6_±_5_._2||
|0_._4|86_._7_±_4_._6|47_._9_±_12_._6|83_._9_±_6_._5|40_._7_±_8_._4||
|0_._6|88_._4_±_3_._9|49_._5_±_7_._8|86_._5_±_4_._3|31_._7_±_8_._3||
|0_._8|89_._1_±_2_._7|59_._9_±_1_._2|87_._9_±_7_._2|39_._3_±_9_._1||
|1_._0|93_._7_±_1_._2|66_._7_±_2_._6|89_._5_±_3_._4|38_._5_±_6_._2||
|1_._2|90_._7_±_2_._0|67_._3_±_4_._8|84_._8_±_4_._0|34_._8_±_1_._7||
|1_._4|90_._9_±_1_._2|63_._2_±_2_._1|85_._2_±_1_._7|38_._0_±_1_._8||
|1_._6|92_._4_±_1_._1|58_._3_±_3_._4|84_._0_±_3_._7|32_._4_±_3_._2||
|1_._8|89_._2_±_2_._5|62_._0_±_3_._6|86_._9_±_8_._0|36_._1_±_3_._2||
|2_._0|91_._3_±_1_._5|61_._5_±_0_._2|84_._9_±_0_._5|37_._7_±_5_._2||
|none|87_._4_±_2_._9|53_._7_±_2_._6|84_._5_±_1_._9|32_._5_±_2_._7||



Table 6: Results of the grid searching noises. The results are averaged over 3 seeds. 

14 

