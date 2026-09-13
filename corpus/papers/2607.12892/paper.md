# UR-VC: Unsupervised Robotic Value Correction for Time-Derived Progress Proxies 

Lirui Zhao, Modi Shi, Li Chen, Qi Liu, Ping Luo, Hongyang Li 

The University of Hong Kong 

https://liruizhao.com/projects/UR-VC 

**_Abstract_ —Modern robot learning systems increasingly rely on dense progress or value signals to evaluate intermediate states, guide policy learning, and detect task completion, making the quality of these signals critical. Since such dense labels are rarely available at scale, normalized time within a demonstration is often used as a scalable substitute: later frames are treated as higher progress. However, this time-derived label is only a noisy proxy for physical task progress. In contact-rich manipulation, a robot may make progress and then lose it through slips, failed grasps, or partial undoing, while the time-derived label continues to increase monotonically. We introduce Unsupervised Robotic Value Correction (UR-VC), an offline, training-free method for correcting time-derived progress labels. UR-VC exploits a simple regularity in demonstration data: similar states often recur across different episodes, but at different timestamps. Instead of trusting the timestamp from a single trajectory, UR-VC retrieves similar states from other episodes and aggregates their time-derived labels to obtain a corrected progress estimate. UR-VC requires no manual progress labels, reward annotations, or additional value model. We evaluate UR-VC on real bimanual cloth flatten-andfold data, a long-horizon deformable-object manipulation task with visible intermediate progress. The corrected labels capture local regressions and non-uniform progress that normalized time cannot represent, while preserving the overall task trend. We further use the corrected signal to construct advantage labels for VLA training, following recent advantage-conditioned policy learning. UR-VC shows a positive trend in real-robot task success under matched data, model, and training settings.** 

## I. INTRODUCTION 

Modern robot learning systems increasingly use intermediate task signals in addition to action supervision. In longhorizon manipulation, such signals estimate how far the current state has advanced toward task completion and whether recent actions have moved the task forward. They commonly take the form of task progress, value, or advantage, and are used for completion detection, value-function learning, and advantage-conditioned policy learning [1, 25, 20, 22, 6, 9]. As vision–language–action (VLA) models are trained on larger and more diverse robot datasets [5, 15, 4, 3], the quality of these dense signals becomes increasingly critical. 

The difficulty is that dense progress or value labels are rarely available at scale. A common scalable substitute is normalized time within a successful demonstration [1, 25], which treats later frames as higher progress since they usually correspond to states closer to task completion. However, the resulting time-derived label is only a noisy proxy for physical task progress. It assumes that progress increases monotonically 

and uniformly with time, while real manipulation, especially contact-rich and deformable manipulation [16, 8], can regress, stall, or advance at different rates. For example, in the real cloth flatten-and-fold episode (Fig. 1), the robot first smooths the garment, but a later grasp slip reintroduces wrinkles and moves the task state backward, so physical progress regresses even as the time-derived label continues to increase linearly. 

This mismatch matters for downstream learning in two ways. Within an episode, this directly affects advantage-style supervision: under a fixed-horizon progress difference, a linear time proxy assigns the same increase to every frame, so it cannot distinguish actions that improve the physical state from actions that stall or undo progress. Across episodes, the problem is subtle but more persistent: similar physical states can receive different labels simply because demonstrations proceed at different speeds or take different recovery motions. A model trained on these noisy labels may then explain label variation using incidental cues such as appearance, texture, or execution speed, rather than physical task progress. 

A common response is to learn this signal from data, for example by training progress estimators or value models using temporal contrast, language–image objectives, value pretraining, optimal transport, or generative modeling [23, 18, 19, 2, 11, 13], or by eliciting value estimates from pretrained models [20, 7, 17]. However, when the supervision target is a systematically biased time-derived proxy, a learned estimator can inherit the same bias: it may smooth over local regressions and encode differences in execution speed or recovery behavior as differences in progress. This motivates a complementary question: _can the proxy labels themselves be corrected before they serve as supervision for estimators or policies?_ 

We introduce **Unsupervised Robotic Value Correction (UR-VC)** , an offline, training-free method for correcting timederived progress labels in demonstration datasets. UR-VC is based on a simple data-driven premise: time-derived labels are noisy mainly because each demonstration has its own temporal distortion. Cross-episode state matches provide anchors for reducing this distortion: when similar physical states are found in independently collected episodes, the state is approximately held fixed while the timestamp varies across trajectories. Aggregating the time-derived labels of these matched states therefore estimates where that state typically lies in the task, rather than where it happened to occur in one particular episode. UR-VC operationalizes this idea by retrieving se- 













<!-- Start of picture text -->
t=0.06 t=0.25 t=0.28 t=0.80 t=0.97<br>1.0<br>time label (proxy)<br>0.8 UR-VC corrected<br>0.6<br>0.4<br>0.2<br>grasp slips   progress regresses<br>(the time label keeps rising)<br>0.0<br>0.0 0.2 0.4 0.6 0.8 1.0<br>episode time (normalized)<br>progress<br><!-- End of picture text -->

Fig. 1: **Time is not progress.** In a real cloth flatten-and-fold episode, the garment is first smoothed, but a slipping grasp reintroduces wrinkles and causes physical progress to regress. The time-derived label (gray) remains monotone by construction, whereas UR-VC estimates corrected progress by retrieving semantically similar frames from other episodes and aggregating their time-derived labels, producing a correction (red) that decreases when the visual state worsens. Across the primary evaluation set, 13.4% of frames receive a negative horizon advantage under UR-VC, capturing regressions the time proxy misses. 

mantically similar frames from other episodes using SigLIP2 visual embeddings [26] and aggregating their time-derived labels into a corrected progress estimate, with at most one matched label contributed by each episode to avoid overweighting temporally adjacent frames. 

We study UR-VC on real bimanual cloth flatten-and-fold data. Cloth manipulation is a natural testbed for this problem because progress is visually meaningful, local regressions occur frequently, and similar intermediate states recur across garments and episodes [28, 8]. Since true physical progress is not directly observed in demonstrations, we first evaluate the conditions that make correction useful: cross-episode coverage, task-state consistency, stability under aggregation, and recovery of non-monotone progress. We find that these conditions hold in our data: high-quality matches are abundant even at moderate dataset sizes (Fig. 2), retrieved frames align with folding state across different garments (Fig. 3), the corrected estimate becomes smoother as the evaluation set grows (Fig. 4), and the labels recover local regressions while preserving the overall task trend (Fig. 1). As a downstream use case, we convert the corrected signal into advantage labels for VLA training, following recent advantage-conditioned policy learning [1, 22], and observe a positive trend in real-robot task success under matched training settings (Tab. I). 

Our contributions are summarized as follows: 

- **Proxy-label perspective.** We formulate normalized time as a noisy proxy for latent physical task progress, and highlight its limitations both within episodes and across episodes in non-monotone manipulation. 

- **Unsupervised correction.** We propose UR-VC, an offline and training-free method that corrects time-derived 

progress labels by aggregating semantically similar states across episodes, requiring no manual progress labels, reward annotations, or an additional value model. 

- **Real-robot evidence.** On real bimanual cloth flatten-andfold data, we validate the conditions needed for correction and demonstrate a downstream use case in advantageconditioned VLA training. 

## II. RELATED WORK 

**Progress and value supervision in robot learning.** Visionlanguage-action models such as RT-2 [5], OpenVLA [15], and the _π_ 0/ _π_ 0 _._ 5 family [4, 3] make large-scale robot policy learning practical, and recent systems consume explicit progress or value signals: _π_ 0<sup>_∗_</sup> _._ 6<sup>[1]learnsavaluefunction</sup> for advantage conditioning, and Gemini Robotics 1.5 [25] uses stage progress for planning and completion judgment. A body of work learns visual progress, reward, or value estimators from contrastive video objectives, language-image representations, value pretraining, optimal transport, or generative modeling [23, 24, 18, 19, 2, 11, 13], or elicits values from large pretrained models [20, 7, 17]. UR-VC is complementary to both lines: it corrects the proxy scores _before_ any estimator is trained, targeting the supervision itself. 

**Cross-episode redundancy and label correction.** URVC is also related to learning with noisy labels [21, 10] and graph/nearest-neighbor label propagation [29, 14]. The difference is the source of redundancy. Rather than propagating sparse human labels over an image graph, UR-VC exploits repeated robot states across independently collected episodes, where each timestamp is a noisy measurement of the same latent physical progress. 

**Deformable manipulation and semantic encoders.** Cloth manipulation has been studied in simulation and real-robot settings, including SoftGym [16], ALOHA/ACT [28], Mobile ALOHA [12], and CLASP [8]. UR-VC studies real bimanual cloth flatten-and-fold data since this setting exposes both within-episode regressions and across-episode timing variation in time-derived progress labels. We use SigLIP-2 [26], an image–text pretrained semantic vision encoder, to retrieve cross-episode state matches. 

## III. METHOD 

We first formalize normalized time as a noisy proxy for latent task progress and explain why its fixed-horizon difference provides a degenerate advantage signal. We then introduce UR-VC, which corrects this proxy by aggregating time-derived labels from semantically matched states across episodes. Finally, we describe how the corrected progress estimate is converted into advantage labels for policy training. 

## _A. Time-derived progress as a noisy proxy_ 

Let episode _e_ contain frames _o_<sup>(</sup> 1<sup>_e_)</sup><sup>_, . . . , o_(</sup> _T_<sup>_e_</sup> _e_<sup>).Acommon</sup> time-derived proxy defines 



treating elapsed time as progress supervision. We assume there is also an unobserved task progress _p_ ( _o_ ) _∈_ [0 _,_ 1], determined by the physical state relevant to the task. This progress _p_ is a conceptual target, not a quantity available to the algorithm. 

The mismatch between _g_ and _p_ appears in two ways. First, within an episode, physical progress can be non-monotone while normalized time is monotone by construction. For instance, if a cloth fold slips or a grasp fails, _p_ ( _o_<sup>(</sup> _t_<sup>_e_))may</sup> decrease even though _gt_<sup>(</sup><sup>_e_)</sup> continues to increase. As a result, raw time labels cannot distinguish actions that improve the task state from actions that stall or locally undo progress. Second, across episodes, the same physical state can appear at different normalized times. Operators may move at different speeds, pause for different durations, or take different recovery paths after local failures. Thus, _gt_<sup>(</sup><sup>_e_)</sup> contains not only information about task progress, but also episode-specific timing distortion. A frame that is physically close to completion may receive a lower time-derived label in a slow episode, while a less advanced state may receive a higher label in a fast episode. 

These two issues directly affect downstream learning. For advantage conditioning with a fixed action horizon _H_ , the finite difference of the raw time proxy is 



constant within episode _e_ . Therefore, within the same episode, naively differencing the raw label gives every frame the same ranking signal, regardless of whether the action improves or worsens the physical state. 

Across episodes the difference varies only through the episode length _Te_ , which primarily reflects execution speed rather than state improvement. More generally, any method 



<!-- Start of picture text -->
99.9%<br>1.0<br>90.4%<br>0.8<br>0.6<br>0.4 primary evaluation set<br>(150 episodes: 98%)<br>0.2<br>similarity   0.90<br>similarity   0.955<br>0.0<br>101 102 103 104<br>number of retrieval episodes<br>frames with a cross-episode match<br><!-- End of picture text -->

Fig. 2: **Cross-episode matches are abundant.** Fraction of task-region frames with at least one semantically similar neighbor in a _different_ episode as more retrieval episodes are added. On real flatten-fold demonstrations, the 150-episode primary evaluation set already covers 98% of frames at similarity _≥_ 0 _._ 90, indicating that independently collected episodes provide sufficient anchors for correcting time-derived progress labels. Coverage further improves at the 10<sup>4</sup> -episode scale, reaching 99.9% at _≥_ 0 _._ 90 and 90.4% at _≥_ 0 _._ 955. 

that consumes raw time labels as progress or value supervision inherits these within-episode monotonicity errors and acrossepisode timing distortions. 



UR-VC corrects a time-derived proxy by exploiting crossepisode recurrence: similar physical states often appear in independently collected demonstrations, but at different normalized times. For an idealized recurring state, episode _e_ ’s time-derived score can be viewed as 



where _p_ is the latent progress and _εe_ is an episode-specific timing error. Averaging proxy scores across independent episodes can reduce this error; averaging many nearby frames from the same episode would not, since their errors are correlated. 

Concretely, let _fi_ be the _L_ 2-normalized SigLIP-2 embedding of query frame _i_ , and let _fi_<sup>_⊤fj_denotecosinesimilarity.</sup> For every other episode _e_ , we form an in-band candidate set 



We then keep only the best representative from that episode, 



and discard it if its similarity is below _ρ_ . Let _Mi_ be the remaining episode representatives, optionally capped to the top _m_ episodes by similarity. The corrected estimate is 





<!-- Start of picture text -->
highly similar frames in other episodes  (similarity shown)<br>query  p=0.09 sim=0.961 sim=0.956 sim=0.956 sim=0.955 sim=0.953<br>query  p=0.80 sim=0.966 sim=0.965 sim=0.964 sim=0.962 sim=0.962<br>query  p=0.84 sim=0.972 sim=0.960 sim=0.960 sim=0.957 sim=0.955<br>query  p=0.96 sim=0.953 sim=0.951 sim=0.950 sim=0.949 sim=0.948<br><!-- End of picture text -->

Fig. 3: **Cross-episode retrieval captures folding state across garment appearances.** For query frames at successive fold stages (left, black borders), nearest neighbors retrieved from _other_ episodes (red borders, similarity shown) exhibit the same folding state across different garment colors. 

If no representative survives, we leave _gi_ unchanged. 

The episode-balanced form matches the statistical assumption above: each episode contributes at most one proxy score to the average, so the estimate aggregates evidence across demonstrations rather than across temporally adjacent frames from a single trajectory. The similarity threshold _ρ_ acts as a conservative match filter, preventing episodes without a sufficiently similar state from contributing to the estimate. The temporal band _τ_ imposes a locality constraint on label correction: all averaged labels satisfy _|gj − gi| ≤ τ_ , and therefore the correction magnitude is bounded as _|g_ ˆ _i −gi| ≤ τ_ . 

UR-VC is an offline label-correction procedure. After computing visual embeddings, it only requires similarity search, per-episode representative selection, and scalar averaging. In implementation, the per-episode argmax is computed with a single masked scatter-max over the offline corpus, so the correction reduces to a small number of matrix operations. UR-VC doesn’t require manual progress labels, reward annotations, online rollouts, or training an additional value model. 

Near the end of an episode, we compute the difference to the final frame and rescale it to the horizon rate: 



Here we follow the indexing convention _o_<sup>(</sup> 1<sup>_e_)</sup><sup>_, . . . , o_(</sup> _T_<sup>_e_</sup> _e_<sup>).Ifthe</sup> implementation uses zero-indexed frames, the denominator should be adjusted accordingly. 

We rank frames by _ri_ and mark the top 20% as positiveadvantage examples. The corresponding training frames receive a plain-text prompt suffix, e.g., “..., advantage: positive”. At deployment, the policy is queried with the same positive-advantage suffix. The policy architecture, training data, and optimization schedule are otherwise unchanged. 

Thus, UR-VC introduces no new policy objective and no separate value model. It only replaces the raw time-derived supervision with a corrected signal, allowing existing progressor advantage-conditioned pipelines to use cross-episode information without additional annotation or critic training. 

## IV. EXPERIMENTS 

## _C. Advantage Labels from Corrected Progress_ 

After correcting the time-derived proxy, we use the resulting estimate _g_ ˆ to construct advantage labels for policy training following _π_ 0<sup>_∗_</sup> _._ 6<sup>.ThisstepisonlyonedownstreamuseofUR-</sup> VC: the correction is performed at the progress-label level, before any policy or value model is trained. 

For an action chunk starting at frame _i_ with horizon _H_ , we define a scalar corrected advantage proxy as 



We evaluate UR-VC on real bimanual cloth flatten-andfold data. Since true physical progress is not directly observed in real demonstrations, we do not claim access to ground-truth progress labels. Instead, our evaluation follows the requirements implied by the method: cross-episode state recurrences should be sufficiently dense, retrieved frames should correspond to the same semantic task state rather than superficial appearance, the corrected labels should express non-monotone progress, and the resulting signal should be usable in a downstream robot-learning pipeline. 



<!-- Start of picture text -->
0.0550 1.00<br>0.0525 primary evaluation  se t<br>0.98<br>0.0500<br>0.0475<br>0.96<br>roughness of raw estimate<br>0.0450 episode-balanced coverage<br>0.0425 0.94<br>0.0400<br>0.92<br>0.0375<br>0.0350 0.90<br>102 103 104<br>number of retrieval episodes<br>2||mean g coverage<br><!-- End of picture text -->

Fig. 4: **The correction improves with more retrieval episodes.** Episode-balanced UR-VC across retrieval-set sizes, reported as mean _±_ std over random subsets. The roughness of the corrected progress estimate, measured by mean _|_ ∆<sup>2</sup> _g_ ˆ _t|_ , decreases by roughly one third from 50 episodes to the 10<sup>4</sup> - episode scale. Meanwhile, coverage remains high and rises from 96% to 99.9%, showing that larger retrieval sets provide both denser support and more stable correction. The 150episode primary evaluation set already lies in a usable regime. 

## _A. Experimental Setup_ 

**Task and data.** We study a long-horizon bimanual cloth flatten-and-fold task. Each episode starts from a garment placed on a table and requires the robot to flatten the cloth and then fold it into the target configuration. This task is a natural setting for evaluating time-derived progress proxies: progress is visually meaningful, intermediate states recur across garments and episodes, and local regressions frequently occur due to slips, failed grasps, wrinkles, or partial undoing. Unless otherwise stated, intrinsic analyses use a primary evaluation set of 150 real robot episodes sampled from a flatten-and-fold dataset released by [27]. We additionally report scaling trends by increasing the retrieval pool up to the 10<sup>4</sup> -episode scale. 

**UR-VC implementation.** For each frame, we compute an _L_ 2-normalized SigLIP-2 visual embedding and retrieve semantically similar frames from other episodes. Unless otherwise stated, we use a temporal band _τ_ = 0 _._ 3 and a cosine-similarity threshold _ρ_ = 0 _._ 90, which provide a conservative locality constraint and filter weak cross-episode matches. 

**Metrics.** We evaluate four aspects of the correction. First, _coverage_ measures the fraction of query frames that have at least one valid match from a different episode. Second, _semantic consistency_ checks whether retrieved frames preserve task state across appearance variation. Third, _non-monotone recovery_ measures whether the corrected estimate produces negative horizon advantages in states where progress locally regresses, while preserving the overall task trend. Finally, we test a downstream use case by using UR-VC-derived advantage labels to train an advantage-conditioned VLA and evaluate it on a dual 7-DoF AgileX robot with absolute joint control. 

## _B. Cross-Episode Coverage and Scaling_ 

UR-VC is useful only if query states have high-quality matches in other episodes. This condition already holds at the scale of the primary evaluation set. With 150 episodes, 98% of frames have a nearest neighbor in a different episode with cosine similarity at least 0.90; 69.9% remain covered under the stricter threshold 0.955. 

Coverage further improves with retrieval-set size. As the retrieval pool increases from the primary set to the 10<sup>4</sup> -episode scale corpus, coverage rises to 99.9% at threshold 0.90 and 90.4% at threshold 0.955 (Fig. 2). This suggests that crossepisode redundancy is not only present in large-scale robot datasets but is already available at moderate collection scales, and it becomes denser as more demonstrations are added. 

## _C. Semantic Consistency of Retrieved States_ 

High coverage alone is insufficient: visually similar frames must also correspond to similar task states. Otherwise, averaging their time-derived labels would blur together unrelated configurations rather than correct temporal distortion. We therefore inspect nearest-neighbor retrievals across different episodes and garments (Fig. 3). The retrieved frames preserve the folding state across garment colors and appearances, indicating that the embedding primarily captures cloth configuration rather than superficial texture. These results support the main assumption behind UR-VC: semantically similar task states recur across independently collected episodes, and retrieval can identify them despite appearance variation. 

## _D. Recovering Non-Monotone Progress_ 

We next examine whether UR-VC expresses progress patterns that normalized time cannot represent. A normalizedtime label is monotone by construction, so its fixed-horizon advantage is non-negative and nearly constant within an episode. It therefore cannot distinguish an action that improves the cloth state from one that stalls or temporarily undoes progress. 

UR-VC preserves the global task trend while allowing local regressions. In the real episode shown in Fig. 1, the corrected estimate decreases when a slipping grasp reintroduces wrinkles and the cloth state visibly worsens, whereas the raw time-derived label continues to increase. Across the primary evaluation set, 13.4% of frames receive a negative horizon advantage under UR-VC, using ( _ri < −_ 0 _._ 02 with a horizon of 5% of the episode length, approximately _≈_ 1.7 s). At the same time, the corrected estimate remains highly correlated with normalized time overall (0.98), which is expected because the temporal band constrains the correction to remain local. Thus, UR-VC does not discard the coarse temporal ordering of the demonstration; it selectively corrects local regions where the visual state suggests that progress has regressed. 

The correction also becomes more stable as the retrieval pool grows. From 50 episodes to the 10<sup>4</sup> -episode scale, the roughness of the corrected estimate, measured by mean _|_ ∆<sup>2</sup> _g_ ˆ _t|_ , decreases by roughly one third, while coverage increases from 96% to 99.9 % (Fig. 4). Larger retrieval sets provide both denser support and smoother estimates. 

TABLE I: **Downstream real-robot evaluation.** Conditions correspond to different tablecloth backgrounds, with 30 trials per condition. UR-VC improves the average success rate from 72.8% to 78.9% and performs better in 5 of 6 conditions. 

|Condition|Baseline|UR-VC|
|---|---|---|
|Bare table|0.90|**0.97**|
|Beige cloth|0.70|**0.73**|
|Blue-gray cloth|**0.50**|0.43|
|Light-yellow cloth|0.63|**0.90**|
|Light-gray cloth|0.77|**0.80**|
|Khaki cloth|0.87|**0.90**|
|Average|0.728|**0.789**|



## _E. Downstream Real-Robot Evaluation_ 

UR-VC produces a corrected progress signal independently of how that signal is consumed. We evaluate one downstream use case: advantage-conditioned VLA training in the style of _π_ 0<sup>_∗_</sup> _._ 6<sup>,usinga</sup><sup>_π_0</sup><sup>_._5backbone.</sup> 

**Setup.** All methods share the same human-collected training set, model architecture, training schedule, and hyperparameters; one policy is trained per labelling scheme. The training mix combines 5700 flatten-and-fold demonstrations with 1795 dedicated _recovery_ demonstrations that pass through degraded cloth states—precisely the regime where time-derived labels are least reliable. The no-advantage baseline trains without advantage labels. The UR-VC variant computes corrected progress with Eq. 6, marks its top-20% advantage frames as positive (Sec. III-C), and conditions the policy on the resulting positive/negative labels. 

**Evaluation protocol.** We evaluate on an AgileX bimanual robot performing flatten-then-fold on short-sleeve garments. The test set contains six table conditions, including a baretable condition and five tablecloth backgrounds. For each condition, we evaluate 6 garments with 5 repetitions each, yielding 30 trials per condition and 180 trials per method. We report per-condition and the average success rates in Tab. I. 

**Results.** UR-VC advantage conditioning improves average success from 0.728 (131/180) to 0.789 (142/180), with higher success in 5 of 6 table conditions. These results provide application-level evidence that UR-VC-derived advantage labels can improve policy performance when inserted into a standard advantage-conditioned VLA pipeline. Importantly, the improvement is obtained without changing the backbone, training data, optimization schedule, or policy objective, suggesting that the gains are attributable to the supervision signal rather than additional model capacity or training changes. 

## V. LIMITATIONS 

UR-VC assumes that visually similar observations indicate similar task progress. This assumption is shared with progress estimator and value models: when visually similar states occur at different task stages or after different action histories, their latent progress can be ambiguous. Episode-balanced averaging reduces trajectory-specific timing noise, but it cannot correct systematic retrieval errors. 

Our downstream evaluation focuses on one representative use case: using UR-VC-derived supervision for advantageconditioned VLA training in real bimanual cloth manipulation. Under matched experimental settings, the corrected labels lead to higher average success, suggesting their practical utility as a drop-in supervision signal. Future work may further examine the method across broader task families, training scales, and hyperparameter settings. 

## VI. CONCLUSION 

UR-VC corrects time-derived progress proxies by averaging semantic state recurrences across independent robot episodes. It exploits dataset redundancy to better align scalable proxy labels with latent physical progress. On real bimanual cloth data, cross-episode matches are abundant and semantic, corrected estimates recover regressions that a monotone time proxy cannot express, and the resulting signal can be used for advantage-conditioned VLA training. Time provides useful and scalable supervision, but it should not be conflated with physical progress. As a model-agnostic preprocessing step, UR-VC refines timestamp-derived labels before downstream model training, without requiring an additional value model or any changes to the model architecture or training objective. 

## REFERENCES 

- [1] Ali Amin, Raichelle Aniceto, Ashwin Balakrishna, Kevin Black, Ken Conley, Grace Connors, James Darpinian, Karan Dhabalia, Jared DiCarlo, et al. _π_ 0<sup>_∗_</sup> _._ 6<sup>:avlathat</sup> learns from experience. _arXiv.org_ , 2511.14759. 

- [2] Chethan Bhateja, Derek Guo, Dibya Ghosh, Anikait Singh, Manan Tomar, Quan Vuong, Yevgen Chebotar, Sergey Levine, and Aviral Kumar. Robotic offline rl from internet videos via value-function pre-training. _arXiv.org_ , 2309.13041. 

- [3] Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Robert Equi, Chelsea Finn, Niccolo Fusai, Manuel Y Galliker, et al. _π_ 0 _._ 5: A vision-language-action model with openworld generalization. In _CoRL_ , 2025. 

- [4] Kevin Black, Noah Brown, Danny Driess, Adnan Esber, Michael Equi, Chelsea Finn, et al. _π_ 0: A vision-languageaction flow model for general robot control. In _RSS_ , 2025. 

- [5] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al. RT-2: Vision-language-action models transfer web knowledge to robotic control. In _CoRL_ , 2023. 

- [6] Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Misha Laskin, Pieter Abbeel, Aravind Srinivas, and Igor Mordatch. Decision transformer: Reinforcement learning via sequence modeling. In _NeurIPS_ , 2021. 

- [7] Shirui Chen, Cole Harrison, Ying-Chun Lee, Angela Jin Yang, Zhongzheng Ren, Lillian J Ratliff, Jiafei Duan, Dieter Fox, and Ranjay Krishna. Topreward: Token 

probabilities as hidden zero-shot rewards for robotics. _arXiv.org_ , 2602.19313. 

- [8] Yuhong Deng, Chao Tang, Cunjun Yu, Linfeng Li, and David Hsu. Clasp: General-purpose clothes manipulation with semantic keypoints. _arXiv.org_ , 2408.08160. 

- [9] Scott Emmons, Benjamin Eysenbach, Ilya Kostrikov, and Sergey Levine. Rvs: What is essential for offline rl via supervised learning? In _ICLR_ , 2022. 

- [10] Benoˆıt Fr´enay and Michel Verleysen. Classification in the presence of label noise: a survey. _IEEE Transactions on Neural Networks and Learning Systems_ , 2013. 

- [11] Yuwei Fu, Haichao Zhang, Di Wu, Wei Xu, and Benoit Boulet. Robot policy learning with temporal optimal transport reward. In _NeurIPS_ , 2024. 

- [12] Zipeng Fu, Tony Z Zhao, and Chelsea Finn. Mobile aloha: Learning bimanual mobile manipulation with lowcost whole-body teleoperation. In _CoRL_ , 2024. 

- [13] Tao Huang, Guangqi Jiang, Yanjie Ze, and Huazhe Xu. Diffusion reward: Learning rewards via conditional video diffusion. In _ECCV_ , 2024. 

- [14] Ahmet Iscen, Giorgos Tolias, Yannis Avrithis, and Ondrej Chum. Label propagation for deep semi-supervised learning. In _CVPR_ , 2019. 

- [15] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. OpenVLA: An open-source vision-language-action model. In _CoRL_ , 2024. 

- [16] Xingyu Lin, Yufei Wang, Jake Olkin, and David Held. Softgym: Benchmarking deep reinforcement learning for deformable object manipulation. In _CoRL_ , 2021. 

      - and Google Brain. Time-contrastive networks: Selfsupervised learning from video. In _ICRA_ , 2018. 

   - [24] Sumedh Sontakke, Jesse Zhang, S´eb Arnold, Karl Pertsch, Erdem Bıyık, Dorsa Sadigh, Chelsea Finn, and Laurent Itti. Roboclip: One demonstration is enough to learn robot policies. In _NeurIPS_ , 2023. 

   - [25] Gemini Robotics Team, Abbas Abdolmaleki, Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Ashwin Balakrishna, Nathan Batchelor, Alex Bewley, Jeff Bingham, et al. Gemini robotics 1.5: Pushing the frontier of generalist robots with advanced embodied reasoning, thinking, and motion transfer. _arXiv.org_ , 2510.03342. 

   - [26] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. Siglip 2: Multilingual visionlanguage encoders with improved semantic understanding, localization, and dense features. _arXiv.org_ , 2502.14786. 

   - [27] Checheng Yu, Chonghao Sima, Gangcheng Jiang, Hai Zhang, Haoguang Mai, Hongyang Li, Huijie Wang, Jin Chen, Kaiyang Wu, Li Chen, et al. _χ_ 0: Resource-aware robust manipulation via taming distributional inconsistencies. _arXiv.org_ , 2602.09021. 

   - [28] Tony Z Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning fine-grained bimanual manipulation with low-cost hardware. In _RSS_ , 2023. 

   - [29] Xiaojin Zhu, Zoubin Ghahramani, and John Lafferty. Semi-supervised learning using gaussian fields and harmonic functions. In _ICML_ , 2003. 

- [17] Jindi Lv, Hao Li, Jie Li, Yifei Nie, Fankun Kong, Yang Wang, Xiaofeng Wang, Zheng Zhu, Chaojun Ni, Qiuping Deng, et al. Viva: A video-generative value model for robot reinforcement learning. _arXiv.org_ , 2604.08168. 

- [18] Yecheng Jason Ma, Vikash Kumar, Amy Zhang, Osbert Bastani, and Dinesh Jayaraman. Liv: Language-image representations and rewards for robotic control. In _ICML_ , 2023. 

- [19] Yecheng Jason Ma, Shagun Sodhani, Dinesh Jayaraman, Osbert Bastani, Vikash Kumar, and Amy Zhang. Vip: Towards universal visual reward and representation via value-implicit pre-training. In _ICLR_ , 2023. 

- [20] Yecheng Jason Ma, Joey Hejna, Chuyuan Fu, Dhruv Shah, Jacky Liang, Zhuo Xu, Sean Kirmani, Peng Xu, Danny Driess, Ted Xiao, et al. Vision language models are in-context value learners. In _ICLR_ , 2025. 

- [21] Nagarajan Natarajan, Inderjit S Dhillon, Pradeep K Ravikumar, and Ambuj Tewari. Learning with noisy labels. In _NeurIPS_ , 2013. 

- [22] Xue Bin Peng, Aviral Kumar, Grace Zhang, and Sergey Levine. Advantage-weighted regression: Simple and scalable off-policy reinforcement learning. _arXiv.org_ , 1910.00177. 

- [23] Pierre Sermanet, Corey Lynch, Yevgen Chebotar, Jasmine Hsu, Eric Jang, Stefan Schaal, Sergey Levine, 

