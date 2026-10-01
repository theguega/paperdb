# **VLA Grounder: Language-Conditioning Space Optimization for Black-Box VLA Models** 

**Damir Shodiev**<sup>2</sup> **, Aleksei Staroverov**<sup>1</sup><sup>_,_2</sup><sup>_,_3</sup> **, Nikita Kachaev**<sup>1</sup> **, Alexey K. Kovalev**<sup>1</sup><sup>_,_2</sup> **, Aleksandr I. Panov**<sup>1</sup><sup>_,_2</sup> 1AXXX, 2MIRAI, 3MISIS 

damir.shodiev.pro@gmail.com 

## **Abstract** 

Vision-Language-Action (VLA) models are commonly treated as end-to-end action policies conditioned on natural-language task descriptions. In practice, however, their behavior often depends sharply on how the instruction is phrased, suggesting that language is not merely a task label but an optimizable conditioning input. We study whether frozen VLA policies can be improved by optimizing language space rather than updating action weights. Our method introduces a language-conditioning space policy that translates a human instruction into a short VLAgrounded command using object appearance, spatial relations, and target-grounding cues. The language-conditioning space policy is initialized with a failure-derived command-space prior and optimized with reinforcement learning from sparse task-completion rewards, while the downstream VLA remains fully frozen. This yields language-conditioning space optimization: RL discovers which VLA-grounded commands best elicit successful behavior from the frozen action policy. Experiments on RL4VLA and VL-Think show that languageconditioning space optimization improves success on instruction-sensitive, symbolic, and multi-object manipulation tasks, demonstrating that language can serve as an optimizable variable for a robot foundation models. 

**Website** : https://tttonyalpha.github. io/vla_grounder 

## **1 Introduction** 

Vision-Language-Action (VLA) models represent a promising paradigm in robotic learning, combining visual perception, natural language understanding, and action generation within a single policy. Recent VLA models leverage large-scale pre-training and robotic demonstrations to produce actions directly from images and language, enabling broad 



Figure 1: Visual overview of language-conditioning space optimization. A failure-derived command-space prior conditions a pre-trained VLM to translate the user instruction and scene image into a short VLA-grounded command. This command conditions a VLA policy, improving action success without updating VLA weights. 

zero-shot manipulation across tasks and embodiments (Black et al., 2024; Zhang et al., 2025; Zhong et al., 2025). 

Despite this progress, the language input to a VLA is often treated as a passive task description, even though it directly shapes the action distribution of the policy. In practice, VLA behavior is highly sensitive to how a goal is expressed (Pugacheva et al., 2025; Kachaev et al., 2025). A command such as _"put bread on plate"_ may fail even when the scene contains a visually clear target, while a more grounded command such as _"pick up the brown round object and put it on the yellow_ 

_plate"_ can succeed. Similar effects appear when an object name is visually misleading: _"put champagne glass on plate"_ can be improved by referring to the same object as _"the tall white glass"_ . These examples suggest that many VLA failures arise from an instruction-to-grounding mismatch between human semantic intent and the perceptual categories available to the robot policy. 

This mismatch is amplified by the limited linguistic diversity of robot demonstration datasets (Walke et al., 2024; Collaboration et al., 2025). Most datasets contain short templated instructions and everyday household objects, while benchmark tasks may include ambiguous objects, abstract references, uncommon categories, or multiple visually similar distractors. As a result, the language channel of a VLA is underused: models reserve substantial capacity for text, but the instruction often fails to provide the visual and spatial cues needed for robust action selection (Zhang et al., 2025; Liu et al., 2025; Black et al., 2024). 

A natural response is to modify the instruction rather than the VLA weights. We therefore propose to insert an additional language-conditioning space policy between the human instruction and the frozen VLA, and ask whether command selection can provide an effective adaptation mechanism for large robot foundation models without updating their action weights. Given the image and the original instruction, this policy translates the user’s goal into a short VLA-grounded command that is better aligned with the perceptual and linguistic priors of the downstream VLA. Under this view, language is not a fixed task label, but an input space in which human intent is converted into an executable command for the robot policy. 

We instantiate this idea with a languageconditioning space policy placed upstream of a frozen VLA, as shown in Figure 1. Given the current image and the original human instruction, the language-conditioning space policy produces a short VLA-grounded command that exposes execution-relevant cues: object appearance, color, shape, spatial position, relative location, and simplified target descriptions. The generated command is then used as the language input to the unchanged VLA policy. Because the downstream action model is kept fixed, the method adapts the conditioning signal rather than the action policy, making it noninvasive for different frozen VLA models. 

We train the language-conditioning space policy (VLA Grounder) with Reinforcement Learn- 

ing (RL) from sparse task-completion rewards while keeping the downstream VLA fully frozen (Figure 1)). This casts adaptation as _languageconditioning space optimization_ : the learned decision variable is not a continuous robot action or a VLA weight update, but the VLA-grounded command through which the frozen action model receives and grounds the task. To make this optimization meaningful, we use a failure-derived command-space prior that restricts exploration to concise, visually grounded, and VLA-compatible commands. Within this structured command space, RL provides an action-grounded signal for discovering which commands most reliably elicit successful behavior from the existing policy. Experiments on VL-Think (Kachaev et al., 2025) and RL4VLA (Liu et al., 2025) benchmarks show that our method creates better grounded guidance commands for frozen _π_ 0 and OpenVLA models, increasing downstream task success across object grounding and multi-object manipulation settings. 

The contributions of this work are as follows: 

1. **Language-conditioning space optimization for VLA models** : We formulate VLA adaptation as learning a language-conditioning space policy that maps human instructions to VLAgrounded commands while keeping the downstream action policy fixed. 

2. **RL in language-conditioning space** : We introduce VLA Grounder, languageconditioning space policy that maps an image and human instruction to a concise VLA-grounded command. We optimize only the language-conditioning space policy from sparse task-completion rewards, showing that downstream robot behavior can improve without updating VLA action weights. 

3. **Empirical and diagnostic evidence** : We evaluate on VL-Think and RL4VLA with frozen _π_ 0 and OpenVLA backbones. The results show improved success on symbolic grounding, multi-object manipulation, and fixedobject settings, and our analyses indicate that optimized commands provide more grounded conditioning signals for the downstream VLA. 

## **2 Related Work** 

### **2.1 Language-Conditioned VLA Policies** 

Large-scale robot policies increasingly separate high-level semantic control from low-level motor 

generation. ACT introduced action chunking for manipulation (Zhao et al., 2023), RT-2 transferred web-scale vision-language pretraining into robot action generation (Brohan et al., 2023), and recent generalist policies such as OpenVLA and _π_ 0 scale this recipe to broad zero-shot manipulation (Kim et al., 2024; Black et al., 2024). Embodied chainof-thought and _π_ 0 _._ 5 further push toward richer intermediate reasoning and semantic control over action generation (Zawalski et al., 2025; Intelligence et al., 2025b). Another line of work makes a VLA more controllable by enriching its conditioning signal. Higher-level semantic prediction in _π_ 0 _._ 5, semantic guidance methods for VLA robustness, and prompt optimization methods in vision-language settings all support the broader idea that behavior depends strongly on how context is represented (Intelligence et al., 2025b; Zhan et al., 2026; Lee et al., 2023). These approaches treat language, metadata, or intermediate semantic variables as mechanisms for conditioning behavior. 

Our paper asks a complementary post-training question: given an already frozen VLA, can we learn a better VLA-grounded command for control? Rather than changing the downstream action policy, we optimize the upstream language-conditioning space policy that conditions it. 

### **2.2 RL Adaptation of VLA Models** 

Reinforcement learning has become a central tool for improving VLA policies after imitation learning. RL4VLA studies PPO (Schulman et al., 2017), GRPO (Shao et al., 2024)-, and preference-based fine-tuning for VLA generalization (Liu et al., 2025); _π_ RL extends online RL to flow-based VLA models (Chen et al., 2026); and _π_ 0<sup>_∗_</sup> _._ 6<sup>showshow</sup> deployment and corrective data can substantially improve a VLA through continued learning (Intelligence et al., 2025a). More broadly, reward optimization for language policies, including ArCHer, PReWrite, and StablePrompt, shows that sparse reward can shape text-producing policies in useful ways (Zhou et al., 2024; Kong et al., 2024; Kwon et al., 2024). In contrast, our method adapts the language policy itself, not the VLA model weights, enabling lightweight post-training in languageconditioning space. 

### **2.3 VLA Language Robustness** 

Recent work shows that language remains an underdeveloped modality in embodied AI. Dataset audits find highly repetitive and template-like in- 

structions in robot corpora (Wanna et al., 2026). Stable Language Guidance frames related failures as modality imbalance, where strong visual priors overpower sparse linguistic signals (Zhan et al., 2026), adversarial paraphrases and irrelevant context can significantly degrade VLA behavior even when task intent is preserved (Pugacheva et al., 2025; Zinkovich et al., 2025), and mechanistic analyses suggest that language-conditioned behavior is present internally but not always reliably exposed by the input instruction (Häon et al., 2025). We build on this diagnosis but reframe it as an optimization problem over command space. Instead of treating language brittleness as only a robustness issue, we treat language conditioning as an actionable input space that can be optimized for downstream execution success. 

## **3 Problem Formulation** 

### **3.1 Partially Observable Control with Language Instructions** 

We consider an episodic partially observable = Markov decision process (POMDP) _M_ ( _S, A, O, P, Z, r, ρ_ 0 _, γ_ ), where _S_ is the latent state space, _A_ the primitive action space, _O_ the observation space, _P_ the transition kernel, _Z_ the observation kernel, _r_ the reward function, _ρ_ 0 the initialstate distribution, and _γ ∈_ [0 _,_ 1) the discount factor. At time _t_ , the environment occupies state _st_ , emits _ot ∼ Z_ ( _· | st_ ), receives action _at_ , produces reward _r_ ( _st, at_ ), and transitions to _st_ +1 _∼ P_ ( _· | st, at_ ). Each episode is paired with a fixed human instruction _u ∈U_ H. We write the discounted return of a trajectory _τ_ as _R_ ( _τ_ ) =<sup>�</sup><sup>_T_</sup> _t_ =0<sup>_−_1</sup><sup>_γtrt_.</sup> 

### **3.2 Frozen Vision-Language-Action Policies** 

A pretrained Vision-Language-Action policy maps an observation and a language command to an action distribution, _π_ VLA : _O × U →_ ∆( _A_ ), where _U_ is the space of commands consumed by the action policy. In standard deployment, the raw instruction is passed directly to the policy, so _at ∼ π_ VLA( _· | ot, u_ ). More generally, each command _u_ ˜ _∈U_ induces a frozen low-level policy _π_ VLA<sup>_u_˜(</sup><sup>_a|o_):=</sup><sup>_π_VLA(</sup><sup>_a|o,_˜</sup><sup>_u_).Distinctcom-</sup> mands that preserve the same task intent may still induce different grounded behavior. 

Let _U_ ( _u_ ) _⊆U_ denote the command space associated with instruction _u_ , i.e., the set of commands that are semantically consistent with _u_ but may differ in wording, specificity, or grounding cues. The 



Figure 2: Model architecture and training loop. The proposed method inserts a scene-conditioned languageconditioning space policy before a frozen VLA action policy. Given the initial task instruction, the scene image, and a failure-derived command-space prior that specifies the desired command format, the language-conditioning space policy generates a VLA-grounded command using object appearance, spatial cues, and target-location information. The command is passed to the unchanged VLA policy for action prediction. During training, only the languageconditioning space policy is optimized: rollouts in the environment produce sparse task-completion rewards, which are used as the RL signal for improving command generation while keeping the VLA weights frozen. 

goal is not to learn a new low-level action policy, but to exploit this command space to obtain better behavior from a frozen one. 

sample the primitive action from _π_ VLA( _· | o,_ ˜ _u_ ). This induces an action-space transformation from primitive actions to language commands. 

### **3.3 Problem Setting** 

We are given a pretrained frozen VLA policy _π_ VLA and seek to maximize return in _M_ by selecting a command ˜ _u ∈U_ ( _u_ ) rather than by changing actionmodel weights. Equivalently, the learned component chooses which member of the frozen family _{π_ VLA<sup>_u_˜:</sup><sup>_u_˜</sup><sup>_∈U_(</sup><sup>_u_)</sup><sup>_}_will generate actions during</sup> the rollout. This formulation is useful only when the policy is sensitive to language-space variation, meaning that semantically valid commands can induce meaningfully different behavior. We assume only black-box access: for any observation _o_ and command _u_ ˜, we may sample from _π_ VLA( _· | o,_ ˜ _u_ ), but we do not use weights, gradients, or internal activations. 

## **4 Method** 

### **4.1 RL in Language-Conditioning space** 

In standard deployment, a VLA acts under the raw instruction _u_ . We instead treat the command itself as the high-level decision variable: at observation _o_ , we first select a command _u_ ˜ _∈U_ ( _u_ ) and then 





These induced kernels define a transformed control problem, denoted _Mu_<sup>lang</sup> , whose action space is _U_ ( _u_ ) and whose dynamics and rewards are given by _P_<sup>lang</sup> and _r_<sup>lang</sup> . This is the sense in which the adaptation problem can be cast as reinforcement learning in language space. We focus on rollout-level optimization: a policy _qϕ_ (˜ _u | o_ 0 _, u_ ) is invoked once at the beginning of the episode, outputs a single command, and keeps it fixed for the full rollout. Sampling _u_ ˜ _∼ qϕ_ ( _· | o_ 0 _, u_ ) therefore selects one member of the frozen family _{π_ VLA<sup>_u_˜:</sup><sup>_u_˜</sup><sup>_∈U_(</sup><sup>_u_)</sup><sup>_}_.</sup> 



where _D_ is the distribution of initial imageinstruction pairs. The raw frozen-VLA baseline 

is recovered by the degenerate policy that always returns the original instruction. 

### **4.2 Language Aliasing and Structured Exploration** 

Language-Conditioning space has additional structure that naive RL does not exploit. In particular, there may exist distinct commands _u_ ˜ _̸_ = _u_ ˜<sup>_′_</sup> such that _π_ VLA( _· | o,_ ˜ _u_ ) _≈ π_ VLA( _· | o,_ ˜ _u_<sup>_′_</sup> ) for many observations _o_ ; we refer to this as _language aliasing_ . It is common for shallow rewrites, where grammatical edits or near-synonymous substitutions are lexically different but behaviorally redundant. Our method therefore biases exploration toward short executable commands that vary along groundingrelevant dimensions such as source-object description, target cues, color, shape, and spatial location; Figure 7 illustrates this distinction. 

### **4.3 Optimizing the Guiding VLM Policy** 

We instantiate _qϕ_ with a Guiding VLM and train it with GRPO (Shao et al., 2024) using only scalar rollout reward. Figure 2 summarizes the languageconditioning space optimization loop: the policy receives the initial image and instruction, proposes candidate commands, and the frozen VLA policy executes a rollout under each one. For each training pair ( _oi, ui_ ), we sample a group of candidate commands _{u_ ˜<sup>(</sup> _i_<sup>_k_)</sup> _}_<sup>_G_</sup> _k_ =1<sup>from</sup><sup>_qϕ_(</sup><sup>_·|oi, ui_),execute the</sup> frozen VLA under each command, and compare the resulting returns _{Ri_<sup>(</sup><sup>_k_)</sup> _}_<sup>_G_</sup> _k_ =1<sup>within the group.</sup> We use the normalized advantage 



where _σi_ is the empirical standard deviation of rewards in the group. Commands with larger relative advantage are up-weighted by the next update, while weaker ones are down-weighted. Appendix C summarizes the procedure. 

### **4.4 VLA Black-Box Deployment** 

At inference time, the system receives a new image and instruction, samples one command _u_<sup>ˆ</sup> ˜ from the trained language-conditioning space policy, and conditions the frozen VLA policy on that command for the full rollout. There is no reward query, no RL update, and no action-model modification at test time. VLA Grounder changes only the languageconditioning input and requires only black-box forward access to the downstream action policy. In 

this paper, we evaluate this setup on frozen _π_ 0 and OpenVLA backbones. 

### **4.5 Why Language Exploration Matters** 

Group-based RL algorithms, such as GRPO, derive a signal by comparing multiple samples for the same task instance. If the sampled set differs only in grammar, then the reward cannot meaningfully separate better and worse conditioning signals. By contrast, if samples vary in robot-relevant grounding cues, for example, color, left-right position, object description, or target relation—the optimization process can assign credit to language transformations that actually change downstream behavior. 

## **5 Experiments** 

We evaluate on two benchmark families. VL-Think (Kachaev et al., 2025) isolates visual-language reasoning under fixed manipulation: the robot places the same source object on a target board matching an abstract concept such as an arrow, sign, shape or weather icon. RL4VLA (Liu et al., 2025) tests broader semantic generalization; we use _MultiPlate_ , _MultiCarrot_ , and fixed-object _Pepper_ , which require distractor handling and objecttarget grounding. All reported VLA backbones are frozen, and we report success rate as mean _±_ standard deviation. 

The primary baseline is the frozen VLA conditioned on the raw human instruction. We also compare against the structured language-conditioning space policy before RL (“w/o GRPO”), generic command rewriting without structured reasoning (“no reasoning”), and structured-guidance variants with different language-conditioning space policy sizes. For Pepper, we include TextGrad (Yuksekgonul et al., 2024) and GEPA (Agrawal et al., 2025), two general-purpose prompt-optimization baselines that rewrite or mutate commands in language space. We organize the empirical section around main research questions: 

**RQ1: Does language-conditioning space optimization improve frozen VLA policies across backbones?** Our experiments on VL-Think and RL4VLA (Table 1, Table 2, Figure 4) reveal a consistent effect: changing only the language input can substantially improve frozen action policies. The effect appears for both _π_ 0 and OpenVLA, which suggests that the method is not exploiting a single backbone-specific quirk. The gains are most visible 

Table 1: Main results on VL-Think. All VLA backbones are frozen. The language-conditioning space policy uses the strongest structured reasoning configuration, Qwen3.5-9B. 

|||_π_0|||OpenVLA||Ope|nVLA|
|---|---|---|---|---|---|---|---|---|
|Task|orig|Qwen3.5-9B<br>w/o GRPO|Qwen3.5-9B|orig|Qwen3.5-9B<br>w/o GRPO|Qwen3.5-9B|TextGrad|GEPA|
|Arrow|4_._2_±_4_._8|4_._7_±_0_._0|**41**_._**7**_±_**2**_._**0**|13_._5_±_3_._2|21_._4_±_2_._0|**62**_._**5**_±_**1**_._**3**|44_._0_±_2_._5|51_._3_±_2_._8|
|Color|32_._3_±_6_._4|25_._0_±_1_._3|**40**_._**1**_±_**4**_._**5**|55_._7_±_8_._7|63_._5_±_4_._5|**82**_._**8**_±_**2**_._**6**|61_._2_±_3_._6|73_._2_±_1_._2|
|Laundry|12_._5_±_1_._3|10_._4_±_0_._7|**25**_._**0**_±_**4**_._**6**|15_._6_±_2_._6|21_._9_±_5_._6|37_._5_±_3_._8|19_._1_±_3_._6|**47**_._**0**_±_**2**_._**1**|
|Public Info|9_._4_±_1_._3|14_._6_±_2_._0|**43**_._**8**_±_**2**_._**2**|21_._4_±_3_._7|27_._1_±_3_._2|**65**_._**6**_±_**3**_._**4**|25_._6_±_1_._6|48_._5_±_3_._1|
|Shape|12_._0_±_5_._9|21_._4_±_3_._2|**43**_._**2**_±_**7**_._**8**|27_._6_±_1_._5|49_._0_±_7_._7|**74**_._**5**_±_**6**_._**4**|54_._2_±_3_._3|59_._5_±_4_._1|
|Traffic|5_._7_±_2_._0|8_._3_±_2_._7|**32**_._**3**_±_**4**_._**5**|16_._7_±_2_._7|28_._1_±_4_._6|**63**_._**0**_±_**4**_._**8**|35_._8_±_3_._4|49_._4_±_5_._1|
|Weather|12_._0_±_2_._0|14_._6_±_2_._7|**44**_._**8**_±_**2**_._**7**|20_._3_±_4_._6|31_._3_±_3_._4|**61**_._**5**_±_**3**_._**9**|30_._4_±_4_._1|57_._2_±_3_._4|
|Avg.|12_._6|14_._1|**38**_._**7**|24_._4|34_._6|**63**_._**9**|38_._6|55_._2|



on VL-Think, where success depends heavily on converting abstract or symbolic targets into visually grounded commands. RL4VLA shows a smaller but still consistent trend, which is expected because these tasks add multi-object distractors and execution errors that cannot be removed by language alone. 

**RQ2: Does GRPO improve over the non-RL language-conditioning space policy and generic prompt-optimization baselines?** The non-RL language-conditioning space policy already provides a useful command prior, but it does not reliably identify which command aliases the frozen action policy will execute. GRPO adds exactly this missing selection pressure: candidate commands are compared by downstream task reward rather than by linguistic plausibility. This distinction is clearest on VL-Think, where the trained policy separates sharply from its non-RL version. On RL4VLA the improvement is more modest, but remains positive despite higher action-side noise. 

**RQ3: How does reward-trained language optimization compare with traditional prompt optimization?** We also tested generic prompt- 



Figure 3: Trajectories comparison on RL4VLA MultiObject with frozen _π_ 0. The default command underspecifies the object and causes unstable motion, while the rewritten command adds visible grounding cues and yields a more direct successful trajectory. 



Figure 4: GRPO training for VL-Think tasks. Each panel reports rollout success rate over training steps for a symbolic grounding category. The upward trends indicate that reward updates to the language-conditioning space policy improve command selection while the downstream VLA action policy remains frozen. 

optimization methods, including TextGrad and GEPA, using the same frozen-VLA setting. Our results in Table 1 show that these methods can improve over the raw instruction baseline, but not nearly as much as GRPO-based RL adaptation. The reason is qualitative as well as quantitative: generic prompt optimizers often modify punctuation, wording, or writing style in ways that are hard to interpret as better robot grounding. VLA Grounder instead updates the language-conditioning space policy from rollout reward, so it favors command changes that actually alter execution success rather than commands that only look cleaner as text. 

### **RQ4: Do better commands lead to better VLA** 

**representations?** We further probe whether improved commands merely change the final text input or also change the internal representations used by the VLA. Using VL-Think Public Info task, we compare OpenVLA representations under the 

Table 2: Results on the RL4VLA semantic generalization tasks. The same structured language-conditioning space policy improves both frozen _π_ 0 and frozen OpenVLA in multi-object and distractor-heavy settings. 

|Task|orig|_π_0<br>Qwen3.5-9B<br>w/o GRPO|Qwen3.5-9B|orig|OpenVLA<br>Qwen3.5-9B<br>w/o GRPO|Qwen3.5-9B|
|---|---|---|---|---|---|---|
|MultiPlate|9_._9_±_1_._5|10_._4_±_3_._2|**17**_._**2**_±_**2**_._**2**|54_._7_±_2_._2|56_._3_±_4_._6|**64**_._**6**_±_**5**_._**2**|
|MultiCarrot|17_._2_±_5_._1|21_._4_±_4_._5|**29**_._**7**_±_**5**_._**6**|59_._4_±_5_._9|60_._9_±_2_._6|**63**_._**5**_±_**5**_._**3**|
|Avg.|13_._6|15_._9|**23**_._**4**|57_._0|58_._6|**64**_._**0**|



Table 3: Common instruction-grounding failures and useful command rewrite dimensions. The learned policy improves frozen VLA behavior by replacing abstract target names, rare object categories, ambiguous references, or weak object names with visually grounded descriptions that expose color, shape, material, symbol, or spatial cues. 

|Failure type|Raw instruction|Useful rewrite dimension|Language policy rewrite|
|---|---|---|---|
|Abstract target|“sunrise icon”|visible symbol/color|“white card with yellow sun”|
|Rare object name|“champagne glass”|shape/material|“tall white glass”|
|Multi-object ambiguity|“plate”|spatial relation|“left yellow plate”|
|Weak object grounding|“bread”|color/shape|“brown round object”|



Table 4: Linear probing accuracy on VL-Think Public Info tasks using OpenVLA hidden states under the default instruction and the rewritten command. 

|Prompt|Accuracy|
|---|---|
|Default|66_._6|
|VLA Grounder|89_._3|



This pattern indicates that the policy is learning how to reduce the semantic burden placed on the frozen action model. The appendix provides Figure 5 prompt-format examples. 

## **6 Discussion** 

### **6.1 What the Method Actually Adapts** 

original task instruction and under the rewritten command. The probing results Table 4, Figure 3 indicate that rewritten commands produce more informative prompt-conditioned representations: task-relevant distinctions become easier to recover from the VLA hidden states, and this aligns with the downstream success improvements. This suggests that language-conditioning space optimization changes the conditioning representation seen by the action model, rather than only changing a superficial instruction string. 

**RQ5: What command transformations does the learned policy discover?** Based on prompt evolution analysis during training (Table 3) we found that the learned policy tends to produce shorter and more executable commands rather than more verbose descriptions. On VL-Think, it often converts symbolic target labels into compact spatial or visual aliases, such as replacing an icon name with a reference to a white card at a particular position. On RL4VLA and Pepper, it usually keeps familiar receptacle names but rewrites brittle source-object names into visible color and shape descriptions. 

The proposed method does not directly improve low-level motor competence. Instead, it changes how existing capabilities are accessed. This distinction matters both scientifically and practically: if a frozen VLA already contains useful action structure, then a substantial part of adaptation may be achievable by changing the VLA-grounded command that exposes that structure. 

### **6.2 Command-Space Design and Model Scale** 

The ablations suggest that the command space and the language model play different roles. Generic rewriting can already help, which confirms that the raw instruction is often a weak conditioning signal. Structured guidance improves the search space by pushing candidates toward object identity, visual attributes, and spatial grounding rather than shallow paraphrases. Scaling the language model then becomes more useful because the model has a richer space of possible aliases to choose from. The benefit is not perfectly monotonic across every category, however, so scale alone is not the explanation; the strongest setting combines structured command variation with reward-based selection. 

|Task|_π_0|+Qwen3-4B<br>no reasoning|+Qwen3-4B<br>reasoning|+Qwen3.5-4B<br>reasoning|+Qwen3.5-9B<br>reasoning|
|---|---|---|---|---|---|
|Arrow|4_._2_±_4_._8|8_._9_±_1_._0|16_._2_±_2_._7|15_._1_±_2_._6|**41**_._**7**_±_**2**_._**0**|
|Color|32_._3_±_6_._4|22_._9_±_3_._0|25_._0_±_0_._0|29_._2_±_8_._7|**40**_._**1**_±_**4**_._**5**|
|Laundry|12_._5_±_1_._3|22_._4_±_1_._0|20_._8_±_1_._0|13_._5_±_2_._0|**25**_._**0**_±_**4**_._**6**|
|Public Info|9_._4_±_1_._3|19_._8_±_1_._5|23_._4_±_1_._3|24_._0_±_1_._5|**43**_._**8**_±_**2**_._**2**|
|Shape|12_._0_±_5_._9|21_._9_±_3_._4|27_._1_±_3_._7|29_._2_±_3_._0|**43**_._**2**_±_**7**_._**8**|
|Traffic|5_._7_±_2_._0|21_._9_±_1_._3|21_._9_±_2_._6|13_._0_±_2_._0|**32**_._**3**_±_**4**_._**5**|
|Weather|12_._0_±_2_._0|25_._0_±_1_._3|22_._4_±_3_._2|21_._9_±_2_._2|**44**_._**8**_±_**2**_._**7**|
|Avg.|11_._0|21_._1|23_._9|23_._3|**38**_._**8**|



Table 5: Ablation on frozen _π_ 0 over VL-Think benchmark. “No reasoning” denotes a generic command-rewriting policy; “reasoning” denotes the structured guidance used by our method. 

Table 6: Fixed-object RL4VLA result. On Pepper, the structured language-conditioning space policy yields a large gain with the _π_ 0 action policy frozen. 

|Task|_π_0|orig|_π_0 + TextGrad|_π_0 + GEPA|_π_0 + Qwen3.5|
|---|---|---|---|---|---|
|Pepper|39_._1|_±_6_._6|44_._3_±_5_._5|61_._7_±_4_._1|**89**_._**6**_±_**3**_._**2**|



### **6.3 Relation to Other Post-Training Strategies** 

Language-Conditioning space optimization is complementary to direct VLA fine-tuning, rewardmodel training, and verifier-based test-time selection. Fine-tuning can change the action policy itself; verifier methods can select better candidates at inference time; our approach improves the upstream command that conditions the frozen action policy. In this sense, language-conditioning space optimization can be viewed as a lightweight posttraining primitive that may compose naturally with broader VLA post-training pipelines. 

### **6.4 Limits of Language-Conditioning Space Optimization** 

The gains are largest when the user instruction is semantically valid but visually under-specified for the action policy. Symbolic VL-Think tasks such as arrows, public-information signs, weather icons, shapes, and traffic signs benefit because the learned policy can replace the symbolic target with a simpler visual or spatial description. Direct perceptual categories such as Color leave less room for improvement because the raw instruction already exposes a usable cue. The method is also limited on tasks where language is not the only bottleneck: clutter, distractors, wrong object grounding, severe visual misperception, missing low-level manipulation skill, or long-horizon reasoning demands can still dominate the rollout outcome. 

### **6.5 Broader Implication** 

The broader implication of this work is that language can function as an optimizable adaptation layer for robot foundation models. If this view is correct, then future post-training of VLAs need not operate only in weight space or action space; it can also operate in the space of semantic conditioning signals that mediate between human intent and embodied action. 

## **7 Conclusion** 

We study adaptation of frozen VLA policies through the language-conditioning input rather than through action-weight updates. The resulting perspective treats language as an optimizable conditioning variable and casts improvement as reinforcement learning in language space. VLA Grounder combines a language-conditioning space policy, a failure-derived command-space prior, and a sparse downstream task reward to discover VLAgrounded commands that better expose latent capability in frozen action policies. The central takeaway is simple but consequential: for large robot foundation models, policy improvement can be lifted from action space into language space. 

## **Limitations** 

Our results show that language-conditioning space optimization can improve frozen VLA policies, but the scope of the evidence is still limited. The method also cannot repair all failures of a frozen action policy. Language-space optimization should be viewed as an adaptation layer over existing VLA capabilities rather than as a replacement for improving perception, robustness, control, or long-horizon reasoning. 

## **Ethical Considerations** 

This work studies language-conditioned robot control in benchmark settings and does not involve human-subject data. 

## **Artifact Licenses** 

We use publicly available benchmark and model artifacts, including VL-Think, RL4VLA, OpenVLA, _π_ 0, and Qwen, under their respective licenses and terms of use. Any released code or checkpoints will be distributed with an explicit license. Our use of these artifacts is limited to research evaluation. 

## **References** 

- Lakshya A Agrawal, Shangyin Tan, Dilara Soylu, Noah Ziems, Rishi Khare, Krista Opsahl-Ong, Arnav Singhvi, Herumb Shandilya, Michael J Ryan, Meng Jiang, and 1 others. 2025. Gepa: Reflective prompt evolution can outperform reinforcement learning. _arXiv preprint arXiv:2507.19457_ . 

- Kevin Black and 1 others. 2024. _π_ 0: A vision-languageaction flow model for general robot control. _arXiv preprint arXiv:2410.24164_ . 

- Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence, Chuyuan Fu, Montse Gonzalez Arenas, Keerthana Gopalakrishnan, Kehang Han, Karol Hausman, Alexander Herzog, Jasmine Hsu, Brian Ichter, and 35 others. 2023. Rt-2: Vision-languageaction models transfer web knowledge to robotic control. _Preprint_ , arXiv:2307.15818. 

- Kang Chen, Zhihao Liu, Tonghe Zhang, Zhen Guo, Si Xu, Hao Lin, Hongzhi Zang, Xiang Li, Quanlu Zhang, Zhaofei Yu, Guoliang Fan, Tiejun Huang, Yu Wang, and Chao Yu. 2026. _π_ RL: Online rl finetuning for flow-based vision-language-action models. _Preprint_ , arXiv:2510.25889. 

- Embodiment Collaboration, Abby O’Neill, Abdul Rehman, Abhinav Gupta, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, Ajinkya Jain, Albert Tung, Alex Bewley, Alex Herzog, Alex Irpan, Alexander Khazatsky, Anant Rai, Anchit Gupta, and 275 others. 2025. Open x- embodiment: Robotic learning datasets and rt-x models. _Preprint_ , arXiv:2310.08864. 

- Bear Häon, Kaylene Stocking, Ian Chuang, and Claire Tomlin. 2025. Mechanistic interpretability for steering vision-language-action models. _Preprint_ , arXiv:2509.00328. 

- Physical Intelligence, Ali Amin, Raichelle Aniceto, Ashwin Balakrishna, Kevin Black, Ken Conley, Grace 

Connors, James Darpinian, Karan Dhabalia, Jared DiCarlo, Danny Driess, Michael Equi, Adnan Esmail, Yunhao Fang, Chelsea Finn, Catherine Glossop, Thomas Godden, Ivan Goryachev, Lachy Groom, and 37 others. 2025a. _π_ 0<sup>_∗_</sup> _._ 6<sup>:a vla that learns from experi-</sup> ence. _Preprint_ , arXiv:2511.14759. 

- Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Manuel Y. Galliker, Dibya Ghosh, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Devin LeBlanc, and 17 others. 2025b. _π_ 0 _._ 5: a vision-language-action model with open-world generalization. _Preprint_ , arXiv:2504.16054. 

- Nikita Kachaev, Mikhail Kolosov, Daniil Zelezetsky, Alexey K Kovalev, and Aleksandr I Panov. 2025. Don’t blind your vla: Aligning visual representations for ood generalization. _arXiv preprint arXiv:2510.25616_ . 

- Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. 2024. Openvla: An open-source vision-language-action model. _Preprint_ , arXiv:2406.09246. 

- W. Kong and 1 others. 2024. Prewrite: Prompt rewriting with reinforcement learning. In _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)_ , pages 594–601. 

- M. Kwon and 1 others. 2024. Stableprompt: Automatic prompt tuning using reinforcement learning for large language model. In _Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP)_ . 

- D. Lee, S. Song, and 1 others. 2023. Read-only prompt optimization for vision-language few-shot learning. In _Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)_ . 

- J. Liu and 1 others. 2025. What can rl bring to vla generalization? an empirical study. _arXiv preprint arXiv:2505.19789_ . 

- Daria Pugacheva, Andrey Moskalenko, Denis Shepelev, Andrey Kuznetsov, Vlad Shakhuro, and Elena Tutubalina. 2025. Bring the apple, not the sofa: Impact of irrelevant context in embodied ai commands on vla models. _Preprint_ , arXiv:2510.07067. 

- John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. _Preprint_ , arXiv:1707.06347. 

- Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan 

Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. _Preprint_ , arXiv:2402.03300. 

- Homer Walke, Kevin Black, Abraham Lee, Moo Jin Kim, Max Du, Chongyi Zheng, Tony Zhao, Philippe Hansen-Estruch, Quan Vuong, Andre He, Vivek Myers, Kuan Fang, Chelsea Finn, and Sergey Levine. 2024. Bridgedata v2: A dataset for robot learning at scale. _Preprint_ , arXiv:2308.12952. 

- Selma Wanna, Agnes Luhtaru, Jonathan Salfity, Ryan Barron, Juston Moore, Cynthia Matuszek, and Mitch Pryor. 2026. Limited linguistic diversity in embodied ai datasets. _Preprint_ , arXiv:2601.03136. 

- Mert Yuksekgonul, Federico Bianchi, Joseph Boen, Sheng Liu, Zhi Huang, Carlos Guestrin, and James Zou. 2024. Textgrad: Automatic" differentiation" via text. _arXiv preprint arXiv:2406.07496_ . 

- Michał Zawalski, William Chen, Karl Pertsch, Oier Mees, Chelsea Finn, and Sergey Levine. 2025. Robotic control via embodied chain-of-thought reasoning. _Preprint_ , arXiv:2407.08693. 

- Zhihao Zhan, Yuhao Chen, Jiaying Zhou, Qinhan Lyu, Hao Liu, Keze Wang, Liang Lin, and Guangrun Wang. 2026. Stable language guidance for vision-language-action models. _Preprint_ , arXiv:2601.04052. 

- Y. Zhang, Y. Qi, and X. Zheng. 2025. Experiences from benchmarking vision-language-action models for robotic manipulation. _arXiv preprint arXiv:2511.11298_ . 

- Tony Z. Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. 2023. Learning fine-grained bimanual manipulation with low-cost hardware. _Preprint_ , arXiv:2304.13705. 

- Y. Zhong and 1 others. 2025. A survey on visionlanguage-action models: An action tokenization perspective. _arXiv preprint arXiv:2507.01925_ . 

- Y. Zhou and 1 others. 2024. Archer: Training language model agents via hierarchical multi-turn rl. In _International Conference on Machine Learning (ICML)_ . 

- Viktoriia Zinkovich, Anton Antonov, Andrei Spiridonov, Denis Shepelev, Andrey Moskalenko, Daria Pugacheva, Elena Tutubalina, Andrey Kuznetsov, and Vlad Shakhuro. 2025. Sparta: Evaluating reasoning segmentation robustness through black-box adversarial paraphrasing in text autoencoder latent space. _Preprint_ , arXiv:2510.24446. 

## **A Appendix Overview** 

This appendix collects the implementation and qualitative material that supports the main paper. Appendix B shows the prompt format used by the language-conditioning space policy and clarifies the output contract: the model may reason about the scene internally, but only the final VLA-grounded command is passed to the frozen action policy. Appendix **??** provides a trajectory-level example comparing the original instruction with an optimized command. Appendix C summarizes the GRPO training procedure and the black-box optimization loop, and Appendix D reports the recovered model, adapter, and optimization parameters used in the archived runs. 

## **B Prompt Format Example** 

Figure 5 illustrates the prompt format used to transform a human instruction into a VLA-grounded command. The input consists of the current scene image, the original task instruction, and a structured instruction that asks the language model to identify the source object, target, and visible grounding cues. The policy is allowed to use this reasoning to choose the command, but the downstream VLA receives only the final text inside the <answer> field. This separation prevents long explanations from entering the action model while preserving the benefit of scene-aware language reasoning. 

## **C Training Algorithm and Optimization Loop** 

The training procedure treats command generation as the only learned component. For each imageinstruction pair, the language-conditioning space policy samples a group of candidate commands. Each candidate is executed by the same frozen VLA policy, and the resulting sparse rollout rewards are compared within the group. GRPO then updates only the language policy, leaving the action model unchanged. 

**Algorithm 1: VLA Grounder Training** 

**Input:** training distribution _D_ over image-instruction pairs ( _o, u_ ), frozen VLA policy _π_ VLA, language-conditioning space policy _qϕ_ , GRPO group size _G_ . **For each training instance:** 

1. Sample a GRPO group of commands _{u_ ˜<sup>(</sup><sup>_k_)</sup> _}_<sup>_G_</sup> _k_ =1<sup>from</sup><sup>_qϕ_(</sup><sup>_· | o, u_).</sup> 

2. For each _u_ ˜<sup>(</sup><sup>_k_)</sup> , run a rollout with the same frozen VLA policy, keep the command fixed for the rollout, and collect scalar return _R_<sup>(</sup><sup>_k_)</sup> . 

3. Compute normalized within-group advantages from _{R_<sup>(</sup><sup>_k_)</sup> _}k_<sup>_G_</sup> 

   - _k_ =1<sup>.</sup> 

4. Apply a GRPO update to _qϕ_ using the grouped commands and advantages; keep _π_ VLA unchanged. 

**Output:** trained language-conditioning space policy _qϕ_ . 

## **D Training Parameters** 

This section reports the implementation details recovered from the archived training artifacts (Table 8, Table 7) . The interface model is adapted with LoRA while the downstream VLA policy is kept frozen. The two archived VL-Think runs share the same model and optimization configuration, differing only in the downstream frozen backbone and output directory. 

|Field|Recovered value|
|---|---|
|Base language model|Qwen/Qwen3.5-9B|
|PEFT method|LoRA|
|LoRA rank_r_|32|
|LoRA scaling_α_|64|
|LoRA dropout|0_._05|
|Train-time precision flag|bf16 = true|
|Target modules|12 projection modules spanning attention and MLP blocks|
|Tokenizer context length|model_max_length = 262144|
|Tokenizer padding / truncation|left padding, left truncation|



Table 7: Language-conditioning space policy configuration 

|Field|_π_0/VL-Think archive|OpenVLA/VL-Think archive|
|---|---|---|
|Gradient accumulation steps|8|8|
|Max training steps|1600|1600|
|Learning rate|5_×_10<sup>_−_6</sup>|5_×_10<sup>_−_6</sup>|
|Scheduler|cosine|cosine|
|Storedwarmup_stepsfield|0_._05|0_._05|
|Weight decay|0_._01|0_._01|
|Adam_β_1,_β_2|0_._9,0_._999|0_._9,0_._999|
|Adam_ϵ_|10<sup>_−_8</sup>|10<sup>_−_8</sup>|
|Max grad norm|1_._0|1_._0|
|Per-device eval batch size|8|8|
|Save strategy / save steps|steps /500|steps /500|



Table 8: GRPO optimization configuration. 



Figure 5: Prompt-format example for VLA Grounder. The language policy receives the scene image and original task, identifies visual cues for the source object and target, and emits one concise VLA-grounded command. Only the command inside the <answer> tags is passed to the frozen VLA policy; intermediate reasoning is used only to select a better command. 



<!-- Start of picture text -->
Language-<br>Inputs Conditioning Space Frozen VLA Policy<br>Scene image  o → Policy → Execute rollout with<br>Human instruction  u Sample one optimized π VLA u ˜<br>command u ˜  ∈U ( u )<br>Grouped<br>Environment<br>Comparison GRPO Update<br>Benchmark-provided → Compare rewards → Update only  qϕ<br>scalar<br>across during training<br>rollout reward  R<br>candidate commands<br><!-- End of picture text -->

Figure 6: VLA Grounder language-conditioning space optimization loop. The image and user instruction define a context in which the language-conditioning space policy selects a command. The frozen VLA policy then executes a rollout under the command-conditioned action model _π_ VLA<sup>_u_˜.During training, grouped reward comparisons update</sup> only the language-conditioning space policy; during inference, the same composition is used without reward queries or parameter updates. 



Figure 7: Language aliasing in command space. Naive rewriting often produces commands that differ only in surface form, so reward comparisons are weak because the frozen action policy behaves similarly under each candidate. VLA Grounder instead biases exploration toward short executable commands that vary along behaviorally meaningful axes such as source-object description, target cues, visible attributes, and spatial relations. 

