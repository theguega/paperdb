# Introducing S1: In-Context Learning for Robotics

Skild AI, "Introducing S1: In-Context Learning for Robotics", Skild AI Blog, August 2026. https://www.skild.ai/blogs/s1

Source: company blog post (not an arXiv paper). No PDF/technical report is published (future posts promised on training details); this is a full-text capture of the blog for corpus search purposes.

Tags: UNSEEN TASKS, 10-MINUTE HORIZONS, ONE VIDEO PROMPT, NO POST-TRAINING.

## Introduction

The evolution of language modeling provides a blueprint for turning capable models into useful general-purpose tools. Earlier transformer approaches such as BERT (Devlin et al., 2019) proved effective at understanding language, but each new application still demanded additional data collection and fine-tuning. The pivotal transition from such early language models to ChatGPT was driven by in-context learning (ICL) — prompting: a user can introduce a novel concept entirely through the prompt, and the model can produce reasonable responses without fine-tuning of the model weights.

Robotics thus far has been stuck in the BERT era: robustly executing a new task still requires collecting hours of data and fine-tuning a specialist policy. To learn robust policies for moderately complex tasks, post-training data must reach tens to hundreds of hours in deployment conditions. Research has shown that when post-training data is sufficiently large and dense, even policies trained from scratch with no pre-training can match the peak performance of a post-trained foundation model (Oh et al., 2026) — raising the question of what pre-training is even for.

The belief driving S1: the main purpose of pre-training is to enable, for robotics, the same shift seen from BERT to GPT-3 (Brown et al., 2020): in-context learning, the ability to learn immediately from one or a few examples. A year prior, Skild released an in-context learner for locomotion (LocoFormer, Liu et al., 2025) that adapts by accumulating live experience in its prompt. S1 is the flagship robotic foundation model for manipulation, built from the ground up as an in-context learner: show it a video of a task, short or long, seen or unseen, and it executes.

## Why In-Context Learning for Robotics?

TL;DR — Tasks are specified by a video demonstration, not language. Pre-training across diverse tasks forces the model to learn intent from demonstration, yielding one set of weights that executes unseen tasks without fine-tuning.

The goal is the same as for language models: show the task, and the robot should execute, even if it has never seen the task. Simple, atomic actions ("hand me the mug") need no elaboration via language, but once a task becomes delicate, nuanced, or long-horizon, people stop describing and start showing (folding a fitted sheet, whisking egg whites to stiff peaks, tying a knot). Community wisdom (Jim Fan, quoted): "Once you have enough data, many behaviors can actually be zero-shot... Whether in-context learning truly works or not also depends on how far away the test is from training."

ICL should therefore be understood along two axes: (1) whether the demonstrated tasks are seen (in-distribution) or unseen (out-of-distribution) during pre-training, and (2) whether tasks are short-horizon/atomic (5-25s) or long-horizon, requiring new skills to be composed. Concurrent approaches to ICL in manipulation (Generalist AI, 2026; Jiang et al., 2026 — RoboTTT) largely cover short-horizon or already-in-distribution tasks. S1 is claimed to be the first robotics foundation model to show ICL on extremely long-horizon tasks (up to 10 minutes) never seen during pre-training.

## S1: An In-Context Learner

TL;DR — S1 translates an in-context video demonstration of the desired task to robot actions.

The training recipe: pre-train on episodic data where the task is specified only through an in-context demonstration. Since the demonstration may come from a different scene, viewpoint, or embodiment, the policy must implicitly learn the demonstrator's intent, functional correspondences, and task progress to predict the appropriate actions. S1 is built on NVIDIA AI infrastructure for accelerated training at scale.

Task diversity and scale drive emergence of ICL capability. In the high-diversity regime, scene ambiguity must be resolved by attending to context, incentivizing the model to learn how to learn from demonstrations (in meta-learning terms, pre-training is the outer loop teaching the policy how to learn from context; at inference time the demonstration drives the inner loop without changing weights). The result is a single policy that performs both familiar behaviors in novel configurations and previously unseen behaviors through ICL, with no fine-tuning or post-training — the same model weights produce every example in the post.

### What does S1 learn from context?

Two axes of difficulty: skill novelty and task horizon. Via a video demonstration prompt, S1 can (1) perform new atomic skills not present during pretraining, and (2) solve long-horizon tasks over 10 minutes long by composing atomic skills in previously unseen ways.

- **Learning out-of-distribution skills.** Robotics data remains scarce and expensive, leaving many tasks outside the pre-training distribution. Rich video context from ICL pre-training encourages an emergent implicit mapping from demonstration to actions, enabling test-time learning of novel behaviors.
- **Composing across long-horizon tasks.** Executing a 10-minute task from context requires chaining new sequences of manipulation primitives, tracking progress, and recovering from mistakes — abilities naturally aligned with well-configured ICL pre-training.

## The Data Engine

TL;DR — No single data source wins on scalability, diversity, and hardware proximity, so all of them are scaled and combined strategically.

Every major robotics data source trades off across three axes:
- **Hardware proximity** (how closely data resembles deployment hardware)
- **Diversity** (how many tasks, scenes, behaviors covered)
- **Scalability** (cost of collecting more)

Teleoperation: high proximity, low diversity, low scalability. UMI: moderate on all three. Egocentric video: low proximity, high diversity, high scalability. Simulation: moderate proximity, low diversity, high scalability. Nothing wins on all three, so Skild scales all of them in-house.

## S1 In Action

TL;DR — S1 performs tasks that run up to ten minutes and do not appear in the training data. ICL scaling gains over language prompting are moderate for seen tasks, but unseen tasks see a 7x performance improvement.

**Seen tasks.** A broad base of manipulation pre-training tasks grants a wide array of capabilities that S1 can later compose to perform unseen tasks.

**Long-horizon unseen tasks.** Four tasks S1 was never trained on: plant potting, pancake cooking, pour-over coffee making, kit assembly. Tasks run up to ten minutes, span dozens of manipulation steps, driven by a single visual demonstration, with strong compositionality and novel test-time behaviors (digging into soil, pressing a coffee filter into a funnel, flipping a pancake). For the plant-potting task, time from a single 6-minute human demonstration to autonomous execution on hardware was 11 minutes total (setup + one demo + execution).

### ICL scaling laws

Controlled study comparing ICL-style demonstration prompting with language prompting (conventional VLA), same data/architecture (besides prompt embedding)/compute, trained on 1k-100k hours.

- **Seen tasks (in-distribution).** At 1k hours, language-conditioned policy: 53% vs ICL: 43%. But ICL overtakes as pretraining scales: at 100k hours, ICL reaches 96% vs language-prompted VLA's 89%.
- **Unseen tasks (out-of-distribution).** Language prompting: 9% success at 100k hours. ICL: 66% success at 100k hours — the gap widens exponentially with more pre-training data. Attributed to (1) novel skills — ICL relies on visual correspondences that transfer better than language grounding for actions never described in training, and (2) compositionality — a demonstration spells out a novel chaining of primitives directly, where language is often too coarse.

### Emergent Properties

TL;DR — S1 withstands perturbations, exhibits common-sense reasoning, recovers from errors, and occasionally improves upon the demonstration. Single-shot ICL performance is comparable to roughly 380 post-training episodes.

- **Robustness to perturbations.** Sliding objects away mid-execution, swapping objects, changing lighting — none shown in the prompt — S1 completed the task regardless.
- **Mistake recovery.** S1 often retries rather than blindly proceeding, observed off the shelf even for out-of-distribution tasks (e.g. assembling a skateboard wheel).
- **Common-sense behavior.** E.g. prompt waters a plant with a watering can but only a cup of water is available — S1 uses the cup instead; prompt fills a glass with juice but the glass is already nearly full — S1 just tops it off.
- **Demonstration correction.** S1 sometimes improves upon flawed demonstrations (e.g. demonstrator drops an egg prematurely and makes a mess; S1 performs the step with controlled motion) — treating the demonstration as a specification of the goal, not a trajectory to reproduce.

### Quantifying ICL robustness to distribution shifts

Two axes: distance from training conditions, and distance from the in-context demonstration.

- **Distance from training conditions (L1-L5):** L2/L3 perturb object poses with increasing magnitude, L4 substitutes objects of matched affordance, L5 forces half the actions onto the opposite arm. Under L5, language-prompted VLA degrades up to 3x as much as ICL (VLA: 89→33% i.e. -56pp; ICL: 96→75% i.e. -21pp from L1 to L5, per the reported deltas).
- **Distance from the in-context demonstration (L1-L5):** ICL is robust to mismatches in object positioning and even substituted objects (L4: 75%); degrades significantly once the demonstration implies a substantially different execution plan (L5: 46%, opposite-arm switch).

### ICL demonstration efficiency

A single in-context demonstration for an unseen task achieves 66% success. Post-training a VLA policy to match that requires roughly 380 post-training episodes (interpolated crossing point) — for long-horizon tasks (4-10 min), collecting 380 demos took 50-100 hours of teleoperation. With 2,000 demonstrations the post-trained VLA eventually reaches 86%, surpassing the single-shot ICL number, but the gap is expected to shrink as ICL pre-training/post-training scale further.

## Closing Remarks

TL;DR — ICL's rapid deployment and strong scaling laws fuel the data flywheel.

Demand for robots is rapidly expanding beyond controlled settings; if every change requires repeated data collection, fine-tuning, and validation cycles, robots will never keep pace with the environments they operate in. Robots should acquire new behaviors the way people do: by observing a single demonstration. Timeline: Sep 2025 LocoFormer → Feb 2026 first in-domain ICL results → May 2026 S1 flips first pancake → Aug 2026 S1 release.

## References

- Brown et al. "Language Models are Few-Shot Learners." NeurIPS, 2020.
- Devlin, Chang, Lee, Toutanova. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL, 2019.
- Generalist AI. "Embodied foundation models are one-shot learners." 2026.
- Jiang, Chebotar, Zheng, Hu, Ge, et al. "RoboTTT: Context Scaling for Robot Policies." arXiv:2607.15275, 2026.
- Liu, Pathak, Agarwal. "LocoFormer: Generalist Locomotion via Long-context Adaptation." arXiv:2509.23745, 2025.
- Oh, Liu, Tao, Han, Shaw, Funabashi, Salakhutdinov, Pathak. "FACTR 2: Learning External Force Sensing for Commodity Robot Arms Improves Policy Learning." arXiv:2606.12406, 2026.

## Citation

```bibtex
@article{skild2026s1,
  author = {Skild AI},
  title  = {Introducing S1: In-Context Learning for Robotics},
  year   = {2026},
  month  = {August},
  url    = {https://skild.ai/blogs/s1},
}
```
