# GEN-1.5: Embodied Foundation Models are One-Shot Learners

Generalist Team, "GEN-1.5: Embodied Foundation Models are One-Shot Learners", Generalist AI Blog, August 19, 2026. https://generalistai.com/blog/gen-1.5

Source: company blog post (not an arXiv paper). No PDF/technical report is published; this is a full-text capture of the blog for corpus search purposes.

## Introduction

Humans perform new physical skills from only one or a few examples. GEN-1.5, Generalist's latest robot foundation model, exhibits the beginnings of that same ability: it can learn a new task in seconds, from a single example, without gradient updates or fine-tuning. It displays broad capabilities across one-shot and few-shot learning from demonstration, as well as zero-shot physical generalization. Tasks are simple and short-horizon, but this is claimed to be the first model for which one-shot and few-shot learning of physical skills have emerged at scale.

For language models, GPT-3 achieved ~45% average one-shot in-context accuracy without training, up to ~65% with few-shot (~100 examples) (Brown et al., 2020). In robotics, the analogous pursuit — learning a task from one or a few demonstrations — traces back to the 1954 Unimate patent's teach-by-guiding and MIT's 1970 Copy Demo, but has predominantly been considered out of reach across a broad, unrestricted range of tasks.

## Introducing GEN-1.5

GEN-1.5 is a large multimodal model that processes video input (30 seconds of memory, alongside other sensor, language, and proprioceptive inputs) and produces 100Hz action trajectories. Capabilities:

- **One-shot learning via in-context prompting.** Learns new tasks in seconds when prompted with 3-12 seconds of a single demonstration, no training required ("physical prompting" — sensorimotor examples in the context window).
- **Compositional generalization.** Given two different physical prompts in context, the model chains them into a single longer-horizon behavior.
- **Zero-shot sim-to-real transfer.** A demonstration recorded in simulation works as a physical prompt for a real-world task, even though pretraining contains no simulation data.
- **Human-to-robot imitation.** A person can demonstrate a task with their own hands, in view of the robot's cameras, and the model reproduces it with the robot's hands.
- **Few-shot adaptation via gradient descent.** Fine-tuned to a new task in 1-10 gradient steps on 1-5 minutes of data (~10-50 demonstrations).
- **Improvising new strategies and tool use.** Generalizes at the level of behavioral strategies — forming new trajectories to reach a goal, using unseen tools (brush, dustpan) to solve tasks demonstrated with other tools, working ambidextrously even when trained on one hand.

These capabilities emerge directly from pretraining on large amounts of physical interaction data, with no architectural changes to promote in-context learning, no meta-learning inner/outer loop, and no auxiliary objectives encouraging improvisation.

Across 10 diverse tasks: 59% (±10% std) average success with one-shot in-context prompting straight from the pretrained model; 83% (±9% std) with few-shot learning via 10 gradient steps on 5 minutes of data (~50 demonstrations) per task. In some cases in-context learning of a new task exceeds the performance of 1-5 gradient steps on the same demonstration data.

Per-task breakdown (10-step/5-min gradient adaptation vs. in-context 12s prompt):
- Retrieve money from purse: 83.3% vs 60.7%
- Fold and crease paper: 69.3% vs 50%
- Twist lid off glass jar: 94.5% vs 60%
- Stack two small cups: 75% vs 67%
- Sweep trash with brush: 99% vs 37.3%
- Open book cover: 82.7% vs 54.7%
- Brush cube into bowl: 71.2% vs 60.8%
- Flip phone upside down: 81% vs 78%
- Unzip pencil pouch: 86% vs 55.5%
- Remove vacuum pad: 86% vs 64%

## Scaling Pretraining for Robotics

Over the past two years, Generalist built a pretraining engine for scaling embodied foundation models from the ground up on physical experience. GEN-0 (announced ~9 months prior) showed predictable scaling laws; GEN-1 (5 months later) demonstrated post-training to mastery at 99%+ success on simple tasks and initial signs of improvisational intelligence. GEN-1.5's pretraining has been running continuously for over 8 months across 3 training phases, with next-action prediction error on held-out validation continuing to improve.

As training progressed, fine-tuning requirements dropped from hundreds, to tens, to eventually 1 gradient step on just one minute of data for new tasks — leading to the question of whether the model could learn purely in-context with zero gradient steps, which it can.

## One-Shot Learning In-Context

GEN-1.5 is prompted with a single demonstration inserted into its 30-second context window; the remainder holds rolling observations. Physical prompts are sensorimotor examples (sensor data + action trajectories), recorded as human data (handheld grippers) or robot rollouts. Once in context, the model performs the task immediately with no training steps. Average one-shot success: 59% across diverse tasks (zippers, jars, money from wallets, etc.) — modest but unexpected given no explicit ICL training. Skills learned in-context are more brittle than fine-tuned models but generalize to some perturbations, improvise, and recover from mistakes.

No explicit training for in-context learning was performed, and tasks tested were not engineered into pretraining data beforehand. Hypotheses for why ICL emerges: "burstiness"/Zipfian structure in physical observation-action distributions (by analogy to language ICL), or naturally repetitive cycles in physical work that the model learned to detect and extend. Pretraining used randomly sampled continuous spans from the data engine (homes, warehouses, factories) with no bespoke infrastructure for packing in-context examples — physical prompts introduce discontinuous time jumps never seen in training.

## Compositional Generalization with Physical Prompt Engineering

Placing demonstrations of two independently-recorded different tasks in context lets GEN-1.5 chain them into one continuous behavior, bridging the two with intermediate motions (repositioning, regrasping, error recovery) that appear in neither demonstration. This opens "physical prompt engineering": assembling a compound task from a small library of short, reusable physical prompts, analogous to chaining instructions in a language prompt.

## Zero-Shot Sim-to-Real Transfer with In-Context Learning

A physical prompt formed entirely from simulated experience (scripted policy, RL agent, or human teleoperating a simulated robot) can prompt the real robot — despite zero simulation data in GEN-1.5 pretraining (neither rendered video nor simulated dynamics). The prompted behavior generalizes to different hands and new object positions/sizes in the real scene, meaning some demonstrations no longer need to be collected physically.

## Human-to-Robot In-Context Learning

In some cases a human demonstrates a task with their own hands, observable through the robot's cameras, and the robot reproduces it immediately afterward — crossing the embodiment gap entirely via in-context transfer.

## Few Gradient Step Adaptation

GEN-1.5 adapts to new physical tasks in as few as 1-10 gradient steps (vs. tens of thousands typically needed for prior robot models), without explicit fast-adaptation machinery (e.g. no second-order MAML-style gradients). Framed as closer to test-time training in an extremely low-data regime. In one-step adaptation (sampling from one minute of data), success on a held-out task is 66.5%, improving with larger batch sizes and higher learning rates — achieved without hyperparameter tuning specific to adaptation. Ten adaptation steps change model weights on held-out tasks by less than 0.15%, suggesting fine-tuning reconfigures existing knowledge rather than building new representations.

## Physical Generalization

Fine-tuned models generalize beyond their demonstrations to new embodiments, object instances, environments, and fundamentally different manipulation strategies:

- **Novel tool use improvisation.** Fine-tuned on 5 minutes of demonstrations brushing a block into a bowl with a brush; the model improvised using a banana as a makeshift brush, and — for a dustpan (a larger strategic departure) — used the dustpan to lift and dump the block into the bowl. Neither fine-tuning nor pretraining data (to their knowledge) contains a dustpan used this way.
- **Ambidexterity, obstacle handling** (removing/replacing a paper covering the bowl not present in fine-tuning data), **removing obstructions** stuck on fingertips using the other hand, **bimanual coordination** for jar-lid twisting despite one-handed training data, **tendency to organize** (sorting blocks by color/category despite training only on single block placement), and **generalizing to new objects** (jar → cups/bottles/to-go containers never seen, requiring different bimanual grasp/twist strategies).

## Looking Ahead

GEN-1.5 was not explicitly built to be a one-shot learner — it emerged from continued investment in a large-scale pretraining data engine. Every trend measured (GEN-0 → GEN-1 → GEN-1.5) pointed toward more pretraining making adaptation faster, cheaper, and more general, with no observed asymptote yet. Past a certain pretraining threshold, the cost of adaptation becomes negligible — closer to "reminding the model of something it nearly knows" than task-specific training in the conventional sense.

## Citation

```bibtex
@article{generalist2026gen15,
author = {Generalist Team},
title = {GEN-1.5: Embodied Foundation Models are One-Shot Learners},
journal = {Generalist AI Blog},
year = {2026},
note = {https://generalistai.com/blog/gen-1.5},
}
```
