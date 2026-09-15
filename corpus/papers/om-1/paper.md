# OM-1: Frontier Robot Intelligence, Learned Firsthand from Humans

Reward AI Team, "OM-1: Frontier Robot Intelligence, Learned Firsthand from Humans", Reward AI Blog, Sep 2026. https://www.rewardai.com/blog/OM-1/

Source: company blog post (not an arXiv paper). No PDF/technical report is published; this is a full-text capture of the blog for corpus search purposes.

## Introduction

People handle the physical world with an ease that hides how hard it is: reaching, grasping, adjusting on contact, mostly without thinking about it. That ease sets the bar. To be genuinely useful alongside people, a robot has to manipulate the world with the speed, fluency, and efficiency of a person, whatever body it happens to have. That level of capability will not come from adding more data or more compute, nor from running today's systems faster: more is different (Philip W. Anderson, "More Is Different: Broken symmetry and the nature of the hierarchical structure of science," Science, 1972). As robots approach human-level efficiency, perception, decision-making, and control must all keep pace, and design choices made for slower, body-specific systems begin to break down. Human-level manipulation therefore requires rethinking how robot intelligence is captured, learned, and deployed.

At Reward AI, the Omnibody stack is built around a single principle: One Model, One Data Interface, Any Body. Omnibody brings the whole pipeline into one system. It captures manipulation behavior at the pace people actually work, learns a general-purpose robot policy from that behavior, and runs the same policy across robot bodies, from industrial arms to humanoids. Data, learning, and control are designed together rather than bolted together, starting with the hand as a shared interface for learning manipulation across bodies.

## Omnibody Hand

The Omnibody stack starts with human manipulation itself. Building on prior work DexCap (Chen Wang et al., "DexCap: Scalable and Portable Mocap Data Collection System for Dexterous Manipulation," RSS, 2024) on scalable, portable motion capture for dexterous manipulation, Omnibody Hand is a wearable device that lets people demonstrate manipulation naturally, without bending their behavior to the kinematics of any particular robot.

Preserving natural manipulation calls for functional dexterity in the behaviors that actually matter. Parallel-jaw grippers can solve many tasks, but usually by constraining approach direction, contact location, and manipulation strategy. Omnibody Hand is designed around the functions that matter: choosing useful contact points, reorienting objects in the hand, and moving smoothly between precision and power grasps. The result is a compact seven-degree-of-freedom design rather than a joint-by-joint copy of the human hand. It captures thumb-index pinching along with flexion of the thumb and index finger, so the thumb can work with the other fingers as well as with the index. For power grasps, the middle, ring, and little fingers move together at the metacarpophalangeal (MCP) joints, with thumb flexion that lets them close around an object.

Ergonomics matters just as much. Poor fit, mechanical constraint, or slippage changes how someone grasps an object and forces them to compensate for the device — a compensated grasp is no longer the behavior meant to be recorded. Omnibody Hand accommodates differences in hand size and finger proportion while following the natural bending and closing motion of the fingers. An integrated distal flexion mechanism absorbs differences in finger length, reducing sensitivity to exact joint alignment and removing the need for per-user link adjustment.

## One Data Interface

Omnibody Hand lets people move naturally. One Data Interface, the next layer of the stack, turns that movement into complete, usable data, pairing sensing of the interaction around the hand with tracking of where the hand travels and how hard it pushes and pulls to get there. It asks nothing of the wearer: working, playing, cooking, or simply going about the day all produce data the model can learn from, with no staged setup and nobody supervising the collection.

Consider a conveyor-belt sorting task. A person can spot an incoming object, reach for it, establish contact, and toss it into a bin in a fraction of a second, and at that pace missing even a brief moment of the interaction makes the recording far less useful. High-frequency tactile feedback, proximity sensing for the distance before contact, and global-shutter in-hand cameras that hold visual context through rapid motion are combined to cover the interaction from approach, through contact, into a stable grasp. This continuous coverage keeps the yield of usable data high even at full human pace and makes sorting at human proficiency learnable at all.

Hand pose is tracked carefully as well. Visual-inertial tracking is the common default, but the visual update rate it localizes against limits how well it follows rapid reversals, and the usual remedies either smooth the trajectory after the fact or ask people to slow down, which moves the data away from the quick reaches and delicate corrections meant to be captured. Visual-inertial tracking is therefore augmented with electromagnetic sensing, which yields a high-fidelity positional signal that does not depend on visual update timing, with a tracking algorithm that compensates for environmental electromagnetic disturbance. To measure the difference, both trackers were rigidly mounted to one structure and moved between two mechanical stops a known distance apart at eight speeds, so that any motion beyond that fixed range is overshoot. Averaged over ten runs per speed, this approach reduced mean overshoot error by 60% at high speed (24.9mm to 9.5mm at the highest speed tested, with narrowing run-to-run spread). Force is recorded along the same trajectory, so a demonstration carries not only the path a person took but the effort it took to follow it: the pull that opens a stuck door, the load that comes with a heavier box.

## Omnibody Model 1 (OM-1)

With natural human behavior captured through One Data Interface, the Omnibody stack learns from it through One Model — OM-1, the in-house general-purpose robot policy and decision-making core of the system. One policy learns from robot-free human data collected while people wear Omnibody Hand, and runs across robot bodies, from industrial arms to humanoids. It supports a wide range of contact-rich tasks, and gets better as the scale and diversity of human data grow.

OM-1 learns from human data alone. No teleoperation and no on-robot experience: neither goes into OM-1, and neither is needed. Rather than routing human behavior through an intermediate robot, OM-1 learns to generate robot actions directly from human motion.

Training follows the same logic: OM-1 has no barrier between pre-training and post-training, because every demonstration arrives through One Data Interface in the same form — the first demonstration ever recorded and the newest one train a single policy in a single stage. Nothing has to be re-collected for a new robot, and nothing has to be set aside as the wrong kind of data. Scaling OM-1 is a matter of adding human data, not of designing a separate training stage for every robot, task, or deployment.

Human-level efficiency also guides the model's design. Fast robots need fast decisions, so a novel architecture is built for efficient inference: OM-1 takes in a rich multimodal history and still produces actions quickly enough to keep a robot moving at the pace of the person it learned from.

OM-1 consumes the multimodal sensory streams collected with Omnibody Hand: images, tactile signals, inter-finger proximity, and hand pose trajectories, and produces human-speed actions for different robot bodies. Each modality carries a different part of the interaction — vision supplies the broader scene context, while proximity and tactile signals carry the approach to contact and the contact itself. Together they let OM-1 reason about events that vision alone reveals only faintly, such as when to initiate a grasp or whether an object is securely held.

### Reactivity / inference-speed design (key architectural details)

- **Native per-modality sampling rate.** OM-1 processes each modality at the native sampling rate of the sensor that produces it, rather than reducing every stream to a common frequency, so high-frequency tactile and motion cues survive alongside lower-frequency visual context. It also consumes a temporal history of these streams, letting it reason about how contact, motion, and task progress evolve over time.
- Actions output by OM-1 carry motion direction, speed, force, and the timing of key events such as grasping and moving.

## Control Any Body

One Model only reaches Any Body if every robot can faithfully execute the actions it produces. OM-1's control layer closes that gap: running underneath at high frequency, it turns those actions (covering manipulation as well as navigation for mobile robots) into actuation and absorbs whatever is particular about the machine at hand. The control layer is trained with reinforcement learning in simulation to account for velocity- and acceleration-dependent dynamics, external disturbances, and system delays. Where a classical controller is pushed off its reference by an unexpected load and never recovers, OM-1 holds the reference through the disturbance and settles back onto it — e.g. pulling open a fully closed refrigerator door without knowing how much the door will resist, or lifting delivery boxes of varying weight.

### Reactivity / inference-speed design (control layer)

- **Asynchronous control layer with its own clock.** Execution cannot pause while the policy computes the next actions. OM-1's control layer runs on its own clock, continuing at high frequency while the policy generates new actions, so variation in inference latency never interrupts robot motion.
- **Online blending between successive predictions.** Running on its own clock brings a challenge: successive actions may not join smoothly when a new prediction arrives, and a discontinuity that is barely noticeable at low speed degrades execution at high speed. The control layer optimizes the transition between successive predictions online, keeping motion smooth and continuous through dynamic behaviors such as tossing and swinging.

## Conclusion

OM-1 learns manipulation directly from natural human behavior, without teleoperation or on-robot training data. Omnibody Hand and One Data Interface capture people working at their own pace; OM-1 learns from those demonstrations to perform tasks across robot bodies, from industrial arms to humanoids. OM-1 picks up a brand-new task, including challenging dynamics and long horizons, from less than 30 minutes of data. This performance comes not only from the policy architecture itself, but from integrating human demonstration capture, sensing, learning, inference, and control into one system that generalizes across robot bodies, tasks, and environments.

## Citation

```bibtex
@article{rewardai2026om1,
  author = {Reward AI Team},
  title = {OM-1: Frontier Robot Intelligence, Learned Firsthand from Humans},
  journal = {Reward AI Blog},
  year = {2026},
  note = {https://rewardai.com/blog/OM-1/}
}
```
