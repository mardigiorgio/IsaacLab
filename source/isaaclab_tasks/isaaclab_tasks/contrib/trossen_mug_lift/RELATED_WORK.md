# Related-work links (citation gathering)

Collected from the research sessions of 2026-08-20. One line each on what
the reference supports in our write-up.

## Core method

- CENIC — convex error-controlled integration (Kurtz & Castro):
  https://arxiv.org/abs/2511.08771
  The adaptive integrator this project implements on the Newton backend.

## Pre-grasp and grasp discovery

- Pavlichenko & Behnke, *Deep RL of Dexterous Pre-grasp Manipulation*
  (CASE 2023): https://www.ais.uni-bonn.de/papers/CASE_2023_Pavlichenko.pdf
  Far/default initialization fails from exploration burden; pre-grasp
  initialization is the enabling factor. Basis for the reset bank.
- Extended journal version: https://arxiv.org/pdf/2307.16752
- DemoGrasp, *Universal Dexterous Grasping from a Single Demonstration*:
  https://arxiv.org/pdf/2509.22149
  One demonstration defines the approach gradient — basis for the
  joint-space pre-grasp pose-matching reward.
- Zhou et al., *Learning to Grasp the Ungraspable with Emergent Extrinsic
  Dexterity*: https://arxiv.org/pdf/2211.01500
- Adaptive motion planning for multi-fingered functional grasp:
  https://arxiv.org/pdf/2401.11977
- Grasp curriculum under sparse rewards (object-size/distance staging):
  https://doi.org/10.3390/fi17100437
- Task-decomposition reward design for pick-and-place:
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10296071/

## Curriculum / reverse curriculum

- Florensa et al., *Reverse Curriculum Generation for RL* (CoRL 2017):
  http://proceedings.mlr.press/v78/florensa17a.pdf
  (arXiv: https://arxiv.org/abs/1707.05300)
  Start-state curriculum growing outward from the goal — basis for the
  home-to-pregrasp interpolated bank with annealed alpha_min.
- Parallelized Reverse Curriculum Generation:
  https://arxiv.org/pdf/2108.02128
- Reverse-Forward Curriculum Learning (ICLR 2024):
  https://proceedings.iclr.cc/paper_files/paper/2024/file/cd062f8003e38f55dcb93df55b2683d6-Paper-Conference.pdf
- TRL: Discriminative Hints for Scalable Reverse Curriculum Learning:
  https://openreview.net/forum?id=rJssAZ-0-
- MaMiC: Macro and Micro Curriculum for Robotic RL:
  https://arxiv.org/pdf/1905.07193
- Reverse curriculum for a contact-rich mounting task (quadruped
  skateboard): https://arxiv.org/pdf/2505.06561

## Demonstrations, teacher-student, sim2real

- TAPG: Teacher-Augmented Policy Gradient with instance segmentation
  (teacher-student for grasping): https://arxiv.org/html/2403.10187v1
- Crossing the human-robot embodiment gap with sim2real RL from ONE human
  demonstration: https://arxiv.org/pdf/2504.12609
- PLANRL: motion planning + imitation to bootstrap RL:
  https://arxiv.org/pdf/2408.04054
- Domain randomization for pre-capture of moving targets (DR practice for
  approach-phase policies): https://arxiv.org/pdf/2406.06460
- Tactile-gated contact-force penalties (approach without displacement) —
  surveyed via: https://doi.org/10.3390/fi17100437

## Platform and tooling

- Isaac Lab framework paper: https://arxiv.org/html/2511.04831v1
- ALOHA 2 (the Stationary AI rig's lineage):
  https://arxiv.org/pdf/2405.02292
- Trossen arm MuJoCo models, Stationary AI pick-and-place and sim2real
  demos: https://github.com/TrossenRobotics/trossen_arm_mujoco
- Trossen AI Isaac integration tutorials:
  https://docs.trossenrobotics.com/trossen_arm/main/tutorials/trossen_ai_isaac.html
- rsl_rl (PPO + Distillation runners used for teacher-student):
  https://github.com/leggedrobotics/rsl_rl
