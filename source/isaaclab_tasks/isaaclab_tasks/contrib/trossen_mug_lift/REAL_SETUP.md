# Real-world setup: Trossen Stationary AI mug tasks

Every placement is defined relative to ONE physical landmark — the LEFT arm's
base-plate center at tabletop level — so the identical setup is reproduced on
the rig with a tape measure. "Forward" is away from the left arm along the
rig's centerline; "left/right" are from the operator standing behind the left
arm facing forward. Sim trains with ZERO spawn jitter: the experiment is
lift-from-THE-spot / slide-from-THE-spot, and the rig must match.

## Mug placement (shared by BOTH tasks)

1. Mark the table's lengthwise MIDPOINT on the centerline — halfway between
   the two short edges (equivalently 45.75 cm forward of the base-plate
   center), under the table camera.
2. Place the mug's bottom-center on the mark, handle pointing toward the
   left arm's base plate (sim spawns the handle at +90° yaw toward the rig).

Sim reference (env frame, meters): mug root at (-0.020, 0.000, 0.021).

## Slide task (mug A -> B without tipping)

- A is the mug placement above.
- B is at the FAR END of the push line: on the centerline, **one mug-base
  radius (4 cm) before the rail at the table's far edge** — a ~57 cm slide.
  Beyond arm reach by design: the policy pushes and releases, friction parks
  the mug. In sim the commanded goal is fixed and rendered as the table
  cross in every clip.
- Success: mug upright at B, at rest. Tipping or leaving the table is a
  failed trial.

## Lift task (pick and carry to a fixed point)

- Same mug placement.
- The carry target is a single fixed point commanded ~25 cm above the table;
  its exact rig-frame projection gets taped during the eval session from the
  sim goal marker.

## Evaluation protocol

- Policies are evaluated from the HOME pose only (the -Play task variants pin
  every episode to home starts; no reset banks in eval).
- Film every trial; success judgments come from the video plus the logged
  metrics, not the reward.
