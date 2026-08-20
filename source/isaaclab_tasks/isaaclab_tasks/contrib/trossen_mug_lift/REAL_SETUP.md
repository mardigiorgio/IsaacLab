# Real-world setup: Trossen spatula lift

Reproduces the simulation's evaluation pose (`...EnvCfg_PLAY`, zero spawn jitter) at the
physical Stationary AI rig with a tape measure. Training adds ±10 cm lateral / ±7.5 cm
forward jitter around this same nominal pose.

## Reference landmark

**The LEFT arm's base plate center, at tabletop level.** All measurements start here.
"Forward" means from the base plate toward the opposite arm; "left" is the operator's
left when standing behind the left arm facing forward.

## Spatula placement (nominal / evaluation pose)

1. Mark the point **33.0 cm forward** of the base plate center, **on the base plate's
   centerline** (lateral offset 0).
2. Lay the spatula **flat on the tabletop** with the **blade center** on that mark.
3. **Handle points to the operator's left**, blade edge facing the arm — the blade's
   7.0 cm width lies across the gripper's approach.

```
        [opposite arm]
              ^
              | forward
              |
   handle <===#####        # = blade, center on the mark
              |(33.0 cm)
              |
      [left arm base plate]
```

## Notes

- The gripper grasps the BLADE (7.0 cm wide), not the handle: the official model's
  closed finger gap is 4.83 cm, wider than the ~2.2 cm handle. See the task module
  docstring for the measured geometry.
- Reachability: this pose sits at the center of the band a trained Stationary AI cube
  policy demonstrably grasped across (June reach-map measurements; band corners
  0.22-0.39 m from the base plate, well inside the arm's measured extension).
- If the physical gripper's true closed gap is measured below ~2 cm (ruler check),
  handle grasp becomes possible and the task geometry should be revisited.
