# Outstanding work

## Grasp form-closure (binding constraint)
The YAM gripper's flat pads **eject** a free cylinder during a static squeeze
("watermelon-seed"): the object is wedged out instead of held.

- Confirmed NOT a force problem: gripper force-limit 5/10/20 N, softer contact
  (solref), friction 4, and condim 6 all fail — too little force = no grip, too
  much = eject.
- Aiming is already fixed (Stage 3 re-rig: pad midpoint on the wrist axis =
  grasp_site [0,0,0.1347]; hand-servo captures the object 3/3, lifts to ~0.18).
- Needs **form-closure pads**: concave / V-groove / taller conforming pad, and
  likely less gripper tilt (the orientation clamp still leaves ~10–16°).
- `test_yam_grasp_lift` is xfail(strict=False) documenting this; its
  compliant-contact (<50 N) + no-crush asserts still run.

Why it matters: per-task analysis of the MT50 run shows the ~15 tasks that
succeed are all non-grasping (press/reach/close/turn). Every grasp-and-hold
task (pick-place, shelf-place, basketball, bin-picking, peg-*, stick-*,
coffee-pull, …) sits at score ≈ −10..−18 (pure jerk penalty, zero progress).
Until the pads achieve form closure the run ceiling is ~30% success no matter
how long it trains. This is an open-ended gripper-reshaping sub-task.
