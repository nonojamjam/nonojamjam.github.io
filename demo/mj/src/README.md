# Panda gripper demo — how the model file is built

- `pickplace.xml`: scene (table, red target box, wider look-alike, green goal zone) that includes Menagerie `mjx_panda.xml` (Franka Panda with hand, Apache-2.0, google-deepmind/mujoco_menagerie).
- `decimate.py`: merges duplicate OBJ vertices, then reduces visual meshes by 80 % (collision meshes unchanged) → `panda_lite/`.
- `build_mjb.mjs panda_lite`: compiles the scene with the official `@mujoco/mujoco` 3.15 WASM build and saves `../pickplace.mjb`.
- `run_pp.mjs`: offline check — runs the same controller (`../pickplace_ctrl.mjs`) five times and reports whether the box ends in the goal zone (5/5 on 2026-10-06; 12/12 at a 4 cm offset).

- `run_sc.mjs`: runs the seven scenarios (same visits, camera noise and planner mistakes for every setting) and prints the comparison table shown on the page.
- Boxes: target 32 mm, look-alike 40 mm (a 52 mm look-alike was pushed by the hand during descent, so the touch check could not read it).
- 2026-10-06 (late): the engine decides **step by step** as in the kinematic demo — up to 12 steps per visit, each from memory (keyed by direction, distance and joint bands) or from the stand-in planner, checked by the gate (progress, joint limits, jerk, hand pointing down; up to 3 regenerations, else skipped); failed memory steps are deleted. Results for random draw #7 and #11/#23 are stable (e.g. no gate 12–13/18 with ~20 unsafe steps, gate 18/18 with 0).
