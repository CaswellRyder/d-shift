# Script guide

Run scripts from the repository root. Many research commands need local data or
artifacts that Git ignores, so a script being here does not mean its historical
inputs are in the repo.

| Task | Entry points |
| --- | --- |
| Verify selected release | `verify_release.py` |
| Assemble v1 from original artifacts | `package_v1.py` (create-only, refuses existing output) |
| Data acquisition/preparation | `download_dtr.py`, `prepare_roboflow_dtr.py` |
| Distill/export | `distill_pi_student.py`, `export_pi_float.py` |
| Student experiments | `run_student_research.py`, `refine_pi_student.py`, `mine_student_negatives.py` |
| Detector experiments | `train_goal_detector.py` (separate optional environment) |
| Pi timing | `pi_research_crops.py`, `pi_temporal_bench.py` |
| Two-method comparison | `competition_compare.py`, `competition_camera.py` (external pixel source required) |
| Runtime build | `tflite_rebuild/build.sh` |

Check `--help` and the relevant [documentation](../docs/README.md) before running
anything. Camera and Pi commands need explicit hardware setup, and you never need
to run them just to install or look around the repository. Never overwrite
original research outputs.
