.PHONY: setup doctor test test-web-policy smoke viewer webcam

setup:
	uv sync --locked --extra dev

doctor:
	uv run --locked dtr doctor

# These local targets require the ignored tests/ directory.
test:
	uv run --locked --extra dev ruff check .
	uv run --locked --extra dev pytest -q

test-web-policy:
	node --test tests/frame-policy.test.cjs tests/goal-view.test.cjs

smoke:
	uv run --locked python scripts/verify_current_models.py --smoke

viewer:
	.venv/bin/tensorboard --logdir runs --host 127.0.0.1 --port 6006

webcam:
	uv run --locked python webcam_app.py --allow-unvalidated \
	  --goal-model models/v1.0.0/goal/student.float.tflite \
	  --balloon-model models/v1.0.0/balloon/student.int8.tflite
