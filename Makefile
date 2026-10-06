.PHONY: setup doctor test test-web-policy smoke viewer webcam

setup:
	uv sync --extra pretrained --extra dev --extra metal --locked

doctor:
	.venv/bin/dtr doctor

test:
	.venv/bin/ruff check src tests scripts
	.venv/bin/pytest -q

test-web-policy:
	node --test tests/frame-policy.test.cjs tests/goal-view.test.cjs

smoke:
	.venv/bin/python scripts/smoke.py

viewer:
	.venv/bin/tensorboard --logdir runs --host 127.0.0.1 --port 6006

webcam:
	.venv/bin/python webcam_app.py --allow-unvalidated
