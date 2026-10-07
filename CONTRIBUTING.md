# Contributing

Discuss substantial model, dataset, or runtime changes before implementation.
Use focused branches and pull requests with plain descriptive titles.

## Setup and checks

Use Python 3.11 and `uv sync --locked --extra dev` on a desktop host.
Run the README's Development checks. Tests use generated fixtures and must not
implicitly download datasets, use credentials, operate cameras, contact a Pi,
or issue actuator commands.

## Preserve evidence

- Never overwrite frozen weights, release assets, historical hashes, or tags.
- `scripts/verify_release.py` checks v1 weights and source-file identities.
  Changing bound files needs a separately designed new-version flow. Do not
  weaken verification or edit the v1 manifest to hide differences.
- Separate crop, full-frame, replay, live capture, and independent field results.
- Keep test sessions out of training and tuning; report regressions as well as gains.

## Data and ownership

Do not commit datasets, recordings, credentials, SSH keys, `.env` files, cached
teacher logits, or third-party pixel code. Use small generated fixtures.
Weights need provenance, class order, preprocessing, hashes, attribution, and
explicit validation status. Only contribute work you have the right to submit;
retain upstream notices and identify adaptations.

## Scope and reporting

D-SHIFT outputs observations. Mission planning and actuator safety belong in
downstream software. Inference and test paths must not implicitly command motors.
Bug reports should include hardware/OS, commit, command, expected/actual behavior,
and sanitized logs. Report vulnerabilities using SECURITY.md, not public details.
