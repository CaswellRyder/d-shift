# Security policy

D-SHIFT is research software. Public source access and version numbers are not
security or flight approval. No guaranteed security-response SLA is offered.

## Reporting

Use GitHub **Security → Report a vulnerability** if private reporting is enabled.
If unavailable, open an issue requesting a private reporting channel **without
exploit details, credentials, private images, or vulnerable endpoints**.

## Known dependency advisories

The frozen v1 environments have open Dependabot alerts involving Keras, Pillow,
PyTorch, setuptools, and pytest. They span `uv.lock`, `pyproject.toml`, and the
separate `requirements-detector.txt` environment. Counts may repeat an advisory
across manifests. Consult GitHub's Security tab for the current list.

These alerts have not been dismissed and exploitability has not been comprehensively
assessed. A tested dependency-upgrade path is still needed. Preserve v1's historical
environment while preparing a new supported environment. Recheck serialization,
exports, output parity, and applicable ARMv6 behavior when upgrading. Numerical
smoke tests do not prove dependency security.

## Operating boundaries

- Load only trusted model files. Hashes establish identity relative to a trusted
  manifest, not the safety of an arbitrary Keras/TensorFlow/PyTorch artifact.
- Keep the viewer on localhost. Do not deploy it as a public or multi-user service.
- Use local, reviewed media, not untrusted uploads.
- Keep camera access, credentials, and control hardware user-managed.
- Select the Pi runtime per process; do not overwrite system libraries.
- A separate controller must enforce watchdogs and safe handling of stale outputs.

Public-release preparation requires a documented dependency-risk decision and
privacy/ownership review. Flight qualification is a different activity.
