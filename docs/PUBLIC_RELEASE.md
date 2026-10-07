# Public-release readiness

Reviewed 2026-10-06. **Repository visibility remains private.** This checklist
concerns publishing research software, not flight approval. No arena tests,
vehicle, or new accuracy threshold are required simply to publish honestly
documented research.

## Prepared

- [x] Concise front page, explicit model identity, clone-ready student examples.
- [x] Documentation index and preserved historical development guide.
- [x] Model card, selected weight lineage, checksums, and versioned release assets.
- [x] Source/runtime/data attribution retained; optional YOLO scope distinguished.
- [x] Contribution guidance, security policy, issue/PR templates, and changelog.
- [x] CI definition with read-only permissions and commit-pinned external actions.
- [x] Local regression: 237 Python tests passed, two skipped; Ruff passed.
- [x] v1 verifier: 30 artifact files and 105 source-file hashes unchanged.
- [x] Targeted credential-pattern scan of 219 reachable Git blobs at review time.
- [x] Both v1 ZIPs and archived Keras JSON metadata checked for the same patterns.
- [x] Datasets, camera images, cached targets, and standalone pixel source excluded
  from the selected release assets.

The targeted scan found no matches for the checked private-key, GitHub/cloud-token,
and previously exposed credential patterns. It is not a complete secret-detection
audit and does not prove that all possible sensitive material is absent.
New changes need another scan. Remote CI passed for cleanup commit `6ddc67a`
([run 37562499662](https://github.com/CaswellRyder/d-shift/actions/runs/37562499662)):
lint, Python and browser-policy tests, artifact verification, and generated model
loading checks. This validates desktop packaging, not Pi or flight performance.
The runner reported action Node.js 20 deprecation and an upcoming `ubuntu-latest`
image migration; both are maintenance notices, not failed checks.

## Before changing visibility

### 1. Decide dependency policy

GitHub currently reports **106 open dependency alerts across 45 distinct
advisories**: 63 high, 31 medium, 12 low. Counts include repeated advisories across
manifests and are a dated snapshot, not a live security status.

| Package | Relevant surface | Next review |
| --- | --- | --- |
| Keras | Model deserialization and training | Trusted-model boundary; patched-version loading/export parity |
| Pillow | Input image decoding/resizing | Patched-version image and prediction regression checks |
| PyTorch | Optional backbone import/detector environment | Separate optional environment upgrade and checkpoint compatibility |
| setuptools | Installation/build tooling | Patched build compatibility |
| pytest | Development tests | Upgrade development tooling independently where compatible |

Recommended: triage/remediate in a new environment while retaining v1 as historical
reproduction evidence. Publishing a clearly labeled research snapshot with known
advisories is possible, but requires an explicit maintainer risk decision; it must
not be described as secure, production-ready, or safe for untrusted inputs.
Do not suppress alerts, rewrite v1 hashes, or claim mitigation solely from localhost
binding. See [SECURITY.md](../SECURITY.md).

### 2. Review historical privacy

Nineteen file paths in the reviewed reachable history contain local machine paths
such as `/Users/<user>/...` or `/home/<user>/...`. Original artifact provenance,
commit author identities, and operational notes can become visible with the repo.
These are not necessarily secrets, but owner approval is still needed.

Review both Git history and release assets. Editing only the current README does
not remove old data from commits, tags, or ZIPs. If sanitization is wanted, choose
an explicit migration/public-mirror plan before rewriting history or frozen assets.
No history rewrite or artifact alteration is authorized by this checklist.

### 3. Confirm ownership and attribution

Confirm that contributed project/team code and the chosen Apache-2.0 publication
are authorized under applicable course, team, and institutional agreements.
Existing notices are retained; this review is not a legal ownership determination.
No private credentials or third-party pixel source may be published. Keep dataset
and pretrained-model attribution, even though datasets are not uploaded.

### 4. Finish publication checks

- [x] Observe the new GitHub Actions run; record any platform-specific limitations.
- [ ] Owner approves historical-path/author disclosures or a sanitization plan.
- [ ] Owner confirms publication rights and dependency-risk policy.
- [ ] Owner explicitly requests changing visibility to public.
- [ ] After publication, verify unauthenticated clone, README links, release/model
  downloads, and the private vulnerability-reporting channel if supported.

Keep `v1.0.0` research status and its limitations visible after publication.
Do not gate public documentation on solving autonomy, capture, or flight control.
