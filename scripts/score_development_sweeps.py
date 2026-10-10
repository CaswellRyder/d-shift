"""Score a directory of photometric sweeps against a pre-registered analysis plan.

Reads `sweep-<name>.json` files written by `research_pixel_photometric.py` with
`--split-merged --oversized-parents`. Model contrasts use the plain `mser` search;
rule contrasts compare searches on the same model, frame and shift. Development
evidence only; synthetic shifts are not real lighting.
"""
import argparse
import itertools
import random
import statistics as st
from pathlib import Path

from dtr.data import read_json, sha256, write_json

LABELS = ("red_balloon", "blue_balloon")
SEEDS = range(42, 48)
ARMS = ("soft-admit", "soft-control", "hard-admit", "hard-control")
REFERENCES = ("photo-42", "photo-43", "nophoto-42", "nophoto-43", "new-views-42", "new-views-43")
SEARCHES = ("mser", "mser_split", "mser_split_oversized")
RULES = (("split", "mser", "mser_split"), ("oversized", "mser_split", "mser_split_oversized"))
SHIFTS = ("darker_0.6", "gamma_0.7", "cool_cast")


def totals(doc):
    out = {}
    for row in doc["rows"]:
        for scope in ("sweep", "identity") if row["transform"] == "identity" else ("sweep",):
            t = out.setdefault(f"{row['search']}|{scope}", {f"{l[0]}_{k}": 0 for l in LABELS
                                                            for k in ("tp", "fp", "fn")})
            for l in LABELS:
                for k in ("tp", "fp", "fn"):
                    t[f"{l[0]}_{k}"] += sum(c[l][k] for c in row["counts"].values())
    return out


def permutation(a, b, exact):
    observed, pool, n = abs(st.mean(a) - st.mean(b)), a + b, len(a)
    if exact:
        splits = itertools.combinations(range(len(pool)), n)
        diffs = [st.mean([pool[i] for i in s]) - st.mean([pool[i] for i in range(len(pool)) if i not in s])
                 for s in splits]
    else:
        rng, diffs = random.Random(0), []
        for _ in range(20000):
            rng.shuffle(pool)
            diffs.append(st.mean(pool[:n]) - st.mean(pool[n:]))
    return sum(abs(d) >= observed - 1e-9 for d in diffs) / len(diffs)


def rule_audit(docs, before, after):
    delta = {f"{l[0]}_{k}": 0 for l in LABELS for k in ("tp", "fp")}
    better = worse = 0
    worse_frames, models_worse = set(), []
    for name, doc in docs.items():
        rows = {(r["transform"], r["search"]): r["frame_counts"] for r in doc["rows"]}
        model_tp = model_fp = 0
        for (shift, search), a in rows.items():
            if search != before:
                continue
            for i, (x, y) in enumerate(zip(a, rows[shift, after], strict=True)):
                tp = sum(y[l]["tp"] - x[l]["tp"] for l in LABELS)
                fp = sum(y[l]["fp"] - x[l]["fp"] for l in LABELS)
                for l in LABELS:
                    for k in ("tp", "fp"):
                        delta[f"{l[0]}_{k}"] += y[l][k] - x[l][k]
                model_tp, model_fp = model_tp + tp, model_fp + fp
                if tp < 0 or (tp == 0 and fp > 0):
                    worse += 1
                    worse_frames.add(i)
                elif tp > 0 or fp < 0:
                    better += 1
        if model_tp < 0 or (model_tp == 0 and model_fp > 0):
            models_worse.append(name)
    tp, fp = delta["r_tp"] + delta["b_tp"], delta["r_fp"] + delta["b_fp"]
    return dict(delta=delta, better=better, worse=worse, worse_frames=sorted(worse_frames),
                models_worse=models_worse,
                replicated=better > worse and tp >= 0 and fp <= 0 and (tp > 0 or fp < 0))


def score(runs):
    runs = Path(runs)
    names = [f"{arm}-{s}" for arm in ARMS for s in SEEDS] + list(REFERENCES)
    docs = {n: read_json(runs / f"sweep-{n}.json") for n in names}
    t = {n: totals(d) for n, d in docs.items()}
    group = {arm: [t[f"{arm}-{s}"] for s in SEEDS] for arm in ARMS}
    contrasts = {}
    for scope, key, direction in (("identity", "b_fp", "lower"), ("sweep", "b_fp", "lower"),
                                  ("sweep", "b_tp", "higher")):
        get = lambda arm: [r[f"mser|{scope}"][key] for r in group[arm]]
        clutter, control = get("soft-admit") + get("hard-admit"), get("soft-control") + get("hard-control")
        p = permutation(clutter, control, False)
        moved = st.mean(clutter) < st.mean(control) if direction == "lower" else \
            st.mean(clutter) > st.mean(control)
        contrasts[f"mser|{scope}|{key}"] = dict(
            groups={arm: get(arm) for arm in ARMS}, predicted=direction,
            pooled_clutter_mean=st.mean(clutter), pooled_control_mean=st.mean(control),
            p_pooled_clutter_vs_control=p, replicated=moved and p < .05,
            p_soft_admit_vs_control=permutation(get("soft-admit"), get("soft-control"), True),
            p_hard_vs_soft_admit=permutation(get("hard-admit"), get("soft-admit"), True))
    rules = {name: rule_audit(docs, before, after) for name, before, after in RULES}
    red_shift = {n: {s: sum(c["red_balloon"]["tp"] for r in d["rows"] if r["transform"] == s
                            and r["search"] == "mser" for c in r["counts"].values())
                     for s in ("identity", *SHIFTS)} for n, d in docs.items()}
    return dict(contrasts=contrasts, rules=rules, red_tp_by_shift=red_shift,
                runs={n: dict(model=d["model"], model_sha256=d["model_sha256"],
                              inputs_sha256=d["inputs_sha256"], script_sha256=d["script_sha256"],
                              totals=t[n]) for n, d in docs.items()},
                scorer_sha256=sha256(__file__))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    result = score(args.runs)
    write_json(args.output, result)
    for k, c in result["contrasts"].items():
        print(k, f"clutter {c['pooled_clutter_mean']:.1f} control {c['pooled_control_mean']:.1f}",
              f"p {c['p_pooled_clutter_vs_control']:.4f}", "replicated" if c["replicated"] else "not replicated")
    for k, r in result["rules"].items():
        print(k, r["delta"], "better", r["better"], "worse", r["worse"],
              "replicated" if r["replicated"] else "not replicated")
