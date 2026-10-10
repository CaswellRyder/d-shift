from dtr.data import write_json
from scripts.score_development_sweeps import ARMS, REFERENCES, SEEDS, permutation, score

SHIFTS = ("identity", "darker_0.6", "gamma_0.7", "cool_cast")


def counts(red_tp, blue_fp):
    return dict(red_balloon=dict(tp=red_tp, fp=0, fn=2-red_tp), blue_balloon=dict(tp=1, fp=blue_fp, fn=0))


def write_runs(root, blue_fp, oversized_frame):
    for name in [f"{a}-{s}" for a in ARMS for s in SEEDS] + list(REFERENCES):
        rows = []
        for shift in SHIFTS:
            for search in ("mser", "mser_split", "mser_split_oversized"):
                frames = [counts(1 + (search != "mser"), blue_fp(name)), counts(2, 0)]
                if search == "mser_split_oversized":
                    frames[1] = oversized_frame
                rows.append(dict(transform=shift, search=search, frame_counts=frames,
                                 counts={"indoor": {l: {k: sum(f[l][k] for f in frames) for k in ("tp", "fp", "fn")}
                                                    for l in ("red_balloon", "blue_balloon")}}))
        write_json(root / f"sweep-{name}.json", dict(rows=rows, model=name, model_sha256="m",
                                                     inputs_sha256="i", script_sha256="s"))


def test_scorer_detects_model_contrast_and_rule_regressions(tmp_path):
    write_runs(tmp_path, lambda n: 1 if "admit" in n else 3, counts(2, 1))
    result = score(tmp_path)
    assert result["contrasts"]["mser|identity|b_fp"]["replicated"]
    assert not result["contrasts"]["mser|sweep|b_tp"]["replicated"]  # no TP difference
    split, oversized = result["rules"]["split"], result["rules"]["oversized"]
    assert split["replicated"] and split["worse"] == 0 and split["delta"]["r_tp"] == 30 * 4
    assert not oversized["replicated"] and oversized["worse"] == 30 * 4 and oversized["worse_frames"] == [1]
    assert result["red_tp_by_shift"]["new-views-42"]["darker_0.6"] == 3


def test_permutation_is_exact_for_small_groups():
    assert permutation([1, 1, 1], [2, 2, 2], True) == 2 / 20
