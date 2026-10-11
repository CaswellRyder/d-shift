import copy

import numpy as np
import pytest

from scripts.audit_decoder_cost_probes import recount, BASE_ROUNDS


def fixture():
    expected = dict(frozen=[[] for _ in range(42)], independent_neighbor=[[] for _ in range(42)])
    rows = [dict(round=r, method=m, index=i, elapsed_ms=1., candidates=0, rescues=0)
            for r, m in enumerate(BASE_ROUNDS) for i in range(42)]
    summary = {m: dict(calls=84, mean_ms=1., p50_ms=1., p95_ms=1., max_ms=1.) for m in BASE_ROUNDS}
    return dict(timings=rows, summary=summary), expected


def test_recounts_every_duration_and_order():
    report, expected = fixture()
    assert recount(report, expected, BASE_ROUNDS) == report['summary']
    for field, value in [('method', 'wrong'), ('index', 1), ('elapsed_ms', np.nan), ('candidates', 1)]:
        bad = copy.deepcopy(report)
        bad['timings'][0][field] = value
        with pytest.raises(ValueError):
            recount(bad, expected, BASE_ROUNDS)


def test_rejects_missing_rows_or_fabricated_summary():
    report, expected = fixture()
    report['summary']['frozen']['mean_ms'] = .5
    with pytest.raises(AssertionError):
        recount(report, expected, BASE_ROUNDS)
    report['timings'].pop()
    with pytest.raises(ValueError):
        recount(report, expected, BASE_ROUNDS)
