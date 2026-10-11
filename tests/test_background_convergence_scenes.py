from scripts.evaluate_background_convergence import paired_survivors


def test_requires_both_seeds_nonregressing():
    names = ['positive_only47', 'positive_only48', 'with_background47', 'with_background48']
    summary = {n: {'nonregression': True} for n in names}
    assert paired_survivors(names, summary) == names
    summary['positive_only48']['nonregression'] = False
    assert paired_survivors(names, summary) == names[2:]
    assert paired_survivors(['with_background47'], summary) == []
    assert paired_survivors([], summary) == []
