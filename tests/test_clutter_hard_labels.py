import numpy as np
import pytest

from scripts.distill_pi_student import row_alphas


def test_clutter_rows_drop_teacher_term_only_when_requested():
    rows = [dict(label="background"), dict(label="background", clutter_ablation_mode="admit"),
            dict(label="red_balloon"), dict(label="background", clutter_ablation_mode="control")]
    np.testing.assert_array_equal(row_alphas(rows, .5, False), [.5, .5, .5, .5])
    np.testing.assert_array_equal(row_alphas(rows, .5, True), [.5, 1, .5, 1])


def test_clutter_hard_labels_require_flagged_rows_and_distillation():
    with pytest.raises(ValueError):
        row_alphas([dict(label="background")], .5, True)
    with pytest.raises(ValueError):
        row_alphas([dict(label="background", clutter_ablation_mode="admit")], 1, True)
