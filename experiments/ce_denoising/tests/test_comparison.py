import pytest
from experiments.ce_denoising.compare_runs import paired_change


def test_paired_ci_matches_realizations_despite_shuffled_order():
    old = [dict(id='a', seed=1, mse=10.), dict(id='b', seed=2, mse=100.)]
    new = [dict(id='b', seed=2, mse=98.), dict(id='a', seed=1, mse=8.)]
    delta, ci = paired_change(old, new, 'mse')
    assert delta == -2
    assert ci == [-2, -2]


def test_paired_comparison_rejects_unpaired_or_duplicate_realizations():
    old = [dict(id='a', seed=1, mse=1.)]
    for new in ([dict(id='b', seed=1, mse=1.)],
                [dict(id='a', seed=2, mse=1.)], old+old, []):
        with pytest.raises(ValueError):
            paired_change(old, new, 'mse')
