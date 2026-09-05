"""Regression test for phylo.qa.opt_branch_recovery.

Pins the numbers the script computes before matplotlib sees them
(qa/CLAUDE.md): the recovery in panel (a), the confounding in panel (b), and
the caption that reports both -- a caption stating a number nobody checked is
the failure this file exists to prevent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from phylo.qa.opt_branch_recovery import (
    SPLITS,
    build_figure,
    main,
    recovery,
    split_profile,
)

from tests._fixtures import EIGHT_TAXA, SMALL_SITES, fixture_path, load_fixture


@pytest.mark.structural
def test_recovery_returns_one_entry_per_estimable_parameter() -> None:
    # The rooted fixture has 14 branches but 13 estimable parameters.
    truth, fitted, spread, hits = recovery(load_fixture(EIGHT_TAXA))
    assert truth.shape == fitted.shape == spread.shape == hits.shape == (13,)
    assert bool((spread > 0.0).all())

    truth, *_ = recovery(load_fixture(SMALL_SITES))
    assert truth.shape == (5,)


@pytest.mark.mathematical
def test_the_root_pair_profile_is_flat_and_the_control_is_not() -> None:
    # Panel (b)'s whole content, as numbers. The threshold separating them is
    # not delicate: the flat curve moves by ~1e-12 and the control by ~90.
    params = load_fixture(EIGHT_TAXA)
    splits, flat = split_profile(params, use_root_pair=True)
    _, curved = split_profile(params, use_root_pair=False)

    assert np.array_equal(splits, np.asarray(SPLITS))
    assert np.abs(flat).max() < 1e-6
    assert np.abs(curved).max() > 10.0


@pytest.mark.simulated_truth
def test_the_control_peaks_at_the_generating_split() -> None:
    # The fixture's two sibling branches are equal, so moving mass away from
    # an even split must lower the likelihood in both directions. Without
    # this, a monotone curve would also pass the test above.
    _, curved = split_profile(load_fixture(EIGHT_TAXA), use_root_pair=False)
    assert curved.argmax() == len(SPLITS) // 2
    assert curved[0] < curved[1]
    assert curved[-1] < curved[-2]


@pytest.mark.edge_case
def test_an_exactly_flat_profile_is_not_called_noise() -> None:
    # "at most 0.0e+00 -- floating-point noise" says two contradictory
    # things. An exact zero gets its own wording.
    params = load_fixture(EIGHT_TAXA)
    flat = (np.asarray(SPLITS), np.zeros(len(SPLITS)))
    curved = (np.asarray(SPLITS), -np.abs(np.asarray(SPLITS) - 0.5) * 100.0)
    _, caption = build_figure(
        recovery(load_fixture(SMALL_SITES)),
        recovery(params),
        flat,
        curved,
        load_fixture(SMALL_SITES),
        params,
    )
    assert "bit-identical" in caption
    assert "0.0e+00" not in caption


@pytest.mark.structural
def test_the_caption_reports_what_it_measured() -> None:
    unrooted_params = load_fixture(SMALL_SITES)
    rooted_params = load_fixture(EIGHT_TAXA)
    root_profile = split_profile(rooted_params, use_root_pair=True)
    sibling_profile = split_profile(rooted_params, use_root_pair=False)

    _, caption = build_figure(
        recovery(unrooted_params),
        recovery(rooted_params),
        root_profile,
        sibling_profile,
        unrooted_params,
        rooted_params,
    )

    flat = float(np.abs(root_profile[1]).max())
    assert ("bit-identical" if flat == 0.0 else f"{flat:.1e}") in caption
    assert f"{np.abs(sibling_profile[1]).max():.1f}" in caption
    assert str(unrooted_params.seed) in caption
    assert str(rooted_params.seed) in caption
    # qa/CLAUDE.md: captions are plain text pulled into LaTeX verbatim.
    assert not set(caption) & set("_%\\&#")


@pytest.mark.structural
def test_main_writes_a_figure_and_caption(tmp_path: Path) -> None:
    written = main(
        [
            "--unrooted-params",
            str(fixture_path(SMALL_SITES)),
            "--rooted-params",
            str(fixture_path(EIGHT_TAXA)),
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert written.figure_path.is_file()
    assert written.caption_path.read_text() == written.caption
