"""Regression test for phylo.qa.opt_recovery.

The figure's claim is that the same model-agnostic fit recovers the
generating parameters of two unrelated models, with intervals that mean what
they say. This pins the numbers the script computes before matplotlib sees
them (qa/CLAUDE.md), including the coverage the caption reports -- a caption
that states a number nobody checked is the failure mode this file exists to
prevent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from phylo.opt.potts import load_potts_params
from phylo.qa.opt_recovery import build_figure, hmm_recovery, main, potts_recovery
from phylo.sim.hmm import load_hmm_params

from tests._fixtures import FIXTURES_DIR

POTTS_FIXTURE = FIXTURES_DIR / "potts_params.yaml"
HMM_FIXTURE = FIXTURES_DIR / "hmm_params.yaml"


@pytest.mark.structural
def test_potts_recovery_returns_one_entry_per_parameter() -> None:
    params = load_potts_params(POTTS_FIXTURE)
    truth, fitted, spread, hits = potts_recovery(params)
    # One coupling plus one field entry per state.
    assert truth.shape == fitted.shape == spread.shape == hits.shape
    assert truth.size == 1 + params.n_states
    assert np.array_equal(truth[1:], params.field)
    assert truth[0] == params.coupling


@pytest.mark.simulated_truth
def test_potts_recovery_covers_every_parameter_at_the_fixture() -> None:
    # Deterministic: fixed seed, so this is a pinned outcome rather than a
    # sample. A single dataset is a draw, not a rate -- the nominal rate is
    # checked over replicates in test_opt_fit.py.
    _, _, spread, hits = potts_recovery(load_potts_params(POTTS_FIXTURE))
    assert bool(hits.all())
    assert bool((spread > 0.0).all())


@pytest.mark.mathematical
def test_hmm_recovery_reports_probabilities_that_normalize() -> None:
    params = load_hmm_params(HMM_FIXTURE)
    truth, fitted, spread, hits = hmm_recovery(params)
    expected = params.n_states + params.n_states**2 + params.n_states * params.n_symbols
    assert truth.size == fitted.size == spread.size == hits.size == expected

    # Initial, then each transition row, then each emission row.
    initial = fitted[: params.n_states]
    np.testing.assert_allclose(initial.sum(), 1.0, rtol=1e-9)
    transition = fitted[params.n_states : params.n_states + params.n_states**2].reshape(
        params.n_states, params.n_states
    )
    np.testing.assert_allclose(
        transition.sum(axis=1), np.ones(params.n_states), rtol=1e-9
    )
    emission = fitted[params.n_states + params.n_states**2 :].reshape(
        params.n_states, params.n_symbols
    )
    np.testing.assert_allclose(
        emission.sum(axis=1), np.ones(params.n_states), rtol=1e-9
    )


@pytest.mark.oracle
def test_hmm_recovery_aligns_the_state_permutation() -> None:
    # Without alignment the fitted emission rows land against the wrong true
    # rows and the figure would show a correct fit as a failure. Checked by
    # requiring each fitted emission row to be closer to its own true row
    # than to any other.
    params = load_hmm_params(HMM_FIXTURE)
    truth, fitted, _, _ = hmm_recovery(params)
    offset = params.n_states + params.n_states**2
    emission = fitted[offset:].reshape(params.n_states, params.n_symbols)
    reference = truth[offset:].reshape(params.n_states, params.n_symbols)
    for state in range(params.n_states):
        own = np.abs(emission[state] - reference[state]).sum()
        others = [
            np.abs(emission[state] - reference[other]).sum()
            for other in range(params.n_states)
            if other != state
        ]
        assert own < min(others)


@pytest.mark.structural
def test_the_caption_reports_the_coverage_it_measured() -> None:
    potts_params = load_potts_params(POTTS_FIXTURE)
    hmm_params = load_hmm_params(HMM_FIXTURE)
    potts = potts_recovery(potts_params)
    hmm = hmm_recovery(hmm_params)

    _, caption = build_figure(potts, hmm, potts_params, hmm_params)

    assert f"{int(potts[3].sum())} of {potts[3].size}" in caption
    assert f"{int(hmm[3].sum())} of {hmm[3].size}" in caption
    assert str(potts_params.seed) in caption
    assert str(hmm_params.seed) in caption
    # qa/CLAUDE.md: captions are plain text pulled into LaTeX verbatim.
    assert not set(caption) & set("_%\\&#")


@pytest.mark.structural
def test_main_writes_a_figure_and_caption(tmp_path: Path) -> None:
    written = main(
        [
            "--potts-params",
            str(POTTS_FIXTURE),
            "--hmm-params",
            str(HMM_FIXTURE),
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert written.figure_path.is_file()
    assert written.caption_path.read_text() == written.caption
