"""The polar construction: one transform, three selection rules, one exact law.

Issue #593. Arikan's erasure recursion conserves ``sum_i I(W_i) = N (1 -
eps)``, asserted exactly at four lengths. Reed--Muller is the same transform
under another rule: the information sets coincide at ``N = 8, k = 4`` (the
extended Hamming (8,4)) and part at ``N = 16``. The Gaussian approximation's
ordering is refereed against error rates measured on the rate-one code.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest
import yaml
from sal.likelihood.ldpc import enumerate_codewords
from sal.likelihood.polar import decode_sc
from sal.sim.fixtures import fixture
from sal.sim.polar import (
    KERNEL,
    MAX_DENSE_LENGTH,
    PolarCode,
    PolarParams,
    bec_polar_code,
    bec_reliability,
    capacity,
    gaussian_polar_code,
    gaussian_reliability,
    parity_check,
    polar_information_set,
    polar_transform,
    reed_muller_code,
    reed_muller_information_set,
)

from tests._fixtures import FIXTURES_DIR
from tests._rows import every_value

#: The information set the ticket states at `N = 16`, rate 1/2, on BEC(0.5).
CI_INFORMATION = (3, 5, 7, 9, 11, 13, 14, 15)

#: The declared instances, in the registry since #826 gave the code its row.
FIXTURES = FIXTURES_DIR / "polar"


def _declared(tier: str = "ci") -> PolarParams:
    """The declared instance at one tier, through the registry (#826)."""
    params = fixture("polar", tier).params
    assert isinstance(params, PolarParams)
    return params


@pytest.mark.analytic
def test_the_transform_is_its_own_inverse_over_gf2() -> None:
    # `F F = I` on the kernel, and the Kronecker power of an involution is
    # one. Every decoder here recovers a source vector by applying the
    # transform a second time rather than inverting it, so this is the
    # property that licenses it.
    assert np.array_equal((KERNEL @ KERNEL) % 2, np.eye(2, dtype=np.int64))
    for stages in range(5):
        transform = polar_transform(stages)
        product = (transform @ transform) % 2
        assert np.array_equal(product, np.eye(2**stages, dtype=np.int64))


@pytest.mark.analytic
def test_the_erasure_recursion_conserves_capacity_exactly() -> None:
    # The one law that holds at every length: the transform moves capacity
    # between synthetic channels and creates none. Asserted as an equality to
    # 1e-12 rather than sampled, because the recursion is closed form.
    def check(stages: int) -> None:
        erasure = 0.5
        total = float(capacity(bec_reliability(stages, erasure)).sum())
        assert total == pytest.approx(2**stages * (1.0 - erasure), abs=1e-12)

    every_value([3, 5, 8, 11], check)


@pytest.mark.analytic
def test_polarisation_shows_as_a_thinning_middle() -> None:
    # Polarisation is asymptotic, so it is reported as a trend and asserted
    # only as monotone: the fraction of channels that are neither good nor bad
    # falls with the length. The values are the ticket's.
    fractions = []
    for stages in (3, 5, 8, 11, 14):
        information = capacity(bec_reliability(stages, 0.5))
        middle = np.logical_and(information > 0.1, information < 0.9)
        fractions.append(float(middle.mean()))

    assert fractions[0] == pytest.approx(0.75, abs=1e-12)
    assert fractions[-1] < 0.07
    assert all(later < earlier for earlier, later in pairwise(fractions))


@pytest.mark.oracle
def test_the_polar_and_reed_muller_rules_coincide_at_eight_and_part_at_sixteen() -> (
    None
):
    # One transform, two rules -- reliability against row weight. At `N = 8`,
    # `k = 4` they pick the same code, which is the extended Hamming (8,4);
    # at `N = 16` they do not, and both halves are asserted so the two rules
    # are not read as one.
    assert bec_polar_code(3, 4, 0.5).information.tolist() == [3, 5, 6, 7]
    assert reed_muller_information_set(3, 1).tolist() == [3, 5, 6, 7]

    assert reed_muller_information_set(4, 1).tolist() == [7, 11, 13, 14, 15]
    assert bec_polar_code(4, 8, 0.5).information.tolist() == list(CI_INFORMATION)


@pytest.mark.oracle
def test_the_code_is_the_span_of_its_generator_and_its_distance_is_four() -> None:
    # `parity_check` is what makes every `ParityCheck` consumer apply, so it
    # is held to the definition: the words it admits are exactly the images of
    # the messages, and the (8,4) code's minimum distance is 4 -- the extended
    # Hamming value, known from outside.
    code = bec_polar_code(3, 4, 0.5)
    check = parity_check(code)
    words = enumerate_codewords(check)

    messages = (np.arange(2**code.n_info)[:, None] >> np.arange(code.n_info)) & 1
    spanned = {tuple(code.encode(message[::-1]).tolist()) for message in messages}
    assert {tuple(word.tolist()) for word in words} == spanned

    weights = words.sum(axis=1)
    assert int(weights[weights > 0].min()) == 4


@pytest.mark.oracle
def test_the_declared_instance_is_the_code_the_ticket_states() -> None:
    # The fixture is the instance every caller shares, so what it builds is
    # asserted rather than assumed: `N = 16`, rate 1/2, and the information
    # set the exact recursion picks.
    with (FIXTURES / "ci.yaml").open() as handle:
        assert yaml.safe_load(handle)["oracle"] == "enumeration"
    code = _declared().code()

    assert (code.n_bits, code.n_info) == (16, 8)
    assert code.rate == 0.5
    assert code.information.tolist() == list(CI_INFORMATION)
    assert code.frozen.tolist() == [0, 1, 2, 4, 6, 8, 10, 12]


@pytest.mark.end2end
def test_the_gaussian_approximation_orders_the_channels_as_a_decoder_does() -> None:
    # The ordering against measured per-channel SC error rates on the
    # rate-one code, not the genie-aided one: Spearman 0.897 against 0.903
    # for the exact erasure recursion; the extremes are exact.
    stages, draws, sigma = 4, 2000, 1.0
    rate_one = PolarCode(stages, np.arange(2**stages))
    errors = np.zeros(2**stages)
    for seed in range(draws):
        rng = np.random.default_rng(seed)
        received = 1.0 + sigma * rng.standard_normal(2**stages)
        errors += decode_sc(rate_one, 2.0 * received / sigma**2).source
    measured = errors / draws

    def spearman(first: np.ndarray, second: np.ndarray) -> float:
        left = np.argsort(np.argsort(first)).astype(float)
        right = np.argsort(np.argsort(second)).astype(float)
        left -= left.mean()
        right -= right.mean()
        return float((left @ right) / np.sqrt((left @ left) * (right @ right)))

    approximate = gaussian_reliability(stages, sigma)
    exact = bec_reliability(stages, 0.5)

    assert spearman(approximate, measured) > 0.85
    assert spearman(exact, measured) > 0.85
    assert spearman(approximate, exact) > 0.99
    assert int(np.argmin(approximate)) == int(np.argmin(measured))
    assert int(np.argmax(approximate)) == int(np.argmax(measured))


@pytest.mark.analytic
def test_the_two_constructions_pick_the_same_code_at_the_declared_instance() -> None:
    # Not a theorem, and recorded because it is convenient rather than
    # assumed: at `N = 16`, rate 1/2, the Gaussian approximation and the exact
    # erasure recursion agree exactly, so the fixture's claims do not depend
    # on which rule built it. They are free to part at a larger length.
    assert (
        gaussian_polar_code(4, 8, 1.0).information.tolist()
        == bec_polar_code(4, 8, 0.5).information.tolist()
    )


@pytest.mark.smoke
def test_a_tie_in_the_reliabilities_breaks_on_the_index() -> None:
    # A stable sort makes the information set a function of the reliabilities
    # alone. With every channel equal the first `k` indices are taken, which
    # is what a reader can predict.
    flat = np.full(8, 0.25)
    assert polar_information_set(flat, 3).tolist() == [0, 1, 2]


@pytest.mark.smoke
def test_the_constructions_refuse_what_they_cannot_answer_for() -> None:
    with pytest.raises(ValueError, match="past MAX_DENSE_LENGTH"):
        polar_transform(int(np.log2(MAX_DENSE_LENGTH)) + 1)
    with pytest.raises(ValueError, match="erasure probability lies in"):
        bec_reliability(3, 1.5)
    with pytest.raises(ValueError, match="noise scale is positive"):
        gaussian_reliability(3, 0.0)
    with pytest.raises(ValueError, match="does not fit"):
        polar_information_set(np.zeros(4), 5)
    with pytest.raises(ValueError, match="0 <= r <= m"):
        reed_muller_information_set(3, 4)
    with pytest.raises(ValueError, match="strictly ascending"):
        PolarCode(2, np.array([2, 1]))
    with pytest.raises(ValueError, match="outside"):
        PolarCode(2, np.array([4]))
    with pytest.raises(ValueError, match="rate-one code has an empty dual"):
        parity_check(PolarCode(2, np.arange(4)))
    with pytest.raises(ValueError, match="the code takes 4"):
        PolarCode(2, np.arange(4)).encode(np.zeros(3))


@pytest.mark.smoke
def test_a_fixture_asking_for_an_impossible_reed_muller_rate_is_refused() -> None:
    # The weight rule fixes `k`, so a fixture cannot ask for another one: the
    # error names the sizes RM does produce rather than silently returning the
    # nearest.
    declared = _declared()
    impossible = replace(declared, construction="reed-muller", n_info=7)
    with pytest.raises(ValueError, match="no RM"):
        impossible.code()


@pytest.mark.analytic
def test_reed_muller_as_a_polar_code_carries_the_weight_rule() -> None:
    # The constructor and the rule are one object, so the code RM builds is
    # the rule's set and its rate follows from the length rather than being
    # chosen.
    code = reed_muller_code(4, 1)
    assert code.information.tolist() == [7, 11, 13, 14, 15]
    assert code.n_info == 5
