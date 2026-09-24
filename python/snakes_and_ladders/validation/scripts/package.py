"""The package's own calls, measured the way a framework's script measures its own (issue #987).

A benchmark pair reads a framework's time and peak resident memory in a
fresh interpreter; this script reads the package's the same way, so the two
figures come from one method in two interpreters of the same kind and the
process under test carries no measurement. ``call`` names an entry of
:data:`CALLS`, which builds the call from the inputs outside the measured
region; the call itself runs under :func:`~snakes_and_ladders.validation.protocol.timed`
inside :func:`~snakes_and_ladders.validation.protocol.peaked`.

``allocate`` is the measure's own referee: it fills ``n`` float64 values,
``8 n`` bytes the peak must read back.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

from snakes_and_ladders.validation.protocol import dump, load, paths, peaked, timed

#: One output mapping from one measured call.
Outputs = dict[str, np.ndarray]

#: A package call: the inputs in, a zero-argument call out whose run is measured.
Build = Callable[[Mapping[str, np.ndarray]], Callable[[], Outputs]]


def _allocate(inputs: Mapping[str, np.ndarray]) -> Callable[[], Outputs]:
    size = int(inputs["n"])

    def call() -> Outputs:
        block = np.empty(size)
        block.fill(0.0)
        return {"total": np.asarray(block.sum())}

    return call


def _ising_cut(inputs: Mapping[str, np.ndarray]) -> Callable[[], Outputs]:
    """The Rust cut with its arrays prebuilt, as the PyMaxflow pair times it (#973)."""
    from snakes_and_ladders import oxisal

    n_nodes = int(inputs["n_nodes"])
    field, edges, coupling = inputs["field"], inputs["edges"], inputs["coupling"]

    def call() -> Outputs:
        states = oxisal.ising_ground_state(n_nodes, field, edges, coupling)
        return {"configuration": np.asarray(states, dtype=np.int64)}

    return call


def _alpha_expansion(inputs: Mapping[str, np.ndarray]) -> Callable[[], Outputs]:
    """Expansion on the Rust cut to convergence, as the gco pair times it (#974).

    With ``move`` set to ``swap``, the alpha-beta swap instead (#997).
    """
    from snakes_and_ladders.backend import Backend
    from snakes_and_ladders.search.alpha_expansion import (
        alpha_beta_swap,
        alpha_expansion,
    )
    from snakes_and_ladders.sim.graph import BoundaryCondition, lattice_graph

    shape = tuple(int(extent) for extent in inputs["shape"])
    graph = lattice_graph(shape, BoundaryCondition.OPEN, float(inputs["coupling"]))
    field = inputs["field"]
    n_states = int(field.shape[1])

    solver = (
        alpha_beta_swap
        if str(inputs.get("move", np.asarray("expansion"))) == "swap"
        else alpha_expansion
    )

    def call() -> Outputs:
        result = solver(graph, field, n_states, backend=Backend.RUST)
        return {"labelling": result.labelling, "energy": np.asarray(result.energy)}

    return call


def _baum_welch(inputs: Mapping[str, np.ndarray]) -> Callable[[], Outputs]:
    """Ten Baum--Welch iterations from the given start, as the hmmlearn pair runs (#975)."""
    import torch

    from snakes_and_ladders.opt.hmm import baum_welch

    observations = inputs["observations"]
    initial, transition, emission = (
        torch.log(torch.as_tensor(inputs[name]))
        for name in ("initial", "transition", "emission")
    )
    n_iter = int(inputs["n_iter"])

    def call() -> Outputs:
        fit = baum_welch(
            observations,
            initial,
            transition,
            emission,
            max_iterations=n_iter,
            tolerance=-np.inf,
        )
        return {"emission": np.exp(fit.log_emission.numpy())}

    return call


def _family_baum_welch(inputs: Mapping[str, np.ndarray]) -> Callable[[], Outputs]:
    """Ten Baum--Welch iterations of a Gaussian or count HMM (#997).

    ``family`` names it: ``gaussian`` (``mean``, ``variance``), ``poisson``
    (``rate``), ``negative_binomial`` (``dispersion``, ``mean``) or
    ``beta_binomial`` (``trials``, ``alpha``, ``beta``); read from the
    parameters when absent. ``backend`` is ``rust`` (the default) or
    ``python``; ``with_table``, ``table_size`` (``-1`` for every cell) and ``approx`` are
    passed through.
    """
    import torch

    from snakes_and_ladders.backend import Backend
    from snakes_and_ladders.emissions import (
        BetaBinomialEmission,
        EmissionFamily,
        GaussianEmission,
        NegativeBinomialEmission,
        PoissonEmission,
    )
    from snakes_and_ladders.opt.hmm import baum_welch_family

    observations = inputs["observations"]
    initial, transition = (
        torch.log(torch.as_tensor(inputs[name])) for name in ("initial", "transition")
    )
    default = "poisson" if "rate" in inputs else "gaussian"
    name = str(inputs.get("family", np.asarray(default)))
    family: EmissionFamily
    if name == "gaussian":
        family = GaussianEmission(inputs["mean"], np.sqrt(inputs["variance"]), 1e-12)
    elif name == "poisson":
        family = PoissonEmission(inputs["rate"])
    elif name == "negative_binomial":
        family = NegativeBinomialEmission(inputs["dispersion"], inputs["mean"])
    else:
        family = BetaBinomialEmission(inputs["trials"], inputs["alpha"], inputs["beta"])
    backend = Backend(str(inputs.get("backend", np.asarray("rust"))))
    with_table = bool(inputs.get("with_table", np.asarray(True)))
    approx = bool(inputs.get("approx", np.asarray(False)))
    size = int(inputs.get("table_size", np.asarray(-1)))
    table_size = None if size < 0 else size
    n_iter = int(inputs["n_iter"])
    # One iteration on the first two positions of two sequences, outside the
    # measured call: torch's first operations in a process set up state that
    # stays, as `_hmc_sample`'s warm-up does for the sampler.
    baum_welch_family(
        np.ascontiguousarray(observations[:2, :2]),
        initial,
        transition,
        family,
        max_iterations=1,
        tolerance=-np.inf,
        backend=backend,
        with_table=with_table,
        table_size=table_size,
        approx=approx,
    )

    def call() -> Outputs:
        fit = baum_welch_family(
            observations,
            initial,
            transition,
            family,
            max_iterations=n_iter,
            tolerance=-np.inf,
            backend=backend,
            with_table=with_table,
            table_size=table_size,
            approx=approx,
        )
        return {"log_likelihood": np.asarray(fit.log_likelihood)}

    return call


def _viterbi(inputs: Mapping[str, np.ndarray]) -> Callable[[], Outputs]:
    """Viterbi paths at the given parameters, as hmmlearn's ``decode`` runs (#997).

    ``family`` is ``gaussian`` (``mean``, ``variance``) or ``poisson``
    (``rate``); ``initial`` and ``transition`` are probabilities. With
    ``score`` set, the summed log-likelihood instead, as hmmlearn's ``score``.
    """
    import torch

    from snakes_and_ladders.emissions import (
        EmissionFamily,
        GaussianEmission,
        PoissonEmission,
    )
    from snakes_and_ladders.opt.hmm import hmm_log_likelihood, viterbi

    observations = inputs["observations"]
    initial, transition = (
        torch.log(torch.as_tensor(inputs[name])) for name in ("initial", "transition")
    )
    family: EmissionFamily = (
        PoissonEmission(inputs["rate"])
        if str(inputs["family"]) == "poisson"
        else GaussianEmission(inputs["mean"], np.sqrt(inputs["variance"]), 1e-12)
    )
    # Outside the measured call, as `_family_baum_welch`'s warm-up.
    viterbi(np.ascontiguousarray(observations[:2, :2]), initial, transition, family)
    if bool(inputs.get("score", np.asarray(False))):

        def scored() -> Outputs:
            value = hmm_log_likelihood(observations, initial, transition, family)
            return {"log_likelihood": np.asarray(value)}

        return scored

    def call() -> Outputs:
        states, log_probability = viterbi(observations, initial, transition, family)
        return {"states": states, "log_probability": np.asarray(log_probability)}

    return call


def _mixture_score(inputs: Mapping[str, np.ndarray]) -> Callable[[], Outputs]:
    """The summed mixture log-likelihood at given parameters, as scikit-learn's ``score_samples`` (#997)."""
    import torch

    from snakes_and_ladders.emissions import GaussianEmission
    from snakes_and_ladders.opt.mixture import mixture_log_likelihood

    observations = torch.from_numpy(np.ascontiguousarray(inputs["observations"]))
    log_weight = torch.log(torch.as_tensor(inputs["weights"]))
    components = GaussianEmission(inputs["mean"], inputs["scale"], 1e-12)
    # Outside the measured call: torch's first operations set up state.
    mixture_log_likelihood(observations[:2], log_weight, components)

    def call() -> Outputs:
        value = mixture_log_likelihood(observations, log_weight, components)
        return {"log_likelihood": np.asarray(float(value))}

    return call


def _mala_sample(inputs: Mapping[str, np.ndarray]) -> Callable[[], Outputs]:
    """MALA on a zero-mean Gaussian at the Langevin step ``h``, as the BlackJAX pair runs (#997)."""
    import torch

    from snakes_and_ladders.sample import langevin
    from snakes_and_ladders.validation.gaussian import GaussianTarget

    target = GaussianTarget(inputs["precision"])
    step_size = float(inputs["step_size"])
    n_draws, seed = int(inputs["n_draws"]), int(inputs["seed"])
    store_chain = bool(inputs.get("store_chain", np.asarray(True)))
    # Outside the measured call, as `_hmc_sample`'s warm-up.
    langevin.mala(
        GaussianTarget(np.ones(2)),
        torch.Generator().manual_seed(0),
        2,
        step_size=0.1,
        store_chain=store_chain,
    )

    def call() -> Outputs:
        chain = langevin.mala(
            target,
            torch.Generator().manual_seed(seed),
            n_draws,
            step_size=step_size,
            store_chain=store_chain,
        )
        return {"acceptance": np.asarray(chain.acceptance_rate)}

    return call


def _swendsen_wang(inputs: Mapping[str, np.ndarray]) -> Callable[[], Outputs]:
    """One Swendsen--Wang sweep on the compiled pass, at q = 3 on an open lattice (#997).

    rustworkx's figure for a sweep is its graph build and
    ``connected_components`` over that sweep's bonds; this is the whole
    sweep, the bond draw and the recolouring included.
    """
    from snakes_and_ladders.backend import Backend
    from snakes_and_ladders.sample.potts_mcmc import swendsen_wang_sweep
    from snakes_and_ladders.sim.graph import BoundaryCondition, lattice_graph
    from snakes_and_ladders.sim.potts import critical_coupling, site_field

    side = int(inputs["side"])
    graph = lattice_graph((side, side), BoundaryCondition.OPEN, critical_coupling(3))
    rng = np.random.default_rng(int(inputs["seed"]))
    state = rng.integers(0, 3, graph.n_nodes)
    rows = site_field(np.zeros(3), graph.n_nodes)
    # Outside the measured call: three sweeps toward equilibrium, which also
    # pay any one-time set-up.
    for _ in range(3):
        swendsen_wang_sweep(state, graph, rows, rng, backend=Backend.RUST)

    def call() -> Outputs:
        swendsen_wang_sweep(state, graph, rows, rng, backend=Backend.RUST)
        return {"state": state}

    return call


def _mixture_em(inputs: Mapping[str, np.ndarray]) -> Callable[[], Outputs]:
    """Ten mixture EM iterations from the given start, as the scikit-learn pair runs (#975)."""
    import torch

    from snakes_and_ladders.emissions import GaussianEmission
    from snakes_and_ladders.opt.mixture import expectation_maximization

    observations = inputs["observations"]
    weights = torch.as_tensor(inputs["weights"])
    start = GaussianEmission(inputs["mean"], inputs["scale"], 1e-12)
    n_iter = int(inputs["n_iter"])

    def call() -> Outputs:
        fit = expectation_maximization(
            observations, weights, start, max_iterations=n_iter, tolerance=-np.inf
        )
        return {"weights": fit.weights.numpy()}

    return call


def _hmc_sample(inputs: Mapping[str, np.ndarray]) -> Callable[[], Outputs]:
    """HMC at unit mass on a zero-mean Gaussian, as the BlackJAX pair runs (#963)."""
    import torch

    from snakes_and_ladders.sample import hmc
    from snakes_and_ladders.validation.gaussian import GaussianTarget

    target = GaussianTarget(inputs["precision"])
    step_size, n_steps = float(inputs["step_size"]), int(inputs["n_steps"])
    n_draws, seed = int(inputs["n_draws"]), int(inputs["seed"])
    # Issue #988: with ``store_chain`` false the chain keeps no draws and
    # estimates the mean of ``x`` instead.
    store_chain = bool(inputs.get("store_chain", np.asarray(True)))
    # With ``observe`` false a chain-free run keeps nothing but its counters,
    # which the compiled route runs (issue #997); true observes ``x``.
    observe = bool(inputs.get("observe", np.asarray(True)))
    operators = None if store_chain or not observe else {"x": lambda x: x}
    # One-time set-up paid outside the measured call, as the JAX scripts
    # exclude compilation (issue #997): a two-draw chain at d = 2 on the same
    # route, since torch's first operations in a process cost 10.9 MB whatever
    # the chain's size.
    hmc.sample(
        GaussianTarget(np.ones(2)),
        torch.Generator().manual_seed(0),
        2,
        step_size=step_size,
        n_steps=1,
        store_chain=store_chain,
        operators=operators,
    )

    def call() -> Outputs:
        chain = hmc.sample(
            target,
            torch.Generator().manual_seed(seed),
            n_draws,
            step_size=step_size,
            n_steps=n_steps,
            store_chain=store_chain,
            operators=operators,
        )
        return {"acceptance": np.asarray(chain.acceptance_rate)}

    return call


def _random_walk_sample(inputs: Mapping[str, np.ndarray]) -> Callable[[], Outputs]:
    """Random-walk Metropolis on a declared target, as the BlackJAX pair runs (#1006).

    ``target`` 0 is the Gaussian of ``precision``, 1 Rosenbrock's function
    with ``constants``; ``warmup`` proposals above zero run the warm-up at
    the optimal random-walk acceptance first.
    """
    import torch

    from snakes_and_ladders.opt.testfunctions import Rosenbrock
    from snakes_and_ladders.sample import hmc, metropolis
    from snakes_and_ladders.validation.gaussian import GaussianTarget

    position = inputs["position"]
    target = (
        Rosenbrock(position.size, *(float(c) for c in inputs["constants"]))
        if int(inputs["target"]) == 1
        else GaussianTarget(inputs["precision"])
    )
    step_size, n_draws = float(inputs["step_size"]), int(inputs["n_draws"])
    seed, warmup = int(inputs["seed"]), int(inputs["warmup"])
    store_chain = bool(inputs["store_chain"])
    adaptation = (
        hmc.Adaptation(warmup, metropolis.RWM_TARGET_ACCEPTANCE, 0.0)
        if warmup
        else None
    )
    theta0 = torch.as_tensor(position)
    # Outside the measured call, as `_hmc_sample`'s set-up.
    metropolis.random_walk(
        GaussianTarget(np.ones(2)),
        torch.Generator().manual_seed(0),
        2,
        step_size=0.1,
        store_chain=store_chain,
    )

    def call() -> Outputs:
        chain = metropolis.random_walk(
            target,
            torch.Generator().manual_seed(seed),
            n_draws,
            step_size=step_size,
            theta0=theta0,
            adaptation=adaptation,
            store_chain=store_chain,
        )
        return {"acceptance": np.asarray(chain.acceptance_rate)}

    return call


def _declared_target(inputs: Mapping[str, np.ndarray]) -> object:
    """The Gaussian of ``precision`` (``target`` 0), Rosenbrock's function with ``constants`` (1), or a mixture of ``values`` at ``n_components`` (2)."""
    from snakes_and_ladders.opt.testfunctions import Rosenbrock
    from snakes_and_ladders.validation.gaussian import GaussianTarget

    if int(inputs["target"]) == 3:
        from snakes_and_ladders.opt.hmm import GaussianHmmObjective

        return GaussianHmmObjective(inputs["values"], int(inputs["n_states"]))
    if int(inputs["target"]) == 2:
        from snakes_and_ladders.opt.mixture import GaussianMixtureObjective

        return GaussianMixtureObjective(inputs["values"], int(inputs["n_components"]))
    if int(inputs["target"]) == 1:
        return Rosenbrock(
            inputs["position"].size, *(float(c) for c in inputs["constants"])
        )
    return GaussianTarget(inputs["precision"])


def _hmc_declared(inputs: Mapping[str, np.ndarray]) -> Callable[[], Outputs]:
    """HMC on a declared target from ``position``, after a warm-up of ``warmup`` proposals if above zero (#1008)."""
    import torch

    from snakes_and_ladders.sample import hmc
    from snakes_and_ladders.validation.gaussian import GaussianTarget

    target = _declared_target(inputs)
    theta0 = torch.as_tensor(inputs["position"])
    step_size, n_steps = float(inputs["step_size"]), int(inputs["n_steps"])
    n_draws, seed, warmup = (
        int(inputs["n_draws"]),
        int(inputs["seed"]),
        int(inputs["warmup"]),
    )
    store_chain = bool(inputs["store_chain"])
    adaptation = (
        hmc.Adaptation(warmup, float(inputs["target_acceptance"]), 0.0)
        if warmup
        else None
    )
    # Outside the measured call, as `_hmc_sample`'s set-up.
    hmc.sample(
        GaussianTarget(np.ones(2)),
        torch.Generator().manual_seed(0),
        2,
        step_size=0.1,
        n_steps=1,
        store_chain=store_chain,
    )

    def call() -> Outputs:
        chain = hmc.sample(
            target,  # type: ignore[arg-type]
            torch.Generator().manual_seed(seed),
            n_draws,
            step_size=step_size,
            n_steps=n_steps,
            theta0=theta0,
            adaptation=adaptation,
            store_chain=store_chain,
        )
        return {"acceptance": np.asarray(chain.acceptance_rate)}

    # The same call once, untimed: a JAX-declared target compiles its chain
    # on first use, and BlackJAX's figure is its second call (issue #1008).
    call()

    return call


def _cluster_labels(inputs: Mapping[str, np.ndarray]) -> Callable[[], Outputs]:
    """The Swendsen--Wang pass's union-find on a bond mask, as the rustworkx pair runs (#976)."""
    from snakes_and_ladders.sample.potts_mcmc import bond_roots

    n_nodes = int(inputs["n_nodes"])
    bonds = np.stack([inputs["first"], inputs["second"]], axis=1)

    def call() -> Outputs:
        return {"roots": bond_roots(n_nodes, bonds)}

    return call


def _gradient(inputs: Mapping[str, np.ndarray]) -> Callable[[], Outputs]:
    """``gradient_at`` at every point, as the JAX pair differentiates them (#991).

    ``route`` picks the torch path: ``autograd`` is ``hmc.gradient_at``, the
    one HMC calls; ``func`` is ``torch.func.grad_and_value``; ``closed`` is
    the Gaussian's ``P x`` with no tape. The measured call is the loop over
    the points; ``per_point`` is each point's median alone.
    """
    import statistics
    import time

    import torch

    from snakes_and_ladders.opt.mixture import GaussianMixtureObjective
    from snakes_and_ladders.sample.hmc import gradient_at
    from snakes_and_ladders.validation.gaussian import GaussianTarget

    points = torch.as_tensor(inputs["points"])
    route = str(inputs.get("route", np.asarray("autograd")))
    if "precision" in inputs:
        target: object = GaussianTarget(inputs["precision"])
        precision = torch.as_tensor(inputs["precision"])
    elif "n_states" in inputs:
        from snakes_and_ladders.opt.hmm import GaussianHmmObjective

        target = GaussianHmmObjective(inputs["observations"], int(inputs["n_states"]))
        precision = None
    else:
        target = GaussianMixtureObjective(
            inputs["observations"], int(inputs["n_components"])
        )
        precision = None

    def one(point: torch.Tensor) -> torch.Tensor:
        if route == "closed":
            assert precision is not None
            return precision * point if precision.ndim == 1 else precision @ point
        if route == "func":
            grad, _ = torch.func.grad_and_value(target)(point)  # type: ignore[arg-type]
            return grad  # type: ignore[no-any-return]
        return gradient_at(target, point)  # type: ignore[arg-type]

    def call() -> Outputs:
        one(points[0])
        seconds = []
        # One block for every gradient, as JAX's script returns them, rather
        # than a kept row per point stacked into a second copy (issue #986).
        # Written through NumPy: a torch indexing op first used here would
        # charge its lazy set-up to the call.
        gradients = np.empty(tuple(points.shape))
        for row, point in enumerate(points):
            start = time.perf_counter()
            gradient = one(point)
            seconds.append(time.perf_counter() - start)
            gradients[row] = gradient.numpy()
        return {
            "gradients": gradients,
            "per_point": np.asarray(statistics.median(seconds)),
        }

    return call


#: The calls this script measures, by name.
CALLS: dict[str, Build] = {
    "allocate": _allocate,
    "ising_cut": _ising_cut,
    "alpha_expansion": _alpha_expansion,
    "baum_welch": _baum_welch,
    "family_baum_welch": _family_baum_welch,
    "viterbi": _viterbi,
    "mixture_em": _mixture_em,
    "mixture_score": _mixture_score,
    "hmc_sample": _hmc_sample,
    "mala_sample": _mala_sample,
    "random_walk_sample": _random_walk_sample,
    "hmc_declared": _hmc_declared,
    "cluster_labels": _cluster_labels,
    "swendsen_wang": _swendsen_wang,
    "gradient": _gradient,
}


def main() -> None:
    """Build the named call, measure it, and write its outputs back."""
    given, returned = paths()
    inputs = load(given)
    call = CALLS[str(inputs["call"])](inputs)
    (outputs, seconds), peak_bytes = peaked(lambda: timed(call))
    dump(returned, outputs, seconds, peak_bytes)


if __name__ == "__main__":
    main()
