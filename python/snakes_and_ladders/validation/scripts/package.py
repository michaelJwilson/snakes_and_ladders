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
    from snakes_and_ladders import oxi_snakes_and_ladders

    n_nodes = int(inputs["n_nodes"])
    field, edges, coupling = inputs["field"], inputs["edges"], inputs["coupling"]

    def call() -> Outputs:
        states = oxi_snakes_and_ladders.ising_ground_state(
            n_nodes, field, edges, coupling
        )
        return {"configuration": np.asarray(states, dtype=np.int64)}

    return call


def _alpha_expansion(inputs: Mapping[str, np.ndarray]) -> Callable[[], Outputs]:
    """Expansion on the Rust cut to convergence, as the gco pair times it (#974)."""
    from snakes_and_ladders.backend import Backend
    from snakes_and_ladders.search.alpha_expansion import alpha_expansion
    from snakes_and_ladders.sim.graph import BoundaryCondition, lattice_graph

    shape = tuple(int(extent) for extent in inputs["shape"])
    graph = lattice_graph(shape, BoundaryCondition.OPEN, float(inputs["coupling"]))
    field = inputs["field"]
    n_states = int(field.shape[1])

    def call() -> Outputs:
        result = alpha_expansion(graph, field, n_states, backend=Backend.RUST)
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
        return {"emission": fit.log_emission.exp().numpy()}

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


#: The calls this script measures, by name.
CALLS: dict[str, Build] = {
    "allocate": _allocate,
    "ising_cut": _ising_cut,
    "alpha_expansion": _alpha_expansion,
    "baum_welch": _baum_welch,
    "mixture_em": _mixture_em,
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
