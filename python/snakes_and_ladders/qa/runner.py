"""Shared command-line entry point for the QA figure and table scripts.

Every script here loads one or more declarative parameters files, computes
what it reports on, and writes a figure or table beside a caption. Only the
middle step differs between them, so the two ends live here: this module owns
the argument parsing, the loading, the writing, and closing the figure
afterwards, and a script supplies the stem, the parameters it takes, and the
builder that turns them into something to render.

The alternative is what this replaced -- the same parser, the same
``--output-dir``, and the same ``try/finally`` around ``plt.close`` repeated
in thirteen modules, where a fix to one of them reached only that one.

Every script now reports what it wrote. Three of them did and ten did not, so
the build log named a third of its own output; uniform reporting is the point
of routing them all through here.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from snakes_and_ladders.opt.potts import load_potts_params
from snakes_and_ladders.qa.figure import (
    QAFigure,
    QATable,
    write_qa_figure,
    write_qa_table,
)
from snakes_and_ladders.sim.hmm import load_hmm_params
from snakes_and_ladders.sim.params import load_simulation_params


@dataclass(frozen=True)
class ParamsArgument:
    """One parameters-file argument, and the loader that reads it.

    Parameters
    ----------
    flag : str
        Long-option name without the leading dashes, e.g. ``"potts-params"``.
        The builder receives the loaded values in the order these are declared,
        so the flag names are the script's interface and the order is the
        builder's.
    load : Callable[[Path], Any]
        Reads the file into whatever the builder expects. The format is the
        model's own, per ``qa/CLAUDE.md``: what matters is that the caption
        reports what actually ran, not that every script reads one layout.
    repeated : bool
        Whether the flag may be given more than once. When true the builder
        receives a list of loaded values rather than one.
    """

    flag: str
    load: Callable[[Path], Any]
    repeated: bool = False

    @property
    def dest(self) -> str:
        """The ``argparse`` destination the flag parses into."""
        return self.flag.replace("-", "_")


@dataclass(frozen=True)
class Option:
    """One command-line option that is not a parameters file.

    Passed to the builder as a keyword argument, so a figure that takes a
    knob declares it here rather than reading ``sys.argv`` itself.

    Parameters
    ----------
    flag : str
        Long-option name without the leading dashes.
    parse : Callable[[str], Any]
        Converts the raw string, e.g. ``int``.
    default : Any
        Value used when the flag is absent.
    """

    flag: str
    parse: Callable[[str], Any]
    default: Any

    @property
    def dest(self) -> str:
        """The ``argparse`` destination the flag parses into."""
        return self.flag.replace("-", "_")


def _parse(
    description: str | None,
    params: Sequence[ParamsArgument],
    options: Sequence[Option],
    argv: list[str] | None,
) -> tuple[list[Any], dict[str, Any], Path]:
    """Parse ``argv`` into loaded parameters, options, and an output directory.

    Returns
    -------
    tuple[list[Any], dict[str, Any], Path]
        The loaded parameter values in declaration order, the option values by
        name, and the directory to write into.
    """
    parser = argparse.ArgumentParser(description=description)
    for argument in params:
        if argument.repeated:
            parser.add_argument(
                f"--{argument.flag}",
                type=Path,
                required=True,
                action="append",
                dest=argument.dest,
            )
        else:
            parser.add_argument(f"--{argument.flag}", type=Path, required=True)
    for option in options:
        parser.add_argument(
            f"--{option.flag}", type=option.parse, default=option.default
        )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    loaded: list[Any] = []
    for argument in params:
        value = getattr(args, argument.dest)
        if argument.repeated:
            loaded.append([argument.load(path) for path in value])
        else:
            loaded.append(argument.load(value))
    option_values = {option.dest: getattr(args, option.dest) for option in options}
    output_dir: Path = args.output_dir
    return loaded, option_values, output_dir


def figure_main(
    *,
    stem: str,
    description: str | None,
    params: Sequence[ParamsArgument],
    build: Callable[..., tuple[Figure, str]],
    options: Sequence[Option] = (),
    argv: list[str] | None = None,
) -> QAFigure:
    """Render one QA figure from the command line.

    Parameters
    ----------
    stem : str
        Output basename, without extension. A document under ``docs/tex/``
        refers to
        the figure by this name, so it is the script's contract with the
        document and is not derived from the module name.
    description : str | None
        Help text; every caller passes its module ``__doc__``.
    params : Sequence[ParamsArgument]
        The parameters files this figure is rendered from.
    build : Callable[..., tuple[Figure, str]]
        Takes the loaded parameters in declaration order and returns the
        figure and its caption. It must not write anything: writing is what
        this function does, so a caption goes through
        ``figure.check_latex_safe`` exactly once.
    options : Sequence[Option]
        Any further options, passed to the builder by keyword.
    argv : list[str] | None
        Argument vector; ``None`` reads ``sys.argv``.

    Returns
    -------
    QAFigure
        Paths written, and the caption.
    """
    loaded, option_values, output_dir = _parse(description, params, options, argv)
    fig, caption = build(*loaded, **option_values)
    try:
        written = write_qa_figure(output_dir, stem, fig, caption)
        print(f"Wrote {written.figure_path} and {written.caption_path}")
        return written
    finally:
        # Closed here rather than in the builder: a builder that returns a
        # figure it has already closed cannot be written by accident, and a
        # write that raises still releases the figure.
        plt.close(fig)


def table_main(
    *,
    stem: str,
    description: str | None,
    params: Sequence[ParamsArgument],
    build: Callable[..., tuple[str, str]],
    options: Sequence[Option] = (),
    argv: list[str] | None = None,
) -> QATable:
    """Render one QA table from the command line.

    The figure counterpart of this function is :func:`figure_main`; a table is
    typeset by LaTeX rather than drawn, per ``qa/CLAUDE.md``, so its builder
    returns a ``tabular`` body instead of a ``Figure`` and there is nothing to
    close.

    Parameters
    ----------
    stem : str
        Output basename, without extension.
    description : str | None
        Help text; every caller passes its module ``__doc__``.
    params : Sequence[ParamsArgument]
        The parameters files this table is rendered from.
    build : Callable[..., tuple[str, str]]
        Takes the loaded parameters in declaration order and returns the
        ``tabular`` body and its caption.
    options : Sequence[Option]
        Any further options, passed to the builder by keyword.
    argv : list[str] | None
        Argument vector; ``None`` reads ``sys.argv``.

    Returns
    -------
    QATable
        Paths written, and the caption.
    """
    loaded, option_values, output_dir = _parse(description, params, options, argv)
    body, caption = build(*loaded, **option_values)
    written = write_qa_table(output_dir, stem, body, caption)
    print(f"Wrote {written.table_path} and {written.caption_path}")
    return written


# The three parameter files the QA scripts read, declared once. A script names
# the ones it takes rather than restating the flag and its loader, so a figure
# and its test cannot disagree about which file the figure was rendered from.
SIMULATION_PARAMS = ParamsArgument("params", load_simulation_params)
SIMULATION_PARAMS_REPEATED = ParamsArgument(
    "params", load_simulation_params, repeated=True
)
POTTS_PARAMS = ParamsArgument("potts-params", load_potts_params)
HMM_PARAMS = ParamsArgument("hmm-params", load_hmm_params)
