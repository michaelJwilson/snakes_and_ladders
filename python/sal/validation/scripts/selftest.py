"""The runner's own subject: NumPy alone, no framework (issue #972).

Returns every input unchanged, and ``doubled`` as twice ``values``, with the
seconds the doubling took; ``fail`` set to a non-zero integer exits with that
code after writing a line to standard error.
"""

from __future__ import annotations

import sys

import numpy as np

from sal.validation.protocol import dump, measured, received


def main() -> None:
    """Read the inputs, double ``values`` under the timer, and write them back."""
    inputs, returned = received()
    code = int(inputs.get("fail", np.int64(0)))
    if code:
        print(f"selftest asked to fail with {code}", file=sys.stderr)
        raise SystemExit(code)
    doubled, seconds, peak_bytes = measured(lambda: 2.0 * inputs["values"])
    dump(returned, {**inputs, "doubled": doubled}, seconds, peak_bytes)


if __name__ == "__main__":
    main()
