"""The skip a test of an external framework carries, read from the registry (issue #1010).

Ten test modules wrote `skipif(not available("<module>"), reason="... is the
validation-<name> extra")` by hand, restating what
:data:`sal.validation.FRAMEWORKS` declares: the module its
script imports and the extra that installs it.
"""

from __future__ import annotations

import pytest
from sal.validation import FRAMEWORKS
from sal.validation.runner import available


def requires(name: str) -> pytest.MarkDecorator:
    """Skip unless the framework ``name`` is installed, naming the extra that installs it."""
    framework = FRAMEWORKS[name]
    return pytest.mark.skipif(
        not available(framework.module),
        reason=f"{framework.distribution} is the {framework.extra} extra",
    )
