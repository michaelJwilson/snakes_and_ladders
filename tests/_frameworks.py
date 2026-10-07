"""The skip a test of an external framework carries, read from the registry (issue #1010).

Ten test modules wrote `skipif(not available("<module>"), reason="... is the
validation-<name> extra")` by hand, restating what
:data:`sal.external.frameworks.FRAMEWORKS` declares: the module its
script imports and the extra that installs it. A source build (OpenGM,
#1279) has no extra: its skip names the script that builds it.
"""

from __future__ import annotations

import pytest
from sal.external.frameworks import FRAMEWORKS, built
from sal.external.runner import installed


def requires(name: str) -> pytest.MarkDecorator:
    """Skip unless the framework ``name`` is installed, naming the extra that installs it."""
    framework = FRAMEWORKS[name]
    if framework.build is not None:
        return pytest.mark.skipif(
            built(framework) is None,
            reason=f"{framework.distribution} is built by `{framework.build}`",
        )
    return pytest.mark.skipif(
        not installed(framework.module),
        reason=f"{framework.distribution} is the {framework.extra} extra",
    )
