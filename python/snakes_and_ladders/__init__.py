"""Top-level package. Re-exports only snakes_and_ladders's own utilities, not submodule contents — import `sim`, `likelihood`, `opt`, and `search` explicitly (e.g. `from snakes_and_ladders.likelihood import ...`)."""

from .oxi_snakes_and_ladders import double

__all__ = ["double"]
