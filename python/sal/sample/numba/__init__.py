"""The ``numba`` twins of ``sample``'s algorithms, one module per algorithm (issue #1059).

Every kernel here touches no Python object and so is ``nogil=True``: a thread
backend runs them concurrently, the rule root ``CLAUDE.md`` states (#604).
The kernels take arrays, never the graph: the caller flattens once and the
kernel walks strides (the layout rule), and ``cache=True`` writes the
compiled object beside the source so the first call in a process pays once.
"""
