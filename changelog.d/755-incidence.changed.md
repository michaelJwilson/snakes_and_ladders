`tests/regression/test_duplication_guards.py` refuses a compressed-sparse
store built by hand: no `offsets` or `indptr` array by `np.cumsum` and no pair
list sorted into row-major order outside `sal.incidence`, which
10 modules build a store through. Three files are excluded against a reason
each: `ragged.py` and `search.maxflow` on the measurements that declined the
fold, and `learn.surrogate`, which the guard found and which is folded or
declined under #755.
