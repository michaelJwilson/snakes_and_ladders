`test_opt_hmm.py` constructed `HmmParams` with `sequence_length` and
`n_sequences`, which #666 made derived properties rather than fields, so
`dataclasses.replace` raised `TypeError` and `main` was red on the full tier
from #667's merge onward. The instance is written as `lengths` -- 600 chains of
15, the equal-length case spelled out -- and the fit is unchanged.

**Every pull-request gate missed it and three pushes to `main` caught it**, the
same one test each time. `test_opt_hmm.py` imports `opt.hmm`, which #667
changed, so `infra/select_tests.py`'s import-graph reasoning should have
selected this file on that pull request. Why it did not is worth its own look;
this restores the green.
