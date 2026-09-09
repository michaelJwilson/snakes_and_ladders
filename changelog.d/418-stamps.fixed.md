A figure's input stamp now hashes the definitions the renderer actually
reaches, parsed as an AST with docstrings stripped, rather than the source text
of every module in its import closure. Rewording a docstring or declaring an
unread constant in `qa/runner.py` staled all 19 cited figures and now stales
none, while editing a function they call still stales exactly those that call
it. `infra/review_gates.sh` gains three checks and runs nine in 29 s.
