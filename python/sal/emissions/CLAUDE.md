# CLAUDE.md --- `sal.emissions`

The root `CLAUDE.md` governs; this file holds the module's principles only.

*   **One construction per density.** The beta-binomial is `log C`, the two log rates and the scaled rising factorials `S(x, m) = R(x, m) - m log x` from `rising.scaled_rising_array`, summed in the order `bb` states, on every NumPy and compiled route: `bb.beta_binomial_log_pmf`, `bb.trial_tables` and the kernels that read them agree bit for bit. The torch `log_density` keeps its own arithmetic and is held to a declared tolerance, not re-recorded (#1332).
*   **No difference of two large `lgamma`.** A term of size `x log x` differenced to a value of size `m log x` is written as a scaled rising factorial, and the `m log x` parts are collected into logarithms of rates.
