`test_the_ci_tier_excludes_the_stress_and_key_tiers` reads the tier
partition over the non-release universe: a `release` test with an `at_scale`
stress case (#756's sampler-efficiency test) counted on one side and not the
other, one off on every local tier from 2026-09-19.
