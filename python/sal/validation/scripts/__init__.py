"""The scripts :func:`sal.validation.runner.run` executes (issue #972).

Each runs in its own interpreter and is the only file that imports its
framework. ``selftest`` imports none and is the runner's own subject.
"""
