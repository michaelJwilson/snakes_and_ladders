# docs/templates/

Templates for the pages the agentic loop publishes, filled by the agent rather
than generated. Each is one self-contained file: no build step, and no
external resource beyond Google Fonts. `tests/regression/docs/test_templates.py`
checks what this file states about each template.

## `work_in_flight.html`: the work-in-flight page

One page answers "what is open, what is next, and what is waiting on whom". It
is published as a private claude.ai artifact, refreshed hourly and whenever
work lands, and republished to the same URL so the link holds (issue #970,
after michaelJwilson/port#335).

### What each section states, and where it comes from

| Section | Content | Source |
| --- | --- | --- |
| header | `{STAMP}`, a `{HEADLINE}` and `{TLDR}` with the result first, and the counts `{N_GREEN}`, `{N_PENDING}`, `{N_RED}`, `{N_QUEUED}` | derived from the sections below |
| merge order | one lane per PR stack (`{LANE_TITLE}`), its PRs in merge order | each PR's base: a PR stacked on another merges after it |
| open pull requests | `{PR}`, `{TITLE}` with its `{HEADLINE_NUMBER}`, `{BASE}`, `{STATE}` and `{NEXT}`, the next action | the open PRs, the check runs on each head, and mergeability |
| in progress and queued | one card per work item: `{TICKET}`, `{NAME}`, `{STATUS}`, and `{RESULT_FIRST}`, the result and then what remains | the session's task list and the requests behind it |
| tickets filed this session | `{N}`, `{NAME}` and `{WHERE_IT_STANDS}` | the issues the session opened |
| footer | `{CAVEATS}` | what the snapshot cannot see, such as a run waiting on a maintainer |

### State chips

- `ok`: green on the last pushed commit, read from its check runs and never
  inferred from an earlier one.
- `wait`: CI pending on the current head, or a work item in progress.
- `bad`: red, conflicted, or waiting on a decision.
- `idle`: queued and not started.

### Rules

- Root `CLAUDE.md` Writing Style 8 at page scale: the TL;DR leads, and every
  row and card leads with its number (for example "19.4× at 318,945
  positions"), not with its narrative.
- A number on the page is one a PR or ticket already states; the page carries
  no measurement of its own.
- Colours are tokens on `:root`, redefined for dark mode under
  `prefers-color-scheme` and again under `[data-theme="dark"]`. The table
  scrolls inside `.tablebox` and the grids collapse, so the page holds at phone
  width.

### Refresh

1. Read the open PRs, the check runs on each head and their mergeability.
   Rebuild the table, the lanes (from each PR's base) and the header counts.
2. Rewrite the queue and ticket cards from the task list.
3. Republish to the same artifact URL.

An hourly check-in does steps 1–3, and anything that lands triggers one.

### Out of scope

A generator script. The headline number and the next action are judgement, so
the page stays a template until a field is found that can be derived
mechanically.
