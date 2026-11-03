# CODING_STANDARDS

Conventions this repository is actually held to. They are not aspirations: each
one has either a machine check or a one-line review rule.

## Layout

- `rnext/` — the deep modules (DSP, ML, tracker client, parsers, scorer). No
  CLI logic here; importable, testable, no `sys.argv`.
- `scripts/` — thin CLI entry points over `rnext/`. Argument parsing, printing,
  exit codes. A script that computes something belongs in `rnext/`.
- `skills/` — agent skills. Prose plus relative links to `scripts/` and
  `rnext/`; never machine-specific paths.
- `research/` — the dataset/model harness. Nothing in the product imports it.
- `tests/` — pytest, plus saved HTML fixtures under `tests/fixtures/`.

## The one-owner rule

A piece of knowledge is implemented once, in one module, and every caller
delegates to it:

- the release scorer is `rnext.planner.score_facts` — `scripts/tracker_hunt.py`
  only wraps it;
- the workspace paths are `rnext/workspace.py` — skills and scripts ask it, they
  never hardcode a directory;
- the secrets are `rnext/secrets.py` — one parser for the cookie file.

A second copy is a bug even when the numbers currently agree: the copies drift
and the flows start ranking the same input differently.

## Errors and data

- No fallbacks instead of required data. A missing source take, a missing
  `data_root`, an empty cookie file, an unparsable page — raise a specific
  exception with the path/value in the message. A silent default turns a broken
  configuration into wrong output.
- No hidden retries. A retry is explicit, bounded, observable, and only for a
  failure known to be transient.
- Catch at the boundary only (CLI `main()`, a subprocess call), keep the cause
  (`raise ... from exc` or the original message), and never swallow it inside
  domain logic.
- No `TODO` markers or stubs standing in for an implementation. An unsupported
  input is an explicit refusal with the reason — not a fake success.
- Secrets are never printed or written to the repo. The cookie scripts report
  cookie *names* and value *lengths* only.

## Naming and style

- English for code, comments, docstrings and commit messages; other languages
  only where the tracked site itself is in another language (the ru/uk patterns
  in the parsers are byte-exact on purpose).
- Comments explain the non-obvious: an invariant, a site quirk, why a value is
  what it is. Do not restate the code.
- Module docstrings state the purpose and the CLI usage; public functions state
  the shape of their input and output.
- Prefer the standard library. Before adding a dependency, check the installed
  version and its docs for the built-in path.

## Tests

- `python3 -m pytest tests/ -q` is the gate; CI runs the same command
  (`.github/workflows/ci.yml`).
- Every parser that touches scraped markup gets a test on a saved fixture under
  `tests/fixtures/`, not on a hand-built string: the fixture is the evidence of
  what the site sends.
- A test asserts behaviour, not implementation trivia. Fixtures and tests must
  run offline: no network, no `/media`, no live cookies, no Transmission.
- A failing test is a result. Fix the code or the expectation with a reason —
  never loosen the assertion to go green.

## Portability

- No absolute machine paths in any tracked file: not in code, not in skills, not
  in configs. Paths come from `rnext.workspace`, from `Path.home()`, or from a
  CLI flag.
- `config/workspace.json` is local and untracked; `config/workspace.example.json`
  is the portable template.
- Skill files are YAML-frontmatter Markdown: quote any frontmatter value that
  contains a colon (`description: "a: b"`), or the skill loader rejects the file.
