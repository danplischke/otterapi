# Contributing to otterapi

First off — thank you for taking the time to contribute! 🦦

`otterapi` is a small, friendly tool that generates type-safe Python clients from
OpenAPI documents. Contributions of all kinds are welcome: bug reports, feature
ideas, documentation fixes, and code.

This guide covers everything you need to get a change merged, including a few
repository rules that are **enforced automatically** (see
[Branch rules & requirements](#branch-rules--requirements)).

---

## Table of contents

- [Code of conduct](#code-of-conduct)
- [Ways to contribute](#ways-to-contribute)
- [Development setup](#development-setup)
- [Project layout](#project-layout)
- [Making a change](#making-a-change)
- [Quality checks](#quality-checks)
- [Commit conventions](#commit-conventions)
- [Signing your commits](#signing-your-commits)
- [Opening a pull request](#opening-a-pull-request)
- [Branch rules & requirements](#branch-rules--requirements)
- [Reporting bugs & requesting features](#reporting-bugs--requesting-features)
- [Security issues](#security-issues)

---

## Code of conduct

Please be respectful and constructive. We want otterapi to be a welcoming
project for everyone. Treat maintainers and fellow contributors the way you'd
like to be treated.

## Ways to contribute

- **Report a bug** — open an issue using the *Bug report* template.
- **Request a feature** — open an issue using the *Feature request* template.
- **Improve docs** — typo fixes and clarifications are always appreciated.
- **Write code** — pick up an open issue or propose a change. For anything
  non-trivial, please open an issue first so we can agree on the approach before
  you invest time.

## Development setup

otterapi uses [uv](https://docs.astral.sh/uv/) for dependency management and
[just](https://github.com/casey/just) as a task runner. If you don't have them
yet:

```bash
# uv (macOS / Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# just — see https://github.com/casey/just#installation
brew install just        # macOS
```

Then fork & clone the repo and run the one-shot dev bootstrap:

```bash
git clone https://github.com/<your-username>/otter.git
cd otter
just dev
```

`just dev` installs the package in editable mode with the `dev` dependency group
(everything needed for tests, linting, and type-checking — `pytest`, `ruff`,
`mypy`, `pandas`, `polars`, `pyarrow`, …) and installs the **pre-commit** hooks
so formatting and lint fixes run automatically on every commit.

Run `just` (or `just --list`) at any time to see all available recipes:

```bash
just            # list recipes
just dev        # editable install + pre-commit hooks
just format     # ruff format
just lint       # ruff check --fix
just chore      # format + lint
just test       # run the test suite
just build      # build the wheel/sdist
```

While developing, run the CLI through uv so it uses your workspace source:

```bash
uv run otter --help
uv run otter generate   # runs against your workspace source
```

## Project layout

```
otterapi/
  cli.py            # Typer CLI entrypoints
  config.py         # otter.yml / CodegenConfig schema
  codegen/          # the code generator (AST builders, templates, runtime/)
  openapi/          # OpenAPI 2.0 / 3.0 / 3.1 / 3.2 spec models
  tests/            # pytest suite (+ golden fixtures)
scripts/            # maintenance scripts
```

Two areas worth knowing:

- **`otterapi/tests/fixtures/golden/`** — byte-exact expected codegen output.
  These files are intentionally excluded from linting/formatting; if a change
  alters generated code, update the golden fixtures in the same PR.

## Making a change

1. Create a topic branch off `main`:
   ```bash
   git switch -c fix/short-description
   ```
2. Make your change, keeping it focused — one logical change per PR.
3. Add or update tests for the behaviour you touched.
4. Run the [quality checks](#quality-checks) locally until they're green.
5. Commit with a [signed](#signing-your-commits), conventional message.

## Quality checks

Most of this runs automatically via the pre-commit hooks installed by
`just dev`, but you can run everything by hand before pushing — CI runs the same
checks, and passing them locally saves a round-trip:

```bash
just chore                # ruff format + ruff check --fix + mypy
just test                 # run the test suite
```

See the [`.justfile`](.justfile) (or run `just --list`) for the full set of
recipes and the exact commands they run.

Notes:

- **Formatting:** single quotes for inline strings, double quotes for
  multiline — this is enforced by ruff, so just run `just format`.
- **Complexity:** functions are capped at a McCabe complexity of 15. If ruff
  flags a function, extract a helper rather than suppressing the rule.
- **Dead code:** `just deadcode` (runs vulture) catches unused symbols. If a
  finding is a false positive (e.g. a Pydantic validator), add it to
  `vulture_whitelist.py` instead of deleting live code.

## Commit conventions

We use [Conventional Commits](https://www.conventionalcommits.org/). The prefix
helps readers (and changelogs) understand intent at a glance:

```
feat: add cursor pagination opt-out for page-size-less APIs
fix: collapse duplicate status codes into a single response type
docs: clarify async client naming in the README
test: cover lenient x-* preservation on security schemes
refactor: extract arrow-type helpers from _python_type_to_arrow
chore: bump ruff to 0.14.2
```

Keep the subject line in the imperative mood and under ~72 characters. Add a body
when the *why* isn't obvious from the diff.

## Signing your commits

**Commit signature verification is required** on this repository — unsigned
commits will be rejected when you open a PR. Set up signing once:

```bash
# Using SSH (simplest if you already push over SSH)
git config --global gpg.format ssh
git config --global user.signingkey ~/.ssh/id_ed25519.pub
git config --global commit.gpgsign true
```

Then add the key to GitHub as a **Signing key** (Settings → SSH and GPG keys).
GPG signing works too — see
[GitHub's guide](https://docs.github.com/authentication/managing-commit-signature-verification).

If you've already made unsigned commits, re-sign them with:

```bash
git rebase --exec 'git commit --amend --no-edit -S' -i main
```

## Opening a pull request

1. Push your branch and open a PR against `main`.
2. Fill in what changed and why; link any related issue (`Fixes #123`).
3. Make sure CI (lint, format, type-check, tests) is green.
4. A maintainer will review. Address feedback by pushing new commits — please
   don't force-push while a review is in progress unless asked, as it can drop
   review context.

Every PR is squash/merge-friendly; keep the branch history reasonable and we'll
handle the final merge.

## Branch rules & requirements

The `main` branch is protected by a ruleset. For your PR to be mergeable it must:

- ✅ Have **signed, verified commits** (see [above](#signing-your-commits)).
- ✅ Receive at least **one approving review** from a code owner.
- ✅ Not require a force-push over the base after approval.
- ✅ Pass the automated quality and code-scanning checks.

Direct pushes to `main`, branch deletion, and force-pushes are blocked — always
work on a topic branch and open a PR.

## Reporting bugs & requesting features

Please use the issue templates:

- **Bug report** — include the OpenAPI spec (or a minimal reproduction), the
  `otter` command you ran, the generated output or traceback, and your
  `otterapi` / Python version.
- **Feature request** — describe the problem you're trying to solve, not just the
  solution you have in mind.

## Security issues

Please **do not** open a public issue for security vulnerabilities. See
[`SECURITY.md`](SECURITY.md), or report privately via GitHub's
[private vulnerability reporting](https://github.com/danplischke/otterapi/security/advisories/new).

---

Thanks again for contributing — happy generating! 🦦
