# Build & publish pipeline

Reproduces this project (quantum-calculator) from a fresh machine.

## 1. Environment (PEP 668 requires a venv)

    python3 -m venv .venv
    .venv/bin/pip install numpy scipy pytest matplotlib

## 2. Tests

    .venv/bin/python -m pytest -q        # 72 tests

`pytest.ini` sets `pythonpath = .` and `testpaths = tests`.

## 3. Install GitHub CLI (no sudo)

Download the latest `gh_<ver>_linux_amd64.tar.gz` from
https://github.com/cli/cli/releases, extract, and copy `bin/gh` to
`~/.local/bin/gh`.

## 4. Authenticate

    printf 'Y\n' | gh auth login --web --hostname github.com --git-protocol https

Then open the printed URL, enter the one-time code, and authorize.
(Pipe stdin: the interactive prompts hang under a PTY because gh's survey
library reads /dev/tty in raw mode.)

## 5. Wire git's credential helper

    gh auth setup-git

Without this, `git push` fails with "could not read Username".

## 6. Create the repo and push

    gh repo create <name> --public --source=. --remote=origin --push

Add a LICENSE (e.g. MIT) before pushing if you want open-source terms.

## Gotchas

- numpy 2.x removed `np.trapz` -> use `np.trapezoid`.
- scipy `eigsh(which='SA')` finds bound (most-negative) states; `'SM'`
  (smallest magnitude) misses negative-energy bound states.
- GitHub's license endpoint returns 404 for a few seconds after pushing a
  LICENSE file (detection lag).
