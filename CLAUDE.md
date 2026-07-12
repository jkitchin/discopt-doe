# discopt-doe — developer & release guide

Design-of-experiments plugin for the `discopt` modeling language. This distribution
packages **only** the `discopt.doe` PEP 420 namespace subpackage (never a top-level
`discopt`), so it merges into the base `discopt` namespace at import time.

## Development

Everything runs through **uv** (the lockfile `uv.lock` is committed; `.python-version` is 3.12).

After cloning, install the deps and the git pre-commit hook once:

```bash
uv sync --all-extras
uv run pre-commit install     # runs ruff + hygiene hooks on every commit
```

| Task | Command | Make target |
| --- | --- | --- |
| Install (with all extras) | `uv sync --all-extras` | — |
| Run tests (fast set) | `uv run pytest tests/ -q` | — |
| Run tests incl. slow | `uv run pytest tests/ -m "slow or not slow"` | `make test` |
| Lint | `uv run ruff check .` | `make lint` |
| Format check | `uv run ruff format --check .` | `make lint` |
| Build the docs | `uv run jupyter-book build docs/ -W` | `make docs` |
| Build artifacts | `rm -rf dist && uv build` | `make build` |
| Check artifacts | `uvx twine check dist/*` | `make check` |
| All pre-flight checks | — | `make preflight` |

Notes:
- The default pytest `addopts` is `-m 'not slow'`, so **slow tests are skipped unless you
  ask for them**. A release must run them (`make test`).
- Docs build with `-W` (warnings-as-errors). Notebooks are pre-executed and committed
  (`execute_notebooks: "off"`); regenerate them with `scripts/build_*_notebook.py` when the
  code they exercise changes.
- Lint/format is **ruff** only (line-length 100, target py310). It runs via
  **pre-commit** (`.pre-commit-config.yaml`); `docs/notebooks/` is excluded (see
  `[tool.ruff] extend-exclude` in `pyproject.toml`).
- **The same hooks run in CI** (the `lint` job in `.github/workflows/ci.yml` calls
  `uvx pre-commit run --all-files`), so local and CI enforcement never drift. Keep the ruff
  `rev` in `.pre-commit-config.yaml` in sync with the ruff version uv resolves.
- Some numerical tests (constrained optimal design via SLSQP) are BLAS/LAPACK-sensitive;
  a green **Linux CI** run is the authoritative gate. If a solver test fails only locally
  on macOS, reproduce it in CI before treating it as a real regression.

## Release checklist

Work top to bottom. The version is a **single source of truth**: `version = "..."` on
line 7 of `pyproject.toml` — there is no `__version__` to keep in sync.

### 0. Prerequisite gate — dependencies must be PyPI-resolvable ⛔

PyPI rejects any distribution that depends on a git/URL source, so every dependency must
resolve from a package index. (This previously blocked releases: `discopt>=0.6` was pinned
to a git rev via `[tool.uv.sources]` until `discopt 0.6.0` shipped to PyPI — that pin has
since been removed.)

- [ ] All runtime dependencies are published on PyPI at the required versions.
- [ ] No `[tool.uv.sources]` git/URL overrides for runtime deps remain in `pyproject.toml`.
- [ ] `grep -rn "git+" pyproject.toml uv.lock` returns nothing for runtime dependencies.
- [ ] `README.md` install instructions match reality (plain `pip install discopt-doe` only
      once this package itself is published).

### 1. Pre-flight — code health

- [ ] On `main`, clean working tree, synced with remote (`git status`, `git pull`).
- [ ] Latest CI run on `main` is green (`gh run list --branch main --limit 1`).
- [ ] Full test suite passes locally **including slow tests**: `make test`
      (= `uv run pytest tests/ -m "slow or not slow"`).
- [ ] No-extras "core" install still degrades gracefully (mirrors the CI `core` job):
      `uv sync --python 3.12 && uv run pytest tests/ -q`.
- [ ] Lint + format clean: `make lint`.
- [ ] Docs build clean: `make docs`.
- [ ] If code feeding the notebooks changed, regenerate them
      (`uv run python scripts/build_*_notebook.py`) and re-commit the executed notebooks.

### 2. Version & metadata

- [ ] Bump `version` in `pyproject.toml` (line 7) per [semver](https://semver.org).
- [ ] Update `CHANGELOG.md`: move `[Unreleased]` entries under the new version + today's date.
- [ ] Re-check `README.md` (install steps, `[gui]`/`[ml]` extras, `discopt-doe-install-skill`).
- [ ] Re-check `pyproject.toml` metadata: `description`, `license`, `authors`,
      `requires-python`, and consider adding `[project.urls]` (Homepage/Docs/Issues) — they
      surface on the PyPI project page.

### 3. Build & verify artifacts

- [ ] `make build` (`rm -rf dist && uv build`) — a stale gitignored `dist/` may exist, so
      always wipe first. Produces an sdist + a wheel.
- [ ] `make check` (`uvx twine check dist/*`) passes.
- [ ] Inspect the wheel: `python -m zipfile -l dist/discopt_doe-*.whl` — it must contain
      `discopt/doe/**` **only** (no top-level `discopt/__init__.py`, which would shadow the
      base package).
- [ ] Smoke test in a clean venv:
      `python -m venv /tmp/dd && /tmp/dd/bin/pip install dist/discopt_doe-*.whl`
      then `/tmp/dd/bin/python -c "import discopt.doe"`, run `discopt-doe-install-skill`,
      and exercise `discopt doe --help`.

### 4. Tag & release

- [ ] Commit the version bump + CHANGELOG (`git commit`), push to `main` (or via PR).
- [ ] Annotated tag: `git tag -a vX.Y.Z -m "vX.Y.Z"` then `git push origin vX.Y.Z`.
- [ ] Create the GitHub release — this **triggers the `publish.yml` workflow** which
      uploads to PyPI:
      `gh release create vX.Y.Z --title vX.Y.Z --notes-file <notes.md>`.
      (Always use `--notes-file`; never inline `--notes` with prose.)

### 5. Post-release verification

- [ ] The **publish** workflow succeeded (Actions tab) and PyPI shows the new version.
- [ ] `pip install "discopt-doe==X.Y.Z"` works in a fresh venv.
- [ ] GitHub Pages docs updated (the `deploy-book` workflow ran on the push to `main`).
- [ ] (Optional) announce the release; open a follow-up to bump to the next dev version.

## PyPI trusted-publisher setup (one-time)

`publish.yml` uses **OIDC trusted publishing** — no API tokens are stored anywhere. The four
repo/workflow values plus the environment must match `publish.yml` exactly, or PyPI refuses
the upload with *"no corresponding publisher (Publisher with matching claims was not found)"*.

### On PyPI

1. Log in at https://pypi.org → account menu → **Publishing**
   (https://pypi.org/manage/account/publishing/).
2. Add a publisher (a **pending publisher** if the project does not exist yet — PyPI creates
   it on first upload) with exactly:
   - **PyPI Project Name:** `discopt-doe`
   - **Owner:** `jkitchin`
   - **Repository name:** `discopt-doe`
   - **Workflow name:** `publish.yml` (filename only, not a path)
   - **Environment name:** `pypi`
3. Click **Add**. No token is generated.

### On GitHub (the environment side)

The workflow references `environment: pypi`, so it must exist:

1. Repo → **Settings → Environments → New environment** → name it `pypi`.
2. (Recommended) add yourself under **Required reviewers** so a publish pauses for a
   one-click approval before uploading. No secrets to add — OIDC handles auth.
