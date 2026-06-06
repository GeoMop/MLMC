# CODEX context for MLMC project

## Project Summary

MLMC is a Python library implementing the Multilevel Monte Carlo method for
uncertainty quantification. It provides sample scheduling, persistent sample
storage, moment and covariance estimation, probability density approximation,
diagnostic plotting, and the `Quantity` abstraction for post-processing stored
simulation outputs.

The project was developed in the GeoMop context and is distributed as the
`mlmc` package. Treat it as reusable scientific library code.

Current 1.x.y versions provide a complete implementation maintained with
backward compatibility. Version 2.x is planned to depart from HDF5-based
storage, add a more general interface, and support sensitivity analysis. That
work may introduce backward-incompatible API changes, but the main project
structure should remain stable.

## Project Structure

- `mlmc/` - package source code.
- `mlmc/sim/` - simulation interfaces and synthetic simulation helpers.
- `mlmc/quantity/` - lazy quantity tree, result format specifications, and
  quantity estimation helpers.
- `mlmc/random/` - random/correlated field utilities.
- `mlmc/plot/` - diagnostic and result plotting utilities.
- `mlmc/tool/` - HDF5 wrappers, distributions, PBS/job helpers, Gmsh/Flow123d
  related tooling, and statistical tests.
- `examples/` - runnable examples and tutorial-like scripts.
- `docs/source/` - Sphinx documentation sources.
- `test/` - pytest tests and small simulation fixtures.
- `test/01_cond_field`, `test/02_conc`, `test/fractures` - repository examples
  and legacy/integration-style fixtures. Do not assume they are part of routine
  tox/CI verification unless current tests prove that.

## CODEX Ignore Folders

- `venv`
- `.tox`
- `.pytest_cache`
- `build`
- `dist`
- `*.egg-info`

## Workflow

- The user reviews changes in `git-cola`. Do not commit changes unless
  explicitly asked.
- Before editing, check the repository state and avoid overwriting unrelated
  user changes.
- At the beginning of work, check the request against `AGENTS.md`, `PLAN.md`,
  `README.rst`, and relevant docs/tests.
- Do not ask for confirmation before making requested changes unless the
  required intent cannot be inferred from the repository context.
- Keep each change focused on one function, module concern, or coherent
  refactoring.
- Do not mix planning edits with code implementation unless explicitly
  requested.
- For larger edits, update `PLAN.md` with the intended steps and unresolved
  questions before implementation.
- Put unresolved project questions or inconsistencies in the last section of
  `PLAN.md` under `AGENT Questions And Remarks`.
- Use the `AGENT log` section in `PLAN.md` for concise completed-work records.
- Treat `AGENT` notes in source comments or documentation as direct
  instructions. When resolved, add a short `Resolved:` line after the note and
  let the user remove the note later.
- For documentation-only changes, tests are not required.
- For code changes, run targeted tests first, then broader verification when
  the change affects shared behavior.

## Coding Rules

Include and adapt: `python_coding.md`.

MLMC-specific interpretation:

- Preserve compatibility in the 1.x.y line unless the user explicitly approves
  an API break.
- Prefer `attrs` for new structured data where a data container is needed; use
  the current `attrs` API.
- Preserve documented data shapes and level/sample conventions in storage,
  sampler, estimator, and quantity code.
- Do not add print-based diagnostics to library code; use logging for
  long-running sampling and storage operations.

## Verification

Preferred verification levels:

- Targeted unit test:
  `python -m pytest test/test_<area>.py`
- Full non-PBS pytest run:
  `python -m pytest -c test/pytest.ini test`
- Tox compatibility run:
  `tox`

Notes:

- `tox.ini` currently defines `py310` and `py312` and runs
  `pytest -m "not pbs"`.
- `test/pytest.ini` uses `addopts = -m "not metacentrum"` and defines
  `slow` and `metacentrum` markers.
- Some tests depend on external tools, PBS/Metacentrum behavior, or historical
  simulation fixtures. Prefer targeted tests while developing and record any
  skipped external verification in the final response and `PLAN.md` QaR when
  relevant.
- `tox` is the ultimate local check, but it is slow. Use it after complex
  refactoring or before PR-ready changes. For normal debugging, run targeted
  pytest commands for the affected tests. CI is expected to run tox before
  merge.

## Mandatory Finish Checklist

Before the final response, verify these items explicitly:

- `PLAN.md` has been reviewed for relevant current work.
- Any touched `AGENT` notes have following `Resolved:` lines.
- New unresolved questions or inconsistencies are recorded in `PLAN.md`.
- Required verification commands were run, or the final response states why
  they were not run.
- The final response mentions open `USER:` questions, missed requirements, and
  failed or skipped verification.
