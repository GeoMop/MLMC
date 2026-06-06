# PLAN

## Current Goals

Goal 1: verify merge of `master` branch and prepare the branch for a pull request.
Goal 2: for a derived branch, implement a Dask-based sampler as an alternative to the PBS sampler.
Goal 3: implement specific simulations and quantities for estimation of Sobol indices.

## Current Repository State

- MLMC is a Python package for Multilevel Monte Carlo uncertainty
  quantification.
- Public package modules live under `mlmc/`.
- Main functional areas:
  - `sampler.py`, `level_simulation.py`, `sampling_pool.py`,
    `sampling_pool_pbs.py` - sample scheduling and execution backends.
  - `sample_storage.py`, `sample_storage_hdf.py`, `tool/hdf5.py` - sample
    persistence and HDF5 storage.
  - `estimator.py`, `moments.py`, `quantity/` - moment estimation, covariance,
    lazy quantities, and post-processing abstractions.
  - `random/` - correlated random field utilities.
  - `plot/` - plotting and diagnostics.
  - `sim/` - simulation interface and synthetic simulation implementation.
  - `tool/` - distributions, PBS/process helpers, Gmsh/Flow123d-related tools,
    and statistical utilities.
- `README.rst` describes package installation, PyPI distribution, Sphinx docs,
  dependencies, and `tox` as the contribution check.
- `docs/source/` contains Sphinx documentation and API pages.
- `test/` contains pytest tests plus integration-like fixtures for Flow/PBS and
  fracture examples.
- `tox.ini` targets Python 3.10 and 3.12 and runs tests excluding marker `pbs`.
- `test/pytest.ini` excludes marker `metacentrum`.
- Several reference planning/instruction files from other projects are present
  as untracked files and were used as style context:
  `AGENTS_HLAVO.md`, `AGENTS_ENDORSE.md`, `AGENTS_RAMPEC.md`,
  `PLAN_HLAVO.md`, `PLAN_ENDORSE.md`, `PLAN_RAMPEC.md`,
  and `python_coding.md`.

## Work Plan

### Goal 1: Verify Merge and Prepare PR

- Inspect current branch and last merge commit.
- Run `tox` to indetify possibly failing tests.
- Run targeted import/compile checks for files changed by the merge.
- Review places marked with TODO: comments during  the last merge. 
  These were conficts that needs a caraful revision and possibly reintroduce 
  some functionality introduced in matster. In particular the while loops 
  in certaing PBS opeartions were introduced in the master, while 
  this branch should have a non-blocking implementation of the similar functionality.
  Verify that.
- Record verification results and any remaining PR blockers in this plan.
- No code changes yet.

1. mlmc/tool/hdf5.py:151: add_level_group() unconditionally creates Levels, but init_header() already creates it. This is now the tox py312 blocker: ValueError: name already exists during test/test_storage.py collection.
2. mlmc/sampling_pool_pbs.py:397: _qstat_pbs_job() still calls .decode() on process.stderr, but PbsCommands already returns decoded strings. Any nonzero qstat status will raise AttributeError instead of handling PBS output.
AGENT: remove wrong decode calls
Resolved: `sampling_pool_pbs.py` now uses decoded `CommandOutput.stderr` directly.

3. mlmc/sampling_pool_pbs.py:408: unknown_job_ids is only initialized inside the qstat-failure branch, then used after successful qstat too. A successful qstat can hit UnboundLocalError.

4. mlmc/sampling_pool_pbs.py:411: the master retry while loops are disabled, matching the non-looping branch direction, but the exception path still sleeps for 30 seconds. That should be checked against the intended non-blocking
  

## AGENT Log

- `2026-06-05`: Continued Goal 1 merge verification on branch `MS_endorse`.
  Current HEAD is `a36d4e2` (`CODEX conditioning.`); the merge under review is
  `118722a` (`origin/master` into `MS_endorse`). `python3 -m py_compile`
  over the conflict-related files fails with `IndentationError` in
  `mlmc/sampling_pool_pbs.py:391`. `tox` required network access for dependency
  installation; after approval, `py310` was skipped because Python 3.10 is not
  installed and `py312` failed during pytest collection with 12 import errors
  caused by missing `import warnings` in `mlmc/sample_storage_hdf.py`.
- `2026-06-05`: Kept `SimpleDistribution`'s documented vector-valued density
  behavior and moved the SciPy scalar callback adaptation into
  `test/test_distribution.py`. The failing distribution target now passes:
  `.tox/py312/bin/python -m pytest -c test/pytest.ini
  test/test_distribution.py::test_pdf_approx_exact_moments -vv`.
- `2026-06-05`: Incorporated user answers into `AGENTS.md` and `PLAN.md`.
  MLMC is now documented as a production scientific library with 1.x
  compatibility, planned 2.x storage/interface changes, targeted pytest for
  routine debugging, and tox as the final PR-level check. Documentation only;
  no tests required.
- `2026-06-06`: Kept HDF5 `result_format` storage heterogeneous by using a
  `result_format` group with one one-row structured dataset per `QuantitySpec`.
  The writer now prepares each `single_format()` result before opening HDF5,
  creates the HDF group/dataset structure first, then writes the records.
  Verified with `.tox/py312/bin/python -m py_compile mlmc/tool/hdf5.py
  mlmc/sample_storage_hdf.py test/test_storage.py` and
  `.tox/py312/bin/python -m pytest -c test/pytest.ini test/test_hdf.py
  test/test_storage.py -vv`.
- `2026-06-05`: Drafted initial MLMC `AGENTS.md` and `PLAN.md` from
  `README.rst`, `docs/source/index.rst`, `tox.ini`, `test/pytest.ini`,
  package layout, and copied reference project instruction files. Documentation
  only; no tests required.

## AGENT Questions And Remarks

- `2026-06-05`: Legacy/external fixture status is only partially classified.
  Treat `test/01_cond_field`, `test/02_conc`, and `test/fractures` as
  repository examples and legacy/integration fixtures, but do not assume they
  are part of routine tox/CI verification unless current tests prove that.
