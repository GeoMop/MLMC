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


  
### Goal 2: Dask Sampling Pool For Sensitivity Sampling

Intent: implement a Dask-backed `SamplingPool` alternative that works with the
existing `Sampler` scheduling loop and can be used from the provided
`sensitivity_sampling.py` script. The Dask client should be supplied by the
caller, e.g. `SamplingPoolDask(client=client, work_dir=..., debug=...)`.

Design direction:

- Keep the MLMC `Sampler` as the master scheduler. It already supports
  asynchronous, iterative sample-count updates through `schedule_samples()`,
  `ask_sampling_pool_for_samples()`, and `process_adding_samples()`.
- Add a new Dask pool implementation, most likely in
  `mlmc/sampling_pool_dask.py`, implementing the existing `SamplingPool`
  interface:
  - `schedule_sample(sample_id, level_sim)` submits one future with
    `client.submit(...)` and records future metadata locally.
  - `get_finished()` collects only completed futures and returns the same tuple
    shape as other pools:
    `(successful_samples, failed_samples, n_running, n_ops)`.
  - `have_permanent_samples(sample_ids)` initially returns `False`, matching
    local pools, unless restart/recovery is explicitly added later.
    AGENT: Support ofr restart of the sampling is must for large sample sizes. 
    So desing a way how to implement that with Dask.
    Resolved: `SamplingPoolDask` persists per-level simulation metadata for
    workspace simulations and uses deterministic Dask task keys and sample
    seeds. On restart, construct the pool with `clean=False`; unfinished stored
    sample ids are submitted again and the worker task receives only the sample
    id, output directory, and seed, then loads the persisted level metadata.
    
- Do not use one large `client.map(...); client.gather(...)` for MLMC runs.
  That pattern waits for a fixed batch and does not fit the adaptive algorithm,
  where estimates are updated while previous futures are still running and new
  samples may be scheduled per level.
- Use Dask communication/result transport for sample results. Workers should
  return `(sample_id, result, err_msg, running_time)` through their futures,
  and the master should write successful/failed samples to `SampleStorage`
  through the existing `Sampler._store_samples()` path.
  AGENT: that could work but be carfull, result for SA will idealy be an xarray object (of about 4 dimensions), and could be quite large definitely tenths of MB
  Resolved: completed futures are released immediately after the master stores
  each result. Large SA result persistence still belongs in the simulation or
  sample-storage design; the Dask pool does not keep completed futures alive.
  
- Reuse `SamplingPool.calculate_sample()` as the worker function to preserve
  deterministic seeding, result-format checks, workspace preparation, and
  exception-to-error-message behavior.
- Track future-to-level metadata on the master, e.g.
  `{future: level_sim}` or `{future: level_id}`, so completed results can be
  partitioned by level and runtime can be accumulated into `n_ops`.
- Prefer Dask `as_completed` or future status polling in `get_finished()`.
  `get_finished()` must be non-blocking or bounded by the outer `Sampler`
  timeout; it should not wait for all scheduled futures.
- Preserve the existing output-directory behavior:
  - if `work_dir` is set, use the `SamplingPool` output directory layout;
  - if `level_sim.need_sample_workspace` is true, each Dask worker must execute
    with a sample-specific workspace prepared by `calculate_sample()`;
  - successful/failed directory moving should happen on the master after the
    future result is received, similarly to `OneProcessPool._process_result()`,
    unless worker-local scratch paths make that impossible.
- Keep Dask as an optional dependency. Avoid importing `dask.distributed` from
  `mlmc/__init__.py` unless packaging dependencies are updated accordingly.
  Import it only in the Dask pool module or in the script that constructs the
  client.
  AGENT: I have updated the environment, so dask and dask.distributed should be available. You just have to add the optional dependency into setup.py.
  Resolved: `setup.py` now defines the `dask` optional extra with `dask` and
  `distributed`; tox test dependencies include both packages.

- Update `sensitivity_sampling.py` to construct/use the MLMC `Sampler` with the
  Dask pool where the current code uses `client.map(single_sample, ...)`, once
  the sensitivity simulation is represented through the MLMC `Simulation`
  interface.
   You can suggest that, but main point is to test sampling_pool_dask be separate unit test.
  
  
Implementation steps:

1. Add `SamplingPoolDask` with constructor parameters
   `client`, `work_dir=None`, `debug=False`, and possibly an optional
   `submit_kwargs` dict if needed by the existing cluster setup.
2. Implement future submission with deterministic seed calculation on the
   master and `pure=False`, so repeated sample ids are not memoized by Dask.
   AGENT: pure =True could be used, as the sample calculation is deteministic using reproducible seeding.
   Resolved: `SamplingPoolDask` submits deterministic-key futures with
   `pure=True`.
   
3. Implement completed-future collection:
   - gather futures whose status is `finished` or `error`;
   - call `future.result()` only for those futures;
   - convert Dask worker exceptions into failed sample entries if they escape
     `calculate_sample()`;
   - release/remove collected futures to avoid retaining results in cluster
     memory.
4. Factor or reuse local pool result processing so success/failure queues,
   runtime accumulation, and sample directory cleanup are consistent across
   `OneProcessPool`, `ProcessPool`, and `SamplingPoolDask`.
    AGENT: not clear to me what you mean by this
    Resolved: `SamplingPoolDask` subclasses `OneProcessPool` and reuses its
    `_process_result()` and `get_finished()` queue conversion, avoiding a
    separate refactor.
    
5. Add focused tests using a local Dask cluster/client and synthetic simulation:
   - initial scheduling stores all samples;
   - repeated polling returns partial completion without blocking for the full
     batch;
   - `process_adding_samples()` can schedule more work while previous futures
     are still running;
   - failed samples are reported through `failed_samples`.
   
6. Add a minimal integration path in `sensitivity_sampling.py`:
   - keep `Client(scheduler)` creation in the script;
   - pass that client into `SamplingPoolDask`;
   - avoid `client.map(...); client.gather(...)` for adaptive MLMC sampling.
7. Update documentation/API references only after the public import path is
   chosen.

Verification plan:

- Targeted compile check:
  `python -m py_compile mlmc/sampling_pool.py mlmc/sampling_pool_dask.py mlmc/sampler.py`.
- Targeted tests for the new pool:
  `python -m pytest -c test/pytest.ini test/test_sampling_pool_dask.py -vv`.
- Existing sampler regression:
  `python -m pytest -c test/pytest.ini test/test_sampler.py test/test_sampling_pools.py -vv`.
- If sensitivity integration is changed, run the smallest available
  `sensitivity_sampling.py` local/Dask smoke command documented by its current
  config. If the required external `endorse`, `chodby_trans`, or Flow123d
  environment is unavailable, record that as skipped verification.


## AGENT Log

- `2026-06-06`: Implemented Goal 2 Dask sampler backend. Added
  `mlmc/sampling_pool_dask.py` with a `SamplingPoolDask` accepting an existing
  Dask `Client`, submitting one deterministic-key future per MLMC sample,
  polling only completed futures in `get_finished()`, releasing futures after
  storage, and resubmitting unfinished workspace sample ids on sampler restart
  from sample ids plus persisted per-level metadata. Added Dask package
  metadata in `setup.py`, tox test deps, and focused
  `test/test_sampling_pool_dask.py` coverage. Verification passed:
  `python3 -m py_compile mlmc/sampling_pool.py mlmc/sampling_pool_dask.py
  mlmc/sampler.py mlmc/sampling_pool_pbs.py setup.py
  test/test_sampling_pool_dask.py`;
  `timeout 60 .tox/py312/bin/python -m pytest -c test/pytest.ini
  test/test_sampling_pool_dask.py -vv`; and
  `.tox/py312/bin/python -m pytest -c test/pytest.ini test/test_sampler.py
  test/test_sampling_pools.py -vv`.
- `2026-06-06`: Planned Goal 2 Dask sampler work. The intended design is a
  `SamplingPoolDask` backend that accepts an existing Dask `Client`, submits
  one future per MLMC sample, polls completed futures in `get_finished()`, and
  leaves adaptive scheduling in the existing `Sampler` rather than using a
  single blocking `client.map(...); client.gather(...)` batch.
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

- `2026-06-06`: Goal 2 assumes that the Dask client is owned by the caller and
  passed into the pool constructor. `SamplingPoolDask` should not start or stop
  the cluster unless a later requirement explicitly asks for that.
- `2026-06-06`: `sensitivity_sampling.py` currently contains a direct Dask
  `client.map(single_sample, sample_args)` workflow around project-specific
  dependencies outside MLMC. To make it use the MLMC adaptive sampler, the
  transport calculation must be wrapped as an MLMC `Simulation` with a
  documented `result_format()`.
- `2026-06-05`: Legacy/external fixture status is only partially classified.
  Treat `test/01_cond_field`, `test/02_conc`, and `test/fractures` as
  repository examples and legacy/integration fixtures, but do not assume they
  are part of routine tox/CI verification unless current tests prove that.
