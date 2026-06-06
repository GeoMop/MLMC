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

### Goal 3: Saltelli Schema Simulation And Sobol Quantities

Intent: implement an MLMC-compatible sensitivity-analysis layer where one MLMC
sample is one full Saltelli row. For `N` uncertain input parameters, the row
contains `2 * (N + 1)` forward model evaluations:

- `A`
- `AB_i` for each parameter `i`
- `BA_i` for each parameter `i`
- `B`

Each row element is itself a paired MLMC forward-model evaluation:
`(fine_result, coarse_result)`. The wrapper simulation should therefore call the
wrapped forward simulation once for each Saltelli term on the fine level and,
for levels above zero, once for each corresponding coarse-level term using the
same parameter vector.

Important interface observation:

- Current `Simulation.calculate(config_dict, seed)` receives only a deterministic
  seed, not the MLMC `sample_id`.
- External QMC/OpenTurns Saltelli row generation cannot be represented safely as
  "derive parameters from seed" if we need the provider to allocate consecutive
  row blocks by requested sample count.
- Goal 3 should therefore add a small optional scheduling hook before samples
  are submitted, so a simulation can reserve input rows for a batch of MLMC
  sample ids and persist the mapping used by workers.

Proposed modules:

- `mlmc/sim/saltelli_simulation.py`
  - `SaltelliRowProvider`
  - `SaltelliRow`
  - `SaltelliSchema`
  - `SaltelliSchemaSimulation`
- `mlmc/quantity/sobol.py`
  - functions/classes constructing derived quantities for Sobol numerators,
    denominators, indices, and level variance diagnostics.
- `test/test_saltelli_simulation.py`
  - local `OneProcessPool` tests only.
  AGENT: plan also unit test oth the sobol, the mean should be abstracted so that the estimators could be
  verified with simple MC mean.
  
Data model:

- Use `attrs` for structured containers:
  - `SaltelliSchema`
    - `parameter_names: list[str]`
      AGENT: no need to know param names (beside error messages possibly)
      So comment here how doyou want to use them.
    - `term_names: list[str]`, ordered as `["A", "AB_0", ..., "AB_N-1",
      "BA_0", ..., "BA_N-1", "B"]`
      AGENT: no point in generating thes names (beside __repr__).Document here how do you want to use them.
      We only need to represent the indexing, that is best done by a 2D matrix: 
      A_mask[i_saltelli, i_param] = 1 if "the term i_saltelli uses for param i_param the matrix A" else 0
      This mask should be constructed and then we only need human readable indices: e.g. A0, AB[i], BA[i], B0
    - `n_parameters`
    - `n_terms = 2 * (n_parameters + 1)`
    - output coordinates: `x`, `y`, `z`, `times`
      AGENT: not important the output array should be flattened before calculations, so work with single generic dimension of the forward model ouput
  - `SaltelliRow`
    - `row_id`
    - `A: np.ndarray`
    - `B: np.ndarray`
    - method or property producing all Saltelli term parameter vectors in the
      schema order.
  - `SaltelliRowProvider`
    - wraps an external row generator function.
    - public method `reserve(n_rows) -> list[SaltelliRow]`.
    - the external function gets only the requested number of rows each time,
      so QMC/OpenTurns can manage its own sequence state.
    AGENT: are QMC generators usefull in this case? What is better: single QMC sequence randomly split between levels r one sequence per level?
    In both cases that is hard to manage localy in each simulation so we may need to generalize the Simulation API and have some Simulation
    global (master) planning, that could prepare sample inputs richer than seed only. For now make a parameter matrices A and B 
    once per level simulation instance, let level simulations crate the sample input vectors that will be passed to the workers (easy for Dask, some small rafactoring for PBS pool).
    
    
Scheduling-row allocation:
- Worker calculation uses the whole input vector scheduled on master.

`SaltelliSchemaSimulation` behavior:

- Constructor accepts:
  - wrapped simulation/factory implementing the existing `Simulation` API;
  - row block external function + a method taking the block rows one by one; creating ne block once necessary
  
  - The the SaltelliSchemaSimulation.calculate simply calls the forward impl for all AB Saltelli combinations.
    Store the result into na array, flatten -> return through Dask or PBS  (but you can assume Dask only for this branch)
  - coordinate metadata for the output grid/time axes; These are fixed by the Simulation LEvel so we can reconstruct the xArray on the master is needed.

- `calculate(...)` for one MLMC sample:
  - loads row `A` and `B`;
  - builds term parameter vectors in schema order:
    `A`, all `AB_i`, all `BA_i`, `B`;
  - for each term, evaluates wrapped fine model and wrapped coarse model using
    the corresponding fine/coarse level configs;
  - stacks fine term outputs into an array shaped like:
    `(i_saltelli, x, y, z, time)` or the agreed flattened equivalent;
  - stacks coarse term outputs with the same shape;
  - returns `(fine_flat, coarse_flat)` compatible with `SampleStorage`.
- For level zero, coarse output remains the existing MLMC zero baseline,
  but still has the full Saltelli term shape.
- Result size and shape checks should remain strict because one row may be
  large.

Quantity specification:

- The wrapper should provide a specific result format representing an xarray-like
  field with dimensions:
  - `i_saltelli`
  - `x`
  - `y`
  - `z`
  - `time`
- Current `QuantitySpec` supports `shape`, `times`, and `locations`, not named
  arbitrary dimensions. Goal 3 should avoid a broad storage rewrite by using a
  conservative mapping first:
  - `shape = (n_saltelli,)` This is orthogonal to the times and locations
  - `times = output_times`
  - `locations`  structured grid  for x,y,z
    quantity machinery requires locations.
- Add helper metadata on `SaltelliSchema` to reconstruct an `xarray.DataArray`
  from flattened quantity samples after loading from storage.
  Once we map to QuantitySpec it goes into HDF5 store and we should process it trhough Quanity, so current approach is not compatible with xarray
  and ther is no point in converting to xarray, only as part of postprocessing.
  
- If named-dimensional result metadata is needed by production SA output, record
  it as a future storage/QuantitySpec extension rather than overloading
  `QuantitySpec` silently.

Sobol derived quantities:

- Add functions in `mlmc/quantity/sobol.py` that operate on the root Saltelli
  quantity and return lazy `Quantity` objects whose means can be estimated with
  existing `qe.estimate_mean()` / `Estimate` machinery.
- Use Saltelli/Jansen-compatible estimators consistently. Choose one formula set
  and document it in code/tests. Candidate formulas using row-wise output arrays:
  - denominator variance from `A` and `B`, preferably centered across both.
  - first-order numerator per parameter from `B * (AB_i - A)` or equivalent
    Saltelli 2010 form.
  - total-order numerator per parameter from `0.5 * (A - AB_i) ** 2` or Jansen
    equivalent.
  - second-order numerator from `BA_i * AB_j - A * B` or the selected Saltelli
    second-order formula, with diagonal omitted or set to zero.
- Provide derived quantities for:
  - first-order numerator field `S1_num[i_param, x, y, z, time]`;
  - total-order numerator field `ST_num[i_param, x, y, z, time]`;
  - second-order numerator field `S2_num[i_param, j_param, x, y, z, time]`;
  - variance denominator field `V[x, y, z, time]`.
  - AGENT: all variance estimators should first estimate mean to form the (Y_i - mean_Y)^2
- Final index calculation divides MLMC mean-estimated numerators by the
  MLMC mean-estimated denominator:
  - `S1 = E_mlmc[S1_num] / E_mlmc[V]`
  - `ST = E_mlmc[ST_num] / E_mlmc[V]`
  - `S2 = E_mlmc[S2_num] / E_mlmc[V]`
- Keep ratio computation outside the per-sample quantity operation unless the
  chosen estimator requires otherwise; this avoids estimating the mean of a
  ratio instead of a ratio of means.

  
Variance decrease diagnostics:

- For every derived quantity expose level variances
  from `QuantityMean.l_vars` / `Estimate.estimate_diff_vars()`.
- Tests should plot the variance diagnostic plots.
  the synthetic problem for each quantity
  - base Saltelli row quantity differences;
  - first-order numerator;
  - total-order numerator;
  - second-order numerator;
  - denominator.
- Add helper assertions or diagnostic function, e.g.
  `assert_level_variance_decreases(quantity, sample_storage, tolerance=...)`,
  local to tests first. Promote to library only if it becomes generally useful.

Testing plan:
- Build a deterministic local analytic forward model with known Sobol indices.
  Use only `OneProcessPool` and in-memory or HDF storage.
- Suggested model:
  - independent parameters `X_i ~ U(0, 1)` supplied by a deterministic local
    row provider;
  - scalar/grid output with known additive and interaction components, e.g.
    `Y = a1 * X1 + a2 * X2 + a12 * X1 * X2 + c`;
  - optional spatial/time scaling factor so output is a small `(x, y, z, time)`
    field while expected indices remain analytically known or easy to compute
    by high-accuracy reference Monte Carlo.
- Wrapped fine/coarse model:
  - fine output = exact model plus level-dependent deterministic bias/noise;
  - coarse output = same model with coarser bias;
  - level differences should have decreasing variance as level parameter
    decreases.
- Tests:
  1. Row provider receives requested batch sizes and returns deterministic
     consecutive Saltelli rows.
  2. `SaltelliSchema` term ordering and shape are exactly
     `2 * (N + 1)`.
  3. `SaltelliSchemaSimulation.result_format()` reports the expected flattened
     xarray-compatible shape.
  4. One scheduled MLMC sample produces fine/coarse arrays containing all
     Saltelli terms in the expected order.
  5. A small local MLMC run estimates first, total, and second-order Sobol
     indices within tolerances against analytic/reference values.
  6. Derived numerator/denominator level variances decrease across levels.
  7. Restart/workspace row mapping is tested if the row mapping is file-backed.

Implementation order:

1. Add `SaltelliSchema` and row-provider abstractions with isolated unit tests.
2. Add the optional sample-preparation hook in `Sampler.schedule_samples()` and
   a regression test proving ordinary simulations are unchanged.
3. Add optional sample-id-aware calculation path in `SamplingPool.calculate_sample()`
   and tests for both old and new simulation calculate APIs.
4. Implement `SaltelliSchemaSimulation` around a small test forward simulation.
5. Implement result-format/xarray reconstruction helpers.
6. Implement Sobol derived quantity functions and ratio-of-means estimation
   helper.
7. Add the full local sampler integration test.
8. Only after local tests pass, connect `sensitivity_sampling.py` to the new
   wrapper if the project-specific transport dependencies are available.

Verification plan:

- Targeted compile:
  `python3 -m py_compile mlmc/sim/saltelli_simulation.py mlmc/quantity/sobol.py
  mlmc/sampler.py mlmc/sampling_pool.py`.
- Focused tests:
  `python3 -m pytest -c test/pytest.ini test/test_saltelli_simulation.py -vv`.
- Existing regressions touched by hooks:
  `python3 -m pytest -c test/pytest.ini test/test_sampler.py
  test/test_sampling_pools.py test/test_sampling_pool_dask.py -vv`.
- Broader non-PBS run if the hook touches shared scheduling behavior:
  `python3 -m pytest -c test/pytest.ini test -m "not pbs"`.
- Full `tox` before PR-ready state if Goal 3 implementation is completed.

Open design questions:

- Confirm exact second-order Saltelli estimator formula to implement and test.
- Decide whether the row mapping persistence format should be JSON/NPZ/Pickle.
  NPZ is attractive for numeric A/B rows; Pickle is simplest for arbitrary
  provider metadata but less transparent.
- Decide whether xarray should become a formal package dependency or whether
  xarray reconstruction should remain an optional helper used by
  `sensitivity_sampling.py`.
- Decide how much named-dimension metadata belongs in `QuantitySpec` now versus
  the planned 2.x storage/interface redesign.


## AGENT Log

- `2026-06-06`: Planned Goal 3 Saltelli/Sobol implementation. The plan covers
  a `SaltelliSchemaSimulation` wrapping an existing forward simulation, external
  batch row generation for A/B Saltelli matrices, a small scheduler hook to
  reserve row mappings for sample ids, xarray-compatible result shape metadata,
  lazy Sobol numerator/denominator quantities estimated through the MLMC mean
  estimator, variance-decrease diagnostics across levels, and local-only tests.
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
- `2026-06-06`: Goal 3 needs a sample-id-aware preparation/calculation path.
  The existing simulation API only passes `config_dict` and `seed` into
  `calculate()`, which is not sufficient for an external QMC/OpenTurns row
  provider that allocates consecutive Saltelli rows by requested batch size.
- `2026-06-05`: Legacy/external fixture status is only partially classified.
  Treat `test/01_cond_field`, `test/02_conc`, and `test/fractures` as
  repository examples and legacy/integration fixtures, but do not assume they
  are part of routine tox/CI verification unless current tests prove that.
