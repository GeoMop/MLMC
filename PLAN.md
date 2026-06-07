# PLAN

## Current Goals

Goal 1: verify merge of `master` branch and prepare the branch for a pull request.
Goal 2: complete; Dask-based sampler implemented, final tests are run by the user.
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
    Resolved: Dask does not expose PBS-like durable jobs in this pool. If the
    Dask master/scheduler/workers are stopped, unfinished samples remain in
    storage and a later sampler run schedules fresh futures. `SamplingPoolDask`
    therefore reports no permanent samples, matching local pools.
    
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


Code Compatibility:
- Review folder and file organization for the pools. Individual pools use
  some pool-specific filesystem operations, which creates duplication and
  compatibility risk. Keep common sample workspace behavior in the base class
  and leave only backend transport files in backend-specific code.

  Overview:
  - `SamplingPool` base creates `work_dir/output`, `work_dir/output/failed`,
    and `work_dir/output/several_successful`; it also owns sample workspace
    creation, common-file copying, successful/failed sample directory moves,
    and cleanup helpers.
    Resolved: yes, the individual sample temporary directory structure is given
    by `SamplingPool`: for sample `Lxx_Syyyyyyy`, the temporary workspace is
    `output/Lxx_Syyyyyyy` when `level_sim.need_sample_workspace` is true. MLMC
    levels themselves do not have separate filesystem directories in the common
    layout; level identity is encoded in the sample id and in storage. PBS adds
    level-specific serialized simulation config files, but that is PBS worker
    transport, not the common sample workspace layout.
    
  - `OneProcessPool`, `ProcessPool`, `ThreadPool`, and `SamplingPoolDask` use
    the base `output` layout and result-moving helpers.
  - `SamplingPoolPBS` uses the base `output` layout plus `work_dir/output/jobs`
    for PBS job scripts, PBS id marker files, per-job YAML result files,
    serialized `PbsJob` process metadata, and scheduled sample metadata.
    PBS also serializes one `level_<id>_simulation_config` file under
    `work_dir/output`.
  - `PbsJob` worker code uses the same base helpers for sample directories and
    success/failure movement, but also writes per-job YAML result files in
    `jobs`.
  
  Suggested reconciliation:
  - Most filesystem interaction is already common in `SamplingPool`; extend
    that base class rather than adding a new filesystem abstraction now.
  - `SamplingPool.__init__()` is responsible for setting `_output_dir`.
    Explicit `work_dir` values must point to an existing directory; otherwise
    construction raises `FileNotFoundError`. If `work_dir` is omitted, the
    constructor falls back to `os.getcwd()`.
  - Keep sample directory creation, common-file copying, successful/failed
    copies, and cleanup in `SamplingPool`.
  - Keep PBS-specific files in PBS code, but centralize PBS path construction
    in helper methods: `jobs`, job scripts, PBS id marker files,
    scheduled/result YAML files, `sample_id_job_id.json`,
    `pbs_process_serialized.txt`, and `level_<id>_simulation_config`.
  - `SamplingPoolDask` should have no direct filesystem interaction beyond
    using the common base helper to obtain `_output_dir` and passing it to
    `SamplingPool.calculate_sample(...)`.

### Goal 3: Saltelli Schema Simulation And Sobol Quantities

Intent: one MLMC sample represents one full Saltelli row. For `N` parameters
the row contains `2 * (N + 1)` forward-model terms: `A`, all `AB_i`, all
`BA_i`, and `B`. Every term is evaluated as a fine/coarse MLMC pair, flattened,
stored through the existing `SampleStorage`, and post-processed through
`Quantity`.

Proposed modules:

- `mlmc/sim/saltelli_simulation.py`
  - `SaltelliSchema`
  - `SaltelliSchemaSimulation`
- `mlmc/quantity/sobol.py`
  - Saltelli term extraction
  - Sobol numerator/denominator quantity builders
  - ratio-of-means index helper
- Tests:
  - `test/test_saltelli_simulation.py`
  - `test/test_sobol_quantity.py`

Core representation:

- `SaltelliSchema` should store only what computation needs:
  - `n_parameters`
  - `n_terms = 2 * (n_parameters + 1)`
  - `A_mask: np.ndarray[bool]` of shape `(n_terms, n_parameters)`, where
    `A_mask[i_term, i_param]` selects matrix `A` for that parameter and
    `False` selects matrix `B`.
  - lightweight human labels such as `A0`, `AB[i]`, `BA[i]`, `B0` for human readable sobol estimate quantities construciton
- The forward-model output is treated as one generic flattened output dimension
  during Sobol calculations. Spatial/time/xarray structure is only metadata for
  reconstructing post-processing output.
- `SaltelliSchemaSimulation` accepts an external matrix-block generator with
  signature `matrix_generator(n_rows: int, n_parameters: int) -> matrix`, with
  shape `(n_rows, n_parameters)` and values in `[0, 1]`. The wrapper calls it
  twice per requested Saltelli row block to obtain `A` and `B`, keeping
  OpenTurns/QMC sequence ownership outside MLMC.
- QMC across adaptive MLMC levels is nontrivial. For Goal 3, generate `A` and
  `B` once per `LevelSimulation` on the master, then let the level simulation
  construct full per-sample input vectors that are passed to workers. Dask is
  the assumed distributed backend for this branch; PBS support can follow after
  the input-vector scheduling interface settles.

Scheduling and simulation API:

- Current workers call `Simulation.calculate(config_dict, seed)`, but Saltelli
  rows need richer sample input than a seed.
- Add a backward-compatible master-side planning hook:
  - `LevelSimulation.prepare_samples(sample_ids)` is a non-abstract method.
  - The default returns `(sample_id, seed)` tuples, preserving seed-based sampling.
  - Saltelli overrides it to return work items such as
    `(sample_id, saltelli_input_vectors)`.
- Add a backward-compatible worker input path:
  - planned sample input vectors are attached to scheduled tasks;
  - `SamplingPool.calculate_sample()` accepts only the fixed
    `(sample_id, sample_input)` tuple shape and calls
    `level_sim._calculate(config, sample_input)`;
  - old `calculate(config, seed)` simulations still work because the default
    level preparation uses the deterministic seed as `sample_input`.
- Worker calculation should receive the whole Saltelli row input vector. It
  should not call the external row generator.

`SaltelliSchemaSimulation` behavior:

- Constructor accepts:
  - wrapped forward simulation/factory;
  - external matrix-block generator; test with simple MC sampler, i.e. independent values from U[0,1] in the matrix.
  - forward-output metadata needed for `QuantitySpec` and optional xarray
    reconstruction.
- `level_instance(...)` delegates fine/coarse level construction to the wrapped
  simulation, then attaches planning metadata and a Saltelli calculate method.
- For one MLMC sample, calculate:
  - construct all Saltelli term vectors using `A_mask`;
  - call the wrapped forward implementation for every term as
    `calculate(forward_config, input_vector)`;
  - stack as `(i_saltelli, output_flat)` for fine and coarse;
  - return flattened arrays compatible with `SampleStorage`.
- Level zero coarse values keep the standard MLMC zero-baseline behavior with
  the same Saltelli/result shape.

Quantity/result format:

- Use existing `QuantitySpec`; do not introduce xarray into storage or `Quantity`
  calculations.
- Map the Saltelli axis to `shape=(n_saltelli,)`. The wrapped model output is
  represented through existing `times` and `locations` where possible, or a
  generic flattened output location when needed.
- xarray conversion is only a post-processing helper on the master after data
  has been estimated through `Quantity`.

Sobol quantities:

- `mlmc/quantity/sobol.py` should build lazy `Quantity` objects from the root
  Saltelli quantity. The mean operation must be abstracted so estimator formulas
  can also be unit-tested with simple MC means outside MLMC.
- Implement and document one consistent estimator family for:
  - first-order indices;
  - total-order indices;
  - second-order indices.
- Variance denominators must use an estimated mean first, then form
  `(Y - mean_Y) ** 2`; do not estimate raw second moment as variance.
- Final Sobol indices are ratios of estimated means:
  - numerator quantities are MLMC-mean estimated;
  - denominator variance quantity is MLMC-mean estimated;
  - index = numerator_mean / denominator_mean.
- Keep ratio computation outside the per-sample quantity operation.

Variance diagnostics:

- For every derived quantity, expose or return level variances from
  `QuantityMean.l_vars` / `Estimate.estimate_diff_vars()`.
- Tests should produce or exercise existing diagnostic variance plots for:
  - base Saltelli row;
  - first-order numerator;
  - total-order numerator;
  - second-order numerator;
  - denominator.

Testing:

- Use only local sampler/pool for Goal 3 tests.
- Build a deterministic analytic test model with known or high-accuracy
  reference Sobol indices, e.g.
  `Y = a1 * X1 + a2 * X2 + a12 * X1 * X2 + c` with independent uniform inputs.
  AGENT: this could be used as basic test of the sobol estimation (using single level MLMC = MC), but not to test MLMC
  since there is not resolution that could be changed through levels.
  Resolved: keep the analytic model for single-level Sobol estimator tests;
  test MLMC variance decrease separately with a level-dependent model.
   
- Add level-dependent fine/coarse perturbations so level-difference variances
  decrease with refinement.
  AGENT: yeah that is an independent test for MLMC already.
  Resolved: keep this as a separate MLMC-level diagnostic test, not as the
  analytic Sobol formula check.
- Unit tests:
  1. `A_mask` creates the exact `A`, `AB_i`, `BA_i`, `B` term matrix.
  2. Row provider is called with expected block sizes and is not called on
     workers.
  3. `SaltelliSchemaSimulation.result_format()` is consistent with
     `(n_saltelli, output_flat)`.
  4. One local sampled row has all fine/coarse Saltelli terms in the expected
     order.
  5. Sobol numerator/denominator formulas pass against simple MC mean.
  6. Local MLMC run estimates first, total, and second-order indices within
     tolerances.
  7. Level variances for derived quantities decrease across levels.

Implementation order:

1. Implement `SaltelliSchema` and `A_mask` tests.
2. Add sample-planning/input-vector hooks while preserving existing simulation
   API behavior.
3. Implement `SaltelliSchemaSimulation` and local row-provider tests.
4. Implement Sobol quantity builders with abstract mean tests.
5. Add local MLMC integration test and variance diagnostics.
6. Connect `sensitivity_sampling.py` only after local tests are stable.

Verification:

- `python3 -m py_compile mlmc/sim/saltelli_simulation.py mlmc/quantity/sobol.py
  mlmc/sampler.py mlmc/sampling_pool.py`
- `python3 -m pytest -c test/pytest.ini test/test_saltelli_simulation.py
  test/test_sobol_quantity.py -vv`
- Regression for touched scheduling/pools:
  `python3 -m pytest -c test/pytest.ini test/test_sampler.py
  test/test_sampling_pools.py test/test_sampling_pool_dask.py -vv`

Open questions:

- Select exact second-order estimator formula.
- Decide how to split or sequence QMC rows across adaptive MLMC levels:
  one global sequence split by level, or one sequence per level.
- Decide persistence format for planned sample inputs if restart support is
  required before the 2.x storage redesign.


## AGENT Log

- `2026-06-07`: Goal 2 is complete, with final project-level tests left for
  user review. Implemented `SamplingPoolDask` as an optional-dependency
  backend that accepts a caller-owned Dask client, submits one deterministic-key
  future per prepared `(sample_id, input_value)` work item, polls completed
  futures without blocking, returns results through Dask futures to the master,
  and releases completed futures after storage. Simplified Dask sample handling
  to match local pools: no Dask-specific permanent sample/restart metadata,
  no backend filesystem behavior beyond passing `_output_dir` into
  `SamplingPool.calculate_sample(...)`, lazy `distributed` import through a
  clear helper called from the constructor, typed/help-string documented public
  methods, and named tuple unpacking instead of tuple indexing. Updated common
  sampler compatibility so `LevelSimulation.prepare_samples()` owns default
  seed/input generation, pools propagate arbitrary sample input, PBS accepts
  prepared sample tuples, and `SamplingPool.__init__()` centralizes `_output_dir`
  fallback plus explicit `work_dir` validation. Verification run during
  implementation passed: `python3 -m py_compile mlmc/sampling_pool.py
  mlmc/sampling_pool_dask.py mlmc/sampling_pool_pbs.py mlmc/tool/pbs_job.py
  test/test_sampling_pool_dask.py test/test_sampling_pools.py`;
  `timeout 60 .tox/py312/bin/python -m pytest -c test/pytest.ini
  test/test_sampling_pool_dask.py -vv`; `.tox/py312/bin/python -m pytest -c
  test/pytest.ini test/test_sampling_pools.py -vv`; and
  `.tox/py312/bin/python -m pytest -c test/pytest.ini test/test_sampler.py
  -vv`.
- `2026-06-06`: Simplified `SaltelliSchemaSimulation` per source notes.
  Removed `SaltelliRowProvider`; the simulation now accepts the
  `matrix_generator(n_rows, n_parameters) -> matrix` callable directly and
  calls it twice for the `A` and `B` Saltelli matrices. Removed the injected
  `parameter_applier`; the wrapper always calls the forward simulation as
  `calculate(forward_config, input_vector)`. Verification passed:
  `python3 -m py_compile mlmc/sampler.py mlmc/sim/saltelli_simulation.py
  test/test_saltelli_simulation.py`; `.tox/py312/bin/python -m pytest -c
  test/pytest.ini test/test_saltelli_simulation.py -vv`; and
  `.tox/py312/bin/python -m pytest -c test/pytest.ini test/test_sampler.py
  test/test_sampling_pools.py -vv`.
- `2026-06-06`: Adapted the Goal 3 Saltelli implementation to source comments.
  `SaltelliRowProvider` now accepts one matrix-block function with signature
  `block_generator(n_rows, n_parameters) -> matrix`, validates shape and
  `[0, 1]` bounds, and calls it twice for `A` and `B`. The default
  `SaltelliSchemaSimulation` path now passes each planned Saltelli input vector
  directly as the second argument to the wrapped forward simulation instead of
  storing per-sample input in the fixed level config; the old config-applier
  path remains available as an optional compatibility adapter. Verification
  passed: `python3 -m py_compile mlmc/level_simulation.py mlmc/sampler.py
  mlmc/sampling_pool.py mlmc/sim/saltelli_simulation.py
  test/test_saltelli_simulation.py`; `.tox/py312/bin/python -m pytest -c
  test/pytest.ini test/test_saltelli_simulation.py -vv`; and
  `.tox/py312/bin/python -m pytest -c test/pytest.ini test/test_sampler.py
  test/test_sampling_pools.py -vv`.
- `2026-06-06`: Followed sampler-plumbing source instructions. Added
  `Simulation.make_level_simulation(...)` for common level finalization, made
  `LevelSimulation.prepare_samples(sample_ids)` the default master planning
  hook, removed the duplicate `_calculate_sample` / `_sample_inputs` execution
  path, moved default seed generation into `LevelSimulation.prepare_samples`,
  and made pools call only `level_sim._calculate(config, sample_input)`.
  Scheduled work items now use the fixed `(sample_id, sample_input)` tuple
  shape; storage still receives plain sample ids. Updated Saltelli to return
  tuple work items, kept Dask as a simple propagator of that work item, and
  routed failed-sample renewal through `prepare_samples(...)`. Verification passed: `python3 -m py_compile
  mlmc/level_simulation.py mlmc/sim/simulation.py mlmc/sampler.py
  mlmc/sampling_pool.py mlmc/sampling_pool_dask.py mlmc/tool/pbs_job.py
  mlmc/sim/saltelli_simulation.py test/test_saltelli_simulation.py
  test/test_sampling_pool_dask.py`; `.tox/py312/bin/python -m pytest -c
  test/pytest.ini test/test_saltelli_simulation.py -vv`;
  `.tox/py312/bin/python -m pytest -c test/pytest.ini test/test_sampler.py
  test/test_sampling_pools.py -vv`; and `timeout 60 .tox/py312/bin/python -m
  pytest -c test/pytest.ini test/test_sampling_pool_dask.py -vv`.
- `2026-06-06`: Documented the new Goal 3 extension points. Added callable
  signatures and usage notes for `SaltelliRowProvider.block_generator`,
  `SaltelliSchemaSimulation.parameter_applier`,
  `SaltelliSchemaSimulation.calculate_sample`, the generated
  `_prepare_samples(sample_ids)` hook, and the optional
  `LevelSimulation._calculate_sample` / `_sample_inputs` path. Verification
  passed: `python3 -m py_compile mlmc/level_simulation.py
  mlmc/sampling_pool.py mlmc/sim/saltelli_simulation.py` and
  `.tox/py312/bin/python -m pytest -c test/pytest.ini
  test/test_saltelli_simulation.py -vv`.
- `2026-06-06`: Started Goal 3 implementation, limited to sampler input-vector
  plumbing and `SaltelliSchemaSimulation`. Added optional `LevelSimulation`
  `_prepare_samples`, `_sample_inputs`, and `_calculate_sample` fields. The
  sampler now calls `_prepare_samples(sample_ids)` once per scheduled level
  batch, and `SamplingPool.calculate_sample()` passes planned sample input to
  simulations that define `_calculate_sample`; old `calculate(config, seed)`
  simulations remain unchanged. Added `mlmc/sim/saltelli_simulation.py` with
  `SaltelliSchema`, `SaltelliRowProvider`, and `SaltelliSchemaSimulation`.
  Added `test/test_saltelli_simulation.py` to verify `A_mask`, propagation of
  planned input vectors, and formation of forward evaluation scenarios through
  a local `ForwardModelSimulation`. Verification passed:
  `python3 -m py_compile mlmc/level_simulation.py mlmc/sampler.py
  mlmc/sampling_pool.py mlmc/sim/saltelli_simulation.py
  test/test_saltelli_simulation.py`;
  `.tox/py312/bin/python -m pytest -c test/pytest.ini
  test/test_saltelli_simulation.py -vv`; and
  `.tox/py312/bin/python -m pytest -c test/pytest.ini test/test_sampler.py
  test/test_sampling_pools.py -vv`.
- `2026-06-06`: Planned Goal 3 Saltelli/Sobol implementation. The plan covers
  a `SaltelliSchemaSimulation` wrapping an existing forward simulation, external
  batch row generation for A/B Saltelli matrices, a small scheduler hook to
  reserve row mappings for sample ids, xarray-compatible result shape metadata,
  lazy Sobol numerator/denominator quantities estimated through the MLMC mean
  estimator, variance-decrease diagnostics across levels, and local-only tests.
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
  AGENT: Confirmed.
- `2026-06-06`: `sensitivity_sampling.py` currently contains a direct Dask
  `client.map(single_sample, sample_args)` workflow around project-specific
  dependencies outside MLMC. To make it use the MLMC adaptive sampler, the
  transport calculation must be wrapped as an MLMC `Simulation` with a
  documented `result_format()`.
  AGENT: Exactly.
- `2026-06-06`: Goal 3 needs a sample-id-aware preparation/calculation path.
  The existing simulation API only passes `config_dict` and `seed` into
  `calculate()`, which is not sufficient for an external QMC/OpenTurns row
  provider that allocates consecutive Saltelli rows by requested batch size.
  AGENT: API change, pass the arbitrary input array produced by the Simulation on the master.
  Resolved: the Saltelli path now returns `(sample_id, input_array)` work items
  from `LevelSimulation.prepare_samples(...)`; `SamplingPool.calculate_sample()`
  passes the input array directly into `level_sim._calculate(...)`.
  Open: for backward compatibility, implement default input generation method in the Simulation base class
  returning the seeds. We unify the implementation generating seeds commonly on master.
- `2026-06-05`: Legacy/external fixture status is only partially classified.
  Treat `test/01_cond_field`, `test/02_conc`, and `test/fractures` as
  repository examples and legacy/integration fixtures, but do not assume they
  are part of routine tox/CI verification unless current tests prove that.
