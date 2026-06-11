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


    
### Goal 3: Saltelli Schema Simulation And Sobol Quantities

Intent: one MLMC sample represents one full Saltelli row. For `N` parameters
the row contains `2 * (N + 1)` forward-model terms: `A`, all `AB_i`, all
`BA_i`, and `B`. Every term is evaluated as a fine/coarse MLMC pair, flattened,
stored through the existing `SampleStorage`, and post-processed through
`Quantity`.

Implemented design reference:

- Scheduled work items use the fixed `(sample_id, sample_input)` shape. Legacy
  seed-based simulations still use deterministic seeds as their default
  `sample_input`. REVIEWED.
- Storage keeps ordinary flattened fine/coarse result arrays and persists
  numeric planned sample inputs for restart/renewal. REVIEWED.

- `mlmc.quantity.sobol.SaltelliSchema` owns the Saltelli term order, masks,
  labels, and `A`/`AB_i`/`BA_i`/`B` term indices.
- `mlmc.sim.saltelli_simulation.SaltelliSchemaSimulation` wraps a forward
  `Simulation`, reserves A/B matrix rows on the master through
  `LevelSimulation.prepare_samples(...)`, and passes complete Saltelli row input
  vectors to workers.
- Sobol post-processing stays in `Quantity`: numerator and denominator
  quantities are MLMC-mean estimated first, then final Sobol indices are formed
  as ratios outside per-sample quantity operations.
- Implemented estimator family: first-order
  `E[(A - mean) * (BA_i - mean)]`, total-order Jansen
  `0.5 * E[(A - AB_i) ** 2]`, and second-order
  `E[(BA_i - mean) * (AB_j - mean)] - first_i - first_j` for `i < j`, all
  divided by the estimated variance denominator.


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

1. Implement `SaltelliSchema` and `A_mask` tests. Done.
2. Add sample-planning/input-vector hooks while preserving existing simulation
   API behavior. Done.
3. Implement `SaltelliSchemaSimulation` and local row-provider tests. Done.
4. Implement Sobol quantity builders with abstract mean tests. Done.
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





## AGENT Log

- `2026-06-11`: Applied the agreed HDF scheduled-sample flat-tail fix. HDF scheduled input splitting now converts each `sample[1:]` directly with `np.asarray(..., dtype=float)` and asserts a 1D tail, then reads stored 1D/2D inputs back as `(sample_id, *tail)` for `SampleStorage.coerce_sample_input()`. Verification passed: `PYTHONPATH=venv/lib/python3.12/site-packages:/home/jb/workspace/MLMC MPLCONFIGDIR=/tmp/mplconfig python3.12 -m pytest -c test/pytest.ini test/test_storage.py test/test_saltelli_simulation.py -vv`.
- `2026-06-11`: Reviewed the Saltelli flattened-tail update and fixed the forwarding path in `SaltelliSchemaSimulation`. The wrapper now converts the incoming tail to a numeric array, reshapes only the Saltelli-owned prefix, and forwards any remaining tail parameters unchanged to the wrapped forward simulation. Updated scheduled-sample coercion/HDF split-load helpers to preserve the new `(sample_id, *tail)` shape while keeping legacy single-input storage working. Added a regression test for forwarding extra parameters through the Saltelli wrapper and updated Saltelli tests to the flattened-tail representation. Verification passed: `PYTHONPATH=venv/lib/python3.12/site-packages:/home/jb/workspace/MLMC MPLCONFIGDIR=/tmp/mplconfig python3.12 -m pytest -c test/pytest.ini test/test_saltelli_simulation.py test/test_storage.py -vv`.
- `2026-06-10`: Fixed the remaining sample-input reconstruction bug in the PBS worker path. `PbsJob.calculate_samples()` now expands stored tails with `(sample_id, *input_value)` before calling `SamplingPool.calculate_sample()`, which avoids re-nesting the input tail as a single list argument. Added `test/test_pbs_job.py::test_pbs_job_reconstructs_flat_sample_input` to cover the flattened-tail contract. Verification pending.
- `2026-06-10`: Replaced `SampleStorageHDF.make_qspec()` with `QuantitySpec` field converters. `QuantitySpec` now normalizes bytes, NumPy arrays, and scalar sequences via `attrs` converters, and HDF loading constructs the spec directly with `QuantitySpec(*res_format[0])`. Added a direct converter regression test alongside the existing HDF roundtrip coverage. Verification passed: `PYTHONPATH=venv/lib/python3.12/site-packages:/home/jb/workspace/MLMC MPLCONFIGDIR=/tmp/mplconfig python3.12 -m pytest -c test/pytest.ini test/test_storage.py -vv`.
- `2026-06-10`: Fixed HDF result-format roundtripping for 1D shapes. `HDF5.single_format()` now sizes the stored `shape` subarray from `len(spec.shape)` instead of hard-coding two entries, and `SampleStorageHDF.make_qspec()` now normalizes loaded shapes and times back to Python scalars. Extended `test/test_storage.py::test_hdf_heterogeneous_result_format` to cover a `QuantitySpec(shape=(14,))` roundtrip and to assert the raw HDF field stores `[14]` rather than `[14, 14]`. Verification passed: `PYTHONPATH=venv/lib/python3.12/site-packages:/home/jb/workspace/MLMC MPLCONFIGDIR=/tmp/mplconfig python3.12 -m pytest -c test/pytest.ini test/test_storage.py -vv`.
- `2026-06-08`: Added `SA_USAGE.md` as auxiliary integration context for
  Sobol sensitivity analysis. It documents the Saltelli row layout,
  `SaltelliSchemaSimulation` forward-simulation contract, matrix generator
  contract, storage/root-quantity selection, `estimate_sobol_indices(...)`
  usage, result diagnostics, formula conventions, and targeted regression
  tests. Documentation-only change; verification was limited to readback and
  stale API-name search.
- `2026-06-08`: Followed Sobol refactoring notes. `SobolIndexEstimate` now
  stores only the already-sliced Saltelli quantities `a`, `b`, `ab`, and `ba`;
  it no longer stores `root` or `schema`. Quantity construction now lives
  directly in property methods, estimated means are cached MLMC properties, and
  the root output mean was renamed to `mean_mlmc`. The factory
  `estimate_sobol_indices(...)` validates the Saltelli axis and builds `ab` and
  `ba` with `Quantity.QArray`, avoiding the current advanced-indexing shape
  issue. Removed the test-only raw Sobol estimator implementation from
  production `sobol.py` and switched tests to SALib as the reference estimator.
  Verification passed: `python3 -m py_compile mlmc/quantity/sobol.py
  test/test_sobol_quantity.py`; `.tox/py312/bin/python -m pytest -c
  test/pytest.ini test/test_sobol_quantity.py -vv`; and
  `.tox/py312/bin/python -m pytest -c test/pytest.ini
  test/test_saltelli_simulation.py test/test_sobol_quantity.py -vv`.
- `2026-06-07`: Resolved current Goal 3 AGENT notes in plan and new Sobol
  sources. Moved `SaltelliSchema` into `mlmc.quantity.sobol`, made Sobol
  functions accept a schema object instead of plain `n_parameters`, kept
  `SobolIndexEstimate` mean diagnostics in internal `_..._mean` fields, and
  added standard-deviation estimates for numerator, denominator, and Sobol
  index ratios. Compacted the implemented Goal 3 design section in `PLAN.md`
  and reviewed `save_global_data`; HDF already rejects incompatible result
  format rewrites, so constructor-based result formats are left as future
  storage-API cleanup. Verification passed: `python3 -m py_compile
  mlmc/quantity/sobol.py mlmc/sim/saltelli_simulation.py
  test/test_sobol_quantity.py test/test_saltelli_simulation.py` and
  `.tox/py312/bin/python -m pytest -c test/pytest.ini
  test/test_saltelli_simulation.py test/test_sobol_quantity.py -vv`.
- `2026-06-07`: Continued Goal 3 by adding `mlmc/quantity/sobol.py`.
  Implemented raw NumPy Saltelli/Jansen estimator samples for first-order,
  total-order, and second-order Sobol indices; lazy `Quantity` builders for
  denominator and all numerator families; and `estimate_sobol_indices(...)`
  that forms ratios after MLMC mean estimation while returning the underlying
  `QuantityMean` diagnostics. Added `test/test_sobol_quantity.py` with an
  analytic two-parameter interaction model and Quantity-backed checks.
  Verification passed: `python3 -m py_compile mlmc/sim/saltelli_simulation.py
  mlmc/quantity/sobol.py mlmc/sampler.py mlmc/sampling_pool.py
  test/test_saltelli_simulation.py test/test_sobol_quantity.py` and
  `.tox/py312/bin/python -m pytest -c test/pytest.ini
  test/test_saltelli_simulation.py test/test_sobol_quantity.py -vv`.
- `2026-06-07`: Fixed scheduled sample persistence for planned sample inputs.
  `Sampler.schedule_samples()` now stores the full prepared
  `(sample_id, sample_input)` work items instead of only sample ids. HDF storage
  keeps scheduled ids in `scheduled` and stores numeric scalar/array inputs in
  a parallel `scheduled_inputs` dataset with shape and dtype fixed by the first
  input per level. `failed_samples()` and HDF unfinished sample queries now
  return stored scheduled work items rather than ids, so
  `Sampler.renew_failed_samples()` no longer regenerates inputs through
  `prepare_samples()`. Verification passed: `python3 -m py_compile
  mlmc/sampler.py mlmc/sample_storage.py mlmc/sample_storage_hdf.py
  mlmc/sampling_pool.py mlmc/sampling_pool_dask.py mlmc/sampling_pool_pbs.py
  mlmc/tool/hdf5.py test/test_storage.py test/test_saltelli_simulation.py
  test/test_sampling_pool_dask.py`; `.tox/py312/bin/python -m pytest -c
  test/pytest.ini test/test_storage.py -vv`; `.tox/py312/bin/python -m pytest
  -c test/pytest.ini test/test_saltelli_simulation.py -vv`;
  `.tox/py312/bin/python -m pytest -c test/pytest.ini test/test_hdf.py -vv`;
  `.tox/py312/bin/python -m pytest -c test/pytest.ini test/test_sampler.py
  test/test_sampling_pools.py -vv`; and `timeout 60 .tox/py312/bin/python -m
  pytest -c test/pytest.ini test/test_sampling_pool_dask.py -vv`.
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
  Resolved: `SamplingPoolDask` requires a caller-owned client and does not
  create, close, or restart Dask clusters.
- `2026-06-06`: `sensitivity_sampling.py` currently contains a direct Dask
  `client.map(single_sample, sample_args)` workflow around project-specific
  dependencies outside MLMC. To make it use the MLMC adaptive sampler, the
  transport calculation must be wrapped as an MLMC `Simulation` with a
  documented `result_format()`.
  AGENT: Exactly.
  Resolved: sensitivity integration remains deferred until a proper MLMC
  `Simulation` wrapper and result format exist.
- `2026-06-06`: Goal 3 needs a sample-id-aware preparation/calculation path.
  The existing simulation API only passes `config_dict` and `seed` into
  `calculate()`, which is not sufficient for an external QMC/OpenTurns row
  provider that allocates consecutive Saltelli rows by requested batch size.
  AGENT: API change, pass the arbitrary input array produced by the Simulation on the master.
  Resolved: the Saltelli path now returns `(sample_id, input_array)` work items
  from `LevelSimulation.prepare_samples(...)`; `SamplingPool.calculate_sample()`
  passes the input array directly into `level_sim._calculate(...)`.
  Resolved: backward-compatible default seed generation is unified on
  `LevelSimulation.prepare_samples(...)`, which returns `(sample_id, seed)`
  work items for legacy simulations.
- `2026-06-05`: Legacy/external fixture status is only partially classified.
  Treat `test/01_cond_field`, `test/02_conc`, and `test/fractures` as
  repository examples and legacy/integration fixtures, but do not assume they
  are part of routine tox/CI verification unless current tests prove that.
