import os.path
import json
import time
import numpy as np
from mlmc.mc_level import Level
from mlmc.logger import Logger


class MLMC:
    """
    Multilevel Monte Carlo method
    """
    def __init__(self, n_levels, sim_factory, step_range, process_options):
        """
        :param n_levels: Number of levels
        :param sim_factory: Object of simulation
        :param step_range: Simulations step range
        :param process_options: Options for processing mlmc samples
                                'output_dir' - directory with sample logs
                                'regen_failed' - bool, if True then failed simulations are generated again
                                'keep_collected' - bool, if True then dirs with finished simulations aren't removed
        """
        # Object of simulation
        self.simulation_factory = sim_factory
        # Array of level objects
        self.levels = []
        self._n_levels = n_levels
        self.step_range = step_range

        self._process_options = process_options
        # Number of simulation steps through whole mlmc
        self.target_time = None
        # Total variance
        self.target_variance = None

    def load_from_setup(self):
        """
        Run mlmc according to setup parameters, load setup {n_levels, step_range} and create levels
        :return: None
        """
        # Load mlmc params from file
        self._load_setup()

        # Create mlmc levels
        self.create_levels()

    def _load_setup(self):
        """
        Load mlmc setup file {n_levels, step_range}
        :return: None
        """
        if self._process_options['output_dir'] is not None:
            setup_file = os.path.join(self._process_options['output_dir'], "mlmc_setup.json")
            with open(setup_file, 'r') as f_reader:
                setup = json.load(f_reader)
                self._n_levels = setup.get('n_levels', None)
                self.step_range = setup.get('step_range', None)

    def _save_setup(self):
        """
        Save mlmc setup file {n_levels, step_range}
        :return: None
        """
        if self._process_options['output_dir'] is not None:
            setup_file = os.path.join(self._process_options['output_dir'], "mlmc_setup.json")
            setup = {'n_levels': self.n_levels, 'step_range': self.step_range}
            with open(setup_file, 'w+') as f_writer:
                json.dump(setup, f_writer)

    def create_levels(self):
        """
        Create level objects, each level has own level logger object
        :return: None
        """
        for i_level in range(self._n_levels):
            previous_level = self.levels[-1] if i_level else None
            if self._n_levels == 1:
                level_param = 1
            else:
                level_param = i_level / (self._n_levels - 1)

            logger = Logger(i_level, self._process_options['output_dir'], self._process_options['keep_collected'])
            level = Level(self.simulation_factory, previous_level, level_param, logger, self._process_options['regen_failed'])
            self.levels.append(level)

        self._save_setup()

    @property
    def n_levels(self):
        """
        Number of levels
        """
        if len(self.levels) > 0:
            return len(self.levels)

    @property
    def n_samples(self):
        """
        Level samples
        """
        return np.array([l.n_samples for l in self.levels])

    @property
    def n_nan_samples(self):
        """
        Level nan samples
        """
        return np.array([len(l.nan_samples) for l in self.levels])

    def estimate_diff_vars(self, moments_fn):
        """
        Estimate moments variance from samples
        :param moments_fn: Moment evaluation functions
        :return: tuple of np.array
        """
        vars = []
        n_samples = []

        for level in self.levels:
            v, n = level.estimate_diff_var(moments_fn)
            vars.append(v)
            n_samples.append(n)
        return np.array(vars), np.array(n_samples)

    def _varinace_regresion(self, raw_vars, n_samples, sim_steps):
        """
        Estimate level variance by regression
        :param raw_vars: moments variances raws
        :param n_samples: number of sample
        :param sim_steps: simulation steps
        :return: np.array
        """
        L, R = raw_vars.shape
        if L < 2:
            return raw_vars

        # estimate of variances, not use until we have model that fits well also for small levels
        S = 1 / (n_samples - 1)  # shape L
        W = np.sqrt(2 * S + 3 * S ** 2 + 2 * S ** 3)  # shape L

        # Use linear regresion to improve estimate of variances V1, ...
        # model log var_{r,l} = a_r  + b * log step_l
        # X_(r,l), j = dirac_{r,j}

        X = np.zeros((L, R-1, R))
        X[:, :, :-1] = np.eye(R-1)
        X[:, :, -1] = np.repeat(np.log(sim_steps[1:]), R-1).reshape((L, R-1))

        # X = X*W[:, None, None]    # scale
        X.shape = (-1, R)
        # solve X.T * X = X.T * V

        log_vars = np.log(raw_vars[:, 1:])     # omit first moment that is constant 1.0
        #log_vars = W[:, None] * log_vars    # scale
        params, res, rank, sing_vals = np.linalg.lstsq(X, log_vars.ravel())
        new_vars = np.empty_like(raw_vars)
        new_vars[:, 0] = raw_vars[:, 0]
        assert np.allclose(raw_vars[:, 0], 0.0)
        new_vars[:, 1:] = np.exp(np.dot(X, params)).reshape(L, -1)
        return new_vars

    def estimate_diff_vars_regression(self, moments_fn):
        """
        Estimate variances
        :param moments_fn: Moment evaluation function
        :return: array of variances
        """
        vars, n_samples = self.estimate_diff_vars(moments_fn)
        sim_steps = np.array([lvl.fine_simulation.step for lvl in self.levels])
        vars[1:] = self._varinace_regresion(vars[1:], n_samples, sim_steps)
        return vars

    def set_initial_n_samples(self, n_samples=None):
        """
        Set target number of samples for each level
        :param n_samples: array of number of samples
        :return: None
        """
        if n_samples is None:
            n_samples = [100, 3]
        # Num of samples to ndarray
        n_samples = np.atleast_1d(n_samples)

        # Just maximal number of samples is set
        if len(n_samples) == 1:
            n_samples = np.array([n_samples[0], 3])

        # Create number of samples for all levels
        if len(n_samples) == 2:
            L = max(2, self.n_levels)
            factor = (n_samples[-1] / n_samples[0])**(1 / (L-1))
            n_samples = n_samples[0] * factor ** np.arange(L)

        for i, level in enumerate(self.levels):
            level.set_target_n_samples(int(n_samples[i]))

    # def set_target_time(self, target_time):
    #     """
    #     For each level counts new N according to target_time
    #     :return: array
    #     TODO: Have a linear model to estimate true time per sample as a function of level step.
    #           This needs some test sampling... that is same as with variance estimates.
    #     """
    #     vars =
    #     amount = self._count_sum()
    #     # Loop through levels
    #     # Count new number of simulations for each level
    #     for level in self.levels:
    #         new_num_of_sim = np.round((target_time * np.sqrt(level.variance / level.n_ops_estimate()))
    #                                   / amount).astype(int)
    #
    #         self.num_of_simulations.append(new_num_of_sim)

    # def reset_moment_fn(self, moments_fn):
    #     for level in self.levels:
    #         level.reset_moment_fn(moments_fn)

    def set_target_variance(self, target_variance, moments_fn=None, fraction=1.0, prescribe_vars=None):
        """
        Estimate optimal number of samples for individual levels that should provide a target variance of
        resulting moment estimate. Number of samples are directly set to levels.
        This also set given moment functions to be used for further estimates if not specified otherwise.
        TODO: separate target_variance per moment
        :param target_variance: Constrain to achieve this variance.
        :param moments_fn: moment evaluation functions
        :param fraction: Plan only this fraction of computed counts.
        :param prescribe_vars: vars[ L, M] for all levels L and moments M safe the (zeroth) constant moment with zero variance.
        :return: np.array with number of optimal samples
        """
        if prescribe_vars is None:
            vars = self.estimate_diff_vars_regression(moments_fn)
        else:
            vars = prescribe_vars

        n_ops = np.array([lvl.n_ops_estimate for lvl in self.levels])

        sqrt_var_n = np.sqrt(vars.T * n_ops)    # moments in rows, levels in cols
        total = np.sum(sqrt_var_n, axis=1)      # sum over levels
        n_samples_estimate = np.round((sqrt_var_n / n_ops).T * total / target_variance).astype(int) # moments in cols
        n_samples_estimate_safe = np.maximum(np.minimum(n_samples_estimate, vars*self.n_levels/target_variance), 2)

        n_samples_estimate_max = np.max(n_samples_estimate_safe, axis=1)

        for level, n in zip(self.levels, n_samples_estimate_max):
            level.set_target_n_samples(int(n*fraction))

        return n_samples_estimate_safe

    def refill_samples(self):
        """
        For each level set fine and coarse simulations and generate samples in reverse order (from the finest sim)
        :return: None
        """
        # Set level coarse sim, it creates also fine simulations if its None
        for level in self.levels:
            level.set_coarse_sim()

        # Generate level's samples in reverse order
        for level in reversed(self.levels):
            level.fill_samples()

    def wait_for_simulations(self, sleep=0, timeout=None):
        """
        Waiting for running simulations
        :param sleep: time for doing nothing
        :param timeout: maximum time for waiting on running simulations
        :return: int, number of running simulations
        """
        if timeout is None:
            timeout = 0
        elif timeout <= 0:
            return 1
        n_running = 1
        t0 = time.clock()
        while n_running > 0:
            n_running = 0
            for level in self.levels:
                n_running += level.collect_samples()

            time.sleep(sleep)
            if 0 < timeout < (time.clock() - t0):
                break

        return n_running

    def estimate_domain(self):
        """
        Estimate domain of the density function.
        TODO: compute mean and variance and use quantiles of normal or lognormal distribution (done in Distribution)
        :return:
        """
        ranges = np.array([l.sample_range() for l in self.levels])

        return np.min(ranges[:, 0]), np.max(ranges[:, 1])

    def subsample(self, sub_samples=None):
        """
        :param sub_samples: None - use all generated samples
                    array of ints, shape = n_levels; choose given number of sub samples from computed samples
        :return: None
        """
        assert len(sub_samples) == self.n_levels
        if sub_samples is None:
            sub_samples = [None]*self.n_levels
        for ns, level in zip(sub_samples, self.levels):
            level.subsample(ns)

    def estimate_moments(self, moments_fn):
        """
        Use collected samples to estimate moments and variance of this estimate.
        :param moments_fn: Vector moment function, gives vector of moments for given sample or sample vector.
        :return: estimate_of_moment_means, estimate_of_variance_of_estimate ; arrays of length n_moments
        """

        means = []
        vars = []
        n_samples = []
        for level in self.levels:
            means.append(level.estimate_diff_mean(moments_fn))
            l_vars, ns = level.estimate_diff_var(moments_fn)
            vars.append(l_vars)
            n_samples.append(ns)
        means = np.sum(np.array(means), axis=0)
        n_samples = np.array(n_samples, dtype=int)

        vars = np.sum(np.array(vars) / n_samples[:, None], axis=0)

        return np.array(means), np.array(vars)

    def estimate_cost(self, level_times=None, n_samples=None):
        """
        Estimate total cost of mlmc
        :param level_times: Number of level executions
        :param n_samples: Number of samples on each level
        :return: total cost
        TODO: Have cost expressed as a time estimate. This requires
        estimate of relation ship to the  task size and complexity.
        """
        if level_times is None:
            level_times = [lvl.n_ops_estimate for lvl in self.levels]
        if n_samples is None:
            n_samples = self.n_samples
        return np.sum(level_times * n_samples)

    def clean_levels(self):
        """
        Reset all levels
        :return: None
        """
        for level in self.levels:
            level.reset()

    def clean_subsamples(self):
        """
        Clean level subsamples
        :return: None
        """
        for level in self.levels:
            level.sample_indices = None
