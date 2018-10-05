import os.path
import json
import time
import numpy as np
from mlmc.mc_level import Level
from mlmc.logger import Logger
import scipy.stats as st
import scipy.integrate as integrate
from mlmc.simulation import Simulation


##################################################

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

    @property
    def sim_steps(self):
        return np.array([Simulation.log_interpolation(self.step_range, lvl.step) for lvl in self.levels])


    def estimate_diff_vars(self, moments_fn):
        """
        Estimate moments variance from samples
        :param moments_fn: Moment evaluation functions
        :return: (diff_variance, n_samples);
            diff_variance - shape LxR, variances of diffs of moments
            n_samples -  shape L, num samples for individual levels.

            Returns simple variance for level 0.
        """
        vars = []
        n_samples = []

        for level in self.levels:
            v, n = level.estimate_diff_var(moments_fn)
            vars.append(v)
            n_samples.append(n)
        return np.array(vars), np.array(n_samples)

    def estimate_level_means(self, moments_fn):
        """
        Estimate means on individual levels.
        :param moments_fn: moments object of size R
        :return: shape (L, R)
        """
        means = []
        for level in self.levels:
            means.append(level.estimate_diff_mean(moments_fn))
        return np.array(means)



    def _variance_of_variance(self, n_samples = None):
        """
        Approximate variance of log(X) where
        X is from ch-squared with df=n_samples - 1.
        Return array of variances for actual n_samples array.

        :param n_samples: Optional array with n_samples.
        :return: array of variances of variance estimate.
        """
        if n_samples is None:
            n_samples = self.n_samples
        if hasattr(self, "_saved_var_var"):
            ns, var_var = self._saved_var_var
            if np.sum(np.abs(ns - n_samples)) == 0:
                return var_var

        vars = []
        for ns in n_samples:
            df = ns - 1

            def log_chi_pdf(x):
                return np.exp(x) * df * st.chi2.pdf(np.exp(x) * df, df=df)

            def compute_moment(moment):
                std_est = np.sqrt(2 / df)
                fn = lambda x, m=moment: x ** m * log_chi_pdf(x)
                return integrate.quad(fn, -100 * std_est, 100 * std_est)[0]

            mean = compute_moment(1)
            second = compute_moment(2)
            vars.append(second - mean ** 2)

        self._saved_var_var = (n_samples, np.array(vars))
        return np.array(vars)

    def _varinace_regression(self, raw_vars, sim_steps):
        """
        Estimate level variance by regression from model:

        log(var_l,r) = A_r + B * log(h_l) + C * log^2(hl),
                                            for l = 0, .. L-1


        :param raw_vars: level variances , shape (L)
        :param sim_steps: simulation steps, shape L
        :return: np.array  (L, )
        """
        L, = raw_vars.shape
        if L <= 2:
            return raw_vars
        L1 = L - 1

        # estimate of variances of variances, compute scaling
        W = 1.0 / np.sqrt(self._variance_of_variance())
        W = W[1:]   # ignore level 0
        #W = np.ones((L - 1,))

        # Use linear regresion to improve estimate of variances V1, ...
        # model log var_{r,l} = a_r  + b * log step_l
        # X_(r,l), j = dirac_{r,j}

        K = 3 # number of parameters
        log_step = np.log(sim_steps[1:])
        X = np.zeros((L1,  K))
        X[:,  0] = 1.0
        X[:,  1] = log_step
        X[:,  2] = log_step ** 2


        WX = X * W[:, None]    # scale

        log_vars = np.log(raw_vars[1:])     # omit first variance, and first moment that is constant 1.0
        log_vars = W[:] * log_vars       # scale RHS

        params, res, rank, sing_vals = np.linalg.lstsq(WX, log_vars)
        new_vars = raw_vars.copy()
        #assert np.isclose(raw_vars[0], 0.0)
        new_vars[1:] = np.exp(np.dot(X, params))
        return new_vars

    def estimate_diff_vars_regression(self, moments_fn=None, raw_vars=None):
        """
        Estimate variances using linear regression model.
        Assumes increasing variance with moments, use only two moments with highest average variance.
        :param moments_fn: Moment evaluation function
        :return: array of variances, shape  L
        """
        # vars shape L x R
        if raw_vars is None:
            assert moments_fn is not None
            raw_vars, n_samples = self.estimate_diff_vars(moments_fn)
        sim_steps = self.sim_steps
        # for evry level max variance over moments
        max_vars = np.max(raw_vars, axis=1)
        vars = self._varinace_regression(max_vars, sim_steps)
        return vars


    def sample_range(self, n0, nL):
        """
        Geometric sequence of L elements decreasing from n0 to nL.
        Useful to set number of samples explicitly.
        :param n0: int
        :param nL: int
        :return: np.array of length L = n_levels.
        """
        return np.round(np.exp2(np.linspace(np.log2(n0), np.log2(nL), self.n_levels))).astype(int)


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
            n0, nL = n_samples
            n_samples = self.sample_range(n0, nL)

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

    def estimate_n_samples_for_target_variance(self, target_variance, moments_fn=None, prescribe_vars=None):
        """
        Estimate optimal number of samples for individual levels that should provide a target variance of
        resulting moment estimate. Number of samples are directly set to levels.
        This also set given moment functions to be used for further estimates if not specified otherwise.
        TODO: separate target_variance per moment
        :param target_variance: Constrain to achieve this variance.
        :param moments_fn: moment evaluation functions
        :param fraction: Plan only this fraction of computed counts.
        :param prescribe_vars: vars[ L, M] for all levels L and moments M safe the (zeroth) constant moment with zero variance.
        :return: np.array with number of optimal samples for individual levels and moments, array (LxR)
        """
        if prescribe_vars is None:
            vars = self.estimate_diff_vars_regression(moments_fn)
        else:
            vars = prescribe_vars

        n_ops = np.array([lvl.n_ops_estimate for lvl in self.levels])

        sqrt_var_n = np.sqrt(vars.T * n_ops)    # moments in rows, levels in cols
        total = np.sum(sqrt_var_n, axis=1)      # sum over levels
        n_samples_estimate = np.round((sqrt_var_n / n_ops).T * total / target_variance).astype(int)# moments in cols

        # Limit maximal number of samples per level
        n_samples_estimate_safe = np.maximum(np.minimum(n_samples_estimate, vars*self.n_levels/target_variance), 2)

        return n_samples_estimate_safe


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

        n_samples =  self.estimate_n_samples_for_target_variance(target_variance, moments_fn, prescribe_vars)
        n_samples = np.max(n_samples, axis=1)
        for level, n in zip(self.levels, n_samples):
            level.set_target_n_samples(int(n*fraction))


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
        if sub_samples is None:
            sub_samples = [None]*self.n_levels
        assert len(sub_samples) == self.n_levels, "{} != {}".format(len(sub_samples), self.n_levels)
        for ns, level in zip(sub_samples, self.levels):
            level.subsample(ns)

    def update_moments(self, moments_fn):
        for level in self.levels:
            level.evaluate_moments(moments_fn, force=True)

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
            level.subsample(None)
