import numpy as np
from mlmc.mc_level import Level
import time

class MLMC:
    """
    Multi level monte carlo method
    """
    def __init__(self, n_levels, sim_factory, pbs=None):
        """
        :param number_of_levels:    Number of levels
        :param sim:                 Instance of object Simulation
        """
        # System interaction object (FlowPBS)
        self._pbs = pbs
        # Object of simulation
        self.simulation_factory = sim_factory
        # Array of level objects
        self.create_levels(n_levels)
        if pbs.collected_log_content is not None:
            self.load_levels(pbs.collected_log_content, pbs.running_log_content)

        # Time of all mlmc
        self.target_time = None
        # Variance of all mlmc
        self.target_variance = None
        # The fines simulation step

    def create_levels(self, n_levels):
        self.levels = []
        for i_level in range(n_levels):
            previous = self.levels[-1].fine_simulation if i_level else None
            if n_levels == 1:
                level_param = 1
            else:
                level_param = i_level / (n_levels - 1)
            level = Level(i_level, self.simulation_factory, previous, level_param)
            self.levels.append(level)

    def load_levels(self, finished_list, running_list):
        # recover finished
        self.clean_levels()
        finished = set()
        for sim in finished_list:
            i_level, i, fine, coarse, value = sim
            self.levels[i_level].finished_simulations.append( (i, fine, coarse) )
            finished.add(i)

        for level in self.levels:
            level.sample_values = np.zeros((level.n_collected_samples, 2))
            level.target_n_samples = level.n_collected_samples
        for sim in finished_list:
            i_level, i, fine, coarse, value = sim
            self.levels[i_level].sample_values[i, :] = value

        # recover running
        for sim in running_list:
            if len(sim) !=5:
                continue
            i_level, i, fine, coarse, _ = sim
            if i not in finished:
                self.levels[i_level].running_simulations.append( (i, fine, coarse) )
        for level in self.levels:
            level.enlarge_samples(level.n_total_samples)
            level.collect_samples(self._pbs)


    # def update_collected(self):
    #     """
    #     Re-create collected sample values from simulation sample directories.
    #     :return:
    #     """
    #     for l in self.levels:
    #         l.update_collected()

    @property
    def n_levels(self):
        return len(self.levels)

    @property
    def n_samples(self):
        return np.array([ l.n_samples for l in self.levels ])


    def estimate_diff_vars(self, moments_fn):
        vars = []
        n_samples = []
        for level in self.levels:
            v, n = level.estimate_diff_var(moments_fn)
            vars.append(v)
            n_samples.append(n)
        return np.array(vars), np.array(n_samples)

    def _varinace_regresion(self, raw_vars, n_samples, sim_steps):
        L, R = raw_vars.shape
        if L < 2:
            return raw_vars

        # estimate of variances, not use until we have model that fits well also for small levels
        S = 1 / (n_samples - 1)  # shape L
        W = np.sqrt(2 * S + 3 * S ** 2 + 2 * S ** 3)  # shape L

        # Use linear regresion to improve esimtae of variances V1, ...
        # model log var_{r,l} = a_r  + b * log step_l
        # X_(r,l), j = dirac_{r,j}

        X = np.zeros(( L, R-1, R))
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
        #print(raw_vars - new_vars)
        return new_vars

    def estimate_diff_vars_regression(self, moments_fn):
        vars, n_samples = self.estimate_diff_vars(moments_fn)
        sim_steps = np.array([lvl.fine_simulation.step for lvl in self.levels])
        vars[1:] = self._varinace_regresion(vars[1:], n_samples, sim_steps)
        return vars

    def set_initial_n_samples(self, n_samples=None):
        if n_samples is None:
            n_samples = [100, 3]
        n_samples = np.atleast_1d(n_samples)
        if len(n_samples) == 1:
            n_samples = np.array([n_samples, 3])
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

    def set_target_variance(self, target_variance, moments_fn=None, fraction = 1.0, prescribe_vars=None):
        """
        Estimate optimal number of samples for individual levels that should provide a target variance of
        resulting moment estimate. Number of samples are directly set to levels.
        This also set given moment functions to be used for further estimates if not specified otherwise.
        TODO: separate target_variance per moment
        :param target_variance: Constrain to achive this variance.
        :param moments_fn: moment evaluation functions
        :param fraction: Plan only this fraction of computed counts.
        :param prescribe_vars: vars[ L, M] for all levels L and moments M safe the (zeroth) constant moment with zero variance.
        :return: np.array with number of optimal samples
        """
        if prescribe_vars is None:
            vars = self.estimate_diff_vars_regression(moments_fn)
        else:
            vars = prescribe_vars
        n_ops = np.array([lvl.n_ops_estimate() for lvl in self.levels])
        sqrt_var_n = np.sqrt(vars.T * n_ops)    # moments in rows, levels in cols
        total = np.sum(sqrt_var_n, axis=1)      # sum over levels
        n_samples_estimate = np.round((sqrt_var_n / n_ops).T * total / target_variance).astype(int) # moments in cols
        n_samples_estimate_safe = np.maximum(np.minimum(n_samples_estimate, vars*self.n_levels/target_variance), 2)
        n_samples_estimate_max = np.max(n_samples_estimate_safe, axis=1)

        #n_samples_estimate_max= np.array([30, 3, 0.3])/target_variance
        #print(n_samples_estimate_max)
        for level, n  in zip(self.levels, n_samples_estimate_max):
            level.set_target_n_samples( int(n*fraction) )
        return  n_samples_estimate_safe

    # @property
    # def number_of_samples(self):
    #     """
    #     List of samples in each level
    #     :return: array
    #     """
    #     return self._number_of_samples

    # @number_of_samples.setter
    # def number_of_samples(self, num_of_sim):
    #     if len(num_of_sim) < self.number_of_levels:
    #         raise ValueError("Number of simulations must be list")
    #
    #     self._number_of_samples = num_of_sim

    #def estimate_n_samples(self):
    #    # Count new number of simulations according to variance of time
    #    if self.target_variance is not None:
    #        self._num_of_samples = self.estimate_n_samples_from_variance()
    #    elif self.target_time is not None:
    #        self._num_of_samples = self.estimate_n_samples_from_time()

    def refill_samples(self):
        """
        For each level counts further number of simulations by appropriate N
        """

        for level in (self.levels):
            level.fill_samples(self._pbs)

        self._pbs.execute()

    def wait_for_simulations(self, sleep = 0, timeout=None):
        if timeout is None:
            timeout = 0
        elif timeout <= 0:
            return 1
        n_running = 1
        t0 = time.clock()
        while n_running > 0:
            n_running=0
            for level in self.levels:
                n_running += level.collect_samples(self._pbs)
            time.sleep(sleep)
            if timeout > 0 and (time.clock() - t0) > timeout:
                break
        return n_running

    def estimate_domain(self):
        """
        Estimate domain of the density function.
        TODO: compute mean and variance and use quantiles of normal or lognormal distribution (done in Distribution)
        :return:
        """
        ranges = np.array([l.sample_range() for l in self.levels])
        return (np.min(ranges[:,0]), np.max(ranges[:,1]))


    # def save_data(self):
    #     """
    #     Save results for future use
    #     """
    #     with open("data", "w") as fout:
    #         for index, level in enumerate(self.levels):
    #             fout.write("LEVEL" + "\n")
    #             fout.write(str(level.number_of_simulations) + "\n")
    #             fout.write(str(level.n_ops_estimate()) + "\n")
    #             for tup in level.data:
    #                 fout.write(str(tup[0])+ " " + str(tup[1]))
    #                 fout.write("\n")

    def subsample(self, sub_samples=None):
        """
        :param sub_samples: None - use all generated samples
                    array of ints, shape = n_levels; choose given number of sub samples from computed samples
        :return:
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

        return means, vars


    def estimate_cost(self, level_times=None, n_samples=None):
        if level_times is None:
            level_times = [ lvl.n_ops_estimate() for lvl in self.levels]
        if n_samples is None:
           n_samples = self.n_samples
        return np.sum(level_times * n_samples)

    def clean_levels(self):
        for level in self.levels:
            level.reset()

