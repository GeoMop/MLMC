import numpy as np


class Result:
    """
    Multilevel Monte Carlo result
    """
    def __init__(self, moments_number):
        self.variance = []
        self._time = 0
        self.average = 0
        self.data = []
        self.moments = []
        self.simulation_results = []
        self.simulation_on_level = []
        self.levels_num_of_steps = []
        self.moments_number = moments_number
        self.levels = None

        self._mc_data = []
        self._mc_levels = []
        self.levels = []
        # Variance of values on each level
        self.levels_dispersion = []
        self._levels_number = None
        # Each level values (fine_step - coarse_step)
        self.levels_data = []

    @property
    def mc_levels(self):
        """
        Monte Carlo method levels
        :return: array of levels
        """
        return self._mc_levels

    @mc_levels.setter
    def mc_levels(self, mc_levels):
        if not mc_levels:
            raise ValueError("mc_levels must be list of levels from monte carlo method")

        self._mc_levels = mc_levels

    @property
    def levels_number(self):
        """
        Number of Monte Carlo method levels
        :return: array, number of levels
        """
        return self._levels_number

    @levels_number.setter
    def levels_number(self, levels_number):
        if levels_number < 1:
            raise ValueError("Levels number must be integer greater than 0")
        self._levels_number = int(levels_number)

    @property
    def time(self):
        """
        Monte Carlo method processing time
        """
        return self._time

    @time.setter
    def time(self, value):
        if value < 0:
            ValueError("Time must be positive value")
        self._time = value

    def process_data(self):
        """
        Prepare mlmc levels data for further use
        :return: None
        """
        if bool(self.mc_levels) is False:
            raise ValueError("mc_levels must be array")

        self.levels_data = [[] for _ in range(len(self.mc_levels))]
        self.levels = [[] for _ in range(len(self.mc_levels))]
        self.simulation_on_level = []

        # Each level in one execution of mlmc
        for index, level in enumerate(self.mc_levels):

            self.levels_num_of_steps.append(level.n_ops_estimate())
            # Add fine - coarse
            for fine_and_coarse in level.data:
                # Array of fine - coarse
                self.levels_data[index].append(fine_and_coarse[0] - fine_and_coarse[1])
            self.simulation_on_level.append(level.number_of_simulations)

        self.result_of_levels()

    def result_of_levels(self):
        """
        Basic result from Monte Carlo method
        """
        self.levels_dispersion = []
        self.average = 0
        self.simulation_results = [0 for _ in range(len(max(self.levels_data,key=len)))]

        for level_data in self.levels_data:
            self.levels_dispersion.append(np.var(level_data))
            for index, data in enumerate(level_data):
                self.simulation_results[index] += data

            self.average += np.mean(level_data)

    def format_result(self):
        """
        Print results
        """
        print("Střední hodnota = ", self.average)
        print("Rozptyl hodnota = ", np.var(self.simulation_results))
        print("Rozptyly na úrovních", self.levels_dispersion)
        print("Počet simulací na jednotlivých úrovních", self.simulation_on_level)

    def level_moments(self):
        """
        Create sum of moments values from all levels
        :return: moments
        """
        moments_pom = []
        moments = [[0, 0] for _ in range(len(self.mc_levels[0].moments))]
        for level in self.mc_levels:
            moments_pom.append(level.moments)
            for index, moment in enumerate(level.moments):
                moments[index][0] += moment[0]
                moments[index][1] += moment[1]

        return [mean for mean, var in moments]

