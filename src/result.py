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

            # Add fine - coarse
            for fine_and_coarse in level.data:
                # Array of fine - coarse
                self.levels_data[index].append(fine_and_coarse[0] - fine_and_coarse[1])

            self.simulation_on_level.append(len(level.data))

        self.result_of_levels()

    def result_of_levels(self):
        """
        Basic result from Monte Carlo method
        """
        self.levels_dispersion = []
        self.average = 0
        self.simulation_results = [0 for _ in self.levels_data[0]]

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
        Create sum of
        :return: moments
        """
        moments_pom = []
        for level in self.mc_levels:
            level.get_moments()
            moments_pom.append(level.moments)
        return [sum(m) for m in zip(*moments_pom)]

