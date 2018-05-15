import time as t
import os
import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../src/')
import shutil
import math
import csv
from test.simulation_test import SimulationTest as TestSim
from test.result import Result
from mlmc.mlmc import MLMC
from mlmc.moments import Monomials, FourierFunctions
import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from src.mlmc.distribution import Distribution
import scipy.stats as st
from scipy.stats import norm
import scipy.misc as sm
import pynverse as pv


def main(*args):


    with open('flow_mlmc.csv', 'w', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=';',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)

        eps = 1e-10
        pocet_urovni = 0
        moments_number = 20

        writer.writerow([" ", "n0", "n1", "level variance", "N", "time", "mean value",  "mean variance", "sum sim steps"])


        files = [["/home/martin/Desktop/FlowResults/FourierFunctions/Results/DATA_MLMC/data_1L_01",
                 "/home/martin/Desktop/FlowResults/FourierFunctions/Results/DATA_MLMC/data_2L",
                 "/home/martin/Desktop/FlowResults/FourierFunctions/Results/DATA_MLMC/data_3L",
                 "/home/martin/Desktop/FlowResults/FourierFunctions/Results/DATA_MLMC/data_4L_5",
                 "/home/martin/Desktop/FlowResults/FourierFunctions/Results/DATA_MLMC/data_5L"
                 ],

               ["/home/martin/Desktop/FlowResults/FourierFunctions/Results/DATA_MLMC/data_1L_02",
                 "/home/martin/Desktop/FlowResults/FourierFunctions/Results/DATA_MLMC/data_2L_1",
                 "/home/martin/Desktop/FlowResults/FourierFunctions/Results/DATA_MLMC/data_3L_1",
                 "/home/martin/Desktop/FlowResults/FourierFunctions/Results/DATA_MLMC/data_4L_2",
                 "/home/martin/Desktop/FlowResults/FourierFunctions/Results/DATA_MLMC/data_5L_1"],

               ["/home/martin/Desktop/FlowResults/FourierFunctions/Results/DATA_MLMC/data_1L_03",
                 "/home/martin/Desktop/FlowResults/FourierFunctions/Results/DATA_MLMC/data_2L_2",
                 "/home/martin/Desktop/FlowResults/FourierFunctions/Results/DATA_MLMC/data_3L_2",
                 "/home/martin/Desktop/FlowResults/FourierFunctions/Results/DATA_MLMC/data_4L_1",
                 "/home/martin/Desktop/FlowResults/FourierFunctions/Results/DATA_MLMC/data_5L_2"]]
        """

        files = [["/home/martin/Desktop/FlowResults/FourierFunctions/Results/DATA_MLMC/data_1L"],
                 ["/home/martin/Desktop/FlowResults/FourierFunctions/Results/DATA_MLMC/data_1L_01"],
                 ["/home/martin/Desktop/FlowResults/FourierFunctions/Results/DATA_MLMC/data_1L_02"],
                 ["/home/martin/Desktop/FlowResults/FourierFunctions/Results/DATA_MLMC/data_1L_03"]]
        """

        for level_n in range(len(files[0])):
            zacatek = t.time()
            pocet_urovni += 1

            simulations_on_level = []
            averages = []
            all_moments_mean = []
            all_moments_variance = []
            l_variances = []

            level_simulations_steps = []

            for i in range(len(files)):
                result = Result(moments_number)
                result.extract_data(files[i][level_n])
                mo = FourierFunctions(moments_number)
                mo.eps = eps

                result.process_data()
                mean = result.average
                mo.mean = mean
                result.add_level_moments_object(mo)
                result.format_result()
                averages.append(mean)
                simulations_on_level.append(result.simulation_on_level)

                mm = []
                mv =[]
                for moment in result.moments:
                    mm.append(moment[0])
                    mv.append(moment[1])

                all_moments_mean.append(mm)
                all_moments_variance.append(mv)

                l_steps = []
                l_variance = []
                for level in result.mc_levels:
                    l_steps.append(level.n_ops_estimate())
                    l_variance.append(np.var(level.result))

                level_simulations_steps.append(l_steps)
                l_variances.append(l_variance)

            konec = t.time()
            time = konec - zacatek
            avg_simulations_on_level = [np.average(e).astype(int) for e in zip(*simulations_on_level)]
            level_simulations_steps = [np.average(e).astype(int) for e in zip(*level_simulations_steps)]
            avg_value = np.average(averages)
            variances = np.mean(l_variances, axis=0)
            whole_variance = 0

            for index, sim_on_level in enumerate(avg_simulations_on_level):
                whole_variance += (variances[index]/ sim_on_level)

            cost = 0
            for index, level_sim in enumerate(avg_simulations_on_level):
                if index > 0:
                    cost += (level_sim * level_simulations_steps[index])# + (level_sim *level_simulations_steps[index-1]))
                else:
                    cost += level_sim * level_simulations_steps[index]

            writer.writerow([str(pocet_urovni) + " úrovňová metoda"])
            for index, n_steps in enumerate(level_simulations_steps):
                if index == 0:
                    writer.writerow(["", "0", n_steps, variances[index], avg_simulations_on_level[index], time, avg_value, whole_variance,
                                    cost])
                else:
                    writer.writerow(["", level_simulations_steps[index-1], n_steps, variances[index], avg_simulations_on_level[index],"",
                                     ])


if __name__ == "__main__":
    main()
