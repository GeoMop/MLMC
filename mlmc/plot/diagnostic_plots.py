import numpy as np
import scipy.stats as st
from scipy import interpolate
import matplotlib

matplotlib.rcParams.update({'font.size': 22})
from matplotlib.patches import Patch
import matplotlib.pyplot as plt


# def log_var_level(variances, l_vars, err_variances=0, err_l_vars=0, moments=[1,2,3,4]):
#     fig, ax1 = plt.subplots(figsize=(8, 5))
#     for m in moments:
#         # line1, = ax1.errorbar(np.log2(variances[m]), yerr=err_variances, label="m={}".format(m), marker="o")
#         # line2, = ax1.errorbar(np.log2(l_vars[:, m]), yerr=err_l_vars, label="m={}".format(m), marker="s")
#         #line1, = ax1.plot(np.log2(variances[m]),  label="m={}".format(m), marker="o")
#         line2, = ax1.plot(np.log2(l_vars[:, m]), label="m={}".format(m), marker="s")
#
#     ax1.set_ylabel('log' + r'$_2$' + 'variance')
#     ax1.set_xlabel('level' + r'$l$')
#     plt.legend()
#     #plt.savefig("MLMC_cost_saves.pdf")
#     plt.show()


def log_var_per_level(l_vars, err_variances=0, err_l_vars=0, moments=[1, 2, 3, 4]):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    for m in moments:
        # line1, = ax1.errorbar(np.log2(variances[m]), yerr=err_variances, label="m={}".format(m), marker="o")
        # line2, = ax1.errorbar(np.log2(l_vars[:, m]), yerr=err_l_vars, label="m={}".format(m), marker="s")
        #line1, = ax1.plot(np.log2(variances[m]),  label="m={}".format(m), marker="o")
        line2, = ax1.plot(np.log2(l_vars[:, m]), label="m={}".format(m), marker="s")

    ax1.set_ylabel('log' + r'$_2$' + 'variance')
    ax1.set_xlabel('level' + r'$l$')
    plt.legend()
    #plt.savefig("MLMC_cost_saves.pdf")
    plt.show()


# def log_mean_level(means, l_means, err_means=0, err_l_means=0, moments=[1,2,3,4]):
#     fig, ax1 = plt.subplots(figsize=(8, 5))
#     for m in moments:
#         # line1, = ax1.errorbar(np.log2(variances[m]), yerr=err_variances, label="m={}".format(m), marker="o")
#         # line2, = ax1.errorbar(np.log2(l_vars[:, m]), yerr=err_l_vars, label="m={}".format(m), marker="s")
#         #line1, = ax1.plot(np.log2(variances[m]),  label="m={}".format(m), marker="o")
#         line2, = ax1.plot(np.log2(np.abs(l_means[:, m])), label="m={}".format(m), marker="s")
#
#     ax1.set_ylabel('log' + r'$_2$' + 'mean')
#     ax1.set_xlabel('level' + r'$l$')
#     plt.legend()
#     #plt.savefig("MLMC_cost_saves.pdf")
#     plt.show()


def log_mean_per_level(l_means, err_means=0, err_l_means=0, moments=[1, 2, 3, 4]):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    for m in moments:
        # line1, = ax1.errorbar(np.log2(variances[m]), yerr=err_variances, label="m={}".format(m), marker="o")
        # line2, = ax1.errorbar(np.log2(l_vars[:, m]), yerr=err_l_vars, label="m={}".format(m), marker="s")
        #line1, = ax1.plot(np.log2(variances[m]),  label="m={}".format(m), marker="o")
        line2, = ax1.plot(np.log2(np.abs(l_means[:, m])), label="m={}".format(m), marker="s")

    ax1.set_ylabel('log' + r'$_2$' + 'mean')
    ax1.set_xlabel('level' + r'$l$')
    plt.legend()
    #plt.savefig("MLMC_cost_saves.pdf")
    plt.show()


def sample_cost_per_level(costs):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    line2, = ax1.plot(np.log2(costs), marker="s")

    ax1.set_ylabel('log' + r'$_2$' + 'cost per sample')
    ax1.set_xlabel('level' + r'$l$')
    plt.legend()
    #plt.savefig("MLMC_cost_saves.pdf")
    plt.show()


def kurtosis_per_level(means, l_means, err_means=0, err_l_means=0, moments=[1, 2, 3, 4]):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    for m in moments:
        # line1, = ax1.errorbar(np.log2(variances[m]), yerr=err_variances, label="m={}".format(m), marker="o")
        # line2, = ax1.errorbar(np.log2(l_vars[:, m]), yerr=err_l_vars, label="m={}".format(m), marker="s")
        #line1, = ax1.plot(np.log2(variances[m]),  label="m={}".format(m), marker="o")
        line2, = ax1.plot(np.log2(np.abs(l_means[:, m])), label="m={}".format(m), marker="s")

    ax1.set_ylabel('log ' + r'$_2$ ' + 'mean')
    ax1.set_xlabel('level ' + r'$l$')
    plt.legend()
    #plt.savefig("MLMC_cost_saves.pdf")
    plt.show()



