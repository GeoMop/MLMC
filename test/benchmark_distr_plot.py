import test.benchmark_distributions as bd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import mlmc.tool.plot


quantile = 0.000001

distributions = [stats.norm(loc=0, scale=10),
                     stats.lognorm(scale=np.exp(1), s=1),
                     bd.TwoGaussians(name='two_gaussians'),
                     bd.FiveFingers(name='five_fingers'),
                     bd.Cauchy(name='cauchy'),
                     bd.Discontinuous(name='discontinuous'),
                     bd.Abyss(name="abyss")]

def plot_distributions():

    for distr in distributions:
        if hasattr(distr, "domain"):
            domain = distr.domain
        else:
            domain = distr.ppf([quantile, 1- quantile])
        x = np.linspace(domain[0], domain[1], 10000)
        plot_distr(x, distr.pdf(x), distr)


def plot_distr(x, density, distr):
    #plt.figure(figsize=(16, 9))
    plt.plot(x, density, color="black")
    plt.xlabel(r'$x$')
    plt.ylabel(r'$f(x)$')

    if 'dist' in distr.__dict__:
        name = "{}".format(distr.dist.name)
    else:
        name = "{}".format(distr.name)

    file = name + ".pdf"
    print("file ", file)
    plt.savefig(file)
    plt.show()


# def plot_for_article():
#
#     shape = (2, 3)
#     fig, axes = plt.subplots(*shape, sharex=True, sharey=True, figsize=(15, 10))
#     # fig.suptitle("Mu -> Lambda")
#     axes = axes.flatten()
#
#
#     for distr, ax in zip(distributions, axes):
#         if hasattr(distr, "domain"):
#             domain = distr.domain
#         else:
#             domain = distr.ppf([quantile, 1-quantile])
#         x = np.linspace(domain[0], domain[1], 10000)
#
#
#         ax.plot(x, distr.pdf(x), color="black")
#
#         ax.set_ylabel(r'$x$')
#         ax.set_xlabel(r'$f(x)$')
#
#         if 'dist' in distr.__dict__:
#             name = "{}".format(distr.dist.name)
#         else:
#             name = "{}".format(distr.name)
#
#     plt.tight_layout()
#     fig.legend()
#
#     # mlmc.plot._show_and_save(fig, "", "mu_to_lambda_lim")
#     mlmc.tool.plot._show_and_save(fig, None, "benchmark_distributions")
#     mlmc.tool.plot._show_and_save(fig, "", "benchmark_distributions")


if __name__ == "__main__":
    #plot_for_article()
    plot_distributions()
