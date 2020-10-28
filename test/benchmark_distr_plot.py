import test.benchmark_distributions as bd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np


quantile = 0.000001


def plot_distributions():
    distributions = [stats.norm(loc=0, scale=10),
                     stats.lognorm(scale=np.exp(1), s=1),
                     bd.TwoGaussians(name='two_gaussians'),
                     bd.FiveFingers(name='five_fingers'),
                     bd.Cauchy(name='cauchy'),
                     bd.Discontinuous(name='discontinuous')]
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


if __name__ == "__main__":
    plot_distributions()