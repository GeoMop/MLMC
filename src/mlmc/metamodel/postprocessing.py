import matplotlib.pyplot as plt
from scipy.stats import ks_2samp


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    #plt.ylim([0, 10])
    plt.yscale("log")
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()


def analyze_results(target, predictions):
    statistics, pvalue = ks_2samp(target, predictions)
    print("KS statistics: {}, pvalue: {}".format(statistics, pvalue))
    # The closer KS statistic is to 0 the more likely it is that the two samples were drawn from the same distribution

    plt.hist(target,  alpha=0.5, label='target', density=True)
    plt.hist(predictions, alpha=0.5, label='predictions', density=True)
    plt.legend(loc='upper right')
    plt.show()