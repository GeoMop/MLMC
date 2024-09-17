import os
import numpy as np
import pandas as pd
from mlmc.tool.hdf5 import HDF5
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

DATA_PATH = "/home/martin/Documents/metamodels/data"

def get_inputs(dir):
    fields_file = os.path.join(dir, "fine_fields_sample.msh")
    input = []
    with open(fields_file, "r") as r:
        fields = r.readlines()
        for f in fields[12:]:
            line = f.split(" ")
            if len(line) > 1:
                input.append(float(line[1]))
            else:
                break
    return input


def preprocess_data():
    dir_path = "/home/martin/Documents/metamodels/data/L1/test/01_cond_field/output"
    hdf = HDF5(file_path="/home/martin/Documents/metamodels/data/L1/test/01_cond_field/mlmc_1.hdf5",
               load_from_file=True)
    level_group = hdf.add_level_group(level_id=str(0))
    collected = zip(level_group.get_collected_ids(), level_group.collected())

    df_values = []
    for sample_id, col_values in collected:
        output_value = col_values[0, 0]
        sample_dir = os.path.join(dir_path, sample_id)
        if os.path.isdir(sample_dir):
            input = get_inputs(sample_dir)
            d = {'x': np.array(input), 'y': output_value}
            df_values.append(d)

    df = pd.DataFrame(df_values)
    df.to_pickle(os.path.join(DATA_PATH, "data.pkl"))


def load_data():
    df = pd.read_pickle(os.path.join(DATA_PATH, "data.pkl"))
    return df


def data_analysis(df):
    print(df.info())
    print(df.y.describe())

    # df.y.plot.hist(bins=50, logx=True)
    # plt.show()
    df.y.plot.kde(bw_method=0.3)
    plt.xlim([-5, df.y.max()])
    plt.show()


def support_vector_regression(df):
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    print("df. info ", df.info)
    train, test = train_test_split(df[:8000], test_size=0.2)
    print("train describe", train.describe())
    print("test describe ", test.describe())

    x = np.stack(train.x.to_numpy(), axis=0)
    y = train.y.to_numpy()

    sc_X = StandardScaler()
    sc_y = StandardScaler()
    x = sc_X.fit_transform(x)
    # print("y.shape ", y.shape)
    # print("y.reshape(-1, 1) ", y.reshape(-1, 1).shape)
    y = sc_y.fit_transform(y.reshape(-1, 1))

    test_x = sc_X.fit_transform(np.stack(test.x.to_numpy(), axis=0))
    test_y = test.y.to_numpy().reshape(-1, 1)
    test_y = sc_y.fit_transform(test.y.to_numpy().reshape(-1, 1))

    # plt.hist(y, bins=50, alpha=0.5, label='train', density=True)
    # plt.hist(test_y, bins=50, alpha=0.5, label='test', density=True)
    # plt.legend(loc='upper right')
    # # plt.xlim(-0.5, 1000)
    # plt.yscale('log')
    # plt.show()
    # exit()

    # print("x.shape ", x.shape)
    # print("y.shape ", y.shape)
    # exit()

    #svr_rbf = SVR(kernel='rbf', verbose=True)  # 'linear' kernel fitting is never-ending and 'poly' kernel gives very bad score (e.g. -2450), sigmoid gives also bad score (e.g. -125)
    svr_rbf = SVR(kernel='poly', degree=50, verbose=True)
    svr_rbf.fit(x, y)

    train_error = svr_rbf.score(x, y)

    test_error = svr_rbf.score(test_x, test_y)

    predictions = svr_rbf.predict(test_x)

    plt.hist(test_y, bins=50, alpha=0.5, label='target', density=True)
    plt.hist(predictions, bins=50, alpha=0.5, label='predictions', density=True)

    # plt.hist(targets - predictions, bins=50, alpha=0.5, label='predictions', density=True)
    plt.legend(loc='upper right')
    # plt.xlim(-0.5, 1000)
    plt.yscale('log')
    plt.show()

    plt.hist(test_y, bins=50, alpha=0.5, label='target', density=True)
    plt.hist(predictions, bins=50, alpha=0.5, label='predictions', density=True)

    # plt.hist(targets - predictions, bins=50, alpha=0.5, label='predictions', density=True)
    plt.legend(loc='upper right')
    # plt.xlim(-0.5, 1000)
    #plt.yscale('log')
    plt.show()

    exit()

    print("train error ", train_error)
    print("test error ", test_error)


def svr_run():
    # preprocess_data()
    df = load_data()
    # data_analysis(df)
    support_vector_regression(df)


if __name__ == "__main__":
    # import cProfile
    # import pstats
    # pr = cProfile.Profile()
    # pr.enable()
    svr_run()
    # my_result = svr_run()
    #
    # pr.disable()
    # ps = pstats.Stats(pr).sort_stats('cumtime')
    # ps.print_stats()



