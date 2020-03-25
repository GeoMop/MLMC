# import mlmc.random.frac_geom as fg
# import numpy as np
#
#
# def _test_frac_geom():
#     np.random.seed(1)
#     box = np.array([[0.0, 0.0],
#                      [2.0, 3.0]])
#
#     n_frac = 50
#     p0 = np.random.rand(n_frac, 2) * (box[1] - box[0]) + box[0]
#     p1 = np.random.rand(n_frac, 2) * (box[1] - box[0]) + box[0]
#     fractures = np.concatenate((p0[:, None, :], p1[:, None, :]), axis=1)
#
#     mesh = fg.make_frac_mesh(box, 0.1, fractures, 0.05)
