import time
import numpy as np
from mlmc.tool import gmsh_io
from mlmc.tool.flow_mc import FlowSim, create_corr_field
from mlmc.tool.flow_mc_2 import FlowSimProcConc
import gstools
from mlmc.random import correlated_field as cf
import matplotlib.pyplot as plt
from mlmc.metamodel.create_graph import get_node_features, extract_mesh_gmsh_io


def create_corr_fields(model='gauss', corr_length=0.125, dim=2, log=True, factor_sigma=1, sigma=1, mode_no=1000,
                       por_top_mean=-1.0, por_bot_mean=-1.0, por_top_sigma=1, por_bot_sigma=1,
                       por_top_len_scale=0.2, por_bot_len_scale=0.2, factor_top_mean=1e-8, factor_bot_mean=1e-8):
    """
    Create random fields
    :return:
    """
    # por_top = cf.SpatialCorrelatedField(
    #     corr_exp='gauss',
    #     dim=2,
    #     corr_length=0.2,
    #     mu=-1.0,
    #     sigma=1.0,
    #     log=True
    # )
    #
    # print("por top ", por_top)


    por_top = cf.GSToolsSpatialCorrelatedField(gstools.Gaussian(dim=2,  len_scale=por_top_len_scale),
                                               log=log, mean=por_top_mean, sigma=por_top_sigma, mode_no=mode_no)

    #print("por top gstools ", por_top_gstools)

    # por_bot = cf.SpatialCorrelatedField(
    #     corr_exp='gauss',
    #     dim=2,
    #     corr_length=0.2,
    #     mu=-1.0,
    #     sigma=1.0,
    #     log=True
    # )

    por_bot = cf.GSToolsSpatialCorrelatedField(gstools.Gaussian(dim=2, len_scale=por_bot_len_scale),
                                               log=log, mean=por_bot_mean, sigma=por_bot_sigma, mode_no=mode_no)

    #por_bot = gstools.Gaussian(dim=dim,  len_scale=0.2, mu=-1.0, sigma=1.0, log=True)

    water_viscosity = 8.90e-4

    factor_top_model = gstools.Gaussian(dim=dim,  len_scale=1)
    factor_bot_model = gstools.Gaussian(dim=dim, len_scale=1)

    fields = cf.Fields([
        cf.Field('por_top', por_top, regions='ground_0'),
        cf.Field('porosity_top', cf.positive_to_range, ['por_top', 0.02, 0.1], regions='ground_0'),
        cf.Field('por_bot', por_bot, regions='ground_1'),
        cf.Field('porosity_bot', cf.positive_to_range, ['por_bot', 0.01, 0.05], regions='ground_1'),
        cf.Field('porosity_repo', 0.5, regions='repo'),
        #cf.Field('factor_top', cf.SpatialCorrelatedField('gauss', mu=1e-8, sigma=1, log=True), regions='ground_0'),

        cf.Field('factor_top', cf.GSToolsSpatialCorrelatedField(factor_top_model, log=log, mean=factor_top_mean, sigma=factor_sigma, mode_no=mode_no),
                 regions='ground_0'),

        #cf.Field('factor_top', gstools.Gaussian(len_scale=1, mu=1e-8, sigma=1.0, log=True), regions='ground_0'),
        # conductivity about
        #cf.Field('factor_bot', cf.SpatialCorrelatedField('gauss', mu=1e-8, sigma=1, log=True), regions='ground_1'),
        #cf.Field('factor_bot', gstools.Gaussian(len_scale=1, mu=1e-8, sigma=1, log=True), regions='ground_1'),
        cf.Field('factor_bot',
                 cf.GSToolsSpatialCorrelatedField(factor_bot_model, log=log, mean=factor_bot_mean, sigma=factor_sigma, mode_no=mode_no),
                 regions='ground_1'),

        # cf.Field('factor_repo', cf.SpatialCorrelatedField('gauss', mu=1e-10, sigma=1, log=True), regions='repo'),
        cf.Field('conductivity_top', cf.kozeny_carman, ['porosity_top', 1, 'factor_top', water_viscosity],
                 regions='ground_0'),
        cf.Field('conductivity_bot', cf.kozeny_carman, ['porosity_bot', 1, 'factor_bot', water_viscosity],
                 regions='ground_1'),
        # cf.Field('conductivity_repo', cf.kozeny_carman, ['porosity_repo', 1, 'factor_repo', water_viscosity], regions='repo')
        cf.Field('conductivity_repo', 0.001, regions='repo')
    ])

    return fields



# def corr_field_sample_time(mesh_file=None, corr_length_config=None):
#     # import matplotlib
#     # from matplotlib import ticker, cm
#     #matplotlib.rcParams.update({'font.size': 22})
#
#     if corr_length_config.get('02_conc', False):
#         return conc_rnd_sample_time(mesh_file, corr_length_config)
#
#     dim = 2
#     log = True
#     cl = 0.1
#     s = 1
#
#     if mesh_file is None:
#         #mesh_file = "/home/martin/Sync/Documents/flow123d_results/flow_experiments/Exponential/corr_length_0_01/l_step_0.0055_common_files/mesh.msh"
#         #mesh_file = "/home/martin/Documents/metamodels/data/5_ele/cl_0_1_s_1/L5/l_step_0.020196309484414757_common_files/mesh.msh"
#         mesh_file = "/home/martin/Documents/metamodels/data/1000_ele/l_step_0.055_common_files/mesh.msh"
#
#
#     start_time = time.process_time()
#     mesh_data = FlowSim.extract_mesh(mesh_file)
#     if corr_length_config is not None:
#         fields = create_corr_field(model="exp", dim=dim,
#                                    sigma=corr_length_config['sigma'],
#                                    corr_length=corr_length_config['corr_length'],
#                                    log=corr_length_config['log'])
#     else:
#         fields = create_corr_field(model="exp", dim=dim,
#                                    sigma=s,
#                                    corr_length=cl,
#                                    log=log)
#     # # Create fields both fine and coarse
#     fields = FlowSim.make_fields(fields, mesh_data, None)
#
#     n_samples = 200
#     for i in range(n_samples):
#
#         fine_input_sample, coarse_input_sample = FlowSim.generate_random_sample(fields, coarse_step=0,
#                                                                              n_fine_elements=len(
#                                                                                  mesh_data['points']))
#
#         len(fine_input_sample["conductivity"])
#         features_log = np.log(fine_input_sample["conductivity"])
#
#         # print("conductivity mean ", np.mean(fine_input_sample["conductivity"]))
#         # print("conductivity var ", np.var(fine_input_sample["conductivity"]))
#         output = 1
#         #
#         # print("fine input sample ", fine_input_sample["conductivity"].shape)
#         #
#         # gmsh_io.GmshIO().write_fields('fields_sample.msh', mesh_data['ele_ids'], fine_input_sample)
#         #
#         # mesh = gmsh_io.GmshIO('fields_sample.msh')
#         # element_data = mesh.current_elem_data
#         # features = list(element_data.values())
#         # print("features ", np.array(features).shape)
#
#     rnd_time = time.process_time() - start_time
#     print("rnd_time / n_samples ", rnd_time / n_samples)
#     return rnd_time / n_samples
#
#     #Xfinal, Yfinal = fields.fields[0].correlated_field.points[:, 0],  fields.fields[0].correlated_field.points[:, 1]
#
#     # cont = ax.tricontourf(Xfinal,
#     #                     Yfinal,
#     #                       fine_input_sample['conductivity'].ravel())#, locator=ticker.LogLocator())
#
#     # fig.colorbar(cont)
#     # fig.savefig("cl_{}_var_{}.pdf".format(cl, s ** 2))
#     # plt.show()
#
#     # print("fields ", fields)
#     # model = gs.Exponential(dim=2, len_scale=cl)
#     # srf = gs.SRF(model, mesh_type="unstructed", seed=20170519, mode_no=1000, generator='RandMeth')
#     # print("model.var ", model.var)
#     # field = srf(
#     #     (fields.fields[0].correlated_field.points[:, 0], fields.fields[0].correlated_field.points[:, 1]))
#     # srf.vtk_export("field")
#     # ax = srf.plot()
#     # ax.set_aspect("equal")


def conc_corr_field(mesh_file, corr_field_config):
    mesh_data = FlowSim.extract_mesh(mesh_file)
    feature_names = [['conductivity_top', 'conductivity_bot', 'conductivity_repo']]

    # por_top_mean = -1.0
    # por_bot_mean = -1.0
    # por_top_sigma = 1
    # por_bot_sigma = 1
    # factor_sigma = 1
    # por_top_len_scale = 1
    # por_bot_len_scale = 1
    # factor_top_mean = 1e-8
    # factor_bot_mean = 1e-8

    all_features = []
    log_all_features = []
    n_samples = 1000
    for i in range(n_samples):
        fields = create_corr_fields(dim=2, log=corr_field_config["log"],
                                    por_top_mean=corr_field_config["por_top_mean"],
                                    por_bot_mean=corr_field_config["por_bot_mean"],
                                    por_top_sigma=corr_field_config["por_top_sigma"],
                                    por_bot_sigma=corr_field_config["por_bot_sigma"],
                                    factor_sigma=corr_field_config["factor_sigma"],
                                    mode_no=corr_field_config["mode_no"],
                                    por_top_len_scale=corr_field_config["por_top_len_scale"],
                                    por_bot_len_scale=corr_field_config["por_bot_len_scale"],
                                    factor_top_mean=corr_field_config["factor_top_mean"],
                                    factor_bot_mean=corr_field_config["factor_bot_mean"]
                                    )

        fields.set_points(mesh_data['points'], mesh_data['point_region_ids'],
                          mesh_data['region_map'])

        fine_input_sample, coarse_input_sample = FlowSimProcConc.generate_random_sample(fields, coarse_step=0,
                                                                                        n_fine_elements=len(
                                                                                            mesh_data['points']))
        fields_file = 'fields_sample.msh'
        gmsh_io.GmshIO().write_fields(fields_file, mesh_data['ele_ids'], fine_input_sample)


        features = get_node_features(fields_file, feature_names)

        #features = np.log(features)
        all_features.append(features)
        log_all_features.append(np.log(features))

    plot_rescale(all_features, log_all_features, mesh_file)

        # mesh_data = extract_mesh_gmsh_io(mesh_file, get_points=True)
        # points = mesh_data['points']
        # X = points[:, 0]
        # Y = points[:, 1]
        # fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        #
        # cont = ax.tricontourf(X, Y, features.ravel(), levels=32)
        # fig.colorbar(cont)
        # plt.title("input")
        # plt.show()


def cond_corr_field(mesh_file=None, corr_length_config=None):
        # import matplotlib
        # from matplotlib import ticker, cm
        # matplotlib.rcParams.update({'font.size': 22})
        dim = 2
        log = True
        cl = 0.1
        s = 1

        feature_names = [['conductivity']]

        if mesh_file is None:
            # mesh_file = "/home/martin/Sync/Documents/flow123d_results/flow_experiments/Exponential/corr_length_0_01/l_step_0.0055_common_files/mesh.msh"
            # mesh_file = "/home/martin/Documents/metamodels/data/5_ele/cl_0_1_s_1/L5/l_step_0.020196309484414757_common_files/mesh.msh"
            mesh_file = "/home/martin/Documents/metamodels/data/1000_ele/l_step_0.055_common_files/mesh.msh"
            mesh_file = "/home/martin/Documents/metamodels/data/cl_0_1_s_1/l_step_0.27232698153315_common_files/mesh.msh"  # L2 10.5 s
            mesh_file = "/home/martin/Documents/metamodels/data/cl_0_1_s_1/l_step_0.07416198487095663_common_files/mesh.msh"  # L3 12s

        start_time = time.process_time()
        mesh_data = FlowSim.extract_mesh(mesh_file)

        all_features = []
        log_all_features = []
        n_samples = 1000
        for i in range(n_samples):
            if corr_length_config is not None:
                fields = create_corr_field(model="exp", dim=dim,
                                           sigma=corr_length_config['sigma'],
                                           corr_length=corr_length_config['corr_length'],
                                           log=corr_length_config['log'])
            else:
                fields = create_corr_field(model="exp", dim=dim,
                                           sigma=s,
                                           corr_length=cl,
                                           log=log)

            # # Create fields both fine and coarse
            fields = FlowSim.make_fields(fields, mesh_data, None)

            # len(fine_input_sample["conductivity"])
            # features_log = np.log(fine_input_sample["conductivity"])

            fields.set_points(mesh_data['points'], mesh_data['point_region_ids'],
                              mesh_data['region_map'])

            fine_input_sample, coarse_input_sample = FlowSim.generate_random_sample(fields, coarse_step=0,
                                                                                    n_fine_elements=len(
                                                                                        mesh_data['points']))

            fields_file = 'fields_sample.msh'
            gmsh_io.GmshIO().write_fields(fields_file, mesh_data['ele_ids'], fine_input_sample)


            features = get_node_features(fields_file, feature_names)

            #features = np.log(features)
            all_features.append(features)
            log_all_features.append(np.log(features))


            ####
            # Plot random field
            ####
            # fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            # mesh_data = extract_mesh_gmsh_io(mesh_file, get_points=True)
            # points = mesh_data['points']
            # X = points[:, 0]
            # Y = points[:, 1]
            # cont = ax.tricontourf(X, Y, features.ravel(), levels=32)
            # fig.colorbar(cont)
            # plt.title("input")
            # plt.show()


        plot_rescale(all_features, log_all_features, mesh_file)

        # print("conductivity mean ", np.mean(fine_input_sample["conductivity"]))
        # print("conductivity var ", np.var(fine_input_sample["conductivity"]))
        output = 1
        #
        # print("fine input sample ", fine_input_sample["conductivity"].shape)
        #
        # gmsh_io.GmshIO().write_fields('fields_sample.msh', mesh_data['ele_ids'], fine_input_sample)
        #
        # mesh = gmsh_io.GmshIO('fields_sample.msh')
        # element_data = mesh.current_elem_data
        # features = list(element_data.values())
        # print("features ", np.array(features).shape)

        rnd_time = time.process_time() - start_time
        print("rnd_time / n_samples ", rnd_time / n_samples)
        return rnd_time / n_samples


def plot_rescale(all_features, log_all_features, mesh_file):
    mean_features = np.mean(all_features, axis=0)
    variance_features = np.var(all_features, axis=0)
    print('mean features ', mean_features)
    print("variance features ", variance_features)

    min_features = np.min(all_features, axis=0)
    max_features = np.max(all_features, axis=0)

    print("min features ", min_features)
    print("max features ", max_features)

    log_mean_features = np.mean(log_all_features, axis=0)
    log_variance_features = np.var(log_all_features, axis=0)
    print('log mean features ', log_mean_features)
    print("log variance features ", log_variance_features)

    log_min_features = np.min(log_all_features, axis=0)
    log_max_features = np.max(log_all_features, axis=0)

    print("log min features ", log_min_features)
    print("log max features ", log_max_features)

    num = 1
    for features, log_features in zip(all_features[:num], log_all_features[:num]):
        print("features[:10] ", features[:10])
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        mesh_data = extract_mesh_gmsh_io(mesh_file, get_points=True)
        points = mesh_data['points']
        X = points[:, 0]
        Y = points[:, 1]
        print("features shape ", features.shape)
        cont = ax.tricontourf(X, Y, features.ravel(), levels=32)
        fig.colorbar(cont)
        plt.title("original features")
        fig.savefig("original_features.pdf")
        plt.show()

        # features -= mean_features
        # print("features - mean ", features)
        # features /= variance_features

        ##################
        ## log features ##
        ##################
        # print("features - min_features ", features - min_features)
        # features = (features - min_features) / (max_features - min_features)
        # print("final features ", features)
        # pl_log_features = np.nan_to_num(log_features)
        # print("log features ", pl_log_features)

        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        mesh_data = extract_mesh_gmsh_io(mesh_file, get_points=True)
        points = mesh_data['points']
        X = points[:, 0]
        Y = points[:, 1]
        cont = ax.tricontourf(X, Y, log_features.ravel(), levels=32)
        fig.colorbar(cont)
        plt.title("log features")
        fig.savefig("log_features.pdf")
        plt.show()

        ###################
        ## norm features ##
        ###################
        # print("features - min_features ", features - min_features)
        norm_features = (features - min_features) / (max_features - min_features)
        # print("final features ", features)
        norm_features = np.nan_to_num(norm_features)
        print("final features ", norm_features)

        # print("final features nan to num ", features)
        #
        # print("final features ", features.shape)

        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        mesh_data = extract_mesh_gmsh_io(mesh_file, get_points=True)
        points = mesh_data['points']
        X = points[:, 0]
        Y = points[:, 1]
        cont = ax.tricontourf(X, Y, norm_features.ravel(), levels=32)
        fig.colorbar(cont)
        plt.title("norm features")
        fig.savefig("norm_features.pdf")
        plt.show()

        #####################
        ## scaled features ##
        #####################
        # print("features - min_features ", features - min_features)
        scaled_features = (features - mean_features) / variance_features

        # print("final features ", features)
        scaled_features = np.nan_to_num(scaled_features)
        print("scaled features ", scaled_features)

        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        mesh_data = extract_mesh_gmsh_io(mesh_file, get_points=True)
        points = mesh_data['points']
        X = points[:, 0]
        Y = points[:, 1]
        cont = ax.tricontourf(X, Y, scaled_features.ravel(), levels=32)
        fig.colorbar(cont)
        plt.title("scaled features")
        fig.savefig("scaled_features.pdf")
        plt.show()

        #######################
        ## norm log features ##
        #######################
        # print("features - min_features ", features - min_features)
        log_norm_features = (log_features - log_min_features) / (log_max_features - log_min_features)
        # print("final features ", features)
        log_norm_features = np.nan_to_num(log_norm_features)
        print("log norm features ", log_norm_features)

        # print("final features nan to num ", features)
        #
        # print("final features ", features.shape)

        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        mesh_data = extract_mesh_gmsh_io(mesh_file, get_points=True)
        points = mesh_data['points']
        X = points[:, 0]
        Y = points[:, 1]
        cont = ax.tricontourf(X, Y, log_norm_features.ravel(), levels=32)
        fig.colorbar(cont)
        plt.title("norm log features")
        fig.savefig("norm_log_features.pdf")
        plt.show()

        #########################
        ## scaled log features ##
        #########################
        # print("features - min_features ", features - min_features)
        log_scaled_features = (log_features - log_mean_features) / log_variance_features

        # print("final features ", features)
        log_scaled_features = np.nan_to_num(log_scaled_features)
        print("scaled features ", log_scaled_features)

        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        mesh_data = extract_mesh_gmsh_io(mesh_file, get_points=True)
        points = mesh_data['points']
        X = points[:, 0]
        Y = points[:, 1]
        cont = ax.tricontourf(X, Y, log_scaled_features.ravel(), levels=32)
        fig.colorbar(cont)
        plt.title("scaled log features")
        fig.savefig("scaled_log_features.pdf")
        plt.show()


if __name__ == "__main__":

    case = "02_conc_cond_case_5"
    case = "cl_0_1_s_1"

    if case == "cl_0_1_s_1":
        corr_file_config = {"02_conc": False, 'sigma': 1, 'corr_length': 0.1, 'log': True}
        mesh_file = "/home/martin/Documents/metamodels/data/mesh_size/l_step_0.07416198487095663_common_files/mesh.msh"

        #mesh_file = "/home/martin/Documents/metamodels/data/mesh_size/l_step_0.020196309484414757_common_files/mesh.msh"

        cond_corr_field(mesh_file, corr_file_config)

    elif case == "cl_0_3_s_4":
        corr_file_config = {"02_conc": False, 'sigma': 4, 'corr_length': 0.3, 'log': True}
        mesh_file = "/home/martin/Documents/metamodels/data/mesh_size/l_step_0.07416198487095663_common_files/mesh.msh"
        cond_corr_field(mesh_file, corr_file_config)

    elif case == "02_conc_cond_case_1":
        corr_file_config = {"02_conc": True, 'log': True, 'mode_no': 1000,
                                                            'por_top_mean': -1.0,
                                                            'por_bot_mean': -1.0,
                                                            'por_top_sigma': 1,
                                                            'por_bot_sigma': 1,
                                                            'factor_sigma': 1,
                                                            'por_top_len_scale': 0.2,
                                                            'por_bot_len_scale': 0.2,
                                                            'factor_top_mean': 1e-8,
                                                            'factor_bot_mean': 1e-8}

        mesh_file = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/l_step_0.1414213562373095_common_files/repo.msh"
        conc_corr_field(mesh_file, corr_file_config)

    elif case == "02_conc_cond_case_2":
        corr_file_config = {"02_conc": True, 'log': True, 'mode_no': 10000,
                            'por_top_mean': -1.0,
                            'por_bot_mean': -1.0,
                            'por_top_sigma': 10,
                            'por_bot_sigma': 10,
                            'factor_sigma': 1,
                            'por_top_len_scale': 0.2,
                            'por_bot_len_scale': 0.2,
                            'factor_top_mean': 1e-8,
                            'factor_bot_mean': 1e-8}

        mesh_file = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/l_step_0.1414213562373095_common_files/repo.msh"
        conc_corr_field(mesh_file, corr_file_config)

    elif case == "02_conc_cond_case_3":
        corr_file_config = {"02_conc": True, 'log': True, 'mode_no': 10000,
                            'por_top_mean': -1.0,
                            'por_bot_mean': -1.0,
                            'por_top_sigma': 0.1,
                            'por_bot_sigma': 0.1,
                            'factor_sigma': 1,
                            'por_top_len_scale': 0.2,
                            'por_bot_len_scale': 0.2,
                            'factor_top_mean': 1e-8,
                            'factor_bot_mean': 1e-8}

        mesh_file = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/l_step_0.1414213562373095_common_files/repo.msh"
        conc_corr_field(mesh_file, corr_file_config)

    elif case == "02_conc_cond_case_4":
        corr_file_config = {"02_conc": True, 'log': True, 'mode_no': 10000,
                            'por_top_mean': -1.0,
                            'por_bot_mean': -1.0,
                            'por_top_sigma': 1,
                            'por_bot_sigma': 1,
                            'factor_sigma': 1,
                            'por_top_len_scale': 1,
                            'por_bot_len_scale': 1,
                            'factor_top_mean': 1e-8,
                            'factor_bot_mean': 1e-8}

        mesh_file = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/l_step_0.1414213562373095_common_files/repo.msh"
        conc_corr_field(mesh_file, corr_file_config)

    elif case == "02_conc_cond_case_5":
        corr_file_config = {"02_conc": True, 'log': True, 'mode_no': 10000,
                            'por_top_mean': -1.0,
                            'por_bot_mean': -1.0,
                            'por_top_sigma': 1,
                            'por_bot_sigma': 1,
                            'factor_sigma': 1,
                            'por_top_len_scale': 0.01,
                            'por_bot_len_scale': 0.01,
                            'factor_top_mean': 1e-8,
                            'factor_bot_mean': 1e-8}

        mesh_file = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_cond/l_step_0.1414213562373095_common_files/repo.msh"
        conc_corr_field(mesh_file, corr_file_config)



    ###


    #corr_file_config = {"02_conc": False, 'log': True, 'corr_length':0.1, 'sigma':1}
    #mesh_file = "/home/martin/Documents/metamodels/data/1000_ele/l_step_0.055_common_files/mesh.msh"


    ###############################
    ### generate conc random sample
    #### 02_conc
    #conc_corr_field(mesh_file, corr_file_config)

    ##
    #### 01_cond_field

