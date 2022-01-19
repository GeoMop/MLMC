import time
import numpy as np
from mlmc.tool import gmsh_io
from mlmc.tool.flow_mc import FlowSim, create_corr_field
from mlmc.tool.flow_mc_2 import FlowSimProcConc
from mlmc.tool.flow_mc_2 import create_corr_field as conc_create_corr_fields
import gstools
from mlmc.random import correlated_field as cf


def conc_rnd_sample_time(mesh_file, corr_field_config):

    start_time = time.process_time()
    mesh_data = FlowSim.extract_mesh(mesh_file)

    fields = conc_create_corr_fields(dim=2, log=corr_field_config["log"], mode_no=10000)

    n_samples = 200
    for i in range(n_samples):
        fields.set_points(mesh_data['points'], mesh_data['point_region_ids'],
                          mesh_data['region_map'])

        fine_input_sample, coarse_input_sample = FlowSimProcConc.generate_random_sample(fields, coarse_step=0,
                                                                                n_fine_elements=len(
                                                                                    mesh_data['points']))
        #print("fine input sample ", fine_input_sample)


    rnd_time = time.process_time() - start_time
    print("rnd_time / n_samples ", rnd_time / n_samples)

    return rnd_time / n_samples


def corr_field_sample_time(mesh_file=None, corr_length_config=None):
    # import matplotlib
    # from matplotlib import ticker, cm
    #matplotlib.rcParams.update({'font.size': 22})

    if corr_length_config.get('02_conc', False):
        return conc_rnd_sample_time(mesh_file, corr_length_config)

    dim = 2
    log = True
    cl = 0.1
    s = 1

    if mesh_file is None:
        #mesh_file = "/home/martin/Sync/Documents/flow123d_results/flow_experiments/Exponential/corr_length_0_01/l_step_0.0055_common_files/mesh.msh"
        #mesh_file = "/home/martin/Documents/metamodels/data/5_ele/cl_0_1_s_1/L5/l_step_0.020196309484414757_common_files/mesh.msh"
        mesh_file = "/home/martin/Documents/metamodels/data/1000_ele/l_step_0.055_common_files/mesh.msh"


    start_time = time.process_time()
    mesh_data = FlowSim.extract_mesh(mesh_file)
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

    n_samples = 200
    for i in range(n_samples):

        fine_input_sample, coarse_input_sample = FlowSim.generate_random_sample(fields, coarse_step=0,
                                                                             n_fine_elements=len(
                                                                                 mesh_data['points']))

        len(fine_input_sample["conductivity"])
        features_log = np.log(fine_input_sample["conductivity"])

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

    #Xfinal, Yfinal = fields.fields[0].correlated_field.points[:, 0],  fields.fields[0].correlated_field.points[:, 1]

    # cont = ax.tricontourf(Xfinal,
    #                     Yfinal,
    #                       fine_input_sample['conductivity'].ravel())#, locator=ticker.LogLocator())

    # fig.colorbar(cont)
    # fig.savefig("cl_{}_var_{}.pdf".format(cl, s ** 2))
    # plt.show()

    # print("fields ", fields)
    # model = gs.Exponential(dim=2, len_scale=cl)
    # srf = gs.SRF(model, mesh_type="unstructed", seed=20170519, mode_no=1000, generator='RandMeth')
    # print("model.var ", model.var)
    # field = srf(
    #     (fields.fields[0].correlated_field.points[:, 0], fields.fields[0].correlated_field.points[:, 1]))
    # srf.vtk_export("field")
    # ax = srf.plot()
    # ax.set_aspect("equal")


if __name__ == "__main__":
    import cProfile
    import pstats
    pr = cProfile.Profile()
    pr.enable()

    corr_file_config = {"02_conc": True, 'log': True}
    mesh_file = "/home/martin/Documents/metamodels/data/mesh_size/02_conc_por/l_step_1.0_common_files/repo.msh"

    # corr_file_config = {"02_conc": False, 'log': True, 'corr_length':0.1, 'sigma':1}
    # mesh_file = "/home/martin/Documents/metamodels/data/1000_ele/l_step_0.055_common_files/mesh.msh"

    my_result = corr_field_sample_time(mesh_file, corr_file_config)

    pr.disable()
    ps = pstats.Stats(pr).sort_stats('cumtime')
    ps.print_stats()

