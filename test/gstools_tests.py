import numpy as np
import gstools as gs
import mlmc.random.correlated_field as cf
import matplotlib.pyplot as plt


def gstools_test():

    seed = gs.random.MasterRNG(19970221)
    rng = np.random.RandomState(seed())
    x = rng.randint(0, 100, size=10000)
    y = rng.randint(0, 100, size=10000)

    # objevuje se pravidelna struktura i pro velky pocet bodu na intervalu 0,1
    x = rng.random_sample(size=10000)
    y = rng.random_sample(size=10000)

    corr_length = 1
    len_scale = corr_length# * 2 * np.pi
    sigma = 1

    model = gs.Exponential(dim=2, var=sigma**2, len_scale=[len_scale], mode_no=1000)
    print("model ", model)
    srf = gs.SRF(model, seed=20170519)
    field = srf((x, y))
    srf.vtk_export("field")
    # Or create a PyVista dataset
    # mesh = srf.to_pyvista()
    ax = srf.plot()
    ax.set_aspect("equal")
    # Jakmile je korelacni delka na urovni desetiny intervalu, tak se objevuji pravidelnosti


def corr_field():
    from mlmc.tool import gmsh_io
    from mlmc.tool.flow_mc import FlowSim, create_corr_field
    import matplotlib
    from matplotlib import ticker, cm
    #matplotlib.rcParams.update({'font.size': 22})
    dim = 2
    log = False
    corr_lengths = [0.1]
    # sigma = [1, 2, 4]
    sigma = [1]

    #mesh_file = "/home/martin/Sync/Documents/flow123d_results/flow_experiments/Exponential/corr_length_0_1/sigma_1/l_step_0.01_common_files/mesh.msh"
    #mesh_file = "/home/martin/Sync/Documents/flow123d_results/flow_experiments/Exponential/corr_length_0_01/l_step_0.0055_common_files/mesh.msh"
    mesh_file = "/home/martin/Sync/Documents/flow123d_results/flow_experiments/Exponential/cl_0_01/svd/l_step_0.015_common_files/mesh.msh"
    fourier = False
    svd = False

    for cl in corr_lengths:
        for s in sigma:
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            mesh_data = FlowSim.extract_mesh(mesh_file)

            if fourier:
                fields = cf.Fields([cf.Field('conductivity', cf.FourierSpatialCorrelatedField('exp', dim=dim,
                                                                                              corr_length=cl,
                                                                                              sigma=s,
                                                                                              log=log))])
            elif svd:
                conductivity = dict(
                    mu=0.0,
                    sigma=s,
                    corr_exp='exp',
                    dim=2,
                    corr_length=cl,
                    log=log
                )
                fields = cf.Fields([cf.Field("conductivity", cf.SpatialCorrelatedField(**conductivity))])
            else:
                fields = create_corr_field(model="exp", dim=dim, sigma=s, corr_length=cl, log=log)


            # # Create fields both fine and coarse
            fields = FlowSim.make_fields(fields, mesh_data, None)

            fine_input_sample, coarse_input_sample = FlowSim.generate_random_sample(fields, coarse_step=0,
                                                                                     n_fine_elements=len(
                                                                                         mesh_data['points']))

            gmsh_io.GmshIO().write_fields('fields_sample.msh', mesh_data['ele_ids'], fine_input_sample)

            Xfinal, Yfinal = fields.fields[0].correlated_field.points[:, 0],  fields.fields[0].correlated_field.points[:, 1]

            cont = ax.tricontourf(Xfinal,
                                Yfinal,
                                  fine_input_sample['conductivity'].ravel())#, locator=ticker.LogLocator())

            fig.colorbar(cont)
            fig.savefig("cl_{}_var_{}.pdf".format(cl, s ** 2))
            plt.show()

            print("fields ", fields)
            model = gs.Exponential(dim=2, len_scale=cl)
            srf = gs.SRF(model, mesh_type="unstructed", seed=20170519, mode_no=1000, generator='RandMeth')
            print("model.var ", model.var)
            field = srf(
                (fields.fields[0].correlated_field.points[:, 0], fields.fields[0].correlated_field.points[:, 1]))
            srf.vtk_export("field")
            ax = srf.plot()
            ax.set_aspect("equal")


def fourier_analysis():
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt

    img = cv2.imread('messi5.jpg', 0)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == "__main__":
    import cProfile
    import pstats
    pr = cProfile.Profile()
    pr.enable()

    #my_result = gstools_test()
    my_result = corr_field()

    pr.disable()
    ps = pstats.Stats(pr).sort_stats('cumtime')
    ps.print_stats()


