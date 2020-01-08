import os
import sys
import yaml

src_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(src_path, '..', '..', 'src'))

import mlmc.mlmc
import mlmc.simulation
import mlmc.moments
import flow_mc as flow_mc
import mlmc.correlated_field as cf
from mlmc import estimate

sys.path.append(os.path.join(src_path, '..'))
import base_process


class FlowProcSim(flow_mc.FlowSim):
    """
    Child from FlowSimulation that defines extract method
    """

    def _extract_result(self, sample):
        """
        Extract the observed value from the Flow123d output.
        :param sample: Sample instance
        :return: None, inf or water balance result (float) and overall sample time
        """
        sample_dir = sample.directory
        if os.path.exists(os.path.join(sample_dir, "FINISHED")):
            # try:
            # extract the flux
            balance_file = os.path.join(sample_dir, "water_balance.yaml")

            with open(balance_file, "r") as f:
                balance = yaml.load(f)

            flux_regions = ['.bc_outflow']
            total_flux = 0.0
            found = False
            for flux_item in balance['data']:
                if flux_item['time'] > 0:
                    break

                if flux_item['region'] in flux_regions:
                    flux = float(flux_item['data'][0])
                    flux_in = float(flux_item['data'][1])
                    if flux_in > 1e-10:
                        raise Exception("Possitive inflow at outlet region.")
                    total_flux += flux  # flux field
                    found = True

            # Get flow123d computing time
            run_time = self.get_run_time(sample_dir)

            if not found:
                raise Exception
            return -total_flux, run_time
        else:
            return None, 0


class CondField(base_process.Process):

    def run(self):
        """
        Run mlmc
        :return: None
        """
        os.makedirs(self.work_dir, mode=0o775, exist_ok=True)

        self.n_moments = 10

        mlmc_list = []
        for nl in [1]:  # , 2, 3, 4,5, 7, 9]:
            mlmc = self.setup_config(nl, clean=True)
            # self.n_sample_estimate(mlmc)
            self.generate_jobs(mlmc, n_samples=[8])
            mlmc_list.append(mlmc)

        self.all_collect(mlmc_list)

    def process(self):
        """
        Use collected data
        :return: None
        """
        assert os.path.isdir(self.work_dir)
        mlmc_est_list = [ 1,3,5,7]

        import time
        for nl in mlmc_est_list:  # high resolution fields
            start = time.time()
            mlmc = self.setup_config(nl, clean=False)
            print("celkový čas ", time.time() - start)
            # Use wrapper object for working with collected data
            mlmc_est = estimate.Estimate(mlmc)
            mlmc_est_list.append(mlmc_est)
        cl = estimate.CompareLevels(mlmc_est_list,
                           output_dir=src_path,
                           quantity_name="Q [m/s]",
                           moment_class=Legendre,
                           log_scale=False,
                           n_moments=21, )

        self.process_analysis(cl)


    def setup_config(self, n_levels, clean):
        """
        Simulation dependent configuration
        :param n_levels: Number of levels
        :param clean: bool, If True remove existing files
        :return: mlmc.MLMC instance
        """

        fields = cf.Fields([
            cf.Field('conductivity', cf.FourierSpatialCorrelatedField('gauss', dim=2, corr_length=0.125, log=True)),
        ])

        # Set pbs config, flow123d, gmsh, ...
        self.set_environment_variables()
        output_dir = os.path.join(self.work_dir, "output_{}".format(n_levels))
        # remove existing files
        if clean:
            self.rm_files(output_dir)

        # Init pbs object
        self.create_pbs_object(output_dir, clean)

        simulation_config = {
            'env': dict(flow123d=self.flow123d, gmsh=self.gmsh, pbs=self.pbs_obj),  # The Environment.
            'output_dir': output_dir,
            'fields': fields,  # correlated_field.FieldSet object
            'yaml_file': os.path.join(self.work_dir, '01_conductivity.yaml'),  # The template with a mesh and field placeholders
            'sim_param_range': self.step_range,  # Range of MLMC simulation parametr. Here the mesh step.
            'geo_file': os.path.join(self.work_dir, 'square_1x1.geo'),  # The file with simulation geometry (independent of the step)
            # 'field_template': "!FieldElementwise {mesh_data_file: \"${INPUT}/%s\", field_name: %s}"
            'field_template': "!FieldElementwise {mesh_data_file: \"$INPUT_DIR$/%s\", field_name: %s}"
        }

        FlowProcSim.total_sim_id = 0

        self.options['output_dir'] = output_dir
        mlmc_obj = mlmc.mlmc.MLMC(n_levels, FlowProcSim.factory(self.step_range, config=simulation_config, clean=clean),
                                  self.step_range, self.options)

        if clean:
            # Create new execution of mlmc
            # Create new execution of mlmc
            mlmc_obj.create_new_execution()
        else:
            # Use existing mlmc HDF file
            mlmc_obj.load_from_file()
        return mlmc_obj


if __name__ == "__main__":
    pr = CondField()
