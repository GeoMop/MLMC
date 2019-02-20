import os
import sys
import yaml
import numpy as np

src_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(src_path, '..', '..', 'src'))

import mlmc.mlmc
import mlmc.simulation
import mlmc.moments
import flow_mc as flow_mc
import mlmc.correlated_field as cf
import mlmc.estimate

sys.path.append(os.path.join(src_path, '..'))
import base_process


class FlowConcSim(flow_mc.FlowSim):
    """
    Child from FlowSimulation that defines extract method
    """

    def _extract_result(self, sample):
        """
        Extract the observed value from the Flow123d output.
        Get sample from the field restriction, write to the GMSH file, call flow.
        :param sample: mlmc.sample Sample
        :return: None, total flux (float) and overall sample time
        """
        sample_dir = sample.directory
        if os.path.exists(os.path.join(sample_dir, "FINISHED")):
            # extract the flux
            balance_file = os.path.join(sample_dir, "mass_balance.yaml")

            with open(balance_file, "r") as f:
                balance = yaml.load(f)

            # it has to be changed for every new input file or different observation.
            # However in Analysis it is already done in general way.
            flux_regions = ['.surface']
            max_flux = 0.0
            found = False

            for flux_item in balance['data']:
                if 'region' not in flux_item:
                    os.remove(os.path.join(sample_dir, "mass_balance.yaml"))
                    return None

                if flux_item['region'] in flux_regions:
                    out_flux = -float(flux_item['data'][0])
                    if not np.isfinite(out_flux):
                        return np.inf
                    # flux_in = float(flux_item['data'][1])
                    # if flux_in > 1e-10:
                    #    raise Exception("Possitive inflow at outlet region.")
                    max_flux = max(max_flux, out_flux)  # flux field
                    found = True

            if not found:
                raise Exception

            # Get flow123d computing time
            run_time = self.get_run_time(sample_dir)

            return max_flux, run_time
        else:
            return None, 0


class ProcConc(base_process.Process):

    def run(self):
        """
        Run mlmc
        :return: None
        """
        os.makedirs(self.work_dir, mode=0o775, exist_ok=True)

        mlmc_list = []
        for nl in [5]:  # , 2, 3, 4,5, 7, 9]:
            mlmc = self.setup_config(nl, clean=True)
            # self.n_sample_estimate(mlmc)
            self.generate_jobs(mlmc, n_samples=[5, 5, 5, 5, 5])
            mlmc_list.append(mlmc)

        self.all_collect(mlmc_list)

    def setup_config(self, n_levels, clean):
        """
        Set simulation configuration, object for generating correlated fields ...
        :param n_levels: Number of levels
        :param clean: bool
        :return: None
        """
        # Set pbs config, flow123d, gmsh, ...
        self.set_environment_variables()
        output_dir = os.path.join(self.work_dir, "output_{}".format(n_levels))
        # remove existing files
        if clean:
            self.rm_files(output_dir)

        # Init pbs object
        self.create_pbs_object(output_dir, clean)

        por_top = cf.SpatialCorrelatedField(
            corr_exp='gauss',
            dim=2,
            corr_length=0.2,
            mu=-1.0,
            sigma=1.0,
            log=True
        )
        por_bot = cf.SpatialCorrelatedField(
            corr_exp='gauss',
            dim=2,
            corr_length=0.2,
            mu=-1.0,
            sigma=1.0,
            log=True
        )
        water_viscosity = 8.90e-4
        fields = cf.Fields([
            cf.Field('por_top', por_top, regions='ground_0'),
            cf.Field('porosity_top', cf.positive_to_range, ['por_top', 0.02, 0.1], regions='ground_0'),
            cf.Field('por_bot', por_bot, regions='ground_1'),
            cf.Field('porosity_bot', cf.positive_to_range, ['por_bot', 0.01, 0.05], regions='ground_1'),
            cf.Field('porosity_repo', 0.5, regions='repo'),
            cf.Field('factor_top', cf.SpatialCorrelatedField('gauss', mu=1e-8, sigma=1, log=True), regions='ground_0'),
            # conductivity about
            cf.Field('factor_bot', cf.SpatialCorrelatedField('gauss', mu=1e-8, sigma=1, log=True), regions='ground_1'),
            # cf.Field('factor_repo', cf.SpatialCorrelatedField('gauss', mu=1e-10, sigma=1, log=True), regions='repo'),
            cf.Field('conductivity_top', cf.kozeny_carman, ['porosity_top', 1, 'factor_top', water_viscosity],
                     regions='ground_0'),
            cf.Field('conductivity_bot', cf.kozeny_carman, ['porosity_bot', 1, 'factor_bot', water_viscosity],
                     regions='ground_1'),
            # cf.Field('conductivity_repo', cf.kozeny_carman, ['porosity_repo', 1, 'factor_repo', water_viscosity], regions='repo')
            cf.Field('conductivity_repo', 0.001, regions='repo')
        ])

        self.step_range = (1, 0.02)  # finest mesh about 18k elements

        simulation_config = {
            'env': dict(flow123d=self.flow123d, gmsh=self.gmsh, pbs=self.pbs_obj),  # The Environment.
            'output_dir': output_dir,
            'fields': fields,
            'time_factor': 1e7,  # max velocity about 1e-8
            'yaml_file': os.path.join(self.work_dir, '02_conc_tmpl.yaml'),
        # The template with a mesh and field placeholders
            'sim_param_range': self.step_range,  # Range of MLMC simulation parametr. Here the mesh step.
            'geo_file': os.path.join(self.work_dir, 'repo.geo'),
        # The file with simulation geometry (independent of the step)
            # 'field_template': "!FieldElementwise {gmsh_file: \"${INPUT}/%s\", field_name: %s}"
            'field_template': "!FieldElementwise {mesh_data_file: \"$INPUT_DIR$/%s\", field_name: %s}"

        }

        FlowConcSim.total_sim_id = 0

        self.options['output_dir'] = output_dir
        mlmc_obj = mlmc.mlmc.MLMC(n_levels, FlowConcSim.factory(self.step_range, config=simulation_config, clean=clean),
                                  self.step_range, self.options)

        if clean:
            # Create new execution of mlmc
            mlmc_obj.create_new_execution()
        else:
            # Use existing mlmc HDF file
            mlmc_obj.load_from_file()
        return mlmc_obj


if __name__ == "__main__":
    pr = ProcConc()
