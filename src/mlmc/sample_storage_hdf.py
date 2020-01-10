from sample_storage import SampleStorage
import hdf5 as hdf


class HDFStorage(SampleStorage):

    def __init__(self, file_name, work_dir):
        self._hdf_object = hdf.HDF5(file_name=file_name,
                                    work_dir=work_dir)

        self._hdf_object.clear_groups()
        self._hdf_object.init_header(step_range=self.step_range,
                                     n_levels=self._n_levels)

        self._hdf_object.add_level_group()

    def save_results(self, res):
        pass

    def save_result_specification(self, res_spec):
        self._hdf_object.


    def write_data(self):
        pass

    def read_data(self):
        pass
