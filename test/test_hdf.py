import os
import shutil
import h5py
import numpy as np

import mlmc.tool.hdf5


"""
test mlmc/tool/hdf5.py methods
"""


def test_hdf5():
    work_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_test_tmp')
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)
    file_path = os.path.join(work_dir, "mlmc_test.hdf5")
    hdf_obj = mlmc.tool.hdf5.HDF5(file_path, load_from_file=False)

    obligatory_attributes = ['version', 'level_parameters']
    init_header(hdf_obj, obligatory_attributes)

    levels = ['1', '2', '3', '7', '8', '9']
    add_level_group(hdf_obj, levels)

    load_from_file(hdf_obj, obligatory_attributes)

    clear_groups(hdf_obj)


def init_header(hdf_obj, obligatory_attributes):
    """
    Initialize hdf file
    :param hdf_obj: mlmc.HDF5 instance
    :param obligatory_attributes: each mlmc.HDF5 hdf file has these attributes
    :return:
    """
    step_range = (0.99, 0.00001)

    hdf_obj.init_header(step_range)

    # Check if all obligatory attributes are actually in HDF5 file
    with h5py.File(hdf_obj.file_name, "r") as hdf_file:
        assert all(attr_name in hdf_file.attrs for attr_name in obligatory_attributes)
        # Group for levels was created
        assert 'Levels' in hdf_file


def add_level_group(hdf_obj, levels):
    """
    Test add level group to file
    :param hdf_obj: HDF5 instance
    :param levels: list of level ids as str
    :return: None
    """
    for level_id in levels:
        hdf_obj.add_level_group(level_id)

    with h5py.File(hdf_obj.file_name, "r") as hdf_file:
        assert all(level_id in hdf_file['Levels'] for level_id in levels)


def clear_groups(hdf_obj):
    """
    Cleare groups in HDF5 - now there is just one group 'Levels'
    :param hdf_obj:
    :return: None
    """
    hdf_obj.clear_groups()

    with h5py.File(hdf_obj.file_name, "r") as hdf_file:
        assert 'Levels' not in hdf_file


def load_from_file(hdf_obj, obligatory_attributes):
    """
    Test loading data from existing file
    :param hdf_obj: mlmc.HDF5 instance
    :param obligatory_attributes: each mlmc.HDF5 hdf file has these attributes
    :return: None
    """
    hdf_obj.load_from_file()
    assert all(attr in hdf_obj.__dict__ for attr in obligatory_attributes)


SCHEDULED_SAMPLES = ['L00_S0000000', 'L00_S0000001', 'L00_S0000002', 'L00_S0000003', 'L00_S0000004']

RESULT_DATA_DTYPE = [("value", np.float), ("time", np.float)]

COLLECTED_SAMPLES = np.array([['L00S0000000', (np.array([10, 20]), np.array([5, 6]))],
                     ['L00S0000001', (np.array([1, 2]), np.array([50, 60]))]])



def test_level_group():
    """
    Test mlmc.tool.hdf.LevelGroup methods
    :return: None
    """
    work_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_test_tmp')
    job_dir = os.path.join(work_dir, 'jobs')
    file_name = os.path.join(work_dir, 'level_test.hdf5')

    level_group_path = 'Levels/1'
    level_id = 1

    # Create hdf file
    with h5py.File(file_name, "w") as hdf_file:
        hdf_file.create_group(level_group_path)

    # Create LevelGroup instance
    hdf_level_group = mlmc.tool.hdf5.LevelGroup(file_name, level_group_path, level_id, job_dir)

    with h5py.File(file_name, "r") as hdf_file:
        assert hdf_file[level_group_path].attrs['level_id'] == level_id == hdf_level_group.level_id

    make_dataset(hdf_level_group)

    make_group_datasets(hdf_level_group)

    append_dataset(hdf_level_group)

    scheduled(hdf_level_group)

    collected(hdf_level_group)


def make_dataset(hdf_level_group, dset_name="test"):
    """
    Test dataset creating
    :param hdf_level_group: mlmc.tool.hdf.LevelGroup instance
    :return: None
    """
    name = hdf_level_group._make_dataset(name=dset_name, shape=(0,), dtype=np.int32, maxshape=(None,), chunks=True)

    assert dset_name == name
    with h5py.File(hdf_level_group.file_name, "r") as hdf_file:
        assert dset_name in hdf_file[hdf_level_group.level_group_path]


def make_group_datasets(hdf_level_group):
    """
    Test if all necessary dataset were created
    :param hdf_level_group: mlmc.tool.hdf.LevelGroup instance
    :return: None
    """
    # Created datasets
    datasets = [attr_prop['name'] for _, attr_prop in mlmc.tool.hdf5.LevelGroup.COLLECTED_ATTRS.items()]
    datasets.extend([hdf_level_group.scheduled_dset, hdf_level_group.failed_dset])
    hdf_level_group._make_groups_datasets()


    with h5py.File(hdf_level_group.file_name, "r") as hdf_file:
        assert all(dset in hdf_file[hdf_level_group.level_group_path] for dset in datasets)


def append_dataset(hdf_level_group, dset_name='test'):
    """
    Test append dataset
    :param hdf_level_group: mlmc.tool.hdf.LevelGroup instance
    :param dset_name: Name of dataset to use
    :return: None
    """
    values = np.random.randint(100, size=10)

    with h5py.File(hdf_level_group.file_name, "r") as hdf_file:
        if dset_name not in hdf_file[hdf_level_group.level_group_path]:
            make_dataset(hdf_level_group, dset_name)

    hdf_level_group._append_dataset(dset_name, values)

    with h5py.File(hdf_level_group.file_name, "r") as hdf_file:
        saved_values = hdf_file[hdf_level_group.level_group_path]['test'][()]
        assert all(orig_value == saved_value for orig_value, saved_value in zip(values, saved_values))


def scheduled(hdf_level_group):
    """
    Test append and read scheduled dataset
    :param hdf_level_group: mlmc.tool.hdf.LevelGroup instance
    :return: None
    """
    hdf_level_group.append_scheduled(SCHEDULED_SAMPLES)
    with h5py.File(hdf_level_group.file_name, "r") as hdf_file:
        assert len(SCHEDULED_SAMPLES) == len(hdf_file[hdf_level_group.level_group_path]['scheduled'][()])

    saved_scheduled = [sample[0].decode() for sample in hdf_level_group.scheduled()]
    assert all(orig_scheduled_id == saved_schedule_id for orig_scheduled_id, saved_schedule_id in zip(SCHEDULED_SAMPLES, saved_scheduled))


def collected(hdf_level_group):
    """
    Test append and read collected dataset
    :param hdf_level_group: mlmc.tool.hdf.LevelGroup instance
    :return: None
    """
    hdf_level_group.append_successful(COLLECTED_SAMPLES)

    results = hdf_level_group.collected(slice(None, None, None)) # all samples
    for col, res in zip(COLLECTED_SAMPLES, results):
        assert (res == np.array(col[1])).all()

    with h5py.File(hdf_level_group.file_name, "r") as hdf_file:
        for _, dset_params in mlmc.tool.hdf5.LevelGroup.COLLECTED_ATTRS.items():
            assert len(COLLECTED_SAMPLES) == len(hdf_file[hdf_level_group.level_group_path][dset_params['name']][()])


if __name__ == '__main__':
    test_hdf5()
    test_level_group()
