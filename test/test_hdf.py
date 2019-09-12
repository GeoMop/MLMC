import os
import sys
import shutil
import h5py
import numpy as np
import pytest

src_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, src_path + '/../src/')
import mlmc.tool.hdf
from mlmc.sample import Sample


"""
test src/mlmc/tool/hdf.py methods
"""


def test_hdf5():
    work_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_test_tmp')
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)
    hdf_obj = mlmc.tool.hdf.HDF5(work_dir, 'test')

    init_header(hdf_obj)

    levels = ['1', '2', '3', '7', '8', '9']
    add_level_group(hdf_obj, levels)

    load_from_file(hdf_obj)

    clear_groups(hdf_obj)


def init_header(hdf_obj):
    step_range = (0.99, 0.00001)
    n_levels = 20
    obligatory_attributes = ['version', 'step_range', 'n_levels']

    hdf_obj.init_header(step_range, n_levels)

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


def load_from_file(hdf_obj):
    """
    Test loading data from existing file
    :param hdf_obj:
    :return: None
    """
    keys = hdf_obj.__dict__.keys()

    del_keys = ['step_range', 'n_levels']
    # Remove class attributes, keep just file name
    for key in del_keys:
        del hdf_obj.__dict__[key]

    hdf_obj.load_from_file()
    assert all(key in hdf_obj.__dict__ for key in keys)


SCHEDULED_SAMPLES = {0:(Sample(sample_id=0, job_id='1', prepare_time=0.01),
                              Sample(sample_id=0, job_id='1', prepare_time=0.011)
                              ),
                     1: (Sample(sample_id=1, job_id='1', prepare_time=0.009),
                          Sample(sample_id=1, job_id='1', prepare_time=0.012)
                          ),
                     2:  (Sample(sample_id=2, job_id='5', prepare_time=0.008),
                          Sample(sample_id=2, job_id='5', prepare_time=0.013)
                          )
                     }

RESULT_DATA_DTYPE = [("value", np.float), ("time", np.float)]


COLLECTED_SAMPLES = [(Sample(sample_id=0, job_id='1', time=0.1,  result=np.array([10, 1.5])),
                      Sample(sample_id=0, job_id='1', time=0.11, result=np.array([11, 0.0012]))),
                     (Sample(sample_id=1, job_id='1', time=0.09, result=np.array([-10.2, 7.854])),
                      Sample(sample_id=1, job_id='1', time=0.12, result=np.array([1.879, 1.00546]))),
                     (Sample(sample_id=2, job_id='5', time=0.08, result=np.array([-7.6, 5.16])),
                      Sample(sample_id=2, job_id='5', time=0.13, result=np.array([15, 100.1])))]

RESULT_STRUCT_FORMAT = np.array(['test_param_1', 'test_value_1'], dtype=[('test_1', 'S20'), ('test_2', 'S20')])


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
    hdf_level_group = mlmc.tool.hdf.LevelGroup(file_name, level_group_path, level_id, job_dir)

    with h5py.File(file_name, "r") as hdf_file:
        assert hdf_file[level_group_path].attrs['level_id'] == level_id == hdf_level_group.level_id

    make_dataset(hdf_level_group)

    make_group_datasets(hdf_level_group)

    append_dataset(hdf_level_group)

    scheduled(hdf_level_group)

    collected(hdf_level_group)

    job_samples(hdf_level_group)


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
    datasets = [attr_prop['name'] for _, attr_prop in mlmc.tool.hdf.LevelGroup.COLLECTED_ATTRS.items()]
    datasets.extend(['scheduled', 'failed_ids'])

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
        assert '1' in hdf_file[hdf_level_group.level_group_path]['Jobs']
        assert '5' in hdf_file[hdf_level_group.level_group_path]['Jobs']
        assert len(SCHEDULED_SAMPLES) == len(hdf_file[hdf_level_group.level_group_path]['scheduled'][()])

    saved_scheduled = hdf_level_group.scheduled()

    for fine_sample, coarse_sample in saved_scheduled:
        assert fine_sample == SCHEDULED_SAMPLES[fine_sample.sample_id][0]
        assert coarse_sample == SCHEDULED_SAMPLES[coarse_sample.sample_id][1]


def collected(hdf_level_group):
    """
    Test append and read collected dataset
    :param hdf_level_group: mlmc.tool.hdf.LevelGroup instance
    :return: None
    """
    hdf_level_group.result_additional_data = RESULT_STRUCT_FORMAT
    hdf_level_group.append_collected(COLLECTED_SAMPLES)

    saved_collected = hdf_level_group.collected()
    for index, (fine_collected, coarse_collected) in enumerate(saved_collected):
        assert fine_collected == COLLECTED_SAMPLES[index][0]
        assert coarse_collected == COLLECTED_SAMPLES[index][1]

    with h5py.File(hdf_level_group.file_name, "r") as hdf_file:
        for _, dset_params in mlmc.tool.hdf.LevelGroup.COLLECTED_ATTRS.items():
            assert len(COLLECTED_SAMPLES) == len(hdf_file[hdf_level_group.level_group_path][dset_params['name']][()])


def job_samples(hdf_level_group):
    """
    Test saved job ids
    :param hdf_level_group: mlmc.tool.hdf.LevelGroup instance
    :return: None
    """
    sample_ids = hdf_level_group.job_samples(['1', '5'])
    assert all(s_id == c_id for s_id, c_id in zip(sample_ids, range(len(COLLECTED_SAMPLES))))


if __name__ == '__main__':
    #test_hdf5()
    test_level_group()
