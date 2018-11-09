import numpy as np
from mlmc.sample import Sample


class LevelGroup:
    # One sample data type for dataset scheduled
    SAMPLE_DTYPE = {'names': ('dir', 'job_id', 'prepare_time', 'queued_time'),
                    'formats': ('S100', 'S5', 'f8', 'f8')}
    # Row format for dataset scheduled
    SCHEDULED_DTYPE = {'names': ('fine_sample', 'coarse_sample'),
                       'formats': (SAMPLE_DTYPE, SAMPLE_DTYPE)}

    # Data that is collected, only this data can by saved to datasets
    # @TODO change docstring
    # {attribute name in class Sample : {name: dataset name, maxshape: dataset maxshape, dtype: dataset dtype}}
    COLLECTED_ATTRS = {"sample_id": {'name': 'collected_ids', 'maxshape': (None, 1), 'dtype': np.int16},
                       "result": {'name': 'collected_values', 'maxshape': (None, 2, None), 'dtype': np.float32},
                       "time": {'name': 'collected_times', 'maxshape': (None, 2, None), 'dtype': np.float32}}

    DATASET_NAMES = {"scheduled": "scheduled",
                     "collected_ids": "collected_ids"}

    def __init__(self, hdf_group, level_id, job_dir):
        # HDF5 level group
        self.level_id = level_id
        # HDF Group object
        self._level_group = hdf_group
        self._n_ops_estimate = None
        # Absolute path to job directory
        self.job_dir = job_dir

        self._level_group.attrs['level_id'] = self.level_id

    def append_scheduled(self, scheduled_samples):
        """
        Save scheduled sample to dataset
        :param scheduled_samples: list of sample objects [(fine_sample, coarse_sample)]
        :return: None
        """
        # Creates a scheduled samples format acceptable for hdf
        sim_hdf_form = []
        # Jobs {job_id: [sample_id, ...]}
        jobs = {}

        # Go through scheduled samples and format them
        for sample_id, (fine_sample, coarse_sample) in scheduled_samples.items():
            sim_hdf_form.append((fine_sample.scheduled_data(), coarse_sample.scheduled_data()))
            # Append sample id to job id
            jobs.setdefault(fine_sample.job_id, set()).add(fine_sample.sample_id)
            if int(self.level_id) != 0:
                jobs.setdefault(coarse_sample.job_id, set()).add(coarse_sample.sample_id)

        # Create dataset scheduled
        if LevelGroup.DATASET_NAMES['scheduled'] not in self._level_group:
            self._create_scheduled_dataset(sim_hdf_form)
        # Dataset scheduled already exists
        else:
            self._append_dataset(LevelGroup.DATASET_NAMES['scheduled'], sim_hdf_form)

        # Save to jobs to datasets
        self._append_jobs(jobs)

    def _create_scheduled_dataset(self, scheduled_samples):
        """
        Create dataset for scheduled samples
        :param scheduled_samples: list in form [((fine sample data, coarse sample data), status), ...]
        :return: None
        """
        # Create dataset with SCHEDULED_DTYPE and with unlimited max shape (required for resize)
        scheduled_dset = self._level_group.create_dataset(
            LevelGroup.DATASET_NAMES['scheduled'],
            shape=(len(scheduled_samples),),
            maxshape=(None,),
            dtype=LevelGroup.SCHEDULED_DTYPE,
            chunks=True)

        # Set values to dataset
        scheduled_dset[()] = scheduled_samples

    def _append_jobs(self, jobs):
        """
        Save jobs to datasets
        :param jobs: dict, {job_id: [sample_id,...], ...}
        :return: None
        """
        if 'Jobs' not in self._level_group:
            self._level_group.create_group('Jobs')

        for job_name, job_samples in jobs.items():
            job_dataset_path = '/'.join(['Jobs', job_name])

            if job_dataset_path not in self._level_group:
                self._level_group.create_dataset(job_dataset_path, data=list(job_samples), maxshape=(None,),
                                                     chunks=True)
            else:
                self._append_dataset('/'.join([self._level_group.name, job_dataset_path]), list(job_samples))

    def append_collected(self, collected_samples):
        """
        Save level collected data to datasets corresponding to COLLECTED_ATTRS
        :param collected_samples: Level sample [(fine sample, coarse sample)], both are Sample() object instances
        :return: None
        """
        collected_data = {}
        # Get sample attributes pairs as zip (fine attribute values, coarse attribute values)
        samples_attr_pairs = self._sample_attributes_pairs(collected_samples)
        # Attribute values with attribute names
        for key, values in zip(LevelGroup.COLLECTED_ATTRS.keys(), samples_attr_pairs):
            collected_data[key] = values

        # Create own dataset for each collected attribute
        for attr_name, data in collected_data.items():
            data = np.array(data)
            # Sample id is same for fine and coarse sample
            if attr_name == 'sample_id':
                data = data[:, 0]
            # Data shape
            shape = data.shape
            # Maxshape as a constant
            maxshape = LevelGroup.COLLECTED_ATTRS[attr_name]['maxshape']

            # Data is squeezed, so expand last dimension to 'maxshape' shape
            if len(shape) == len(maxshape) - 1:
                data = np.expand_dims(data, axis=len(maxshape) - 1)
                shape = data.shape

            # Dataset already exists, append new values
            if LevelGroup.COLLECTED_ATTRS[attr_name]['name'] in self._level_group:
                self._append_dataset(LevelGroup.COLLECTED_ATTRS[attr_name]['name'], data)
                continue

            # Create resizable dataset
            dataset = self._level_group.create_dataset(
                LevelGroup.COLLECTED_ATTRS[attr_name]['name'],
                shape=shape,
                maxshape=maxshape,
                dtype=LevelGroup.COLLECTED_ATTRS[attr_name]['dtype'],
                chunks=True)
            # Set data
            dataset[()] = data

    def _sample_attributes_pairs(self, fine_coarse_samples):
        """
        Merge fine sample and coarse sample collected values to one dictionary
        :param fine_coarse_samples: list of tuples; [(Sample(), Sample()), ...]
        :return: zip
        """
        fine_coarse_data = np.array([(f_sample.collected_data_array(LevelGroup.COLLECTED_ATTRS),
                                      coarse_sample.collected_data_array(LevelGroup.COLLECTED_ATTRS))
                                     for f_sample, coarse_sample in fine_coarse_samples])

        samples_attr_values = (zip(*[zip(f, c) for f, c in zip(fine_coarse_data[:, 0], fine_coarse_data[:, 1])]))
        return samples_attr_values

    def save_failed(self, failed_samples):
        """
        Save level failed sample ids (not append samples)
        :param failed_samples: set; Level sample ids
        :return: None
        """
        # Dataset already exists, append new values
        if 'failed_ids' not in self._level_group:
            # Create resizable dataset
            self._level_group.create_dataset(
                'failed_ids',
                data=list(failed_samples),
                maxshape=(None,),
                chunks=True)
            # Set data
        else:
            dataset = self._level_group['failed_ids']
            dataset.resize((len(failed_samples),))
            dataset[()] = list(failed_samples)

    def _append_dataset(self, dataset_name, values):
        """
        Append values to existing dataset
        :param dataset_name: Path to dataset
        :param values: list of values (tuple or single value)
        :return: None
        """
        if dataset_name not in self._level_group:
            raise Exception("Dataset doesn't exist")
        # Resize dataset
        self._level_group[dataset_name].resize(self._level_group[dataset_name].shape[0] +
                                               len(values), axis=0)
        # Append new values to the end of dataset
        self._level_group[dataset_name][-len(values):] = values

    def scheduled(self):
        """
        Read level dataset with scheduled samples
        :return: generator, each item is in form (Sample(), Sample())
        """
        # Generator of scheduled dataset values
        scheduled_dset = self._level_group.get(LevelGroup.DATASET_NAMES['scheduled'], None)

        if scheduled_dset is None:
            return

        # Create fine and coarse samples
        for sample_id, (fine, coarse) in enumerate(scheduled_dset):
            yield (Sample(sample_id=sample_id,
                          directory=fine[0].decode('UTF-8'),
                          job_id=fine[1].decode('UTF-8'),
                          prepare_time=fine[2], queued_time=fine[3]),
                   Sample(sample_id=sample_id,
                          directory=coarse[0].decode('UTF-8'),
                          job_id=coarse[1].decode('UTF-8'),
                          prepare_time=coarse[2], queued_time=coarse[3]))

    def collected(self):
        """
        Read all level datasets with collected data (reading dataset values via self._dataset_values() method),
        create fine and coarse samples as Sample() instances
        :return: generator; one item is tuple (Sample(), Sample())
        """
        # Collected ids dataset exists in Level group
        if LevelGroup.DATASET_NAMES['collected_ids'] not in self._level_group:
            return

        # Number of collected samples
        num_collected_samples = self._level_group[LevelGroup.DATASET_NAMES['collected_ids']].len()

        # Collected data as dictionary according to lookup table COLLECTED_ATTRS
        # dictionary format -> {COLLECTED_ATTRS key: corresponding dataset values generator, ...}
        collected_data = {sample_attr_name: self._dataset_values('/'.join([self._level_group.name, dset_params['name']]))
                          for sample_attr_name, dset_params in LevelGroup.COLLECTED_ATTRS.items()}

        # For each item create two Sample() instances - fine sample and coarse sample
        for _ in range(num_collected_samples):
            fine_values = {}
            coarse_values = {}
            # Loop through all datasets (in form of generator) with collected data
            for sample_attr_name, values_generator in collected_data.items():
                value = next(values_generator)
                # Sample id is store like single value for both samples
                if sample_attr_name == 'sample_id':
                    fine_values[sample_attr_name] = value[0]
                    coarse_values[sample_attr_name] = value[0]
                else:
                    fine_values[sample_attr_name] = value[0]
                    coarse_values[sample_attr_name] = value[1]
            # Create tuple of Sample() instances
            yield (Sample(**fine_values), Sample(**coarse_values))

    def _dataset_values(self, dataset_name):
        """
        Read dataset values
        :param dataset_name: dataset name in Level group
        :return: generator
        """
        # Dataset doesn't exist
        if dataset_name not in self._level_group:
            return
        else:
            # Scheduled dataset
            dataset = self._level_group[dataset_name]

            # Return dataset items
            for dset_value in dataset:
                yield dset_value

    def level_jobs(self):
        """
        Get level job ids
        :param level_id: Level id
        :return: list of job ids
        """
        if 'Jobs' not in self._level_group:
            return []
        return list(self._level_group['Jobs'].keys())

    def failed_samples(self):
        if 'failed_ids' not in self._level_group:
            return set()
        return set(self._level_group['failed_ids'])

    def job_samples(self, job_dataset_names):
        """
        Get sample ids from job datasets
        :param job_dataset_names: Job dataset names
        :return: set; sample ids
        """
        # HDF path to level Jobs group
        if 'Jobs' not in self._level_group:
            self.level_group.create_group('Jobs')

        # Path 'Jobs' group
        jobs_group_hdf_path = "/".join([self._level_group.name, 'Jobs'])
        sample_ids = set()
        # Get job samples
        for dset_name in job_dataset_names:
            dataset = self._level_group["/".join([jobs_group_hdf_path, dset_name])]
            sample_ids.update(dataset.value)

        return sample_ids

    @property
    def n_ops_estimate(self):
        return self._n_ops_estimate

    @n_ops_estimate.setter
    def n_ops_estimate(self, n_ops_estimate):
        self._n_ops_estimate = self._level_group.attrs['n_ops_estimate'] = float(n_ops_estimate)

    # def change_scheduled_sample_status(self, sample_id, status):
    #     """
    #     Change scheduled sample status, it is common for fine sample and coarse sample too
    #     :param level_id: Level id
    #     :param sample_id: Sample id
    #     :param status: status, e.g. "RUN"
    #     :return: None
    #     """
    #     if LevelGroup.DATASET_NAMES['scheduled'] not in self._level_group:
    #         raise Exception("Dataset is not in HDF5 file")
    #     else:
    #         sample_row = self._level_group[LevelGroup.DATASET_NAMES['scheduled']][sample_id]
    #         sample_row[-1] = status
    #         self._level_group[LevelGroup.DATASET_NAMES['scheduled']][sample_id] = sample_row
