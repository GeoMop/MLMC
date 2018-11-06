import numpy as np


class Sample:
    def __init__(self, **kwargs):
        """
        Create Sample() instance
        :param kwargs: 
                sample_id: Sample unique identifier
                directory: Directory with sample simulation data
                job_id: Id of pbs job with this sample
                prepare_time: Time needed for creating sample
                queued_time: Time when job was queued
                result: sample simulation result
                time: overall time
        """
        self.sample_id = kwargs.get('sample_id')
        self.directory = kwargs.get('directory', '')
        self.job_id = kwargs.get('job_id', '')
        self.prepare_time = kwargs.get('prepare_time', 0)
        self.queued_time = kwargs.get('queued_time', 0)
        self._result = kwargs.get('result')
        # @TODO is this attr used?
        self.running_time = kwargs.get('running_time')
        self._time = kwargs.get('time', None)


    # def set_values(self, attributes):
    #     """
    #     Set sample attributes
    #     :param attributes:
    #     :return:
    #     """
    #     for name, value in attributes.items():
    #         setattr(self, name, value)

    @property
    def time(self):
        if self._time is None:
            self.time = self.prepare_time + self.running_time
        return self._time

    @time.setter
    def time(self, time):
        self._time = time

    def scheduled_data(self):
        """
        Sample data that are used for scheduled simulations in HDF5
        :return: tuple
        """
        return self.directory, self.job_id, self.prepare_time, self.queued_time

    @property
    def result(self):
        return np.squeeze(self._result)

    @result.setter
    def result(self, res):
        self._result = res

    def collected_data_array(self, attributes):
        """
        Get sample attribute values
        :param attributes: list of required sample attributes
        :return: list of collected values
        """
        coll_attributes = []
        for name in attributes:
            coll_attributes.append(getattr(self, name))

        return coll_attributes
