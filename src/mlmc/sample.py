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
        self.job_id = kwargs.get('job_id', 'jobId')
        self.prepare_time = kwargs.get('prepare_time', 0.0)
        self.queued_time = kwargs.get('queued_time', 0)
        self._result = kwargs.get('result', None)
        self.running_time = kwargs.get('running_time', 0.0)
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
        """
        Total time for sample
        :return: float
        """
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
        """
        Sample result
        :return: numpy array or np.Inf
        """
        if self._result != np.Inf and self._result is not None:
            return np.squeeze(self._result)
        return self._result

    @result.setter
    def result(self, res):
        self._result = res

    def collected_data_array(self, attributes):
        """
        Get sample attribute values
        :param attributes: list of required sample attributes
        :return: list of collected values of attributes
        """
        coll_attributes = []
        for name in attributes:
            coll_attributes.append(getattr(self, name))

        return coll_attributes

    def add_scheduled_attrs(self, scheduled_sample):
        """
        Add sample attributes - which were previously saved as as scheduled sampled - to 'collected' sample
        :param scheduled_sample: Sample()
        :return: None
        """
        # Sample attributes which are in scheduled dataset
        scheduled_attr_saved = ['directory', 'job_id', 'prepare_time', 'queued_time']
        for attr_name in scheduled_attr_saved:
            setattr(self, attr_name, getattr(scheduled_sample, attr_name))

    def __eq__(self, other):
        return self.sample_id == other.sample_id and \
               self.prepare_time == other.prepare_time and\
               self.queued_time == other.queued_time and \
               self.result == other.result

    def __str__(self):
        return "sample id: {}, result: {}, prepare time: {}, queued time: {} ".format(self.sample_id, self.result,
                                                                                      self.prepare_time,
                                                                                      self.queued_time)
