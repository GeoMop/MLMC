class Sample:
    # @TODO: some attributes not yet used
    def __init__(self, directory='', job_id='id', prepare_time=0, queued_time=0, sample_id=None, sample_tag=None):
        self.sample_id = sample_id
        self.directory = directory
        self.job_id = job_id
        self.prepare_time = prepare_time
        self.queued_time = queued_time
        self._result = None
        self.running_time = 0
        # @TODO: Not yet used
        self.sample_tag = sample_tag

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
        return self.prepare_time + self.running_time

    def scheduled_data(self):
        """
        Sample data that are used for scheduled simulations in HDF5
        :return: tuple
        """
        return self.directory, self.job_id, self.prepare_time, self.queued_time

    @property
    def result(self):
        return self._result

    @result.setter
    def result(self, res):
        self._result = res

    def collected_data(self, attributes):
        """
        Get sample selected attributes
        :param attributes: list of required sample attributes
        :return: dict {attribute name: attribute value}
        """
        coll_attributes = {}
        for name in attributes:
            coll_attributes[name] = getattr(self, name)

        return coll_attributes
