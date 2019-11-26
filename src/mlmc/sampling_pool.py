from abc import ABC, abstractmethod
from multiprocessing import Pool as ProcPool
from multiprocessing.pool import ThreadPool


class SamplingPool(ABC):

    @abstractmethod
    def plan_samples(self, config, calc_method, files=None):
        """
        Plan samples
        """

    @abstractmethod
    def have_permanent_sample(self, sample_id):
        """
        Is sample serialized?
        """

    @abstractmethod
    def get_finished(self):
        """
        Return finished samples
        """

    # @abstractmethod
    # def add_samples(self, fine_sample, coarse_sample):
    #     pass
    #
    # @abstractmethod
    # def read_data(self):
    #     pass


# class InMemory(SamplingPool):
#     def plan_samples(self, config, calc_method, files=None):
#         print("config ", config)
#         print("calc method ", calc_method)
#         print("files ", files)


class PBSPool(SamplingPool):
    def add_samples(self, fine_sample, coarse_sample):
        pass


class ProcessPool(SamplingPool):

    def __init__(self, n_processes):
        print("n processes ", n_processes)
        self._pool = ProcPool(n_processes)

    def plan_samples(self, config, calc_method, files=None):
        # print("self._pool ", self._pool)

        # print("type calc method ", calc_method)
        # print("type calc method ", type(calc_method))

        # fine_result, coarse_result = calc_method(config)
        #
        # print("fine result ", fine_result)
        # print("coarse result ", coarse_result)
        # jobs = []
        # for i in range(5):
        #     p = self._pool.apply_async(self.worker, args=(i,))
        #     jobs.append(p)

        result = self._pool.apply_async(calc_method, args=(config, ), callback=self.result_callback,
                                        error_callback=self.error_callback)
        return result

    def result_callback(self, res):
        print("RESULT ", res)

    def error_callback(self, error):
        print("Error ", error)

    def have_permanent_sample(self, sample_id):
        """
        List of permanent samples or find per call?
        """
        pass

    def get_finished(self):
        """

        """




"""
PBS - postupně přidávat metody compute() a data a pak spustit v jendom procesu, až se naplní počet simulací nebo step atd. prostě to co bylo v minulé verzi

ProcessPool - přidáváme metodu a dictionary s daty ze simulace (data ze simulace získá Sampler v podobně dictionary a pak je předá do SamplingPoolu spolu s metodou, která se má pustit )
              v momentě kdy je naplněný stanovený počet simulací, tak se vytvoří process (případně thread) a spustí se v něm jeden pythonní soubor s těmi metodami, které se mají pustit a jejich parametry (dictionry ze simulace)
              
Není to tak, že co volání metody compute() to nový pythonní script.

Jak si poradit s obnovením, když to selže, tak ID simulací bude znát jenom Sampler, jak zjistit jestli ty samply doběhly, umí ProcessPool vrátit spadnuté simulace resp. procesy??

Jak vlastně probíhá Picklování a samotné volání metody compute a dictionary a vytvoří simulaci a spustí ji

Zkusit otestovat na jednoduché simulaci
"""


class ThreadPool(SamplingPool):

    def __init__(self):
        self._samples = []
        self._pool = ThreadPool()

    def add_samples(self, fine_sample, coarse_sample):
        self._samples.append(fine_sample)
        self._samples.append(coarse_sample)

    def run(self, sample):
        sample.run

    def run_samples(self):
        self._pool.map(self.run, self._samples)
