import statprof
from contextlib import contextmanager




@contextmanager
def stat_profiler():
    statprof.start()
    yield  statprof
    statprof.stop()
    statprof.display()

