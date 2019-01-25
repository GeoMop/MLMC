

def profile(fun, skip=False):
    """
    Statistical profiling of a given function.
    :param fun:
    :param skip:
    :return:
    """
    import statprof
    if skip:
        return fun()
    statprof.start()
    try:
        result = fun()
    finally:
        statprof.stop()
    statprof.display()
    return result