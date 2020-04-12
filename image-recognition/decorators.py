import time


def time_it(func):
    """
    Decorator to calculate execution time of a give function.
    :param func: function to be measured
    :return: original function wrapped
    """
    def wrapper(*args, **kw):
        start = time.time()
        print("Computing {}...".format(func.__name__))
        func(*args, **kw)
        print("Done in {0:.3f}s.".format(time.time() - start))
    return wrapper
