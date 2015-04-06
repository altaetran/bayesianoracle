
""" Timing functions """

def tic():
    """ Timing function, counter initializer """
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    """ Timing function, counter setter """
    import time
    if 'startTime_for_tictoc' in globals():
        return time.time() - startTime_for_tictoc
    else:
        return -1
