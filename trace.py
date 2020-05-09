class Trace():
    """
    Class that stores the logs of running an optimization method and plots
    the trajectory
    .
    """
    def __init__(self, trace_length=500):
        self.xs = []
        self.ts = []
        self.its = []