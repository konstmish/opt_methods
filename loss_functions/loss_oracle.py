class Oracle():
    """
    Base class for all objectives. Can provide objective values,
    gradients and its Hessians as functions that take parameters as input.
    Takes as input the values of l1 and l2 regularization.
    """
    def __init__(self, l1=0, l2=0):
        if l1 < 0.0:
            raise ValueError("Invalid value for l1 regularization: {}".format(l1))
        if l2 < 0.0:
            raise ValueError("Invalid value for l2 regularization: {}".format(l2))
        self.l1 = l1
        self.l2 = l2
    
    def value(self, x):
        pass
    
    def gradient(self, x):
        pass
    
    def hessian(self, x):
        pass
    
    def smoothness(self):
        pass
    
    def max_smoothness(self):
        pass
    
    def average_smoothness(self):
        pass
    
    def density(self, x):
        if hasattr(x, "toarray"):
            dty = float(x.nnz) / (x.shape[0]*x.shape[1])
        else:
            dty = 0 if x is None else float((x!=0).sum()) / x.size
        return dty