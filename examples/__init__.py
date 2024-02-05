class OperatorWrapper:
    def __init__(self, operator, scale=1., shift=0.):
        self.operator = operator
        self.scale = scale
        self.shift = shift

    def __call__(self, model, x, importance=None):
        Tf, f = self.operator(model, x, importance)
        return self.scale * Tf + self.shift * f, f
