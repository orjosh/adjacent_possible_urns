from powerlaw import Distribution, trim_to_range

class Weibull(Distribution):
    def parameters(self, params):
        self.Shape = params[0]
        self.parameter1 = self.Shape
        self.parameter1_name = "Shape"
        self.Scale = params[1]
        self.parameter2 = self.Scale
        self.parameter2_name = "Scale"
    
    @property
    def name(self):
        return "weibull"

    def _initial_parameters(self, data):
        from numpy import mean
        return (1/mean(data), 1)
    
    def _in_standard_parameter_range(self):
        return self.Shape > 0 and self.Scale > 0
    
    def _pdf_base_function(self, x):
        from numpy import exp
        return (self.Shape / self.Scale) * pow(x/self.Scale, (self.Shape-1)) \
            * exp(-1*pow(x/self.Scale, self.Shape))

    def _cdf_base_function(self, x):
        from numpy import exp
        return 1 - exp(-1*pow(x/self.Scale, self.Shape))


import powerlaw
import numpy as np
from matplotlib import pyplot as plt

shape = 5

data = np.random.weibull(a=shape, size=10000)

model = Weibull(data=data)
model.fit(data)

fig, ax = plt.subplots()

model.plot_pdf(data=data, ax=ax)
ax.set_xscale("linear")
ax.set_yscale("linear")
fig.savefig("weibull_test.png")