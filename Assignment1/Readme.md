

**Problem 1a:** (10 Points) The heights of adult men in the United States are approximately normally distributed with a mean of 70 inches and a standard deviation of 3 inches. Heights of adult women are approximately normally distributed with a mean of 64.5 inches and a standard deviation of 2.5 inches.

Graph the two distributions from 0 to 100 inches using the plotting framework of your choice ( Matplotlib, Seaborn etc.)

**Solution:**

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

#x-axis ranges from -5 and 5 with .001 steps
x = np.arange(0, 100, 0.5)

#define multiple normal distributions
plt.plot(x, norm.pdf(x, 70, 3), label='Male-> μ: 70, σ: 3')
plt.plot(x, norm.pdf(x, 64.5, 2.5), label='Female--> μ:64.5, σ: 2.5')

#add legend to plot
plt.legend()