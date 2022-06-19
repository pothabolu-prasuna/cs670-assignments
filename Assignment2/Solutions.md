**Part 1 (30 points):**

A) Let X = amount of time (in minutes) a ice cream man gets a new customer at his cart. The time is known to have an exponential distribution with the average amount of time between a new customer being four minutes.

Plot the probablity density function of the exponential distribution of this ice cream man getting a customer every 4 minutes. (10 points)

**Answer:**

X = amount of time (in minutes) a ice cream man gets a new customer at his cart

X is a continuous random variable since time is measured. It is given that(mean) μ=4 minutes

Average of time between a new customer for four minutes= lambda=1/μ=1/4=0.25

rate=1/4=0.25

A common parameterization for expon is in terms of the rate parameter lambda, such that  **pdf = lambda * exp(-lambda * x)**.

This parameterization corresponds to using **scale = 1 / lambda**.


Plot 

#Importing required modules

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import expon
 
#Applying the expon class methods
x = np.linspace(1,20, 10)

#Using probability density function
pdf = expon.pdf(x,loc=0,scale=4)
 
#Visualizing the results

print(pdf)

plt.plot(x, pdf , label='expon pdf' , color = 'b')

plt.xlabel('intervals')

plt.ylabel('Probability Density')

plt.show()

Output:

