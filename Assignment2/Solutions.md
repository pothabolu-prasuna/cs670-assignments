**Part 1 (30 points):**

**A) Let X = amount of time (in minutes) a ice cream man gets a new customer at his cart. The time is known to have an exponential distribution with the average amount of time between a new customer being four minutes.**

Plot the probablity density function of the exponential distribution of this ice cream man getting a customer every 4 minutes. (10 points)

**Answer:**

X = amount of time (in minutes) a ice cream man gets a new customer at his cart

X is a continuous random variable since time is measured. It is given that(mean) μ=4 minutes

Average of time between a new customer for four minutes= lambda=1/μ=1/4=0.25

rate=1/4=0.25

A common parameterization for expon is in terms of the rate parameter lambda, such that  **pdf = lambda * exp(-lambda * x)**.

This parameterization corresponds to using **scale = 1 / lambda**.


**Plot data set with lambda 0.25 so scale=4
**
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

**Output:**

![image](https://user-images.githubusercontent.com/106718885/174466785-f3701891-15d1-47cc-a1d3-b55612ac658d.png)


**B) Now assume on a very hot day the ice cream man gets X customers and each new customer comes every 4 minutes. Generate X samples from the exponential distribution where X = 200 and the rate = 4. Plot the samples on a graph to show how they look graphically. Does it look similar to the graph above? (20 points)**

**code:**

we use random.exponential function with values scale=4 and size=200 to generate 200 samples

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns  

#Using exponential() method

exponval = np.random.exponential(4, 200)

#print(exponval )

plt.title("Exponential distribution of 200 data samples")

plt.xlabel('intervals')

plt.ylabel('Probability Density')

plt.hist(exponval, 10, density = True)

plt.show()



**output:**
