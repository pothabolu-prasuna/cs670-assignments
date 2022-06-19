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

![image](https://user-images.githubusercontent.com/106718885/174467073-3df17a51-c5c0-4251-accf-822bede0e671.png)


**Part 2 (30 points)
Assume that you are given the customer data generated in Part 1, implement a Gradient Descent algorithm from scratch that will estimate the Exponential distribution according to the Maximum Likelihood criterion.**

Answer the following:

**Plot the negative log likelihood of the exponential distribution. (10 points)**


Plot the negative log likelihood of the exponential distribution:


log likelihood of the exponential distribution: nlog(λ)- λ sum(x) 

Negative log likelihood of the exponential distribution=-[ nlog(λ)- λ sum(x)

Steps:

	1.Take set of values for λ ex: we took here from 0.01 to 1 with step size 0.01
 
	2.Take x data set here we took randomly generated 200 samples
 
	3.Calculate negative log likelihood value using above formula using data set(step2) , each lambda
 
	4. use lambdaset and calculated negative likelihood values to plot graph
	
	
Code:

from scipy.stats import expon
#pdf = expon.pdf(x,loc=0,scale=4)

def plot_EXpll(x): 

 plt.figure(figsize=(8,8)) 
 
 plt.title("Neg Log Likelihood of Exponential Distribution") 
 
 plt.xlabel("lambda Estimate") 
 
 plt.ylabel("Negative log Likelyhood") 

 lambda_set = np.arange(0.01,1, 0.01)
 
 #print(lambda_set)

 exll_array = [] 
 
 n=x.size
 
 #print(n)
 
 for lambdaval in lambda_set: 
 
    loglikelyval= (n*np.log(lambdaval))-(lambdaval*np.sum(x)) # The LL function
    
    #print(loglikelyval)
    
    exll_array.append(-loglikelyval) # negative LL 
    
 print(exll_array)
 
 #Plot the results
 
 plt.plot(lambda_set, exll_array , label='expon pdf' , color = 'b')

plot_EXpll(xdata);


**Output:**

![image](https://user-images.githubusercontent.com/106718885/174467560-7f7dd1e5-8fb7-4f55-bbaa-2ba8ba2fba98.png)

**Part 2.2 What is the lambda MLE of the generated data? (10 points)**

log likelihood of the exponential distribution: ll(λ:x1,x2..xn)=nlog(λ)- λ*sum(x)

derivative LL with respect to λ=n/λ - sum(x)

**by using this formula MLE of parameter λ=n/sum(x)**

code: find MLE of data generated in previous problem

import numpy as np 

def exp_lamda_MLE(x):

    n = len(x)
    
    sumval = np.sum(x)
    
    return n/sumval

print("lambda estimate:", exp_lamda_MLE(xdata))



