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

**output is: lambda estimate: 0.24911656443887148**

**part 2.3Plot the estimated lambda vs iterations to showcase convergence towards the true lambda (10 points)
**

radient descent algorithm to find parameter set: 

Step 1: Take derivative (Gradient) of loss function for each parameter in it 

Step 2: pick random values for the parameters 

Step 3: Plug the parameter values into derivatives 

Step 4: calculate the step sizes : step size=gradient of loss function*learning rate 

Step 5: calculate the new parameters = old parameter- step size Repeat this 
steps 3 to step 5 till we step sizes are very small or we reach maximum number of steps 

We apply above steps to estimate optimal values for lambda. 


Step1: consider random values lambda=100, iterations = 1000 and learning rate =0.01 

Step2: take derivatives of MLE of lambda (λ), using above formulas. der_lambda=(n/lamval)-np.sum(data)

We denote these value as d_lambda.

Step 3: find new lambda using step size est_lambda = est_lambda - (-learning_rate*d_lambda)

step 4: repeat above steps for given number of iterations 

step 5: find optimal values for lambda
steep6: calculate exponential likelihood using new lambda 

formula is (n*np.log(lambdaval))-(lambdaval*np.sum(data))

step 7: we stop epochs when likelihood of exponential distribution value has no much difference with previous epoch values. We used threshold 0.001 for difference


code:

import numpy as np 

def partial_deriv_lambda(data,lamval): 

 #derivate of log of likelhood with respect to lambda is ∂LL/∂lambda =(n/lambda)-(x1+x2+...xn) for n=1 to N 
 
 n=data.size
 
 der_lambda=(n/lamval)-np.sum(data)
 
 return der_lambda 
  
def log_likelihood(data,lambdaval): 

 n=data.size
 
 loglikeval= (n*np.log(lambdaval))-(lambdaval*np.sum(data)) # The LL function
 
 #print("ll",loglikeval)
 
 return loglikeval 

def gradient_descent_exp(data, est_lambda,learning_rate, epochs): 

 min_step_lambda=99999
 
 threshold = 1e-3
 
 prev_ll=0
 
 est_ll = [] 
 
 lam_set=[]
 
 #print("lam",est_lambda)
 
 for k in range(epochs): 
 
  d_lambda = partial_deriv_lambda(data,est_lambda ) 
  
  #print("lambda=",est_lambda," dlambda=",d_lambda)
  #print(d_lambda)
  
  estll_val=log_likelihood(data,est_lambda) 
  
  est_ll.append(-estll_val) 
  
  if min_step_lambda>(learning_rate*d_lambda): 
  
     opt_lambda=est_lambda - (learning_rate*d_lambda) 
  
 
  est_lambda = est_lambda - (-learning_rate*d_lambda) 
  
  lam_set.append(est_lambda)
  
  if(abs(prev_ll-estll_val) <threshold):
  
    print("entered optimal value")
    
    break
    
  prev_ll=estll_val
 
 print("Final parameters with given iterations:") 
 
 print(f"lambda value {est_lambda} Gradient of lambda is",partial_deriv_lambda(data,est_lambda) ) 
 
 print(f"\n optimal lambda= {opt_lambda}" ) 
 
 plt.figure(figsize=(8,8)) 
 
 plt.title("Gradient descent of Lambda Exponential Distribution") 
 
 plt.xlabel("lambda Estimate")
 
 plt.ylabel("Negative log Likelyhood") 
 
 plt.plot(lam_set, est_ll , label='expon pdf' , color = 'b')
 
gradient_descent_exp(xdata,100,0.1,100) 


output:


