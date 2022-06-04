

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

**Output:**

![image](https://user-images.githubusercontent.com/106718885/172023125-fdc52773-dd05-4716-b7ac-74ba05843ea0.png)

**Problem 1b:** (10 Points) What is the probability that a female is between the heights of 65 inches and 70 inches? What is the probability that a female is between the heights of 65 inches and 70 inches given that they are at least 60 inches tall ( You can use 100 as the upper limit of the distribution )?

(Hint: This is a probability assignment, not a calculus assignment. You do not need to calculate any integrals, just show the steps that you would take and the final result)

**Answer 1b:**

Given Mean: 64.5 inches and a standard deviation : 2.5 inches.

probability that a female is between the heights of 65 inches and 70 inches:

zscore(65)=(x-mean)/stardarddeviation=(65-64.5)/2.5=0.5/2.5=0.2

zscore(70)=(70-64.5)/2.5=2.2

using ztable =zscore(0.2)=0.5793

zscore(2.2) =0.9861

Then we will subtract the smaller value from the larger value:0.9861-0.5793=0.4068

probability that a female is between the heights of 65 inches and 70 inches:40.68%

the probability that a female is between the heights of 65 inches and 70 inches given that they are at least 60 inches tall:

p(60) using z score=(60-64.5)/2.5=-1.8 using ztable value is=0.0359

P(65<t<70|t>=60)=p(65∩70)/1-P(60)=40.68/(1-0.0359)=42.19


**Problem 1c:** (20 Points) Lets say you conduct an experiment with a 100 trials where you measure a random man’s height. Lets say the measurement that you take is always rounded down to an integer

( ie. both a person with a height of 75.2 inches and a person of height 75.8 inches would be recorded as 75 inches thus making the distribution a discrete distribution instead of continuous).

What do you expect the count of men with a height of 70 inches to be? What type of distribution do you expect it to be?

( You do not need to answer these questions, it is simply something to think about to aid you with the next part )

Calculate the probability distribution function of the “counts” of people out of 100 with a height of 70 inches.

( Hint: You will have to find the categorical probability that a man is of height 70 )

Simulate the experiment 1000 times to show the relationship on a plot. What is the relationship between number of times the experiment is run and how close it is to the true distribution

( Hint: numpy has many functions that can allow you to simulate distribution functions)

**Problem 1c ANSWER:**
since we are not considering continous values, and consideing discrete values , we can use binomial distributon

from scipy.stats import binom

import matplotlib.pyplot as plt



n = 100

p = 0.01 #if consider 1 to 100 inches for height



prob=binom.pmf(70,100,0.01)

print("categorical probability that a man is of height 70 inches:",prob)

o/p is:categorical probability that a man is of height 70 inches: 2.172673073333347e-115


**Plot**

import numpy 
import matplotlib.pyplot as plt 

prob=[] # pobability list to store outcome 2 probility for range n=10 to 1000 

ntrails_no=[] # List to store n trails values 

ntrails=1000 

count70=0 

--# it is to get 

for n in range(1,ntrails+1): 

 x = numpy.random.multinomial(100, [1/100.]*100, size=1) 
 
 count70=x[0][69] 
 
 probability=count70/(100*n) 
 
 count70=0 
 
 prob.append(probability) 
 
 ntrails_no.append(n) 


plt.plot(ntrails_no,prob,color='green') 

plt.xlabel("Trials") 

plt.ylabel("Probability") 

plt.title("Probability of 70inches height using multinomial distribution") 

plt.show()

![image](https://user-images.githubusercontent.com/106718885/172027661-84bcf1eb-93b3-4e37-83c8-d8e439635e29.png)
