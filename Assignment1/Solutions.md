

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

**P(65<t<70|t>=60)=p(65∩70)/1-P(60)=40.68/(1-0.0359)=42.19**


**Problem 1c:** (20 Points) Lets say you conduct an experiment with a 100 trials where you measure a random man’s height. Lets say the measurement that you take is always rounded down to an integer

( ie. both a person with a height of 75.2 inches and a person of height 75.8 inches would be recorded as 75 inches thus making the distribution a discrete distribution instead of continuous).

What do you expect the count of men with a height of 70 inches to be? What type of distribution do you expect it to be?

( You do not need to answer these questions, it is simply something to think about to aid you with the next part )

Calculate the probability distribution function of the “counts” of people out of 100 with a height of 70 inches.

( Hint: You will have to find the categorical probability that a man is of height 70 )

Simulate the experiment 1000 times to show the relationship on a plot. What is the relationship between number of times the experiment is run and how close it is to the true distribution

( Hint: numpy has many functions that can allow you to simulate distribution functions)

**Problem 1c ANSWER:**
**since we are not considering continous values, and consideing discrete values , we can use binomial distributon or multinomial distribution. the probability is constants with number of experiments increases**

from scipy.stats import binom

import matplotlib.pyplot as plt



n = 100

p = 0.01 #if consider 1 to 100 inches for height



prob=binom.pmf(70,100,0.01)

print("categorical probability that a man is of height 70 inches:",prob)

o/p is:categorical probability that a man is of height 70 inches: 2.172673073333347e-115

x = numpy.random.binomial(100, 0.01,size=1000) 

plt.hist(x,density="true")
plt.show()

![image](https://user-images.githubusercontent.com/106718885/172071561-917cde9e-4df8-4253-a7fd-600b9ea494cb.png)

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

**
Using binomial distribution**


import numpy 

import matplotlib.pyplot as plt 

from scipy.stats import binom

prob=[] # pobability list to store outcome 2 probility for range n=10 to 1000 

ntrails_no=[] # List to store n trails values 

ntrails=1000 

count_of_70=0 


for n in range(0,ntrails+1): 

 probheight = binom.pmf(70,100,0.01)

 prob.append(probheight)
 
 ntrails_no.append(n) 
 
plt.plot(ntrails_no,prob,color='green') 

plt.xlabel("Trials") 

plt.ylabel("Probability") 

plt.title("Probability of height 70 inches in 1000 experiments using binomial distribution") 

plt.show()

![image](https://user-images.githubusercontent.com/106718885/172027719-9543ca3a-a8bf-41df-a4af-fd2f9bc9ec7d.png)

**Problem 2a: (15 Points)**
Given the circuit below, the probability that any switch  is closed (current passes through ) is ? What is the probability that there is a signal at the output? Give your answer in terms of p.

![image](https://user-images.githubusercontent.com/106718885/172027860-718b8846-9b71-40c5-bdc8-bf260b348ae0.png)

**Answer 2a:**

a) The probability of a signal at the output here is computed as:

probability of current passes through switch s closed is p. Probability that a parallel of two switches(S1 and S3) work is computed as =p(S1US2)=1-p(S1c∩S2c)=1-((1-p)*(1-p)) = 1 - (1 - p)2

probability of top layer using (S1 ,S2,S3) is computed as:

= p* [ 1 - (1 - p)2 ]

= p* [ 2p - p2]

= p2(2 - p)

probability of bottom layer using (S4, S5, S6) is computed as: P(S4∩S5US6): = p2(2 - p)

Therefore, the probability that the whole system works now is computed here as: =1-[1-p2 (2 - p)]*[1-p2 (2 - p)] = 1 - [ 1 - p2 (2 - p) ]2

= 1-[1+ p4(2 - p)2 -2p2 (2 - p)]

=1-[1+ p4(4 +p2- 4p) -2p2 (2 - p)]

=1-1- p4(4 +p2- 4p)+ 2p2 (2 - p)

=4p2 - 2p3- 4p4 -p6+ 4p5

that is 4*(p pow5)-2 (p pow 3)-4(p pow 4)-(p pow 6)+4(p Pow 5)


**Problem 2b: (15 Points)**
Given the same circuit above, if a signal is observed at the end, what is the probability that  is open ( no current going through ) . Give your answer in terms of p.



Given that the signal is observed at the end, the probability that S3 is open here is computed using Bayes theorem here as:

= Probability that S3 is open and the below series works / P( system works )

=(1 - p) p2 (2 - p) /(4p2 - 2p3- 4p4 -p6+ 4p5)

=(1 - p) p2 (2 - p)/[p2 (2 - p)*[2 - p2(2-p) ]]

= (1-p) / [ 2 - p2(2-p) ]


**Problem 3a (20 points)** It follows that those with a larger height will be generally heavier than those with a smaller height. This is just a broad generalization and does not always apply. Here is a link to a data set that contains anonymous entries on peoples’ gender, height and weight. Download this dataset ( It is pretty small don’t worry ). You might need to make a Kaggle account. Kaggle is an online community of data scientists and has a large collection of open source datasets for many different purposes.

Familiarize yourself with the package pandas, as you can use it to easily unpack the csv into manipulatable datatypes. If you are using colab, which you most likely are, ensure that you have logged in with the university Google account.

Using the data find two values for covariance between height and weight. There should be one value for male and female.

(Hint: Make sure to store all intermediate values like averages and counts as they might be useful for the extra credit)




**Answer 3a:**

import pandas as pd

import math

df = pd.read_csv('weight-height.csv')

# covariance=sum((x-xmean)*(y-ymean))/n

Maledatarows=len(df[df['Gender']=='Male'])

Fdatarows=len(df[df['Gender']=='Female'])

Mheightsmean=df[df['Gender']=='Male']["Height"].mean()

Fheightsmean=df[df['Gender']=='Female']["Height"].mean()

Mweightsmean=df[df['Gender']=='Male']["Weight"].mean()

Fweightsmean=df[df['Gender']=='Female']["Weight"].mean()


print("Male heightsmean=",Mheightsmean," Male weightsmean=",Mweightsmean)

print("Female heightsmean=",Fheightsmean," Female weightsmean=",Fweightsmean)


df["heightsmeanDiff"]= ""

df["WeightsmeanDiff"]=""

df["Mulheightweightdiff"]=""

df["heightMeandiffsquare"]=""

df["weightMeandiffsquare"]=""

df.loc[df['Gender']=='Male',"heightsmeanDiff"]=df["Height"]-Mheightsmean

df.loc[df['Gender']=='Female',"heightsmeanDiff"]=df["Height"]-Fheightsmean

df.loc[df['Gender']=='Male',"WeightsmeanDiff"]=df["Weight"]-Mweightsmean

df.loc[df['Gender']=='Female',"WeightsmeanDiff"]=df["Weight"]-Fweightsmean

df["Mulheightweightdiff"]=df["heightsmeanDiff"]*df["WeightsmeanDiff"]

df["heightMeandiffsquare"]=df['heightsmeanDiff']**2

df["weightMeandiffsquare"]=df['WeightsmeanDiff']**2

MdataCov=df[df['Gender']=='Male']["Mulheightweightdiff"].mean()

FdataCov=df[df['Gender']=='Female']["Mulheightweightdiff"].mean()

print ("Covarience of Male height and weight",MdataCov)

print ("Covarience of Female height and weight",FdataCov)

df.loc[df['Gender']=='Male'].cov()


**Output:**

![image](https://user-images.githubusercontent.com/106718885/172030243-d9b60f4b-3935-46ea-b8c6-9061ea8b9c12.png)

**Problem 3b:** (10 points) Find the correlation between height and weight for Males and Females

#correlation=cov(x,y)/(standarddeviation(x)* standarddeviation(y))

#ststandarddeviation(x)=sqrt(sqare(x-mean)/n)


stdMHeight=math.sqrt(df[df['Gender']=='Male']["heightMeandiffsquare"].sum()/Maledatarows)

stdMweight=math.sqrt(df[df['Gender']=='Male']["weightMeandiffsquare"].sum()/Maledatarows)

stdFHeight=math.sqrt(df[df['Gender']=='Female']["heightMeandiffsquare"].sum()/Fdatarows)

stdFweight=math.sqrt(df[df['Gender']=='Female']["weightMeandiffsquare"].sum()/Fdatarows)

Mcorr=MdataCov/(stdMHeight*stdMweight)

Fcorr=FdataCov/(stdFHeight*stdFweight)

print("Correlation of Male height and weight=",Mcorr)

print("Correlation of Female height and weight=",Fcorr)

**output:**

Correlation of Male height and weight= 0.8629788486163136

Correlation of Female height and weight= 0.8496085914186011

**Extra credit: (5 points)**
Using matplotlib’s or seaborn’s 3D graphing functionality, create a wireframe graph of the multivariate probability distribution of heights and weights for either men or women ( You don’t have to do both ). Use the data and calculated values from problem 3 to solve this problem.

( Hint: You can assume both distributions are normal. Use this link to help in understanding )

**Answer:**
import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt

mean = [Mheightsmean, Mweightsmean]

mcov = df.loc[df['Gender']=='Male'].cov()

print(mcov)
plt.style.use('seaborn-dark')

plt.rcParams['figure.figsize']=14,6

fig = plt.figure()

distr = multivariate_normal(cov = mcov, mean = mean)

X, Y = np.meshgrid(df[df['Gender']=='Male']["Height"],df[df['Gender']=='Male']["Weight"])

pdf = np.zeros(X.shape)

for i in range(X.shape[0]):

  for j in range(X.shape[1]):
  
    pdf[i,j] = distr.pdf([X[i,j], Y[i,j]])

fig = plt.figure(figsize=(13, 7))

ax = plt.axes(projection='3d')

#print(pdf[0:1000,:])

w = ax.plot_wireframe(X, Y, pdf)

ax.set_xlabel('Height')

ax.set_ylabel('Weight')

ax.set_zlabel('PDF')

ax.set_title('Wireframe plot of Gaussian');

**output:**

![image](https://user-images.githubusercontent.com/106718885/172030601-aa48e4a4-5128-4a58-95a7-de7b1679bb89.png)



