# * Use Monte Carlo methods to aid in modeling of COVID-19 spread.
# 
# ---
# With the global pandemic, there are a multitude of research efforts to model the impact and spread of the COVID-19 disease.  Some of this work has been published in papers using compartmental models including:
# 
# The compartments represent three groups:
# - Susceptible (S)
# - Infected (I)
# - Recovered (R)
# 
# In short, the _Susceptibles_ are healthy humans that can potentially become infected.  The _Infected_ group has received the virus and may be symptomatic or non-symptomatic.  In this model, someone who has been infected can recover and be part of the __Recovered__ group.  This model can be represented as a system of equations as:  
# 
# ### Basic SIR Model: 
# 
# $\frac{dS}{dt} = -bSI,$
# 
# $\frac{dI}{dt} = +bSI - cI,$
# 
# $\frac{dR}{dt} = +cI$
# 
# We are going to apply this model to the spread of infection in the state of Michigan to help us answer questions about COVID-19.
# 

# In[1]:


# Basic SIR Model
#
# This code runs the basic SIR model for a set of initial conditions.

# Import the usual libraries
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.integrate import odeint
from IPython.display import display, clear_output, set_matplotlib_formats


# The model is expressed in terms of ODEs and this is where
# they are defined in a manner consistent with what odeint wants to see.
# solve the system dy/dt = f(y, t)
# Define a function of the derivatives of this system
def derivatives_sir(y, t, b,c):
        Snow = y[0]
        Inow = y[1]
        Rnow = y[2]
        # the model equations 
        dSdt = -b*Snow*Inow
        dIdt = b*Snow*Inow-c*Inow
        dRdt = c*Inow
      
        return [dSdt, dIdt, dRdt]

# initial conditions
total_population = 9986857  # population of Michigan
susceptible_percent = 10  # percent of total population susceptible to COVID-19
S0 = total_population*susceptible_percent/100  # initial susceptible population  
I0 = 1                  # initial infected population
R0 = 0                  # initial recovered population
y0 = [S0, I0, R0]       # initial condition vector
t  = np.arange(0, 300, 1)   # time grid

# rate constants
total_population_modeled = S0+I0+R0
b =  8E-7    # infect rate
c = 5E-2  # recover rate

# solve the ODEs
soln = odeint(derivatives_sir, y0, t, args=(b, c))
S = soln[:, 0]
I = soln[:, 1]
R = soln[:, 2]

plt.figure(figsize=(6,6))
# plot results
plt.subplot(311)
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
#plt.ylim(0,10000000)
plt.xlabel('Days since start of infection')
plt.ylabel('Population')
plt.legend()


# ## Model Limitations
# 
# All models have limitations and assumptions that do not necessarily mirror reality.  
# This model contains numerous limitations and assumptions.  Discuss some of the limitations of this model with your group members. List at least three major assumptions below.

# <font size="+3">&#9998;</font>
# 
# 1. Lack of places in hospital
# 
# 
# 2. Peaple do not social distancing
# 
# 
# 3. population

# ## SIR MODEL with Social Distancing
# 
# A key strategy world-wide is to implement social distancing, which is really physical distancing.  The intention of this strategy is to slow the spread of the disease such that healthcare systems are not overly taxed all at once.  Moreover, the hope is that we can slow the spread to allow time to develop therapies and additional mitigation strategies.  
# 
# ### SIR Model with distancing:
# $\frac{dS_c}{dt} = -b_cS_cI,$
# 
# $\frac{dS_d}{dt} = -b_dS_dI,$
# 
# $\frac{dI}{dt} = +b_cS_cI + b_dS_dI - cI,$
# 
# $\frac{dR}{dt} = +cI$
# 
# where $S_c$ are the susceptible people not practicing distancing (close to each other) and $b_c$ is the infection rate for those not practicing distancing, $S_d$ and $b_d$ are the population and rate for those practicing distancing.  We can assume that a fixed portion of the population, $S_c$, will not be able to practice social distancing because they have essential responsibilities (healthcare workers, food+grocery industries, basic utilities, etc).  We will assume that those able to practice social distancing will have a lower infection rate, $b_d$, compared to those not distanced.
# 
# **A very important question on all of our minds is:**
# <h2><center> Q1: When can we end social distancing policies and return to normal life?</center></h2>
# 
# We will use the model below to help us think about answers to this question.
# 
# First, make a graphical diagram of this 4 compartment model.  Draw the model on paper and take photo; or use a text-based drawing (S --> I, etc) below
# 
# 

# In[8]:


# Place drawing here
s------------->I--------------R
  sc,bc,bd,sd


# **Run the code below to make sure it works and produces a plot. Review the code and make sure you understand how it works.**

# In[3]:


# SIR Model w/ Social Distancing
#
# This code runs the basic SIR model for a set of initial conditions.

# The model is expressed in terms of ODEs and this is where
# they are defined in a manner consistent with what odeint wants to see.
# solve the system dy/dt = f(y, t)
# Define a function of the derivatives of this system
def derivatives_sir_distance(y, t, b_c,b_d,c):
        S_c_now = y[0]
        S_d_now = y[1]
        Inow = y[2]
        Rnow = y[3]
        # the model equations 
        dScdt = -b_c*S_c_now*Inow
        dSddt = -b_d*S_d_now*Inow
        dIdt = b_c*S_c_now*Inow  +  b_d*S_d_now*Inow  -  c*Inow
        dRdt = c*Inow
           
        return [dScdt, dSddt, dIdt, dRdt]

# initial conditions
total_population = 9986857  # population of Michigan
susceptible_percent = 10  # percent of total population susceptible to COVID-19
S0 = total_population*susceptible_percent/100  # initial population susceptible
S_c0 = S0* .1           # initial population close to each other, assume 10% of population
S_d0 = S0* .9           # initial population distancing from each other, assume 90% of population
I0 = 1                  # initial infected population
R0 = 0                  # initial recovered population
y0 = [S_c0, S_d0, I0, R0]       # initial condition vector
t  = np.arange(0, 300, 1)   # time grid

# Rate constants
total_population_modeled = S0+I0+R0
b_c =  8E-7     # base infection rate for those close to each other
b_d =  b_c*.2   # infect rate for those distant to each other, assume 20% of those close to each other
c = 5E-2        # recover rate

# solve the ODEs
soln = odeint(derivatives_sir_distance, y0, t, args=(b_c,b_d, c))
S = soln[:, 0] + soln[:, 1]   # sum of all susceptible people
I = soln[:, 2]
R = soln[:, 3]

plt.figure(figsize=(6,6))
# plot results
plt.subplot(311)
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
#plt.ylim(0,10000000)
plt.xlabel('Days since start of infection')
plt.ylabel('Population')
plt.legend()


# ### Determining time of plateau of infection curve
# 
# To answer the question when can we end the social distancing, we would like to know when the cases of infections plateaus.  In this model, the plateau will occur when the infection curve peaks. The following function will determine the time of the peak of a curve.
# 

# In[4]:


# Determine time of maximum value of a curve
def find_time_max(curve, t):
    '''
    Inputs:
    curve : 1D numpy array
    t : 1D numpy array of time

    Returns : 
    t_max : the time when curve is at its maximum
    max : maximum value of curve
    '''
    index = np.argmax(curve)
    t_max = t[index]
    return t_max, np.max(curve)


# ## Simulating different conditions
# 
# As you are well aware, one of the main reasons we do not have an answer to when we can end social distancing is because there are many factors that we do not know yet about COVID-19.  We do not have clear evidence of the infection rate, recovery rate, the reason some people are susceptible and others not, etc.  Even in our simple model, we had to make many assumptions.  For the purposes of this assignment, we would like to simulate different conditions.  In other words, vary some of the parameters to look at the range of potential impact.
# 
# **Free parameters of interest:**
# 1. percent of total population susceptible to COVID-19, `susceptible_percent`
# 2. base infection rate of susceptible people, `b_c`
# 3. recovery rate of infected people, `c`
# 
# Considering there are many, many possible combinations of values for these three parameters, we will not be able to simulate all of these combinations.  Rather, we will adopt a Monte Carlo approach of trying random combinations of settings.  In order to do this, complete the code below which will perform multiple realizations (multiple simulations) of different conditions.
# 
# Answer the question: **What is the mean and range of times till peak infection levels using this model?**

# In[21]:


# Complete this code, Look for "FINISH" section to identify missing components

#  Package model solver into a single function for easy calling
#  This function will be called with different values of susceptible_percent,infect_rate,recover_rate
#  HINT: You can use code above to complete this


def solve_sir_w_distance(derivatives_sir_distance, susceptible_percent,infect_rate,recover_rate):
    '''
    Inputs:
    derivatives_sir_distance : function for derivatives
    susceptible_percent : percent of total population susceptible to COVID-19
    infect_rate : base infection rate
    recover_rate : recovery rate

    Output:
    I : number of infections for all t
    t : array of times 
    '''

    # initial conditions
    total_population = 9986857  # population of Michigan
    # susceptible_percent = 10  # percent of total population susceptible to COVID-19
    S0 = total_population*susceptible_percent/100  # initial population susceptible
    S_c0 = S0* .1           # initial population close to each other, assume 10% of population
    S_d0 = S0* .9           # initial population distancing from each other, assume 90% of population
    I0 = 1                  # initial infected population
    R0 = 0                  # initial recovered population
    y0 = [S_c0, S_d0, I0, R0]       # initial condition vector
    t  = np.arange(0, 300, 1)   # time grid

    # Rate constants
    total_population_modeled = S0+I0+R0
    # FINISH: PLACE assigment below
    b_c =  0.1         # base infection rate for those close to each other
    b_d =  b_c*.2   # infect rate for those distant to each other, assume 20% of those close to each other
    c = recover_rate        # recover rate
    
    # solve the ODEs
    # FINISH: PLACE odeint() call here

    
    S = soln[:, 0] + soln[:, 1]   # sum of all susceptible people
    I = soln[:, 2]
    R = soln[:, 3]

    # FINISH: PLACE RETURN STATEMENT HERE
    return I, t


# In[33]:


# Complete this code, Look for "FINISH" section to identify missing components

# Monte Carlo Simulation of multiple conditions

list_t_at_max=[]  # list of time to peak infection
list_susceptible_percent = []  # # list of simulated susceptible_percent values

num_realizations = 100;             # Number of conditions to simulate
for i in range(num_realizations):
    # Free parameters of interest:
    #     1. percent of total population susceptible to COVID-19, susceptible_percent
    #            Assume values uniformly distributed between 5-15 percent 
    #     2. base infection rate of susceptible people, b_c
    #             Assume values uniformly distributed between 3-10E-7
    #     3. recovery rate of infected people, c
    #             Assume values uniformly distributed between 0.003-0.007

    susceptible_percent_now = (15-5)*np.random.random()+5  # Range [5 15]
    b_c_now = (10E-7-3E-7)*np.random.random()+3E-7  # base infection rate, Range [3E-7 10E-7]
    # FINISH
    c_now = (0.007-0.003)*np.random.random()+0.003

    I_now, t_now = solve_sir_w_distance(derivatives_sir_distance, susceptible_percent_now,b_c_now,c_now)

    t_at_max, I_max = find_time_max(I_now, t_now);
    list_t_at_max.append(t_at_max)
    list_susceptible_percent.append(susceptible_percent_now)
    plt.figure(1)
    plt.plot(t_now,I_now)
    plt.xlabel('days after start infection')
    plt.ylabel('number infected')

# FINISH: Report mean and range of times to peak infection
print('mean peak time of infection: ',np.mean(list_t_at_max))
print('Range of time: ',min(list_t_at_max),max(list_t_at_max))


# In[34]:


# Put your code here


# ### Relationship of time till peak infection level and percent of population susceptible to COVID-19
# 
# What does the plot of all of your simulated conditions suggest in order for us to answer the question of when we can end social distancing?
# 
# Considering we may not be able to address our Q1, let's looks at a sub-question.
# We would like to answer the question: 
# <h3><center>Q2: How does the time till peak infection change with the number of people susceptible to COVID-19? </center></h3>
# 
# Specifically, it would be helpful to be able to state:  **For every 1% increase in the number of people susceptible to COVID-19, the time till peak infection will go down by YYY days.**
# 
# First, make a scatter plot of the time till peak infection versus total population susceptible to COVID-19.
# 
# Then, use the tools you learned previously to identify the linear relationship between days till peak and percent of total population susceptible to COVID-19.  Complete the statement (fill in YYY in statement above).

# In[45]:


# Put your code here
t_max = list()
x_list = list()
for i in range (100):
    t,c=solve_sir_w_distance(derivatives_sir_distance, i,0.5,0.03)
    t_max.append(c)
    x_list.append(i)
plt.plot(x_list,t_max)
# Complete this statement
print('For every 1% increase in the number of people susceptible to COVID-19, ')
print('   the time till peak infection will go down by','YYYY',' days.')


