import sys, math, getopt, scipy,os
import numpy as np
import matplotlib.pyplot as plt
#does it need a list of time steps?

time_step = 10 #by default

if len(sys.argv) > 1:
    s = sys.argv[1].split("=")
    if s[0] == '--plot':
        degree = float(s[1])

half_life = 5700
mass = 10**(-9) #g
N_A = 6.022*(10**(23))
atomic_mass = 14.003241 #g/mol
mole = mass/atomic_mass #mol
N_i = mole * N_A
tau = half_life / math.log(2)
t = time_step
N_n = N_i
N_a = N_i
t_list = [0]
numeric_N_list = [N_n]
analytical_N_list = [N_a]
while t <= 20000:
    N_n = N_n - ((N_n/ tau)*time_step) #use numeric way
    s = - t / tau 
    N_a = N_i * math.exp(s) #use analytical way
    second_order = ((((time_step/tau)**2)/2)*N_n)
    if (time_step == 1000 and t == 12000) or t == 2 * half_life :
        print("Time step of " + str(time_step) + " at 2* half-lives:")
        print("Numeric Analysis Value: " + str(N_n))
        print("Analytical Analysis Value: " + str(N_a))
        print("The percentage deviation is " + str(N_n/N_a)) #percentage deviation
        print("The second order is "+ str(second_order)) #second-order
        print("Difference among two analysis is " + str(N_a-N_n)) #difference among two analysis
        print("Difference among the deviation and second order is " + str((N_a-N_n)-second_order)) #difference among the deviation and second order
    t_list.append(t)
    numeric_N_list.append(N_n)
    analytical_N_list.append(N_a)
    t += time_step
plt.plot(t_list, numeric_N_list,label = "numeric solution")
plt.plot(t_list, analytical_N_list, linestyle = "--" ,label = "analytical solution")
plt.legend()
plt.title("the activity of the sample over 20000 years with time step of " + str(time_step))
plt.xlabel("time(years)")
plt.ylabel("activity(number of atom)")
plt.savefig(str(time_step)+".pdf")
plt.show()




