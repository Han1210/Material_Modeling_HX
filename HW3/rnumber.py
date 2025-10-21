import random
import math
import matplotlib.pyplot as plt
import sys, math, getopt, scipy,os

if len(sys.argv) > 1:
    s = sys.argv[1].split("=")
    if s[0] == '--part':
        part = float(s[1])

def plt_R(N, bin):
    x_list=[]
    for i in range(N):
        n=random.uniform(0,1)
        x_list.append(n)
        i+=1
    plt.hist(x_list,bins=bin,label=str(bin)+" subdivision")
    plt.legend()

def Part1():
    plt.figure()
    #1000 number, 10 subdivision
    plt_R(1000,10)
    #1000 number, 20 subdivision
    plt_R(1000,20)
    #1000 number, 50 subdivision
    plt_R(1000,50)
    #1000 number, 100 subdivision
    plt_R(1000,100)
    plt.savefig("1000 subdivision.pdf")
    plt.figure()
    #1000,000 number, 10 subdivision
    plt_R(1000000,10)
    #1000,000 number, 20 subdivision
    plt_R(1000000,20)
    #1000,000 number, 50 subdivision
    plt_R(1000000,50)
    #1000,000 number, 100 subdivision
    plt_R(1000000,100)
    plt.savefig("1000000 subdivision.pdf")

def Gaussian(x_list):
    y_list = []
    sigma = 1.0
    max_P = 1/(math.sqrt(2*math.pi))
    for x in x_list:
        p1 = 1/(sigma*math.sqrt(2*math.pi))
        p2 = math.exp(-(x**2)/(2*(sigma**2)))
        y_list.append(p1*p2*100000)
    return y_list

def get_x():
    x_list=[]
    for x in range(-400,400):
        x_list.append(x*0.01)
        x+=1
    return x_list

def Part2(N):
    z0_list=[]
    for i in range(N):
        u1=random.uniform(0,1)
        u2=random.uniform(0,1)
        z0 = math.sqrt(-2*math.log(u1))*math.cos(2*math.pi*u2)
        z0_list.append(z0)
        i+=1
    
    z_list = z0_list
        
    x_list = get_x()
    y_list = Gaussian(x_list)

    plt.hist(z_list,bins=100)
    plt.plot(x_list,y_list)
    plt.savefig("rnumber_p_2.pdf")
    plt.show()

#build system to call
if part == 1:
    Part1()
elif part == 2:
    Part2(1000000)
    
    






