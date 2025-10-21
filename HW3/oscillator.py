import sys, math, getopt, scipy,os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys, math, getopt, scipy,os
if len(sys.argv) > 1:
    s = sys.argv[1].split("=")
    if s[0] == '--part':
        part = float(s[1])

g=9.8 #m/s^2
l=9.8 #m
m = 1 #kg
gamma=0.25 #s^-1
aD=0.2 #rad/s^2
theta=math.radians(60) #initial 60 degree
dt=0.1
t=0
omega=0
ohm=0.71
theta_list_RK, theta_list_EC=[math.degrees(theta)],[math.degrees(theta)]
time_list=[t]
omega_list_RK,omega_list_EC =[math.degrees(omega)],[math.degrees(omega)]
ohm_list = []
n=0.05

while n <= 2:
    ohm_list.append(n)
    n+=0.001

#calculating the k value for Runge Kutta 4th order
def k_value(theta, omega, g,l,t,ohm, gamma, dt, aD):
    k_omega =-g/l*theta-2*gamma*omega+aD*math.sin(ohm*t)
    k_theta = omega #w=dtheta/dt dw=dtheta^2/dt^2
    return k_theta,k_omega

#calculating the k value for Runge Kutta 4th order part 4
def part4_k_value(theta, omega, g,l,t,ohm, gamma, dt, aD):
    k_omega =-g/l*math.sin(theta)-2*gamma*omega+aD*math.sin(ohm*t)
    k_theta = omega #w=dtheta/dt dw=dtheta^2/dt^2
    return k_theta,k_omega

#calculate the omega and theta value with Runge Kutta 4th order method
def Runge_Kutta(omega, theta, dt , g,l,t,ohm, gamma, aD):
    k1_theta,k1_omega=k_value(theta, omega, g,l,t,ohm, gamma,dt, aD)
    k2_theta,k2_omega=k_value(theta+(dt/2)*k1_theta, omega+(dt/2)*k1_omega, g,l,t+dt/2,ohm, gamma,dt, aD)
    k3_theta,k3_omega=k_value(theta+(dt/2)*k2_theta, omega+(dt/2)*k2_omega, g,l,t+dt/2,ohm, gamma,dt, aD)
    k4_theta,k4_omega=k_value(theta+dt*k3_theta, omega+dt*k3_omega, g,l,t+dt,ohm, gamma,dt, aD)
    omega_new = omega + (1/6) * (k1_omega + 2*k2_omega + 2*k3_omega + k4_omega)*dt
    theta_new = theta + (1/6) * (k1_theta + 2*k2_theta + 2*k3_theta + k4_theta)*dt
    return omega_new, theta_new

#calculate the omega and theta value with Runge Kutta 4th order method
def part4_Runge_Kutta(omega, theta, dt , g,l,t,ohm, gamma, aD):
    k1_theta,k1_omega=part4_k_value(theta, omega, g,l,t,ohm, gamma,dt,aD)
    k2_theta,k2_omega=part4_k_value(theta+(dt/2)*k1_theta, omega+(dt/2)*k1_omega, g,l,t+dt/2,ohm, gamma,dt,aD)
    k3_theta,k3_omega=part4_k_value(theta+(dt/2)*k2_theta, omega+(dt/2)*k2_omega, g,l,t+dt/2,ohm, gamma,dt,aD)
    k4_theta,k4_omega=part4_k_value(theta+dt*k3_theta, omega+dt*k3_omega, g,l,t+dt,ohm, gamma,dt, aD)
    omega_new = omega + (1/6) * (k1_omega + 2*k2_omega + 2*k3_omega + k4_omega)*dt
    theta_new = theta + (1/6) * (k1_theta + 2*k2_theta + 2*k3_theta + k4_theta)*dt
    return omega_new, theta_new

#calculate the omega and theta value with euler cromer method
def Euler_Cromer(theta, omega, g,l,t,ohm, gamma,dt, aD):
    domega_dt =-(g/l)*theta-2*gamma*omega+aD*math.sin(ohm*t)
    new_omega = omega + domega_dt*dt
    new_theta = theta + new_omega*dt
    return new_omega,new_theta

#calculate the omega and theta value with euler cromer method
def part4_Euler_Cromer(theta, omega, g,l,t,ohm, gamma,dt, aD):
    domega_dt =-(g/l)*math.sin(theta)-2*gamma*omega+aD*math.sin(ohm*t)
    new_omega = omega + domega_dt*dt
    new_theta = theta + new_omega*dt
    return new_omega,new_theta

#calculate the theta and omega values with two different methods
def calc_theta(theta, omega, g,l,ohm, gamma, dt, aD, part, trange):
    RK_theta = theta
    RK_omega = omega
    EC_theta = theta
    EC_omega = omega
    theta_list_RK,theta_list_EC,time_list,omega_list_RK,omega_list_EC=[],[],[],[],[]
    diff_theta = 0
    diff_omega = 0
    t=0
    while t<=trange:
        t += dt
        if part==2:
            EC_omega,EC_theta = Euler_Cromer(EC_theta, EC_omega, g,l,t,ohm, gamma,dt, aD)
            RK_omega,RK_theta = Runge_Kutta(RK_omega, RK_theta, dt, g,l,t,ohm, gamma, aD)
        elif part==4:
            EC_omega,EC_theta = part4_Euler_Cromer(EC_theta, EC_omega, g,l,t,ohm, gamma,dt, aD)
            RK_omega,RK_theta = part4_Runge_Kutta(RK_omega, RK_theta, dt, g,l,t,ohm, gamma, aD)

        theta_list_RK.append(math.degrees(RK_theta))
        theta_list_EC.append(math.degrees(EC_theta))
        time_list.append(t)
        omega_list_RK.append(math.degrees(RK_omega))
        omega_list_EC.append(math.degrees(EC_omega))
        diff_theta += abs(EC_theta-RK_theta)
        diff_omega += abs(EC_omega-RK_omega)
    #print("average difference in theta is "+str(diff_theta/t))
    #print("average difference in omega is "+str(diff_omega/t))
    return time_list, theta_list_RK, theta_list_EC, omega_list_RK,omega_list_EC

def RK_only(theta, omega, g,l,ohm, gamma, dt, trange, aD):
    RK_theta = theta
    RK_omega = omega
    t = 0
    theta_list = [theta]
    omega_list = [omega]
    time_list = [t]
    diff_theta = 0
    diff_omega = 0
    while t<=trange:
        t += dt
        RK_omega,RK_theta = Runge_Kutta(RK_omega, RK_theta, dt, g,l,t,ohm, gamma, aD)
        theta_list.append(RK_theta)
        omega_list.append(RK_omega)
        time_list.append(t)
    return time_list, theta_list, omega_list

def amp_list(ohm_list,theta, omega, g,l,t,ohm, gamma, dt, aD):
    time_list, theta_list = [],[]
    A_list, phi_list = [],[]
    for ohm in ohm_list:
        time_list, theta_list, omega_list=RK_only(theta, omega, g,l,ohm, gamma, dt, 100, aD)
        time_list = time_list[500:]
        theta_list = theta_list[500:]
        def sin_func(t, A, phi,D):
            return A*np.sin(ohm*t+phi)+D
        parameters, covariance = curve_fit(sin_func, time_list, theta_list)
        A, phi, D = parameters
        if A < 0:
            A = -A
            phi += np.pi
        phi = -((phi + np.pi) % (2 * np.pi) - np.pi)
        A_list.append(A)
        phi_list.append(phi)
    return A_list, phi_list

def FWHM(A_list,ohm_list, aD):
    half_width = max(A_list)/2
    min_diff = 100
    ohm = 0
    reached = False
    count = 0
    t1 = 0
    t2 = 0
    while count < len(A_list):
        A = A_list[count]
        if A==max(A_list):
            reached = True
            min_diff = 100
            ohm = 0
        difference = abs(A-(max(A_list)/2))
        if difference <= min_diff and not reached:
            ohm=A
            t1=count
            min_diff = difference
        elif difference <= min_diff and reached:
            ohm=A
            t2=count
            min_diff = difference
        count += 1
    return(ohm_list[t2]-ohm_list[t1])

def func(theta, omega, g,l,t,ohm, gamma, dt, ohm_list, aD):
    time_list, theta_list_RK, theta_list_EC, omega_list_RK,omega_list_EC = calc_theta(theta, omega, g,l,ohm, gamma, dt, aD, 2, 50)
    A_list, phi_list = amp_list(ohm_list,theta, omega, g,l,t,ohm, gamma, dt, aD)
    FWHM_v = FWHM(A_list,ohm_list, aD)
    return time_list, theta_list_RK, theta_list_EC, omega_list_RK,omega_list_EC, A_list, phi_list,FWHM_v

def Part_2_plot(theta, omega, g,l,t,ohm, gamma, dt, ohm_list, aD):
    time_list, theta_list_RK, theta_list_EC, omega_list_RK,omega_list_EC, A_list, phi_list,FWHM_v = func(theta, omega, g,l,t,ohm, gamma, dt, ohm_list, aD)
    print(FWHM_v)
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(time_list, theta_list_RK,label = "Runge-Kutta 4th order method")
    plt.plot(time_list, theta_list_EC,label = "Euler-Cromer method", linestyle='--')
    plt.title("Angle")
    plt.xlabel("time(s)")
    plt.ylabel("θ(t)(degree)")
    plt.legend()
    
    plt.subplot(2,1,2)
    plt.plot(time_list,omega_list_RK,label = "Runge-Kutta 4th order method")
    plt.plot(time_list, omega_list_EC,label = "Euler-Cromer method", linestyle='--')
    plt.title("Angular Velocity")
    plt.xlabel("time(s)")
    plt.ylabel("ω(t)(degree/rads)")
    plt.legend()
    plt.legend()
    plt.tight_layout()
    plt.savefig("ECvsRK.pdf")
    plt.show()

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(ohm_list, A_list,label = "amplitude")
    plt.title("amplitude")
    plt.xlabel("ohm(s^-2)")
    plt.ylabel("θ(ohm)(degree)")

    plt.subplot(2,1,2)
    plt.plot(ohm_list,phi_list,label = "phase shift")
    plt.title("phase shift")
    plt.xlabel("ohm(s^-2)")
    plt.ylabel("phi(ohm)(s)")
    plt.tight_layout()
    plt.savefig("amp_and_phase_shift.pdf")
    plt.show()

def Calc_E(ohm_list,theta, omega, g,l,t,ohm, gamma, dt, m, aD):
    A_list, phi_list = amp_list(ohm_list,theta, omega, g,l,t,ohm, gamma, dt, aD)
    ohm_i = A_list.index(max(A_list))
    ohm_max = ohm_list[ohm_i]
    time_list, theta_list, omega_list = RK_only(theta, omega, g,l,ohm_max, gamma, dt, 50, aD)
    KE = []
    PE = []
    E = []
    for i in range(len(theta_list)):
        omega = omega_list[i]
        theta = theta_list[i]
        ke = (1/2)*m*((omega*l)**2)
        pe = m*g*l*(1/2)*(theta**2)
        e = ke+pe
        KE.append(ke)
        PE.append(pe)
        E.append(e)
    return KE,PE,E,time_list

def Part_3_plot(ohm_list,theta, omega, g,l,t,ohm, gamma, dt, m, aD):
    KE,PE,E,time_list = Calc_E(ohm_list,theta, omega, g,l,t,ohm, gamma, dt, m, aD)
    plt.plot(time_list,KE,label = "Kinetics Energy")
    plt.plot(time_list,PE,label = "Potential Energy")
    plt.plot(time_list,E,label = "Total Energy")
    plt.title("Energy over time")
    plt.xlabel("t(s)")
    plt.ylabel("E(J)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("energy.pdf")
    plt.show()

def Part_4_calc(ohm_list,theta, omega, g,l,t, gamma, dt, m, aD):
    A_list, phi_list = amp_list(ohm_list,theta, omega, g,l,t,ohm, gamma, dt, aD)
    ohm_i = A_list.index(max(A_list))
    ohm_max = ohm_list[ohm_i]
    time_list, theta_list_RK_4, theta_list_EC, omega_list_RK_4,omega_list_EC=calc_theta(theta, omega, g,l, ohm_max, gamma, dt, aD, 4, 100)
    time_list, theta_list_RK_2, theta_list_EC, omega_list_RK_2,omega_list_EC=calc_theta(theta, omega, g,l, ohm_max, gamma, dt, aD, 4, 100)
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(time_list, theta_list_RK_4,label = "non-linear")
    plt.plot(time_list, theta_list_RK_2,label = "linear", linestyle='--')
    plt.title("Angle with non-linear effect,aD: " + str(aD))
    plt.xlabel("time(s)")
    plt.ylabel("θ(t)(degree)")
    plt.legend()
    
    plt.subplot(2,1,2)
    plt.plot(time_list,omega_list_RK_4,label = "non-linear")
    plt.plot(time_list, omega_list_RK_2,label = "linear", linestyle='--')
    plt.title("Angular Velocity with non-linear effect,aD: " + str(aD))
    plt.xlabel("time(s)")
    plt.ylabel("ω(t)(degree/rads)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ECvsRK" + str(aD)+".pdf")
    plt.show()

def Part_4_plot(ohm_list,theta, omega, g,l,t, gamma, dt, m):
    Part_4_calc(ohm_list,theta, omega, g,l,t, gamma, dt, m, 0.2)
    Part_4_calc(ohm_list,theta, omega, g,l,t, gamma, dt, m, 1.2)

def linear(a,b,x):
    return a*x+b

def Part_5_calc(theta, omega, g,l,t, gamma, dt, m, aD):
    theta = math.radians(60)
    time_list, theta_list_RK1, theta_list_EC1, omega_list_RK1,omega_list_EC1=calc_theta(theta, omega, g,l, ohm, gamma, dt, aD, 4, 100)
    theta = math.radians(60)+0.001
    time_list, theta_list_RK2, theta_list_EC2, omega_list_RK2,omega_list_EC2=calc_theta(theta, omega, g,l, ohm, gamma, dt, aD, 4, 100)
    delta_theta = []
    a_list = []
    for i in range(len(theta_list_RK1)):
        diff = abs(theta_list_RK1[i]-theta_list_RK2[i])
        delta_theta.append(abs(diff))
        a = math.log(diff/0.001)
        a_list.append(a)
    return time_list, a_list #delta_theta

def Part_5_plot(omega, g,l,t, gamma, dt, m):
    ohm = 0.666
    plt.figure()
    plt.subplot(3,1,1)
    aD = 0.2
    time_list, delta_theta = Part_5_calc(theta, omega, g,l,t, gamma, dt, m, aD)
    parameters, covariance = curve_fit(linear, time_list, delta_theta)
    print("lambda:" + str(parameters[1]))

    plt.plot(time_list,delta_theta)
    plt.title("delta theta for non-linear pendulum with aD=0.2")
    plt.xlabel("time(s)")
    plt.ylabel("delta theta(t)(degree)")
    
    plt.subplot(3,1,2)
    aD = 0.5
    time_list, delta_theta = Part_5_calc(theta, omega, g,l,t, gamma, dt, m, aD)
    parameters, covariance = curve_fit(linear, time_list, delta_theta)
    print("lambda:" + str(parameters[1]))
    plt.plot(time_list,delta_theta)
    plt.title("delta theta for non-linear pendulum with aD=0.5")
    plt.xlabel("time(s)")
    plt.ylabel("delta theta(t)(degree)")

    plt.subplot(3,1,3)
    aD = 1.2
    time_list, delta_theta = Part_5_calc(theta, omega, g,l,t, gamma, dt, m, aD)
    parameters, covariance = curve_fit(linear, time_list, delta_theta)
    print("lambda:" + str(parameters[1]))
    plt.plot(time_list,delta_theta)
    plt.title("delta theta for non-linear pendulum with aD=1.2")
    plt.xlabel("time(s)")
    plt.ylabel("delta theta(t)(degree)")

    plt.savefig("part5.pdf")
    plt.tight_layout()
    plt.show()

if part == 2:
    Part_2_plot(theta, omega, g,l,t,ohm, gamma, dt, ohm_list, aD)
elif part == 3:
    Part_3_plot(ohm_list,theta, omega, g,l,t,ohm, gamma, dt, m, aD)
elif part == 4:
    Part_4_plot(ohm_list,theta, omega, g,l,t, gamma, dt, m)
elif part == 5:
    Part_5_plot(omega, g, l, t, gamma, dt, m)