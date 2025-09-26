import sys, math, getopt, scipy,os
import numpy as np
import matplotlib.pyplot as plt
degree = 45
DEBUG = False

if len(sys.argv) > 1:
    s = sys.argv[1].split("=")
    if s[0] == '--plot':
        degree = float(s[1])

theta = math.radians(degree) #degree
m = 0.046 #kg
vi = 70 #m/s
rou = 1.29 #kg/m^3
A = 0.0014 #m^2
g = 9.81 #m/s^2

#initialization
time_step = 0.0001 #step
Xx_i = 0 # ideal trajectory
Xy_i = 0 
Xx_list_i = [0] 
Xy_list_i = [Xy_i]
Xx_s = 0 #smooth golf ball with drag
Xy_s = 0
Xx_list_s = [Xx_s] 
Xy_list_s = [Xy_s]
Xx_d = 0 # dimpled golf ball with drag
Xy_d = 0
Xx_list_d = [Xx_d] 
Xy_list_d = [Xy_d]
Xx_sd = 0 # dimpled golf ball with drag and spin
Xy_sd = 0
Xx_list_sd =[Xx_sd]
Xy_list_sd = [Xy_sd]
t = time_step
v_x = vi* math.cos(theta) # initial velocity used for calculating F drag
v_y = vi* math.sin(theta)
v_s_x = vi* math.cos(theta) # initial velocity used for calculating F drag
v_s_y = vi* math.sin(theta)
v_d_x = vi* math.cos(theta) # initial velocity used for calculating F drag
v_d_y = vi* math.sin(theta)
v_s_d_x = vi* math.cos(theta) # initial velocity used for calculating F drag and spin
v_s_d_y = vi* math.sin(theta)

while (Xy_i >= 0) or (Xy_s >= 0) or (Xy_d >= 0) or (Xy_sd >= 0):
    #ideal case
    if Xy_i >= 0:
        v_x = v_x
        v_y = v_y - g * time_step
        Xx_i += v_x * time_step
        Xy_i += v_y * time_step
        if Xy_i > 0:
            v_x_temp = v_x
            v_y_temp = v_y
            vmag_temp  = math.sqrt((v_x_temp ** 2)+(v_y_temp ** 2))
            Xx_list_i.append(Xx_i)
            Xy_list_i.append(Xy_i)

    #smooth golf ball with drag
    if Xy_s >= 0:
        v_s = math.sqrt((v_s_x ** 2)+(v_s_y ** 2))
        C = 1/2
        v_F_x = ((-C*rou*A*abs(v_s)*v_s_x)/m) * time_step
        v_F_y = (((-C*rou*A*(abs(v_s)*v_s_y))/m)-g) * time_step
        v_s_x = v_s_x + v_F_x
        v_s_y = v_s_y + v_F_y
        Xx_s += (v_s_x) * time_step
        Xy_s += (v_s_y)* time_step 
        if Xy_s > 0:  
            v_s_x_temp = v_s_x
            v_s_y_temp = v_s_y
            v_s_temp = v_s
            C_temp_s = C
            Xx_list_s.append(Xx_s) 
            Xy_list_s.append(Xy_s)    

    #dimpled golf ball with drag
    if Xy_d >= 0:
        v_d = math.sqrt((v_d_x ** 2)+(v_d_y ** 2))
        if v_d <= 14:
            C = 1/2
        else:
            C = 7/v_d

        v_F_x = ((-C*rou*A*(abs(v_d)*v_d_x))/m) * time_step
        v_F_y = (((-C*rou*A*(abs(v_d)*v_d_y))/m)-g) * time_step
        v_d_x = v_d_x + v_F_x
        v_d_y = v_d_y + v_F_y
        Xx_d += (v_d_x) * time_step
        Xy_d += (v_d_y)* time_step
        if Xy_d > 0:
            v_d_x_temp = v_d_x
            v_d_y_temp = v_d_y
            v_d_temp = v_d
            C_temp_d = C
            Xx_list_d.append(Xx_d) 
            Xy_list_d.append(Xy_d)
    
    #dimpled golf ball with drag and spin
    if Xy_sd >= 0:
        v_sd = math.sqrt((v_s_d_x ** 2)+(v_s_d_y ** 2))
        if v_sd <= 14:
            C = 1/2
        else:
            C = 7/v_sd
        v_F_s_d_x = (((-C*rou*A*(abs(v_sd)*v_s_d_x))/m)-(0.25* v_s_d_y))*time_step
        v_F_s_d_y = (((-C*rou*A*(abs(v_sd)*v_s_d_y))/m)+(0.25* v_s_d_x)-g)*time_step
        v_s_d_x = v_s_d_x + v_F_s_d_x
        v_s_d_y = v_s_d_y + v_F_s_d_y
        Xx_sd += v_s_d_x * time_step
        Xy_sd += v_s_d_y * time_step
        if Xy_sd > 0:
            v_s_d_x_temp = v_s_d_x
            v_s_d_y_temp = v_s_d_y
            v_sd_temp = v_sd
            C_temp_sd = C
            Xx_list_sd.append(Xx_sd) 
            Xy_list_sd.append(Xy_sd)           

    t += time_step

if DEBUG:
    print("Ideal case \n x: " + str(Xx_list_i[-1]) + "m \n y: "
                 + str(Xy_list_i[-1]) + "m \n vmag: " + str(vmag_temp) + "m/s \n vx: " 
                 + str(v_x_temp) + "m/s \n vy:" + str(v_y_temp) + "m/s \n")
    print("Smooth golf ball with drag case \n x: " + str(Xx_list_s[-1]) + "m \n y: "
                 + str(Xy_list_s[-1]) + "m \n vmag: " + str(v_s_temp) + "m \n C: "
                  + str(C_temp_s) + "\n vx: " + str(v_s_x_temp) + "m/s \n vy:" + str(v_s_y_temp) + "m/s \n")
    print("Dimpled golf ball with drag case \n x: " + str(Xx_list_d[-1]) + "m \n y: "
                 + str(Xy_list_d[-1]) + "m \n vmag: " + str(v_d_temp) + "m \n C: "
                  + str(C_temp_d) + "\n vx: " + str(v_d_x_temp) + "m/s \n vy:" + str(v_d_y_temp) + "m/s \n") 
    print("Dimpled golf ball with drag and spin case \n x: " + str(Xx_list_sd[-1]) + "m \n y: "
                 + str(Xy_list_sd[-1]) + "m \n vmag: " + str(v_sd_temp) + "m \n C: "
                  + str(C_temp_sd) + "\n vx: " + str(v_s_d_x_temp) + "m/s \n vy:" + str(v_s_d_y_temp) + "m/s \n")    

plt.plot(Xx_list_i, Xy_list_i,label = "ideal trajectory")
plt.plot(Xx_list_s, Xy_list_s,label = "Smooth golf ball with drag")
plt.plot(Xx_list_d, Xy_list_d,label = "Dimpled golf ball with drag")
plt.plot(Xx_list_sd, Xy_list_sd,label = "Dimpled golf ball with drag and spin")

plt.legend()
plt.title("The trajectory of a golf ball with the starting angle degree of " + str(degree))
plt.xlabel("x(m)")
plt.ylabel("y(m)")
plt.savefig("The trajectory at "+ str(degree)+".pdf")
plt.show()