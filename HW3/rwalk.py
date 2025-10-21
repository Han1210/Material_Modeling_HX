import numpy as np
import matplotlib.pyplot as plt
import sys, math, getopt, scipy,os
import sys, math, getopt, scipy,os
if len(sys.argv) > 1:
    s = sys.argv[1].split("=")
    if s[0] == '--part':
        part = float(s[1])

#initialization
n = 100
walker_list = []
n_list = []
for i in range(n):
    walker_list.append([0,0])

def random_assign(walker):
    step = np.random.choice(['up', 'down', 'left', 'right'])
    if step == 'left':
        walker[0] -= 1
    elif step == 'right':
        walker[0] += 1
    elif step == 'up':
        walker[1] += 1
    elif step == 'down':
        walker[1] -= 1

def iteration(walker_list):
    max = 10**4
    x_list = []
    x_2_list = []
    r_2_list = []
    n_list = []
    for i in range(max):
        n_list.append(i)
        x = 0
        x_2 = 0
        r_2 = 0
        for walker in walker_list:
            random_assign(walker)
            x += walker[0]
            x_2 += (walker[0])**2
            r_2 += walker[0]**2 + walker[1]**2
        x = x/len(walker)
        x_2 = x_2/len(walker)
        r_2 = r_2/len(walker)
        x_list.append(x)
        x_2_list.append(x_2)
        r_2_list.append(r_2)
    return x_list, x_2_list, r_2_list, n_list

def part_1(x_list, x_2_list):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(x_list, label='⟨x⟩', color='blue')
    plt.title('Mean Position in x-direction vs Steps')
    plt.xlabel('Number of Steps (n)')
    plt.ylabel('Mean Position ⟨x⟩')
    plt.axhline(0, color='grey', lw=0.5, ls='--')
    plt.legend()
    plt.grid()

    # Plot D⟨(xn)²⟩
    plt.subplot(1, 2, 2)
    plt.plot(x_2_list, label='D(x^2)', color='green')
    plt.title('Mean Square Distance in x-direction vs Steps')
    plt.xlabel('Number of Steps (n)')
    plt.ylabel('<x^2>')
    plt.grid()

    plt.tight_layout()
    plt.savefig("rwalker_p1.pdf")
    plt.show()


def part_2(r_2_list, n_list):
    fit_x = np.polyfit(n_list[2000:], r_2_list[2000:], 1)
    print("The diffusion constant is "+ str(fit_x[0])) # 53.67995319387034
    plt.plot(r_2_list, label='⟨x⟩', color='blue')
    plt.title('Mean square distance from origin vs Steps')
    plt.xlabel('time (t)')
    plt.ylabel('Mean square distance from origin ⟨r^2⟩')
    plt.axhline(0, color='grey', lw=0.5, ls='--')
    plt.legend()
    plt.grid()
    plt.savefig("rwalker_p2.pdf")
    plt.show()

x_list, x_2_list, r_2_list, n_list = iteration(walker_list)
if part == 1:
    part_1(x_list, x_2_list)
elif part == 2:
    part_2(r_2_list, n_list)
            
