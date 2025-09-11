_author_ = "Han Xie"
_date_ = '8.30.2025'

import sys, math, getopt, scipy,os
import numpy as np
import matplotlib.pyplot as plt

#read from file and print if needed
def read_file(read, is_print, print_):
    if not os.path.exists(read):
        print(f"Warning: The file '{read}' does not exist.")
        exit
    with open(read) as infile:
         title = infile.readline()
         title = title.split(" ")
         nested_list = [[] for i in range(len(title)-1)]
         for line in infile:
            l = line.rstrip()
            l_split = l.split(" ")
            for a in range(len(l_split)):
                nested_list[a].append(float(l_split[a]))
    
    for j in range(1,len(nested_list)):
        plt.plot(nested_list[0], nested_list[j], label = str(title[j]))
    plt.legend()
    plt.title("The Graph From the Read File")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    if is_print:
        print_split = print_.split(",")
        for form in print_split:
            plt.savefig("read_file_figure."+form, format=form)
    plt.show()
    plt.close()

# write the file into the function from arguments
def write_file(all_list, write_, plot):
    outfile = open(write_, 'w')
    n = 0
    a = 0

    outfile.write("x ")
    for sign in plot:
        outfile.write(sign + " ")
    outfile.write("\n")

    while (n < len(all_list[0])):
         a = 0
         while (a < len(all_list)):
             outfile.write(str(all_list[a][n])+ " ")
             a += 1
         outfile.write("\n")
         n+= 1
    outfile.close()

# read in the function and show the plot
def make_plot(is_plot, is_write, is_read,
             is_print, plot, write_, read, print_):
 n = 0
 if not(is_plot):
     print("no plot need to be printed")
     exit
 else:
    plot_ = plot.split(",") 
    x = np.arange(-10,10.05,0.05).tolist()
    all_list = [x]
    for math_sign in plot_:
        n += 1
        match math_sign:
            case "sin":
                y = [math.sin(x) for x in x]
            case "cos":
                y= [math.cos(x) for x in x]
            case "sinc":
                y = [scipy.special.sinc(x) for x in x]
        all_list.append(y)
        plt.plot(x, y, label = str(math_sign))
    plt.legend()
    plt.title(str(plot) + " graph between -10 and 10")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    if is_print:
        print_split = print_.split(",")
        for form in print_split:
            plt.savefig("function_figure."+form, format=form)
    plt.show()
    if is_write:
        write_file(all_list, write_, plot)
 if is_read:
    read_file(read, is_print, print_)

# read in the sys argument
splited_args = sys.argv[1:]

# initialization
is_plot = False
is_write = False
is_read = False
is_print = False
plot = []
print_= ""
write_= ""
read = ""

# split the arguments
for item in splited_args:
     s = item.split("=")
     if s[0] == '--function':
         is_plot = True
         plot = s[1]
     if s[0] == '--print':
         is_print = True
         print_= s[1]
     if s[0] == '--write':
         is_write = True
         write_ = s[1]
     if s[0] == '--read_from_file':
         is_read = True
         read = s[1]
         break

# send into function
make_plot(is_plot, is_write, is_read,
             is_print, plot, write_, read, print_)