import numpy as np
import matplotlib.pyplot as plt
import sys, math, getopt, scipy,os
if len(sys.argv) > 1:
    s = sys.argv[1].split("=")
    if s[0] == '--part':
        part = float(s[1])

def jacobi_relaxation(tol, grid_size=15, max_iterations=10000):
    # Initialize the potential grid to zero
    Phi_j = np.zeros((grid_size, grid_size))

    # Set the boundary conditions 
    Phi_j[:, 10] = 0
    Phi_j[:, -10] = 0
    Phi_j[10, :] = 0
    Phi_j[-10, :] = 0

    # Set the dipole charges
    Q = 1
    Phi_j[grid_size // 2, grid_size // 2] = Q
    Phi_j[grid_size // 2, grid_size // 2 - 1] = -Q

    for it in range(max_iterations):
        Phi_new_j = np.copy(Phi_j)

        # Update potential values
        for i in range(1, grid_size - 1):
            for j in range(1, grid_size - 1):
                if not ((i == grid_size // 2 and j == grid_size // 2) or 
                        (i == grid_size // 2 and j == grid_size // 2 - 1)):
                    Phi_new_j[i, j] = (Phi_j[i + 1, j] + Phi_j[i - 1, j] + 
                                     Phi_j[i, j + 1] + Phi_j[i, j - 1]) / 4
                
        # Calculate the maximum change
        delta_phi_j = np.max(np.abs(Phi_new_j - Phi_j))
        Phi_j = Phi_new_j

        # Check for convergence
        if delta_phi_j < tol:
            return (it + 1), Phi_j  # Return the number of iterations taken to converge

    return max_iterations, Phi_j  # If it doesn't converge, return max iterations

def Part_1():
    itr, phi = sor_method(1e-3)
    #plotting
    plt.figure(figsize=(8, 8))
    contour = plt.contourf(phi, levels=50)
    plt.colorbar(label='Potential (V)')
    plt.title('Electric Potential Field Using Jacobi Method')
    plt.xlabel('Grid X')
    plt.ylabel('Grid Y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("poission_p_1.pdf")
    plt.show()

def Part_2():
    # Range of tolerances to test
    tolerances = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    N_iter_list = [] 

    # Run Jacobi relaxation for each tolerance and record iterations
    for tolerance in tolerances:
        n_iter,phi = jacobi_relaxation(tolerance)
        N_iter_list.append(n_iter)
        print(f'Tolerance: {tolerance}, Iterations: {n_iter}')

    #plotting
    plt.figure(figsize=(10, 6))
    plt.loglog(tolerances, N_iter_list, marker='o')
    plt.title('Number of Iterations vs Tolerance')
    plt.xlabel('Tolerance (ε)')
    plt.ylabel('Number of Iterations (N_iter)')
    plt.grid(True, which="both", ls="--")
    plt.savefig("poission_p_2.pdf")
    plt.show()

def sor_method(tol, grid_size=15, max_iterations=10000000, a=0.5):
    # Initialize the potential grid to zero
    Phi = np.zeros((grid_size, grid_size))

    # Set the boundary conditions 
    Phi[:, 0] = 0
    Phi[:, -1] = 0
    Phi[0, :] = 0
    Phi[-1, :] = 0

    # Set the dipole charges
    Q = 1
    Phi[grid_size // 2, grid_size // 2] = Q  # Positive charge
    Phi[grid_size // 2, grid_size // 2 - 1] = -Q  # Negative charge

    for it in range(max_iterations):
        Phi_new = np.copy(Phi)  # Create new grid for the updated values

        # Update potential values
        for i in range(1, grid_size - 1):
            for j in range(1, grid_size - 1):
                if not ((i == grid_size // 2 and j == grid_size // 2) or 
                        (i == grid_size // 2 and j == grid_size // 2 - 1)):
                    Phi_new[i, j] = (Phi[i + 1, j] + Phi[i - 1, j] + 
                                     Phi[i, j + 1] + Phi[i, j - 1]) / 4
                
        # Calculate the maximum change
        delta_phi = np.max(np.abs(Phi_new - Phi))
        Phi = delta_phi * a + Phi

    return (it + 1), Phi  # Return the number of iterations taken to converge


def Part_3():
    # Range of tolerances to test
    tolerances = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    N_iter_list = [] 

    # Run Jacobi relaxation for each tolerance and record iterations
    for tolerance in tolerances:
        n_iter,phi = sor_method(tolerance)
        N_iter_list.append(n_iter)
        print(f'Tolerance: {tolerance}, Iterations: {n_iter}')

    #plotting
    plt.figure(figsize=(10, 6))
    plt.loglog(tolerances, N_iter_list, marker='o')
    plt.title('Number of Iterations vs Tolerance')
    plt.xlabel('Tolerance (ε)')
    plt.ylabel('Number of Iterations (N_iter)')
    plt.grid(True, which="both", ls="--")
    plt.savefig("poission_p_3.pdf")
    plt.show()

if part == 1:
    Part_1()
elif part == 2:
    Part_2()
elif part == 3:
    Part_3()