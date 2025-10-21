import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the Gaussian function for fitting
def gaussian(x, A, mu, sigma):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

# Parameters
L = 100           
nx = 100            
dx = L / (nx - 1)   
D = 2               
dt = 0.1            
nt = 201             
x = np.linspace(0, L, nx)

# Initial density
initial_density = np.zeros(nx)
box_width = 5  
initial_density[nx//2 - box_width//2 : nx//2 + box_width//2] = 1.0 

density = np.zeros((nt, nx))
density[0, :] = initial_density.copy()

for t in range(1, nt):
    for i in range(1, nx - 1):
        density[t, i] = density[t - 1, i] + D * dt / dx**2 * (density[t - 1, i + 1] - 2 * density[t - 1, i] + density[t - 1, i - 1])


times_to_fit = [50, 100, 150, 200]  
fitted_mu = []
fitted_sigma = []

plt.figure(figsize=(12, 10))

for idx, t in enumerate(times_to_fit):
    plt.subplot(3, 2, idx + 1)
    

    norm_density = density[t, :] / np.sum(density[t, :]) 

    p0 = [1.0, L/2, 5] 
    popt, _ = curve_fit(gaussian, x, norm_density, p0=p0)


    A, mu, sigma = popt
    fitted_mu.append(mu)
    fitted_sigma.append(sigma)
    
    # Plot the density profile and fit
    plt.plot(x, norm_density, label='Density Profile', color='blue')
    plt.plot(x, gaussian(x, *popt), label='Fitted', linestyle='--', color='orange')
    plt.title(f'Time Step: {t}')
    plt.xlabel('Position (x)')
    plt.ylabel('Normalized Density')
    plt.legend()
    plt.grid()

plt.tight_layout()
plt.savefig("diffusion.pdf")
plt.show()

# Calculate and print σ(t)
t_values = [dt * t for t in times_to_fit]
expected_sigma = [np.sqrt(2 * D * t) for t in t_values]

# Print fitted sigma and expected sigma values
for t, fitted, expected in zip(t_values, fitted_sigma, expected_sigma):
    print(f'Time: {t:.1f} | Fitted σ: {fitted:.4f} | Expected σ: {expected:.4f}')