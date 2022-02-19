#--------------------------------------------------
# EE2703 Assignment 6: Simulation of a tubelight
# Nandha Varman
# EE19B043
#--------------------------------------------------



# importing necessary modules
import sys
from pylab import *

n = 100                     # Spatial grid size
M = 5                       # Number of electrons injected per turn
Msig = 2                    # Standard deviation of electrons injected per turn
nk = 500                    # Number of turns to simulate
u0 = 5                      # Threshold velocity
p = 0.25                    # Probability that ionisation will occur


if len(sys.argv) > 7:
    print("Error!! Too many command line arguments passed.")
    exit()

# getting parameter values from optional command line arguments
else:
    for i in range(len(sys.argv)):
        if i == 1:
            n = int(sys.argv[1])     
        elif i == 2:
            M = int(sys.argv[2])       
        elif i == 3:
            Msig = int(sys.argv[3])
        elif i == 4:
            nk = int(sys.argv[4])       
        elif i == 5:
            u0 = int(sys.argv[5])
        elif i == 6:
            p = float(sys.argv[6])       
            

# empty arrays to store electrons information
xx = zeros(n * M)           # Electron position
u = zeros(n * M)            # Electron velocity
dx = zeros(n * M)           # Displacement in current turn

# empty lists for intensity, position and velocity
I = []                      # Intensity of emitted light
X = []                      # Electron position
V = []                      # Electron velocity

# getting the locations which are occupied by electrons
ii = where(xx > 0)[0]
# loop simulating tubelight operation
for i in range(nk):
    # updating electrons positions and velocity
    dx[ii] = u[ii] + 0.5
    xx[ii] += dx[ii]
    u[ii] += 1
    
    # setting the parameters of electrons that have hit the anode to zero
    hit_anode = where(xx[ii] > n)[0]
    xx[ii[hit_anode]] = 0
    u[ii[hit_anode]] = 0
    dx[ii[hit_anode]] = 0
    
    # finding the electrons that are ionized
    kk = where(u >= u0)[0]
    ll = where(rand(len(kk)) <= p)[0]
    kl = kk[ll]
    # resetting their velocities after ionizaiton to zero
    u[kl] = 0
    # assigning the position of the ionized electrons randomly between the previous and current xi
    rho = rand(len(kl))
    xx[kl] -= (dx[kl] * rho)
    # adding a photon at the location of electron excitaion
    I.extend(xx[kl].tolist())
    
    # determining the actual number of electrons injected
    m = int(randn() * Msig + M)
    # finding the unused indices
    not_ii = where(xx == 0)
    # finding the number of electrons to fill 
    empty_slots = min(len(not_ii), m)        
    # filling electrons in the unused slots
    xx[not_ii[:empty_slots]] = 1
    u[not_ii[:empty_slots]] = 0
    dx[not_ii[:empty_slots]] = 0
    
    # getting the locations which are occupied by electrons
    ii = where(xx > 0)[0]
    
    # adding location and velocity of electrons to the X and V list
    X.extend(xx.tolist())
    V.extend(u.tolist())


# Plotting the Electron Density Histogram
figure(0)
hist(X, bins = arange(0, 101, 1), rwidth = 0.8, color = 'g')
title('Electron Density')
xlabel('Position$\\rightarrow$')
ylabel('Number of electrons$\\rightarrow$')
show()

# Plotting the Light Intensity Histogram
figure(1)
pop_counts,bins,_ = hist(I, bins = arange(0, 101, 1), rwidth = 1.5, color = 'white', ec = 'red')
title('Light Intensity')
xlabel('Position$\\rightarrow$')
ylabel('Intensity$\\rightarrow$')
show()

# Plotting the Electron Phase Space
figure(2)
plot(X, V, 'bo', markersize = 4)
title("Electron Phase Space")
xlabel('Position$\\rightarrow$')
ylabel('Velocity of electron$\\rightarrow$')
show()


# finding the middle point of the bins
xpos = 0.5*(bins[0:-1] + bins[1:])

# Printing the Intensity table
print("Intensity data:\nxpos\tcount")
for i in range(len(bins) - 1):
    print(" ", xpos[i],"\t ",int(pop_counts[i]),sep="")


