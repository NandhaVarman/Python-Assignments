# importing necessary modules
import numpy as np
import matplotlib.pyplot as plt

# defining constants
PI = np.pi
mu0 = 4*PI*1e-7

# Question 2:

   # Breaking the volume into a 3x3x1000 mesh, with mesh points separated
   # by 1cm

  
x = np.linspace(-0.01 , 0.01 ,3)
y = np.linspace(-0.01, 0.01,3)
z = np.linspace(0.01, 10,1000)

X,Y,Z = np.meshgrid(x,y,z,indexing = 'ij')


# Question 4:

    # Obtaining the vectors rd_l and dl_l, where l indexes the segments of the loop.

# number of sections
sec = 100       
# radius of the loop
a = 0.1
k = 1/a
# array to store the indices of the loop segments
l = np.array(list(range(sec)))
# angular locations of the segments
phi = (l+0.5)*2*PI/sec
# dl vector
dl = 2*PI*a*(1/sec)*np.array([-np.sin(phi), np.cos(phi)]).T
# rd (r') vector
rd = a*np.array([np.cos(phi), np.sin(phi)]).T

# Question 3:

   # Plotting the currents in the x-y plane as a quiver plot


I = 4*PI*(1/mu0)*np.array([-np.cos(phi)*np.sin(phi), np.cos(phi)*np.cos(phi)]).T
plt.figure(1)
# quiver plot of currents
plt.quiver(rd[:,0],rd[:,1],I[:,0],I[:,1])
plt.scatter(rd[:,0], rd[:,1], s = 5, c = 'r')
plt.xlabel("x $\longrightarrow$")
plt.ylabel("y $\longrightarrow$")
plt.title("Quiver Plot of Currents in the loop Antenna")
plt.grid()
plt.show()

# Question 5,6:

   # Defining function calc(l) to calculate Rijk_l and then extending it to find
   # Aijk_l


def calc(l):
    rl = rd[l]
    # Matix of distances from segment l to r
    Rijk_l = np.sqrt((X-rl[0])**2 + (Y - rl[1])**2 + (Z )**2 )
    # Aijk_l matrix
    Aijk_l = np.cos(l*2*PI/sec)*np.exp(-1j*k*Rijk_l)/Rijk_l
    return Aijk_l

# Question 7:

   # Using the calc(l) function to compute the vector potential
   # Aijk( A_x and A_y)

# We have used a for loop to reduce space complexity. Vectorized code would
# require us to compute multiplication of large matrices. 

Aijk_x = np.zeros(X.shape)
Aijk_y = np.zeros(Y.shape)

for l in range(sec): 
  Aijk_l = calc(l)
  # potential in x direction
  Aijk_x = Aijk_x + Aijk_l*dl[l,0]
  # potential in y direction
  Aijk_y = Aijk_y + Aijk_l*dl[l,1]
  

# Question 8:

   # Compution B along the z axis (B_z)

# we divide by 1e-4 to get the field magnitude in SI units
B_z = (Aijk_y[-1,1,:] - Aijk_x[1,-1,:] - Aijk_y[0,1,:] + Aijk_x[1,0,:])/(4*1e-4)

# Question 9:

   # Plotting the magnetic field

plt.figure(2)
plt.loglog(z,np.abs(B_z),label = '$B_z$')
plt.title('Loglog plot of magnetic field')
plt.xlabel("z $\longrightarrow$")
plt.ylabel("$B_{z}$ $\longrightarrow$")
plt.grid()

# Question 10:

   # Fitting the field B_z to c*z^b
    

M = np.zeros((len(z[100:]),2))
M[:,0] = np.log(z[100:])
M[:,1] = 1
b,logc = np.linalg.lstsq(M, np.log(np.abs(B_z[100:])).T, rcond=None)[0]
print("The values of the parameters fitted to c*b^z are b = ", "{0:.5f}".format(b), " and c = ","{0:.5E}".format(np.exp(logc)),".")

# Plotting the fitted model upon the magnetic field plot
plt.loglog(z, np.exp(logc)*np.power(z,b), label = "$Fitted model$")
plt.legend()
plt.show()




