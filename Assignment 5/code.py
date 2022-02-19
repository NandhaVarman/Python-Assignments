# importing necessary modules
from pylab import *
import mpl_toolkits.mplot3d.axes3d as p3
from sys import argv


Nx= 30     # size along x
Ny= 30     # size along y
radius= 8   # radius of central lead
Niter= 4000 # number of iterations to perform

if(len(argv) == 5):    # extracting from optional command line arguments
    Nx = int(argv[1])
    Ny = int(argv[2])
    radius = int(argv[3])
    Niter = int(argv[4])

phi = zeros((Ny, Nx))   # initializing voltage potential array

y = arange(Ny)  
x = arange(Nx)
X,Y=meshgrid(x,y)       # creating mesh grid

ii = where((X-Nx/2)*(X- Nx/2)+ (Y-Ny/2)*(Y-Ny/2)<= radius*radius) 
phi[ii] = 1.0           # setting voltge of 1V region


figure(1)
contourf(X, Y, phi, cmap=cm.jet)
axes().set_aspect('equal')
colorbar(orientation = 'vertical')
title('Initial contour plot of potential')
xlabel('$x\longrightarrow$')
ylabel('$y\longrightarrow$')
show()

errors = zeros(Niter)           # initializing error array
for k in range(Niter):          # updating potential
    oldphi = phi.copy()
    # updating phi array
    phi[1:-1,1:-1]=0.25*(phi[1:-1,0:-2] + phi[1:-1,2:] + phi[0:-2,1:-1] + phi[2:,1:-1])
    # asserting boundaries
    phi[1:-1,0]=phi[1:-1,1]
    phi[1:-1,-1] = phi[1:-1,-2]
    phi[-1,1:-1] = phi[-2,1:-1]
    # setting potential at electrode
    phi[ii] = 1.0
    # error calculation
    errors[k]=(abs(phi-oldphi)).max()
    
figure(2)
semilogy(range(Niter),errors)   # semilog plot of errors
xlabel('Number of iterations$\longrightarrow$')
ylabel('$Error\longrightarrow$')
title('Semi-log plot of error over the iterations')
show()

figure(3)
loglog(range(Niter), errors)    # loglog plot of errors
xlabel('Number of iterations$\longrightarrow$')
ylabel('$Error\longrightarrow$')
title('Log-log plot of error over the iterations')
show()

def lstsq_estimate(iterations, y_vec):   # function to fit the data and return the estimate
    num_iter = len(iterations)
    coeff_matrix = zeros((num_iter,2),dtype = float)
    const_matrix = zeros((num_iter), dtype = float)
    coeff_matrix[:,0] = 1.0
    coeff_matrix[:,1] = iterations
    const_matrix = log(y_vec)
    fit = lstsq(coeff_matrix, const_matrix, rcond = None)[0]
    estimate = coeff_matrix@fit
    return estimate, fit

iterations = array(range(Niter))        # array spanning from 0 to Niter-1


figure(4)
semilogy(range(Niter),errors, 'b', label = 'true plot')
xlabel('Number of iterations$\longrightarrow$')
ylabel('$Error\longrightarrow$')
title('Semi-log plot of error over the iterations')
estimate_all = lstsq_estimate(iterations, errors)[0]
print(lstsq_estimate(iterations, errors)[1])
semilogy(arange(0,Niter,100), exp(estimate_all[::100]), 'or', markersize = 5, label = 'estimate using all error values')
estimate_after500 = lstsq_estimate(array(iterations[501:]), array(errors[501:]))[0]
print(lstsq_estimate(array(iterations[501:]), array(errors[501:]))[1])
semilogy(iterations[501::100], exp(estimate_after500[::100]), 'og', markersize = 5, label = 'estimate using error values after 500 iterations') 
legend(loc = 'upper right')
show()

figure(5)                   # 3D surface plot of potential 
ax=p3.Axes3D(figure(5))          
title('The 3-D surface plot of the potential')
ax.plot_surface(X, Y, phi, rstride=1, cstride=1, cmap=cm.jet)
xlabel('$x\longrightarrow$')
ylabel('$y\longrightarrow$')
show()

figure(6)                   # contour plot of potential
cs = contourf(X ,Y, phi, cmap=cm.jet)
colorbar(orientation = 'vertical')
title('Contour plot of potential')
xlabel('$x\longrightarrow$')
ylabel('$y\longrightarrow$')
show()

figure(7)
Jx = zeros((Nx, Ny))
Jy = zeros((Nx, Ny))
Jx[1:-1, 1:-1] = (phi[1:-1, 0:-2] - phi[1:-1, 2:])*0.5
Jy[1:-1, 1:-1] = (phi[0:-2, 1:-1] - phi[2:, 1:-1])*0.5
h = quiver(X, Y,Jx[::,::],Jy[::,::], scale = 3)
scatter(x[ii[0]], y[ii[1]], color = 'r', s =12)
title("Vector plot of current flow")
xlabel('$x\longrightarrow$')
ylabel('$y\longrightarrow$')
show()