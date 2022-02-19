#--------------------------------------------------
# EE2703 Assignment 4: Fourier Approximations
# Nandha Varman
# EE19B043
#--------------------------------------------------

#importing necessary modules
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

PI = np.pi

# functions for exp(x) and cos(cos(x))
def f1(x):
    return np.exp(x)
def f2(x):
    return np.cos(np.cos(x))


t = np.linspace(-2*PI, 4*PI, 500 )
t = t[0:-1]

# lists for locations and labels for ticks
locs = [-2*PI, -3/2*PI, -PI,-PI/2, 0, PI/2, PI, 3*PI/2, 2*PI, 5*PI/2, 3*PI, 7*PI/2, 4*PI]
labels = ['$-2\pi$', '$-3\pi/2$', '$-\pi$','$-\pi/2$', 0, '$\pi/2$', '$\pi$', '$3\pi/2$', '$2\pi$', '$5\pi/2$', '$3\pi$','$7\pi/2$', '$4\pi$']


# plotting exp(x) as a semilog plot
plt.figure(1)
plt.xticks(locs, labels) 
plt.semilogy(t,f1(t),'k', label = 'True function')
plt.semilogy(t,f1(t%(2*PI)),'r', label = 'Expected')
plt.legend(loc='upper right')
plt.title('$exp(x): semilog plot$')
plt.grid()
plt.show()

# plotting cos(cos(x))
plt.figure(2)
plt.xticks(locs, labels) 
plt.plot(t,f2(t),'k', label = 'True function')
plt.plot(t,f2(t%(2*PI)),'--', label = 'Expected')
plt.legend(loc='upper right')
plt.title('$cos(cos(x))$')
plt.grid()
plt.show()

# defining functions for the integrands
def cos_coeff(x, k, h):
    return h(x)*np.cos(k*x)
def sin_coeff(x, k, h):
    return h(x)*np.sin(k*x)

# function to find the first 51 fourier series coefficients
def fourier_coeff_51(h):
    coeff = np.zeros(51)
    coeff[0] = quad(cos_coeff, 0, 2*PI, args =(0, h))[0]/(2*PI)
    for i in range(1,26):
        coeff[2*i-1] = quad(cos_coeff, 0, 2*PI, args =(i, h))[0]/(PI)
        coeff[2*i] = quad(sin_coeff, 0, 2*PI, args =(i, h))[0]/(PI)
    return  coeff

# arrays holding the first 51 FS coefficients in the form [ a0, a1, b1, a2, b2,..]
exp_coeff = fourier_coeff_51(f1)
coscos_coeff = fourier_coeff_51(f2)

# generating lists for location and labels for the coefficents plot
n_locs = range(51)
n_labels = ['$a_0$']
for i in range(1,26):
    n_labels.append('$a_{'+str(i)+'}$')
    n_labels.append('$b_{'+str(i)+'}$') 

# semi-log plot for FS coefficients of exp(x)
plt.figure(3)
plt.xticks(n_locs,n_labels , rotation = '90')
plt.tick_params(axis='x', labelsize=7)
plt.semilogy(0, np.abs(exp_coeff[0]), 'ro')
plt.semilogy(np.arange(1,51,2), np.abs(exp_coeff[1::2]),'ro', label = 'Cosine coefficients')
plt.semilogy(np.arange(2,51,2), np.abs(exp_coeff[2::2]),'bo', label = 'Sine coefficents')
plt.legend(loc='upper right')
plt.title('$exp(x)$ fourier coefficients:semilog plot')
plt.grid()
plt.show()

# log-log plot for FS coefficients of exp(x)
plt.figure(4)
plt.loglog(0, np.abs(exp_coeff[0]), 'ro')
plt.loglog(np.arange(1,51,2), np.abs(exp_coeff[1::2]),'ro', label = 'Cosine coefficients')
plt.loglog(np.arange(2,51,2), np.abs(exp_coeff[2::2]),'bo', label = 'Sine coefficents')
plt.legend(loc='upper right')
plt.title('$exp(x)$ fourier coefficients:log-log plot')
plt.grid()
plt.show()

# semi-log plot for FS coefficients of cos(cos(x))
plt.figure(5)
plt.xticks(n_locs,n_labels , rotation = '90')
plt.tick_params(axis='x', labelsize=7)
plt.semilogy(0, np.abs(coscos_coeff[0]), 'ro')
plt.semilogy(np.arange(1,51,2), np.abs(coscos_coeff[1::2]),'ro', label = 'Cosine coefficients')
plt.semilogy(np.arange(2,51,2), np.abs(coscos_coeff[2::2]),'bo', label = 'Sine coefficents')
plt.legend(loc='upper right')
plt.title('$cos(cos(x))$ fourier coefficients:semilog plot')
plt.grid()
plt.show()

# log-log plot for FS coefficients of cos(cos(x))
plt.figure(6)
plt.loglog(0, np.abs(coscos_coeff[0]), 'ro')
plt.loglog(np.arange(1,51,2), np.abs(coscos_coeff[1::2]),'ro', label = 'Cosine coefficients')
plt.loglog(np.arange(2,51,2), np.abs(coscos_coeff[2::2]),'bo', label = 'Sine coefficents')
plt.legend(loc='upper right')
plt.title('$cos(cos(x))$ fourier coefficients:log-log plot')
plt.grid()
plt.show()

# function to find the first 51 coefficients by least squares estimation
def Lstsq_FS_coeff(f):
    x=np.linspace(0,2*PI,401)
    x=x[:-1] # drop last term to have a proper periodic integral
    b=f(x) # f has been written to take a vector
    A=np.zeros((400,51)) # allocate space for A
    A[:,0]=1 # col 1 is all ones
    for k in range(1,26):
        A[:,2*k-1]=np.cos(k*x) # cos(kx) column
        A[:,2*k]=np.sin(k*x) # sin(kx) column
    #endfor
    ls_coeffs=np.linalg.lstsq(A,b,rcond = -1)[0] # the ’[0]’ is to pull out the
    # best fit vector. lstsq returns a list.
    return ls_coeffs

# arrays holding the first 51 FS coefficients obtained by least squares estimation
exp_coeff_lstsq = Lstsq_FS_coeff(f1)
coscos_coeff_lstsq = Lstsq_FS_coeff(f2)

# semi-log comparison plot of the FS coefficents obtained by integrationa and
# Least squares estimation for exp(x)
plt.figure(7)
plt.xticks(n_locs, n_labels, rotation = '90')
plt.tick_params(axis='x', labelsize=7)
plt.semilogy(range(51), np.abs(exp_coeff),'ro', label = 'by Integration')
plt.semilogy(range(51), np.abs(exp_coeff_lstsq),'go', label = 'by Least Squares')
plt.legend(loc='upper right')
plt.title('$exp(x)$ fourier coefficients comparison:semilog plot')
plt.grid()
plt.show()

# semi-log comparison plot of the FS coefficents obtained by integrationa and
# Least squares estimation for exp(x)
plt.figure(8)
plt.xticks(n_locs, n_labels, rotation = '90')
plt.tick_params(axis='x', labelsize=7)
plt.loglog(range(51), np.abs(exp_coeff),'ro', label = 'by Integration')
plt.loglog(range(51), np.abs(exp_coeff_lstsq),'go', label = 'by Least Squares')
plt.legend(loc='upper right')
plt.title('$exp(x)$ fourier coefficients comparison:log-log plot')
plt.grid()
plt.show()

# semi-log comparison plot of the FS coefficents obtained by integrationa and
# Least squares estimation for cos(cos(x))
plt.figure(9)
plt.xticks(n_locs, n_labels, rotation = '90')
plt.tick_params(axis='x', labelsize=7)
plt.semilogy(range(51), np.abs(coscos_coeff),'ro', label = 'by Integration')
plt.semilogy(range(51), np.abs(coscos_coeff_lstsq),'go', label = 'by Least Squares')
plt.legend(loc='upper right')
plt.title('$cos(cos(x))$ fourier coefficients comparison:semilog plot')
plt.grid()
plt.show()

# log-log comparison plot of the FS coefficents obtained by integrationa and
# Least squares estimation for cos(cos(x))
plt.figure(10)
plt.xticks(n_locs, n_labels, rotation = '90')
plt.tick_params(axis='x', labelsize=7)
plt.loglog(range(51), np.abs(coscos_coeff),'ro', label = 'by Integration')
plt.loglog(range(51), np.abs(coscos_coeff_lstsq),'go', label = 'by Least Squares')
plt.legend(loc='upper right')
plt.title('$cos(cos(x))$ fourier coefficients comparison: loglog plot')
plt.grid()
plt.show()

# finding the maximum absolute deviation between the FS coefficients obtained by the two methods
abs_error_exp = [abs(exp_coeff[i]-exp_coeff_lstsq[i]) for i in range(len(exp_coeff))]
abs_error_coscos = [abs(coscos_coeff[i]-coscos_coeff_lstsq[i]) for i in range(len(coscos_coeff))]


print('Largest deviation for exp(x): '+str(max(abs_error_exp)))
print('Largest deviation for cos(cos(x)): '+str(max(abs_error_coscos)))

x = np.linspace(-2*PI, 4*PI, 1001)
x = x[:-1]

A = np.zeros((1000,51))
A[:,0] = 1
for k in range(1,26):
    A[:,2*k-1]=np.cos(k*x)
    A[:,2*k]=np.sin(k*x)

# plot showing reconstruction of exp(x) using the obtained FS coefficients 
plt.figure(11)
plt.xticks(locs, labels) 
plt.plot(x, np.matmul(A, exp_coeff ), 'go', label='Integration')
plt.plot(x, np.matmul(A, exp_coeff_lstsq), 'r', label='Least squares')
plt.title('Reconstruction of $exp(x)$')
plt.legend()
plt.grid()

# plot showing reconstruction of cos(cos(x)) using the obtained FS coefficients
plt.figure(12)
plt.xticks(locs, labels) 
plt.plot(x, np.matmul(A, coscos_coeff ), 'go', label='Integration')
plt.plot(x, np.matmul(A, coscos_coeff_lstsq), 'r', label='Least squares')
plt.title('Reconstruction of $cos(cos(x))$')
plt.legend()
plt.grid()