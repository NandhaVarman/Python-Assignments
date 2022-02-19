# importing necessary modules
import scipy.signal as sp
import numpy as np
import matplotlib.pyplot as plt

# function to find the time response for given decay rate
def Time_resp(a):
    X_num = np.poly1d([1, a])
    X_den = np.polymul([1,2*a,a**2 +2.25],[1,0,2.25])
    X = sp.lti(X_num, X_den)
    t,x_imp=sp.impulse(X,None,np.linspace(0,50,5000))
    return t,x_imp

# time response for decay rate of 0.5
t, x1_imp = Time_resp(0.5)
plt.figure(1)
# plotting the time response
plt.plot(t, x1_imp, label='decay=0.5')
plt.xlabel(r"$t \to$")
plt.ylabel(r"$x(t) \to$")
plt.title(r'Solution of $\ddot{x} + 2.25x = e^{-0.5t}cos(1.5t)u(t)$')
plt.legend()
plt.grid()
plt.show()

# time response for decay rate of 0.05
t, x2_imp = Time_resp(0.05)
plt.figure(2)
# plotting the time response
plt.plot(t, x2_imp, label='decay=0.05')
plt.xlabel(r"$t \to$")
plt.ylabel(r"$x(t) \to$")
plt.title(r'Solution of $\ddot{x} + 2.25x = e^{-0.05t}cos(1.5t)u(t)$')
plt.legend()
plt.grid()
plt.show()

# function for input signal
def f(t, decay, w):
    return np.cos(w*t)*np.exp(-decay*t)

# transfer function 
H = sp.lti([1], [1, 0, 2.25])
plt.figure(3)
# time vector goin from 0 to 50
t_vec = np.linspace(0, 50, 1000)
# plotting for different cosine frequencies
for w in np.arange(1.4, 1.6, 0.05):
    t, y, _ = sp.lsim(H, f(t_vec, 0.05, w), t_vec)
    plt.plot(t, y, label='$w = {} rad/s$'.format(w))
    plt.legend()
plt.xlabel(r"$t \to$")
plt.ylabel(r"$x(t) \to$")
plt.title(r"Response of LTI system to various frequencies")
plt.grid()
plt.show()

# Laplace transform of x(t)
X = sp.lti([1, 0, 2], [1, 0, 3, 0])
# getting time response
t, x = sp.impulse(X, None, np.linspace(0, 20, 5000))
# Laplace transform of y(t)
Y = sp.lti([2], [1, 0, 3, 0])
# getting time response
t, y = sp.impulse(Y, None, np.linspace(0, 20, 5000))

# plotting x(t) and y(t)
plt.figure(4)
plt.plot(t, y, label=r"$y(t)$")
plt.plot(t, x, label=r"$x(t)$")
plt.xlabel(r"$t \to$")
plt.title(r"$\ddot{x}+(x-y)=0$" "\n" r"$\ddot{y}+2(y-x)=0$" "\n" r"ICs: $x(0)=1,\ \dot{x}(0)=y(0)=\dot{y}(0)=0$",fontsize=7) 
plt.legend()
plt.show()

# component values of given RLC network
R = 100
L = 1e-6
C = 1e-6
# transfer function
TF = sp.lti([1], [L*C, R*C, 1])
# frequency, magnitude in dB and phase in degree are returned
w,S,phi=TF.bode()

# Magnitude plot of the transfer function
plt.figure(5)
plt.semilogx(w, S)
plt.xlabel(r"$\omega \ \to$")
plt.ylabel(r"$\|H(jw)\|\ (in\ dB)$")
plt.title("Magnitude plot of the given RLC network")
plt.show()

# Phase plot of the transfer function
plt.figure(6)
plt.semilogx(w, phi)
plt.xlabel(r"$\omega \ \to$")
plt.ylabel(r"$\angle H(jw)\ (in\ ^o)$")
plt.title("Phase plot of the given RLC network")
plt.show()

# time vector
t_vec = np.linspace(0, 0.1, int(1e6))
# input signal array
vi = np.cos(1e3*t_vec) - np.cos(1e6*t_vec)
# getting vo using sp.lsim
t, vo, _ = sp.lsim(TF, vi, t_vec)

# plot of vo(t)
plt.figure(7)
plt.plot(t, vo)
plt.xlabel(r"$t\ \to$")
plt.ylabel(r"$v_o(t)\ \to$")
plt.title(r"$v_o(t)$" " given $v_i(t)=cos(10^3t)-cos(10^6t)$")
plt.show()

# zoomed in view of vo(t) to show high frequency
plt.figure(8)
plt.plot(t, vo)
# we are zooming in to see the high frequency oscillations
plt.xlim(0.0124, 0.0129)
plt.ylim(0.92, 1.02)
plt.xlabel(r"$t\ \to$")
plt.ylabel(r"$v_o(t)\ \to$")
plt.title('Zoomed in view of $v_o(t)$')
plt.show()

# magnified view of vo(t) for t< 30 micro s
plt.figure(9)
plt.plot(t, vo)
plt.xlim(0, 3e-5)
plt.ylim(0, 0.35)
plt.xlabel(r"$t\ \to$")
plt.ylabel(r"$v_o(t)\ \to$")
plt.title(r"$v_o(t)$ for $t<30\ us$")
plt.show()






