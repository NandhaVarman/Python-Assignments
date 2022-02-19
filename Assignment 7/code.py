# importing necessary modules
import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt
import sympy as sym

PI = np.pi
s = sym.symbols('s')                # invoking symbolic variable "s"
t = np.linspace(0, 0.1, int(1e6))   # time array


# function to return output of low-pass filter
def lowPass(R1, R2, C1, C2, G, Vi):
    s = sym.symbols('s')
    A = sym.Matrix([[0, 0, 1, -1/G],
                    [-1/(1+s*R2*C2), 1, 0, 0],
                    [0, -G, G, 1],
                    [-1/R1-1/R2-s*C1, 1/R2, 0, s*C1]])
    b = sym.Matrix([0,
                    0,
                    0,
                    -Vi/R1])
    V = A.inv() * b
    return V[3]

# function to extract numerator and denominator coefficients from 
# sympy expression
def sympyToLTI(sym_expr):
    # extracting the numerator and denominator
    num, denom = sym_expr.as_numer_denom()
    # converting numerator and denominator to polynomials of 's'
    num = sym.Poly(num, s)
    denom = sym.Poly(denom, s)
    # extracting the coefficents
    num_coeffs = num.all_coeffs()
    denom_coeffs = denom.all_coeffs()
    # converting the numerator and denominator coefficients to float
    for i in range(len(num_coeffs)):
        x = float(num_coeffs[i])
        num_coeffs[i] = x
    for j in range(len(denom_coeffs)):
        x = float(denom_coeffs[j])
        denom_coeffs[j] = x
    return num_coeffs, denom_coeffs

# Question 1
Vo = lowPass(10000, 10000, 1e-9, 1e-9, 1.586, 1)
# extracting numerator and denominator as a list from Vo
voNum, voDenom = sympyToLTI(Vo)
# getting a rational transfer function using the numerator and denominator
TF1 = sp.lti(voNum, voDenom)

## Bode Plot of Transfer Function
w, mag, phase = sp.bode(TF1, w=np.linspace(1, int(1e6), int(1e6)))
fig1 = plt.figure(1)
fig1.suptitle(r'Bode Plot of Transfer function of lowpass filter')
plt.subplot(211)
plt.semilogx(w, mag)
plt.ylabel(r'$20log(\|H(j\omega)\|)$')
plt.subplot(212)
plt.semilogx(w, phase)
plt.xlabel(r'$\omega \ \to$')
plt.ylabel(r'$\angle H(j\omega)$')


## Calculate step response
# using the "sp.step" funciton to calculate the step response
time, voStep = sp.step(TF1, None, t)
plt.figure(2)
plt.plot(time, voStep)
plt.title(r'Step Response of Lowpass filter')
plt.xlabel(r'$t\ \to$')
plt.ylabel(r'$V_o(t)\ \to$')
plt.xlim(0, 1e-3)
plt.grid(True)


# Question 2
vi = np.heaviside(t, 1)*(np.sin(2e3*PI*t)+np.cos(2e6*PI*t))     # input
# "np.heaviside(t,1)" represents the unit step function

plt.figure(3)
plt.plot(t, vi)
plt.title(r'$V_i(t)=(sin(2x10^3\pi t)+cos(2x10^6\pi t))u(t)$ to Lowpass filter')
plt.xlabel(r'$t\ \to$')
plt.ylabel(r'$V_i(t)\ \to$')
plt.xlim(0, 1e-3)
plt.grid(True)

# getting vOut using the lsim function 
time, vOut, _ = sp.lsim(TF1, vi, t)
plt.figure(4)
plt.plot(time, vOut)
plt.title(r'$V_o(t)$ for $V_i(t)=(sin(2x10^3\pi t)+cos(2x10^6\pi t))u(t)$ for Lowpass filter')
plt.xlabel(r'$t\ \to$')
plt.ylabel(r'$V_o(t)\ \to$')
plt.xlim(0, 1e-3)
plt.grid(True)


# Question 3
# function to return output of high-pass filter
def highPass(R1, R3, C1, C2, G, Vi):
    s = sym.symbols('s')
    A = sym.Matrix([[0, -1, 0, 1/G],
                    [s*C2*R3/(s*C2*R3+1), 0, -1, 0],
                    [0, G, -G, 1],
                    [-s*C2-1/R1-s*C1, 0, s*C2, 1/R1]])

    b = sym.Matrix([0,
                    0,
                    0,
                    -Vi*s*C1])
    return (A.inv()*b)[3]

Vo = highPass(10000, 10000, 1e-9, 1e-9, 1.586, 1)
# extracting numerator and denominator as a list from Vo
voNum, voDenom = sympyToLTI(Vo)
# getting a rational transfer function using the numerator and denominator
TF2 = sp.lti(voNum, voDenom)

## Bode Plot of Transfer Function
w, mag, phase = sp.bode(TF2, w=np.linspace(1, int(1e6), int(1e6)))
fig5 = plt.figure(5)
fig5.suptitle(r'Bode Plot of Transfer function of highpass filter')
plt.subplot(211)
plt.semilogx(w, mag)
plt.ylabel(r'$20log(\|H(j\omega)\|)$')
plt.subplot(212)
plt.semilogx(w, phase)
plt.xlabel(r'$\omega \ \to$')
plt.ylabel(r'$\angle H(j\omega)$')


vi = np.heaviside(t, 1)*(np.sin(2e3*PI*t)+np.cos(2e6*PI*t))     # input

# plotting the input as a function of time
plt.figure(6)
plt.plot(t, vi)
plt.title(r'$V_i(t)=(sin(2x10^3\pi t)+cos(2x10^6\pi t))u(t)$ to Highpass filter')
plt.xlabel(r'$t\ \to$')
plt.ylabel(r'$V_i(t)\ \to$')
plt.xlim(0, 1e-3)
plt.grid(True)


# getting the output using the sp.lsim function
time, vOut, _ = sp.lsim(TF2, vi, t)
# plotting the output
plt.figure(7)
plt.plot(time, vOut)
plt.title(r'$V_o(t)$ for $V_i(t)=(sin(2x10^3\pi t)+cos(2x10^6\pi t))u(t)$ for Highpass filter')
plt.xlabel(r'$t\ \to$')
plt.ylabel(r'$V_o(t)\ \to$')
plt.xlim(0, 1e-3)
plt.grid(True)



# Question 4

# damped low frequency input
viDampedLowFreq = np.heaviside(t, 1)*(np.sin(2*PI*t))*np.exp(-t)
# damped high frequency input
viDampedHighFreq = np.heaviside(t, 1)*(np.sin(2e5*PI*t))*np.exp(-t)

## Output for low frequency damped sinusoid
time, vOutDampedLowFreq, _ = sp.lsim(TF2, viDampedLowFreq, t)
# plotting the output
plt.figure(8)
plt.plot(time, vOutDampedLowFreq)
plt.title(r'$V_o(t)$ for $V_i(t)=sin(2\pi t)e^{-t}u(t)$ for Highpass filter')
plt.xlabel(r'$t\ \to$')
plt.ylabel(r'$V_o(t)\ \to$')
plt.xlim(0, 1e-3)
plt.grid(True)

## Output for high frequency damped sinusoid
time, vOutDampedHighFreq, rest = sp.lsim(TF2, viDampedHighFreq, t)
# plotting the output
plt.figure(9)
plt.plot(time, vOutDampedHighFreq)
plt.title(r'$V_o(t)$ for $V_i(t)=sin(2x10^5\pi t)e^{-t}u(t)$ for Highpass filter')
plt.xlabel(r'$t\ \to$')
plt.ylabel(r'$V_o(t)\ \to$')
plt.xlim(0, 1e-3)
plt.grid(True)


# Question 5
# getting step response using the sp.step function
time, voStep = sp.step(TF2, None, t)
# plotting the step response as a function of time
plt.figure(10)
plt.plot(time, voStep)
plt.title(r'Step Response of Highpass filter')
plt.xlabel(r'$t\ \to$')
plt.ylabel(r'$V_o(t)\ \to$')
plt.xlim(0, 1e-3)
plt.grid(True)
plt.show()