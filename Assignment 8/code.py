# Importing modules
import numpy as np
import scipy.fftpack as fft
import matplotlib.pyplot as plt

PI = np.pi

# FFT and IFFT on random sequence
x_actual = np.random.rand(128)
X = fft.fft(x_actual)
xComputed = fft.ifft(X)
plt.figure(0)
t = np.linspace(-40, 40, 129)
t = t[:-1]
plt.plot(t, x_actual, color='orange', label='Original $x(t)$', lw=2)
plt.plot(t, np.abs(xComputed), 'c', label='Computed $x(t)$', lw=2)
plt.xlabel(r'$t\ \to$')
plt.grid()
plt.legend()
plt.title('Comparison of actual and computed $x(t)$')
maxError = max(np.abs(xComputed-x_actual))
print(r'Magnitude of maximum error between actual and computed values of the random sequence: ', maxError)     # order of 1e-15


# Spectrum of sin(5t)
x = np.linspace(0, 2*PI, 129)
x = x[:-1]
y = np.sin(5*x)
Y = fft.fftshift(fft.fft(y))/128.0
fig1 = plt.figure(1)
fig1.suptitle(r'FFT of $sin(5t)$')
YMag = np.abs(Y)
YPhase = np.angle(Y)
presentFreqs = np.where(YMag > 1e-3)
w = np.linspace(-40, 40, 129)
w = w[:-1]
plt.subplot(211)
plt.plot(w, YMag, lw=2)
plt.xlim([-10, 10])
plt.ylabel(r'$\|Y\|$')
plt.grid()
plt.subplot(212)
plt.xlim([-10, 10])
plt.ylim([-np.pi, np.pi])
plt.ylabel(r'$\angle Y$')
plt.xlabel(r'$k\ \to$')
plt.plot(w[presentFreqs], YPhase[presentFreqs], 'ro', lw=2)
plt.grid()


# AM Modulation with (1 + 0.1cos(t))cos(10t)
x = np.linspace(-4*PI, 4*PI, 513)
x = x[:-1]
y = (1+0.1*np.cos(x))*np.cos(10*x)
Y = fft.fftshift(fft.fft(y))/512.0
fig2 = plt.figure(2)
fig2.suptitle(r'AM Modulation with $(1+0.1cos(t))cos(10t)$')
YMag = np.abs(Y)
YPhase = np.angle(Y)
presentFreqs = np.where(YMag > 1e-3)
w = np.linspace(-40, 40, 513)
w = w[:-1]
plt.subplot(211)
plt.plot(w, YMag, lw=2)
plt.xlim([-15, 15])
plt.ylabel(r'$\|Y\|$')
plt.grid()
plt.subplot(212)
plt.xlim([-15, 15])
plt.ylim([-np.pi, np.pi])
plt.ylabel(r'$\angle Y$')
plt.xlabel(r'$k\ \to$')
plt.plot(w[presentFreqs], YPhase[presentFreqs], 'ro', lw=2)
plt.grid()


# Spectrum of sin^3(t)
x = np.linspace(-4*PI, 4*PI, 513)
x = x[:-1]
y = (np.sin(x))**3
Y = fft.fftshift(fft.fft(y))/512.0
fig3 = plt.figure(3)
fig3.suptitle(r'Spectrum of $sin^3(t)$')
YMag = np.abs(Y)
YPhase = np.angle(Y)
presentFreqs = np.where(YMag > 1e-3)
w = np.linspace(-40, 40, 513)
w = w[:-1]
plt.subplot(211)
plt.plot(w, YMag, lw=2)
plt.xlim([-5, 5])
plt.ylabel(r'$\|Y\|$')
plt.grid()
plt.subplot(212)
plt.plot(w[presentFreqs], YPhase[presentFreqs], 'ro', lw=2)
plt.xlim([-5, 5])
plt.ylim([-np.pi, np.pi])
plt.ylabel(r'$\angle Y$')
plt.xlabel(r'$k\ \to$')
plt.grid()


# Spectrum of cos^3(t)
x = np.linspace(-4*PI, 4*PI, 513)
x = x[:-1]
y = (np.cos(x))**3
Y = fft.fftshift(fft.fft(y))/512.0
fig4 = plt.figure(4)
fig4.suptitle(r'Spectrum of $cos^3(t)$')
YMag = np.abs(Y)
YPhase = np.angle(Y)
presentFreqs = np.where(YMag > 1e-3)
w = np.linspace(-40, 40, 513)
w = w[:-1]
plt.subplot(211)
plt.plot(w, YMag, lw=2)
plt.xlim([-5, 5])
plt.ylabel(r'$\|Y\|$')
plt.grid()
plt.subplot(212)
plt.plot(w[presentFreqs], YPhase[presentFreqs], 'ro', lw=2)
plt.xlim([-5, 5])
plt.ylim([-np.pi, np.pi])
plt.ylabel(r'$\angle Y$')
plt.xlabel(r'$k\ \to$')
plt.grid()


# Spectrum of cos(20t + 5cos(t))
x = np.linspace(-4*PI, 4*PI, 513)
x = x[:-1]
y = np.cos(20*x + 5*np.cos(x))
Y = fft.fftshift(fft.fft(y))/512.0
fig5 = plt.figure(5)
fig5.suptitle(r'Spectrum of $cos(20t + 5cos(t))$')
YMag = np.abs(Y)
YPhase = np.angle(Y)
presentFreqs = np.where(YMag > 1e-3)
w = np.linspace(-40, 40, 513)
w = w[:-1]
plt.subplot(211)
plt.plot(w, YMag, lw=2)
plt.xlim([-50, 50])
plt.ylabel(r'$\|Y\|$')
plt.grid()
plt.subplot(212)
plt.plot(w[presentFreqs], YPhase[presentFreqs], 'ro', lw=2)
plt.xlim([-50, 50])
plt.ylim([-np.pi, np.pi])
plt.ylabel(r'$\angle Y$')
plt.xlabel(r'$k\ \to$')
plt.grid()


# Spectrum of Gaussian:

# Phase and Magnitude of estimated Gaussian Spectrum
                #considering a window from [-8pi, 8pi] with 512 points

t =  np.linspace(-8*PI, 8*PI, 513)
t = t[:-1]
xTrueGaussian = np.exp(-(t**2)/2)
Y = fft.fftshift(fft.fft(fft.ifftshift(xTrueGaussian)))*8/512.0
fig6 = plt.figure(6)
fig6.suptitle(r'Spectrum of Gaussian')
YMag = np.abs(Y)
YPhase = np.angle(Y)
absentFreqs = np.where(YMag < 1e-3)
YPhase[absentFreqs] = 0
w = np.linspace(-40, 40, 513)
w = w[:-1]
plt.subplot(221)
plt.plot(w, YMag, lw=2)
plt.xlim([-10, 10])
plt.ylabel(r'$\|Y\|$')
plt.title("Estimated")
plt.grid()
plt.subplot(223)
plt.plot(w, YPhase, 'ro', lw=2)
plt.xlim([-10, 10])
plt.ylim([-np.pi, np.pi])
plt.ylabel(r'$\angle Y$')
plt.xlabel(r'$k\ \to$')
plt.grid()

# Phase and Magnitude of true Gaussian spectrum
trueY = np.exp(-(w**2)/2)/np.sqrt(2*PI)
trueYMag = np.abs(trueY)
trueYPhase = np.angle(trueY)
plt.subplot(222)
plt.plot(w, trueYMag)
plt.xlim([-10, 10])
plt.title("True")
plt.grid()
plt.subplot(224)
plt.plot(w, trueYPhase, 'ro')
plt.xlim([-10, 10])
plt.ylim([-np.pi, np.pi])
plt.xlabel(r'$k\ \to$')
plt.grid()

meanError = np.mean(np.abs(trueY - Y))
print(r'Magnitude of mean error between actual and computed values of the Gaussian: ', meanError)     #


plt.show()
