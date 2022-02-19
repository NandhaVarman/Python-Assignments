#-----------------------------------------
#EE2703 Assignment 3
#EE19B043
#Nandha Varman
#-----------------------------------------

# Importing modules
from pylab import *
import scipy.special as sp
import sys

# PART 1:
# Run generate_data.py program

# PART 2:
try:
    input_data = np.loadtxt("fitting.dat", usecols =range(1,10))
except OSError:
    sys.exit("fitting.dat not found!!")
# Extracting data
data_columns = [[],[],[],[],[],[],[],[],[]]
for i in range(len(input_data)):
    for j in range(len(input_data[0])):
        data_columns[j].append(input_data[i][j])

# PART 3:
t = linspace(0,10,101)
sigma = logspace(-1,-3,9)
# Rounding off(3 places)
sigma = around(sigma,3)

# opening a new plot
figure(0)
# plotting the data for different sigma values
for i in range(len(data_columns)):
    plot(t,data_columns[i],label='$\sigma_{} = {}$'.format(i, sigma[i]))

# PART 4:
# defining a function with same general shape as data
def g(t, A, B):
    return A*sp.jn(2,t) + B*t
A = 1.05
B = -0.105
fitting_fn = g(t, A, B)
# Plotting
plot(t, fitting_fn, label='true value', color='#000000')
xlabel('$t\longrightarrow$')
ylabel('$f(t)+noise\longrightarrow$')
title('Data to be fitted to theory')
legend()
grid()
show()

# PART 5:
# opening a new plot
figure(1)
xlabel('$t\longrightarrow$')
ylabel('$f(t)\longrightarrow$')
title('Data points for $\sigma =$ 0.10 along with exact function')
plot(t, fitting_fn, label='f(t)', color='#000000')
# Errorbar plot
errorbar(t[::5], data_columns[0][::5], 0.1, fmt='ro', label=' Error bar')
legend()
grid()
show()

# PART 6:
# Creating column vector for least-squares estimation
jColumn = sp.jn(2,t)
M = c_[jColumn, t]
p = array([A, B])
# Constructiong matrix out of the column vectors
actual = c_[t,fitting_fn]

# PART 7:
# Calculating the fitting error for different combinations of A and B
A = arange(0,2,0.1)
B = arange(-0.2,0,0.01)
epsilon = zeros((len(A), len(B)))
for i in range(len(A)):
    for j in range(len(B)):
            epsilon[i][j] = mean(square(data_columns[0][:] - g(t[:], A[i], B[j])))

# PART 8:
# opening a new plot
figure(2)
# Contour plot of epsilon with A and B as axes
contour_plot=contour(A,B,epsilon,levels=20)
xlabel("A$\longrightarrow$")
ylabel("B$\longrightarrow$")
title("Contours of $\epsilon_{ij}$")
clabel(contour_plot, inline=1, fontsize=10)
# Annotating the graph with exact location of minima
plot([1.05], [-0.105], 'ro')
grid()
annotate("Exact Location\nof Minima", (1.05, -0.105), xytext=(-50, -40), textcoords="offset points", arrowprops={"arrowstyle": "->"})
show()

# PART 9:
# Least squares estimation
p= lstsq(M,fitting_fn,rcond=None)[0]

# PART 10:
# opening a new plot
figure(3)
perr= zeros((9, 2))
# Least square estimation taking different columns
for k in range(len(data_columns)):
    perr[k], *rest = lstsq(M, data_columns[k], rcond=None)
# Calculating Aerr and Berr for each least square estimation
Aerr = array([square(x[0]-p[0]) for x in perr])
Berr = array([square(x[1]-p[1]) for x in perr])
plot(sigma, Aerr, 'mo--', label='$A_{err}$')
plot(sigma, Berr, 'yo--', label='$B_{err}$')
xlabel("Noise standard deviation$\longrightarrow$")
title("Variation of error with noise")
ylabel("MS error$\longrightarrow$")
legend()
grid()
show()

# PART 11:
# opening a new plot
figure(4)
# Plotting Aerr and Berr vs. sigma in a log-log scale
loglog(sigma, Aerr, 'ro', label="$A_{err}$")
loglog(sigma, Berr, 'bo', label="$B_{err}$")
legend()
errorbar(sigma, Aerr, std(Aerr), fmt='ro')
errorbar(sigma, Berr, std(Berr), fmt='bo')
xlabel("$\sigma_{n}\longrightarrow$")
title("Variation of error with noise")
ylabel("MSerror$\longrightarrow$")
legend(loc='upper right')
grid()
show()