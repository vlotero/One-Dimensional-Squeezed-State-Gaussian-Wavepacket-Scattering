import math as m
import cmath as cm
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt

#This program models the quantum tunneling of a Gaussian wavepacket encountering a step potential

#Declaring a function that solves the tridiagonal elements of the Crank-Nicolson equation
#============================================================================
def LU(ldiag, mdiag, udiag, d, n):
    if (mdiag[1] == 0): print("b = 0; Main diagonal vanishes"); return
    for i in range(2,n+1):                                    
        ldiag[i] /= mdiag[i-1]
        mdiag[i] -= ldiag[i]*udiag[i-1]
        if (mdiag[i] == 0): print("Main diagonal vanishes"); return
        d[i] -= ldiag[i]*d[i-1]

    d[n] /= mdiag[n]                                       
    for i in range(n-1,0,-1): d[i] = (d[i] - udiag[i]*d[i+1])/mdiag[i]
    #backward substitution running from i=N-1 to i=0
    
# Declaring Wavepacket parameters to be called by the wavepacket function in atomic units:
p0 = 2.71           # initial momentum of the wavepacket
x0 = -40            # initial position of wave packet
beta = 2.67         # wave packet parameter
L = 100             # straight line where the wavepacket moves
hbar = 1            #reduced Planck constant
C = -1              #Correlation coefficient
arb = 1             #arbitrary number
#Other parameters for the system:
hx = 0.005          # spatial step size
tf = 1              # maximum propagation time (5 seconds)
ht = 0.005          # time step

#Defining a function that construct the initial Gaussian wavepacket that is independent on time t
#-------------------------------------------------------------------------------------------------------------#
# The wavepacket from Robinett's study:                                                                       #
# psi(x,0) = 1/sqrt(sqrt(pi)*beta*hbar(1+iC)) * exp[-p0(x-x0)^2/hbar] * exp[-(x-x0)^2/(2(beta*hbar)^2)(1+iC))]#
#-------------------------------------------------------------------------------------------------------------#
def Wavepacket(Psi, x, totspace, x0, beta, p0, hbar, arb, C):
    w = 1/cm.sqrt(cm.sqrt(cm.pi)*beta*complex(arb,C))
    z =-1/(2*beta*beta*complex(arb,C))
    for i in range(1,totspace+1):
        xspace = x[i] - x0
        dy= w * cm.exp(z*xspace*xspace)
        Psi[i] = dy * complex(cm.cos((p0*xspace)/hbar),cm.sin((p0*xspace)/hbar))

#Gaussian wave packet for momentum distribution
def Gaussian_wavepacket(x, beta, x0, p0, C):
    return ((1 / (np.sqrt(np.sqrt(np.pi) * beta * (1 + 1j*C))))  
           * (np.cos(p0*(x - x0) / hbar)+ 1j * np.sin(p0*(x_space - x0) / hbar)) 
           *np.exp((-((x - x0)**2)) / (2*beta*beta * (1 + 1j*C))) )

#Defining a function that calculates |psi(x,0)|^2  and its integral using Trapezoidal method
def PsiSquared(Psi, Psi2, totspace, hx):
    for i in range(1,totspace+1):
        # The modulus square of the wavepacket
        Psi2[i] = abs(Psi[i])*abs(Psi[i])
        #Imposing a condition for very small values of |psi(x,0)|^2
        if (Psi2[i] <= 1e-10): Psi2[i] = 0e0
    #Calculating the norm of the wavepacket by invoking trapezoidal method of integration
    PsiDensity = 0.5e0*(Psi2[1] + Psi2[totspace])    #Taking the average of the wavepackets at the tail ends
    for i in range(2,totspace): PsiDensity += Psi2[i]
    PsiDensity *= hx
    # The probability density of the wavepacket should be unity
    for i in range(1,totspace+1): Psi2[i] /= PsiDensity 
    return PsiDensity

#Constructing the output for the quantum mechanical system:
outstep = 100                         # output every nout steps
fr = int(tf / float(outstep * ht))
totspace = 2*int(L/hx + 0.5) + 1      # the number of spatial nodes (in odd number)
tottime = int(tf/ht + 0.5)            # number of time steps
sp = int(totspace/2)                  #symmetry in the x-axis

#Declaring an empty array for the wavepacket and its probability density
Psi = [complex(0,0)]*(totspace+1)     
Psi2 = [0]*(totspace+1)               # The probability density array

#Declaring an array for the position space of the wavepacket
x = [0]*(totspace+1) 
total_N_point = 40000
x_space = hx * (np.arange(total_N_point) - 0.5 * total_N_point)
for i in range(1,totspace+1):                 
    x[i] = (i-sp-1)*hx
    
#Defining the step potential barrier, which encountered by the wave packet.
#-------------------------------------------------------------------------------
def Potential(x, a, L, V0):                                             
    b = (2/3*L)*0.75*a
    return V0 if x > b else 0e0

#For plotting
def Potential2(x, a, L, V02):                                             
    b = (2/3*L)*0.75*a
    return V02 if x > b else 0e0

#For momentum distribution

def step_condition(x_space):
    x_space = np.asarray(x_space)
    y = np.zeros(x_space.shape)
    y[x_space > 0] = 1.0
    return y

def step_potential_barrier(x_space, barrierheight):
    return barrierheight * (step_condition(x_space))

#-----------------------------------------------------------------------------
#Potential Barrier parameter:
a = 0               # position of  the potential barrier
V0 = 3.67           # height of the step potential barrier
V02 = 0.25          # height of potential barrier (for plotting)

#Array for the potential barrier
V = [0]*(totspace+1)     # array for the real potential
V2 = [0]*(totspace+1)          # potential for plotting
for i in range(1,totspace+1):                 
    V[i] = Potential(x[i],a,L,V0)
    V2[i] = Potential2(x[i],a,L,V02)    #potential for plotting

step_potential = step_potential_barrier(x_space, V0) #For MD
#--------------------------------------------------------------------------------------------------
#Average energy of the wave packet, E = p0^2/2m, in terms of electron volts in Hartree atomic units
#--------------------------------------------------------------------------------------------------
m = 1              #mass of the particle
E = p0*p0/2*m

if E < V0:
    print("The average energy of the wavepacket is less than the height of the barrier potential.")
elif E == V0:
    print("The average energy of the wavepacket is equal to the height of the barrier potential.")
else:
    print("The average energy of the wavepacket is greater than the height of the barrier potential.")
#---------------------------------------------------------------------------------

# Call the fucntion for the defined initial wavepacket
Wavepacket(Psi, x, totspace, x0, beta, p0, hbar, arb, C)
initial_wavepacket = Gaussian_wavepacket(x_space, beta, x0, p0, C)

def CrankNicolson(Psi, V, totspace, hx, ht):
    #array for lower diagional elements
    ldiag = [complex(0,0)]*(totspace+1) 
    #array for the Main diagonal elements
    mdiag = [complex(0,0)]*(totspace+1)   
    #array for the upper diagonal elements
    udiag = [complex(0,0)]*(totspace+1)              

    const = ht/(4e0*hx*hx)                         

    mdiag[1] = 1; udiag[1] = 0          
    Psii = Psi[1]
    for i in range (2,totspace):
        Psi1 = Psii; Psii = Psi[i]            # save initial wave packet values
        W = 2*const + 0.5*ht*V[i]
        ldiag[i] = -const
        mdiag[i] = complex(W,-1)
        udiag[i] = -const
        Psi[i] = const*Psi1 - complex(W,1)*Psii + const*Psi[i+1]  # constant term (column matrix)

    ldiag[totspace] = 0; mdiag[totspace] = 1
                                      
    LU(ldiag,mdiag,udiag,Psi,totspace)            # solution Psi: propagated wave packet

print("THE TIME-EVOLUTION OF WAVE PACKET")
for it in range(1,tottime+1): # time loop
    t = it*ht
    CrankNicolson(Psi, V, totspace, hx, ht) # propagate solution by tridiagonal solver
    
    PsiDensity = PsiSquared(Psi,Psi2,totspace,hx) # probability density
    if (it % outstep == 0 or it == tottime): # output every nout steps
        fname = "wavepacket_{0:4.2f}.txt".format(t)
        out = open(fname,"w")
        for i in range(1,totspace+1):
            out.write("{0:10.5f}{1:10.5f}\n".\
                format(x[i],Psi2[i]))
        out.close
        
        plt.plot(x, Psi2, label = "line 1")
        plt.plot(x, V2, label = "line 2")

        plt.xlim(-100, 100)
        plt.ylim(-0.02, 0.4)
        txt = "Probability Distribution Evolution, t = {time:.2f}"
        plt.title(txt.format(time = t))
        plt.xlabel('x')
        plt.ylabel('$|\Psi(x,t)|^2$')
        plt.grid(False)

        plt.savefig("t = {time:.2f}.jpg".format(time = t))
        plt.show()



animate_evolution = animation.FuncAnimation(plot_fig, evolution, init_func=labels,
                               frames=fr, interval=30, blit=True)
animate_evolution.save('CASE 1 C-4.mp4', fps=15, extra_args=['-vcodec', 'libx264'])
plt.show()
