import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

ve = 10.
re = -1000 

tf = (10 - re) /  ve
t = np.linspace(0, tf, 1000)

ke = .001
k = 1.
d = 1.
yp, ym = d/2., -d/2.

po = [re, 0, yp, ym, ve, 0, 0, 0, 0, 0, 0, 0]

# vector of functions is (xe, ye, yu, yd, vxe, vye, vyu, vyd, axe, aye, ayu, ayd)
# vector of initial conditions is (re, 0, yp, ym, ve, 0, 0, 0, 0, 0, 0, 0)

def Fc(x1, y1, x2, y2):
    r = np.sqrt((x1-x1)**2 + (y1-y2)**2)
    return ke / r**2

def f(p, t):
    '''Returns derivative of vector (xe, ye, yu, yd, vxe, vye, vyu, vyd, axe, aye, ayu, ayd)
    '''
    xe, ye, yu, yd, vxe, vye, vyu, vyd, axe, aye, ayu, ayd = p
    
    def Sphi(z):
        return np.sin(np.arctan( (z-ye)/xe ))
    
    def Cphi(z):
        return np.cos(np.arctan( (z-ye)/xe ))
        
    daxedt = - Fc(xe, ye, 0, yu)*Sphi(yu) - Fc(xe, ye, 0, yd)*Sphi(yd)
    dayedt = - Fc(xe, ye, 0, yu)*Cphi(yu) + Fc(xe, ye, 0, yd)*Cphi(yd)
    
    dayudt = -k*(yu - yp) + Fc(xe, ye, 0, yu)*Sphi(yu)
    dayddt = -k*(yd - ym) + Fc(xe, ye, 0, yd)*Sphi(yd)
    
    dpdt = [vxe, vye, vyu, vyd, axe, aye, ayu, ayd, daxedt, dayedt, dayudt, dayddt]

    return dpdt


sol = integrate.odeint(f, po, t, full_output=1)


ims = []

for s in range(1000):
    x = sol[0][s, 0], 0, 0
    y = sol[0][s, 1:4]
    ims.append((plt.scatter(x, y, s=10), ))

im_ani = animation.ArtistAnimation(fig1, ims, interval=50, repeat_delay=3000, blit=True)

plt.show()