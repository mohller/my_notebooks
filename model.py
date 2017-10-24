import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

ve = 10.
re = -1000

tf = (10 - re) / ve
t = np.linspace(0, tf, 1000)

ke = .01
k = 1
d = 1.
yp, ym = d / 2., -d / 2.

# vector of initial conditions is (re, 0, yp, ym, ve, 0, 0, 0, 0, 0, 0, 0)
po = [re, 0, yp * 1.1, ym*1.1, ve, 0, 0, 0, 0, 0, 0, 0]

# vector of functions is (xe, ye, yu, yd, vxe, vye, vyu, vyd, axe, aye, ayu, ayd)


def Fc(x1, y1, x2, y2):
    r = np.sqrt((x1-x1)**2 + (y1-y2)**2)
    return ke / r**2


def f(p, t):
    '''Returns derivative of vector (xe, ye, yu, yd, vxe, vye, vyu, vyd, axe, aye, ayu, ayd)
    '''
    xe, ye, yu, yd, vxe, vye, vyu, vyd, axe, aye, ayu, ayd = p

    def Sphi(z):
        return np.sin(np.arctan((z - ye) / xe))

    def Cphi(z):
        return np.cos(np.arctan((z - ye) / xe))

    daxedt = - Fc(xe, ye, 0, yu) * Sphi(yu) - Fc(xe, ye, 0, yd) * Sphi(yd)
    dayedt = - Fc(xe, ye, 0, yu) * Cphi(yu) + Fc(xe, ye, 0, yd) * Cphi(yd)

    dayudt = -k * (yu - yp) + Fc(xe, ye, 0, yu) * Sphi(yu)
    dayddt = -k * (yd - ym) + Fc(xe, ye, 0, yd) * Sphi(yd)

    dpdt = [vxe, vye, vyu, vyd, axe, aye, ayu, ayd, daxedt, dayedt, dayudt, dayddt]

    return dpdt


sol = integrate.odeint(f, po, t, full_output=1)

fig1, ax = plt.subplots(1 , 1, figsize=(12, 8))
ax.set_xlim((re, 10))
ax.set_ylim((d, -d))

ims = []

for s in range(1000):
    x = sol[0][s, 0], 0, 0
    y = sol[0][s, 1:4]
    ims.append((plt.scatter(x, y, color='b', s=10), ))

im_ani = animation.ArtistAnimation(fig1, ims, interval=1, repeat_delay=3000)

plt.show()

plt.scatter

# from numpy import sin, cos
# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.integrate as integrate
# import matplotlib.animation as animation

# G = 9.8  # acceleration due to gravity, in m/s^2
# L1 = 1.0  # length of pendulum 1 in m
# L2 = 1.0  # length of pendulum 2 in m
# M1 = 1.0  # mass of pendulum 1 in kg
# M2 = 1.0  # mass of pendulum 2 in kg


# def derivs(state, t):

#     dydx = np.zeros_like(state)
#     dydx[0] = state[1]

#     del_ = state[2] - state[0]
#     den1 = (M1 + M2)*L1 - M2*L1*cos(del_)*cos(del_)
#     dydx[1] = (M2*L1*state[1]*state[1]*sin(del_)*cos(del_) +
#                M2*G*sin(state[2])*cos(del_) +
#                M2*L2*state[3]*state[3]*sin(del_) -
#                (M1 + M2)*G*sin(state[0]))/den1

#     dydx[2] = state[3]

#     den2 = (L2/L1)*den1
#     dydx[3] = (-M2*L2*state[3]*state[3]*sin(del_)*cos(del_) +
#                (M1 + M2)*G*sin(state[0])*cos(del_) -
#                (M1 + M2)*L1*state[1]*state[1]*sin(del_) -
#                (M1 + M2)*G*sin(state[2]))/den2

#     return dydx

# # create a time array from 0..100 sampled at 0.05 second steps
# dt = 0.05
# t = np.arange(0.0, 20, dt)

# # th1 and th2 are the initial angles (degrees)
# # w10 and w20 are the initial angular velocities (degrees per second)
# th1 = 120.0
# w1 = 0.0
# th2 = -10.0
# w2 = 0.0

# # initial state
# state = np.radians([th1, w1, th2, w2])

# # integrate your ODE using scipy.integrate.
# y = integrate.odeint(derivs, state, t)

# x1 = L1*sin(y[:, 0])
# y1 = -L1*cos(y[:, 0])

# x2 = L2*sin(y[:, 2]) + x1
# y2 = -L2*cos(y[:, 2]) + y1

# fig = plt.figure()
# ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
# ax.set_aspect('equal')
# ax.grid()

# line, = ax.plot([], [], 'o-', lw=2)
# time_template = 'time = %.1fs'
# time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


# def init():
#     line.set_data([], [])
#     time_text.set_text('')
#     return line, time_text


# def animate(i):
#     thisx = [0, x1[i], x2[i]]
#     thisy = [0, y1[i], y2[i]]

#     line.set_data(thisx, thisy)
#     time_text.set_text(time_template % (i*dt))
#     return line, time_text

# ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y)),
#                               interval=25, blit=True, init_func=init)

# # ani.save('double_pendulum.mp4', fps=15)
# plt.show()