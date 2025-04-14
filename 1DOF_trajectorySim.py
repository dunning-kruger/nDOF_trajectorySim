# 1 Degree of Freedom (1-DOF) Trajectory Simulation
# TODO: add option to import and use thrust/mass motor profiles
# TODO: replace kinemtic equations with numerical integration

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.interpolate

# environmental constants
g0     = 9.80665     # acceleration due to gravity [m/s^2]
gamma  = 1.4         # adiabatic ratio
R      = 287.05287   # gas constant [N⋅m/kg⋅K]
Bs     = 1.458e-6    # dynamic viscocity [N⋅s/m^2]
S      = 110.4       # Sutherland constant [K]
G      = 6.67430e-11 # Newtonian constant of gravitation [m^3/kg⋅s^2]
mEarth = 5.972e24    # mass of the Earth [kg]
rEarth = 6378137     # radius of the Earth [m]

# rocket engine performance parameters 
timeBurnout  = 1.8   # time to engine burnout [sec]
engineThrust = 21525 # total engine thrust [N]

# rocket mass parameters
massInitial = 65.6                    # rocket inert mass [kg]
massInert   = 45.2                    # engine propellant mass [kg]
massProp    = massInitial - massInert # engine propellant mass [kg]

# drag coefficient experimental measurement data
CDmach_lookup  = [0,     0.5,   0.75,  0.9,   0.95,  1.1,   1.2,   1.3 ,  1.5,   2.0,   3.0]
CDCoast_lookup = [0.292, 0.264, 0.277, 0.392, 0.474, 0.557, 0.557, 0.545, 0.492, 0.428, 0.335]
CDBoost_lookup = [0.148, 0.127, 0.129, 0.167, 0.197, 0.245, 0.245, 0.241, 0.227, 0.207, 0.174]

# create drag coefficient functions
fCoast = scipy.interpolate.CubicSpline(CDmach_lookup,CDCoast_lookup)
fBoost = scipy.interpolate.CubicSpline(CDmach_lookup,CDBoost_lookup)

# vehicle structure parameters
dia = .122             # rocket diameter [m]  
A   = np.pi*(dia/2)**2 # rocket frontal area [m^2]

# simulation initialization
pos  = [210]  # initail position [m]
vel  = [.001] # initial velocity [m/s]
acc  = [0]    # initial acceleration [m/s^2]
time = [0]    # initial time [sec]
dt   = .01    # time step [sec]

# function to determine mass relative to engine burnout time
def getMass(time,massInert,massProp):
    if time < timeBurnout: # engine enabled
        mass = massInert + massProp*(1-(time/timeBurnout)) # [kg]
    else: # engine disabled
        mass = massInert # [kg]
    return mass

# function to determine thrust relative to engine burnout time
def getThrust(time,engineThrust):
    if time < timeBurnout: # engine enabled
        thrust = engineThrust # [N]
    else: # engine disabled
        thrust = 0 # [N]
    return thrust

# function to calculate aerodynamic drag
def getDrag(time,fBoost,fCoast,s,rho,velocity,mach):
    # select which drag coefficient function to use
    if time < timeBurnout: # engine enabled
        cD_function = fBoost
    else: # engine disabled
        cD_function = fCoast
    
    # get drag coefficient from the function
    cD = cD_function(mach)

    # calculate aerodynamic drag
    drag = 0.5 * rho * cD * s * velocity**2  # [N]
    return drag

# function to calculate standard atmosphere properties
def atmosphericConditions(altitude):
    # calculate gas properties in Earth's atmosphere
    #                          index    lapse rate   base Temp        base alt             base pressure
    #                          i        Ki (°C/m)    Ti (°K)          Hi (m)               P (Pa)
    atmLayerTable = np.array([[1,       -.0065,      288.15,          0,                   101325],
                              [2,       0,           216.65,          11000,               22632.0400950078],
                              [3,      .001,         216.65,          20000,               5474.87742428105],
                              [4,       .0028,       228.65,          32000,               868.015776620216],
                              [5,       0,           270.65,          47000,               110.90577336731],
                              [6,       -.0028,      270.65,          51000,               66.9385281211797],
                              [7,       -.002,       214.65,          71000,               3.9563921603966],
                              [8,       0,           186.94590831019, 84852.0458449057,    0.373377173762337]])

    # extract layer data from lookup table
    atmLayerK = atmLayerTable[:,1] # layer lapst rate [°K/m]
    atmLayerT = atmLayerTable[:,2] # layer base temp [°K]
    atmLayerH = atmLayerTable[:,3] # layer altitude [m]
    atmLayerP = atmLayerTable[:,4] # layer pressure [Pa]

    # set upper limit of atmospheric model
    altitudeMax = 90000 # [m]

    # troposphere
    if altitude <= atmLayerH[1]:
        i     = 0
        TonTi = 1 + atmLayerK[i] * (altitude - atmLayerH[i]) / atmLayerT[i]
        T     = TonTi * atmLayerT[i]
        PonPi = TonTi ** (-g0 / (atmLayerK[i] * R))
        P     = atmLayerP[i] * PonPi

    # tropopause
    if (altitude <= atmLayerH[2]) & (altitude > atmLayerH[1]):
        i     = 1
        T     = atmLayerT[i]
        PonPi = np.exp(-g0 * (altitude - atmLayerH[i]) / (atmLayerT[i] * R))
        P     = PonPi * atmLayerP[i]

    # stratosphere 1
    if (altitude <= atmLayerH[3]) & (altitude > atmLayerH[2]):
        i     = 2
        TonTi = 1 + atmLayerK[i] * (altitude - atmLayerH[i]) / atmLayerT[i]
        T     = TonTi*atmLayerT[i]
        PonPi = TonTi ** (-g0 / (atmLayerK[i] * R))
        P     = PonPi * atmLayerP[i]

    # stratosphere 2
    if (altitude <= atmLayerH[4]) & (altitude > atmLayerH[3]):
        i     = 3
        TonTi = 1 + atmLayerK[i] * (altitude - atmLayerH[i]) / atmLayerT[i]
        T     = TonTi*atmLayerT[i]
        PonPi = TonTi ** (-g0 / (atmLayerK[i] * R))
        P     = PonPi * atmLayerP[i]

    # stratopause
    if (altitude <= atmLayerH[5]) & (altitude > atmLayerH[4]):
        i     = 4
        T     = atmLayerT[i]
        PonPi = np.exp(-g0 * (altitude - atmLayerH[i]) / (atmLayerT[i] * R))
        P     = PonPi * atmLayerP[i]

    # mesosphere 1
    if (altitude <= atmLayerH[6]) & (altitude > atmLayerH[5]):
        i     = 5
        TonTi = 1 + atmLayerK[i] * (altitude - atmLayerH[i]) / atmLayerT[i]
        T     = TonTi*atmLayerT[i]
        PonPi = TonTi ** (-g0 / (atmLayerK[i] * R))
        P     = PonPi * atmLayerP[i]

    # mesosphere 2
    if (altitude <= atmLayerH[7]) & (altitude > atmLayerH[6]):
        i     = 6
        TonTi = 1 + atmLayerK[i] * (altitude - atmLayerH[i]) / atmLayerT[i]
        T     = TonTi*atmLayerT[i]
        PonPi = TonTi ** (-g0 / (atmLayerK[i] * R))
        P     = PonPi * atmLayerP[i]

    # mesopause
    if (altitude <= altitudeMax) & (altitude > atmLayerH[7]):
        i     = 7
        T     = atmLayerT[i]
        PonPi = np.exp(-g0 * (altitude - atmLayerH[i]) / (atmLayerT[i] * R))
        P     = PonPi * atmLayerP[i]

    # thermosphere
    if altitude > altitudeMax:
        print('WARNING: altitude above atmospheric upper limit')
        T = 0
        P = 0

    # calculate outputs
    rho = P / (T * R)                       # air density [kg/m^3]
    a   = (gamma * R * T) ** 0.5            # acoustic sound [m/s]
    u   = (Bs * (T**1.5) / (T + S)) / rho   # kinematic viscosity [m^2/s]
    return T, P, rho, a, u

# simulation loop
dragArray = [0]
machArray = [0]
while pos[-1] < 20000 and pos[-1] > -.1:
    # calculate current system parameters
    m               = getMass(time[-1],massInert,massProp)
    T, P, rho, a, u = atmosphericConditions(pos[-1])
    
    # get rocket body forces
    thrust = getThrust(time[-1],engineThrust)
    drag   = getDrag(time[-1],fBoost,fCoast,A,rho,vel[-1],vel[-1]/a)
    weight = -G*mEarth*m/((pos[-1] + rEarth)**2) # rocket weight in [N]

    # calculate acceleration using Newton's second law
    F = thrust + weight - drag # net force in [N]
    
    # update velocity and acceleration data vectors
    time      = np.append([time],time[-1]+dt)
    dragArray = np.append([dragArray],drag)
    machArray = np.append([machArray],vel[-1]/a)
    
    # return results
    acc = np.append([acc],F/m)
    vel = np.append([vel],vel[-1]+acc[-1]*dt)
    pos = np.append([pos],pos[-1]+vel[-1]*dt)

# plot results
plt.figure(figsize=(10, 5))
plt.subplot(3,1,1)
plt.plot(time,pos,label="Position",color="r")
plt.title('Apogee of %.2f m at time %.2f sec' % (np.max(pos),time[np.argmax(pos)]))
plt.ylabel("Position (m)")
plt.legend()
plt.grid()

plt.subplot(3,1,2)
plt.plot(time,machArray,label="Relative Mach",color="g")
plt.ylabel("Mach")
plt.legend()
plt.grid()

plt.subplot(3,1,3)
plt.plot(time,dragArray,label="Drag")
plt.xlabel("Time (s)")
plt.ylabel("Drag (N)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
