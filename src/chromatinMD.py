import numpy as np
import scipy as sp
from scipy.integrate import odeint
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy import special
from tqdm import tqdm
import matplotlib
import matplotlib.pylab as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import sys
import pickle
import os
import copy
import datetime
import fast_calc
import sim_utils

matplotlib.rcParams.update({'font.size': 20})
plt.rcParams['contour.negative_linestyle'] = 'solid'
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

class strand:
    """ This is the main class in the MD simulation.
        L is the number of segments. (I am using standard 128; Bryan used 740.)
        B and C are bend and twist moduli (divided by kT).
        SL is the strand length.
        psiEnd is the twist at the right edge.
        The principal class attribute is the strand r vector: r = {x,y,z,psi}."""
    def __init__(self, L=128, d=1.E-8, B=43.E-9, C=89.E-9, rd=1.2E-9, psiEnd=0.0,
            thetaEnd=0.0, uniformlyTwisted=False):
        self.L = L
        self.B = B                # in m·kT_room
        self.C = C                # in m·kT_room
        self.d = d      # in m
        self.rd = rd              # hydrodynamic radius in m
        self.psiEnd = psiEnd
        self.thetaEnd = thetaEnd
        self.r = np.zeros(( self.L, 4 ))
        self.r[:,1] = self.d * np.sin(self.thetaEnd) * np.arange(self.L)
        self.r[:,2] = self.d * np.cos(self.thetaEnd) * np.arange(self.L)
        if uniformlyTwisted:
            self.r[:,3] = self.psiEnd * np.arange( self.L ) / self.L

    def tangent_vectors(self):
        """Tangent vectors for each rod (Array[(L, 3)]) scaled with rod length.

        t_n = r_{n+1} - r_n, n in [0, 1, ..., L-1].

        The last element is appropriately adjusted for boundary conditions.
        """
        r = self.r[...,:3]
        tangent = 0.0 * r
        tangent[:-1,...] = r[1:,...] - r[:-1,...]
        tangent[-1,1] = self.d * np.sin(self.thetaEnd)
        tangent[-1,2] = self.d * np.cos(self.thetaEnd)

        return tangent

    def distanceMatrix(self):
        """ Returns Euclidean distances betwen segments.
            Note: I am using the starting position of each segment."""
        return squareform( pdist( self.r[:3] ) )

    def oldJacobian( self, tangent=None ):
        """Returns the jacobian of the alpha to r transformation.
           Shape: (L, 4, 4).
           alpha (rows) is ordered as {delta, phi, theta, psi}.
           r (columns) is ordered as {x,y,z,psi}.
           * We don't have psi depend on r.
           ** We should check with * numerically."""
        if tangent is None: tangent=self.tangent_vectors()
        t = tangent
        D = np.sqrt( t[...,0]**2 + t[...,1]**2 + t[...,2]**2 )
        p = np.sqrt( t[...,0]**2 + t[...,1]**2 + 1.E-16 )

        J = np.zeros(( self.L, 4, 4 ))
        J[..., 0, 0] = -t[:,0] / D
        J[..., 0, 1] = -t[:,1] / D
        J[..., 0, 2] = -t[:,2] / D
        J[..., 1, 0] = -t[:,1] / p**2
        J[..., 1, 1] = t[:,0] / p**2
        J[..., 2, 0] = -t[:,0] * t[:,2] / ( p * D**2 )
        J[..., 2, 1] = -t[:,1] * t[:,2] / ( p * D**2 )
        J[..., 2, 2] = p / D**2

        return J

    def jacobianBV( self, tangent=None ):
        """Returns the jacobian of the alpha to r transformation.
           Shape: (L, 4, 4).
           alpha (rows) is ordered as {delta, phi, theta, psi}.
           r (columns) is ordered as {x,y,z,psi}.
           * We don't have psi depend on r.
           ** We should check with * numerically."""
        if tangent is None: tangent=self.tangent_vectors()
        t = tangent
        D = np.sqrt( t[...,0]**2 + t[...,1]**2 + t[...,2]**2 )
        p = np.sqrt( t[...,0]**2 + t[...,1]**2 + 1.E-16 )

        J = np.zeros(( self.L, 4, 4 ))
        J[..., 0, 0] = -t[:,0] / D
        J[..., 0, 1] = -t[:,1] / D
        J[..., 0, 2] = -t[:,2] / D
        J[..., 1, 0] = -t[:,1] / p**2
        J[..., 1, 1] = t[:,0] / p**2
        J[..., 2, 0] = -t[:,0] * t[:,2] / ( p * D**2 )
        J[..., 2, 1] = -t[:,1] * t[:,2] / ( p * D**2 )
        J[..., 2, 2] = p / D**2
        J[..., 3, 0] = -t[:,1] * t[:,2] / ( p**2 * D )
        J[..., 3, 1] = t[:,0] * t[:,2] / ( p**2 * D )
        J[..., 3, 2] = 0.0

        return J

    def fastJacobian( self, tangent=None ):
        """Returns the jacobian of the alpha to r transformation.
           Shape: (L, 4, 4).
           alpha (rows) is ordered as {delta, phi, theta, psi}.
           r (columns) is ordered as {x,y,z,psi}.
           * We don't have psi depend on r.
           ** We should check with * numerically."""
        if tangent is None:
            tangent = self.tangent_vectors()
        return fast_calc.md_jacobian(tangent)

    jacobian = fastJacobian
#    jacobian = oldJacobian
#    jacobian = jacobianBV

    def removeLocalStretch( self, tangent=None ):
        """ Updates r vector to remove local stretch """
        if tangent is None: tangent=self.tangent_vectors()
        t = normalize( tangent, self.d )
        self.r[1:, :3] = np.cumsum( t, axis=0 )[:-1]

def rDotOld( r, time, strandClass, force=1.96E-12, inextensible=True, eta=9.22E-4,
            tangent=None, jacobian=None, torques=None ):
    """ Returns dr / dt at zero temperature.
        Shape: (L, 4).
        Evolves [r_1, ..., r_L-1] components of r. r_0 doesn't move.
        Enter r in one-dimensional shape: (4 L,)
        [0,0,0,0,x1, y1, z1, psi1, ..., x_{L-1}, y_{L-1}, z_{L-1}, psi_{L-1}]
        inextensible is true if the strand does not stretch locally.
        Use .reshape((L,4)) in the result to have it in (L,4) shape."""
    sc = strandClass
    sc.r = r.reshape(( sc.L, 4 ))

    kT0 = 4.09E-21
    GammaR, GammaPsi = effectiveViscosities( sc, eta )

    if tangent is None: tangent = sc.tangent_vectors()
    if jacobian is None: jacobian = sc.jacobian( tangent=tangent )
    J = jacobian

    if torques is None:
        dnaAngle = angular( sc, tangent=tangent )
        torques = dnaAngle.effectiveTorques()
    tau = torques

    x = np.einsum('...ji,...j', J, tau) # x_i = J_{ji} tau_j

    drdt = np.zeros(( sc.L, 4 ))
    drdt[1:,:3] += - GammaR * kT0 * ( x[1:,:3] - x[:-1,:3] )
    drdt[-1,:3] += GammaR * force * ( tangent[-1] / sc.d )

    drdt[1:,:3] += GammaR * electrostaticForces( sc )[1:,:]

    if inextensible:
        tDot = np.zeros(( sc.L, 3 ))
        tDot[:-1] += drdt[1:,:3] - drdt[:-1,:3]
        tDot = projectPerp( tDot, normalize(tangent) )
        drdt[1:,:3] = np.cumsum( tDot[:-1], axis=0 )

    drdt[1:,3] += -GammaPsi * kT0 * tau[1:,3]

    return drdt.flatten()

def rDot( r, time, strandClass, force=1.96E-12, inextensible=True, eta=9.22E-4,
            tangent=None, jacobian=None, torques=None ):
    """ Returns dr / dt at zero temperature.
        Shape: (L, 4).
        Evolves [r_1, ..., r_L-1] components of r. r_0 doesn't move.
        Enter r in one-dimensional shape: (4 L,)
        [0,0,0,0,x1, y1, z1, psi1, ..., x_{L-1}, y_{L-1}, z_{L-1}, psi_{L-1}]
        inextensible is true if the strand does not stretch locally.
        Use .reshape((L,4)) in the result to have it in (L,4) shape."""
    sc = strandClass
    sc.r = r.reshape(( sc.L, 4 ))

    if tangent is None: tangent = sc.tangent_vectors()

    drdt = np.zeros(( sc.L, 4 ))
    drdt[1:,:] += elasticForces( sc, jacobian=jacobian, tangent=tangent, torques=torques)[1:,:]
    drdt[-1,:3] += force * ( tangent[-1] / sc.d )
#    drdt[1:,:3] += electrostaticForces( sc )[1:,:]

    if inextensible:
        tDot = np.zeros(( sc.L, 3 ))
        tDot[:-1] += drdt[1:,:3] - drdt[:-1,:3]
        tDot = projectPerp( tDot, normalize(tangent) )
        drdt[1:,:3] = np.cumsum( tDot[:-1], axis=0 )

    GammaR, GammaPsi = effectiveViscosities( sc, eta )
    drdt[:,:3] *= GammaR
    drdt[:,3] *= GammaPsi

    return drdt.flatten()

def elasticForces( strandClass, jacobian=None, tangent=None, torques=None ):
    """ """
    sc = strandClass
    kT0 = 4.09E-21
    if tangent is None: tangent = sc.tangent_vectors()
    if jacobian is None: jacobian = sc.jacobian( tangent=tangent )
    if torques is None: torques = angular(sc, tangent=tangent ).effectiveTorques()

    x = np.einsum('...ji,...j', jacobian, torques)

    ef = np.zeros((sc.L, 4))
    ef[1:,:3] = - kT0 * ( x[1:,:3] - x[:-1,:3] )
    ef[1:,3] = - kT0 * torques[1:,3]

    return ef

def electrostaticForces( strandClass, lambD=0.8E-9, nu=8.4E9, T=293.15 ):
    """ lambD = 0.8E-9 is the Debye length at 0.14M NaCl solution. """
    kT = 1.38E-23 * T
    LB = 0.7E-9 # e^2 / ( epsilon k T_room )
    sc = strandClass
    r = 1. * sc.r[:,:3]
    rdiff = np.zeros((sc.L, sc.L, 3))
    for i in range( sc.L ):
        for j in range(i+1, sc.L):
            rdiff[i,j,:] = r[i,:] - r[j,:]
    rdiff -= np.swapaxes( rdiff, 0, 1 )
    rdiffNorm = np.linalg.norm( rdiff, axis=2 ) + 1.E-16
    aux = special.k1( rdiffNorm / lambD ) / rdiffNorm
    force = np.zeros(( sc.L, 3))
    for i in range(3):
        for j in range(sc.L):
            force[:,i] += aux[:,j] * rdiff[:,j,i]
    return kT * nu**2 * LB * sc.d * force / lambD

def simpleEuler(dxdt, x0, times, args ):
    """ Simplest implementation of an euler integration algorithm.
        To be used similarly to odeint. """
    sc = args[0]
    dt = times[1:] - times[:-1]

    st = x0 # Solution at time t.
    sol = [ st ]

    for i in range( len(dt) ):
        st += dxdt( st, times[i], *args ) * dt[i] 

        sc.r = st.reshape(( sc.L, 4 ))
        sc.removeLocalStretch()
        st = sc.r.flatten()

        sol.append( st )
    return np.array( sol )

def eulerMaruyamaOS(dxdt, x0, times, args, T=293.15, eta=9.22E-4 ):
    """ Simplest implementation of an euler-maruyama integration algorithm.
        To be used similarly to odeint. """
    sc = args[0]
    dt = times[1:] - times[:-1]

    kT = 1.38E-23 * T
    GammaR, GammaPsi = effectiveViscosities( sc, eta )

    dW = np.random.randn( sc.L, 4 , len(dt) )
    dW[0,...] *= 0.
    for i in range(len(dt)):
        dW[:,:3,i] *= np.sqrt( 1. - np.exp( -2.* GammaR * kT * dt[i] / sc.d**2 ) ) * sc.d
        dW[:,3,i] *= np.sqrt( 1. - np.exp( -2.* GammaPsi * kT * dt[i] ) )

    st = x0 # Solution at time t.
    sol = [ st ]

    for i in range( len(dt) ):
        half_times = np.arange(0., dt[i]/2., dt[i]/20.0)
        st = odeint( dxdt, st, half_times, args=args)[-1]

        st += dW[...,i].flatten()
        sc.r = st.reshape(( sc.L, 4 ))
        sc.removeLocalStretch()
        st = sc.r.flatten()

        st = odeint( dxdt, st, half_times, args=args)[-1]

        sol.append( st )
    return np.array( sol )

def effectiveViscosities( strandClass, eta=9.22E-4):
    """ Returns 1/ (d zeta) and 1 / (d lambda)"""
    sc = strandClass
    return ( ( np.log( sc.B / sc.rd ) / ( 2. * np.pi * eta ) ) / sc.d,
            ( 1. / ( 2. * np.pi * eta * sc.rd**2 ) ) / sc.d )
 
def makeFilename( directory, elementList, extension, dated=True ):
    """ Creates a filename in directory with a list of string elements.
        If dated, add date and time at the beginning of the filename.
        Examples of elementList: ['Brf','vs','A','indexJ'].
        Examples of extension: 'png' (without '.')."""
    filename = directory
    if dated:
        day, time = str( datetime.datetime.today() ).split()
        y = day.split('-') + time.split('.')[0].split(':')[:2]
        filename += ''.join(y)
        filename += '-'
    filename += '-'.join( elementList )
    filename += '.' + extension
    return filename

def project( vecA, vecB ):
    u""" Returns ( vecA · vecB ) vecB.

    Vectors A and B must have shape (N,3)."""
    AdotB = np.einsum('...j,...j', vecA, vecB)
    vec = vecB.copy()
    for i in range(3):
        vec[:, i] *= AdotB
    return vec

def projectPerp( vecA, vecB ):
    u"""Returns vecA - ( vecA · vecB ) vecB.

    Vectors A and B must have shape (N,3)."""
    return vecA - project( vecA, vecB )

def normalize( vector, N=1.0 ):
    """ Returns vector field with norm N.
        Enter vector of shape (len(vector), 3)."""
    norm = np.linalg.norm(vector, axis=1)
    x = N * vector
    for i in range(3):
        x[:, i] /= norm
    return x

# WORKING HERE.
class angular(sim_utils.AngularDescription):
    """ This class gives the angular description of the DNA strand."""
    def __init__( self, strandClass, tangent=None ):
        temperature = sim_utils.Environment.ROOM_TEMP
        super().__init__(
            strandClass.L, strandClass.B, strandClass.C, temperature,
            strandClass.d * strandClass.L,
            euler=self.alpha( strandClass, tangent )[...,1:],
            end=np.array([0., strandClass.thetaEnd, strandClass.psiEnd]))
        self.RStart = self.startRotationMatrix( strandClass )
        self.REnd = self.endRotationMatrix( strandClass )

    def alpha( self, strandClass, tangent=None ):
        """ alpha = {Delta, phi, theta, psi}."""
        if tangent is None:
            t = strandClass.tangent_vectors()
        else:
            t = tangent

        t_norm = np.linalg.norm(t, axis=1)
        r = strandClass.r

        x = 0.0 * r
        x[:, 0] = t_norm
        x[:, 1] = np.arctan2(t[:, 0], t[:, 1])
        x[:, 2] = np.arccos(t[:, 2] / t_norm)
        x[:, 3] = r[:, 3]

        return x

    def startRotationMatrix( self, strandClass ):
        """ """
        # Question: Why phi == psi == 0 here?
        theta = strandClass.thetaEnd
        R = np.zeros(( 3, 3 ))
        R[0, 0] = 1.0
        R[0, 1] = 0.0
        R[1, 0] = 0.0
        R[1, 1] = np.cos(theta)
        R[1, 2] = np.sin(theta)
        R[2, 0] = 0.0
        R[2, 1] = -np.sin(theta)
        R[2, 2] = np.cos(theta)
        return R

    def endRotationMatrix( self, strandClass ):
        """ """
        # Question: Why phi == 0 here?
        theta = strandClass.thetaEnd
        psi = strandClass.psiEnd
        R = np.zeros(( 3, 3 ))
        R[0, 0] = np.cos(psi)
        R[0, 1] = np.sin(psi)
        R[1, 0] = -np.cos(theta) * np.sin(psi)
        R[1, 1] = np.cos(theta) * np.cos(psi)
        R[1, 2] = np.sin(theta)
        R[2, 0] = np.sin(theta) * np.sin(psi)
        R[2, 1] = -np.sin(theta) * np.cos(psi)
        R[2, 2] = np.cos(theta)
        return R

    def oldDerivativeRotationMatrices( self ):
        """ Returns rotation matrices along the DNA string"""
        sinPhi = np.sin( self.euler[..., 0])
        cosPhi = np.cos( self.euler[..., 0])
        sinTheta = np.sin( self.euler[..., 1] )
        cosTheta = np.cos( self.euler[..., 1] )
        sinPsi = np.sin( self.euler[..., 2] )
        cosPsi = np.cos( self.euler[..., 2])

        DR = np.zeros(( self.L, 3, 3, 3 ))
        DR[..., 0, 0, 0] = -sinPhi * cosPsi - cosPhi * cosTheta *sinPsi
        DR[..., 1, 0, 0] = -cosPhi * cosPsi + sinPhi * cosTheta * sinPsi
        DR[..., 0, 1, 0] = cosPhi * cosTheta * cosPsi - sinPhi * sinPsi
        DR[..., 1, 1, 0] = -sinPhi * cosTheta * cosPsi - cosPhi * sinPsi
        DR[..., 0, 2, 0] = cosPhi * sinTheta
        DR[..., 1, 2, 0] = -sinPhi * sinTheta
        DR[..., 0, 0, 1] = sinPhi * sinTheta * sinPsi
        DR[..., 1, 0, 1] = cosPhi * sinTheta * sinPsi
        DR[..., 2, 0, 1] = cosTheta * sinPsi
        DR[..., 0, 1, 1] = -sinPhi * sinTheta * cosPsi
        DR[..., 1, 1, 1] = -cosPhi * sinTheta * cosPsi
        DR[..., 2, 1, 1] = -cosTheta * cosPsi
        DR[..., 0, 2, 1] = sinPhi * cosTheta
        DR[..., 1, 2, 1] = cosPhi * cosTheta
        DR[..., 2, 2, 1] = -sinTheta
        DR[..., 0, 0, 2] = -sinPhi * cosTheta * cosPsi - cosPhi * sinPsi
        DR[..., 1, 0, 2] = -cosPhi * cosTheta * cosPsi + sinPhi * sinPsi
        DR[..., 2, 0, 2] = sinTheta * cosPsi
        DR[..., 0, 1, 2] = cosPhi * cosPsi - sinPhi * cosTheta * sinPsi
        DR[..., 1, 1, 2] = -sinPhi * cosPsi - cosPhi * cosTheta * sinPsi
        DR[..., 2, 1, 2] = sinTheta * sinPsi
        return DR

    def rotationMatrices(self, *args, **kwargs):
        return self.rotation_matrices(*args, **kwargs)

    def derivativeRotationMatrices( self ):
        """ Returns rotation matrices along the DNA string"""
        return fast_calc.md_derivative_rotation_matrices(self.euler)

    def effectiveTorquesAV( self, Rs=None, DRs=None ):
        """ Returns the effective torques per temperature. Shape (L, 4)."""
        if Rs is None:
            Rs = self.rotationMatrices()[:-1]

        if DRs is None: DRs = self.derivativeRotationMatrices()

        tau = np.zeros(( self.L, 4))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if k==2:
                        c = -( self.C + 2.0*self.B ) / ( 2.0*self.d )
                    else:
                        c = -self.C / ( 2.0*self.d )

                    tau[0, i] += c * ( self.RStart[j,k] + Rs[1,j,k] ) * DRs[0,j,k,i]
                    tau[-1,i] += c * ( Rs[-2,j,k] + self.REnd[j,k] ) * DRs[-1,j,k,i]
                    tau[1:-1,i] += c * ( Rs[:-2,j,k] + Rs[2:,j,k] ) * DRs[1:-1,j,k,i]
        return np.roll( tau, 1, axis=1 ) #THis rolling here is ugly. I should fix this sometime.

    effectiveTorques = effectiveTorquesAV

    def effectiveTorquesBV( self, Rs=None, DRs=None ):
        """ Returns the effective torques per temperature."""
        if Rs is None:
            Rs = self.rotationMatrices()

        if DRs is None: DRs = self.derivativeRotationMatrices()

        return fast_calc.md_effective_torques(
            self.RStart, Rs, self.REnd, DRs, self.L, self.C, self.B, self.d)
