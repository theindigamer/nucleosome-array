import numpy as np
import scipy as sp
from scipy.integrate import odeint
import matplotlib
import matplotlib.pylab as plt
import pickle
import os
import copy
import datetime
matplotlib.rcParams.update({'font.size': 20})
plt.rcParams['contour.negative_linestyle'] = 'solid'
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

class strand( object ):
    """ This is the main class in the MD simulation.
        L is the number of segments. (I am using standard 128; Bryan used 740.)
        B and C are bend and twist moduli (divided by kT).
        SL is the strand length.
        psiEnd is the twist at the right edge.
        The principal class attribute is the strand r vector: r = {x,y,z,psi}."""
    def __init__( self, L=128, B=43.E-9, C=89.E-9, SL=740.E-9, rd=1.2E-9, psiEnd=0.0,
                thetaEnd=0.0 ):
        self.L = L
        self.B = B
        self.C = C
        self.d = SL / (1.*self.L)
        self.rd = rd
        self.r = np.zeros(( self.L, 4 ))
        self.r[:,2] = np.arange( 0.0, self.L*self.d, self.d )
        self.psiEnd = psiEnd
        self.thetaEnd = thetaEnd

    def tangent( self ):
        """ Returns the tangent vector field.
            t_n = r_{n+1}-r_n
            Shape: (L, 3)
            n in [0,1,...,L-1]
            We take t_L to be d e_z."""
        r = self.r[...,:3]
        tangent = 0.0 * r
        tangent[:-1,...] = r[1:,...] - r[:-1,...]
        tangent[-1,1] = self.d * np.sin(self.thetaEnd)
        tangent[-1,2] = self.d * np.cos(self.thetaEnd)
#        tangent[-1,2] = self.d

        return tangent 

    def jacobian( self, tangent=None ):
        """Returns the jacobian of the alpha to r transformation.
           Shape: (L, 4, 4).
           alpha (rows) is ordered as {delta, phi, theta, psi}.
           r (columns) is ordered as {x,y,z,psi}.
           * We don't have psi depend on r.
           ** We should check with * numerically.""" 
        if tangent==None: tangent=self.tangent()
        t = tangent
        D = np.sqrt( t[...,0]**2 + t[...,1]**2 + t[...,2]**2 )
        p = np.sqrt( t[...,0]**2 + t[...,1]**2 ) #+ 1.E-16

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

def parameters( strandClass, eta=9.22E-4 , kT=4.09E-21 ):
    """ Returns cR and cPsi parameters. (See notes) """
    zeta = 2. * np.pi * eta / np.log( strandClass.B / strandClass.rd )
    lamb = 2. * np.pi * eta * strandClass.rd**2

    return kT / ( strandClass.d * zeta ), kT / ( strandClass.d * lamb )

def rDot( r, time, strandClass, tangent=None, jacobian=None, torques=None,
            params=None, force=4.8E8, inextensible=True, flattened=True ):
    """ Returns dr / dt
        Shape: (L-1, 4)
        Evolves [r_1, ..., r_L-1] components of r. r_0 is immobile.
        Enter r in one-dimensional shape: (4 (L-1),)
        [x1, y1, z1, psi1, ..., x_{L-1}, y_{L-1}, z_{L-1}, psi_{L-1}]
        Inexitensible is true if the strand does not stretch locally."""
    strandClass.r = r.reshape(( strandClass.L, 4 ))

    if params==None: params = parameters( strandClass )
    cR, cPsi = params

    if tangent==None: tangent = strandClass.tangent()

    if jacobian==None: jacobian = strandClass.jacobian( tangent=tangent )
    J = jacobian

    if torques==None:
        dnaAngle = angular( strandClass, tangent=tangent )
        torques = dnaAngle.effectiveTorques()
    tau = torques

    x = 0.0 * tau
    for i in xrange(3):
        for j in xrange(3):
            x[:,i] += J[:, j, i] * tau[:, j]

    drdt = np.zeros(( strandClass.L, 4 ))
    for i in xrange(3):
        drdt[1:,i] -= cR * ( x[1:,i] - x[:-1,i] )
#    drdt[-1, 2] += cR * force
    drdt[-1,:3] += cR * force * tangent[-1] / strandClass.d
    #tangent[-1] / np.sqrt(tangent[-1,0]**2+tangent[-1,1]**2+tangent[-1,2]**2)

    if inextensible:
        tDot = np.zeros(( strandClass.L, 3 ))
        tDot[:-1] += drdt[1:,:3] - drdt[:-1,:3]
        tDot = projectPerp( tDot, normalize(tangent) )
        drdt[1:,:3] = np.cumsum( tDot[:-1], axis=0 )

    drdt[1:,3] -= cPsi * tau[1:,3]   

    if flattened:
        return drdt.flatten()
    else:
        return drdt

def makeFilename( directory, elementList, extension, dated=True ):
    """ Creates a filename in directory with a list of string elements.
        If dated, add date and time at the beginning of the filename.
        Examples of elementList: ['Brf','vs','A','indexJ'].
        Examples of extension: 'png' (without '.')."""
    filename = directory
    if dated:
        x = str( datetime.datetime.today() ).split()
        y = x[0].split('-') + x[1].split('.')[0].split(':')[:2]
        filename += ''.join( x[0].split('-') + x[1].split('.')[0].split(':')[:2] )
        filename += '-'
    filename += '-'.join( elementList )
    filename += '.' + extension
    return filename

def projectPerp( vecA, vecB ):
    """ Returns vecA - ( vecA . vecB ) vecB.
        Vectors A and B must have shape (N,3)."""
    AdotB = vecA[:,0]*vecB[:,0] + vecA[:,1]*vecB[:,1] + vecA[:,2]*vecB[:,2]
    vec = 1.0*vecA
    for i in xrange(3):
        vec[:,i] -= AdotB * vecB[:,i]
    return vec

def project( vecA, vecB ):
    """ Returns ( vecA . vecB ) vecB.
        Vectors A and B must have shape (N,3)."""
    AdotB = vecA[:,0]*vecB[:,0] + vecA[:,1]*vecB[:,1] + vecA[:,2]*vecB[:,2]
    vec = 1.0*vecB
    for i in xrange(3):
        vec[:,i] *= AdotB
    return vec

def normalize( vector, N=1.0 ):
    """ Returns vector field with norm N.
        Enter vector of shape (len(vector), 3)."""
    norm = np.sqrt( vector[:,0]**2 + vector[:,1]**2 + vector[:,2]**2 )

    x = N * vector
    for i in xrange(3):
        x[:,i] /= norm
    return x

# WORKING HERE.
class angular( object ):
    """ This class gives the angular description of the DNA strand."""
    def __init__( self, strandClass, tangent=None ):
        self.L = strandClass.L
        self.B = strandClass.B
        self.C = strandClass.C
        self.d = strandClass.d
        self.euler = self.alpha( strandClass, tangent )[...,1:]
        self.RL = self.edgeRotationMatrix( strandClass )
#self.edgeRotationMatrix( strandCLass )

    def alpha( self, strandClass, tangent=None ):
        """ alpha = {Delta, phi, theta, psi}."""
        if tangent==None:
            t = strandClass.tangent()
        else:
            t = tangent

        tabs = np.sqrt( t[:,0]**2 + t[:,1]**2 + t[:,2]**2 )
        r = strandClass.r

        x = 0.0 * r
        x[...,0] = tabs
        x[...,1] = np.arctan( t[:,0] / ( t[:,1] + 1.E-12 ) )
        x[...,2] = np.arccos( t[:,2] / tabs )
        x[...,3] = r[...,3]

        return x

    def edgeRotationMatrix( self, strandClass ):
        """ """
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

    def rotationMatrices( self ):
        """ Returns rotation matrices along the DNA string"""
        phi = self.euler[...,0]
        theta = self.euler[...,1]
        psi = self.euler[...,2]
        R = np.zeros(( self.L, 3, 3 ))
        R[..., 0, 0] = np.cos(phi) * np.cos(psi) - np.cos(theta) * np.sin(phi) * np.sin(psi)
        R[..., 1, 0] = -np.cos(psi) * np.sin(phi) - np.cos(theta) * np.cos(phi) * np.sin(psi)
        R[..., 2, 0] = np.sin(theta) * np.sin(psi)
        R[..., 0, 1] = np.cos(phi) * np.sin(psi) + np.cos(theta) * np.cos(psi) * np.sin(phi)
        R[..., 1, 1] = -np.sin(phi) * np.sin(psi) + np.cos(theta) * np.cos(phi) * np.cos(psi)
        R[..., 2, 1] = -np.cos(psi) * np.sin(theta)
        R[..., 0, 2] = np.sin(theta) * np.sin(phi)
        R[..., 1, 2] = np.cos(phi) * np.sin(theta)
        R[..., 2, 2] = np.cos(theta)
        return R

    def derivativeRotationMatrices( self ):
        """ Returns rotation matrices along the DNA string"""
        phi = self.euler[...,0]
        theta = self.euler[...,1]
        psi = self.euler[...,2]
        DR = np.zeros(( self.L, 3, 3, 3 ))
        DR[..., 0, 0, 0] = -np.sin(phi) * np.cos(psi) - np.cos(phi) * np.cos(theta) * np.sin(psi) 
        DR[..., 1, 0, 0] = -np.cos(phi) * np.cos(psi) + np.sin(phi) * np.cos(theta) * np.sin(psi)
#        DR[..., 2, 0, 0] = 0.0
        DR[..., 0, 1, 0] = np.cos(phi) * np.cos(theta) * np.cos(psi) - np.sin(phi) * np.sin(psi)
        DR[..., 1, 1, 0] = -np.sin(phi) * np.cos(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi)
#        DR[..., 2, 1, 0] = 0.0
        DR[..., 0, 2, 0] = np.cos(phi) * np.sin(theta)
        DR[..., 1, 2, 0] = -np.sin(phi) * np.sin(theta)
#        DR[..., 2, 2, 0] = 0.0
        DR[..., 0, 0, 1] = np.sin(phi) * np.sin(theta) * np.sin(psi)
        DR[..., 1, 0, 1] = np.cos(phi) * np.sin(theta) * np.sin(psi)
        DR[..., 2, 0, 1] = np.cos(theta) * np.sin(psi)
        DR[..., 0, 1, 1] = -np.sin(phi) * np.sin(theta) * np.cos(psi)
        DR[..., 1, 1, 1] = -np.cos(phi) * np.sin(theta) * np.cos(psi)
        DR[..., 2, 1, 1] = -np.cos(theta) * np.cos(psi)
        DR[..., 0, 2, 1] = np.sin(phi) * np.cos(theta)
        DR[..., 1, 2, 1] = np.cos(phi) * np.cos(theta)
        DR[..., 2, 2, 1] = -np.sin(theta)
        DR[..., 0, 0, 2] = -np.sin(phi) * np.cos(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi) 
        DR[..., 1, 0, 2] = -np.cos(phi) * np.cos(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)
        DR[..., 2, 0, 2] = np.sin(theta) * np.cos(psi)
        DR[..., 0, 1, 2] = np.cos(phi) * np.cos(psi) - np.sin(phi) * np.cos(theta) * np.sin(psi)
        DR[..., 1, 1, 2] = -np.sin(phi) * np.cos(psi) - np.cos(phi) * np.cos(theta) * np.sin(psi)
        DR[..., 2, 1, 2] = np.sin(theta) * np.sin(psi)
#        DR[..., 0, 2, 2] = 0.0
#        DR[..., 1, 2, 2] = 0.0
#        DR[..., 2, 2, 2] = 0.0
        return DR

    def effectiveTorques( self, Rs=None, DRs=None ):
        """ Returns the effective torques per temperature."""
        RLp1 = np.zeros(( self.L+1, 3, 3))
        if Rs==None:
            RLp1[:-1,...] = self.rotationMatrices()
        else:
            RLp1[:-1,...] = Rs
        RLp1[-1,...] = self.RL

        if DRs==None: DRs = self.derivativeRotationMatrices()

        tau = np.zeros(( self.L, 4))
        for i in xrange(3):
            for j in xrange(3):
                for k in xrange(3):
                    if k==2:
                        c = -( self.C + 2.0*self.B ) / ( 2.0*self.d ) 
                    else:
                        c = -self.C / ( 2.0*self.d )
                    tau[0,i] += c * ( RLp1[1,j,k] + np.kron(j,k) ) * DRs[0,j,k,i]
                    tau[1:,i] += c * ( RLp1[2:,j,k] + RLp1[:-2,j,k] ) * DRs[1:,j,k,i]
        return np.roll( tau, 1, axis=1 )

    def deltaMatrices( self, Rs=None ):
        """ Returns delta matrices. """
        RLp1 = np.zeros(( self.L+1, 3, 3))
        if Rs==None:
            RLp1[:-1,...] = self.rotationMatrices()
        else:
            RLp1[:-1,...] = Rs
        RLp1[-1,...] = self.RL
        
        a = np.swapaxes( RLp1[:-1], 1, 2 )
        b = RLp1[1:]
        return np.array([ np.dot(a[i], b[i]) for i in xrange(len(a)) ])

#
# MC stuff
#
class nakedDNA( object ):
    """ Creates the naked DNA class.
        Euler angles are ordered as phi, theta and psi.
        Euler angle parametrization... """
    def __init__(self, L=740, B=43.0, C=89.0, d=1.0):
        self.L = L
        self.B = B
        self.C = C
        self.d = d
        self.euler = np.zeros(( self.L, 3))
        self.oddMask = np.array([ i & 0x1 for i in xrange( self.L -2 ) ], dtype=bool)
        self.evenMask = np.roll( self.oddMask, 1 )

    def rotationMatrices( self ):
        """ Returns rotation matrices along the DNA string"""
        phi = self.euler[...,0]
        theta = self.euler[...,1]
        psi = self.euler[...,2]
        R = np.zeros(( self.L, 3, 3 ))
        R[..., 0, 0] = np.cos(phi) * np.cos(psi) - np.cos(theta) * np.sin(phi) * np.sin(psi)
        R[..., 1, 0] = -np.cos(psi) * np.sin(phi) - np.cos(theta) * np.cos(phi) * np.sin(psi)
        R[..., 2, 0] = np.sin(theta) * np.sin(psi)
        R[..., 0, 1] = np.cos(phi) * np.sin(psi) + np.cos(theta) * np.cos(psi) * np.sin(phi)
        R[..., 1, 1] = -np.sin(phi) * np.sin(psi) + np.cos(theta) * np.cos(phi) * np.cos(psi)
        R[..., 2, 1] = -np.cos(psi) * np.sin(theta)
        R[..., 0, 2] = np.sin(theta) * np.sin(phi)
        R[..., 1, 2] = np.cos(phi) * np.sin(theta)
        R[..., 2, 2] = np.cos(theta)
        return R

    def deltaMatrices( self, Rs=None ):
        """ Returns delta matrices. """
        if Rs==None:
            Rs = self.rotationMatrices()
        a = np.swapaxes( Rs[:-1], 1, 2 )
        b = Rs[1:]
        return np.array([ np.dot(a[i], b[i]) for i in xrange(len(a)) ])

    def twistBendAngles( self, Ds=None, squared=True ):
        """ Returns the twist and bending angles."""
        if Ds==None:
            Ds = self.deltaMatrices()
        if squared:
            betaSq = 2.0 * ( 1 - Ds[...,2,2] )
            GammaSq = 1.0 - Ds[...,0,0] - Ds[...,1,1] + Ds[...,2,2]
            return betaSq, GammaSq
        else:
            beta1 = ( Ds[...,1,2] - Ds[...,2,1] ) / 2.0
            beta2 = ( Ds[...,2,0] - Ds[...,0,2] ) / 2.0
            Gamma = ( Ds[...,0,1] - Ds[...,1,0] ) / 2.0
            return beta1, beta2, Gamma

    def bendingEnergyDensity( self, angles=None, squared=True ):
        """ Returns the bending energy density.
            Enter angles in a tuple( arrays ) format."""
        if angles==None:
            angles = self.twistBendAngles( squared=squared )

        if squared:
            return self.B * angles[0] / (2.*self.d)
        else:
            return self.B * ( angles[0]**2 + angles[1]**2 ) / (2.*self.d)

    def bendingEnergy( self, squared=True, bendEnergyDensity=None ):
        """ Returns the total bending energy."""
        if bendEnergyDensity==None:
            bendEnergyDensity = self.bendingEnergyDensity( squared=squared )

        return np.sum( bendEnergyDensity )

    def twistEnergyDensity( self, angles=None, squared=True ):
        """ Returns the twist energy density."""
        if angles==None:
            angles = self.twistBendAngles( squared=squared )
        if squared:
            return self.C * angles[-1] / (2.*self.d)
        else:
            return self.C * angles[-1]**2 / (2.*self.d)

    def twistEnergy( self, squared=True, twistEnergyDensity=None ):
        """ Returns the total twist energy. """
        if twistEnergyDensity==None:
            twistEnergyDensity = self.twistEnergyDensity( squared=squared )

        return np.sum( twistEnergyDensity )

    def stretchEnergyDensity( self, tangent=None, force=1.96 ):
        """ Returns the stretching energy density.
            Enter the force in pN
            Our energy is in unit of kT.
            Es = force * tangent * prefactor
            prefactor = 10E-12 10-9/ (1.38E-23 296.65).
            Change prefactor to change the temperature."""
        prefactor = 0.24
        if tangent==None:
            tangent = self.tVector()
        return -force * prefactor * tangent[:-1,2]

    def stretchEnergy( self, force=1.96, stretchEnergyDensity=None ):
        """ Returns the total stretching energy. """
        if stretchEnergyDensity==None:
            stretchEnergyDensity = self.stretchEnergyDensity( force=force )
        return np.sum( stretchEnergyDensity )

    def totalEnergyDensity( self, squared=True, force=1.96 ):
        """ Returns the total energy density."""
        E = self.bendingEnergyDensity( squared=squared )
        E += self.twistEnergyDensity( squared=squared ) 
        E += self.stretchEnergyDensity( force=force )
        return E 

    def totalEnergy( self, squared=True, force=1.96, totalEnergyDensity=None ):
        """ Returns the total energy. """
        if totalEnergyDensity==None:
            totalEnergyDensity=self.totalEnergyDensity( squared=squared, force=force )
        return np.sum( totalEnergyDensity )

    def tVector( self ):
        """ Returns the tangent vectors. """
        ( phi, theta ) = ( self.euler[...,0], self.euler[...,1] )
        t = np.zeros(( self.L, 3 ))
        t[..., 0] = np.sin(theta) * np.sin(phi)
        t[..., 1] = np.cos(phi) * np.sin(theta)
        t[..., 2] = np.cos(theta)        
        return t

    def rVector( self, t=None ):
        """ Returns the end points of the t-vectors."""
        if t==None:
            t = self.tVector()
        return np.cumsum( t, axis=0 ) 

def metropolisUpdate( dnaClass, sigma=0.1, squared=True, force=1.96, E0=None ):
    """ Update dnaClass Euler angles using Metropolis algorithm.
        Return the total energy density. """
    if E0==None:
        E0 = dnaClass.totalEnergyDensity( squared=squared, force=force )

    moves = sigma * np.random.randn( dnaClass.L-2, 3 )
    moves *= ( np.abs(moves) < 5.0*sigma )

    for i in xrange(3):
        dnaClass.euler[1:-1, i] += moves[..., i] * dnaClass.oddMask
        Ef = dnaClass.totalEnergyDensity( squared=squared, force=force )
        deltaE = ( Ef - E0 )[:-1] + ( Ef - E0 )[1:]

        reject = 1.0 * dnaClass.oddMask
        reject *= deltaE > 0
        #reject *= np.random.rand( dnaClass.L - 2 ) > np.exp( - deltaE )

        dnaClass.euler[1:-1,i] -= moves[..., i] * reject

        E0 = dnaClass.totalEnergyDensity( squared=squared, force=force )
        
        dnaClass.euler[1:-1, i] += moves[..., i] * dnaClass.evenMask

        Ef = dnaClass.totalEnergyDensity( squared=squared, force=force )
        deltaE = ( Ef - E0 )[:-1] + ( Ef - E0 )[1:]

        reject = 1.0 * dnaClass.evenMask
        reject *= deltaE > 0
        #reject *= np.random.rand( dnaClass.L - 2 ) > np.exp( - deltaE )

        dnaClass.euler[1:-1,i] -= moves[..., i] * reject
        E0 = dnaClass.totalEnergyDensity( squared=squared, force=force )   
    return E0

def mcRelaxation( dnaClass, sigma=0.1, squared=True, force=1.96, E0=None, mcSteps=100 ):
    """ Monte Carlo relaxation using Metropolis algorithm. """
    energyList = []
    xList = []
    if E0==None:
        E0 = dnaClass.totalEnergyDensity( squared=squared, force=force )
    energyList.append( np.sum( E0 ) )
    for i in xrange(mcSteps):
        E0 = metropolisUpdate( dnaClass, sigma, squared, force, E0 )
        energyList.append( np.sum(E0) )
        xList.append( np.sum( dnaClass.tVector()[:,2] ) )
    return np.array( energyList ), np.array( xList )

def torsionProtocol( dnaClass, sigma=0.1, squared=True, force=1.96, E0=None, mcSteps=100,
            twists=np.arange( np.pi/2.0 , 30.0 * np.pi, np.pi/2.0 ) ):
    """ Simulate a torsion protocol defined by twists. """
    energyList, extensionList = [], [] 
    for x in twists:
        dnaClass.euler[-1, 2] = x
        energy, extension= mcRelaxation( dnaClass, sigma, squared, force, E0, mcSteps )
        energyList.append( energy[-1] )
        extensionList.append( extension[-1] )
    return energyList, extensionList

