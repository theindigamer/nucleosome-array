import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pylab as plt
import pickle
import os
import copy

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
        phi = self.euler[:, 0]
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        theta = self.euler[:, 1]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        psi = self.euler[:, 2]
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)

        R = np.zeros(( self.L, 3, 3 ))
        R[:, 0, 0] = cos_phi * cos_psi - cos_theta * sin_phi * sin_psi
        R[:, 1, 0] = -cos_psi * sin_phi - cos_theta * cos_phi * sin_psi
        R[:, 2, 0] = sin_theta * sin_psi
        R[:, 0, 1] = cos_phi * sin_psi + cos_theta * cos_psi * sin_phi
        R[:, 1, 1] = -sin_phi * sin_psi + cos_theta * cos_phi * cos_psi
        R[:, 2, 1] = -cos_psi * sin_theta
        R[:, 0, 2] = sin_theta * sin_phi
        R[:, 1, 2] = cos_phi * sin_theta
        R[:, 2, 2] = cos_theta
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
            return self.B * angles[0] / (2.0*self.d)
        else:
            return self.B * ( angles[0]**2 + angles[1]**2 ) / (2.0*self.d)

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
        ( phi, theta ) = ( self.euler[:,0], self.euler[:,1] )
        t = np.zeros(( self.L, 3 ))
        t[:, 0] = np.sin(theta) * np.sin(phi)
        t[:, 1] = np.cos(phi) * np.sin(theta)
        t[:, 2] = np.cos(theta)
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

    moves = np.random.normal(loc=0.0, scale=sigma, size=(dnaClass.L - 2, 3))
    moves *= ( np.abs(moves) < 5.0*sigma )

    for i in xrange(3):
        dnaClass.euler[1:-1, i] += moves[:, i] * dnaClass.oddMask
        Ef = dnaClass.totalEnergyDensity( squared=squared, force=force )
        deltaE = ( Ef - E0 )[:-1] + ( Ef - E0 )[1:]

        reject = 1.0 * dnaClass.oddMask
        reject[deltaE <= 0] = 0
        #reject *= np.random.rand( dnaClass.L - 2 ) > np.exp( - deltaE )

        dnaClass.euler[1:-1,i] -= moves[:, i] * reject

        E0 = dnaClass.totalEnergyDensity( squared=squared, force=force )

        dnaClass.euler[1:-1, i] += moves[:, i] * dnaClass.evenMask

        Ef = dnaClass.totalEnergyDensity( squared=squared, force=force )
        deltaE = ( Ef - E0 )[:-1] + ( Ef - E0 )[1:]

        reject = 1.0 * dnaClass.evenMask
        reject[deltaE <= 0] = 0
        #reject *= np.random.rand( dnaClass.L - 2 ) > np.exp( - deltaE )

        dnaClass.euler[1:-1,i] -= moves[:, i] * reject
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
