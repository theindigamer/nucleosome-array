import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pylab as plt
import pickle
import os
import copy
import time

ntimers = 10
timers = np.zeros(ntimers)
timer_descr = {
    0 : "Half of inner loop in metropolisUpdate",
    1 : "Half of half of inner loop in metropolisUpdate",
    2 : "Calls to bendingEnergyDensity",
    3 : "Calls to twistEnergyDensity",
    4 : "Calculating Rs in deltaMatrices via rotationMatrices",
    5 : "Calculating a in deltaMatrices",
    6 : "Calculating b in deltaMatrices",
    7 : "Calculating deltas in deltaMatrices",
    8 : "Total time",
}

dot_ufunc = np.frompyfunc(np.dot, 2, 1)

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
        self.oddMask = np.array([i % 2 == 1 for i in range(self.L - 2)])
        self.evenMask = np.roll(self.oddMask, 1)

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
        start = time.clock()

        if Rs is None:
            Rs = self.rotationMatrices()

        timers[4] += time.clock() - start
        start = time.clock()

        a = np.swapaxes( Rs[:-1], 1, 2 )

        timers[5] += time.clock() - start
        start = time.clock()

        b = Rs[1:]

        timers[6] += time.clock() - start
        start = time.clock()

        deltas = np.array([ np.dot(a[i], b[i]) for i in range(len(a)) ])

        timers[7] += time.clock() - start

        return deltas

    def twistBendAngles(self, Ds=None, squared=True):
        """ Returns the twist and bending angles."""
        if Ds is None:
            Ds = self.deltaMatrices()
        if squared:
            betaSq = 2.0 * (1 - Ds[:, 2, 2])
            GammaSq = 1.0 - Ds[:, 0, 0] - Ds[:, 1, 1] + Ds[:, 2, 2]
            return betaSq, GammaSq
        else:
            beta1 = (Ds[:, 1, 2] - Ds[:, 2, 1]) / 2.0
            beta2 = (Ds[:, 2, 0] - Ds[:, 0, 2]) / 2.0
            Gamma = (Ds[:, 0, 1] - Ds[:, 1, 0]) / 2.0
            return beta1, beta2, Gamma

    def bendingEnergyDensity( self, angles=None, squared=True ):
        """ Returns the bending energy density.
            Enter angles in a tuple( arrays ) format."""
        if angles is None:
            angles = self.twistBendAngles( squared=squared )

        if squared:
            energy_density = self.B * angles[0] / (2.0*self.d)
        else:
            energy_density = self.B * ( angles[0]**2 + angles[1]**2 ) / (2.0*self.d)

        return energy_density, angles

    def bendingEnergy( self, squared=True, bendEnergyDensity=None ):
        """ Returns the total bending energy."""
        if bendEnergyDensity is None:
            bendEnergyDensity, _ = self.bendingEnergyDensity( squared=squared )

        return np.sum( bendEnergyDensity )

    def twistEnergyDensity( self, angles=None, squared=True ):
        """ Returns the twist energy density."""
        if angles is None:
            angles = self.twistBendAngles( squared=squared )
        if squared:
            return self.C * angles[-1] / (2.*self.d)
        else:
            return self.C * angles[-1]**2 / (2.*self.d)

    def twistEnergy( self, squared=True, twistEnergyDensity=None ):
        """ Returns the total twist energy. """
        if twistEnergyDensity is None:
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
        if tangent is None:
            tangent = self.tVector()
        return -force * prefactor * tangent[:-1,2]

    def stretchEnergy( self, force=1.96, stretchEnergyDensity=None ):
        """ Returns the total stretching energy. """
        if stretchEnergyDensity is None:
            stretchEnergyDensity = self.stretchEnergyDensity( force=force )
        return np.sum( stretchEnergyDensity )

    def totalEnergyDensity( self, squared=True, force=1.96 ):
        """ Returns the total energy density."""
        start = time.clock()

        E, angles = self.bendingEnergyDensity( squared=squared )

        timers[2] += time.clock() - start
        start = time.clock()

        E += self.twistEnergyDensity(squared=squared, angles=angles)

        timers[3] += time.clock() - start

        E += self.stretchEnergyDensity( force=force )
        return E

    def totalEnergy( self, squared=True, force=1.96, totalEnergyDensity=None ):
        """ Returns the total energy. """
        if totalEnergyDensity is None:
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
        if t is None:
            t = self.tVector()
        return np.cumsum( t, axis=0 )

    def metropolisUpdate(self, sigma=0.1, squared=True, force=1.96, E0=None):
        """ Updates dnaClass Euler angles using Metropolis algorithm.
        Returns the total energy density. """
        if E0 is None:
            E0 = self.totalEnergyDensity( squared=squared, force=force )

        moves = np.random.normal(loc=0.0, scale=sigma, size=(self.L - 2, 3))
        moves[np.abs(moves) >= 5.0*sigma] = 0

        for i in range(3):
            start = time.clock()

            self.euler[1:-1, i] += moves[:, i] * self.oddMask
            Ef = self.totalEnergyDensity( squared=squared, force=force )
            deltaE = ( Ef - E0 )[:-1] + ( Ef - E0 )[1:]

            timers[1] += time.clock() - start

            reject = 1.0 * self.oddMask
            reject[deltaE <= 0] = 0

            self.euler[1:-1,i] -= moves[:, i] * reject
            E0 = self.totalEnergyDensity( squared=squared, force=force )

            timers[0] += time.clock() - start

            self.euler[1:-1, i] += moves[:, i] * self.evenMask

            Ef = self.totalEnergyDensity( squared=squared, force=force )
            deltaE = ( Ef - E0 )[:-1] + ( Ef - E0 )[1:]

            reject = 1.0 * self.evenMask
            reject[deltaE <= 0] = 0

            self.euler[1:-1,i] -= moves[:, i] * reject
            E0 = self.totalEnergyDensity( squared=squared, force=force )

        return E0

    def mcRelaxation(self, sigma=0.1, squared=True, force=1.96, E0=None, mcSteps=100):
        """ Monte Carlo relaxation using Metropolis algorithm. """
        energyList = []
        xList = []
        if E0 is None:
            E0 = self.totalEnergyDensity( squared=squared, force=force )
        energyList.append( np.sum( E0 ) )
        for i in range(mcSteps):
            E0 = self.metropolisUpdate(sigma, squared, force, E0 )
            energyList.append( np.sum(E0) )
            xList.append( np.sum( self.tVector()[:,2] ) )
        return np.array( energyList ), np.array( xList )

    def torsionProtocol(self, sigma=0.1, squared=True, force=1.96, E0=None, mcSteps=100,
                twists=np.arange( np.pi/2.0 , 30.0 * np.pi, np.pi/2.0 ) ):
        """ Simulate a torsion protocol defined by twists. """
        start = time.clock()

        energyList, extensionList = [], []
        for x in twists:
            self.euler[-1, 2] = x
            energy, extension = self.mcRelaxation(sigma, squared, force, E0, mcSteps )
            energyList.append( energy[-1] )
            extensionList.append( extension[-1] )

        global timers
        timers[8] += time.clock() - start
        timings = {s : timers[i] for (i, s) in timer_descr.items()}
        timers = np.zeros(ntimers)

        return {
            "energy" : energyList,
            "extension" : extensionList,
            "timing" : timings
        }

    def relaxationProtocol(self, sigma=0.1, squared=True, force=1.96, E0=None,
                           mcSteps=1000, nsamples=4):
        """
        Simulate a relaxation for an initial twist profile.

        The DNA should be specified with the required profile already applied.
        """
        start = time.clock()

        angles = []

        nsteps = [mcSteps // nsamples] * nsamples if mcSteps >= 10 else []
        if not mcSteps % nsamples == 0:
            nsteps.append(mcSteps % nsamples)

        for nstep in nsteps:
            _, _ = self.mcRelaxation(sigma, squared, force, E0, nstep)
            angles.append(copy.deepcopy(self.euler))

        global timers
        timers[8] += time.clock() - start
        timings = {s : timers[i] for (i, s) in timer_descr.items()}
        timers = np.zeros(ntimers)

        return {
            "angles" : np.array(angles),
            "tsteps" : np.cumsum(nsteps),
            "timing" : timings
        }
