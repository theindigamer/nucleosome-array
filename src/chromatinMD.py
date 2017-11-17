import numpy as np
import scipy as sp
from scipy.integrate import odeint
import matplotlib
import matplotlib.pylab as plt
import pickle
import os
import copy
import datetime
import fast_calc
from sim_utils import AngularDescription

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
    def __init__(self, L=128, B=43.E-9, C=89.E-9, SL=740.E-9, rd=1.2E-9, psiEnd=0.0,
                 thetaEnd=0.0, uniformlyTwisted=False):
        # TODO: Document rd. Is it hydrodynamic radius?
        self.L = L
        self.B = B                # in m·kT
        self.C = C                # in m·kT
        self.d = SL / (1.*self.L) # in m
        self.rd = rd              # in m
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

    def jacobian( self, tangent=None ):
        """Returns the jacobian of the alpha to r transformation.
           Shape: (L, 4, 4).
           alpha (rows) is ordered as {delta, phi, theta, psi}.
           r (columns) is ordered as {x,y,z,psi}.
           * We don't have psi depend on r.
           ** We should check with * numerically."""
        if tangent is None:
            tangent = self.tangent_vectors()
        return fast_calc.md_jacobian(tangent)

    def removeLocalStretch( self, tangent=None ):
        """ Updates r vector to remove local stretch """
        if tangent is None: tangent=self.tangent_vectors()
        t = normalize( tangent, self.d )
        self.r[1:, :3] = np.cumsum( t, axis=0 )[:-1]

def parameters( strandClass, eta=9.22E-4 , T=293.15 ):
    """ Returns cR and cPsi parameters. (See notes) """
    k_B = 1.38E-23
    zeta = 2. * np.pi * eta / np.log( strandClass.B / strandClass.rd )
    lamb = 2. * np.pi * eta * strandClass.rd**2

    return k_B * T / ( strandClass.d * zeta ), k_B * T / ( strandClass.d * lamb )

def rDot( r, time, strandClass, force=4.8E8, inextensible=True, tangent=None,
          jacobian=None, torques=None, params=None, flattened=True ):
    """ Returns dr / dt
        Shape: (L-1, 4)
        Evolves [r_1, ..., r_L-1] components of r. r_0 is immobile.
        Enter r in one-dimensional shape: (4 (L-1),)
        [x1, y1, z1, psi1, ..., x_{L-1}, y_{L-1}, z_{L-1}, psi_{L-1}]
        inextensible is true if the strand does not stretch locally."""
    # NOTE: Force used above is actually F / kT, units: Newton / Joule == m^-1.
    # The value 4.8E8 corresponds to 1.96 pN at T = 293 K.
    strandClass.r = r.reshape(( strandClass.L, 4 ))

    if params is None: params = parameters( strandClass )
    cR, cPsi = params

    if tangent is None: tangent = strandClass.tangent_vectors()

    if jacobian is None: jacobian = strandClass.jacobian( tangent=tangent )
    J = jacobian

    if torques is None:
        dnaAngle = angular( strandClass, tangent=tangent )
        torques = dnaAngle.effectiveTorques()
        # FIXME: direction of force should be taken into account
        # print("{0:.1E}".format(
        #     dnaAngle.total_energy(np.array([0., 0., 1.96E-12]))))
    tau = torques

    # x = 0.0 * tau
    # for i in range(3):
    #     for j in range(3):
    #        x[:,i] += J[:, j, i] * tau[:, j]
    x = np.einsum('...ji,...j', J, tau)

    drdt = np.zeros(( strandClass.L, 4 ))
    drdt[1:,:3] -= cR * ( x[1:,:3] - x[:-1,:3] )
    drdt[-1,:3] += cR * force * tangent[-1] / strandClass.d

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
        day, time = str( datetime.datetime.today() ).split()
        y = day.split('-') + time.split('.')[0].split(':')[:2]
        filename += ''.join(y)
        filename += '-'
    filename += '-'.join( elementList )
    filename += '.' + extension
    return filename

def project( vecA, vecB ):
    """ Returns ( vecA . vecB ) vecB.
        Vectors A and B must have shape (N,3)."""
    AdotB = np.einsum('...j,...j', vecA, vecB)
    vec = vecB.copy()
    for i in range(3):
        vec[:,i] *= AdotB
    return vec

def projectPerp( vecA, vecB ):
    """ Returns vecA - ( vecA . vecB ) vecB.
        Vectors A and B must have shape (N,3)."""
    return vecA - project( vecA, vecB )

def normalize( vector, N=1.0 ):
    """ Returns vector field with norm N.
        Enter vector of shape (len(vector), 3)."""
    norm = np.linalg.norm(vector, axis=1)
    x = N * vector
    for i in range(3):
        x[:,i] /= norm
    return x

# WORKING HERE.
class angular(AngularDescription):
    """ This class gives the angular description of the DNA strand."""
    def __init__( self, strandClass, tangent=None ):
        super().__init__(
            strandClass.L, strandClass.B, strandClass.C, strandClass.d * strandClass.L,
            euler=self.alpha( strandClass, tangent )[...,1:])
        self.RL = self.edgeRotationMatrix( strandClass )

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

    def edgeRotationMatrix( self, strandClass ):
        """ """
        # TODO: Explain why are there no phi terms here.
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

    def derivativeRotationMatrices( self ):
        """ Returns rotation matrices along the DNA string"""
        return fast_calc.md_derivative_rotation_matrices(self.euler)

    def oldEffectiveTorques( self, Rs=None, DRs=None ):
        """ Returns the effective torques per temperature. Shape (L, 4)."""
        # Rotation matrices for the start of the L rods plus 1 at the end
        # describing the tunable boundary condition.
        RLp1 = np.zeros(( self.L+1, 3, 3))
        if Rs is None:
            RLp1[:-1,...] = self.rotation_matrices()
        else:
            RLp1[:-1,...] = Rs
        RLp1[-1,...] = self.RL

        if DRs is None: DRs = self.derivativeRotationMatrices()

        def kronDelta(i1, i2): return (1 if i1 == i2 else 0)

        tau = np.zeros(( self.L, 4))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if k==2:
                        c = -( self.C + 2.0*self.B ) / ( 2.0*self.d )
                    else:
                        c = -self.C / ( 2.0*self.d )
                    tau[0, i] += c * ( RLp1[1,j,k] + kronDelta(j, k) ) * DRs[0,j,k,i]
                    tau[1:,i] += c * ( RLp1[2:,j,k] + RLp1[:-2,j,k] ) * DRs[1:,j,k,i]
        return np.roll( tau, 1, axis=1 )

    def effectiveTorques( self, Rs=None, DRs=None ):
        """ Returns the effective torques per temperature."""
        # Rotation matrices for the start of the L rods plus 1 at the end
        # describing the tunable boundary condition.
        RLp1 = np.zeros((self.L + 1, 3, 3))
        if Rs is None:
            RLp1[:-1,...] = self.rotation_matrices()
        else:
            RLp1[:-1,...] = Rs
        RLp1[-1,...] = self.RL

        if DRs is None: DRs = self.derivativeRotationMatrices()

        return fast_calc.md_effective_torques(
            RLp1, DRs, self.L, self.C, self.B, self.d)

    def deltaMatrices( self, Rs=None ):
        """ Returns delta matrices. """
        # TODO: Explain why the first dimension has size L + 1 instead of L.
        # There are L rods, so I think there should be L - 1 delta matrices.
        RLp1 = np.zeros(( self.L+1, 3, 3))
        if Rs is None:
            RLp1[:-1,...] = self.rotation_matrices()
        else:
            RLp1[:-1,...] = Rs
        RLp1[-1,...] = self.RL

        a = np.swapaxes( RLp1[:-1], 1, 2 )
        b = RLp1[1:]
        return a @ b

    def stretch_energy_density(self, force, tangents=None):
        """Computes stretching energy for all but the last rod.

        Computes -F·t_i for i in [0, ..., L-1).

        Args:
            force (Array[(3,)]): x, y, z components of applied force in Newtons.
            tangents (Array[(L, 3)]): tangent vectors for each rod in m.

        Returns:
            energy_density (Array[(L-1,)]) in Joules.

        Note:
            Overrides base class method parameters. Does not require an
            explicit temperature value but all components of force are required.
        """
        if tangents is None:
            tangents = self.tangent_vectors()
        energy_density = np.einsum('i,...i', -force, tangents[:-1])
        return energy_density

    def total_energy_density(self, force, tangents=None):
        """Computes total energy for each rod.

        The energy has three pieces:

            E_tot = E_bend + E_twist + E_stretch

        Args:
            force (Array[(3,)]): x, y, z components of applied force in Newtons.
            tangents (Array[(L, 3)]): tangent vectors for each rod in m.

        Returns:
            energy_density (Array[(L-1,)]) in units of Joules.

        Note:
            1. Caller should ensure that B / d and C / d are in Joules.
            2. Overrides base class method parameters. Does not require an explicit
               temperature value but all components of force are required.
        """
        energy_density, twist_bends = self.bend_energy_density()
        energy_density += self.twist_energy_density(twist_bends=twist_bends)[0]
        energy_density += self.stretch_energy_density(
            force, tangents=tangents)
        return energy_density
