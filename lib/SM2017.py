#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

"""
NE2001 for extragalactic work.
"""


from astropy.constants import kpc, c
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
import numpy as np
import os
import logging
from scipy.special import gamma

__author__ = 'Paul Hancock'
__date__ = '2017-02-23'
SFG=0
AGN=1

seconds_per_year = 3600 * 24 * 365.25

class SM(object):
    """
    :param ha_file:
    :param err_file:
    :param nu: freq in Hz
    :param d: distance in kpc
    :param v: in m/s
    :param log:
    """
    def __init__(self, ha_file, err_file=None, nu=185e6, log=None, d=1, v=10e3):

        if log is None:
            logging.basicConfig(format="%(module)s:%(levelname)s %(message)s")
            self.log = logging.getLogger("SM2017")
            self.log.setLevel(logging.DEBUG)
        else:
            self.log=log

        # define some of the constants that we need
        # i'm saving these here to allow for different instances to have different values
        self.nu = nu  # Hz
        self.kpc = kpc.value  # in m
        self.t4 = 0.8  # t/1e4 K
        self.lo = 1e18/(self.kpc*1e-3)  # pc
        self.eps = 1
        self.D = d  # kpc - distance to the screen
        self.c = c.value
        self.beta = 11/3
        self.re = 2.817e-15  # m
        self.rf = np.sqrt(self.c * self.D * self.kpc / (2*np.pi*self.nu))  # Fresnel scale
        self.v = v  # relative velocity of source/observer in m/s
        #self.log.debug("data:{0} err:{1}".format(ha_file,err_file))
        self.file = ha_file
        self.err_file = err_file
        self._load_file()

    def _load_file(self):
        self.hdu = fits.getheader(self.file, ignore_missing_end=True)
        self.wcs = WCS(self.hdu)
        self.data = fits.open(self.file, memmap=True, ignore_missing_end=True)[0].data
        if self.err_file:
            self.err_hdu = fits.getheader(self.err_file,ignore_missing_end=True)
            self.err_wcs = WCS(self.err_hdu)
            self.err_data = fits.open(self.err_file, memmap=True, ignore_missing_end=True)[0].data
        else:
            self.err_hud = self.err_wcs = self.err_data = None

        return

    def get_halpha(self, position):
        """
        Return the Halpha for a given location on the sky.
        :param position: astropy.coordinates.SkyCoord
        :return:
        """
        # The coordinates we request need to be the same as that in the WCS header
        # for the files in this repo, this currently means galactic coordinates.
        x, y = zip(*self.wcs.all_world2pix(zip(position.galactic.l.degree, position.galactic.b.degree), 0))
        x = np.int64(np.floor(x))
        x = np.clip(x, 0, self.hdu['NAXIS1'])
        y = np.int64(np.floor(y))
        y = np.clip(y, 0, self.hdu['NAXIS2'])
        iha = self.data[y, x]
        err_iha = self.err_data[y, x]
        return iha, err_iha

    def get_sm(self, position):
        """
        Return the scintillation measure for a given location on the sky.
        Units are kpc m^{-20/3}
        :param position: astropy.coordinates.SkyCoord
        :return:
        """
        iha, err_iha = self.get_halpha(position)
        # Cordes2002
        sm2 = iha/198 * self.t4**0.9 * self.eps**2/(1+self.eps**2) * self.lo**(-2/3)
        err_sm2= (err_iha/iha)*sm2
        return sm2, err_sm2

    def get_rdiff(self, position):
        """
        Calculate the diffractive scale at the given sky coord
        :param position: astropy.coordinates.SkyCoord
        :return: parameter r_diff in m
        """
        sm2, err_sm2 = self.get_sm(position)
        # ^ units are kpc m^{-20/3}, but we want m^{-17/3} so we have to multiply by kpc below
        # r_diff as per Mcquart & Koay 2013, eq 7a.
        rdiff = (2**(2-self.beta) * (np.pi * self.re**2 * (self.c/self.nu)**2 * self.beta) * sm2 * self.kpc *
                 gamma(-self.beta/2)/gamma(self.beta/2))**(1/(2-self.beta))
        err_rdiff = abs((1/(2-self.beta))*(err_sm2/sm2)*rdiff)
        return rdiff, err_rdiff

    def get_rref(self, position):
        """
        Calculate the refractive scale at the given sky coord
        :param position: astropy.coordinates.SkyCoord
        :return: parameter r_ref in m
        """
        # Narayan 1992 eq 4.2
        rdiff, err_rdiff = self.get_rdiff(position)
        rref = self.rf**2 / rdiff
        err_rref = (err_rdiff / rdiff) * rref
        return rref, err_rref

    def get_xi(self, position):
        """
        calculate the parameter ξ for a given sky coord
        Parameter is dimensionless
        :param position: astropy.coordinates.SkyCoord
        :return: parameter ξ
        """
        rdiff, err_rdiff = self.get_rdiff(position)
        # Narayan 1992, uses r_F/r_diff = \xi without explicitly stating that this is being done
        # Compare Narayan 1992 eq 3.5 with Walker 1998 eq 6
        xi = self.rf / rdiff
        err_xi = (err_rdiff/rdiff)*xi
        return xi, err_xi

    def get_theta(self, position):
        """
        calculate the size of the scattering disk for a given sky coord
        :param position: astropy.coordinates.SkyCoord
        :return: scattering disk in degrees
        """
        # See Narayan 1992 eq 4.10 and discussion immediately prior
        r_ref, err_r_ref = self.get_rref(position)
        theta = np.degrees(r_ref / (self.D*self.kpc))
        err_theta = np.degrees(err_r_ref / (self.D*self.kpc))
        return theta, err_theta

    def get_m(self, position, ssize=0):
        """
        calculate the modulation index using parameter ξ for a given sky coord
        :param position: astropy.coordinates.SkyCoord
        :param stype: Ignored
        :param ssize: source size in deg
        :return:
        """
        xi, err_xi = self.get_xi(position)
        theta, err_theta = self.get_theta(position)
        m = xi ** (-1. / 3.)*(theta / ssize) ** (7. / 6.)
        err_m = (1. / 3.) * (err_xi / xi) * m * (7. / 6.) * (err_theta / theta)

        #theta, err_theta = self.get_theta(position)
        #large = np.argwhere(ssize > theta)
        #m[large] *= (theta[large] / ssize[large]) ** (7. / 6.)
        #err_m[large] *= (7. / 6.) * (err_theta[large] / theta[large])
        return m, err_m

    def get_timescale(self, position):
        """
        calculate the refractive timescale using parameter ξ for a given sky coord
        timescale is in years
        :param position: astropy.coordinates.SkyCoord
        :return:
        """
        xi, err_xi = self.get_xi(position)
        tref = self.rf *xi / self.v / seconds_per_year
        err_tref = (err_xi/xi)*tref
        return tref, err_tref

    def get_rms_var(self, position, stype=AGN, ssize=0, nyears=1):
        """
        calculate the expected RMS variation in nyears at a given sky coord
        rms variability is fraction/year
        :param position: astropy.coordinates.SkyCoord
        :param nyears: timescale of interest
        :return:
        """
        tref, err_tref=self.get_timescale(position)
        m, err_m = self.get_m(position, stype, ssize)
        #basic uncertainty propagation, can probably change.
        t = m/tref * nyears
        err_t = ((err_m/m)+(err_tref/tref))*t
        return t, err_t

    def get_vo(self, position):
        """
        Calculate the transition frequency at a given sky location
        :param position:
        :return: Transition frequency in GHz
        """
        sm2, _ = self.get_sm(position)
        pow = (1 / (2 - self.beta))
        A = (2 ** (2 - self.beta) * (np.pi * self.re ** 2 * self.beta) * sm2 * self.kpc *
                gamma(-self.beta / 2) / gamma(self.beta / 2)) ** pow
        vo = self.c * (np.sqrt(self.D*self.kpc/(2*np.pi)) / A)**(1/(0.5 - 2*pow))
        return vo/1e9

def test_all_params():
    print("Testing with single positions")
    sm = SM(os.path.join('data', 'Halpha_map.fits'), os.path.join('data', 'Halpha_error.fits'), nu=1e8)
    pos = SkyCoord([0], [0], unit=(u.hour, u.degree))
    print("Hα = {0}".format(sm.get_halpha(pos)))
    print("ξ = {0}".format(sm.get_xi(pos)))
    print("m = {0}".format(sm.get_m(pos)))
    print("sm = {0} (m^-17/3)".format(sm.get_sm(pos)[0]*sm.kpc))
    print("t0 = {0} (sec)".format(sm.get_timescale(pos)))
    print("r_diff = {0} (m)".format(sm.get_rdiff(pos)))
    print("r_ref = {0} (m)".format(sm.get_rref(pos)))
    print("r_F = {0} (m)".format(sm.rf))
    print("rms = {0}".format(sm.get_rms_var(pos)))
    print("theta = {0} (rad)".format(np.radians(sm.get_theta(pos))))
    print("nu_0 = {0} (GHz)".format(sm.get_vo(pos)))


def test_multi_pos():
    print("Testing with list of positions")
    sm = SM(os.path.join('data', 'Halpha_map.fits'), os.path.join('data', 'Halpha_error.fits'))
    pos = SkyCoord([0, 4, 8, 12, 16, 20]*u.hour, [-90, -45, 0, 45, 90, -26]*u.degree)
    print("Hα = {0}".format(sm.get_halpha(pos)))
    print("ξ = {0}".format(sm.get_xi(pos)))
    print("m = {0}".format(sm.get_m(pos)))
    print("sm = {0}".format(sm.get_sm(pos)))
    print("t0 = {0}".format(sm.get_timescale(pos)))
    print("rms = {0}".format(sm.get_rms_var(pos)))
    print("theta = {0}".format(sm.get_theta(pos)))


if __name__ == "__main__":
    test_all_params()
    test_multi_pos()
