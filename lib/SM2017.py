#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

"""
NE2001 for extragalactic work.
"""


from astropy.constants import kpc, c, au
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
import numpy as np
import os
import logging
from scipy.special import gamma

__author__ = ['Paul Hancock', 'Elliott Charlton']
__date__ = '2019-06-07'

SECONDS_PER_YEAR = 3600 * 24 * 365.25

class SM(object):
    """
    :param ha_file:
    :param err_file:
    :param nu: freq in Hz
    :param d: distance in kpc
    :param v: in m/s
    :param log:
    """
    def __init__(self, ha_file, err_file=None, nu=185e6, log=None, d=None, v=10e3):

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
        self.lo = 1e18/(self.kpc*1e-3)  # 1e18m expressed in pc (also armstrong_electron_1985 !)
        self.eps = 1
        self.D = d  # kpc - distance to the screen
        self.c = c.value
        self.beta = 11/3
        self.re = 2.817e-15  # m
        self.v = v  # relative velocity of source/observer in m/s
        self.file = ha_file
        self.err_file = err_file
        self._load_file()

    def _load_file(self):
        self.hdu = fits.getheader(self.file, ignore_missing_end=True)
        self.wcs = WCS(self.hdu)
        self.data = fits.open(self.file, memmap=True, ignore_missing_end=True)[0].data
        if self.err_file:
            self.err_hdu = fits.getheader(self.err_file, ignore_missing_end=True)
            self.err_wcs = WCS(self.err_hdu)
            self.err_data = fits.open(self.err_file, memmap=True, ignore_missing_end=True)[0].data
        else:
            self.err_hud = self.err_wcs = self.err_data = None

        return

    def get_distance(self, position):
        """
        :param position: sky position
        :return: Distance to scattering screen in kpc
        """
        if self.D is not None:
            return np.ones(np.shape(position))*self.D
        # TODO: sort out gal_r and find a reference for it
        gal_r = 16.2  # kpc
        sun_r = 8.09  # kpc
        gal_h = 1.   # kpc
        theta = position.galactic.l.radian # angle from the GC along the plane
        phi = position.galactic.b.radian   # angle from the GC perp to the plane
        far_edge = sun_r*np.cos(theta) + np.sqrt(gal_r**2 - (sun_r*np.sin(theta))**2)
        top = (gal_h/2.) / np.abs(np.sin(phi))
        mask = np.where(top > far_edge)
        screen_dist = top
        screen_dist[mask] = far_edge[mask]
        return screen_dist/2.

    def get_rf(self, position):
        """
        :param position: Sky position
        :return: Fresnel scale in m
        """
        return np.sqrt(self.c * self.get_distance(position) * self.kpc / (2 * np.pi * self.nu))

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
        sm2 = iha / 198 * self.t4 ** 0.9 * self.eps ** 2 / (1 + self.eps ** 2) * self.lo ** (-2 / 3)
        err_sm2 = (err_iha / iha) * sm2
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
        rdiff = (2 ** (2 - self.beta) * (
                    np.pi * self.re ** 2 * (self.c / self.nu) ** 2 * self.beta) * sm2 * self.kpc *
                 gamma(-self.beta / 2) / gamma(self.beta / 2)) ** (1 / (2 - self.beta))
        err_rdiff = abs((1 / (2 - self.beta)) * (err_sm2 / sm2) * rdiff)
        return rdiff, err_rdiff


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
        rf = self.get_rf(position)
        xi = rf / rdiff
        err_xi = (err_rdiff/rdiff)*xi
        return xi, err_xi

    def get_theta(self, position):
        """
        calculate the size of the scattering disk for a given sky coord
        :param position: astropy.coordinates.SkyCoord
        :return: scattering disk in degrees
        """
        # See Narayan 1992 eq 4.10 and discussion immediately prior
        rdiff, err_rdiff = self.get_rdiff(position)
        theta = np.degrees((self.c/self.nu)/(2.* np.pi*rdiff))
        err_theta = np.degrees(err_rdiff / rdiff)*theta
        return theta, err_theta

    def get_m(self, position, ssize=0):
        """
        calculate the modulation index using parameter ξ for a given sky coord
        :param position: astropy.coordinates.SkyCoord
        :param ssize: source size in deg
        :return:
        """
        ssize = np.zeros(len(position)) + ssize
        xi, err_xi = self.get_xi(position)
        m = xi ** (-1. / 3.)
        err_m = (1. / 3.) * (err_xi / xi) * m
        theta, err_theta = self.get_theta(position)

        mask = np.where(ssize > theta)
        m[mask] = m[mask] * (theta[mask] / ssize[mask]) ** (7. / 6.)
        err_m[mask] = np.sqrt((err_m[mask]/m[mask]) ** (2.0) + ((7. / 6.) * (err_theta[mask] / theta[mask])) ** 2.) * m[mask]
        return m, err_m

    def get_timescale(self, position, ssize=0):
        """
        calculate the refractive timescale using parameter ξ for a given sky coord
        timescale is in years
        :param position: astropy.coordinates.SkyCoord
        :param ssize: source size in deg
        :return:
        """

        xi, err_xi = self.get_xi(position)
        ssize = np.zeros(len(position)) + ssize
        rf = self.get_rf(position)
        tref = rf * xi / self.v / SECONDS_PER_YEAR
        err_tref = (err_xi/xi)*tref

        # timescale is longer for 'large' sources
        theta, err_theta = self.get_theta(position)
        large = np.where(ssize > theta)
        tref[large] = tref[large] * ssize[large] / theta[large]
        err_tref[large] = tref[large] * np.sqrt((err_tref[large]/tref[large])**2.  + (err_theta[large]/theta[large])**2.)
        return tref, err_tref

    def get_rms_var(self, position, ssize=0, nyears=1):
        """
        calculate the expected modulation index observed when measured on nyears timescales
        at a given sky coord
        :param position: astropy.coordinates.SkyCoord
        :param ssize: source size in deg
        :param nyears: timescale of interest
        :param ssize: source size in deg
        :return:
        """
        ssize = np.zeros(len(position)) + ssize
        tref, err_tref = self.get_timescale(position, ssize=ssize)
        m, err_m = self.get_m(position, ssize=ssize)

        short = np.where(nyears * SECONDS_PER_YEAR < tref)
        m[short] *= (nyears / tref[short])
        err_m[short] = np.sqrt((err_m[short]/m[short]) ** 2. + (err_tref[short] / tref[short]) ** 2.) * m[short]
        return m, err_m

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
        D = self.get_distance(position)
        vo = self.c * (np.sqrt(D*self.kpc/(2*np.pi)) / A)**(1/(0.5 - 2*pow))
        return vo/1e9


def test_all_params():
    print("Testing with single positions")
    #original map
    #sm = SM(os.path.join('data', 'Halpha_map.fits'), os.path.join('data', 'Halpha_error.fits'), nu=1e8)
    #new map
    sm = SM(os.path.join('data', 'Ha_map_new.fits'), os.path.join('data', 'Ha_err_new.fits'), nu=1e8)
    pos = SkyCoord([0], [0], unit=(u.hour, u.degree))
    print("Hα = {0}".format(sm.get_halpha(pos)))
    print("ξ = {0}".format(sm.get_xi(pos)))
    print("m = {0}".format(sm.get_m(pos)))
    print("sm = {0} (m^-17/3)".format(sm.get_sm(pos)[0]*sm.kpc))
    print("t0 = {0} (sec)".format(sm.get_timescale(pos)))
    print("r_diff = {0} (m)".format(sm.get_rdiff(pos)))
    print("r_F = {0} (m)".format(sm.get_rf(pos)))
    print("rms = {0}".format(sm.get_rms_var(pos)))
    print("theta = {0} (rad)".format(np.radians(sm.get_theta(pos))))
    print("nu_0 = {0} (GHz)".format(sm.get_vo(pos)))
    print("Distance = {0}".format(sm.get_distance(pos)))

def test_multi_pos():
    print("Testing with list of positions")
    # original map
    # sm = SM(os.path.join('data', 'Halpha_map.fits'), os.path.join('data', 'Halpha_error.fits'), nu=1e8)
    # new map
    sm = SM(os.path.join('data', 'Ha_map_new.fits'), os.path.join('data', 'Ha_err_new.fits'), nu=1e8)
    pos = SkyCoord([0, 4, 8, 12, 16, 20]*u.hour, [-90, -45, 0, 45, 90, -26]*u.degree)
    print("Hα = {0}".format(sm.get_halpha(pos)))
    print("ξ = {0}".format(sm.get_xi(pos)))
    print("m = {0}".format(sm.get_m(pos)))
    print("sm = {0}".format(sm.get_sm(pos)))
    print("t0 = {0}".format(sm.get_timescale(pos)))
    print("rms = {0}".format(sm.get_rms_var(pos)))
    print("theta = {0}".format(sm.get_theta(pos)))
    print("Distance = {0}".format(sm.get_distance(pos)))

def write_multi_pos():
    from astropy.table import Table, Column
    RA=np.append(np.arange(0,360),np.arange(0,360))
    DEC=np.append(np.append(np.append(np.arange(-90,90),np.arange(-90,90)),np.arange(-90,90)),np.arange(-90,90))
    #Original map
    #sm = SM(os.path.join('data', 'Halpha_map.fits'), os.path.join('data', 'Halpha_error.fits'), nu=1e8)
    sm = SM(os.path.join('data', 'Ha_map_new.fits'), os.path.join('data', 'Ha_err_new.fits'), nu=1e8)



    pos = SkyCoord(RA * u.degree, DEC * u.degree)
    mvar=int(1)
    Ha,err_Ha=sm.get_halpha(pos)
    mod,err_m=sm.get_m(pos)
    t0,err_t0=sm.get_timescale(pos)
    theta,err_theta=sm.get_theta(pos)
    #tau,err_tau=sm.get_tau(pos)
    datatab1 = Table()
    datafile ='SM2017_test_m{0}.csv'.format(mvar)
    ### DATA FILE
    datatab1.add_column(Column(data=RA, name='RA'))
    datatab1.add_column(Column(data=DEC, name='DEC'))
    datatab1.add_column(Column(data=Ha, name='H_Alpha'))
    datatab1.add_column(Column(data=err_Ha, name='H_Alpha err'))
    #datatab1.add_column(Column(data=tau, name='Tau'))
    #datatab1.add_column(Column(data=err_tau, name='Tau err'))
    datatab1.add_column(Column(data=mod, name='Modulation'))
    datatab1.add_column(Column(data=err_m, name='Modulation err'))
    datatab1.add_column(Column(data=t0, name='Timescale'))
    datatab1.add_column(Column(data=err_t0, name='Timescale err'))
    datatab1.add_column(Column(data=theta, name='Theta'))
    datatab1.add_column(Column(data=err_theta, name='Theta err'))
    datatab1.write(datafile, overwrite=True)


if __name__ == "__main__":
    test_all_params()
    #test_multi_pos()
    #write_multi_pos()