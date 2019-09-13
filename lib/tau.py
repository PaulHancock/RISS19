#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

from astropy.constants import kpc, c
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.table import Table, Column
import numpy as np
import os
import logging

from scipy.special import gamma
import matplotlib.pyplot as plt
SFG = 0
AGN = 1
seconds_per_year = 3600 * 24 * 365.25
alpha = 3.86
import warnings
warnings.filterwarnings("ignore")
class SM(object):
    """
    :param ha_file:
    :param err_file:
    :param nu: freq in Hz
    :param d: distance in kpc
    :param v: in m/s
    :param log:
    """

    def __init__(self, ha_file, err_file=None, nu=185e6, log=None, v=10e3):

        if log is None:
            logging.basicConfig(format="%(module)s:%(levelname)s %(message)s")
            self.log = logging.getLogger("SM2017")
            self.log.setLevel(logging.DEBUG)
        else:
            self.log = log

        # define some of the constants that we need
        # i'm saving these here to allow for different instances to have different values
        self.nu = nu  # Hz
        self.kpc = kpc.value  # in m
        self.t4 = 0.8  # t/1e4 K
        self.lo = 1e18 / (self.kpc * 1e-3)  # pc
        self.eps = 1
        self.c = c.value
        self.beta = 11 / 3.
        self.seconds_per_year = 3600 * 24 * 365.25
        self.re = 2.817e-15  # m
        self.v = v  # relative velocity of source/observer in m/s
        # self.log.debug("data:{0} err:{1}".format(ha_file,err_file))
        self.file = ha_file
        self.err_file = err_file
        self.tau_file = "data/tau_map_near.fits"
        #self.tau_file = "data/ymw16tau_map_lin.fits"
        self._load_file()

    def _load_file(self):
        self.hdu = fits.getheader(self.file, ignore_missing_end=True)
        self.wcs = WCS(self.hdu)
        self.data = fits.open(self.file, memmap=True, ignore_missing_end=True)[0].data
        self.tau = fits.open(self.tau_file, memmap=True, ignore_missing_end=True)[0].data
        if self.err_file:
            self.err_hdu = fits.getheader(self.err_file, ignore_missing_end=True)
            self.err_wcs = WCS(self.err_hdu)
            self.err_data = fits.open(self.err_file, memmap=True, ignore_missing_end=True)[0].data
        else:
            self.err_hud = self.err_wcs = self.err_data = None

        return

    def get_tau(self, position):
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
        tau = self.tau[y, x]
        alpha = 3.86
        tau = (tau/1e3) * ((self.nu / 1e9) ** (-alpha)) #In seconds
        err_tau = 0.1*tau
        return tau, err_tau

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
        # rdiff = metres
        return rdiff, err_rdiff

    def get_rf(self, position):
        rdiff, err_rdiff = self.get_rdiff(position)
        tau, err_tau = self.get_tau(position)

        rf = rdiff * (4.*np.pi*self.nu*tau) ** (1. / 2.)

        err_rf = ((err_rdiff / rdiff)**2.0 + (1. / 2. * (err_tau / tau))**2.0) * rf
        # rf = rdiff (m) x sqrt(tau(s) x nu(1/s))
        # rf = metres
        return rf, err_rf

    def get_rref(self, position):
        """
        Calculate the refractive scale at the given sky coord
        :param position: astropy.coordinates.SkyCoord
        :return: parameter r_ref in m
        """
        # Narayan 1992 eq 4.2
        rdiff, err_rdiff = self.get_rdiff(position)
        rf, err_rf = self.get_rf(position)
        rref = (rf ** 2.0) / rdiff
        err_rref = np.sqrt((err_rdiff / rdiff)**2.0 + (2.0* (err_rf / rf))**2.0) * rref
        # rref = metres
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
        rf, err_rf = self.get_rf(position)
        xi = rf / rdiff
        err_xi = np.sqrt((err_rdiff / rdiff)**2.0 + (err_rf / rf)**2.0) * xi
        # xi = unitless
        return xi, err_xi

    def get_theta(self, position):
        """
        calculate the size of the scattering disk for a given sky coord
        :param position: astropy.coordinates.SkyCoord
        :return: scattering disk in degrees
        """
        # See Narayan 1992 eq 4.10 and discussion immediately prior
        rdiff, err_rdiff = self.get_rdiff(position)
        theta = np.degrees((self.c / (2.*np.pi*self.nu)) / rdiff)
        err_theta = theta * np.degrees(err_rdiff / rdiff)
        #theta = radians then converted to degrees
        return theta, err_theta

    def get_mold(self, position, ssize=0.):
        """
        calculate the modulation index using parameter ξ for a given sky coord
        :param position: astropy.coordinates.SkyCoord
        :param stype: Ignored
        :param ssize: source size in deg
        :return:
        """
        xi, err_xi = self.get_xi(position)
        m = xi ** (-1. / 3.)
        err_m = (1. / 3.) * (err_xi / xi) * m

        theta, err_theta = self.get_theta(position)
        large = np.where(ssize > theta)
        if len(large[0]) >= 1:
            m[large] *= (theta[large] / ssize[large]) ** (7. / 6.)
            err_m[large] = np.sqrt(err_m[large]**2.0 +((7. / 6.) * (err_theta[large] / theta[large]))**2.0) *m[large]
        return m, err_m


    def get_m(self, position, ssize=0.):
        xi, err_xi = self.get_xi(position)
        m = xi ** (-1. / 3.)
        err_m = (1. / 3.) * (err_xi / xi) * m
        theta, err_theta = self.get_theta(position)
        large = np.where(ssize > theta)
        if len(large[0]) > 1:
            m[large] =m[large] * (theta[large] / ssize[large]) ** (7. / 6.)
            err_m[large] = np.sqrt((err_m[large]/m[large])**(2.0) + ((7. / 6.) * (err_theta[large] / theta[large]))**2.) * m[large]
        # m = unitless
        return m, err_m

    def get_timescale(self, position, ssize=0.):
        """
        calculate the refractive timescale using parameter ξ for a given sky coord
        timescale is in years
        :param position: astropy.coordinates.SkyCoord
        :return:
        """
        xi, err_xi = self.get_xi(position)
        rf, err_rf = self.get_rf(position)
        tref = rf * xi / self.v / self.seconds_per_year
        err_tref = np.sqrt((err_xi / xi)**2.0 + (err_rf / rf)**2.0) * tref

        # timescale is longer for 'large' sources
        theta, err_theta = self.get_theta(position)
        large = np.where(ssize > theta)
        if len(large[0]) > 1:
            tref[large] *= ssize / theta[large]
            err_tref[large] = np.sqrt((err_tref[large]/tref[large])**2. + (err_theta[large] / theta[large])**2.0) * tref[large]
        #tref = m/(m/s) = seconds converted into years
        return tref, err_tref

    def get_rms_var(self, position, ssize=0., nyears=1):
        """
        calculate the expected modulation index observed when measured on nyears timescales
        at a given sky coord
        :param position: astropy.coordinates.SkyCoord
        :param nyears: timescale of interest
        :param ssize: source size in deg
        :return:
        """

        tref, err_tref = self.get_timescale(position, ssize=ssize)
        m, err_m = self.get_m(position, ssize=ssize)

        short = np.where(nyears  < tref)
        if (len(short[0]) > 1):
            m[short] *= (nyears / tref[short])
            err_m[short] = np.sqrt((err_m[short]/m[short])**2. + (err_tref[short] / tref[short])**2.) * m[short]
        return m, err_m

    def get_all(self, position, ssize=0):
        Ha, err_Ha = self.get_halpha(position)
        xi, err_xi = self.get_xi(position)
        theta, err_theta = self.get_theta(position)
        sm, err_sm = self.get_sm(position)
        m, err_m = self.get_m(position, ssize)
        t0, err_t0 = self.get_timescale(position)
        rms, err_rms = self.get_rms_var(position, ssize)
        tau, err_tau = self.get_tau(position)

        return Ha, err_Ha, xi, err_xi, theta, err_theta, sm,err_sm, m, err_m, t0, err_t0, rms, err_rms, tau, err_tau


def test_all_params():
    print("Testing with single positions")
    #original map
    #sm = SM(os.path.join('data', 'Halpha_map.fits'), os.path.join('data', 'Halpha_error.fits'), nu=1e8)
    #new map
    sm = SM(os.path.join('data', 'Ha_map_new.fits'), os.path.join('data', 'Ha_err_new.fits'), nu=1e8)
    pos = SkyCoord([0], [0], unit=(u.hour, u.degree))
    print("Hα = {0}".format(sm.get_halpha(pos)))
    print("Tau = {0}".format(sm.get_tau(pos)))
    print("ξ = {0}".format(sm.get_xi(pos)))
    print("m = {0}".format(sm.get_m(pos)))
    print("sm = {0} (m^-17/3)".format(sm.get_sm(pos)[0]*sm.kpc))
    print("t0 = {0} (sec)".format(sm.get_timescale(pos)))
    print("r_diff = {0} (m)".format(sm.get_rdiff(pos)))
    print("r_ref = {0} (m)".format(sm.get_rref(pos)))
    print("r_F = {0} (m)".format(sm.get_rf))
    print("rms = {0}".format(sm.get_rms_var(pos)))
    print("theta = {0} (rad)".format(np.radians(sm.get_theta(pos))))

gl=np.arange(0,360,1)
def test_multi_pos():
    print("Testing with list of positions")
    # original map
    # sm = SM(os.path.join('data', 'Halpha_map.fits'), os.path.join('data', 'Halpha_error.fits'), nu=1e8)
    # new map
    sm = SM(os.path.join('data', 'Ha_map_new.fits'), os.path.join('data', 'Ha_err_new.fits'), nu=1e8)
    pos = SkyCoord([0, 4, 8, 12, 16, 20]*u.hour, [-90, -45, 0, 45, 90, -26]*u.degree)
    print("Hα = {0}".format(sm.get_halpha(pos)))
    print("Tau = {0}".format(sm.get_tau(pos)))
    print("ξ = {0}".format(sm.get_xi(pos)))
    print("m = {0}".format(sm.get_m(pos)))
    print("sm = {0}".format(sm.get_sm(pos)))
    print("t0 = {0}".format(sm.get_timescale(pos)))
    print("rms = {0}".format(sm.get_rms_var(pos)))
    print("theta = {0}".format(sm.get_theta(pos)))

def write_multi_pos():
    RA=[]
    DEC=[]
    mult=1./10.
    for i in range(0, int(360*mult)):
        for j in range(int(-90*mult), int(90*mult)):
            #print(i)
            RA.append(np.float(i*(1./mult)))
            DEC.append(np.float(j*(1./mult)))
    #RA=np.append(np.arange(0,360),np.arange(0,360))
    #DEC=np.append(np.append(np.append(np.arange(-90,90),np.arange(-90,90)),np.arange(-90,90)),np.arange(-90,90))
    #Original map
    #sm = SM(os.path.join('data', 'Halpha_map.fits'), os.path.join('data', 'Halpha_error.fits'), nu=1e8)
    # new map
    sm = SM(os.path.join('data', 'Ha_map_new.fits'), os.path.join('data', 'Ha_err_new.fits'), nu=1e8)



    pos = SkyCoord(RA * u.degree, DEC * u.degree)
    mvar=int(1)
    Ha,err_Ha=sm.get_halpha(pos)
    mod,err_m=sm.get_m(pos)
    #print(mod)
    t0,err_t0=sm.get_timescale(pos)
    theta,err_theta=sm.get_theta(pos)
    tau,err_tau=sm.get_tau(pos)
    datatab1 = Table()
    datafile ='tau_skytest_m{0}.csv'.format(mvar)
    #print(np.shape(err_tau),np.shape(Ha), np.shape(RA))
    #print('mod_mean', np.mean(mod))
    ### DATA FILE
    datatab1.add_column(Column(data=RA, name='RA'))
    datatab1.add_column(Column(data=DEC, name='DEC'))
    datatab1.add_column(Column(data=Ha, name='H_Alpha'))
    datatab1.add_column(Column(data=err_Ha, name='H_Alpha err'))
    datatab1.add_column(Column(data=tau, name='Tau'))
    datatab1.add_column(Column(data=err_tau, name='Tau err'))
    datatab1.add_column(Column(data=mod, name='Modulation'))
    datatab1.add_column(Column(data=err_m, name='Modulation err'))
    datatab1.add_column(Column(data=t0, name='Timescale'))
    datatab1.add_column(Column(data=err_t0, name='Timescale err'))
    datatab1.add_column(Column(data=theta, name='Theta'))
    datatab1.add_column(Column(data=err_theta, name='Theta err'))
    datatab1.write(datafile, overwrite=True)

def test_poss():
    r=np.arange(0,360,1)
    d=np.arange(-90,90,1)
    RA=np.random.choice(r,size=10000)
    DEC=np.random.choice(d,size=10000)


    sm = SM(os.path.join('data', 'Ha_map_new.fits'), os.path.join('data', 'Ha_err_new.fits'), nu=1e8)
    pos = SkyCoord(RA * u.degree, DEC * u.degree)
    print('rdiff',np.where(np.isnan(sm.get_rdiff(pos))==True))

    """    Ha, err_Ha, xi, err_xi, theta, err_theta, sm,err_sm, m, err_m, t0, err_t0, rms, err_rms, tau, err_tau= sm.get_all(pos)
    print('ha',np.where(np.isnan(Ha)==True))
    print('tau',np.where(np.isnan(tau)==True))
    print('xi',np.where(np.isnan(xi)==True))
    print('m',np.where(np.isnan(m)==True))
    #print('sm',np.where(np.isnan(theta)==True))
    print('time',np.where(np.isnan(t0)==True))
    print('rms',np.where(np.isnan(rms)==True))
    print('theta',np.where(np.isnan(theta)==True))
    """
if __name__ == "__main__":
    #test_all_params()
    #test_multi_pos()
    #write_multi_pos()
    test_poss()

#sm = SM(os.path.join('data', 'Ha_map_new.fits'), os.path.join('data', 'Ha_err_new.fits'), nu=1e8)
#sm2 = SM(os.path.join('data', 'Halpha_map.fits'), os.path.join('data', 'Halpha_error.fits'), nu=1e8)
#pos = SkyCoord([0, 4, 8, 12, 16, 20]*u.hour, [-90, -45, 0, 45, 90, -26]*u.degree)
#pos = SkyCoord([0, 4, 8, 12, 16, 20]*u.hour, [-90, -45, 0, 45, 90, -26]*u.degree)
#pos2 = SkyCoord([0, 4, 8, 12, 16, 20]*u.degree, [-90, -45, 0, 45, 90, -26]*u.degree)



