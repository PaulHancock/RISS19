#!/usr/bin/python
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
from scipy.special import gamma

__author__ = 'Paul Hancock'
__date__ = '2017-02-23'


class SM(object):
    def __init__(self, ha_file, err_file=None):
        # define some of the constants that we need
        self.nu = 185e6  # MHz
        self.kpc = kpc.value  # in m
        self.t4 = 0.8  # t/1e4 K
        self.lo = 1e18/(self.kpc*1e-3)  # pc
        self.eps = 1
        self.D = 1  # kpc
        self.c = c.value
        self.beta = 11/3
        self.re = 2.817e-15  # m
        self.file = ha_file
        self.err_file = err_file
        self._load_file()

    def _load_file(self):
        self.hdu = fits.getheader(self.file)
        self.wcs = WCS(self.hdu)
        self.data = fits.open(self.file, memmap=True)[0].data
        if self.err_file:
            self.err_hdu = fits.getheader(self.err_file)
            self.err_wcs = WCS(self.err_hdu)
            self.err_data = fits.open(self.err_file, memmap=True)[0].data
        else:
            self.err_hud = self.err_wcs = self.err_data = None

        return

    def get_halpha(self, position):
        """
        Return the Halpha for a given location on the sky.
        :param position: astropy.coordinates.SkyCoord
        :return:
        """
        ra = position.ra.degree
        dec = position.dec.degree
        x, y = self.wcs.all_world2pix([(ra, dec)], 0)[0]
        x = np.int(np.floor(x))
        y = np.int(np.floor(y))
        iha = self.data[x, y]
        return iha

    def get_sm(self, position):
        """
        Return the scintillation measure for a given location on the sky.
        :param position: astropy.coordinates.SkyCoord
        :return:
        """
        iha = self.get_halpha(position)
        # Cordes2002
        sm2 = iha/198 * self.t4**0.9 * self.eps**2/(1+self.eps**2) * self.lo**(-2/3)
        return sm2

    def get_xi(self, position):
        """
        calculate the parameter ξ for a given sky coord
        : param position: astropy.coordinates.SkyCoord
        :return: parameter ξ
        """
        sm2 = self.get_sm(position)
        rdiff = (2**(2-self.beta) * (np.pi * self.re**2 * (self.c/self.nu)**2 * self.beta) * sm2 * self.kpc *
                 gamma(-self.beta/2)/gamma(self.beta/2))**(1/(2-self.beta))
        # Fresnel scale assuming that D = 1kpc
        rf_1kpc = np.sqrt( self.c * self.kpc / (2*np.pi*self.nu))
        xi = rf_1kpc / rdiff
        return xi


def test_get_xi():
    sm = SM(os.path.join('data', 'Halpha_map.fits'), os.path.join('data', 'Halpha_error.fits'))
    pos = SkyCoord(0, 0, unit=(u.hour, u.degree))
    print(sm.get_xi(pos))


if __name__ == "__main__":
    test_get_xi()