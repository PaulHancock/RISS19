#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

"""
NE2001 for extragalactic work.
"""

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import os
from scipy.constants import c
from scipy.special import gamma

__author__ = 'Paul Hancock'
__date__ = '2017-02-23'


nu = 185e6 #MHz

def get_halpha(position):
    """
    Return the Halpha +\- err for a given location on the sky.
    :param position:
    :return:
    """


def get_xi(position):
    """
    Calculate the ξ parameter
    :param posiotin:
    :return:
    """
    hmap = os.path.join('data', 'Halpha_map.fits')
    iha_wcs = WCS(fits.getheader(hmap))

    # convert the (ra,dec) position into (x, y) pixel coords)
    x, y = iha_wcs.all_world2pix([position], 0)[0]
    x = np.int(np.floor(x))
    y = np.int(np.floor(y))
    iha = fits.open(hmap, memmap=True)[0].data[x, y]

    kpc = 3.086e19  # in m
    t4 = 0.8  # 1e4 K
    lo = 1e18/(kpc*1e-3) # pc
    eps = 1
    D = 1 #kpc

    # Cordes2002
    sm2 = iha/198 * t4**0.9 * eps**2/(1+eps**2) * lo**(-2/3)

    # Marcquart & Koay 2013
    # z = 0
    beta = 11/3
    re = 2.817e-15  # m
    # l1ghz = c / 1e9  # wavelength at 1GHz
    # sm2 * kpc converts SM into SI units.
    rdiff = (2**(2-beta) * (np.pi * re**2 * (c/nu)**2 * beta) * sm2 * kpc * gamma(-beta/2)/gamma(beta/2))**(1/(2-beta))



    # Fresnel scale assuming that D = 1kpc
    rf_1kpc = np.sqrt( c * kpc / (2*np.pi*nu))
    tref = rf_1kpc**2 / rdiff / 1e4 / 3600 / 24 / 365.25  # years

    xi = rf_1kpc / rdiff

    m = xi**(-1/3.)

    thetaF = rf_1kpc /kpc # rad
    theta_ref = xi*np.degrees(thetaF)*3600e3 # milli-arcsec
    return xi

if __name__ == "__main__":
    print(get_xi((0, 0)))