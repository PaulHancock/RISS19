#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

from astropy.coordinates import SkyCoord
from astropy.table import Table, Column
import astropy.units as u
from astropy.utils.exceptions import AstropyWarning

from lib.SM2017 import SM
import logging
import os
import sys
import argparse
import numpy as np

# Turn off the stupid warnings that Astropy emits when loading just about any fits file.
import warnings
warnings.simplefilter('ignore', category=AstropyWarning)

# configure logging
logging.basicConfig(format="%(module)s:%(levelname)s %(message)s")
log = logging.getLogger("varcalc")
log.setLevel(logging.INFO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    group1 = parser.add_argument_group('Output parameter selection')
    group1.add_argument('-H', '--Halpha', dest='halpha', action='store_true', default=False,
                        help='Calculate Hα intensity (Rayleighs)')
    group1.add_argument('-x', '--xi', dest='xi', action='store_true', default=False,
                        help='Calculate ξ (dimensionless)')
    group1.add_argument('-m', '--modulation', dest='m', action='store_true', default=False,
                        help='Calculate modulation index (fraction)')
    group1.add_argument('-s', '--sm', dest='sm', action='store_true', default=False,
                        help='Calculate scintillation measure (kpc m^{-20/3})')
    group1.add_argument('-t', '--timescale', dest='t0', action='store_true', default=False,
                        help='Calculate timescale of variability (years)')
    group1.add_argument('-r', '--rms', dest='rms', action='store_true', default=False,
                        help='Calculate rms variability over 1 year (fraction/year)')
    group1.add_argument('-d', '--theta', dest='theta', action='store_true', default=False,
                        help='Calculate the scattering disk size (deg)')
    group1.add_argument('-v', '--nuzero', dest='nuzero', action='store_true', default=False,
                        help='Calculate the transition frequency (GHz)')
    group1.add_argument('-f', '--fzero', dest='fzero', action='store_true', default=False,
                        help='Calculate the Fresnel zone (deg)')
    group1.add_argument('--distance', dest='dist', action='store_true', default=False,
                        help='Calculate the model distance')
    group1.add_argument('--all', dest='do_all', action='store_true', default=False,
                        help='Include all parameters')

    group2 = parser.add_argument_group('Input and output data')
    group2.add_argument('--in', dest='infile', default=None, type=argparse.FileType('r'),
                        help="Table of coordinates")
    group2.add_argument('--incol', dest='cols', default=('ra', 'dec'), nargs=2, type=str,
                        help='Column names to read from input. [ra,dec]')
    group2.add_argument('--out', dest='outfile', default=None, type=str,
                        help="Table of results")
    group2.add_argument('--append', dest='append', action='store_true', default=False,
                        help="Append the data to the input data (write a new file)")
    group2.add_argument('--pos', dest='pos', default=None, nargs=2, type=float,
                        help="Single coordinates in ra/dec degrees")
    group2.add_argument('-g', '--galactic', dest='galactic', action='store_true', default=False,
                        help='Interpret input coordinates as l/b instead of ra/dec (default False)')
    group2.add_argument('--debug', dest='debug', action='store_true', default=False,
                        help='Debug mode (default False)')

    group3 = parser.add_argument_group('Input parameter settings')
    group3.add_argument('--freq', dest='frequency', default=185, type=float,
                        help="Frequency in MHz")
    group3.add_argument('--dist', dest='distance', default=1, type=float,
                        help="Distance to scattering screen in kpc")
    group3.add_argument('--vel', dest='velocity', default=10, type=float,
                        help="Relative motion of screen and observer in km/s")

    results = parser.parse_args()

    if results.debug:
        log.setLevel(logging.DEBUG)

    if results.do_all:
        results.halpha = results.sm = results.m = results.rms = True
        results.xi = results.t0 = results.theta = results.nuzero = results.fzero = True
        results.dist = True

    # data is stored in the data dir, relative to *this* file
    datadir = os.path.join(os.path.dirname(__file__), 'data')

    nu = results.frequency*1e6
    d = results.distance
    v = results.velocity * 1e3
    # For doing a one off position calculation

    if results.pos is None and results.infile is None:
        parser.print_usage()
        sys.exit(0)

    if results.galactic:
        log.info("Using galactic coordinates")
        frame = 'galactic'
    else:
        log.info("Using fk5 coordinates")
        frame = 'fk5'

    if results.pos:
        ra, dec = results.pos
        pos = SkyCoord([ra]*u.degree, [dec]*u.degree, frame=frame)
        log.info(os.path.join(datadir, 'Halpha_error.fits'))
        sm = SM(ha_file=os.path.join(datadir, 'Halpha_map.fits'),
                err_file=os.path.join(datadir, 'Halpha_error.fits'),
                nu=nu,
                log=log,
                d=d,
                v=v)
        if results.halpha:
            logging.debug(sm.get_halpha(pos))
            val,err=sm.get_halpha(pos)
            print("Halpha: ", val, "(Rayleighs)")
            print("err_Halpha: ", err, "(Rayleighs)")
        if results.xi:
            val, err = sm.get_xi(pos)
            print("xi: ", val)
            print("err_xi: ", err)
        if results.sm:
            val, err = sm.get_sm(pos)
            print("sm: ", val, "kpc m^{-20/3}")
            print("err_sm: ", err, "kpc m^{-20/3}")
        if results.m:
            val, err = sm.get_m(pos)
            print("m: ", val*100, "%")
            print("err_m: ", err*100, "%")
        if results.t0:
            val, err = sm.get_timescale(pos)
            print("t0: ", val, "years")
            print("err_t0: ", err, "years")
        if results.rms:
            val, err = sm.get_rms_var(pos)
            print("rms: ", val*100, "%/1year")
            print("err_rms: ", err*100, "%/1year")
        if results.theta:
            val, err = sm.get_theta(pos)
            print("theta: ", val, "deg")
            print("err_theta: ", err, "deg")
        if results.nuzero:
            val = sm.get_vo(pos)
            print("nu0: ", val, "GHz")
        if d == 0:
            d = sm.get_distance(pos)
            print("D: ", d, "kpc")
            rf = sm.get_rf(pos)
            print("theta_F0: ", np.degrees(rf/(d*sm.kpc)), "deg")
        sys.exit(0)

    if results.infile:
        if not results.outfile:
            print("Output file is required")
            sys.exit(1)
        # read the input data
        tab = Table.read(results.infile)
        ra = tab[results.cols[0]]
        dec = tab[results.cols[1]]
        # create the sky coordinate
        pos = SkyCoord(ra*u.degree, dec*u.degree, frame=frame)
        # make the SM object
        sm = SM(ha_file=os.path.join(datadir, 'Halpha_map.fits'),
                err_file=os.path.join(datadir, 'Halpha_error.fits'),
                nu=nu,
                log=log,
                d=d,
                v=v)
        # make a new table for writing and copy the ra/dec unless we are appending to the old file
        if not results.append:
            tab = Table()
            tab.add_column(ra)
            tab.add_column(dec)
        else:
            print("Appending results to existing table")
        if results.halpha:
            val,err=sm.get_halpha(pos)
            tab.add_column(Column(data=val, name='Halpha'))
            tab.add_column(Column(data=err, name='err_Halpha'))
        if results.dist:
            val =sm.get_distance(pos)
            tab.add_column(Column(data=val, name='Distance'))
        if results.xi:
            val, err = sm.get_xi(pos)
            tab.add_column(Column(data=val, name='xi'))
            tab.add_column(Column(data=err, name='err_xi'))
        if results.sm:
            val, err = sm.get_sm(pos)
            tab.add_column(Column(data=val, name='sm'))
            tab.add_column(Column(data=err, name='err_sm'))
        if results.m:
            val, err = sm.get_m(pos)
            tab.add_column(Column(data=val, name='m'))
            tab.add_column(Column(data=err, name='err_m'))
        if results.t0:
            val, err = sm.get_timescale(pos)
            tab.add_column(Column(data=val, name='t0'))
            tab.add_column(Column(data=err, name='err_t0'))
        if results.rms:
            val, err = sm.get_rms_var(pos)
            tab.add_column(Column(data=val, name='rms1yr'))
            tab.add_column(Column(data=err, name='err_rms1yr'))
        if results.theta:
            val, err = sm.get_theta(pos)
            tab.add_column(Column(data=val, name='theta_r'))
            tab.add_column(Column(data=err, name='err_theta_r'))
        if results.nuzero:
            val = sm.get_vo(pos)
            tab.add_column(Column(data=val, name='nu0'))
        if d == 0:
            val = sm.get_distance(pos)
            tab.add_column(Column(data=val, name='D'))
        print("Writing to {0}".format(results.outfile))
        tab.write(results.outfile, overwrite=True)
