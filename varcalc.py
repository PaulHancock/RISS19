#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

from astropy.coordinates import SkyCoord
from astropy.table import Table, Column
import astropy.units as u
from astropy.utils.exceptions import AstropyWarning

from lib.SM2017 import SM
import os
import sys
import argparse

# Turn off the stupid warnings that Astropy emits when loading just about any fits file.
import warnings
warnings.simplefilter('ignore', category=AstropyWarning)


__author__ = 'Paul Hancock'
__date__ = '2017-02-24'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    group1 = parser.add_argument_group('Parameter selection')
    group1.add_argument('-H', '--Halpha', dest='halpha', action='store_true', default=False,
                        help='Calculate Hα intensity (Rayleighs)')
    group1.add_argument('-x', '--xi', dest='xi', action='store_true', default=False,
                        help='Calculate ξ')
    group1.add_argument('-m', '--modulation', dest='m', action='store_true', default=False,
                        help='Calculate modulation index')
    group1.add_argument('-s', '--sm', dest='sm', action='store_true', default=False,
                        help='Calculate scintillation measure')
    group1.add_argument('-t', '--timescale', dest='t0', action='store_true', default=False,
                        help='Calculate timescale of variability')
    group1.add_argument('-r', '--rms', dest='rms', action='store_true', default=False,
                        help='Calculate rms variability over 1 year')
    group1.add_argument('--all', dest='do_all', action='store_true', default=False,
                        help='Include all parameters')

    group2 = parser.add_argument_group('Input and output data')
    group2.add_argument('--in', dest='infile', default=None, type=argparse.FileType('r'),
                        help="Table of coordinates")
    group2.add_argument('--incol', dest='cols', default=('ra', 'dec'), nargs=2, type=str,
                        help='Column names to read from input. [ra,dec]')
    group2.add_argument('--out', dest='outfile', default=None, type=str,
                        help="Table of results")
    group2.add_argument('--pos', dest='pos', default=None, nargs=2, type=float,
                        help="Single coordinates in ra/dec degrees")

    group3 = parser.add_argument_group('Input parameter settings')
    group3.add_argument('--freq', dest='frequency', default=185, type=float,
                        help="Frequency in MHz")

    results = parser.parse_args()

    if results.do_all:
        results.halpha = results.sm = results.m = results.rms = results.xi = results.t0 = True

    nu = results.frequency*1e6
    # For doing a one off position calculation
    if results.pos:
        ra, dec = results.pos
        pos = SkyCoord([ra]*u.degree, [dec]*u.degree)
        sm = SM(os.path.join('data', 'Halpha_map.fits'), nu=nu)
        print(sm.get_halpha(pos))
        sys.exit()

    if results.infile:
        if not results.outfile:
            print("Output file is required")
            sys.exit(1)
        # read the input data
        tab = Table.read(results.infile)
        ra = tab[results.cols[0]]
        dec = tab[results.cols[1]]
        # create the sky coordinate
        pos = SkyCoord(ra*u.degree, dec*u.degree)
        # make the SM object
        sm = SM(os.path.join('data', 'Halpha_map.fits'), nu=nu)
        # make a new table for writing
        tab = Table()
        tab.add_column(ra)
        tab.add_column(dec)
        if results.halpha:
            tab.add_column(Column(data=sm.get_halpha(pos), name='Halpha'))
        if results.xi:
            tab.add_column(Column(data=sm.get_xi(pos), name='xi'))
        if results.sm:
            tab.add_column(Column(data=sm.get_sm(pos), name='sm'))
        if results.m:
            tab.add_column(Column(data=sm.get_m(pos), name='m'))
        if results.t0:
            tab.add_column(Column(data=sm.get_timescale(pos), name='t0'))
        if results.rms:
            tab.add_column(Column(data=sm.get_rms_var(pos), name='rms1yr'))
        tab.write(results.outfile, overwrite=True)





