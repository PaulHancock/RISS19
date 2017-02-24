#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

from astropy.io import fits
from astropy.table import Table

import os
import sys
import argparse

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

    group2 = parser.add_argument_group('Sky Coordinates')
    group2.add_argument('--in', dest='infile', default=None, type=argparse.FileType('r'),
                        help="Table of coordinates")
    group2.add_argument('--incol', dest='cols', default=('ra', 'dec'), nargs=2, type=str,
                        help='Column names to read from input. [ra,dec]')
    group2.add_argument('--out', dest='outfile', default=None, type=argparse.FileType('w'),
                        help="Table of results")

    results = parser.parse_args()
    print(results)


