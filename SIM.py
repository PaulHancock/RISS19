from __future__ import print_function, division

import os
import datetime
import healpy as hp
import numpy as np
from AegeanTools.regions import Region
import cPickle
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table, Column
import astropy.units as u
from lib.SM2017 import SM

datadir = os.path.join(os.path.dirname(__file__), 'data')

SFG=0
AGN=1
stypes=[SFG, AGN]
sprobs=[0.5, 0.5]

def pos_gen(num):
    """
    A function to generate a number of random points in RA/DEC
    Input:  Number of points to generate
    Output: List of RA/DEC (2,1) array
    """
    lim = int(num * 2.5)
    x = []
    y = []
    z = []
    r = []
    i = 0
    x1 = np.array(np.random.uniform(-1.0, 1.0, lim))
    y1 = np.array(np.random.uniform(-1.0, 1.0, lim))
    z1 = np.array(np.random.uniform(-1.0, 1.0, lim))
    rad = (x1 ** 2.0 + y1 ** 2.0 + z1 ** 2.0) ** (0.5)
    for i in range(0, len(rad)):
        if rad[i] <= 1.:
            x.append(x1[i])
            y.append(y1[i])
            z.append(z1[i])
            r.append(rad[i])
    x, y, z = np.array(x) / np.array(r), np.array(y) / np.array(r), np.array(z) / np.array(r)
    r0 = (x ** 2.0 + y ** 2.0 + z ** 2.0) ** (0.5)
    theta = np.arccos(z / r0) * 180 / (np.pi)
    theta = theta - 90
    theta = theta[:num]
    phi = np.arctan2(y, x) * 180 / (np.pi)
    phi = phi + 180
    phi = phi[:num]
    return phi, theta


def region_gen(num, reg_file):
    """
    Takes in a list of positions and removes points outside the MIMAS region
    Input:  RA/DEC positions and MIMAS region file.
    Output: List of RA/DEC inside the correct region.
    """
    reg_ind = []
    reg_ra = []
    reg_dec = []
    region = cPickle.load(open(reg_file, 'rb'))
    while len(reg_ind) < num:
        RA, DEC = pos_gen(num * 40)
        reg_arr = region.sky_within(RA, DEC, degin=True)
        for i in range(0, len(reg_arr)):
            if reg_arr[i] == True and len(reg_ra) < num:
                reg_ind.append(i)
                reg_ra.append(RA[i])
                reg_dec.append(DEC[i])
    return np.array(reg_ra), np.array(reg_dec)


def flux_gen(region_arr, fdl, upp_lim):
    """
    Function to distribute flux across all points
    Input:  Flux Density limit, RA/DEC positions, source distribution function
    Output: Flux for each RA/DEC point
    """
    low_lim=fdl
    flux_arr = []
    for i in range(0, len(region_arr)):
        flux_arr.append(np.random.uniform(low_lim, upp_lim))
    return flux_arr

#def stype_gen(region_pos, soruce_size):
def stype_gen(arr):
    """
    Function to determine if a source is of type AGN or SFR
    Input:  RA/DEC list (source_size?)
    Output: AGN (1?) or SFR (0?)
    """
    stype_arr = []
    for i in range(0, len(arr)):
        stype_arr.append(np.random.choice(stypes, p=sprobs))
    return stype_arr
def ssize_gen(flux,stype):
    """
    Generates source size based on flux and stype given.
    Input: Flux and source type
    Output: Source size
    """
    arcsec=3600*180/np.pi
    ssize_arr=[]
    for i in range(0,len(stype)):
        if stype[i] == AGN:
            ssize_arr.append(1/(arcsec*1000))
        elif stype[i]== SFG:
            ssize_arr.append((30/1000)/arcsec)

    return ssize_arr

def file_gen(ar1, ar2, ar3, ar4, ar5, output_name):
    """
    Function to create output csv file for required data
    Input: arrays for RA/DEC
    Output: CSV file
    """
    output_table= Table([ar1,ar2,ar3,ar4, ar5], names=('RA','DEC','Flux','stype', 'ssize'), meta={'name': output_name})
    output_table.write(output_name, overwrite=True)
fdl=0.001
upp_lim=10
table_name='test.fits'
RA,DEC= region_gen(1000,'testreg.mim')
flux=flux_gen(RA,fdl,upp_lim)
stype=stype_gen(RA)
ssize=ssize_gen(flux,stype)
file_gen(RA,DEC,flux,stype,ssize, table_name)

frame = 'fk5'
outfile='outtest.csv'
file_name='testreg.mim'
nu = 185*1e6

tab = Table.read(table_name)
ra,dec= RA,DEC
# create the sky coordinate
pos = SkyCoord(ra*u.degree, dec*u.degree, frame=frame)
# make the SM object
sm = SM(ha_file=os.path.join(datadir, 'Halpha_map.fits'),
        err_file=os.path.join(datadir, 'Halpha_error.fits'),
        nu=nu)
#halpha
val, err = sm.get_halpha(pos)
tab.add_column(Column(data=val, name='Halpha'))
tab.add_column(Column(data=err, name='err_Halpha'))
#xi
val, err = sm.get_xi(pos)
tab.add_column(Column(data=val, name='xi'))
tab.add_column(Column(data=err, name='err_xi'))
#sm
val, err = sm.get_sm(pos)
tab.add_column(Column(data=val, name='sm'))
tab.add_column(Column(data=err, name='err_sm'))
#mod

val, err = sm.get_m(pos)
tab.add_column(Column(data=val, name='m'))
tab.add_column(Column(data=err, name='err_m'))
#t0
val, err = sm.get_timescale(pos)
tab.add_column(Column(data=val, name='t0'))
tab.add_column(Column(data=err, name='err_t0'))
#rms
val, err = sm.get_rms_var(pos)
tab.add_column(Column(data=val, name='rms1yr'))
tab.add_column(Column(data=err, name='err_rms1yr'))
#theta
val, err = sm.get_theta(pos)
tab.add_column(Column(data=val, name='theta_r'))
tab.add_column(Column(data=err, name='err_theta_r'))
tab.write(outfile, overwrite=True)


