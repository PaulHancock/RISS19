from __future__ import print_function, division
import os
import logging
import cPickle
import argparse
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table, Column
import astropy.units as u
from lib.SM2017 import SM
from astropy.utils.exceptions import AstropyWarning
import warnings
warnings.filterwarnings("always")
warnings.simplefilter('ignore', category=AstropyWarning)

datadir = os.path.join(os.path.dirname(__file__), 'data')
SFG=0
AGN=1
stypes=[SFG, AGN]
sprobs=[0.5, 0.5]

###################
# INPUT VARIABLES #
###################
"""
Variables I want to be able to be read in from the user but havent fixed yet
placeholders below
"""
mod_cutoff=0.05
#tscale_cutoff=??
low_Flim=0.001  #Jy
upp_Flim=1.0    #Jy
table_name='test.fits' #Name of table you want to write to (FILE GEN)
output_name = 'outtest.csv' #Name of outfile (OUTPUT GEN)
#Above Table/Output could be same? Think I chose differently as to not overwrite, can double check later
region_name = 'testreg.mim' #Region file name
region = cPickle.load(open(region_name, 'rb'))
area=region.get_area(degrees=True)
obs_time=30.*24.*60.*60.
############################################

class SIM(object):
    def __init__(self, log=None):

        if log is None:
            logging.basicConfig(format="%(module)s:%(levelname)s %(message)s")
            self.log = logging.getLogger("SIM_new")
            self.log.setLevel(logging.DEBUG)
        else:
            self.log=log
        #Variables
        self.nu = 185 * 1e6
        self.mod_cutoff = 0.05
        self.low_Flim = 0.001  # Jy
        self.upp_Flim = 1.0  # Jy
        self.table_name = 'test.fits'  # Name of table you want to write to (FILE GEN)
        self.output_name = 'outtest.csv'  # Name of outfile (OUTPUT GEN)
        # Above Table/Output could be same? Think I chose differently as to not overwrite, can double check later
        self.region_name = 'testreg.mim'  # Region file name
        self.area = region.get_area(degrees=True)
        self.obs_time = 600. * 24. * 60. * 60.
        self.loops=10
        self.num_scale=40

    def flux_gen(self):
        """
        Function to distribute flux across all points
        Input:  Flux Density limit, RA/DEC positions, source distribution function
        Output: Flux for each RA/DEC point
        """
        low=self.low_Flim
        upp=self.upp_Flim
        def dn_func(val, a, b):
            output = (a * val ** (1.0 - b)) / (1.0 - b)
            return output

        def diff_counts(a, b, low, upp):
            bins = np.logspace(low, upp)
            Ni = []
            mpoint = []
            dS = []
            for i in range(0, len(bins) - 1):
                Ni.append(dn_func(bins[i + 1], a, b) - dn_func(bins[i], a, b))
                mpoint.append(np.sqrt(bins[i + 1] * bins[i]))
                dS.append(bins[i + 1] - bins[i])
            Ni = np.array(Ni)
            mpoint = np.array(mpoint)
            dS = np.array(dS)
            S = Ni / dS  # * mpoint**2.5
            return bins, S, mpoint, Ni, dS

        a = 3900
        x1, y1, mp = diff_counts(a, 1.6, -4., 0.)[0:3]
        x = mp
        y = y1 * mp ** 2.5
        p = np.polyfit(x, y, 1)
        q = np.poly1d(p)


        flux = np.logspace(np.log10(low), np.log10(upp), num=10)
        numS = q(flux)
        FLUX = []
        Area = self.area * (np.pi ** 2.) / (180 ** 2.)
        for i in range(0, len(flux) - 1):
            rang = np.arange(flux[i], flux[i + 1], (flux[i + 1] - flux[i]) / 10., dtype=float)
            run = np.int((numS[i + 1] - numS[i]) * Area)
            for j in range(0, run):
                FLUX.append(np.random.choice(rang))
        flux_arr = np.random.permutation(np.array(FLUX))

        return flux_arr, len(flux_arr)

    def pos_gen(self):
        """
        A function to generate a number of random points in RA/DEC
        Input:  Number of points to generate
        Output: List of RA/DEC (2,1) array
        """
        num=self.flux_gen()[1]
        num=num*self.num_scale
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
        r0 = (x ** 2.0 + y ** 2.0 + z ** 2.0) ** 0.5
        theta = np.arccos(z / r0) * 180 / np.pi
        theta = theta - 90
        theta = theta[:num]
        phi = np.arctan2(y, x) * 180 / (np.pi)
        phi = phi + 180
        phi = phi[:num]
        return phi, theta

    def region_gen(self, reg_file):
        """
        Takes in a list of positions and removes points outside the MIMAS region
        Input:  RA/DEC positions and MIMAS region file.
        Output: List of RA/DEC inside the correct region.
        """

        reg_ind = []
        reg_ra = []
        reg_dec = []
        region = cPickle.load(open(reg_file, 'rb'))
        num=self.flux_gen()[1]
        while len(reg_ind) < num:
            RA, DEC = self.pos_gen()

            reg_arr = region.sky_within(RA, DEC, degin=True)
            for i in range(0, len(reg_arr)):
                if reg_arr[i] == True and len(reg_ra) < num:
                    reg_ind.append(i)
                    reg_ra.append(RA[i])
                    reg_dec.append(DEC[i])
        return np.array(reg_ra), np.array(reg_dec)

    def stype_gen(self,arr):
        """
        Function to determine if a source is of type AGN or SFR
        Input:  RA/DEC list (source_size?)
        Output: AGN (1?) or SFR (0?)
        """
        stype_arr = []
        for i in range(0, len(arr)):
            stype_arr.append(np.random.choice(stypes, p=sprobs))
        return stype_arr

    def ssize_gen(self,flux, stype):
        """
        Generates source size based on flux and stype given.
        Input: Flux and source type
        Output: Source size
        """
        arcsec = 3600 * 180 / np.pi
        ssize_arr = []
        for i in range(0, len(stype)):
            if stype[i] == AGN:
                ssize_arr.append(1 / (3600 * 1000))
            elif stype[i] == SFG:
                ssize_arr.append((30 / (3600 * 1000)))

        return ssize_arr

    def file_gen(self,ar1, ar2, ar3, ar4, ar5, output_name):
        """
        Function to create output csv file for required data
        Input: arrays for RA/DEC
        Output: CSV file
        """
        output_table = Table([ar1, ar2, ar3, ar4, ar5], names=('RA', 'DEC', 'Flux', 'stype', 'ssize'),
                             meta={'name': output_name})
        output_table.write(output_name, overwrite=True)

    def output_gen(self, ra, dec, stype, ssize):
        nu = self.nu
        frame = 'fk5'
        tab = Table.read(self.table_name)

        # create the sky coordinate
        pos = SkyCoord(ra * u.degree, dec * u.degree, frame=frame)
        # make the SM object
        sm = SM(ha_file=os.path.join(datadir, 'Halpha_map.fits'),
                err_file=os.path.join(datadir, 'Halpha_error.fits'),
                nu=nu)
        # halpha
        val, err = sm.get_halpha(pos)
        tab.add_column(Column(data=val, name='Halpha'))
        tab.add_column(Column(data=err, name='err_Halpha'))
        # xi
        val, err = sm.get_xi(pos)
        tab.add_column(Column(data=val, name='xi'))
        tab.add_column(Column(data=err, name='err_xi'))
        # sm
        val, err = sm.get_sm(pos)
        tab.add_column(Column(data=val, name='sm'))
        tab.add_column(Column(data=err, name='err_sm'))
        # mod
        val, err = sm.get_m(pos, stype, ssize)
        tab.add_column(Column(data=val, name='m'))
        tab.add_column(Column(data=err, name='err_m'))
        # t0
        val, err = sm.get_timescale(pos)
        tab.add_column(Column(data=val, name='t0'))
        tab.add_column(Column(data=err, name='err_t0'))
        # rms
        val, err = sm.get_rms_var(pos, stype, ssize)
        tab.add_column(Column(data=val, name='rms1yr'))
        tab.add_column(Column(data=err, name='err_rms1yr'))
        # theta
        val, err = sm.get_theta(pos)
        tab.add_column(Column(data=val, name='theta_r'))
        tab.add_column(Column(data=err, name='err_theta_r'))
        tab.write(self.output_name, overwrite=True)
        return tab['m'], tab['t0']

    def areal_gen(self):
        flux, num = self.flux_gen()
        RA, DEC = self.region_gen(self.region_name)
        stype = self.stype_gen(RA)
        ssize = self.ssize_gen(flux, stype)
        mod, t0 = self.output_gen(RA, DEC, stype, ssize)
        obs_yrs = self.obs_time / (3600. * 24. * 365.25)
        mcount = 0
        var = []
        for i in range(0, len(t0) - 1):
            if obs_yrs <= t0[i]:
                mod[i] = mod[i] * (obs_yrs/t0[i])
        for i in range(0, len(mod)):
            if mod[i] >= self.mod_cutoff:
                mcount = mcount + 1
                var.append(mod[i])
        areal = mcount / self.area
        return areal, mod, var, len(flux), len(RA)

    def repeat(self):
        areal_arr = []
        count=0
        for i in range(0, self.loops):
            areal_arr.append(self.areal_gen()[0])
            count=count+1
        avg_areal = np.mean(areal_arr)
        return np.array(areal_arr), avg_areal, count


def test():
    sim=SIM()
    results=sim.repeat()
    print("Array: {0}".format(results[0]))
    print("Avg Areal: {0}".format(results[1]))
    print("Loops: {0}".format(results[2]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    test()