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
sprobs=[0.84839, 1-0.84839]

parser = argparse.ArgumentParser()

parser.add_argument('-FUL', action='store', dest='FUL', default=1.,
                    help='Store upper flux limit (Jy)')
parser.add_argument('-FLL', action='store', dest='FLL', default=0.001,
                    help='Store lower flux limit (Jy)')
parser.add_argument('-mc', action='store', dest='mc', default=0.05,
                    help='Store modulation cut off value')
parser.add_argument('-t', action='store', dest='obs_time', default=183,
                    help='observation time in days')
parser.add_argument('-a', action='store', dest='a', default=1150,
                    help='Scaling Constant for source counts')
parser.add_argument('-f', action='store', dest='nu', default=185.,
                    help='Frequency in MHz')
parser.add_argument('-i', action='store', dest='loops', default=20,
                    help='Number of iterations to run program through (30+ recommended)')
parser.add_argument('-reg', action='store', dest='region_name',
                    help='read in region file')
parser.add_argument('--out', dest='outfile', default=False, type=str,
                        help="Table of results")
parser.add_argument('--version', action='version', version='%(prog)s 1.0')

results = parser.parse_args()
outfile=results.outfile
class SIM(object):
    def __init__(self, log=None):


        if log is None:
            logging.basicConfig(format="%(module)s:%(levelname)s %(message)s")
            self.log = logging.getLogger("SIM_new")
            self.log.setLevel(logging.DEBUG)
        else:
            self.log=log
        #Variables
        self.nu = np.float(results.nu) * 1e6
        self.arcsec = np.pi / (180. * 3600.)
        self.mod_cutoff = np.float(results.mc)
        self.low_Flim = np.float(results.FLL)  # Jy
        self.upp_Flim = np.float(results.FUL) # Jy
        self.region_name = results.region_name
        #self.region_name=('testreg.mim')
        region=cPickle.load(open(self.region_name, 'rb'))
        self.area = region.get_area(degrees=True)
        print(self.area)
        self.obs_time = np.float(results.obs_time) * 24. * 60. * 60.
        self.loops=np.int(results.loops)
        self.num_scale=40
        self.a=results.a

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
            low=np.log10(low)
            upp=np.log10(upp)
            bins = np.logspace(low, upp, num=100)
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
            dNdS= Ni / dS  # * mpoint**2.5
            return bins, dNdS, mpoint, Ni, dS

        a = self.a
        bins, dnds,mid,Ni,width=diff_counts(a,1.6,low,upp)
        FLUX = []
        Area = self.area * (np.pi ** 2.) / (180. ** 2.)

        for i in range(0, len(bins) - 1):
            rang = np.logspace(np.log10(bins[i]), np.log10(bins[i]+width[i]), dtype=float)
            FLUX.extend(np.random.choice(rang, size=int(Ni[i]*Area)))
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

        ssize_arr = []
        for i in range(0, len(stype)):
            if stype[i] == AGN:
                ssize_arr.append(0.25/(3600.*1000.)) #(0.0979/(3600.)) actual values
            elif stype[i] == SFG:
                ssize_arr.append(30./(3600.*1000.)) #(0.2063/(3600.)) actual values

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
        tab = Table()

        # create the sky coordinate
        pos = SkyCoord(ra * u.degree, dec * u.degree, frame=frame)
        # make the SM object
        sm = SM(ha_file=os.path.join(datadir, 'Halpha_map.fits'),
                err_file=os.path.join(datadir, 'Halpha_error.fits'),
                nu=nu)
        """
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
        #tab.write(self.output_name, overwrite=True)
        return tab['m'], tab['t0'], tab['Halpha'], tab['theta_r']
        """
        # Ha
        val1, err1 = sm.get_halpha(pos)
        # xi
        #val2, err2 = sm.get_xi(pos)
        # sm
        #val3, err3 = sm.get_sm(pos)
        # mod
        val4, err4 = sm.get_m(pos, stype, ssize)
        # t0
        val5, err5 = sm.get_timescale(pos)
        # rms
        #val6, err6 = sm.get_rms_var(pos, stype, ssize)
        # theta
        val7, err7 = sm.get_theta(pos)
        return val4, val5, val1, val7
    def areal_gen(self):
        flux, num = self.flux_gen()
        #print('flux', flux, num)
        RA, DEC = self.region_gen(self.region_name)
        #print('RA')
        stype = self.stype_gen(RA)
        print(np.sum(stype))
        ssize = self.ssize_gen(flux, stype)
        #print('SS')
        mod, t0, Ha, theta= self.output_gen(RA, DEC, stype, ssize)
        #print('mod')
        obs_yrs = self.obs_time / (3600. * 24. * 365.25)
        mcount = 0
        var = []
        for i in range(0, len(t0) - 1):
            if obs_yrs <= t0[i]:
                print(mod[i])
                mod[i] = mod[i] * (np.float(obs_yrs/t0[i]))
                print(obs_yrs, t0[i], mod[i])
        for i in range(0, len(mod)):
            if mod[i] >= self.mod_cutoff:
                mcount = mcount + 1
                var.append(mod[i])
        print(mcount, len(var))
        areal = mcount / self.area
        print(self.area)

        return areal, mod, t0, Ha, theta

    def repeat(self):
        areal_arr = []
        mod_arr=np.empty((self.loops,2))
        t0_arr=np.empty((self.loops,2))
        Ha_arr=np.empty((self.loops,2))
        theta_arr=np.empty((self.loops,2))
        NSources=[]
        count=0

        for i in range(0, self.loops):
            INPUT = self.areal_gen()
            areal_arr.append(INPUT[0])
            mod_arr[i,:]=[np.mean(INPUT[1]), np.std(INPUT[1])]
            t0_arr[i,:]=[np.mean(INPUT[2]), np.std(INPUT[2])]
            Ha_arr[i,:]=[np.mean(INPUT[3]), np.std(INPUT[3])]
            theta_arr[i,:]=[np.mean(INPUT[4]), np.std(INPUT[4])]
            count=count+1
            NSources.append(len(INPUT[1]))
        areal_arr=np.array(areal_arr)
        NSources= np.array(NSources)
        return areal_arr, mod_arr,t0_arr, Ha_arr, theta_arr, count, NSources, self.area, self.low_Flim, self.upp_Flim, self.obs_time, self.nu


def test():
    sim=SIM()
    areal_arr, mod_arr, t0_arr, Ha_arr, theta_arr, count, NSources, area, low_Flim, upp_Flim, obs_time, nu=sim.repeat()
    datatab=Table()
    resultstab=Table()
    if outfile!= False:
        datafile=outfile[:-4]+'_data'+outfile[-4:]
        ### DATA FILE
        datatab.add_column(Column(data=np.arange(1,len(areal_arr)+1,1), name='Interations'))
        datatab.add_column(Column(data=Ha_arr[:,0], name='H_Alpha Mean'))
        datatab.add_column(Column(data=Ha_arr[:,1], name='H_Alpha STD'))
        datatab.add_column(Column(data=mod_arr[:,0], name='Modulation Mean'))
        datatab.add_column(Column(data=mod_arr[:,1], name='Modulation STD'))
        datatab.add_column(Column(data=t0_arr[:,0], name='Timescale Mean'))
        datatab.add_column(Column(data=t0_arr[:,1], name='Timescale STD'))
        datatab.add_column(Column(data=theta_arr[:,0], name='Theta Mean'))
        datatab.add_column(Column(data=theta_arr[:,1], name='Theta STD'))
        datatab.add_column(Column(data=areal_arr, name='Areal Sky Density'))
        datatab.write(datafile, overwrite=True)

        ##RESUTLS FILE
        resultsfile = outfile[:-4] + '_results' + outfile[-4:]
        Stats=['H_Alpha Mean','H_Alpha STD', 'Modulation Mean', 'Modulation STD', 'Timescale Mean (yrs)', 'Timescale STD (yrs)',
               'Theta Mean (deg)', 'Theta STD (deg)', 'Areal Sky Desnity Mean',  'Areal Sky Desnity STD']
        Stats_vals=[np.mean(Ha_arr[:,0]),np.std(Ha_arr[:,0]),np.mean(mod_arr[:,0]),np.std(mod_arr[:,0]),
                    np.mean(t0_arr[:,0]),np.std(t0_arr[:,0]), np.mean(theta_arr[:,0]),np.std(theta_arr[:,0]),
                    np.mean(areal_arr), np.std(areal_arr)]
        Params=['Avg # Sources', 'Area (deg^2)', 'Lower Flux Limit (Jy)', 'Upper Flux Limit (Jy)', 'Observation time (days)', 'Frequency (MHz)']
        Params.extend(["","","",""])
        Param_vals=[np.mean(NSources), area, low_Flim, upp_Flim, obs_time, nu]
        Param_vals.extend(["", "", "", ""])

        resultstab.add_column(Column(data=Stats, name='Statistics'))
        resultstab.add_column(Column(data=Stats_vals, name='Results'))
        resultstab.add_column(Column(data=Params, name='Parameters'))
        resultstab.add_column(Column(data=Param_vals, name='Values'))
        resultstab.write(resultsfile, overwrite=True)

    print("Array: {0}".format(areal_arr))
    print("Avg Areal: {0}".format(np.mean(areal_arr)))
    print("Iterations: {0}".format(len(areal_arr)))
    print("Num Sources: {0}".format(np.mean(NSources)))


test()
