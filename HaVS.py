from __future__ import print_function, division
import os
import logging
import cPickle
import argparse
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table, Column
import astropy.units as u
from lib.new_SM17 import SM
from astropy.utils.exceptions import AstropyWarning
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
#warnings.filterwarnings("always")
#warnings.simplefilter('ignore', category=AstropyWarning)

datadir = os.path.join(os.path.dirname(__file__), 'data')
SFG=0
AGN=1
stypes=[SFG, AGN]
#sprobs=[0.84839, 1-0.84839] #0.15161
sprobs=[1-0.84839,0.84839]
parser = argparse.ArgumentParser()

parser.add_argument('-FUL', action='store', dest='FUL', default=1.,
                    help='Store upper flux limit (Jy)')
parser.add_argument('-FLL', action='store', dest='FLL', default=50e-3,
                    help='Store lower flux limit (Jy)')
parser.add_argument('-mc', action='store', dest='mc', default=0.05,
                    help='Store modulation cut off value')
parser.add_argument('-t', action='store', dest='obs_time', default=365.,
                    help='observation time in days')
parser.add_argument('-a', action='store', dest='a', default=3300.,
                    help='Scaling Constant for source counts')
#parser.add_argument('-scount', action='store', dest='scount', default=False, help='Number of sources')
parser.add_argument('-f', action='store', dest='nu', default=185.,
                    help='Frequency in MHz')
parser.add_argument('-i', action='store', dest='loops', default=20,
                    help='Number of iterations to run program through (30+ recommended)')
parser.add_argument('-reg', action='store', dest='region_name',
                    help='read in region file')
parser.add_argument('-map', action='store', dest='map', default=0,
                    help='Select old (0) or new (1) Ha maps')
parser.add_argument('--out', dest='outfile', default=False, type=str,
                        help="Output file name for results including file type (.csv)")
parser.add_argument('--fig', dest='figure', default=False,
                        help="Save Figure?")

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
        self.figure=results.figure
        self.nu = np.float(results.nu) * 1e6 #Hz, Default 185 MHz
        self.arcsec = np.pi / (180. * 3600.)
        self.mod_cutoff = np.float(results.mc) #Default 0.05
        self.low_Flim = np.float(results.FLL)  # Jy, Default 50e-3 Jy
        self.upp_Flim = np.float(results.FUL) # Jy, Default 1 Jy
        self.region_name = results.region_name
        region=cPickle.load(open(self.region_name, 'rb'))
        self.area = region.get_area(degrees=True)
        self.obs_time = np.float(results.obs_time) * 24. * 60. * 60. # seconds, Default 183 days
        self.loops=np.int(results.loops) #Default 20
        self.num_scale=40
        self.a=np.float(results.a) #Default 3300
        self.map=float(results.map)
        if self.map==1:
            self.ha_file = 'Ha_map_new.fits'
            self.err_file = 'Ha_err_new.fits'
        elif self.map==0:
            self.ha_file = 'Halpha_map.fits'
            self.err_file = 'Halpha_error.fits'

        #self.scount=float(results.scount)

    def flux_gen(self, alpha=0.8):
        """
        Function to distribute flux across all points
        Input:  Flux Density limit, RA/DEC positions, source distribution function
        Output: Flux for each RA/DEC point
        """

        def franz_counts(mids):
            a = [3.52, 0.307, -0.388, -0.0404, 0.0351, 0.006]
            source_counts = []
            for ii in range(0, len(mids)):
                x = (mids[ii])
                sum_counts = 0.
                for i in range(0, 6):
                    sum_counts = sum_counts + (a[i] * (np.log10(x)) ** i)
                sum_counts = 10 ** (sum_counts)
                source_counts.append(sum_counts)
            return (source_counts)

        def hopkins_counts(mids):
            a = [0.859, 0.508, 0.376, -0.049, -0.121, 0.057, -0.008]
            source_counts = []
            mids = mids * 1e3
            for ii in range(0, len(mids)):
                x = (mids[ii])
                sum_counts = 0.
                for i in range(0, 7):
                    sum_counts = sum_counts + (a[i] * (np.log10(x)) ** i)
                sum_counts = 10 ** (sum_counts)
                source_counts.append(sum_counts)
            return (source_counts)

        def linscale(f, f0, low=50e-3, upp=1., alpha=-0.8, inc=1e-3):
            freq_ratio = (np.float(f) / f0) ** (alpha)
            low_flux = np.float(low)
            upp_flux = np.float(upp)
            edges = np.arange(low_flux, upp_flux, inc)
            mids = []
            ds = []
            for i in range(0, len(edges) - 1):
                mids.append((edges[i + 1] + edges[i]) / 2.)
                ds.append(edges[i + 1] - edges[i])
            mids = np.array(mids)
            ds = np.array(ds)
            return mids, ds, edges, mids * freq_ratio, ds * freq_ratio, edges * freq_ratio, freq_ratio

        def scount(low, upp, alpha=0.8, freq=185e6, inc=1e-3):
            f0_f = 154e6
            f0_h = 1400e6
            low_f = 1e-3
            low_h = 0.05e-3
            upp_f = 75.
            upp_h = 1.

            franz_stats = linscale(freq, f0_f, low_f, upp_f, alpha, inc)
            mid_f, ds_f, edg_f, mids_f, fr_f = franz_stats[0], franz_stats[1], franz_stats[2], franz_stats[3], \
                                               franz_stats[6]
            hopkins_stats = linscale(freq, f0_h, low_h, upp_h, alpha, inc)
            mid_h, ds_h, edg_h, mids_h, fr_h = hopkins_stats[0], hopkins_stats[1], hopkins_stats[2], hopkins_stats[3], \
                                               hopkins_stats[6]
            counts_f = franz_counts(mid_f)
            counts_h = hopkins_counts(mid_h)

            normcounts_f = counts_f * ds_f * mid_f ** (-2.5)
            normcounts_h = counts_h * ds_h * mid_h ** (-2.5)

            sum_f = np.sum(normcounts_f)
            sum_h = np.sum(normcounts_h)

            if freq <= f0_f:
                sumcounts = sum_f
                mids = mid_f
                normcounts = normcounts_f
                edges = edg_f
            elif freq >= f0_h:
                sumcounts = sum_h
                mids = mid_h
                normcounts = normcounts_h
                edges = edg_h

            else:
                # print bins
                finc = inc / fr_f
                hinc = inc / fr_h
                franz_stats = linscale(freq, f0_f, low_f, upp_f, alpha, finc)
                mid_f, ds_f, edg_f, mids_f = franz_stats[0], franz_stats[1], franz_stats[2], franz_stats[3]
                hopkins_stats = linscale(freq, f0_h, low_h, upp_h, alpha, hinc)
                mid_h, ds_h, edg_h, mids_h = hopkins_stats[0], hopkins_stats[1], hopkins_stats[2], hopkins_stats[3]
                counts_f = franz_counts(mid_f)
                counts_h = hopkins_counts(mid_h)

                normcounts_f = counts_f * ds_f * mid_f ** (-2.5)
                normcounts_h = counts_h * ds_h * mid_h ** (-2.5)

                sum_f = np.sum(normcounts_f)
                sum_h = np.sum(normcounts_h)

                dF = np.abs(freq - f0_f)
                dH = np.abs(freq - f0_h)
                fw1 = 1 - (np.abs(dF) / (dF + dH))
                fw2 = 1 - (np.abs(dH) / (dF + dH))
                franz_upp = np.max(mids_f)
                franz_low = np.min(mids_f)
                hop_low = np.min(mids_h)
                hop_upp = np.max(mids_h)

                m1, m2, m3 = [], [], []
                n1, n2, n3 = [], [], []
                # OVERLAP
                maskff = np.where((mids_f >= hop_low) & (mids_f <= hop_upp))
                maskhh = np.where((mids_h >= franz_low) & (mids_h <= franz_upp))
                m2 = (mids_f[maskff] * fw1) + (mids_h[maskhh] * fw2)
                n2 = (normcounts_f[maskff] * fw1) + (normcounts_h[maskhh] * fw2)
                mask_lim = np.where((m2 >= low) & (m2 <= upp))
                m2 = m2[mask_lim]
                n2 = n2[mask_lim]
                if franz_upp < upp or franz_low > low or hop_upp < upp or hop_low > low:
                    if franz_low > low:
                        maskh = np.where((mids_h >= low) & (mids_h <= franz_low))
                        n1 = normcounts_h[maskh]
                        m1 = mids_h[maskh]
                    elif franz_upp < upp:
                        maskh = np.where((mids_h >= franz_upp) & (mids_h <= upp))
                        n3 = normcounts_h[maskh]
                    elif hop_low > low:
                        maskf = np.where((mids_f <= hopp_low) & (mids_f >= low))
                        n1 = normcounts_f[maskf]
                        m1 = mids_f[maskf]
                    elif hop_upp < upp:
                        maskf = np.where((mids_f <= upp) & (mids_f >= hop_upp))
                        n3 = normcounts_f[maskf]
                        m3 = mids_f[maskf]


                normcounts = np.concatenate([n1, n2, n3])
                mids = np.concatenate([m1, m2, m3])
                sumcounts = np.sum(normcounts)
                edges = mids - 0.5 * inc
            return normcounts, sumcounts, mids, edges
        norm_counts,  total_counts, mids, edges= scount(self.low_Flim, self.upp_Flim, 0.8, self.nu)
        Area = self.area * (np.pi ** 2.) / (180. ** 2.)
        FLUX = []
        num_sources = norm_counts * Area
        total_counts = total_counts * Area
        for i in range(0, len(edges) - 1):
            count = num_sources[i]
            p = count - int(count)
            leftover_count = np.random.choice([0, 1], p=[1 - p, p])
            count = int(count) + leftover_count
            FLUX.extend(np.random.uniform(edges[i], edges[i + 1], size=int(count)))
        flux_arr = np.random.permutation(np.array(FLUX))
        return flux_arr, len(flux_arr)

    def pos_gen(self):
        """
        A function to generate a number of random points in RA/DEC
        Input:  Number of points to generate from flux_gen function
        Output: List of RA/DEC (2,1) array in (2D) Cartesian coordiantes.
        """

        num_sources=self.flux_gen()[1]

        num=num_sources*100.
        lim = int(num * 2.5)
        x = []
        y = []
        z = []
        r = []
        i = 0
        #Generating cube
        x1 = np.array(np.random.uniform(-1.0, 1.0, lim))
        y1 = np.array(np.random.uniform(-1.0, 1.0, lim))
        z1 = np.array(np.random.uniform(-1.0, 1.0, lim))
        rad = (x1 ** 2.0 + y1 ** 2.0 + z1 ** 2.0) ** (0.5)
        #Getting points inside sphere of radius 1
        for i in range(0, len(rad)):
            if rad[i] <= 1.:
                x.append(x1[i])
                y.append(y1[i])
                z.append(z1[i])
                r.append(rad[i])
        x, y, z = np.array(x) / np.array(r), np.array(y) / np.array(r), np.array(z) / np.array(r)
        r0 = (x ** 2.0 + y ** 2.0 + z ** 2.0) ** 0.5
        #converting back to cartesian cooridantes
        theta = np.arccos(z / r0) * 180 / np.pi
        theta = theta - 90.
        theta = theta[:num_sources]
        phi = np.arctan2(y, x) * 180. / (np.pi)
        phi = phi + 180.
        phi = phi[:num_sources]
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
        flux, num=self.flux_gen()
        while len(reg_ind) < num:
            RA, DEC = self.pos_gen()
            reg_arr = region.sky_within(RA, DEC, degin=True)
            for i in range(0, len(reg_arr)):
                if len(reg_ra) < num:
                    reg_ind.append(i)
                    reg_ra.append(RA[i])
                    reg_dec.append(DEC[i])
        return np.array(reg_ra[:num]), np.array(reg_dec[:num]), flux, num

    def stype_gen(self,arr):
        """
        Function to determine if a source is of type compact or extended
        Input:  RA/DEC list (source_size?)
        Output: compact (1?) or extended (0?)
        """
        stype_arr = []
        for i in range(0, len(arr)):
            stype_arr.append(np.random.choice(stypes, p=sprobs))
        return stype_arr

    def ssize_gen(self, stype):
        """
        Generates source size based stype given.
        Input: Flux and source type
        Output: Source size
        """

        ssize_arr=[]
        for i in range(0, len(stype)):

            if stype[i] == AGN:
                #  milli arc secs
                ssize_arr.append((0.5*1e-3)/3600.) #(0.0979/(3600.)) actual values
            elif stype[i] == SFG:
                #  milli arc secs
                ssize_arr.append((10*1e-3)/3600.) #(0.2063/(3600.)) actual values

        return np.array(ssize_arr)


    def output_gen(self, ra, dec, ssize):
        """
        Function to use SM2017 to get Modulation, Timescale, Halpha, Theta and other values.
        Input: RA, DEC, Source Size
        Output: Modulation, Timescale, Halpha, Theta
        """
        nu = np.float(self.nu)
        frame = 'galactic'
        #frame='fk5'
        tab = Table()

        # create the sky coordinate
        pos = SkyCoord(ra * u.degree, dec * u.degree, frame=frame)
        # make the SM object

        sm = SM(ha_file=os.path.join(datadir, self.ha_file),
                err_file=os.path.join(datadir, self.err_file),
                nu=nu)
        # Halpha
        Ha, err_Ha = sm.get_halpha(pos)
        # xi
        #xi, err_xi = sm.get_xi(pos)
        # theta

        theta, err_theta = sm.get_theta(pos)
        #sm
        #sm, err_sm = sm.get_sm(pos)
        # mod

        m, err_m = sm.get_m(pos, ssize)
        # t0
        t0, err_t0 = sm.get_timescale(pos,ssize)
        # rms
        #val6, err6 = sm.get_rms_var(pos, stype, ssize)

        #tau
        #tau, err_tau=sm.get_tau(pos)
        tau=1
        err_tau=1
        return m, err_m, t0, err_t0, Ha, err_Ha , theta, err_theta, tau, err_tau

    def areal_gen(self):
        """
        Function to generate the areal sky density (ASD) values
        Uses: Flux, Region, Stype, Ssize, Output (Ha, mod, t0, theta), Obs_Yrs
        Output: ASD, modulation, timescale, Ha, Theta
        """
        RA, DEC, flux, num = self.region_gen(self.region_name)

        #print('RA')
        stype = self.stype_gen(RA)
        ssize = self.ssize_gen(stype)
        #print('SS')
        mod, err_m, t0, err_t0, Ha, err_Ha, theta, err_theta, tau, err_tau= self.output_gen(RA, DEC, ssize)
        obs_yrs = self.obs_time / (3600. * 24. * 365.25)
        t_mask=np.where(np.float(obs_yrs)<=t0)
        mod[t_mask] = mod[t_mask] * (np.float(obs_yrs)/ t0[t_mask])
        err_m[t_mask] = err_m[t_mask] * (np.float(obs_yrs) / t0[t_mask])

        mp = np.random.normal(loc=mod, scale=err_m)

        v_mask=np.where(mp*flux>=self.low_Flim*3.)
        m_mask=np.where(mp>=self.mod_cutoff)
        var_mask=np.where((mp*flux>=self.low_Flim*3.) & (mp>=self.mod_cutoff))

        vcount = len(v_mask[0])
        mcount = len(m_mask[0])
        varcount = len(var_mask[0])
        #print(vcount,mcount)
        #print(np.nanmean(theta*3600))
        mareal = float(mcount) / self.area
        vareal = float(vcount) / self.area
        varareal=float(varcount)/ self.area
        print(mcount, vcount, varcount)

        datatab1 = Table()
        mvar=int(self.map)
        datafile = self.region_name[8:-4] + '_test_19' +'_m{0}_data.csv'.format(mvar)
        #print('mod_mean',np.mean(mod))
        ### DATA FILE
        datatab1.add_column(Column(data=RA, name='RA'))
        datatab1.add_column(Column(data=DEC, name='DEC'))
        datatab1.add_column(Column(data=flux, name='flux'))
        datatab1.add_column(Column(data=Ha, name='H_Alpha'))
        datatab1.add_column(Column(data=err_Ha, name='H_Alpha err'))
        datatab1.add_column(Column(data=mod, name='Modulation'))
        datatab1.add_column(Column(data=err_m, name='Modulation err'))
        datatab1.add_column(Column(data=t0, name='Timescale'))
        datatab1.add_column(Column(data=err_t0, name='Timescale err'))
        datatab1.add_column(Column(data=theta, name='Theta'))
        datatab1.add_column(Column(data=err_theta, name='Theta err'))
        #datatab1.add_column(Column(data=tau, name='Tau'))
        #datatab1.add_column(Column(data=err_tau, name='Tau err'))
        datatab1.write(datafile, overwrite=True)
        return mareal, mp, t0, Ha, theta, flux, mareal, vareal, varareal

    def repeat(self):
        """
        Function to repeate the ASD calculation
        Input: Number of iterations set at beginning
        Output: Arrays of Modulation, Timescale, Halpha, Theta as well as other statistics.
        """
        areal_arr = []
        mod_arr = np.empty((self.loops,2))
        t0_arr = np.empty((self.loops,2))
        Ha_arr = np.empty((self.loops,2))
        theta_arr = np.empty((self.loops,2))
        NSources = []
        count = 0

        for i in range(0, self.loops):
            INPUT = self.areal_gen()
            areal_arr.append(INPUT[0])
            mod_arr[i,:]=[np.mean(INPUT[1]), np.std(INPUT[1])]
            t0_arr[i,:]=[np.mean(INPUT[2]), np.std(INPUT[2])]
            Ha_arr[i,:]=[np.mean(INPUT[3]), np.std(INPUT[3])]
            theta_arr[i,:]=[np.mean(INPUT[4]), np.std(INPUT[4])]
            count=count+1
            NSources.append(len(INPUT[1]))
        areal_arr = np.array(areal_arr)
        NSources = np.array(NSources)

        return areal_arr, mod_arr,t0_arr, Ha_arr, theta_arr, count, NSources, self.area, self.low_Flim, self.upp_Flim, self.obs_time, self.nu, self.mod_cutoff


def test():
    """
    This section collects runs the previous functions and outputs them to two different files.
    Data file: Includes raw data from each iteration.
    Results file: Returns averaged results.
    """

    sim=SIM()
    areal_arr, mod_arr, t0_arr, Ha_arr, theta_arr, count, NSources, area, low_Flim, upp_Flim, obs_time, nu, mod_cutoff=sim.repeat()
    datatab=Table()
    resultstab=Table()
    if outfile != False:
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
        Stats = ['H_Alpha Mean','H_Alpha STD', 'Modulation Mean', 'Modulation STD', 'Timescale Mean (yrs)', 'Timescale STD (yrs)',
               'Theta Mean (deg)', 'Theta STD (deg)', 'Areal Sky Desnity Mean',  'Areal Sky Desnity STD']
        Stats_vals = [np.mean(Ha_arr[:,0]),np.std(Ha_arr[:,0]),np.mean(mod_arr[:,0]),np.std(mod_arr[:,0]),
                    np.mean(t0_arr[:,0]),np.std(t0_arr[:,0]), np.mean(theta_arr[:,0]),np.std(theta_arr[:,0]),
                    np.mean(areal_arr), np.std(areal_arr)]
        Params = ['Avg # Sources', 'Avg Variables','Area (deg^2)', 'Lower Flux Limit (Jy)', 'Upper Flux Limit (Jy)', 'Observation time (days)', 'Frequency (MHz)', 'Modulation Cutoff']
        Params.extend(["",""])
        Param_vals =[ np.mean(NSources),area*np.mean(areal_arr), area, low_Flim, upp_Flim, obs_time/(24.*3600.), nu/(1E6), mod_cutoff]
        Param_vals.extend(["", ""])

        resultstab.add_column(Column(data=Stats, name='Statistics'))
        resultstab.add_column(Column(data=Stats_vals, name='Results'))
        resultstab.add_column(Column(data=Params, name='Parameters'))
        resultstab.add_column(Column(data=Param_vals, name='Values'))
        resultstab.write(resultsfile, overwrite=True)
    if outfile == False:
        print("Array: {0}".format(areal_arr))
        print("Avg Areal: {0}".format(np.mean(areal_arr)))
        print("Iterations: {0}".format(len(areal_arr)))
        print("Num Sources: {0}".format(np.mean(NSources)))
        print("Area: {0}".format(area))
        print("Num Variable: {0}".format(np.mean(areal_arr)*area))
        print("% Variable: {0}".format(np.mean(areal_arr) * area*100./np.mean(NSources)))

test()
