from __future__ import print_function, division
import os
import logging
import cPickle
import argparse
import numpy as np
import numpy.polynomial.polynomial as poly
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
#sprobs=[1-0.84839,0.84839]
#Chetri2017 Strong Scint
#sprobs=[1/-37./347.,37./347.]
#Chetri2017 Strong + Mod Scint
sprobs=[1.-(37.+91.)/347.,(37.+91.)/347.]


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
        self.alpha=-0.8
        if self.map==1:
            self.ha_file = 'Ha_map_new.fits'
            self.err_file = 'Ha_err_new.fits'
        elif self.map==0:
            self.ha_file = 'Halpha_map.fits'
            self.err_file = 'Halpha_error.fits'

        #self.scount=float(results.scount)

    def flux_gen(self):
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

        def linscale(freq, f0, low=50e-3, upp=1., alpha=-0.8, inc=1e-4):
            edges = np.arange(low, upp, inc)
            mids = []
            ds = []
            for i in range(0, len(edges) - 1):
                mids.append((edges[i + 1] + edges[i]) / 2.)
                ds.append(edges[i + 1] - edges[i])
            mids = np.array(mids)
            ds = np.array(ds)
            return mids, ds, edges

        def fran_gen(freq=185e6, alpha=-0.8, inc=1e-4):
            f0 = 154e6
            low = 1e-3
            upp = 75.
            fr = ((freq * 1.0) / f0) ** (alpha)
            stats = linscale(freq, f0, low, upp, alpha, inc / fr)
            mid, ds, edg = stats
            counts = franz_counts(mid)
            numcounts = counts * ds * mid ** (-2.5)
            return mid * fr, numcounts, np.sum(numcounts), fr

        def hop_gen(freq=185e6, alpha=-0.8, inc=1e-4):
            f0 = 1400e6
            low = 0.05e-3
            upp = 1.
            fr = ((freq * 1.0) / f0) ** (alpha)
            stats = linscale(freq, f0, low, upp, alpha, inc / fr)
            mid, ds, edg = stats
            counts = hopkins_counts(mid)
            numcounts = counts * ds * mid ** (-2.5)
            return mid * fr, numcounts, np.sum(numcounts), fr

        def weight(freq=185e6, alpha=-0.8, inc=1e-4):
            fmid, fcounts, ftotal, fratio = fran_gen(freq, alpha=alpha, inc=inc)
            hmid, hcounts, htotal, hratio = hop_gen(freq, alpha=alpha, inc=inc)
            f0_f = 154e6
            f0_h = 1400e6
            # WEIGHTING
            dF = np.abs(freq - f0_f)
            dH = np.abs(freq - f0_h)
            fw1 = 1. - (np.abs(dF) / (dF + dH))
            fw2 = 1. - (np.abs(dH) / (dF + dH))
            if freq <= 154e6:
                mids = np.array(fmid)
                ncounts = np.array(fcounts)
            elif freq >= 1400e6:
                mids = np.array(hmid)
                ncounts = np.array(hcounts)
            else:
                franz_upp = np.max(fmid)
                franz_low = np.min(fmid)
                hop_low = np.min(hmid)
                hop_upp = np.max(hmid)

                # OVERLAP
                maskff = np.where((fmid >= hop_low) & (fmid <= hop_upp))
                maskhh = np.where((hmid >= franz_low) & (hmid <= franz_upp))
                m1 = (fmid[maskff] * fw1) + (hmid[maskhh] * fw2)
                n1 = (fcounts[maskff] * fw1) + (hcounts[maskhh] * fw2)
                # OUTER EDGES
                maskf1 = np.where(fmid < hop_low)
                m2 = fmid[maskf1]
                n2 = fcounts[maskf1]
                maskf2 = np.where(fmid > hop_upp)
                m4 = fmid[maskf2]
                n4 = fcounts[maskf2]

                maskh1 = np.where(hmid < franz_low)
                m3 = hmid[maskh1]
                n3 = hcounts[maskh1]
                maskh2 = np.where(hmid > franz_upp)
                m5 = hmid[maskh2]
                n5 = hcounts[maskh2]

                ncounts = np.concatenate([n2, n3, n1, n4, n5])
                mids = np.concatenate([m2, m3, m1, m4, m5])
                mids = np.array(mids)
                ncounts = np.array(ncounts)

            return mids, ncounts, np.sum(ncounts), fmid, hmid, fcounts, hcounts, ftotal, htotal

        def limit(low, upp, freq=185e6, alpha=-0.8, inc=1e-4):
            mids, ncounts, tcounts, fmid, hmid, fcounts, hcounts, ftotal, htotal = weight(freq, alpha, inc)
            x = np.log10(mids)
            y = np.log10(ncounts)
            xf = np.log10(fmid)
            yf = np.log10(fcounts)
            xh = np.log10(hmid)
            yh = np.log10(hcounts)
            deg = 15
            z = poly.polyfit(x, y, deg=deg)
            zf = poly.polyfit(xf, yf, deg=deg)
            zh = poly.polyfit(xh, yh, deg=deg)
            x0 = np.arange(low, upp, inc)
            x0 = np.log10(x0)
            p = 10 ** poly.polyval(x0, z)
            pf = 10 ** poly.polyval(x0, zf)
            ph = 10 ** poly.polyval(x0, zh)
            x0 = 10 ** x0
            mids, edges, fmid, hmid = x0[:-1] + inc, x0, x0[:-1] + inc, x0[:-1] + inc
            ncounts, fcounts, hcounts = p[:-1], pf[:-1], ph[:-1]
            return mids, ncounts, np.sum(ncounts), edges

        mids, norm_counts, total_counts, edges= limit(self.low_Flim, self.upp_Flim, self.nu, self.alpha)
        Area = self.area * (np.pi ** 2.) / (180. ** 2.)
        FLUX = []
        num_sources = norm_counts * Area
        tcounts = total_counts * Area
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
        num=num_sources*200.
        lim = int(num)
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
        theta = theta
        phi = np.arctan2(y, x) * 180. / (np.pi)
        phi = phi + 180.
        phi = phi
        return np.array(phi), np.array(theta)

    def region_gen(self, reg_file):
        """
        Takes in a list of positions and removes points outside the MIMAS region
        Input:  RA/DEC positions and MIMAS region file.
        Output: List of RA/DEC inside the correct region.
        """

        # reg_ra = []
        # reg_dec = []
        # region = cPickle.load(open(reg_file, 'rb'))
        # flux, num=self.flux_gen()
        # while len(reg_ra) < num:
        #     RA, DEC = self.pos_gen()
        #     reg_arr = region.sky_within(RA, DEC, degin=True)
        #     reg_ra.extend(RA[reg_arr])
        #     reg_dec.extend(DEC[reg_arr])

        reg_ra = []
        reg_dec = []
        region = cPickle.load(open(reg_file, 'rb'))
        flux, num = self.flux_gen()
        while len(reg_ra) < num:
            RA, DEC = self.pos_gen()
            c = SkyCoord(l=RA*u.degree, b=DEC*u.degree, frame='galactic')
            ra=c.fk5.ra.deg
            dec=c.fk5.dec.deg
            reg_arr = region.sky_within(ra, dec, degin=True)
            print(max(RA), max(DEC))
            #for i in range(0, len(reg_arr)):
            reg_ra.extend(RA[reg_arr])
            reg_dec.extend(DEC[reg_arr])
        print(max(reg_ra), max(reg_dec))
        reg_dec= np.array(reg_dec[:num])
        reg_ra = np.array(reg_ra[:num])
        return reg_ra, reg_dec, flux, num

    def stype_gen(self):
        """
        Function to determine if a source is of type compact or extended
        Input:  RA/DEC list (source_size?)
        Output: compact (1?) or extended (0?)
        """
        ra,dec,flux=self.region_gen(self.region_name)[0:3]
        arr=ra

        stype_arr=(np.random.choice(stypes, p=sprobs, size=len(arr)))

        return stype_arr, ra, dec, flux

    def ssize_gen(self):
        """
        Generates source size based stype given.
        Input: Flux and source type
        Output: Source size
        """
        stype,ra, dec, flux=self.stype_gen()
        def ang_size(flux, freq, alpha=-0.8):
            f0 = 1400e6
            flux = np.array(flux)
            fr = ((freq * 1.0) / f0) ** (alpha)
            Sn = (flux) * fr
            a = 2. * Sn ** 0.3
            return a/3600., Sn
        ssize_arr=ang_size(flux,freq=self.nu)[0]
        ssize_arr=np.array(ssize_arr)
        agn_mask = np.where(stype == 1)


        if len(agn_mask[0])>=1:
            agn_ssize=(1e-3) / 3600.
            ssize_arr[agn_mask]=agn_ssize

        return ssize_arr, stype, ra, dec ,flux


    def output_gen(self):
        """
        Function to use SM2017 to get Modulation, Timescale, Halpha, Theta and other values.
        Input: RA, DEC, Source Size
        Output: Modulation, Timescale, Halpha, Theta

        """
        ssize, stype, ra, dec ,flux=self.ssize_gen()
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
        print(max(m))
        return m, err_m, t0, err_t0, Ha, err_Ha , theta, err_theta, tau, err_tau,ssize, stype, ra, dec ,flux

    def areal_gen(self):
        """
        Function to generate the areal sky density (ASD) values
        Uses: Flux, Region, Stype, Ssize, Output (Ha, mod, t0, theta), Obs_Yrs
        Output: ASD, modulation, timescale, Ha, Theta
        """

        #stype = self.stype_gen()
        #ssize = self.ssize_gen()

        mod, err_m, t0, err_t0, Ha, err_Ha, theta, err_theta, tau, err_tau,ssize, stype, RA, DEC ,flux= self.output_gen()
        obs_yrs = self.obs_time / (3600. * 24. * 365.25)
        t_mask=np.where(np.float(obs_yrs)<=t0)
        mod[t_mask] = mod[t_mask] * (np.float(obs_yrs)/ t0[t_mask])
        err_m[t_mask] = err_m[t_mask] * (np.float(obs_yrs) / t0[t_mask])

        #mp = np.random.normal(loc=mod, scale=err_m)
        print(np.max(mod))
        mp= np.random.uniform(low=mod-err_m, high= mod+err_m)
        print(np.max(mp))
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
        return varareal, mp, t0, Ha, theta, flux, mareal, vareal, varareal,ssize, stype, RA, DEC ,flux

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
            varareal, mp, t0, Ha, theta, flux, mareal, vareal, varareal, ssize, stype, RA, DEC, flux = INPUT
            areal_arr.append(INPUT[0])
            mod_arr[i,:]=[np.mean(INPUT[1]), np.std(INPUT[1])]
            t0_arr[i,:]=[np.mean(INPUT[2]), np.std(INPUT[2])]
            Ha_arr[i,:]=[np.mean(INPUT[3]), np.std(INPUT[3])]
            theta_arr[i,:]=[np.mean(INPUT[4]), np.std(INPUT[4])]
            count=count+1
            NSources.append(len(INPUT[1]))
        areal_arr = np.array(areal_arr)
        NSources = np.array(NSources)

        return areal_arr, mod_arr,t0_arr, Ha_arr, theta_arr, count, NSources, self.area, self.low_Flim, self.upp_Flim, self.obs_time, self.nu, self.mod_cutoff,ssize, stype, RA, DEC ,flux


def test():
    """
    This section collects runs the previous functions and outputs them to two different files.
    Data file: Includes raw data from each iteration.
    Results file: Returns averaged results.
    """

    sim=SIM()
    areal_arr, mod_arr, t0_arr, Ha_arr, theta_arr, count, NSources, area, low_Flim, upp_Flim, obs_time, nu, mod_cutoff,ssize, stype, RA, DEC ,flux=sim.repeat()
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
        print("Area: {0}".format(np.round(area,2)))
        print("Num Variable: {0}".format(np.mean(areal_arr)*area))
        print("% Variable: {0}".format(np.mean(areal_arr) * area*100./np.mean(NSources)))
        print("Avg Modulation: {0}".format(np.round(np.mean(mod_arr),5)))
        print("Avg TScatt: {0}".format(np.round(np.mean(theta_arr),5)))
        print("Avg Source Size: {0}".format(np.round(np.mean(ssize),5)))


test()
