import imp
import numpy as np
import astropy
from astropy.io import fits
from astropy.wcs import WCS
from astropy.constants import kpc, c
from astropy.coordinates import SkyCoord
import astropy.units as u
SM = imp.load_source('SM', '/home/elliottcharlton/PycharmProjects/SM2017/lib/SM2017.py')
hdulist = fits.open('/home/elliottcharlton/PycharmProjects/SM2017/data/Ha_map_new.fits')
data = hdulist[0].data
wcs = WCS(hdulist[0].header, naxis=2)
d, r = 0, 0
point= SkyCoord(r, d,  frame='galactic', unit=(u.degree, u.degree))

def SMF(r,d, dist=False, size=False, ang=False, t=False, freq=185e6, modthresh=0.05, figure=False, figname=False):
    point = SkyCoord([r] * u.degree, [d] * u.degree)
    sm = SM.SM('/home/elliottcharlton/PycharmProjects/SM2017/data/Ha_map_new.fits','/home/elliottcharlton/PycharmProjects/SM2017/data/Ha_err_new.fits',nu=freq)
    #sm = TAU.SM('/home/elliottcharlton/PycharmProjects/SM2017/data/Ha_map_new.fits','/home/elliottcharlton/PycharmProjects/SM2017/data/Ha_err_new.fits',nu=freq)
    if dist & size == False:
        if ang == False:
            print('Input Distance and Linear size of source or its Angular Resolution, using defualt angular size of 1 milli arcsec')
            ang=1e-3/3600.
        if ang != False:
            ssize=float(ang)
    elif dist & size != False:
        ang = np.degrees(size/dist)
    ssize=float(ang)
    mod, err_mod=sm.get_m(point, ssize)
    if t == False:
        tlen=np.linspace(1e-5,1e2, 10000)
        tarr=sm.get_rms_var(point,ssize,tlen)[0]
        tarr1=sm.get_rms_var(point,ssize*1.3,tlen)[0]
        tarr2=sm.get_rms_var(point,ssize*0.7,tlen)[0]
        tres=np.array([tarr,tarr1,tarr2])
        
    elif t != False:
        tres=sm.get_rms_var(point,ssize,t)     
    tmod=tlen[np.where(tres[0]>=modthresh)]
    print 'Timescale for {0}% modulation = {1} years'.format(int(modthresh*100),round(tmod[0],3))    
    return mod, tres,tlen, tmod[0]
