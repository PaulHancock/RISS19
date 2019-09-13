from astropy.constants import kpc, c
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
import numpy as np
import os
import logging
from scipy.special import gamma

class SM(object):
    """
    :param ha_file:
    :param err_file:
    :param nu: freq in Hz
    :param d: distance in kpc
    :param v: in m/s
    :param log:
    """
    def __init__(self, ha_file, err_file=None, nu=185e6, log=None, d=1, v=10e3):

        if log is None:
            logging.basicConfig(format="%(module)s:%(levelname)s %(message)s")
            self.log = logging.getLogger("SM2017")
            self.log.setLevel(logging.DEBUG)
        else:
            self.log=log

        # define some of the constants that we need
        # i'm saving these here to allow for different instances to have different values
        self.nu = nu  # Hz
        self.kpc = kpc.value  # in m
        self.t4 = 0.8  # t/1e4 K
        self.lo = 1e18/(self.kpc*1e-3)  # pc
        self.eps = 1
        self.D = d  # kpc - distance to the screen
        self.c = c.value
        self.beta = 11/3
        self.re = 2.817e-15  # m
        self.rf = np.sqrt(self.c * self.D * self.kpc / (2*np.pi*self.nu))  # Fresnel scale
        self.v = v  # relative velocity of source/observer in m/s
        #self.log.debug("data:{0} err:{1}".format(ha_file,err_file))
        self.file = ha_file
        self.err_file = err_file
        self._load_file()
		#self.tau_file= tau_file

    def _load_file(self):
        self.hdu = fits.getheader(self.file, ignore_missing_end=True)
        self.wcs = WCS(self.hdu)
        self.data = fits.open(self.file, memmap=True, ignore_missing_end=True)[0].data
		#self.tau = fits.open(self.tau_file, memmap=True, ignore_missing_end=True)[0].data
        if self.err_file:
            self.err_hdu = fits.getheader(self.err_file,ignore_missing_end=True)
            self.err_wcs = WCS(self.err_hdu)
            self.err_data = fits.open(self.err_file, memmap=True, ignore_missing_end=True)[0].data
        else:
            self.err_hud = self.err_wcs = self.err_data = None

        return

    def get_distance(self,position):
        """
        :param position: sky position
        :return: Distance to scattering screen in kpc
        """
        gal_r = 40.  # kpc
        sun_r = 8.   # kpc
        gal_h = 1.   # kpc
        theta = position.galactic.l.radian  # angle from the GC along the plane
        phi = position.galactic.b.radian  # angle from the GC perp to the plane
        far_edge = sun_r*np.cos(theta) + np.sqrt(gal_r**2. - sun_r**2.*np.sin(theta)**2.)
        top =  (gal_h/2. / np.abs(np.sin(phi)))
        mask = np.where(top>far_edge)
        screen_dist = top
        if len(mask[0])>=1:
            screen_dist[mask] = far_edge[mask]
        
        return screen_dist/2.


 

    def get_halpha(self, position):
        """
        Return the Halpha for a given location on the sky.
        :param position: astropy.coordinates.SkyCoord
        :return:
        """
        # The coordinates we request need to be the same as that in the WCS header
        # for the files in this repo, this currently means galactic coordinates.
        x, y = zip(*self.wcs.all_world2pix(zip(position.galactic.l.degree, position.galactic.b.degree), 0))
        x = np.int64(np.floor(x))
        x = np.clip(x, 0, self.hdu['NAXIS1'])
        y = np.int64(np.floor(y))
        y = np.clip(y, 0, self.hdu['NAXIS2'])
        iha = self.data[y, x]
        err_iha = self.err_data[y, x]
        return iha, err_iha

    def get_tau(self, position):
        """
        Return the Halpha for a given location on the sky.
        :param position: astropy.coordinates.SkyCoord
        :return:
        """
        # The coordinates we request need to be the same as that in the WCS header
        # for the files in this repo, this currently means galactic coordinates.
        x, y = zip(*self.wcs.all_world2pix(zip(position.galactic.l.degree, position.galactic.b.degree), 0))
        x = np.int64(np.floor(x))
        x = np.clip(x, 0, self.hdu['NAXIS1'])
        y = np.int64(np.floor(y))
        y = np.clip(y, 0, self.hdu['NAXIS2'])
        tau = self.data[y, x]
		#m = tau ** (3. / 2.)
        return tau

	def get_m(self, position):
		tau=self.get_tau(position)
		m=tau**(3./2.)
		return m



sm = SM(os.path.join('data', 'Ha_map_new.fits'), os.path.join('data', 'Ha_err_new.fits'), nu=1e8)
pos = SkyCoord([0, 4, 8, 12, 16, 20]*u.hour, [-90, -45, 0, 45, 90, -26]*u.degree)
print pos

print sm.get_tau(pos)




	
