from .math_module import xp, xcipy, ensure_np_array
from scoob_llowfsc import utils
from scoob_llowfsc.imshows import imshow1, imshow2, imshow3

import numpy as np
import scipy
import astropy.units as u
from astropy.io import fits
import poppy
import time
import copy
import os
from pathlib import Path
from IPython.display import clear_output
from datetime import datetime
today = int(datetime.today().strftime('%Y%m%d'))

try:
    from scoobpy import utils as scoob_utils
    import purepyindi
    import purepyindi2
    from magpyx.utils import ImageStream
    import ImageStreamIOWrap as shmio
except ImportError:
    print('SCoOB interface does not have the required packages to operate.')

def create_shmim(name, dims, dtype=shmio.ImageStreamIODataType.FLOAT, shared=1, nbkw=8):
    # if ImageStream objects didn't auto-open on creation, you could create and return that instead. oops.
    img = shmio.Image()
    # not sure if I should try to destroy first in case it already exists
    # img.create(name, dims, dtype, shared, nbkw)
    buffer = np.zeros(dims)
    img.create(name, buffer, -1, True, 8, 1, dtype, 1)

def move_psf(x_pos, y_pos, client):
    client.wait_for_properties(['stagepiezo.stagepupil_x_pos', 'stagepiezo.stagepupil_y_pos'])
    scoob_utils.move_relative(client, 'stagepiezo.stagepupil_x_pos', x_pos)
    time.sleep(0.25)
    scoob_utils.move_relative(client, 'stagepiezo.stagepupil_y_pos', y_pos)
    time.sleep(0.25)

def home_block(client, delay=2):
    client.wait_for_properties(['stagelinear.home'])
    client['stagelinear.home.request'] = purepyindi.SwitchState.ON
    time.sleep(delay)

def move_block_in(client, delay=2):
    client.wait_for_properties(['stagelinear.presetName'])
    client['stagelinear.presetName.block_in'] = purepyindi.SwitchState.ON
    time.sleep(delay)

def move_block_out(client, delay=2):
    client.wait_for_properties(['stagelinear.presetName'])
    client['stagelinear.presetName.block_out'] = purepyindi.SwitchState.ON
    time.sleep(delay)

def set_camsci_roi(xc, yc, npix, client, delay=0.25):
    # update roi parameters
    client.wait_for_properties(['camsci.roi_region_x', 'camsci.roi_region_y', 
                                'camsci.roi_region_h' ,'camsci.roi_region_w', 
                                # 'camsci.roi_region_bin_x' ,'camsci.roi_region_bin_y', 
                                'camsci.roi_set'])
    client['camsci.roi_region_x.target'] = xc
    client['camsci.roi_region_y.target'] = yc
    client['camsci.roi_region_h.target'] = npix
    client['camsci.roi_region_w.target'] = npix
    time.sleep(delay)
    client['camsci.roi_set.request'] = purepyindi.SwitchState.ON
    time.sleep(delay)

def set_fib_atten(value, client, delay=0.1):
    client['fiberatten.atten.target'] = value
    time.sleep(delay)
    print(f'Set the fiber attenuation to {value:.1f}')

def set_camsci_exp_time(exp_time, client, delay=0.25):
    if exp_time<3.2e-5:
        print('Minimum exposure time is 3.2E-5 seconds. Setting exposure time to minimum.')
        exp_time = 3.2e-5
    client.wait_for_properties(['camsci.exptime'])
    client['camsci.exptime.target'] = exp_time
    time.sleep(delay)
    print(f'Set the CAMSCI exposure time to {exp_time:.2e}s')

def set_camsci_gain(gain, client, delay=0.1):
    client.wait_for_properties(['camsci.emgain'])
    client['camsci.emgain.target'] = gain
    time.sleep(delay)
    print(f'Set the CAMSCI gain setting to {gain:.1f}')

def set_camsci_blacklevel(val, client, delay=0.1):
    client.wait_for_properties(['camsci.blacklevel'])
    client['camsci.blacklevel.target'] = val
    time.sleep(delay)
    print(f'Set the CAMSCI blacklevel to {val:.1f}')

def normalize_camsci_image(image, im_params, ref_psf_params):
    image_ni = image/ref_psf_params['Imax']
    image_ni *= (ref_psf_params['texp']/im_params['texp'])
    image_ni *= 10**((im_params['atten']-ref_psf_params['atten'])/10)
    image_ni *= 10**(-im_params['gain']/20 * 0.1) / 10**(-ref_psf_params['gain']/20 * 0.1)
    return image_ni

def set_camlo_exp_time(exp_time, client, delay=0.25):
    if exp_time<3.2e-5:
        print('Minimum exposure time is 3.2E-5 seconds. Setting exposure time to minimum.')
        exp_time = 3.2e-5
    client.wait_for_properties(['camnsv.exptime'])
    client['camnsv.exptime.target'] = exp_time
    time.sleep(delay)
    print(f'Set the CAMLO exposure time to {exp_time:.2e}s')

def set_camlo_gain(gain, client, delay=0.1):
    client.wait_for_properties(['camnsv.emgain'])
    client['camnsv.emgain.target'] = gain
    time.sleep(delay)
    print(f'Set the CAMLO gain setting to {gain:.1f}')

class SCOOBI():

    def __init__(
            self, 
            dm_channel,
            camsci_channel=None,
            camlo_channel=None,
            dm_ref=np.zeros((34,34)),
            Ncamsci=150,
        ):
        self.camsci_stream = ImageStream(camsci_channel) if camsci_channel is not None else None
        self.camlo_stream = ImageStream(camlo_channel) if camlo_channel is not None else None
        self.dm_stream = scoob_utils.connect_to_dmshmim(channel=dm_channel) # channel used for writing to DM
        self.dm_delay = 0.1

        self.wavelength_c = 633e-9
        self.total_pupil_diam = 2.4 # assumed total telescope diameter
        self.fsm_beam_diam = 7.1e-3
        self.dm_beam_diam = 9.1e-3 # as measured in the Fresnel model
        self.lyot_pupil_diam = 9.1e-3
        self.lyot_diam = 8.6e-3
        self.lyot_ratio = self.lyot_diam/self.lyot_pupil_diam
        self.llowfsc_fl = 200e-3
        self.camsci_pxscl = 4.6e-6
        self.camsci_pxscl_lamDc = 0.307
        self.camlo_pxscl = 3.76e-6
        self.camlo_pxscl_lamDc = self.camlo_pxscl / (self.llowfsc_fl * self.wavelength_c / self.lyot_pupil_diam)

        # Init all DM settings
        self.Nact = 34
        self.Nacts = 952
        self.dm_shape = (self.Nact,self.Nact)
        self.act_spacing = 300e-6
        self.dm_ref = dm_ref
        self.reset_dm()
        
        xx = (np.linspace(0, self.Nact-1, self.Nact) - self.Nact/2 + 1/2) * self.act_spacing
        x,y = np.meshgrid(xx,xx)
        r = np.sqrt(x**2 + y**2)
        self.dm_mask = r<10.5e-3/2

        self.NCAMSCI = 1
        self.NCAMLO = 1
        self.Ncamsci = 150
        self.Ncamlo = 96
        self.camsci_x_shift = 0
        self.camsci_y_shift = 0
        self.camlo_x_shift = 0
        self.camlo_y_shift = 0

        self.atten = 1
        self.texp = 1
        self.gain = 1
        self.texp_locam = 1
        self.gain_locam = 1
        
        self.camsci_ref_params = None
        self.dark_frame = None
        self.subtract_dark = False
        self.return_ni = False

    def getattr(self, attr):
        return getattr(self, attr)
    
    def setattr(self, attr, val):
        setattr(self, attr, val)

    def set_fib_atten(self, value, client, delay=0.1):
        client['fiberatten.atten.target'] = value
        time.sleep(delay)
        self.atten = value
        print(f'Set the fiber attenuation to {value:.1f}')

    def set_camsci_exp_time(self, exp_time, client, delay=0.25):
        if exp_time<3.2e-5:
            print('Minimum exposure time is 3.2E-5 seconds. Setting exposure time to minimum.')
            exp_time = 3.2e-5
        client.wait_for_properties(['camsci.exptime'])
        client['camsci.exptime.target'] = exp_time
        time.sleep(delay)
        self.texp = exp_time
        print(f'Set the ZWO exposure time to {self.texp:.2e}s')

    def set_camsci_gain(self, gain, client, delay=0.1):
        client.wait_for_properties(['camsci.emgain'])
        client['camsci.emgain.target'] = gain
        time.sleep(delay)
        self.gain = gain
        print(f'Set the ZWO gain setting to {gain:.1f}')
    
    def zero_dm(self):
        self.dm_stream.write(np.zeros(self.dm_shape))
        time.sleep(self.dm_delay)
    
    def reset_dm(self):
        self.dm_stream.write(ensure_np_array(self.dm_ref))
        time.sleep(self.dm_delay)
    
    def set_dm(self, dm_command):
        self.dm_stream.write(ensure_np_array(dm_command)*1e6)
        time.sleep(self.dm_delay)
    
    def add_dm(self, dm_command):
        dm_state = ensure_np_array(self.get_dm())
        self.dm_stream.write( 1e6*(dm_state + ensure_np_array(dm_command)) )
        time.sleep(self.dm_delay)
               
    def get_dm(self):
        return xp.array(self.dm_stream.grab_latest())/1e6
    
    def close_dm(self):
        self.dm_stream.close()

    def normalize_camsci(self, image):
        if self.camsci_ref_params is None:
            raise ValueError('Cannot normalize because reference PSF not specified.')
        image_ni = image/self.camsci_ref_params['Imax']
        image_ni *= (self.camsci_ref_params['texp']/self.texp)
        image_ni *= 10**((self.atten-self.camsci_ref_params['atten'])/10)
        image_ni *= 10**(-self.gain/20 * 0.1) / 10**(-self.camsci_ref_params['gain']/20 * 0.1)
        return image_ni

    def snap_camsci(self, plot=False, vmin=None):
        im = np.mean( self.camsci_stream.grab_many(self.NCAMSCI), axis=0)
        
        im = xp.array(im)
        im = xcipy.ndimage.shift(im, (self.camsci_y_shift, self.camsci_x_shift), order=0)
        im = utils.pad_or_crop(im, self.Ncamsci)

        if self.subtract_dark and self.df is not None:
            im -= self.df
            print(xp.sum(im<0))
            im[im<0] = 0.0
            
        if self.return_ni:
            im = self.normalize_camsci(im)
        
        return im
    
    def snap_camlo(self, normalize=False, plot=False, vmin=None):
        im = np.mean( self.camlo_stream.grab_many(self.NCAMLO), axis=0)

        im = xp.array(im)
        im = xcipy.ndimage.shift(im, (self.camlo_y_shift, self.camlo_x_shift), order=0)
        im = utils.pad_or_crop(im, self.Ncamlo)

        return im
    
    
        
        
