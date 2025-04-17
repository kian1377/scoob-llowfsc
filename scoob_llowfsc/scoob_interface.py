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

def create_shmim(name, dims, dtype=None, shared=1, nbkw=8):
    # if ImageStream objects didn't auto-open on creation, you could create and return that instead. oops.
    img = shmio.Image() # not sure if I should try to destroy first in case it already exists
    buffer = np.zeros(dims)
    if dtype is None: dtype = shmio.ImageStreamIODataType.FLOAT
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

def set_fib_atten(value, client, delay=0.1):
    client['fiberatten.atten.target'] = value
    time.sleep(delay)
    print(f'Set the fiber attenuation to {value:.1f}')

def get_fib_atten(client,):
    fib_atten = client['fiberatten.atten.current']
    return fib_atten

# CAMERA Functions
def set_cam_roi(xc, yc, npix, client, cam_name='camsci', bin_mode=2, delay=0.25):
    # update roi parameters
    client.wait_for_properties([
        f'{cam_name}.roi_region_x', f'{cam_name}.roi_region_y', 
        f'{cam_name}.roi_region_h' , f'{cam_name}.roi_region_w', 
        f'{cam_name}.roi_region_bin_x', f'{cam_name}.roi_region_bin_y', 
        f'{cam_name}.roi_set',
    ])
    client[f'{cam_name}.roi_region_bin_x.target'] = bin_mode
    client[f'{cam_name}.roi_region_bin_y.target'] = bin_mode
    client[f'{cam_name}.roi_region_x.target'] = xc
    client[f'{cam_name}.roi_region_y.target'] = yc
    client[f'{cam_name}.roi_region_h.target'] = npix
    client[f'{cam_name}.roi_region_w.target'] = npix
    time.sleep(delay)
    client[f'{cam_name}.roi_set.request'] = purepyindi.SwitchState.ON
    time.sleep(delay)
    print(f'Set {cam_name} ROI.')

def set_cam_exp_time(exp_time, client, cam_name='camsci', delay=0.25):
    client.wait_for_properties([f'{cam_name}.exptime'])
    client[f'{cam_name}.exptime.target'] = exp_time
    time.sleep(delay)
    print(f'Set the {cam_name} exposure time to {exp_time:.2e}s')

def set_cam_gain(gain, client, cam_name='camsci', delay=0.1):
    client.wait_for_properties([f'{cam_name}.emgain'])
    client[f'{cam_name}.emgain.target'] = gain
    time.sleep(delay)
    print(f'Set the {cam_name} gain setting to {gain:.1f}')

def set_cam_blacklevel(val, client, cam_name='camsci', delay=0.1):
    client.wait_for_properties([f'{cam_name}.blacklevel'])
    client[f'{cam_name}.blacklevel.target'] = val
    time.sleep(delay)
    print(f'Set the {cam_name} blacklevel to {val:.1f}')

def set_camnsv_roi(xc, vcropoffset, client, delay=0.25):
    # update roi parameters
    client.wait_for_properties(['camnsv.roi_region_x', 'camnsv.vcropoffset', 'camnsv.roi_set'])
    client['camnsv.roi_region_x.target'] = xc
    client['camnsv.vcropoffset.target'] = vcropoffset
    time.sleep(delay)
    client['camnsv.roi_set.request'] = purepyindi.SwitchState.ON
    time.sleep(delay)
    print('Set CAMLO ROI.')

def get_im_params(client0, client, cam_name='camsci'):
    client0.wait_for_properties([f'{cam_name}.exptime', f'{cam_name}.emgain', ])
    exp_time = client[f'{cam_name}.exptime.target']
    gain = client[f'{cam_name}.emgain.target']
    fib_atten = client['fiberatten.atten.target']

    im_params = {
        'texp':exp_time,
        'gain':gain,
        'atten':fib_atten,
    }
    return im_params

def normalize_image(image, im_params, ref_psf_params):
    image_ni = image / ref_psf_params['Iref']
    image_ni *= ref_psf_params['texp'] / im_params['texp']
    image_ni *= 10**( (im_params['atten'] - ref_psf_params['atten']) / 10)
    image_ni *= 10**( -im_params['gain']/20 * 0.1) / 10**(-ref_psf_params['gain']/20 * 0.1)
    return image_ni

def snap(cam_stream, NFRAMES=1, dark=0, im_params=None, ref_psf_params=None):
    im = np.mean( cam_stream.grab_many(NFRAMES), axis=0) - dark
    if im_params is not None and ref_psf_params is not None: 
        im = normalize_image(im, im_params, ref_psf_params)
    return im

def zero_dm(dm_stream, delay=0.01):
    dm_stream.write( np.zeros((34,34)) )
    time.sleep(delay)

def set_dm(dm_stream, dm_command, delay=0.01):
    dm_stream.write( 1e6*ensure_np_array(dm_command) )
    time.sleep(delay)

def add_dm(dm_stream, dm_command, delay=0.01):
    current_command = dm_stream.grab_latest()
    dm_stream.write( current_command + 1e6*ensure_np_array(dm_command) )
    time.sleep(delay)

def set_kilo_mod_amp(amp, client, process_name='kiloModulator', delay=0.25):
    client.wait_for_properties([f'{process_name}.amp'])
    client[f'{process_name}.amp.target'] = amp
    time.sleep(delay)

def set_kilo_mod_rate(freq, client, process_name='kiloModulator', delay=0.25):
    client.wait_for_properties([f'{process_name}.frequency'])
    client[f'{process_name}.frequency.target'] = freq
    time.sleep(delay)

import subprocess
def start_kiloModulator(delay=0.5):
    subprocess.run(['xctrl', 'start', 'kiloModulator'])
    time.sleep(delay)

def stop_kiloModulator(delay=0.5):
    subprocess.run(['xctrl', 'stop', 'kiloModulator'])
    time.sleep(delay)

def toggle_kilo_mod(toggle, client, process_name='kiloModulator', delay=0.25):
    if toggle:
        client.wait_for_properties([f'{process_name}.trigger', f'{process_name}.modulating'])
        client[f'{process_name}.trigger.toggle'] = purepyindi.SwitchState.OFF
        time.sleep(delay)
        client[f'{process_name}.modulating.toggle'] = purepyindi.SwitchState.ON
        time.sleep(delay)
    else:
        client.wait_for_properties([f'{process_name}.trigger', f'{process_name}.modulating', f'{process_name}.zero'])
        client[f'{process_name}.modulating.toggle'] = purepyindi.SwitchState.OFF
        time.sleep(delay)
        client[f'{process_name}.trigger.toggle'] = purepyindi.SwitchState.ON
        time.sleep(delay)
        client[f'{process_name}.zero.request'] = purepyindi.SwitchState.ON
        time.sleep(delay)

from matplotlib.colors import LogNorm

def monitor_camsci(
        camsci_stream, 
        im_params,
        ref_psf_params,
        dark_frame,
        control_mask, 
        NFRAMES=10,
        duration=60,
        plot=False, 
        clear=True, 
        save_path=None,
    ):
    all_ims = []
    try:
        print('Streaming camsci data ...')
        i = 0
        start = time.time()
        while (time.time()-start)<duration:
            im = np.mean(camsci_stream.grab_many(NFRAMES), axis=0) - dark_frame
            im_ni = normalize_image(im, im_params, ref_psf_params)
            all_ims.append(im_ni)
            i += 1
            contrast = xp.mean(im_ni[control_mask])
            print(f'Mean NI = {contrast:.2e}')
            if plot:
                utils.imshow([im_ni], norms=[LogNorm(1e-9)])
            if clear:
                clear_output(wait=True)
    except KeyboardInterrupt:
        print('Stopping camsci stream!')
    if save_path is not None:
        utils.save_fits(save_path, np.array(all_ims))

    return np.array(all_ims)


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

    def set_camsci_exp_time(self, exp_time, client, camsci_name='camsci', delay=0.25):
        if exp_time<3.2e-5:
            print('Minimum exposure time is 3.2E-5 seconds. Setting exposure time to minimum.')
            exp_time = 3.2e-5
        client.wait_for_properties([f'{camsci_name}.exptime'])
        client[f'{camsci_name}.exptime.target'] = exp_time
        time.sleep(delay)
        self.texp = exp_time
        print(f'Set the CAMSCI exposure time to {self.texp:.2e}s')

    def set_camsci_gain(self, gain, client, camsci_name='camsci', delay=0.1):
        client.wait_for_properties([f'{camsci_name}.emgain'])
        client[f'{camsci_name}.emgain.target'] = gain
        time.sleep(delay)
        self.gain = gain
        print(f'Set the CAMSCI gain setting to {gain:.1f}')
    
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
    
    
        
        
