from .math_module import xp, xcipy, ensure_np_array
from scoob_llowfsc import utils
from scoob_llowfsc.imshows import imshow1, imshow2, imshow3
import scoob_llowfsc.props as props
import scoob_llowfsc.dm as dm

import numpy as np
import astropy.units as u
import copy
import poppy
from scipy.signal import windows

class single():

    def __init__(
            self,
            wavelength=633e-9*u.m, 
            entrance_flux=None, 
        ):
        
        self.wavelength_c = 633e-9*u.m
        self.dm_beam_diam = 9.1*u.mm # as measured in the Fresnel model
        self.lyot_pupil_diam = 9.1*u.mm
        self.lyot_diam = 8.6*u.mm
        self.lyot_ratio = (self.lyot_diam/self.lyot_pupil_diam).decompose().value
        self.rls_diam = 25.4*u.mm
        self.imaging_fl = 140*u.mm
        self.llowfsc_fl = 200*u.mm
        self.llowfsc_fnum = self.llowfsc_fl.to_value(u.mm)/self.lyot_diam.to_value(u.mm)
        self.llowfsc_defocus = 2.5*u.mm
        self.camsci_pxscl = 4.6*u.um / u.pix
        self.camsci_pxscl_lamDc = 0.307
        self.camlo_pxscl = 3.76*u.um / u.pix
        self.camlo_pxscl_lamDc = (self.camlo_pxscl / (self.llowfsc_fl * self.wavelength_c / self.lyot_pupil_diam)).decompose().value

        self.wavelength = wavelength
        self.use_vortex = False
        self.plot_vortex = False
        
        self.npix = 1000
        self.def_oversample = 2.048 # default oversample
        self.rls_oversample = 3 # reflective lyot stop oversample
        self.Ndef = int(self.npix*self.def_oversample)
        self.Nrls = int(self.npix*self.rls_oversample)
        self.ncamsci = 150
        self.ncamlo = 96

        ### INITIALIZE APERTURES ###
        pwf = poppy.FresnelWavefront(beam_radius=self.dm_beam_diam/2, npix=self.npix, oversample=1)
        rls_wf = poppy.FresnelWavefront(beam_radius=self.dm_beam_diam/2, npix=self.npix, oversample=self.rls_oversample)
        self.APERTURE = poppy.CircularAperture(radius=self.dm_beam_diam/2).get_transmission(pwf)
        self.LYOTSTOP = poppy.CircularAperture(radius=self.lyot_diam/2).get_transmission(pwf)
        rls_ap = poppy.CircularAperture(radius=self.rls_diam/2).get_transmission(rls_wf)
        self.RLS = rls_ap - utils.pad_or_crop( self.LYOTSTOP, self.Nrls)
        rls_ap = 0
        self.oap_ap = poppy.CircularAperture(radius=15*u.mm/2).get_transmission(rls_wf)
        self.use_locam = False

        self.LYOT = self.LYOTSTOP
        self.oversample = self.def_oversample
        self.N = self.Ndef # default to not using RLS

        self.BAP_MASK = self.APERTURE>0
        self.AMP = xp.ones((self.npix,self.npix))
        self.OPD = xp.zeros((self.npix,self.npix))

        self.Imax_ref = 1
        self.entrance_flux = entrance_flux
        if self.entrance_flux is not None:
            pixel_area = (self.pupil_diam/self.npix)**2
            flux_per_pixel = self.entrance_flux * pixel_area
            self.APERTURE *= xp.sqrt(flux_per_pixel.to_value(u.photon/u.second))

        ### INITIALIZE DM PARAMETERS ###
        self.Nact = 34
        act_spacing = 300e-6*u.m
        dm_pxscl = self.dm_beam_diam.to_value(u.m)/self.npix
        inf_sampling = act_spacing.to_value(u.m)/dm_pxscl
        inf_fun = dm.make_gaussian_inf_fun(act_spacing=act_spacing, sampling=inf_sampling, coupling=0.15, Nact=self.Nact+2)
        self.DM = dm.DeformableMirror(inf_fun=inf_fun, inf_sampling=inf_sampling, name='DM (pupil)')
        self.dm_mask = self.DM.dm_mask
        self.Nacts = self.DM.Nacts
        self.dm_ref = xp.zeros((self.Nact, self.Nact))

        ### INITIALIZE VORTEX PARAMETERS ###
        self.oversample_vortex = 4.096
        self.N_vortex_lres = int(self.npix*self.oversample_vortex)
        self.vortex_win_diam = 30 # diameter of the window to apply with the vortex model
        self.lres_sampling = 1/self.oversample_vortex # low resolution sampling in lam/D per pixel
        self.lres_win_size = int(self.vortex_win_diam/self.lres_sampling)
        w1d = xp.array(windows.tukey(self.lres_win_size, 1, False))
        self.lres_window = utils.pad_or_crop(xp.outer(w1d, w1d), self.N_vortex_lres)
        self.vortex_lres = props.make_vortex_phase_mask(self.N_vortex_lres)

        self.hres_sampling = 0.025 # lam/D per pixel; this value is chosen empirically
        self.N_vortex_hres = int(np.round(self.vortex_win_diam/self.hres_sampling))
        self.hres_win_size = int(self.vortex_win_diam/self.hres_sampling)
        w1d = xp.array(windows.tukey(self.hres_win_size, 1, False))
        self.hres_window = utils.pad_or_crop(xp.outer(w1d, w1d), self.N_vortex_hres)
        self.vortex_hres = props.make_vortex_phase_mask(self.N_vortex_hres)

        y,x = (xp.indices((self.N_vortex_hres, self.N_vortex_hres)) - self.N_vortex_hres//2) * self.hres_sampling
        r = xp.sqrt(x**2 + y**2)
        self.hres_dot_mask = r>=0.15

    def getattr(self, attr):
        return getattr(self, attr)
    
    def setattr(self, attr, val):
        setattr(self, attr, val)

    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, wl):
        self._wavelength = wl
        self.camsci_pxscl_lamD = self.camsci_pxscl_lamDc * (self.wavelength_c/wl).decompose().value
        self.camlo_pxscl_lamD = self.camlo_pxscl_lamDc * (self.wavelength_c/wl).decompose().value

    def reset_dm(self):
        self.set_dm(self.dm_ref)

    def zero_dm(self):
        self.set_dm(xp.zeros((self.Nact,self.Nact)))
    
    def set_dm(self, command):
        self.DM.command = command
        
    def add_dm(self, command):
        self.DM.command += command
        
    def get_dm(self):
        return self.DM.command

    def use_llowfsc(self, use=True):
        if use:
            self.use_locam = True
            self.N = self.Nrls
            self.oversample = self.rls_oversample
            self.LYOT = self.RLS
        else:
            self.use_locam = False
            self.N = self.Ndef
            self.oversample = self.def_oversample
            self.LYOT = self.LYOTSTOP

    def apply_vortex(self, pupwf, plot=False):
        lres_wf = utils.pad_or_crop(pupwf, self.N_vortex_lres) # pad to the larger array for the low res propagation
        fp_wf_lres = props.fft(lres_wf)
        fp_wf_lres *= self.vortex_lres * (1 - self.lres_window) # apply low res (windowed) FPM
        pupil_wf_lres = props.ifft(fp_wf_lres)
        pupil_wf_lres = utils.pad_or_crop(pupil_wf_lres, self.N,)
        if plot: 
            imshow2(xp.abs(pupil_wf_lres), xp.angle(pupil_wf_lres), 
                            'FFT Pupil Amplitude', 'FFT Pupil Phase', 
                            npix=int(1.5*self.npix), cmap2='twilight', 
                            )

        fp_wf_hres = props.mft_forward(pupwf, self.npix, self.N_vortex_hres, self.hres_sampling, convention='-')
        fp_wf_hres *= self.vortex_hres * self.hres_window * self.hres_dot_mask # apply high res (windowed) FPM
        pupil_wf_hres = props.mft_reverse(fp_wf_hres, self.hres_sampling, self.npix, self.N, convention='+')
        if plot: 
            imshow2(
                xp.abs(pupil_wf_hres), xp.angle(pupil_wf_hres), 
                'MFT Pupil Amplitude', 'MFT Pupil Phase',
                npix=int(1.5*self.npix), cmap2='twilight', 
            )

        post_vortex_pup_wf = (pupil_wf_lres + pupil_wf_hres)
        if plot: 
            imshow2(
                xp.abs(post_vortex_pup_wf), xp.angle(post_vortex_pup_wf), 
                'Total Pupil Amplitude', 'Total Pupil Phase',
                npix=int(1.5*self.npix), cmap2='twilight', 
            )

        return post_vortex_pup_wf

    def calc_wfs(self, save_wfs=True, plot=False): # method for getting the PSF in photons
        wfs = []
        wf = self.APERTURE.astype(complex)
        if save_wfs: wfs.append(copy.copy(wf))

        wf *= self.AMP * xp.exp(1j * 2*xp.pi/self.wavelength.to_value(u.m) * self.OPD )
        if save_wfs: wfs.append(copy.copy(wf))
        if plot: imshow2(xp.abs(wf), xp.angle(wf), 'EP WF', cmap2='twilight', npix=int(1.5*self.npix))

        dm_surf = utils.pad_or_crop(self.DM.get_surface(), self.npix)
        wf *= xp.exp(1j * 4*xp.pi/self.wavelength.to_value(u.m) * dm_surf)
        if save_wfs: wfs.append(copy.copy(wf))
        if plot: imshow2(xp.abs(wf), xp.angle(wf), 'After DM WF', cmap2='twilight', npix=int(1.5*self.npix))

        if self.use_vortex: wf = self.apply_vortex(wf, plot=plot)
        if save_wfs: wfs.append(copy.copy(wf))

        if self.use_locam:
            wf = props.ang_spec(wf, self.wavelength, -150*u.mm, self.lyot_pupil_diam/(self.npix*u.pix))
            wf *= self.oap_ap
            wf = props.ang_spec(wf, self.wavelength, 150*u.mm, self.lyot_pupil_diam/(self.npix*u.pix))

            wf *= utils.pad_or_crop(self.LYOT, wf.shape[0]).astype(complex)
            if save_wfs: wfs.append(copy.copy(wf))
            if plot: imshow2(xp.abs(wf), xp.angle(wf), 'After Lyot Stop WF', cmap2='twilight')

            # Use TF and MFT to propagate to defocused image
            self.llowfsc_fnum = self.llowfsc_fl.to_value(u.mm)/self.lyot_diam.to_value(u.mm)
            tf = props.get_fresnel_TF(
                self.llowfsc_defocus.to_value(u.m) * self.rls_oversample**2, 
                self.Nrls, 
                self.wavelength.to_value(u.m), 
                self.llowfsc_fnum,
            )
            wf = props.mft_forward(tf*wf, self.npix*self.lyot_ratio, self.ncamlo, self.camlo_pxscl_lamD)
            if plot: imshow2(xp.abs(wf)**2, xp.angle(wf), cmap2='twilight',)
            if save_wfs: 
                wfs.append(copy.copy(wf))
                return wfs
            else:
                return wf
            
        wf *= utils.pad_or_crop(self.LYOT, wf.shape[0]).astype(complex)
        if save_wfs: wfs.append(copy.copy(wf))
        if plot: imshow2(xp.abs(wf), xp.angle(wf), 'After Lyot Stop WF', cmap2='twilight', npix=int(1.5*self.npix))

        wf = props.mft_forward(wf, self.npix*self.lyot_ratio, self.ncamsci, self.camsci_pxscl_lamD)
        if save_wfs: wfs.append(copy.copy(wf))
        if plot: imshow2(xp.abs(wf)**2, xp.angle(wf), 'Image Plane WF', cmap2='twilight',)

        if save_wfs:
            return wfs
        else:
            return wf
    
    def calc_wf(self):
        self.use_llowfsc(False)
        fpwf = self.calc_wfs(save_wfs=False) / xp.sqrt(self.Imax_ref)
        return fpwf
    
    def snap_camsci(self):
        self.use_llowfsc(False)
        image = xp.abs(self.calc_wfs(save_wfs=False))**2 / self.Imax_ref
        return image
    
    def snap_camlo(self):
        self.use_llowfsc()
        camlo_im = xp.abs(self.calc_wfs(save_wfs=False))**2
        return camlo_im
    


