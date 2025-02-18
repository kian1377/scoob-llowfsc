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
            dm_ref=xp.zeros((34,34)),
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
        self.plot_oversample = 1.5
        
        self.npix = 1000
        self.def_oversample = 2.048 # default oversample
        self.rls_oversample = 4.096 # reflective lyot stop oversample
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
        self.OAP_AP = poppy.CircularAperture(radius=15*u.mm/2).get_transmission(rls_wf)
        self.use_camlo = False

        self.LYOT = self.LYOTSTOP
        self.oversample = self.def_oversample
        self.N = self.Ndef # default to not using RLS

        self.BAP_MASK = self.APERTURE>0

        # Initialize pupil data
        self.AMP = xp.ones((self.npix,self.npix))
        self.OPD = xp.zeros((self.npix,self.npix))

        # Initialize flux and normalization params
        self.Imax_ref = 1
        self.entrance_flux = entrance_flux
        if self.entrance_flux is not None:
            pixel_area = (self.pupil_diam/self.npix)**2
            flux_per_pixel = self.entrance_flux * pixel_area
            self.APERTURE *= xp.sqrt(flux_per_pixel.to_value(u.photon/u.second))

        ### INITIALIZE DM PARAMETERS ###
        self.Nact = 34
        self.dm_shape = (self.Nact, self.Nact)
        self.act_spacing = 300e-6*u.m
        self.dm_pxscl = self.dm_beam_diam.to_value(u.m) / self.npix
        self.inf_sampling = self.act_spacing.to_value(u.m)/self.dm_pxscl
        self.inf_fun = dm.make_gaussian_inf_fun(
            act_spacing=self.act_spacing, 
            sampling=self.inf_sampling, 
            coupling=0.15, 
            Nact=self.Nact+2,
        )
        self.Nsurf = self.inf_fun.shape[0]

        y,x = (xp.indices((self.Nact, self.Nact)) - self.Nact//2 + 1/2)
        r = xp.sqrt(x**2 + y**2)
        self.dm_mask = r<(self.Nact/2 + 1/2)
        self.Nacts = int(self.dm_mask.sum())

        self.inf_fun_fft = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(self.inf_fun,)))

        xc = self.inf_sampling*(xp.linspace(-self.Nact//2, self.Nact//2-1, self.Nact) + 1/2) # DM command coordinates
        yc = self.inf_sampling*(xp.linspace(-self.Nact//2, self.Nact//2-1, self.Nact) + 1/2)

        fx = xp.fft.fftshift(xp.fft.fftfreq(self.Nsurf)) # Influence function frequncy sampling
        fy = xp.fft.fftshift(xp.fft.fftfreq(self.Nsurf))

        self.Mx = xp.exp(-1j*2*np.pi*xp.outer(fx,xc)) # forward DM model MFT matrices
        self.My = xp.exp(-1j*2*np.pi*xp.outer(yc,fy))

        self.dm_ref = copy.copy(dm_ref)
        self.dm_command = copy.copy(dm_ref)

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
        self.set_dm(copy.copy(self.dm_ref))

    def zero_dm(self):
        self.set_dm(xp.zeros((self.Nact,self.Nact)))
    
    def set_dm(self, command):
        self.dm_command = copy.copy(command)
        
    def add_dm(self, command):
        self.dm_command += copy.copy( command )
        
    def get_dm(self):
        return copy.copy( self.dm_command )

    def compute_dm_phasor(self):
        mft_command = self.Mx @ self.dm_command @ self.My
        fourier_surf = self.inf_fun_fft * mft_command
        dm_surf = xp.fft.fftshift( xp.fft.ifft2( xp.fft.ifftshift( fourier_surf, ))).real
        dm_phasor = xp.exp(1j * 4*xp.pi/self.wavelength.to_value(u.m) * dm_surf )
        dm_phasor = utils.pad_or_crop(dm_phasor, self.N)
        return dm_phasor

    def apply_vortex(self, pupwf, plot=False):
        lres_wf = utils.pad_or_crop(pupwf, self.N_vortex_lres) # pad to the larger array for the low res propagation
        fp_wf_lres = props.fft(lres_wf)
        fp_wf_lres *= self.vortex_lres * (1 - self.lres_window) # apply low res (windowed) FPM
        pupil_wf_lres = props.ifft(fp_wf_lres)
        # pupil_wf_lres = utils.pad_or_crop(pupil_wf_lres, self.N,)
        if plot: 
            imshow2(xp.abs(pupil_wf_lres), xp.angle(pupil_wf_lres), 
                            'FFT Lyot Pupil Amplitude', 'FFT Lyot Pupil Phase', 
                            npix=int(self.plot_oversample*self.npix), cmap2='twilight', 
                            )

        fp_wf_hres = props.mft_forward(pupwf, self.npix, self.N_vortex_hres, self.hres_sampling, convention='-')
        fp_wf_hres *= self.vortex_hres * self.hres_window * self.hres_dot_mask # apply high res (windowed) FPM
        pupil_wf_hres = props.mft_reverse(fp_wf_hres, self.hres_sampling, self.npix, self.N_vortex_lres, convention='+')
        if plot: 
            imshow2(
                xp.abs(pupil_wf_hres), xp.angle(pupil_wf_hres), 
                'MFT Lyot Pupil Amplitude', 'MFT Lyot Pupil Phase',
                npix=int(self.plot_oversample*self.npix), cmap2='twilight', 
            )

        post_vortex_pup_wf = (pupil_wf_lres + pupil_wf_hres)
        if plot: 
            imshow2(
                xp.abs(post_vortex_pup_wf), xp.angle(post_vortex_pup_wf), 
                'Total Lyot Pupil Amplitude', 'Total Lyot Pupil Phase',
                npix=int(self.plot_oversample*self.npix), cmap2='twilight', 
            )

        return post_vortex_pup_wf

    def calc_wfs_camsci(self, return_all=True, plot=False): # method for getting the PSF in photons
        WFE = self.AMP * xp.exp(1j * 2*xp.pi/self.wavelength.to_value(u.m) * self.OPD )
        E_EP =  self.APERTURE.astype(complex) * WFE
        E_EP = utils.pad_or_crop(E_EP, self.N)
        if plot: imshow2(xp.abs(E_EP), xp.angle(E_EP), 'EP WF', cmap2='twilight', npix=int(self.plot_oversample*self.npix))

        DM_PHASOR = self.compute_dm_phasor()
        E_DM = E_EP * DM_PHASOR
        if plot: imshow2(xp.abs(E_DM), xp.angle(E_DM), 'After DM WF', cmap2='twilight', npix=int(self.plot_oversample*self.npix))

        if self.use_vortex: 
            E_LP = self.apply_vortex(E_DM, plot=plot)
        else: 
            E_LP = copy.copy(E_DM)
            
        E_LS = E_LP * utils.pad_or_crop(self.LYOT, E_LP.shape[0]).astype(complex)
        if plot: imshow2(xp.abs(E_LS), xp.angle(E_LS), 'After Lyot Stop WF', cmap2='twilight', npix=int(self.plot_oversample*self.npix))

        E_CAMSCI = props.mft_forward(E_LS, self.npix*self.lyot_ratio, self.ncamsci, self.camsci_pxscl_lamD)
        if plot: imshow2(xp.abs(E_CAMSCI)**2, xp.angle(E_CAMSCI), 'CAMSCI WF', lognorm1=1, cmap2='twilight',)

        if return_all:
            return E_EP, DM_PHASOR, E_DM, E_LP, E_LS, E_CAMSCI
        else:
            return E_CAMSCI
    
    def calc_wfs_camlo(self, return_all=True, plot=False): # method for getting the PSF in photons
        WFE = self.AMP * xp.exp(1j * 2*xp.pi/self.wavelength.to_value(u.m) * self.OPD )
        E_EP =  self.APERTURE.astype(complex) * WFE
        E_EP = utils.pad_or_crop(E_EP, self.N)
        if plot: imshow2(xp.abs(E_EP), xp.angle(E_EP), 'EP WF', cmap2='twilight', npix=int(self.plot_oversample*self.npix))

        DM_PHASOR = self.compute_dm_phasor()
        E_DM = E_EP * DM_PHASOR
        if plot: imshow2(xp.abs(E_DM), xp.angle(E_DM), 'After DM WF', cmap2='twilight', npix=int(self.plot_oversample*self.npix))

        if self.use_vortex: 
            E_LP = self.apply_vortex(E_DM, plot=plot)
        else: 
            E_LP = copy.copy(E_DM)

        E_LP = props.ang_spec(E_LP, self.wavelength, -150*u.mm, self.lyot_pupil_diam/(self.npix*u.pix))
        E_LP *= utils.pad_or_crop(self.OAP_AP, E_LP.shape[0])
        # print(E_LP.shape)
        E_LP = props.ang_spec(E_LP, self.wavelength, 150*u.mm, self.lyot_pupil_diam/(self.npix*u.pix))

        E_RLS = E_LP * utils.pad_or_crop(self.RLS, E_LP.shape[0]).astype(complex)
        if plot: imshow2(xp.abs(E_RLS), xp.angle(E_RLS), 'After RLS WF', cmap2='twilight')

        # Use TF and MFT to propagate to defocused image
        self.llowfsc_fnum = self.llowfsc_fl.to_value(u.mm)/self.lyot_diam.to_value(u.mm)
        camlo_tf = props.get_fresnel_TF(
            self.llowfsc_defocus.to_value(u.m) * self.rls_oversample**2, 
            self.Nrls, 
            self.wavelength.to_value(u.m), 
            self.llowfsc_fnum,
        )
        E_CAMLO = props.mft_forward(camlo_tf*E_RLS, self.npix*self.lyot_ratio, self.ncamlo, self.camlo_pxscl_lamD)
        if plot: imshow2(xp.abs(E_CAMLO)**2, xp.angle(E_CAMLO), cmap2='twilight',)
            
        if return_all:
            return E_EP, DM_PHASOR, E_DM, E_LP, E_RLS, E_CAMLO
        else:
            return E_CAMLO
    
    def calc_wf_camsci(self):
        fpwf = self.calc_wfs_camsci( return_all=False ) / xp.sqrt(self.Imax_ref)
        return fpwf
    
    def snap_camsci(self):
        image = xp.abs(self.calc_wfs_camsci(return_all=False))**2 / self.Imax_ref
        return image
    
    def snap_camlo(self):
        camlo_im = xp.abs(self.calc_wfs_camlo(return_all=False))**2
        return camlo_im
    


