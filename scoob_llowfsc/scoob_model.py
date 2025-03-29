from .math_module import xp, xcipy, ensure_np_array
from scoob_llowfsc import utils, dm, props

import numpy as np
import astropy.units as u
import copy
import poppy
from scipy.signal import windows

from matplotlib.colors import LogNorm, Normalize, CenteredNorm

try:
    import ray
except ImportError:
    print('Could not import ray. Parallelized model unavailble.')

class single():

    def __init__(
            self,
            wavelength=633e-9, 
            dm_ref=xp.zeros((34,34)),
            entrance_flux=None, 
        ):
        
        self.wavelength_c = 633e-9
        self.total_pupil_diam = 2.4 # assumed total telescope diameter
        self.fsm_beam_diam = 7.1e-3
        self.dm_beam_diam = 9.1e-3 # as measured in the Fresnel model
        self.lyot_pupil_diam = 9.1e-3
        self.lyot_diam = 8.6e-3
        self.lyot_ratio = self.lyot_diam/self.lyot_pupil_diam
        self.rls_diam = 25.4e-3
        self.d_oap_ls = 150e-3
        self.imaging_fl = 140e-3
        self.llowfsc_fl = 200e-3
        self.llowfsc_fnum = self.llowfsc_fl/self.lyot_diam
        self.llowfsc_defocus = 2.75e-3
        self.camsci_pxscl = 4.6e-6
        self.camsci_pxscl_lamDc = 0.307
        self.camlo_pxscl = 3.76e-6
        self.camlo_pxscl_lamDc = self.camlo_pxscl / (self.llowfsc_fl * self.wavelength_c / self.lyot_pupil_diam)

        self.wavelength = wavelength
        self.use_vortex = False
        self.plot_vortex = False
        self.plot_oversample = 1.5
        
        self.npix = 1000
        self.def_oversample = 2.048 # default oversample
        self.rls_oversample = 3 # reflective lyot stop oversample
        self.Ndef = int(self.npix*self.def_oversample)
        self.Nrls = int(self.npix*self.rls_oversample)
        self.ncamsci = 150
        self.ncamlo = 96

        self.tt_pv_to_rms = 1/4
        self.as_per_radian = 206264.806

        ### INITIALIZE APERTURES ###
        self.npix_rls = int( np.round( self.npix * self.rls_diam / self.lyot_pupil_diam ))
        pwf = poppy.FresnelWavefront(beam_radius=self.dm_beam_diam/2*u.m, npix=self.npix, oversample=1)
        self.APERTURE = poppy.CircularAperture(radius=self.dm_beam_diam/2*u.m).get_transmission(pwf)
        self.LYOTSTOP = poppy.CircularAperture(radius=self.lyot_diam/2*u.m).get_transmission(pwf)

        pwf_rls = poppy.FresnelWavefront(beam_radius=self.dm_beam_diam/2*u.m, npix=self.npix, oversample=self.rls_oversample)
        rls_ap = poppy.CircularAperture(radius=self.rls_diam/2*u.m).get_transmission(pwf_rls)
        self.RLS = rls_ap - utils.pad_or_crop( self.LYOTSTOP, self.Nrls)
        rls_ap = 0

        self.OAP_AP = poppy.CircularAperture(radius=15*u.mm/2).get_transmission(pwf_rls)
        self.use_camlo = False

        self.LYOT = self.LYOTSTOP
        self.oversample = self.def_oversample
        self.N = self.Ndef # default to not using RLS

        self.BAP_MASK = self.APERTURE>0

        # Initialize pupil data
        self.PREFPM_AMP = xp.ones((self.npix,self.npix))
        self.PREFPM_OPD = xp.zeros((self.npix,self.npix))

        self.POSTFPM_AMP = xp.ones((self.npix,self.npix))
        self.POSTFPM_OPD = xp.zeros((self.npix,self.npix))

        self.RLS_AMP = xp.ones((self.npix,self.npix))
        self.RLS_OPD = xp.zeros((self.npix,self.npix))

        self.PTT_MODES = utils.create_zernike_modes(self.APERTURE, nmodes=3, remove_modes=0) # define tip/tilt modes
        self.FSM_PTT = np.zeros(3) # [OPD in m, arcsec, arcsec]
        self.FSM_OPD = 0*self.PTT_MODES[0]

        # Initialize flux and normalization params
        self.Imax_ref = 1
        self.entrance_flux = entrance_flux
        if self.entrance_flux is not None:
            pixel_area = (self.total_pupil_diam*u.m/self.npix)**2
            flux_per_pixel = self.entrance_flux * pixel_area
            self.APERTURE *= xp.sqrt(flux_per_pixel.to_value(u.photon/u.second))

        ### INITIALIZE DM PARAMETERS ###
        self.Nact = 34
        self.dm_shape = (self.Nact, self.Nact)
        self.act_spacing = 300e-6
        self.dm_pxscl = self.dm_beam_diam / self.npix
        self.inf_sampling = self.act_spacing / self.dm_pxscl
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
        self.Mx_back = xp.exp(1j*2*np.pi*xp.outer(xc,fx)) # adjoint DM model MFT matrices
        self.My_back = xp.exp(1j*2*np.pi*xp.outer(fy,yc))

        self.dm_ref = copy.copy(dm_ref)
        self.dm_channels = xp.zeros((10,34,34))
        self.dm_channels[0] = self.dm_ref
        self.dm_total = xp.sum(self.dm_channels, axis=0)

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

        # DETECTOR PARAMETERS
        self.CAMLO = None
        self.NCAMLO = 1
        self.camlo_shear = None

        self.camsci_shear = None

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
        self.camsci_pxscl_lamD = self.camsci_pxscl_lamDc * self.wavelength_c/wl
        self.camlo_pxscl_lamD = self.camlo_pxscl_lamDc * self.wavelength_c/wl

    def zero_fsm(self,):
        self.FSM_PTT = np.array([0,0,0])
        self.FSM_OPD = 0*self.PTT_MODES[0]

    def set_fsm(self, ptt):
        self.FSM_PTT = ptt
        # self.FSM_OPD = self.FSM_PTT[0]*self.PTT_MODES[0] + self.FSM_PTT[1]*self.PTT_MODES[1] + self.FSM_PTT[2]*self.PTT_MODES[2]
        
        tip_at_pupil_pv = np.tan(self.FSM_PTT[1]/self.as_per_radian) * self.fsm_beam_diam
        tilt_at_pupil_pv = np.tan(self.FSM_PTT[2]/self.as_per_radian) * self.fsm_beam_diam

        tip_at_pupil_rms = tip_at_pupil_pv * self.tt_pv_to_rms
        tilt_at_pupil_rms = tilt_at_pupil_pv * self.tt_pv_to_rms

        self.FSM_OPD = self.FSM_PTT[0]*self.PTT_MODES[0] + tip_at_pupil_rms*self.PTT_MODES[1] + tilt_at_pupil_rms*self.PTT_MODES[2]

    def add_fsm(self, ptt):
        self.FSM_PTT = self.FSM_PTT + ptt
        # self.FSM_OPD = self.FSM_PTT[0]*self.PTT_MODES[0] + self.FSM_PTT[1]*self.PTT_MODES[1] + self.FSM_PTT[2]*self.PTT_MODES[2]

        tip_at_pupil_pv = np.tan(self.FSM_PTT[1]/self.as_per_radian) * self.fsm_beam_diam
        tilt_at_pupil_pv = np.tan(self.FSM_PTT[2]/self.as_per_radian) * self.fsm_beam_diam

        tip_at_pupil_rms = tip_at_pupil_pv * self.tt_pv_to_rms
        tilt_at_pupil_rms = tilt_at_pupil_pv * self.tt_pv_to_rms

        self.FSM_OPD = self.FSM_PTT[0]*self.PTT_MODES[0] + tip_at_pupil_rms*self.PTT_MODES[1] + tilt_at_pupil_rms*self.PTT_MODES[2]
    
    def get_fsm(self):
        return self.FSM_PTT

    def reset_dm(self):
        self.dm_channels = xp.zeros((10,34,34))
        self.dm_channels[0] = self.dm_ref
        self.dm_total = xp.sum(self.dm_channels, axis=0)

    def zero_dm(self, channel=1):
        self.dm_channels[channel] = xp.zeros((34,34))
        self.dm_total = xp.sum(self.dm_channels, axis=0)

    def set_dm(self, command, channel=1):
        self.dm_channels[channel] = copy.copy(command)
        self.dm_total = xp.sum(self.dm_channels, axis=0)

    def add_dm(self, command, channel=1):
        old = self.dm_channels[channel]
        self.dm_channels[channel] = copy.copy(old + command)
        self.dm_total = xp.sum(self.dm_channels, axis=0)

    def get_dm(self, channel=1):
        return copy.copy(self.dm_channels[channel])

    def get_dm_total(self):
        return self.dm_total

    def compute_dm_phasor(self):
        mft_command = self.Mx @ self.dm_total @ self.My
        fourier_surf = self.inf_fun_fft * mft_command
        dm_surf = xp.fft.fftshift( xp.fft.ifft2( xp.fft.ifftshift( fourier_surf, ))).real
        dm_phasor = xp.exp(1j * 4*xp.pi/self.wavelength * dm_surf )
        dm_phasor = utils.pad_or_crop(dm_phasor, self.N)
        return dm_phasor

    def apply_vortex(self, pupwf, plot=False):
        N = pupwf.shape[0]

        lres_wf = utils.pad_or_crop(pupwf, self.N_vortex_lres) # pad to the larger array for the low res propagation
        fp_wf_lres = props.fft(lres_wf)
        fp_wf_lres *= self.vortex_lres * (1 - self.lres_window) # apply low res (windowed) FPM
        pupil_wf_lres = props.ifft(fp_wf_lres)
        pupil_wf_lres = utils.pad_or_crop(pupil_wf_lres, N) # crop to the desired wavefront dimension
        if plot: 
            utils.imshow(
                [xp.abs(pupil_wf_lres), xp.angle(pupil_wf_lres)], 
                titles=['FFT Lyot Pupil Amplitude', 'FFT Lyot Pupil Phase'], 
                npix=[int(self.plot_oversample*self.npix)]*2, 
                cmaps=['magma', 'twilight'], 
            )

        fp_wf_hres = props.mft_forward(pupwf, self.npix, self.N_vortex_hres, self.hres_sampling, convention='-')
        fp_wf_hres *= self.vortex_hres * self.hres_window * self.hres_dot_mask # apply high res (windowed) FPM
        pupil_wf_hres = props.mft_reverse(fp_wf_hres, self.hres_sampling, self.npix, N, convention='+')
        if plot: 
            utils.imshow(
                [xp.abs(pupil_wf_hres), xp.angle(pupil_wf_hres)], 
                titles=['MFT Lyot Pupil Amplitude', 'MFT Lyot Pupil Phase'], 
                npix=[int(self.plot_oversample*self.npix)]*2, 
                cmaps=['magma', 'twilight'], 
            )

        post_vortex_pup_wf = (pupil_wf_lres + pupil_wf_hres)
        if plot: 
            utils.imshow(
                [xp.abs(post_vortex_pup_wf), xp.angle(post_vortex_pup_wf)], 
                titles=['Total Lyot Pupil Amplitude', 'Total Lyot Pupil Phase'], 
                npix=[int(self.plot_oversample*self.npix)]*2, 
                cmaps=['magma', 'twilight'], 
            )

        return post_vortex_pup_wf

    def calc_wfs_camsci(self, return_all=True): # method for getting the PSF in photons
        FSM_PHASOR = xp.exp(1j * 4*xp.pi/self.wavelength * self.FSM_OPD )
        PREFPM_WFE = self.PREFPM_AMP * xp.exp(1j * 2*xp.pi/self.wavelength * self.PREFPM_OPD )
        E_EP =  self.APERTURE.astype(complex) * PREFPM_WFE * FSM_PHASOR
        E_EP = utils.pad_or_crop(E_EP, self.N)

        DM_PHASOR = self.compute_dm_phasor()
        E_DM = E_EP * DM_PHASOR

        if self.use_vortex: 
            E_LP = self.apply_vortex(E_DM)
        else: 
            E_LP = copy.copy(E_DM)

        POSTFPM_WFE = self.POSTFPM_AMP * xp.exp(1j * 2*xp.pi/self.wavelength * self.POSTFPM_OPD )
        E_LP =  E_LP * utils.pad_or_crop(POSTFPM_WFE, E_LP.shape[0])

        E_LS = E_LP * utils.pad_or_crop(self.LYOT, E_LP.shape[0]).astype(complex)

        E_CAMSCI = props.mft_forward(E_LS, self.npix*self.lyot_ratio, self.ncamsci, self.camsci_pxscl_lamD)
        if self.camsci_shear is not None: # shift the CAMLO image to simulate detector lateral shift
            E_CAMSCI = xcipy.ndimage.shift(E_CAMSCI, (self.camsci_shear[1], self.camsci_shear[0]), order=3)
 
        if return_all:
            return E_EP, DM_PHASOR, E_DM, E_LP, E_LS, E_CAMSCI
        else:
            return E_CAMSCI
    
    def calc_wfs_camlo(self, return_all=True): # method for getting the PSF in photons
        FSM_PHASOR = xp.exp(1j * 4*xp.pi/self.wavelength * self.FSM_OPD )
        PREFPM_WFE = self.PREFPM_AMP * xp.exp(1j * 2*xp.pi/self.wavelength * self.PREFPM_OPD )
        E_EP =  self.APERTURE.astype(complex) * PREFPM_WFE * FSM_PHASOR
        E_EP = utils.pad_or_crop(E_EP, self.N)

        DM_PHASOR = self.compute_dm_phasor()
        E_DM = E_EP * DM_PHASOR

        if self.use_vortex: 
            E_DM = utils.pad_or_crop(E_DM, self.Nrls)
            E_LP = self.apply_vortex(E_DM)
        else: 
            E_LP = copy.copy(E_DM)
        # print(E_LP.shape)

        E_LP = props.ang_spec(E_LP, self.wavelength, -self.d_oap_ls, self.lyot_pupil_diam/self.npix)
        E_LP *= utils.pad_or_crop(self.OAP_AP, E_LP.shape[0])
        E_LP = props.ang_spec(E_LP, self.wavelength, self.d_oap_ls, self.lyot_pupil_diam/self.npix)
        
        RLS_WFE = self.RLS_AMP * xp.exp(1j * 2*xp.pi/self.wavelength * self.RLS_OPD )
        E_RLS =  E_LP * utils.pad_or_crop(self.RLS, E_LP.shape[0]).astype(complex) * utils.pad_or_crop(RLS_WFE, E_LP.shape[0])

        # Use TF and MFT to propagate to defocused image
        self.llowfsc_fnum = self.llowfsc_fl/self.lyot_diam
        camlo_tf = props.get_fresnel_TF(
            self.llowfsc_defocus * self.rls_oversample**2, 
            self.Nrls, 
            self.wavelength, 
            self.llowfsc_fnum,
        )
        E_CAMLO = props.mft_forward(camlo_tf*E_RLS, self.npix*self.lyot_ratio, self.ncamlo, self.camlo_pxscl_lamD)
        if self.camlo_shear is not None: # shift the CAMLO image to simulate detector lateral shift
            E_CAMLO = xcipy.ndimage.shift(E_CAMLO, (self.camlo_shear[1], self.camlo_shear[0]), order=3)
 
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
        if self.CAMLO is not None:
            noisy_im = 0.0
            for i in range(self.NCAMLO):
                noisy_im += self.CAMLO.add_noise(camlo_im)
            return noisy_im/self.NCAMLO
        return camlo_im

class parallel():
    def __init__(
            self,
            ACTORS,
        ):

        self.ACTORS = ACTORS
        self.Nactors = len(ACTORS)

        self.wavelength_c = self.getattr('wavelength_c')
        self.total_pupil_diam = self.getattr('total_pupil_diam')
        self.fsm_beam_diam = self.getattr('fsm_beam_diam')
        self.dm_beam_diam = self.getattr('dm_beam_diam')
        self.lyot_pupil_diam = self.getattr('lyot_pupil_diam')
        self.lyot_diam = self.getattr('lyot_diam')
        self.lyot_ratio = self.getattr('lyot_ratio')
        self.rls_diam = self.getattr('rls_diam')
        self.imaging_fl = self.getattr('imaging_fl')
        self.llowfsc_fl = self.getattr('llowfsc_fl')
        self.llowfsc_fnum  = self.getattr('llowfsc_fnum')
        self.llowfsc_defocus = self.getattr('llowfsc_defocus')
        self.camsci_pxscl = self.getattr('camsci_pxscl')
        self.camsci_pxscl_lamDc = self.getattr('camsci_pxscl_lamDc')
        self.camlo_pxscl = self.getattr('camlo_pxscl_lamDc')
        self.camlo_pxscl_lamDc = self.getattr('camlo_pxscl_lamDc')

        self.PTT_MODES = ray.get(ACTORS[0].getattr.remote('PTT_MODES'))

        self.Nact = ray.get(ACTORS[0].getattr.remote('Nact'))
        self.dm_mask = ray.get(ACTORS[0].getattr.remote('dm_mask'))
        self.dm_ref = ray.get(ACTORS[0].getattr.remote('dm_ref'))
        self.reset_dm()

        # DETECTOR PARAMETERS
        self.CAMSCI = None
        self.NCAMSCI = 1
        self.Imax_ref = 1

        self.CAMLO = None
        self.NCAMLO = 1

    def getattr(self, attr):
        return ray.get(self.ACTORS[0].getattr.remote(attr))
    
    def set_actor_attr(self, attr, value):
        for i in range(len(self.ACTORS)):
            self.ACTORS[i].setattr.remote(attr, value)
    
    def zero_fsm(self,):
        for i in range(len(self.ACTORS)):
            self.ACTORS[i].zero_fsm.remote()

    def set_fsm(self, ptt):
        for i in range(len(self.ACTORS)):
            self.ACTORS[i].set_fsm.remote( ptt )
        
    def add_fsm(self, ptt):
        for i in range(len(self.ACTORS)):
            self.ACTORS[i].add_fsm.remote( ptt )

    def get_fsm(self):
        return self.getattr('FSM_PTT')

    def reset_dm(self):
        for i in range(len(self.ACTORS)):
            self.ACTORS[i].reset_dm.remote()

    def zero_dm(self, channel=1):
        for i in range(len(self.ACTORS)):
            self.ACTORS[i].zero_dm.remote(channel)

    def set_dm(self, command, channel=1):
        for i in range(len(self.ACTORS)):
            self.ACTORS[i].set_dm.remote(command, channel)

    def add_dm(self, command, channel=1):
        for i in range(len(self.ACTORS)):
            self.ACTORS[i].add_dm.remote(command, channel)

    def get_dm(self, channel=1):
        return copy.copy(self.getattr('dm_channels')[channel])

    def get_dm_total(self):
        return self.getattr('dm_total')
    
    def snap_camsci(self):
        pending_ims = []
        for i in range(self.Nactors):
            future_ims = self.ACTORS[i].snap_camsci.remote()
            pending_ims.append(future_ims)

        ims = ray.get(pending_ims)
        ims = xp.array(ims)
        camsci_im = xp.sum(ims, axis=0)/self.Imax_ref

        return camsci_im

    def snap_camlo(self):
        pending_ims = []
        for i in range(self.Nactors):
            future_ims = self.ACTORS[i].snap_camlo.remote()
            pending_ims.append(future_ims)
            
        ims = ray.get(pending_ims)
        ims = xp.array(ims)
        camlo_im = xp.sum(ims, axis=0)

        if self.CAMLO is not None:
            noisy_im = 0.0
            for i in range(self.NCAMLO):
                noisy_im += self.CAMLO.add_noise(camlo_im)

            camlo_im = noisy_im / self.NCAMLO

        return camlo_im






