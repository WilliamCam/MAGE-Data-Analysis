from matplotlib import pyplot as plt
from scipy.signal import welch
from lmfit.models import LorentzianModel, ConstantModel, LinearModel
import numpy as np
import gc

#constants
kb = 1.380649e-23 #Boltzmann constant

class Detector():
    """A detector contains multiple channels."""
    def __init__(self, parent, name: str):
        self.parent = parent
        self.name = name
        self.channels = {}

    def __repr__(self):
        return f"<Detector {self.name} with {len(self.channels)} channels>"
    
    def add_channel(self, channel):
        self.channels[channel.name] = channel

    def get_channel(self, name: str):
        return self.channels.get(name)
    
    
    # Nice Pythonic access: det["ch0"]
    def __getitem__(self, key):
        return self.channels[key]


class Channel():
    # ...existing code...
    def __init__(self, parent, name, data = {}, is_IQ = True, **kwargs):
        self.parent = parent
        self.data = data
        self.is_IQ = is_IQ
        self.IQ_identifier = kwargs.pop('IQ_identifier', ('-I', '-Q'))
        self.name = name
        self.fit_result = {}
        self.events = {}

    def apply(self, func, **kwargs):
        """Apply a processing function to the channel."""
        self.data = func(self.data, **kwargs)

    def clear_data(self):
        """Remove the channel's loaded datasets and cached fit results from memory.

        Preserve event metadata so detected or defined events remain available.
        """
        if isinstance(self.data, dict):
            self.data.clear()
        else:
            self.data = {}

        if isinstance(self.fit_result, dict):
            self.fit_result.clear()
        else:
            self.fit_result = {}

        gc.collect()

    def consolidate_iq_pair(self, iq_channel):
        """Consolidate this channel with an I/Q pair channel.
        
        Args:
            iq_channel: The paired I/Q channel to consolidate with
        """
        if not self.is_IQ or not iq_channel.is_IQ:
            return
            
        # Store I and Q data in a dictionary
        if self.IQ_identifier[0] in self.name and self.IQ_identifier[1] in iq_channel.name:
            # This is I channel, iq_channel is Q channel
            self.data = {
                'I': self.data,
                'Q': iq_channel.data
            }
        elif self.IQ_identifier[1] in self.name and self.IQ_identifier[0] in iq_channel.name:
            # This is Q channel, iq_channel is I channel
            self.data = {
                'I': iq_channel.data,
                'Q': self.data
            }
        
        # Update channel name to remove I/Q suffix
        base_name = self.name
        for suffix in self.IQ_identifier:
            base_name = base_name.replace(suffix, '')
        self.name = base_name
        self.is_IQ = False  # No longer an I/Q channel

    def fit_lorentzian(self, fs, nfft, **kwargs):
        # Determine PSD of channel data and fit lorentzian to mode thermal peak
        if not self.data:
            raise ValueError("No data initilised in Channel ${self.name}")
        if not self.is_IQ:
            dataI = self.data['I']
            dataQ = self.data['Q']
            fn, SdataI = welch(dataI, fs=fs, nperseg = 2*nfft-1, scaling = 'density')
            fn, SdataQ = welch(dataQ, fs=fs, nperseg = 2*nfft-1, scaling = 'density')
            fit_result_I = lorentzian_fit_thermalpeak(SdataI, fn, **kwargs)
            fit_result_Q = lorentzian_fit_thermalpeak(SdataQ, fn, **kwargs)
            self.fit_result['I'] = fit_result_I
            self.fit_result['Q'] = fit_result_Q
            return fit_result_I, fit_result_Q
        else:
            data = self.data
            fn, Sdata = welch(data, fs=fs, nperseg = 2*nfft-1, scaling = 'density')
            fit_result = lorentzian_fit_thermalpeak(Sdata, fn, **kwargs)
            self.fit_result = fit_result
            return fit_result
        
    def event_plot(self, event_name, span=1000):
        if not self.events[event_name]:
            raise KeyError("Event does not exist")
        if not self.data:
            raise ValueError("Data not initalised in Channel")
        event_index = self.event[event_name]['index']
        if not self.is_IQ:
            dataI = self.data['I']
            dataQ = self.data['Q']
        plt.ion()
        fig = plt.figure()
        plt.axis('tight')
        fig.clf()
        ax = fig.add_subplot(111)
        ax.plot(dataI[span//2-event_index:span//2+event_index])
        ax.plot(dataI[span//2-event_index:span//2+event_index])
        ax.legend()
        plt.pause(0.05)
        plt.draw()
        
        
    def calibrate_strain(self, f, Q, Rlambda, Meff, G, simulate_with_noise = False):
            if not self.data:
                raise ValueError("No data initilised in Channel ${self.name}")
            if not self.is_IQ:
                dataI = self.data['I']
                dataQ = self.data['Q']
            if simulate_with_noise:
                dataI_noise = np.random.normal(np.mean(dataI), np.std(dataI), size = len(dataI))
                dataQ_noise = np.random.normal(np.mean(dataQ), np.std(dataQ), size = len(dataQ))
                R = np.sqrt(dataI_noise**2+dataQ_noise**2)
            else:    
                R = np.sqrt(dataI**2+dataQ**2)
            kappa = np.sqrt(f*2*np.pi*Meff/(Q * Rlambda))
            strain = np.fft.ifft(np.fft.fft(R[1:]/G)/(kappa*f*2*np.pi)).real
            Temperature = np.mean(0.5*(f*2*np.pi)**2*Meff*strain**2/kb)
            return strain, Temperature

    
def lorentzian_fit_thermalpeak(mag, fn, fdemod=0.0, Plot=False, span=300, peak = None):
    """
    Fit a Lorentzian model to thermal peak data and return the fitted parameters, their errors, and the Q-factor.

    Parameters:
    - mag: The magnitude data (spectrum).
    - fn: The frequency vector corresponding to the magnitude data.
    - fdemod: The demodulation frequency to adjust the resonance frequency.
    - Plot: Whether to plot the fit (default is False).
    - start: Start index for the fitting range (default is 0).
    - stop: Stop index for the fitting range (default is 1600).
    
    Returns:
    - f_res: The fitted resonance frequency (Hz).
    - sigma: The fitted Lorentzian width (Hz).
    - integral: The amplitude of the Lorentzian peak.
    - Q: The Q-factor of the resonance.
    - const: The constant offset in the data (from the linear background).
    - f_res_err: The error on the resonance frequency.
    - sigma_err: The error on the Lorentzian width.
    - Q_err: The error on the Q-factor.
    - const_err: The error on the constant background.
    """
    # Lorentzian and constant background models
    lor_mod = LorentzianModel(prefix='lor_')
    lin_mod = ConstantModel(prefix='lin_')
    if not peak:
        peak_index = np.where(mag==np.max(mag))[0][0]
    else:
        peak_index = np.abs(fn - peak).argmin()
    if peak_index-span < 1:
        fn_fit = fn[0:peak_index+span]
        linear_mag = mag[0:peak_index+span]
    else:
        fn_fit = fn[peak_index-span:peak_index+span]
        linear_mag = mag[peak_index-span:peak_index+span]

    # Make initial guesses for the parameters
    pars = lor_mod.guess(linear_mag, x=fn_fit)
    pars.update(lin_mod.make_params())

    # Add the Lorentzian and constant models together
    mod = lin_mod + lor_mod

    # Perform the fit
    out = mod.fit(linear_mag, pars, x=fn_fit)

    # Extract the fitted parameters
    Gamma = out.params["lor_sigma"].value * 2  # Lorentzian width (FWHM), multiply by 2
    f_res = out.params["lor_center"].value + fdemod  # Resonance frequency, adjusted by demodulation frequency
    Q = f_res / Gamma  # Quality factor
    integral = out.params["lor_amplitude"].value  # Amplitude of the Lorentzian peak
    height = out.params["lor_height"].value #Unnormalised height of Lorentzian peak
    sigma = out.params["lor_sigma"].value  # Lorentzian width (standard deviation)
    noise_val = out.params["lin_c"].value

    # Calculate the errors (standard errors of the parameters)
    f_res_err = out.params["lor_center"].stderr if out.params["lor_center"].stderr else 0.99  # Error on f_res
    sigma_err = out.params["lor_sigma"].stderr if out.params["lor_sigma"].stderr else 0.99  # Error on sigma
    Q_err = Q * np.sqrt((f_res_err / f_res) ** 2 + (sigma_err / Gamma) ** 2)  # Error on Q

    # Plot the results if requested
    if Plot:
        plt.ion()
        fig = plt.figure("IMPA DOWNLOAD")
        plt.axis('tight')
        fig.clf()
        ax = fig.add_subplot(111)
        ax.plot(fn_fit, linear_mag, marker= 'o', markersize=0.4)
        ax.set_title(fdemod)
        ax.plot(fn_fit, out.best_fit, '-', label='fit')
        ax.set_yscale('log')
        #ax.set_xscale('log')
        ax.legend()
        plt.pause(0.05)
        plt.draw()

    # Return the fitted parameters and their errors
    ret = {'centre_freqeuncy': f_res, 'linewidth': sigma, 'amplitude': integral, 'Q_factor' : Q, 'center_frequency_error': f_res_err, 'linewidth_error': sigma_err, 'Q_factor_error': Q_err, 'height': height, 'noise_level': noise_val}
    return ret

