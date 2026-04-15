import h5py
import numpy as np
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import os
from os import listdir
from scipy import signal
from scipy.signal import welch, csd
from scipy.signal import find_peaks
import lmfit
from scipy.fft import fft, fftfreq
import scipy.constants
from lmfit.models import LorentzianModel, ConstantModel, LinearModel
import sys
import os

def optimal_filter(data, template, Fs, NFFT):
    fft = np.fft.fft(data) # fourier transformed data
    zero_pad = np.zeros(data.size - template.size) # zero pad template to match data size
    template_pad = np.append(template, zero_pad)
    fft_template = np.fft.fft(template_pad) # fourier transformed padded template
    plt.plot()
    power_dat, freq_PSD = plt.psd(data, Fs=Fs, NFFT = NFFT, visible = True)
    freq_dat = np.fft.fftfreq(data.size)*Fs #fourier frequencies corresponding to data partition
    power_spec = np.interp(freq_dat, freq_PSD, power_dat)
    
    val_cal = np.max(template)
    OF = np.fft.ifft(fft_template*fft_template.conjugate()/power_spec).real
    K = val_cal/np.amax(OF)
    
    
    df = np.abs(freq_dat[1] - freq_dat[2])
    opt_filter = K * fft * fft_template.conjugate() / power_spec #optimal filter
    dat_filt = np.fft.ifft(opt_filter) #revert to time domain for filter output
    
    sigmasq = 2*(K**2 * fft_template * fft_template.conjugate() / power_spec).sum() * df 
    sigma = np.sqrt(np.abs(sigmasq))
    SNR = np.abs(2*dat_filt) / (sigma)
    return SNR, dat_filt

def get_resistances(filename):
    fresistance = open(filename= '/home/leo_maria/Desktop/UWA/MAGE/calibration/SQUID gain models.txt')
    resistances = np.genfromtxt(fresistance, delimiter='\t')
    return fresistance

def get_squid_gain(filename):
    gains = open(filename='home/leo_maria/Desktop/UWA/MAGE/October/Singletone_last/Vphi_last_singletone.txt')
    return gains

#TODO: generalise naming convention for Demod Freqs, date/time string, Fs, etc.
def get_meta_data(filename, IQ_channels = True):
    '''Reads file of experimental run and retrieves important information'''
    meta_dict={}
    f = h5py.File(filename, 'r')
    N_detectors = len(f.keys())
    detector_names = list(f.keys())
    if IQ_channels:
        N_channels = len(f[detector_names[0]].keys()) // 2
    else:
        N_channels = len(f[detector_names[0]].keys())
    Fs = f[detector_names[0]].attrs['Fs']
    _lo_frequencies={}
    for ai,name in enumerate(detector_names):
        lo_frequencies = np.zeros(N_channels)
        for ch in range(N_channels):
            lo_frequencies[ch] = f['AI ' + str(ai)].attrs['Demod freqs AI ' + str(ai)][ch]
        _lo_frequencies[name]=lo_frequencies 
    
    t_start_string = str(f[detector_names[0]].attrs['date/time string'])
    t_start = datetime.strptime(t_start_string, 'UTC %d-%m-%y %H:%M:%S.%f ')
    meta_dict['N_detectors'] = N_detectors
    meta_dict['N_samples'] = len(f[detector_names[0] + '/CH 1-I/Data'][:])
    meta_dict['N_channels'] = N_channels
    meta_dict['sample_rate'] = Fs
    meta_dict['lo_frequencies'] = _lo_frequencies
    meta_dict['start_time'] = t_start
    return meta_dict


def get_meta_data_from_first_file(filename):
    '''Reads first file of experimental run and retrieves important information. Returns number of inputs (2), number of channels (16), number of samples in file, sample frequency, time interval between consecutive samples, demodulation frequencies, date of start'''
    f = h5py.File(filename, 'r')
    Ninputs = len(f.keys())
    Nchannels = len(f['AI 0'].keys())//2
    Nsample = len(f['AI 0/CH 1-I/Data'][:])
    Fs = f['AI 0'].attrs['Fs']
    dt = 1/Fs
    f_demods=np.zeros((Ninputs, Nchannels))
    for ai in range(Ninputs):
        for ch in range(Nchannels):
            f_demods[ai, ch] = f['AI ' + str(ai)].attrs['Demod freqs AI ' + str(ai)][ch]
    
    t_start_string = str(f['AI 0'].attrs['date/time string'])
    t_start = datetime.strptime(t_start_string, 'UTC %d-%m-%y %H:%M:%S.%f ')
    return Ninputs, Nchannels, Nsample, Fs, dt, f_demods, t_start

def retrieve_IQ_td(filename, Ninputs=2, Nchannels=16, AI_scaling_gain =3.09758E-5 ):  # check that adc conversion to volts is correct
    f = h5py.File(filename, 'r')
    Ninputs = len(f.keys())
    Nchannels = len(f['AI 0'].keys())//2
    Nsample = len(f['AI 0/CH 1-I/Data'][:])
    data=np.zeros((Ninputs, Nchannels, 2, Nsample))
    for AI in range(Ninputs):
        for channel in range(Nchannels):
            dataI = f['AI ' + str(AI) + '/CH ' + str(channel+1) + '-I/Data'][:]*AI_scaling_gain**2
            dataQ = f['AI ' + str(AI) + '/CH ' + str(channel+1) + '-Q/Data'][:]*AI_scaling_gain**2
            data[AI, channel, 0, :] = dataI
            data[AI, channel, 1, :] = dataQ
    return data

def psd_from_IQ_td(iqdata, Fs, NFFT=2**13):
    Ninputs = np.size(iqdata, axis=0)
    Nchannels = np.size(iqdata, axis=1)
    Sx = np.zeros((Ninputs, Nchannels, NFFT))
    Sy = np.zeros_like(Sx)
    Sr = np.zeros_like(Sx)
    fns = np.zeros_like(Sx)
    for AI in range(Ninputs):
        for channel in range(Nchannels):
            fn, SdataI = welch(iqdata[AI, channel, 0, :], fs=Fs, nperseg = 2*NFFT-1, scaling = 'density')
            fn, SdataQ = welch(iqdata[AI, channel, 1, :], fs=Fs, nperseg = 2*NFFT-1, scaling = 'density')
            fn, SdataR = welch(np.sqrt(iqdata[AI, channel, 0, :]**2 + iqdata[AI, channel, 1, :]**2), fs=Fs, nperseg = 2*NFFT-1, scaling = 'density' )
            Sx[AI, channel, :] = SdataI
            Sy[AI, channel, :] = SdataQ
            Sr[AI, channel, :] = SdataR
            fns[AI, channel, :] = fn
    return fns, Sx, Sy, Sr

def spectrum_from_IQ_td(iqdata, Fs, NFFT=2**13):
    Ninputs = np.size(iqdata, axis=0)
    Nchannels = np.size(iqdata, axis=1)
    Sx = np.zeros((Ninputs, Nchannels, NFFT))
    Sy = np.zeros_like(Sx)
    Sr = np.zeros_like(Sx)
    fns = np.zeros_like(Sx)
    for AI in range(Ninputs):
        for channel in range(Nchannels):
            fn, SdataI = welch(iqdata[AI, channel, 0, :], fs=Fs, nperseg = 2*NFFT-1, scaling = 'spectrum')
            fn, SdataQ = welch(iqdata[AI, channel, 1, :], fs=Fs, nperseg = 2*NFFT-1, scaling = 'spectrum')
            fn, SdataR = welch(np.sqrt(iqdata[AI, channel, 0, :]**2 + iqdata[AI, channel, 1, :]**2), fs=Fs, nperseg = 2*NFFT-1, scaling = 'spectrum' )
            Sx[AI, channel, :] = SdataI
            Sy[AI, channel, :] = SdataQ
            Sr[AI, channel, :] = SdataR
            fns[AI, channel, :] = fn
    return fns, Sx, Sy, Sr

def lorentzian_fit_thermalpeak(mag, fn, fdemod, Plot=False, span=300):
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
    peak_index = np.where(mag==np.max(mag))[0][0]
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
        plt.pause(0.05)
        plt.draw()
        fig.clf()
        plt.plot(fn_fit, linear_mag, 'o', label='Data')
        plt.title('Fitted Lorentzian Peak: Frequency = {:.2f} Hz'.format(f_res))
        plt.plot(fn_fit, out.best_fit, '-', label='Best Fit')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD (V/√Hz)')
        plt.legend()
        plt.show()

    # Return the fitted parameters and their errors
    ret = {'centre_freqeuncy': f_res, 'linewidth': sigma, 'amplitude': integral, 'Q_factor' : Q, 'center_frequency_error': f_res_err, 'linewidth_error': sigma_err, 'Q_factor_error': Q_err, 'height': height, 'noise_level': noise_val}
    return ret

def lorentzian_fit_thermalpeak_bis_onlyFandQ(mag, fn, fdemod, ai, ch, Plot=False, span=300):
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

    # Select the frequency and magnitude data to fit
    peak_index = np.where(mag==np.max(mag))[0][0]
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
    sigma = out.params["lor_sigma"].value  # Lorentzian width (standard deviation)
    const = out.params["lin_c"].value  # Constant offset (background)

    # Calculate the errors (standard errors of the parameters)
    f_res_err = out.params["lor_center"].stderr if out.params["lor_center"].stderr else 0.99  # Error on f_res
    sigma_err = out.params["lor_sigma"].stderr if out.params["lor_sigma"].stderr else 0.99  # Error on sigma
    Q_err = Q * np.sqrt((f_res_err / f_res) ** 2 + (sigma_err / Gamma) ** 2)  # Error on Q
    const_err = out.params["lin_c"].stderr if out.params["lin_c"].stderr else 0.99  # Error on constant background

    # Plot the results if requested
    if Plot:
        plt.plot(fn_fit, linear_mag, 'o', label='Data')
        plt.title('Fitted Lorentzian Peak: Frequency = {:.2f} Hz'.format(f_res))
        plt.plot(fn_fit, out.best_fit, '-', label='Best Fit')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD (V/√Hz)')
        plt.legend()
        plt.show()

    # Return the fitted parameters and their errors
    return f_res, Q

def generate_template(ai, channel, f_res, Qs, Fs, dt, Nsamples):
    """
    Generate an exponential template signal and its corresponding time vector.
    
    Args:
    - ai: index for AI channel.
    - channel: the channel index.
    - f_res: array of resonant frequencies.
    - Fs: sampling frequency.
    - dt: time step.
    - Nsamples: number of samples.
    
    Returns:
    - signal: the exponential decay signal.
    - t_sig: time vector for the signal.
    """
    tau1 = Qs[ai, channel] /np.pi / f_res[ai, channel]  # Decay constant, assumed formula
    Nfilter = int(Fs * 5 * tau1)  # Number of filter taps, based on the decay constant
    
    t_sig = dt * np.linspace(0, Nfilter, Nfilter)
    signal = 1e-5 * np.exp(-t_sig / tau1)  # Exponential decay signal
    
    if signal.size == 0:
        raise ValueError("Generated signal is empty.")
    
    return signal, t_sig, tau1

def pad_and_roll_signal(signal, Rdat, roll_offset=50000):
    """
    Zero-pad and roll the signal to match the data size.
    
    Args:
    - signal: the template signal.
    - Rdat: reference data array to match the size.
    - roll_offset: number of points to roll the signal.
    
    Returns:
    - template_pad: the padded and rolled signal.
    """
    if signal.size == 0 or Rdat.size == 0:
        raise ValueError("Signal or reference data array is empty.")
    
    # Check if signal is shorter than Rdat and pad with zeros to match Rdat size
    if signal.size >= Rdat.size:
        raise ValueError("Signal should be shorter than Rdat.")
    
    # Zero pad the signal to match the size of Rdat
    padded_signal = np.pad(signal, (0, Rdat.size - signal.size), 'constant', constant_values=0)
    
    # Roll the signal by the specified offset
    template_pad = np.roll(padded_signal, roll_offset)
    
    return template_pad

def add_noise_to_signal(template_pad, Rdat):
    """
    Add noise to the signal.
    
    Args:
    - template_pad: the template signal after padding and rolling.
    - Rdat: the reference data used to determine the noise level.
    
    Returns:
    - h_inject: noisy signal with template and Gaussian noise.
    - h_inject2: purely noise (Gaussian) signal.
    - h_inject3: real data plus the template signal.
    """
    noise_std = 1.66 * np.std(Rdat)
    h_inject = template_pad + np.abs(np.random.normal(0, noise_std, size=len(Rdat)))
    h_inject2 = np.abs(np.random.normal(0, noise_std, size=len(Rdat)))
    h_inject3 = template_pad + Rdat
    return h_inject, h_inject2, h_inject3

def plot_filtered_data(tn, nn, span, h_inject, h_inject3, dat_filt, dat_filt3):
    """
    Plot the filtered data and compare different signals.
    
    Args:
    - tn: time vector.
    - nn: index for the time region of interest.
    - span: the width of the time window for plotting.
    - h_inject: the noisy signal.
    - h_inject3: the real data plus the simulated signal.
    - dat_filt: filtered data for noisy signal.
    - dat_filt3: filtered data for real data plus simulated signal.
    """
    plt.plot(tn[nn-span//2:nn+span//2], h_inject[nn-span//2:nn+span//2], label='Random Noise Simulated Signal')
    plt.plot(tn[nn-span//2:nn+span//2], h_inject3[nn-span//2:nn+span//2], label='Real Data + Simulated Signal')
    plt.legend()
    plt.show()

    plt.plot(tn[nn-span//2:nn+span//2], abs(dat_filt)[nn-span//2:nn+span//2], label='Random Noise + Simulated Signal POST FILTERING')
    plt.plot(tn[nn-span//2:nn+span//2], abs(dat_filt3)[nn-span//2:nn+span//2], label='Real Data + Simulated Signal POST FILTERING')
    plt.legend()
    plt.show()

def plot_snr_colormesh(SNR_arr, tau_vec, tn, nn, Span, tau, Fs, NFFT):
    """
    Generate a colormesh plot of the Signal-to-Noise Ratio (SNR) for varying tau values.
    
    Args:
    - SNR_arr: Array of SNR values for each template and time span.
    - tau_vec: Vector of tau values for the templates.
    - tn: Time vector.
    - nn: Index for the region of interest in the time vector.
    - Span: The width of the time window for plotting.
    - tau: Reference tau value for scaling the tau vector.
    - Fs: Sampling frequency (for context).
    - NFFT: Number of FFT points for filtering (for context).
    
    Returns:
    - fig: The figure object for the plot.
    - ax: The axes object for the plot.
    """
    # Create meshgrid for plotting
    XX, YY = np.meshgrid(tn[nn-Span//2:nn+Span//2], tau_vec/tau)
    ZZ = SNR_arr[:,:]
    
    # Create figure and axis for plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Generate the colormesh plot
    cax = ax.pcolormesh(XX, YY, ZZ, cmap='viridis')
    
    # Set the scale for the y-axis to logarithmic
    ax.set_yscale('log')
    
    # Add color bar with label
    fig.colorbar(cax, ax=ax, label='SNR')
    
    # Label the axes
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$\tau$ (scaled)')
    ax.set_title('Signal-to-Noise Ratio (SNR) vs. Time and Tau')
    
    # Show the plot
    plt.show()
    
    return fig, ax

# Function to generate the decay template
def generate_decay_template(t_sig, tau):
    return np.exp(-t_sig / (2 * tau))

# Function to apply quality cuts and detect diverging transient features
def check_for_transients(SNR, transient_SNR1, transient_SNR2, peaks):
    discarded = []
    for i, event_i in enumerate(peaks[0]):
        print(f"{SNR[event_i]:1.2f}, {transient_SNR1[event_i]:1.2f}, {transient_SNR2[event_i]:1.2f}")
        if SNR[event_i] < transient_SNR1[event_i] or SNR[event_i] < transient_SNR2[event_i]:
            print(f"Transiently divergent feature detected of SNR {SNR[event_i]:1.2f}, performing quality cut")
            discarded.append(event_i)
    return discarded

# Function to catalog detected events
def catalog_event(event_catalogue, event_name, event_time, SNR, ai, channel, fdemods, Rdat, subrun, event_i):
    event_name = f"{event_name}-AI{ai}-ch{channel}-SNR {SNR[event_i]:1.2f}"
    event_catalogue[event_name] = {
        'time': event_time,
        'SNR': SNR[event_i],
        'input AI': ai,
        'channel': channel + 1,
        'frequency': fdemods[ai, channel],
        'noise': np.mean(Rdat),
        'file N': subrun,
        'index': event_i
    }
    print(f'Event {event_name} Saved')

# Main loop to filter data and detect events
def filter_and_detect_events(Ninputs, Nchannels, Nsamples, Qs, f_res, iq, Fs, NFFT, fdemods, dt, file_start, subrun, event_catalogue):
    Nfilter = NFFT
    tn = np.linspace(0, Nsamples * dt, Nsamples)
    t_sig = dt * np.linspace(0, Nfilter, Nfilter)
    max_sigma = 10
    max_error = 0.2
    
    for ai in range(Ninputs):
        for channel in range(Nchannels):
            tau = Qs[ai, channel] / np.pi / f_res[ai, channel]
            template = generate_decay_template(t_sig, tau)
            
            # Create signal magnitude from the I/Q data
            Rdat = np.sqrt(iq[ai, channel, 0, :]**2 + iq[ai, channel, 1, :]**2)
            
            # Apply optimal filter
            SNR, dat_filt = optimal_filter(Rdat, template, Fs, NFFT)
            
            # Find peaks in the SNR
            threshold = 0.25
            peaks = find_peaks(SNR, height=threshold, distance=int(tau * Fs), width=[100, 5e6], rel_height=1.0)
            
            # Create event day string
            event_day_string = datetime.strftime(file_start, "%d-%m-%y")
            
            # Check if there are any peaks
            if len(peaks[0]) > 0:
                # Generate diverging templates for transient detection
                diverge_template1 = generate_decay_template(t_sig, tau / 10.0)
                diverge_template2 = generate_decay_template(t_sig, tau / 100.0)
                diverge_template3 = generate_decay_template(t_sig, tau / 1000.0)
                
                # Apply optimal filter with diverging templates
                transient_SNR1, dat_filt1 = optimal_filter(Rdat, diverge_template1, Fs, NFFT)
                transient_SNR2, dat_filt2 = optimal_filter(Rdat, diverge_template2, Fs, NFFT)
                transient_SNR3, dat_filt3 = optimal_filter(Rdat, diverge_template3, Fs, NFFT)
                
                # Plot filtered output
                #plot_filtered_output(SNR, tn, peaks, ai, channel, subrun, start=1400, stop=1600)
                
                # Check for transients and quality cuts
                discarded = check_for_transients(SNR, transient_SNR1, transient_SNR2, peaks)
                
                # Catalog detected events
                for event_i in peaks[0]:
                    if event_i not in discarded:
                        event_time = file_start + timedelta(seconds=event_i * dt)
                        event_name = datetime.strftime(event_time, "%d%m%y-%H:%M:%S")
                        catalog_event(event_catalogue, event_name, event_time, SNR, ai, channel, fdemods, Rdat, subrun, event_i)
    
    return event_catalogue

# Function to plot filtered output for inspection
def plot_filtered_output(SNR, tn, peaks, ai, channel, subrun, start, stop):
    fig = plt.figure("Filtered Output", figsize=(15, 15))
    ax = fig.add_subplot(211)
    axx = fig.add_subplot(212)
    ax.set_xlim(start, stop)
    axx.set_xlim(start, stop)
    ax.set_title(f"File {subrun} Input AI {ai}, Channel {channel}")
    ax.plot(tn, Rdat / np.mean(Rdat), label='Normalised Magnitude')
    ax.plot(tn, SNR, label='SNR')
    ax.plot(peaks[0] * dt, peaks[1]['peak_heights'], linestyle=' ', marker='x', color='black')
    plt.show()

def R_to_strain(Rdata, Fs, fdemods, Vphi, Min, kappa):
    R_fd = np.fft.rfft(Rdata)
    freqs_fft = np.fft.fftfreq(R_fd.size, 1.0 / Fs )
    u_fd = 1 / (Min*2*np.pi*(fdemods + freqs_fft) * kappa ) / (2000*Vphi) * R_fd
    u =np.fft.irfft(u_fd)
    return u

def get_eff_mass(path):
    feffective_mass = open(path, 'r')
    mode_distributions = np.genfromtxt(feffective_mass, delimiter=',', skip_header=1)
    meff = mode_distributions[:,1]
    return meff

def filter_strain_data_one_file(strain, tau, Fs, NFFT, file_start, fdemods, subrun, Ninputs=2, Nchannels=16):
    SNR = np.zeros_like(strain)
    filtered_strain = np.zeros_like(strain)
    event_catalogue={}
    for ai in range(Ninputs):
        for ch in range(Nchannels):
            Nfilter=int(Fs*5*tau[ai,ch])
            t_sig = 1/Fs*np.linspace(0, Nfilter, Nfilter)
            template = np.exp(-t_sig/(2 * tau[ai,ch])) # template construction
            SNR[ai,ch], filtered_strain[ai,ch] = optimal_filter(strain[ai,ch], template, Fs, NFFT)

            # Find peaks in the SNR
            threshold = 0.25
            try:
                peaks = find_peaks(SNR[ai,ch], height=threshold, distance=int(tau[ai,ch] * Fs), width=[100, 5e6], rel_height=1.0)
            except:
                print('Had issues with peaks for ai' + str(ai) + ', channel ' + str(ch) )
                continue

            # Create event day string
            event_day_string = datetime.strftime(file_start, "%d-%m-%y")

            # Check if there are any peaks
            if len(peaks[0]) > 0:
                # Generate diverging templates for transient detection
                diverge_template1 = np.exp(-t_sig/(2 * tau[ai,ch]/10)) # template construction
                diverge_template2 = np.exp(-t_sig/(2 * tau[ai,ch]/100)) # template construction
                diverge_template3 = np.exp(-t_sig/(2 * tau[ai,ch]/10000)) # template construction
                
                # Apply optimal filter with diverging templates
                transient_SNR1, dat_filt1 = optimal_filter(strain[ai,ch], diverge_template1, Fs, NFFT)
                transient_SNR2, dat_filt2 = optimal_filter(strain[ai,ch], diverge_template2, Fs, NFFT)
                transient_SNR3, dat_filt3 = optimal_filter(strain[ai,ch], diverge_template3, Fs, NFFT)
                
                # Plot filtered output
                #plot_filtered_output(SNR, tn, peaks, ai, channel, subrun, start=1400, stop=1600)
                
                # Check for transients and quality cuts
                discarded = check_for_transients(SNR[ai,ch, :], transient_SNR1, transient_SNR2, peaks)
                
                # Catalog detected events
                for event_i in peaks[0]:
                    if event_i not in discarded:
                        event_time = file_start + timedelta(seconds=event_i / Fs)
                        event_name = datetime.strftime(event_time, "%d%m%y-%H:%M:%S")
                        catalog_event(event_catalogue, event_name, event_time, SNR[ai,ch], ai, ch, fdemods, strain[ai,ch], subrun, event_i)

    return SNR, filtered_strain, event_catalogue

#Helper function for reading calibration data in txt files.
def read_two_column_data(file_path):
    data = np.loadtxt(file_path)
    column_1 = data[:, 0]  # First column
    column_2 = data[:, 1]  # Second column
    return np.array([column_1, column_2])

