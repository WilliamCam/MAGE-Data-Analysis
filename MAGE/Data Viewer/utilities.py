# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 20:26:11 2024

@author: 21958742
"""


from scipy import signal
from scipy.signal import welch, csd
import matplotlib.pyplot as plt
from os import listdir
import numpy as np
from scipy import fft
from scipy.stats import norm
import pickle
from datetime import datetime

from lmfit.models import LorentzianModel
from lmfit.models import ConstantModel

#Class to store meta data to access a single data channel
class data_channel:
    def __init__(self, File, AI, channel):
        File = self.File
        AI = self.AI
        channel = self.channel

#Performs optimal filter on a segemnt of data given a template of the same size
def optimal_filter(data, template, Fs, *args, **kwargs):
    zero_pad = np.zeros(data.size - template.size) # zero pad template to match data size
    template_pad = np.append(template, zero_pad)
    fft_template = np.fft.fft(template_pad) # fourier transformed padded template

    power_dat, freq_PSD = plt.psd(data, Fs=Fs, **kwargs)
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

def fit_spectrum(data_channel, *args, plotting=True, return_peak = False, trim=(0,-1), **kwargs):
    
    f = data_channel.File
    Fs = f['AI ' + str(data_channel.AI)].attrs['Fs']
    dataI = f['AI ' + str(data_channel.AI) + '/CH ' + str(data_channel.channel+1) + '-I/Data'][:]*9.86e-10
    dataQ = f['AI ' + str(data_channel.AI) + '/CH ' + str(data_channel.channel+1) + '-Q/Data'][:]*9.86e-10
    
    fn, SdataI = welch(dataI, fs=Fs, **kwargs)
    fn, SdataQ = welch(dataQ, fs=Fs, **kwargs)
    Sx_trim = SdataI[trim[0]:trim[1]]
    Sy_trim = SdataQ[trim[0]:trim[1]]
    fn_trim = fn[trim[0]:trim[1]]
    
    peak_model = LorentzianModel()
    noise_model= ConstantModel()
    
    pars = peak_model.guess(Sx_trim, x=fn_trim)
    pars += noise_model.guess(Sx_trim, x=fn_trim)
    pars2 = peak_model.guess(Sx_trim, x=fn_trim)
    pars2 += noise_model.guess(Sx_trim, x=fn_trim)
    model = peak_model + noise_model

    out = model.fit(Sx_trim, pars, x=fn_trim)
    out2 = model.fit(Sy_trim, pars2, x=fn_trim)
    Gamma1, Gamma2 = out.params["sigma"].value, out2.params["sigma"].value
    fcenter1, fcenter2 = out.params["center"].value, out2.params["center"].value
    h1, h2 = out.params["height"].value + out.params["c"].value, out2.params["height"].value + out2.params["c"].value
    f_demod = f['AI ' + str(data_channel.AI)].attrs['Demod freqs AI ' + str(data_channel.AI)][data_channel.channel] # demodulation frequency

    Q1, Q2 = (fcenter1+f_demod)/(2*Gamma1), (fcenter2+f_demod)/(2*Gamma2)
    if out.params["sigma"].stderr == None:
        error1 = 0.99
    else:
        error1 =  out.params["sigma"].stderr/out.params["sigma"].value
    if out2.params["sigma"].stderr == None:
        error2 = 0.99
    else:
        error2 =  out2.params["sigma"].stderr/out2.params["sigma"].value
    #if Gamma1 > Gamma_max or Gamma2 > Gamma_max or error1 > error_max or error2 > error_max: #Ignore bad fits
        #print("Input AI " + str(AI) + ", Channel " + str(channel+1) + ":WARNING: Bad mode detected, channel will be ignored")
    if plotting==True:
        fig = plt.figure("IMPA DOWNLOAD")
        plt.axis('tight')
        ax = fig.add_subplot(111)
        ax.plot(fn_trim, Sx_trim, 'o', markersize=0.2)
        ax.set_title("Input AI " + str(data_channel.AI) + ", Channel " + str(data_channel.channel+1))
        #plt.plot(fn_n, out.init_fit, '--', label='initial fit')
        ax.plot(fn_trim, out.best_fit, '-', label='best fit I')
        ax.plot(fn_trim, out2.best_fit, '-', label='best fit Q')
        ax.set_yscale('log')
        #ax.set_xscale('log')
        ax.legend()
    if return_peak:
        return Q1,Q2,Gamma1,Gamma2,fcenter1,fcenter2, h1, h2
    else:
        return Q1,Q2,Gamma1,Gamma2,fcenter1,fcenter2