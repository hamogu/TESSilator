# imports
import logging
__all__ = ['check_for_jumps', 'gauss_fit', 'gauss_fit_peak', 'get_second_peak', 'is_period_cont', 'logger', 'mean_of_arrays', 'moving_average', 'run_ls', 'sin_fit']


import warnings

# Third party imports
import numpy as np
import os

from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.timeseries import LombScargle
from astropy.wcs import WCS
from astropy.io import fits
import astropy.units as u
from astropy.stats import akaike_info_criterion_lsq


from scipy.stats import median_abs_deviation as MAD
from scipy.optimize import curve_fit
import itertools as it

from collections.abc import Iterable

# Local application imports
from .fixedconstants import *


# initialize the logger object
logger = logging.getLogger(__name__)
#logger_aq = logging.getLogger("astroquery")
logger.setLevel(logging.ERROR)    






def check_for_jumps(time, flux, eflux, lc_part, n_avg=10, thresh_diff=10.):
    '''Identify if the lightcurve has jumps.
    
    A jumpy lightcurve is one that has small contiguous data points that change in flux significantly compared to the amplitude of the lightcurve. These could be due to some instrumental noise or response to a non-astrophysical effect. They may also be indicative of a stellar flare or active event.
    
    This function takes a running average of the differences in flux, and flags lightcurves if the absolute value exceeds a threshold. These will be flagged as "jumpy" lightcurves.

    parameters
    ----------
    time : `Iterable`
        The time coordinate
    flux : `Iterable`
        The original, normalised flux values
    eflux : `Iterable`
        The error on "flux"
    lc_part : `Iterable`
        The running index for each contiguous data section in the lightcurve
    n_avg : `int`, optional, default=10
        The number of data points to calculate the running average
    thresh_diff : `float`, optional, default=10.
        The threshold value, which, if exceeded, will yield a "jumpy" lightcurve

    returns
    -------
    jump_flag : `Boolean`
        This will be True if a jumpy lightcurve is identified, otherwise False.
    '''
    
    jump_flag = False
    
    for lc in np.unique(lc_part):
        g = np.array(lc_part == lc)
        f_mean = moving_average(flux[g], n_avg)
        t_mean = moving_average(time[g], n_avg)
        
        f_shifts = np.abs(np.diff(f_mean))

        median_f_shifts = np.median(f_shifts)
        max_f_shifts = np.max(f_shifts)
        if max_f_shifts/median_f_shifts > thresh_diff:
            jump_flag = True
            return jump_flag

    return jump_flag


def gauss_fit(x, a0, x_mean, sigma):
    '''Construct a simple Gaussian.

    Return Gaussian values from a given amplitude (a0), mean (x_mean) and
    uncertainty (sigma) for a distribution of values

    parameters
    ----------
    x : `Iterable`
        list of input values
    a0 : `float`
        Amplitude of a Gaussian
    x_mean : `float`
        The mean value of a Gaussian
    sigma : `float`
        The Gaussian uncertainty

    returns
    -------
    gaussian : `list`
        A list of Gaussian values.
    '''

    gaussian = a0*np.exp(-(x-x_mean)**2/(2*sigma**2))
    return gaussian


def gauss_fit_peak(period, power):
    '''
    Applies the Gaussian fit to the periodogram. If there are more than 3 data
    points (i.e., more data points than fixed parameters), the "gauss_fit"
    module is used to return the fit parameters. If there are 3 or less points,
    the maximum peak is located and 9 data points are interpolated between the
    2 neighbouring data points of the maximum peak, and the "gauss_fit" module
    is applied.
    
    parameters
    ----------
    period : `Iterable`
        The period values around the peak.
    power : `Iterable`
        The power values around the peak.
        
    returns
    -------
    popt : `list`
        The best-fit Gaussian parameters: A, B and C where A is the amplitude,
        B is the mean and C is the uncertainty.
    ym : `list`
        The y values calculated from the Gaussian fit.
    '''
    if len(period) > 3:
        try:
            popt, _ = curve_fit(gauss_fit, period, power,
                                bounds=([0, period[0], 0],
                                        [1., period[-1], period[-1]-period[0]]))
            ym = gauss_fit(period, *popt)
        except:
            print(f"Couldn't find the optimal parameters for the Gaussian fit!")
            logger.error(f"Couldn't find the optimal parameters for the Gaussian fit!")
            p_m = np.argmax(power)
            peak_vals = [p_m-1, p_m, p_m+1]
            x = period[peak_vals]
            y = power[peak_vals]
            xvals = np.linspace(x[0], x[-1], 9)
            yvals = np.interp(xvals, x, y)
            popt, _ = curve_fit(gauss_fit, xvals, yvals,
                                bounds=(0, [1., np.inf, np.inf]))
            ym = gauss_fit(xvals, *popt)     

    else:
        p_m = np.argmax(power)
        peak_vals = [p_m-1, p_m, p_m+1]
        x = period[peak_vals]
        y = power[peak_vals]
        xvals = np.linspace(x[0], x[-1], 9)
        yvals = np.interp(xvals, x, y)
        popt, _ = curve_fit(gauss_fit, xvals, yvals,
                            bounds=(0, [1., np.inf, np.inf]))
        ym = gauss_fit(xvals, *popt)        
    return popt, ym
    
    
def get_second_peak(power):
    '''An algorithm to identify the second-highest peak in the periodogram

    parameters
    ----------
    power : `Iterable`
        A set of power values calculated from the periodogram analysis.

    returns
    -------
    a_g : `list`
        A list of indices corresponding to the Gaussian around the peak power.
    a_o : `list`
        A list of indices corresponding to all other parts of the periodogram.
    '''
    # Get the left side of the peak
    a = np.arange(len(power))

    p_m = np.argmax(power)
    x = p_m
    while (power[x-1] < power[x]) and (x > 0):
        x = x-1
    p_l = x
    p_lx = 0
    while (power[p_l] > 0.85*power[p_m]) and (p_l > 1):
        p_lx = 1
        p_l = p_l - 1
    if p_lx == 1:
        while (power[p_l] > power[p_l-1]) and (p_l > 0):
            p_l = p_l - 1
    if p_l < 0:
        p_l = 0

    # Get the right side of the peak
    x = p_m
    if x < len(power)-1:
        while (power[x+1] < power[x]) and (x < len(power)-2):
            x = x+1
        p_r = x
        p_rx = 0
        while (power[p_r] > 0.85*power[p_m]) and (p_r < len(power)-2):
            p_rx = 1
            p_r = p_r + 1
        if p_rx == 1:
           while (power[p_r] > power[p_r+1]) and (p_r < len(power)-2):
                p_r = p_r + 1
        if p_r > len(power)-1:
            p_r = len(power)-1
        a_g = a[p_l:p_r+1]
        a_o = a[np.setdiff1d(np.arange(a.shape[0]), a_g)] 
    elif x == len(power)-1:
        a_g = a[x]
        a_o = a[0:x]
    return a_g, a_o


def is_period_cont(d_target, d_cont, t_cont, frac_amp_cont=0.5):
    '''Identify neighbouring contaminants that may cause the periodicity.

    If the user selects to measure periods for the neighbouring contaminants
    this function returns a flag to assess if a contaminant may actually be
    the source causing the observed periodicity.

    parameters
    ----------
    d_target : `dict`
        A dictionary containing periodogram data of the target star.
    d_cont : `dict`
        A dictionary containing periodogram data of the contaminant star.
    t_cont : `astropy.table.Table`
        A table containing Gaia data for the contaminant star
    frac_amp_cont : `float`, optional, default=0.5
        The threshold factor to account for the difference in amplitude
        of the two stars. If this is high, then the contaminants will be
        less likely to be flagged as the potential source
    
    returns
    -------
    output : `str`
        | Either ``a``, ``b`` or ``c``.
        | (a) The contaminant is probably the source causing the periodicity
        | (b) The contaminant might be the source causing the periodicity
        | (c) The contaminant is not the source causing the periodicity
        
    '''
    per_targ = d_target["period_best"]
    per_cont = d_cont["period_best"]
    err_targ = d_target["Gauss_fit_peak_parameters"][2]
    err_cont = d_cont["Gauss_fit_peak_parameters"][2]
    amp_targ = d_target["pops_vals"][1]
    amp_cont = d_cont["pops_vals"][1]
    flux_frac = 10**(t_cont["log_flux_frac"])

    if abs(per_targ - per_cont) < (err_targ + err_cont):
        if amp_targ/amp_cont > (frac_amp_cont*flux_frac):
            output = 'a'
        else:
            output = 'b'
    else:
        output = 'c'
    return output
    
    
def mean_of_arrays(arr, num):
    '''Calculate the mean and standard deviation of an array which is split into N components.
    
    parameters
    ----------
    arr : `Iterable`
        The input array
    num : `int`
        The number of arrays to split the data (equally) into

    returns
    -------
    mean_out : `float`
        The mean of the list of arrays.
    std_out : `float`
        The standard deviation of the list of arrays.
    '''
    x = np.array_split(arr, num)
    ar = np.array(list(it.zip_longest(*x, fillvalue=np.nan)))
    mean_out, std_out = np.nanmean(ar, axis=0), np.nanstd(ar, axis=0) 
    return mean_out, std_out
    

def moving_average(x, w):
    '''Calculate the moving average of an array.
    
    parameters
    ----------
    x : `Iterable`
        The input data to be analysed.
    w : `int`
        The number of data points that the moving average will convolve.

    returns
    -------
    z : `np.array`
        An array of the moving averages
    '''
    
    z = np.convolve(x, np.ones(w), 'valid') / w
    return z


def run_ls(cln, n_sca=10, p_min_thresh=0.05, p_max_thresh=100., samples_per_peak=10, check_jump=False):
    '''Run Lomb-Scargle periodogram and return a dictionary of results.

    parameters
    ----------
    cln : `dict`
        A dictionary containing the lightcurve data. The keys must include
        | "time" -> The time coordinate relative to the first data point
        | "nflux" -> The detrended, cleaned, normalised flux values
        | "enflux" -> The uncertainty for each value of nflux
        | "lc_part" -> An running index describing the various contiguous sections
    p_min_thresh : `float`, optional, default=0.05
        The minimum period (in days) to be calculated.
    p_max_thresh : `float`, optional, default=100.
        The maximum period (in days) to be calculated.
    samples_per_peak : `int`, optional, default=10
        The number of samples to measure in each periodogram peak.
    check_jump : `bool`, optional, default=False
        Choose to check the lightcurve for jumpy data, using the "check_for_jumps"
        function.

    returns
    -------
    LS_dict : `dict`
        A dictionary of parameters calculated from the periodogram analysis. These are:
        | "median_MAD_nLC" : The median and median absolute deviation of the normalised lightcurve.
        | "jump_flag" : A flag determining if the lightcurve has sharp jumps in flux.
        | "period" : A list of period values from the periodogram analysis.
        | "power" :  A list of power values from the periodogram analysis.
        | "period_best" : The period corrseponding to the highest power output.
        | "power_best" : The highest power output.
        | "time" : The time coordinate corresponding to the normalised lightcurve.
        | "y_fit_LS" : The best fit sinusoidal function.
        | "AIC_sine" : The Aikaike Information Criterion value of the best-fit sinusoid
        | "AIC_line" : The Aikaike Information Criterion value of the best-fit linear function.
        | "FAPs" : The power output for the false alarm probability values of 0.1, 1 and 10%
        | "Gauss_fit_peak_parameters" : Parameters for the Gaussian fit to the highest power peak
        | "Gauss_fit_peak_y_values" : The corresponding y-values for the Gaussian fit
        | "period_around_peak" : The period values covered by the Gaussian fit
        | "power_around_peak" : The power values across the period range covered by the Gaussian fit
        | "period_not_peak" : The period values not covered by the Gaussian fit
        | "power_not_peak" : The power values across the period range not covered by the Gaussian fit
        | "period_second" : The period of the second highest peak.
        | "power_second" : The power of the second highest peak.
        | "phase_fit_x" : The time co-ordinates from the best-fit sinusoid to the phase-folded lightcurve.
        | "phase_fit_y" : The normalised flux co-ordinates from the best-fit sinusoid to the phase-folded lightcurve.
        | "phase_x" : The time co-ordinates from the phase-folded lightcurve.
        | "phase_y" : The normalised flux co-ordinates from the phase-folded lightcurve.
        | "phase_chisq" : The chi-square fit between the phase-folded lightcurve and the sinusoidal fit.
        | "phase_col" : The cycle number for each data point.
        | "pops_vals" : The best-fit parameters from the sinusoidal fit to the phase-folded lightcurve.
        | "pops_cov" : The corresponding co-variance matrix from the "pops_val" parameters.
        | "phase_scatter" : The typical scatter in flux around the best-fit.
        | "frac_phase_outliers" : The fraction of data points that are more than 3 median absolute deviation values from the best-fit.
        | "Ndata" : The number of data points used in the periodogram analysis.
    '''
    LS_dict = dict()
    cln = cln[cln["pass_clean"]]
    time = np.array(cln["time"])
    nflux = np.array(cln["nflux_dt2"])
    enflux = np.array(cln["nflux_err"])
    
    if check_jump:
        lc_part = np.array(cln["lc_part"])
        jump_flag = check_for_jumps(time, nflux, enflux, lc_part)
    med_f, MAD_f = np.median(nflux), MAD(nflux, scale='normal')
    ls = LombScargle(time, nflux, dy=enflux)
    frequency, power = ls.autopower(minimum_frequency=1./p_max_thresh,
                                    maximum_frequency=1./p_min_thresh,
                                    samples_per_peak=samples_per_peak)
    FAP = ls.false_alarm_probability(power.max())
    probabilities = [0.1, 0.05, 0.01]
    FAP_test = ls.false_alarm_level(probabilities)
    p_m = np.argmax(power)

    y_fit_sine = ls.model(time, frequency[p_m])
    y_fit_sine_param = ls.model_parameters(frequency[p_m])
    chisq_model_sine = np.sum((y_fit_sine-nflux)**2/enflux**2)/(len(nflux)-3-1)
    line_fit, _,_,_,_ = np.polyfit(time, nflux, 1, full=True)
    y_fit_line = np.polyval(line_fit, time)
    chisq_model_line = np.sum((y_fit_line-nflux)**2/enflux**2)/(len(nflux)-len(line_fit)-1)

    AIC_sine, AIC_line = 2.*3. + chisq_model_sine, 2.*2. + chisq_model_line

    period_best = 1.0/frequency[p_m]
    power_best = power[p_m]
    period = 1./frequency[::-1]
    power = power[::-1]
    # a_g: array of datapoints that form the Gaussian around the highest power
    # a_o: the array for all other datapoints
    if len(power) == 0:
        LS_dict['median_MAD_nLC'] = -999
        LS_dict['jump_flag'] = -999
        LS_dict['period'] = -999
        LS_dict['power'] = -999
        LS_dict['period_best'] = -999
        LS_dict['power_best'] = -999
        LS_dict['time'] = -999
        LS_dict['y_fit_LS'] = -999
        LS_dict['AIC_sine'] = -999
        LS_dict['AIC_line'] = -999
        LS_dict['FAPs'] = -999
        LS_dict['Gauss_fit_peak_parameters'] = -999
        LS_dict['Gauss_fit_peak_y_values'] = -999
        LS_dict['period_around_peak'] = -999
        LS_dict['power_around_peak'] = -999
        LS_dict['period_not_peak'] = -999 
        LS_dict['power_not_peak'] = -999 
        LS_dict['period_second'] = -999
        LS_dict['power_second'] = -999
        LS_dict['phase_fit_x'] = -999
        LS_dict['phase_fit_y'] = -999
        LS_dict['phase_x'] = -999
        LS_dict['phase_y'] = -999
        LS_dict['phase_chisq'] = -999
        LS_dict['phase_col'] = -999
        LS_dict['pops_vals'] = -999    
        LS_dict['pops_cov'] = -999
        LS_dict['phase_scatter'] = -999
        LS_dict['frac_phase_outliers'] = -999
        LS_dict['Ndata'] = -999
        return LS_dict

    a_g, a_o = get_second_peak(power)
    if isinstance(a_g, Iterable):
        pow_r = max(power[a_g])-min(power[a_g])
        a_g_fit = a_g[power[a_g] > min(power[a_g]) + .05*pow_r]
        popt, ym = gauss_fit_peak(period[a_g_fit], power[a_g_fit])
    else:
        if period[a_g] == p_max_thresh:
            popt = [1.0, p_max_thresh, 50.]
            a_g_fit = np.arange(a_g-10, a_g)
            ym = power[a_g_fit]
        elif period[a_g] == p_min_thresh:
            popt = [1.0, p_min_thresh, 50.]
            a_g_fit = np.arange(a_g, a_g+10)
            ym = power[a_g_fit]
        else:
            popt = [-999, -999, -999]
            a_g_fit = np.arange(a_g-2, a_g+3)
            ym = power[a_g_fit]
    
    per_a_o, power_a_o = period[a_o], power[a_o]
    per_2 = per_a_o[np.argmax(power[a_o])]
    pow_2 = power_a_o[np.argmax(power[a_o])]
    pow_pow2 = 1.0*power_best/pow_2
    tdiff = np.array(time-min(time))
    nflux = np.array(nflux)
    pha, cyc = np.modf(tdiff/period_best)
    pha, cyc = np.array(pha), np.array(cyc)
    f = np.argsort(pha)
    p = np.argsort(tdiff/period_best)
    pha_fit, nf_fit, ef_fit, cyc_fit = pha[f], nflux[f], enflux[f], cyc[f].astype(int)
    pha_plt, nf_plt, ef_plt, cyc_plt = pha[p], nflux[p], enflux[p], cyc[p].astype(int)
    try:
        pops, popsc = curve_fit(sin_fit, pha_fit, nf_fit,
                                bounds=(0, [2., 2., 1000.]))
    except Exception:
        logger.warning(Exception)
        pops, popsc = np.array([1., 0.001, 0.5]), 0
        pass

    # order the phase folded lightcurve by phase and split into N even parts.
    # find the standard deviation in the measurements for each bin and use
    # the median of the standard deviation values to represent the final scatter
    # in the phase curve.
     
    sca_mean, sca_stdev = mean_of_arrays(nf_fit, n_sca)
    sca_median = np.median(sca_stdev)

    Ndata = len(nflux)
    yp = sin_fit(pha_fit, *pops)
    chi_sq = np.sum(((yp-pha_fit)/ef_fit)**2)/(len(pha_fit)-len(pops)-1)
    chi_sq = np.sum((yp-pha_fit)**2)/(len(pha_fit)-len(pops)-1)
    
    pha_sct = MAD(yp - nflux, scale='normal')
    fdev = 1.*np.sum(np.abs(nflux - yp) > 3.0*pha_sct)/Ndata
    LS_dict['median_MAD_nLC'] = [med_f, MAD_f]
    if check_jump:
        LS_dict['jump_flag'] = jump_flag
    else:
        LS_dict['jump_flag'] = -999
    LS_dict['period'] = period
    LS_dict['power'] = power
    LS_dict['period_best'] = period_best
    LS_dict['power_best'] = power_best
    LS_dict['time'] = time 
    LS_dict['y_fit_LS'] = y_fit_sine
    LS_dict['AIC_sine'] = AIC_sine
    LS_dict['AIC_line'] = AIC_line
    LS_dict['FAPs'] = FAP_test
    LS_dict['Gauss_fit_peak_parameters'] = popt
    LS_dict['Gauss_fit_peak_y_values'] = ym
    LS_dict['period_around_peak'] = period[a_g_fit]
    LS_dict['power_around_peak'] = power[a_g_fit]
    LS_dict['period_not_peak'] = period[a_o] 
    LS_dict['power_not_peak'] = power[a_o] 
    LS_dict['period_second'] = per_2
    LS_dict['power_second'] = pow_2
    LS_dict['phase_fit_x'] = pha_fit
    LS_dict['phase_fit_y'] = yp
    LS_dict['phase_x'] = pha_plt
    LS_dict['phase_y'] = nf_plt
    LS_dict['phase_chisq'] = chi_sq
    LS_dict['phase_col'] = cyc_plt
    LS_dict['pops_vals'] = pops    
    LS_dict['pops_cov'] = popsc
    LS_dict['phase_scatter'] = sca_median
    LS_dict['frac_phase_outliers'] = fdev
    LS_dict['Ndata'] = Ndata
    return LS_dict


def sin_fit(x, y0, A, phi):
    '''
    Returns the best parameters (y_offset, amplitude, and phase) to a regular
    sinusoidal function.

    parameters
    ----------
    x : `Iterable`
        list of input values
    y0 : `float`
        The midpoint of the sine curve
    A : `float`
        The amplitude of the sine curve
    phi : `float`
        The phase angle of the sine curve

    returns
    -------
    sin_fit : `list`
        A list of sin curve values.
    '''
    sin_fit = y0 + A*np.sin(2.*np.pi*x + phi)
    return sin_fit 
