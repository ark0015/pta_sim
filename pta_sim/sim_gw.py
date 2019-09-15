#!/usr/bin/env python
# coding: utf-8

# Code for running NANOGrav DSA2000 Simulation using the Optimal Statistic
# Libstempo is used to do the simulating

from __future__ import division, print_function

import numpy as np
import sys, os, glob, json, pickle, copy
from collections import OrderedDict

import libstempo as T2
import libstempo.toasim as LT
import libstempo.plot as LP
from shutil import copyfile, copy2
import scipy.stats as sci_stats

# sys.path.insert(0,'/home/jeffrey.hazboun/nanograv/dsa2000_simulations/enterprise/')
import enterprise
from enterprise.pulsar import Pulsar
import enterprise.signals.parameter as parameter
from enterprise.signals import utils
from enterprise.signals import signal_base
from enterprise.signals import selections
from enterprise.signals.selections import Selection
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals
import enterprise.constants as const

from enterprise_extensions import models, model_utils
from enterprise_extensions.frequentist import optimal_statistic as OS

import argparse

######## Convenience Functions ##########
day_in_sec = 24.*3600
year_in_sec = 365.25*day_in_sec

def psr_name(x, rhs='.'):
    return x.split('/')[-1].split(rhs)[0]


######## Simulations ############

class Simulation(object):

    def __init__(self, parfiles, timfiles, ephem='DE436',verbose=True):

        #### Make a list of libstempo.tempopulsar objects.
        libs_psrs = []
        for p, t in zip(parfiles, timfiles):

            psr = T2.tempopulsar(p, t, ephem=ephem, maxobs=30000)
            libs_psrs.append(psr)
            if verbose:
                print('\rPSR {0} loaded.'.format(psr_name(p)),
                      flush=True,end='')

        print('')
        self.libs_psrs = libs_psrs
        self.psrs = None
        self.ephem = ephem
        self.first_toa = np.amin([p.toas().min() for p in libs_psrs])
        self.last_toa = np.amax([p.toas().max() for p in libs_psrs])
        self.toa_cuts = [self.last_toa]
        self.gwb_added = False

    def add_white_noise(self,noisedict,efac=True,seed_efac=None,equad=True,seed_equad=None,
                        ecorr=False,seed_ecorr=None):
        """add white noise using libstempo
    
        Parameters
        ----------
        noisedict : dict
            the noise dictionary containing the white noise values of the pulsars
        efac : bool, optional
            Turns on efac loading from noisedict
        equad : bool, optional
            Turns on equad loading from noisedict
        ecorr : bool, optional
            Turns on ecorr loading from noisedict
        seed_efac : int, optional
            seed for efac pseudorandom-number-generator, if assigned, we can replicate the injection
        seed_equad : int, optional
            seed for equad pseudorandom-number-generator, if assigned, we can replicate the injection
        seed_ecorr : int, optional
            seed for ecorr pseudorandom-number-generator, if assigned, we can replicate the injection

        """
        for ii,p in enumerate(self.libs_psrs):
            if p.name in noisedict:
                ## make ideal
                LT.make_ideal(p)

                if efac:
                    if seed_efac != None:
                        seed_efac += ii
                    ## add efacs
                    LT.add_efac(p, efac=noisedict[p.name]['efac'],seed=seed_efac)

                if equad:
                    if seed_equad != None:
                        seed_equad += ii
                    ## add equads
                    LT.add_equad(p, equad=noisedict[p.name]['equad'],seed=seed_equad)

                if ecorr:
                    if seed_ecorr != None:
                        seed_ecorr += ii
                    ## add jitter
                    try: #Only NANOGrav Pulsars have ECORR
                        LT.add_jitter(p, ecorr = noisedict[p.name]['ecorr'],  
                                      coarsegrain = 1.0/86400.0, seed=seed_jitter)
                    except KeyError:
                        pass
            else:
                print(p.name,' is not in noisedict')

    def add_red_noise(self,noisedict,seed=None):
        """add red noise using libstempo
    
        Parameters
        ----------
        noisedict : dict
            the noise dictionary containing the white noise values of the pulsars
        seed : int, optional
            seed for pseudorandom-number-generator, if assigned, we can replicate the injection
        
        """
        for p in self.libs_psrs:
            LT.add_rednoise(p, noisedict[p.name]['RN_Amp'], noisedict[p.name]['RN_gamma'], 
                            components=30, seed=seed)


    def createGWB(self, A_gwb, gamma_gw= 13./3, seed=None, fit=None, noCorr=False):
        """create GWB using libstempo.toasim
        
        Parameters
        ----------
        A_gwb : float
            Amplitude of the gravitational wave background
        gamma_gw : float
            Power law spectrum of the GWB
        seed : int, optional
            seed for pseudorandom-number-generator, if assigned, we can replicate the injection
        fit : int, optional
            Number of refits with libstempo
        noCorr : bool, optional
            Inject correlated signal, default is no

        """
        if A_gwb!=0:
            LT.createGWB(self.libs_psrs, A_gwb, gamma_gw, seed=seed, noCorr=noCorr)
            # Fit libstempo psr
            if fit is not None:
                for psr in self.libs_psrs:
                    psr.fit(iters=fit)
        else:
            pass
        self.gwb_added = True
        self.seed = seed

    def init_ePulsars(self, **kwarg):
        if not self.gwb_added:
            raise ValueError('Must add GWB signal (createGWB) before '
                             'initializing enterprise.Pulsar objects. If none '
                             'desired set A_gwb=0.')

        psrs = []
        for p in self.libs_psrs:
            psr = Pulsar(p, ephem=self.ephem, **kwarg)

            ### Check first toa ####
            #mn = psr.residuals[:100].mean()
            #std = psr.residuals[:100].std()
            #check_val = mn + 4*std
            #if any([abs(res)>abs(check_val) for res in psr.residuals]):
            #    mask = np.where(abs(psr.residuals)>abs(check_val),False,True)
            #    model_utils.mask_filter(psr,mask)
            # mask = np.ones_like(psr.toas,dtype=bool)
            # mask[0] = False
            # model_utils.mask_filter(psr,mask)

            psrs.append(psr)

        self.psrs=psrs

    def filter_by_mjd(self,end_time):
        """Filter the TOAs by MJD"""
        #### Filter TOAs by time ##########
        if np.amin(self.toa_cuts)<end_time:
            raise ValueError('Can only filter the TOAs to earlier times. Must'
                             ' start again and start with latest cuts first.')
        if self.psrs is None:
            raise ValueError('Enterprise pulsars have not be initialized yet!'
                             ' Use init_ePulsars().')
        idxs = []
        first_toa = np.amin([p.toas.min() for p in self.psrs])/day_in_sec

        for psr in self.psrs:
            psr.filter_data(start_time=self.first_toa, end_time=end_time)
            if psr.toas.size==0:
                idxs.append(self.psrs.index(psr))
            else:
                timespan = (psr.toas[-1]-psr.toas[0])/year_in_sec
                if timespan<3.0:
                    idxs.append(self.psrs.index(psr))

        #### Remove empty pulsars, Reverse to keep correct idx order.

        for ii in reversed(idxs):
            del self.psrs[ii]

        if np.amin(self.toa_cuts)==end_time:
            pass
        else:
            self.toa_cuts.append(end_time)

def model_simple(psrs, psd='powerlaw', components=30, freqs=None,
                 gamma_common=None, upper_limit=False, bayesephem=False,
                 select='backend', red_noise=False, Tspan=None):
    """
    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with the most simple model allowable for enterprise:
    per pulsar:
        1. fixed EFAC per backend/receiver system at 1.0
        2. Linear timing model.
        3. Red noise modeled as a power-law with
            30 sampling frequencies. Default=False
    global:
        1.Common red noise modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']
        2. Optional physical ephemeris modeling.
    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover' 'spectrum']. 'powerlaw' is default
        value.
    :param gamma_common:
        Fixed common red process spectral index value. By default we
        vary the spectral index over the range [0, 7].
    :param upper_limit:
        Perform upper limit on common red noise amplitude. By default
        this is set to False. Note that when performing upper limits it
        is recommended that the spectral index also be fixed to a specific
        value.
    :param bayesephem:
        Include BayesEphem model. Set to False by default
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    if Tspan is None:
        Tspan = model_utils.get_tspan(psrs)

    # timing model
    model = gp_signals.TimingModel()

    #Only White Noise is EFAC set to 1.0
    selection = selections.Selection(selections.by_backend)
    efac = parameter.Constant(1.00)
    model += white_signals.MeasurementNoise(efac=efac, selection=selection)

    # common red noise block
    if upper_limit:
        log10_A_gw = parameter.LinearExp(-18,-12)('gw_log10_A')
    else:
        log10_A_gw = parameter.Uniform(-18,-12)('gw_log10_A')

    gamma_gw = parameter.Constant(4.33)('gw_gamma')
    pl = signal_base.Function(utils.powerlaw, log10_A=log10_A_gw,
                              gamma=gamma_gw)
    if freqs is None:
        gw = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan)
    else:
        gw = gp_signals.FourierBasisGP(spectrum=pl, modes=freqs)

    model += gw

    if red_noise:
        # red noise
        model += models.red_noise_block(prior=amp_prior, Tspan=Tspan,
                                        components=components)

    # ephemeris model
    if bayesephem:
        model += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

    # set up PTA
    pta = signal_base.PTA([model(p) for p in psrs])

    return pta

def create_pta_analysis(psrs, gammas = None, psd='powerlaw', components=30, freqs=None,
                 upper_limit=False, bayesephem=False, select='backend',
                 white_noise=None, red_noise=False, Tspan=None,orf=None):
    """Generates PTA for various analysis models in enterprise

    Parameters
    ----------
    psrs : list
        A list of enterprise Pulsar instance

    gammas : array-like, optional
        An array of background powerlaw index, dictates how many backgrounds are searched over
        Default is None, If None, then uses a varied gamma search
    psd : str, {'powerlaw', 'turnover' 'spectrum'}
        the power spectral density to use for common red noise signal.
    components : int, optional
        Number of sample frequencies
    freqs : array-like, optional
        frequencies at which to inject the background
    upper_limit : bool, optional
        Perform upper limit on common red noise amplitude. By default
        this is set to False. Note that when performing upper limits it
        is recommended that the spectral index also be fixed to a specific
        value.
    bayesephem : bool, optional
        Include BayesEphem model. Set to False by default
    select : str, {'backend','other'}
        'backend' separates noise per backend/receiver system
        'other' sets no selections
    white_noise : str, optional
        Includes fixed white noise parameters if set to filepath (ie. '/path/to/file/filename.json')
        If None, sets a fixed EFAC per backend/receiver system at 1.0
    red_noise : bool, optional
        Includes search over pulsar red noise parameters
    Tspan : float, optional
        The maximum observation time span to set GW frequency sampling
    orf : str, {None,'hd','dipole','monopole'}
        Sets which overlap reduction function to use. By default we do not use any spatial correlations.

    Returns
    -------
    pta : obj
        An instantiated enterprise pta

    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    if Tspan is None:
        Tspan = model_utils.get_tspan(psrs)

    # timing model
    model = gp_signals.TimingModel()

    if select is 'backend':
        selection = selections.Selection(selections.by_backend)
    else:
        selection = selections.Selection(selections.no_selection)

    if white_noise == None:
        #Only White Noise is EFAC set to 1.0
        efac = parameter.Constant(1.00)
        model += white_signals.MeasurementNoise(efac=efac, selection=selection)
    else:
        #Set White Noise to file values
        model += models.white_noise_block(select=select)

    # common red noise block
    if gammas != None:
        #Fixed gamma
        for idx,gamma in enumerate(gammas):
            label = 'gw_{}'.format(idx+1)

            if freqs is None:
                gw = models.common_red_noise_block(psd=psd, prior=amp_prior,
                                   Tspan=Tspan, gamma_val=gamma, name=label,orf=orf)
            else:
                gw = models.common_red_noise_block(psd=psd, prior=amp_prior,
                                   Tspan=Tspan, gamma_val=gamma, name=label,orf=orf,modes=freqs)
    else:
        #Varied gamma and amplitude
        gw = models.common_red_noise_block(psd=psd, prior=amp_prior,
                                   Tspan=Tspan, gamma_val=None, name='gw',orf=orf)
        
    model += gw

    if red_noise:
        # red noise
        model += models.red_noise_block(psd=psd,prior=amp_prior, Tspan=Tspan,
                                        components=components)

    # ephemeris model
    if bayesephem:
        model += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

    # set up PTA
    pta = signal_base.PTA([model(p) for p in psrs])

    return pta

######## Red Noise Parameters ############
#Since there is no red noise we make a dictionary of low RN Values

def get_rn_dict(pta):
    #psr_list = pta
    pass


######## File I/O ###############
def save(outpath):
    pass

# if __name__=='__main__':
