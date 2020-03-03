# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 11:37:06 2018

@author: dkorff
"""

"""====================================================================="""
"""===========================Model Functions==========================="""
"""====================================================================="""
import cantera as ct
import numpy as np
from li_ion_battery_p2d_inputs import Inputs

def set_state(offset, SV, ed, surf, el, conductor, ptr, z_k_rxn):
                    
    ed.X = [SV[offset + ptr['X_ed'][-1]], 1 - SV[offset + ptr['X_ed'][-1]]]
    ed.electric_potential = SV[offset + ptr['Phi_ed']]
    
    conductor.electric_potential = SV[offset + ptr['Phi_ed']]
    
    el.X = SV[offset + ptr['X_k_elyte']]
    el.electric_potential = SV[offset + ptr['Phi_dl']] + SV[offset + ptr['Phi_ed']]
    
    state = {}
    state['sdot'] = surf.get_net_production_rates(el)
    state['sdot_full'] = surf.net_production_rates
    state['phi_ed'] = SV[offset + ptr['Phi_ed']]
    state['phi_el'] = SV[offset + ptr['Phi_dl']] + SV[offset + ptr['Phi_ed']]
    state['rho_ed'] = ed.density_mole
    state['rho_el'] = el.density_mole
    state['X_k_el'] = SV[offset + ptr['X_k_elyte']]
    state['T'] = SV[offset + ptr['T']]
    
    state['h_ed'] = ed.partial_molar_enthalpies
    state['h_el'] = el.partial_molar_enthalpies
    state['h_cond'] = np.zeros([1])
    state['h_tot'] = np.concatenate((state['h_ed'], state['h_cond'], state['h_el']))
    state['phi_k'] = np.concatenate((np.ones([3])*state['phi_ed'], 
                                     np.ones([el.n_species])*state['phi_el']))
    state['e_k'] = state['h_tot'] + z_k_rxn*ct.faraday*state['phi_k']
    
    return state

"""====================================================================="""

def set_state_sep(offset, SV, el, ptr):
    
    el.X = SV[offset + ptr['X_k_elyte']]
    el.electric_potential = SV[offset + ptr['Phi']]
    
    state = {}
    state['phi_el'] = SV[offset + ptr['Phi']]
    state['rho_el'] = el.density_mole
    state['X_k_el'] = SV[offset + ptr['X_k_elyte']]
    state['T'] = SV[offset + ptr['T']]
    
    return state

"""====================================================================="""

def dilute_flux(s1, s2, dyInv, ed, D_k, D_migr_k):
    F = ct.faraday; R = ct.gas_constant; T = Inputs.T
    z_k = Inputs.z_k_elyte
    # Calculate ionic flux in electrolyte
    
    # Total molar concentration in electrolyte
    C_0 = (s2['rho_el'] + s1['rho_el'])/2.
    
    N_io_p = (- D_k*C_0*(s2['X_k_el'] - s1['X_k_el'])*dyInv 
              - D_migr_k*(z_k*F/R/T)*(s2['phi_el'] - s1['phi_el'])*dyInv)
    
#    N_io_p = (-ed.u_Li_elyte*(R*T*C_0*(s2['X_k_el'] - s1['X_k_el'])
#    + Inputs.z_k_elyte*F*C_k*(s2['phi_el'] - s1['phi_el']))*dyInv)
    
    i_io_p = np.dot(N_io_p, Inputs.z_k_elyte)*F
    
    return N_io_p, i_io_p

"""====================================================================="""

def solid_flux(SV, offset, ptr, s1, ed):
    
    X_Li = SV[offset + ptr['X_ed']]
    DiffFlux = np.zeros([ed.nshells+1])
    DiffFlux[1:-1] = ed.D_Li_ed*(X_Li[1:] - X_Li[0:-1])/ed.dr
    DiffFlux[-1] = -s1['sdot'][ed.ptr['iFar']]/s1['rho_ed']
    
    return DiffFlux

def setup_plots(plt, rate_tag):
    if Inputs.plot_potential_profiles == 1:
        fig1, axes1 = plt.subplots(sharey="row", figsize=(14,6), nrows=1, 
                                   ncols = 2+(Inputs.flag_re_equil*Inputs.phi_time))
        plt.subplots_adjust(wspace = 0.15, hspace = 0.4)
        fig1.text(0.15, 0.8, rate_tag, fontsize=20, 
                  bbox=dict(facecolor='white', alpha=0.5))
        
    if Inputs.plot_electrode_profiles == 1:
        nrows = Inputs.flag_anode + Inputs.flag_cathode
        ncols = 2 + Inputs.flag_re_equil
        fig2, axes2 = plt.subplots(sharey="row", figsize=(18,9), nrows=nrows,
                                   ncols=ncols)
        plt.subplots_adjust(wspace=0.15, hspace=0.4)
        fig2.text(0.15, 0.8, rate_tag, fontsize=20, 
                  bbox=dict(facecolor='white', alpha=0.5))
        
    if Inputs.plot_elyte_profiles == 1:
        nrows = Inputs.flag_anode + Inputs.flag_cathode + Inputs.flag_sep
        ncols = 2 + Inputs.flag_re_equil
        fig3, axes3 = plt.subplots(sharey="row", figsize=(18,9), nrows=nrows,
                                    ncols=ncols)
        plt.subplots_adjust(wspace=0.15, hspace=0.4)
        fig3.text(0.15, 0.8, rate_tag, fontsize=20, 
                  bbox=dict(facecolor='white', alpha=0.5))
        
    if Inputs.plot_temp_flag == 1:
        nrows = 1 #Inputs.flag_anode + Inputs.flag_cathode + Inputs.flag_sep
        ncols = 2 + Inputs.flag_re_equil
        fig4, axes4 = plt.subplots(sharey="row", figsize=(18,9), nrows=nrows,
                                   ncols=ncols)
        plt.subplots_adjust(wspace=0.15, hspace=0.4)
        fig4.text(0.15, 0.8, rate_tag, fontsize=20,
                  bbox=dict(facecolor='white', alpha=0.5))
        
    return fig1, axes1, fig2, axes2, fig3, axes3, fig4, axes4

"""====================================================================="""

def thermal_terms(state, q_m, q_p, dyInv, A_s, h_CC, T_inf):
    
    qdot = {}
    qdot['cond'] = Inputs.flag_conduction*(q_m['cond'] - q_p['cond'])*dyInv
    qdot['conv'] = Inputs.flag_convection*(q_m['conv'] - q_p['conv'])*dyInv
    qdot['ohm'] = Inputs.flag_ohmic*(q_m['ohm'] + q_p['ohm'])*0.5
    qdot['chem'] = Inputs.flag_chemical*(-A_s)*np.dot(state['sdot_full'][0:-1], state['e_k'])
    qdot['loss'] = 0 #h_CC*(state['T'] - T_inf)
    
    return qdot