"""
    Cables Parametric Model
"""

#  This file is part of FAST : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2019  ONERA/ISAE
#  FAST is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import math
import yaml
import sympy
import pdb
import matplotlib.pyplot as plt
import sympy as sp
from pathlib import Path
from scipy.optimize import curve_fit
import statistics
import matplotlib.cm as cm
import matplotlib.lines as mlines
import math as m
import statistics as stats

def func_poly_m(cur, a, b):
    """
    Polynomial used for data interpolation in matrix form
    """
    return a * np.power(cur, b)
# end

def func_poly_n(cur, a, b):
    """
    Polynomial used for data interpolation in scalar form
    """
    return a * cur ** b
# end

def func_lin(volt, a, b):
    """
    Linear function used for data interpolation in matrix form
    """
    return a * volt + b
# end

def sw_func_Cu(A_cable, U_cbl):
    """
    Computes length specific weight given cross sectional area and transmission voltage for copper cables

    From :cite:'stuckl:2018`, 3.1.2. Conventional Materials for Power Transmission
    """
    lsw = 0.00908 * A_cable + 0.00256 *U_cbl+ 0.00145 * A_cable ** (1 / 2) *U_cbl+ 0.0132 * A_cable ** (
            1 / 2) + 3.82e-5 *U_cbl** 2 + 0.0138
    return lsw


def sw_func_Al(A_cable, U_cbl):
    """
    Computes length specific weight given cross sectional area and transmission voltage for copper cables

    From :cite:'stuckl:2018`, 3.1.2. Conventional Materials for Power Transmission
    """
    lsw = 0.00288 * A_cable + 0.00256 *U_cbl+ 0.00145 * A_cable ** (1 / 2) *U_cbl+ 0.0132 * A_cable ** (
            1 / 2) + 3.82e-5 **U_cbl** 2 + 0.0138
    return lsw


class ComputeCables:
    """
    Cable Parametric model

    Computes mass and losses of HV cables at transmission current and voltage level

    :param U_cbl: (unit=V) Cable transmission votlage
    :param I_cbl: (unit=A) Cable transmission current
    :param conductor: Conductor type (Aluminum or Copper)
    :param length: (unit=m) Cable length

    Based on
    :cite:'stuckl:2018`, 3.1.2. Conventional Materials for Power Transmission
    """

    def __init__(self, cables_param):

        self.I_cbl = cables_param['current']
        self.U_cbl = cables_param['voltage']
        self.conductor = cables_param['conductor']
        self.length = cables_param['length']
        self.plot_graphs = False

        # Load and interpolate across cable data

        # Copper Cables
        if self.conductor == "Cu":

            # Load Cable Data

            # Area vs current
            data_str_Cu = Path(__file__).parent / "Data/Copper_area_current.yaml"
            # Area vs specific weight
            data_str_Cu_ref = Path(__file__).parent / "Data/N2XY.yaml"

            stream_Cu = open(data_str_Cu, 'r')
            stream_Cu_ref = open(data_str_Cu_ref, 'r')

            Cu_current_v_area = yaml.safe_load(stream_Cu)
            N2XY_ref = yaml.safe_load(stream_Cu_ref)

            sz = len(Cu_current_v_area)
            sz_ref = len(N2XY_ref)

            area = np.empty((sz, 1)).flatten()
            current = np.empty((sz, 1)).flatten()

            area_ref = np.empty((sz_ref, 1)).flatten()
            sw_ref = np.empty((sz_ref, 1)).flatten()

            # Populate arrays with data

            for i in range(0, sz):
                area[i] = Cu_current_v_area[i]['Conductor_Area']
                current[i] = Cu_current_v_area[i]['Current']
            # end

            for i in range(0, sz_ref):
                area_ref[i] = N2XY_ref[i]['cross_section']
                sw_ref[i] = N2XY_ref[i]['specific_weight']
            # end

            self.rho = 8.9 / 1e6  # Copper density [kg/mm^3]
            self.alpha = 0.00386  # Temperature coefficient per degree of resistance for Copper [1/K]
            self.area_ref = area_ref
            self.sw_ref = sw_ref
        # end

        # Load aluminum cable data
        elif self.conductor == "Al":

            # Load Cable Data
            data_str_Al = Path(__file__).parent / "Data/Aluminum_area_current.yaml"
            data_str_Al_ref = Path(__file__).parent / "Data/NA2XY.yaml"

            stream_Al = open(data_str_Al, 'r')
            stream_Al_ref = open(data_str_Al_ref, 'r')

            Al_current_v_area = yaml.safe_load(stream_Al)
            NA2XY_ref = yaml.safe_load(stream_Al_ref)

            sz = len(Al_current_v_area)
            sz_ref = len(NA2XY_ref)

            area = np.empty((sz, 1)).flatten()
            current = np.empty((sz, 1)).flatten()

            area_ref = np.empty((sz_ref, 1)).flatten()
            sw_ref = np.empty((sz_ref, 1)).flatten()

            for i in range(0, sz):
                area[i] = Al_current_v_area[i]['Conductor_Area']
                current[i] = Al_current_v_area[i]['Current']
            # end

            for i in range(0, sz_ref):
                area_ref[i] = NA2XY_ref[i]['cross_section']
                sw_ref[i] = NA2XY_ref[i]['specific_weight']
            # end

            self.rho = 2.7 / 1e6  # Aluminum density <kg/mm^3>
            self.alpha = 0.00429  # Aluminum temperature coefficient per degree C
            self.area_ref = area_ref
            self.sw_ref = sw_ref
        # end
        else:
            raise Exception("Invalid conductor type entered. Available types are 'Cu' (Copper), or 'Al' (Aluminum).")
        # end

        # Coefficients for interpolation between current and cable cross-sectional area
        self.popt_CA, pcov1 = curve_fit(func_poly_m, current, area)
        popt_CA = self.popt_CA

        # Plot cable area interpolation results
        if self.plot_graphs:

            conductor = 'Cu'

            # Calculate Error
            err = np.zeros((len(current), 1)).flatten()
            for i in range(0, len(current)):
                err[i] = abs(area[i] - func_poly_n(current[i], *popt_CA)) / area[i] * 100
            # end

            fig, ax = plt.subplots()
            plt.scatter(current, area, c='C3')
            plt.plot(current, func_poly_m(current, *popt_CA))
            plt.xlabel("Current [A]")
            plt.ylabel("Conductor Cross-sectional Area [$mm^2$]")
            plt.title(conductor + " Cable Regression")
            plt.ylim(bottom=0)  # adjust the bottom leaving top unchanged
            plt.legend(['Fitted', 'Experimental'])

            textstre = '\n'.join((
                r"Efficiency Error" + '\n',
                r"$\bar{\mu} = {%.2f}$ %%" % stats.mean(err),
                r"$\mu_{max} = {%.2f}$ %%" % max(err)))

            props = dict(boxstyle='round', facecolor='beige', alpha=0.5, pad=1)

            plt.text(0.03, 0.65, textstre, transform=ax.transAxes, fontsize=10,
                     verticalalignment='top', bbox=props, horizontalalignment='left')

            props = dict(boxstyle='round', facecolor='beige', alpha=0.5)

            textstr = r'$A_{req}[mm^2] = {%.4f}~I[A]^{~%.4f}}$' % (popt_CA[0], popt_CA[1])
            plt.text(0.5, 0.450, textstr, transform=ax.transAxes, fontsize=10,
                     verticalalignment='top', bbox=props)

            plt.show()
        # end


        # Interpolate Isolation Thickness
        # Load Cable Data
        if __name__ == '__main__':
            data_str = 'Data/Volt_Iso.yaml'
        else:
            data_str = 'components/cables/Data/Volt_Iso.yaml'
        # end

        stream_Iso = open(data_str, 'r')
        voltage_v_iso_thick = yaml.safe_load(stream_Iso)
        sz = len(voltage_v_iso_thick)

        voltage = np.empty((sz, 1)).flatten()
        thickness = np.empty((sz, 1)).flatten()

        for i in range(0, sz):
            voltage[i] = voltage_v_iso_thick[i]['voltage']
            thickness[i] = voltage_v_iso_thick[i]['thickness']
        # end

        # Coefficients for interpolation between voltage and isolation thickness
        self.popt_iso, pcov1 = curve_fit(func_lin, voltage, thickness)
        popt_iso = self.popt_iso

        # Plot isolation thickness interpolation results
        if self.plot_graphs:

            # Calculate Error
            err = np.zeros((len(voltage), 1)).flatten()
            for i in range(0, len(voltage)):
                err[i] = abs(thickness[i] - func_poly_n(voltage[i], *popt_iso)) / thickness[i] * 100
            # end

            fig, ax = plt.subplots()
            plt.scatter(voltage, thickness, c='C3')
            plt.plot(voltage, func_lin(voltage, *popt_iso))
            plt.xlabel("Voltage [$kV$]")
            plt.ylabel("Isolation Thickness[$mm$]")
            plt.title("Cable Isolation Regression")
            plt.ylim(bottom=0)  # adjust the bottom leaving top unchanged

            plt.legend(['Fitted', 'Experimental'])

            props = dict(boxstyle='round', facecolor='beige', alpha=0.5)
            textstr = r'$Thickness_{iso}[mm] = {%.4f} * U[kV] + {~%.4f}$' % (popt_iso[0], popt_iso[1])
            plt.text(0.3, 0.450, textstr,transform=ax.transAxes,fontsize=10,
                    verticalalignment='top', bbox=props)

            textstre = '\n'.join((
                r"Efficiency Error" + '\n',
                r"$\bar{\mu} = {%.2f}$ %%" % stats.mean(err),
                r"$\mu_{max} = {%.2f}$ %%" % max(err)))

            props = dict(boxstyle='round', facecolor='beige', alpha=0.5, pad=1)

            plt.text(0.03, 0.25, textstre, transform=ax.transAxes, fontsize=10,
                     verticalalignment='top', bbox=props, horizontalalignment='left')

            plt.show()
        # end

    def compute_cable_mass(self):
        """
        Computes mass of transmission cable. Based on model by Stuckl

        :param popt_CA: Polynomial fit coefficients for cross area as a function of transmission current
        :param popt_iso: Polynomial fit coefficients for isolation thickness as a function of transmission voltage
        :param U_cbl: (unit=V) Cable transmission votlage
        :param I_cbl: (unit=A) Cable transmission current
        :param conductor: Conductor type (Aluminum or Copper)
        :param length: (unit=m) Cable length
        :param a_ref: (unit=m^2) Cross-sectional area data of reference cable
        :param sw_ref: (unit=kg/m) Specific weight data of reference cable
        :param rho: (unit=kg/m^3) Conductor density
        
        :return cbl_mass: (unit=kg)  Cable Mass

        From :cite:'stuckl:2018`, 3.1.2. Conventional Materials for Power Transmission
        """

        popt_CA = self.popt_CA
        popt_iso = self.popt_iso
        rho = self.rho
        I_cbl = self.I_cbl
        U_cbl= self.U_cbl
        area_ref = self.area_ref
        sw_ref = self.sw_ref
        length = self.length
        conductor = self.conductor

        # Calculate area using coefficients and current transmitted
        A_cable = popt_CA[0] * I_cbl ** popt_CA[1]

        # Calculation isolation thickness using coefficients and cable voltage
        t_iso = popt_iso[0] * U_cbl+ popt_iso[1]

        rho_XLPE = 1.4 / 1e6  # EVA isolation density [kg/mm^3]
        rho_PVC = 1.3 / 1e6  # PVC Sheath density [kg/mm^3]

        # Calculate cable weight
        lsw_con = A_cable * rho  # Conductor length specific weight [kg/mm]
        r_con = (A_cable / np.pi) ** 0.5  # Conductor radius [mm]
        lsw_iso = r_con * 2 * np.pi * rho_XLPE * t_iso  # Isolation length specific weight [kg/mm]

        # Total cable radius
        r_iso = t_iso + r_con
        t_shth = 0.035 * r_iso * 2 + 1  # Sheath thickness [mm]
        lsw_shth = rho_PVC * t_shth * r_iso * 2 * np.pi     # Sheath linear specific weight [kg/mm]
        lsw_cbl = 1000 * (lsw_shth + lsw_iso + lsw_con)     # Cable linear specific weight [kg/mm]
        cbl_mass = lsw_cbl * length                         # Total cable mass

        if self.plot_graphs:
            
            U_cbl= np.array([400, 1000, 2000, 3000, 4000]) / 1000

            for i in range(0, 5):

                I_cbl = np.linspace(1, 1600, 100)
                P_in = I_cbl * U_cbl[i]

                A_cable = popt_CA[0] * I_cbl ** popt_CA[1]
                t_iso = popt_iso[0] * U_cbl[i] + popt_iso[1]

                rho_XLPE = 1.4 / 1e6  # EVA isolation density <kg/mm^3>
                rho_PVC = 1.3 / 1e6  # PVC Sheath density <kg/mm^3>

                # Calculate cable weight
                lsw_con = A_cable * rho  # Conductor length specific weight <kg/mm>
                r_con = (A_cable / np.pi) ** 0.5  # Conductor radius <mm>
                lsw_iso = r_con * 2 * np.pi * rho_XLPE * t_iso  # Isolation length specific weight <kg/mm>
                r_iso = t_iso + r_con
                t_shth = 0.035 * r_iso * 2 + 1  # Sheath thickness <mm>
                lsw_shth = rho_PVC * t_shth * r_iso * 2 * np.pi  # Sheath linear specific weight <kg/mm>

                lsw_cbl = 1000 * (lsw_shth + lsw_iso + lsw_con)


                # Calculate Error in specific weight. Compare to NXY and NAXY data
                err = np.zeros((len(sw_ref), 1)).flatten()
                for j in range(0, len(sw_ref)):

                    if conductor == "Cu":
                        err[j] = abs(sw_ref[j] - sw_func_Cu(area_ref[j], 0.4)) / sw_ref[j] * 100
                    elif conductor == "Al":
                        err[j] = abs(sw_ref[j] - sw_func_Al(area_ref[j], 0.4)) / sw_ref[j] * 100
                    else:
                        raise Exception(
                            "Invalid conductor type entered. Available types are 'Cu' (Copper), or 'Al' (Aluminum).")
                    # end
                # end

                if i == 0:
                    fig, ax1 = plt.subplots()
                    ax1.plot(A_cable, lsw_cbl)
                    ax1.scatter(area_ref, sw_ref, c='C3')
                    plt.title(conductor + " Cable Regression | 0.4 kV")
                    plt.xlabel("Conductor Cross-Sectional Area [mm$^2$]")
                    plt.ylabel("Length Specific Weight [kg/m]")

                    textstre = '\n'.join((
                        r"Error" + '\n',
                        r"$\bar{\mu} = {%.2f}$ %%" % stats.mean(err),
                        r"$\mu_{max} = {%.2f}$ %%" % max(err)))

                    props = dict(boxstyle='round', facecolor='beige', alpha=0.5, pad=1)

                    plt.text(0.03, 0.8, textstre, transform=ax1.transAxes, fontsize=10,
                             verticalalignment='top', bbox=props, horizontalalignment='left')

                    if conductor == "Cu":
                        textstrSci = r'$\rho_{cable} = 9.08e-3~A_{req} + 2.56e-3~U_{[kV]} + $ ' \
                                     '\n ' \
                                     r'$ A_{req}^{0.5}~(1.45e-3~U_{[kV]} +$' \
                                     r'$ 0.0132) + $' \
                                     '\n' \
                                     r'$3.82e-5~U_{[kV]}^2 + 0.0138$'
                    elif conductor == "Al":
                        textstrSci = r'$\rho_{cable} = 2.88e-3*A_{req~[mm^2]} + 2.56e-3*U_{[kV]} + 1.45e-3*A_{req~[mm^2]}^{0.5}*U_{[kV]} +' \
                                     r' 0.0132*A_{req~[mm^2]}^{0.5} + 3.82e-5*U_{[kV]}^2 + 0.0138$'
                    else:
                        raise Exception(
                            "Invalid conductor type entered. Available types are 'Cu' (Copper), or 'Al' (Aluminum).")
                    # end
                    props = dict(boxstyle='round', facecolor='beige', alpha=0.5)
                    plt.text(0.35, 0.25, textstrSci, transform=ax1.transAxes, fontsize=10,
                             verticalalignment='top', bbox=props)
                    plt.legend(['Fitted', 'Experimental'])
                    plt.show()

                else:
                    plt.plot(P_in, lsw_cbl)
                # end

            plt.xlabel("Power Transmitted [kW]")
            plt.ylabel("Specific Weight [kg/m]")
            plt.title(conductor + " Cable Regression")

            #textstrSci = r'$\rho_{cable} = 9.08e-3*A_{req~[mm^2]} + 2.56e-3*U_{[kV]} + 1.45e-3*A_{req~[mm^2]}^{0.5}*U_{[kV]} +' \
            #             r' 0.0132*A_{req~[mm^2]}^{0.5} + 3.82e-5*U_{[kV]}^2 + 0.0138$'
            props = dict(boxstyle='round', facecolor='beige', alpha=0.5)
            plt.text(0.4, 0.250, textstrSci, fontsize=10,transform=ax1.transAxes,
                     verticalalignment='top', bbox=props)
            plt.legend(['1000V', '2000V', '3000V', '4000V'])

            plt.show()
        # end

        return cbl_mass

    def compute_cable_loss(self):
        """
        Computes losses of transmission cable at voltage and current level specified

        :param popt_CA: Polynomial fit coefficients for cross area as a function of transmission current
        :param I_cbl: (unit=A) Transmission current
        :param U_cbl: (unit=V) Transmission voltage
        :param conductor: Conductor type (Aluminum or Copper)
        :param alpha: (unit=1/K) Temperature coefficient of resistance
        :return P_L: (unit=W) Cable power losses

        From :cite:'stuckl:2018`, 3.1.2. Conventional Materials for Power Transmission
        """

        popt_CA = self.popt_CA
        alpha = self.alpha
        conductor = self.conductor
        I_cbl = self.I_cbl
        plot = self.plot_graphs

        A_cable = popt_CA[0] * I_cbl ** popt_CA[1]
        R20 = 0.0225  # Specific resistivity. Assume cables at 90 C
        R = R20 * (1 + alpha * abs(90 - 25)) / A_cable  # Cable resistance
        P_L = I_cbl ** 2 * R  # Losses [W]

        if plot:
            # Cable Losses
            U_cbl= [1000, 2000, 3000, 4000]
            fig, ax2 = plt.subplots()
            for i in range(0, 4):

                I_cbl = np.linspace(1, 1600, 100)

                A_cable = popt_CA[0] * I_cbl ** popt_CA[1]
                P_in = U_cbl[i] * I_cbl
                R20 = 0.0225  # Specific resistivity. Assume cables at 90 C
                R = R20 * (1 + alpha * abs(90 - 25)) / A_cable  # Cable resistance
                P_L = I_cbl ** 2 * R  # Losses

                plt.plot(P_in / 1000, P_L)
            # end


            textstrSci = r'$P_L[W] = \left(\frac{P}{U}\right) ^ 2~R_{20}~\frac{(1 + \alpha~\Delta~T)}{ {%.3f}~ \left(\frac{P}{U}\right) ^ {%.3f}}$' % (
                popt_CA[0], popt_CA[1])
            props = dict(boxstyle='round', facecolor='beige', alpha=0.5)
            plt.text(0.4, 0.50, textstrSci, fontsize=10,transform=ax2.transAxes,
                     verticalalignment='top', bbox=props)

            plt.xlabel("Power Transmitted [kW]")
            plt.ylabel("Power Loss [kW]")
            plt.title(conductor + " Cable Regression")
            plt.legend(['1000V', '2000V', '3000V', '4000V'])
            plt.show()
        # end

        return P_L


if __name__ == '__main__':
    cables_param = {
        'current': 350,             # [A] current transmitted by cable
        'voltage': 300,             # [V] voltage carried by cable
        'conductor': 'Cu',
        'length': 10,               # [m] Cable length in powertrain
    }

    myCables = ComputeCables(cables_param)

    cable_mass = myCables.compute_cable_mass()
    cable_losses = myCables.compute_cable_loss()


