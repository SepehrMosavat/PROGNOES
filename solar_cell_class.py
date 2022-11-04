import time
import numpy as np


class solar_cell():
    """
        This class instantiates a Solar Cell object that contains many functions.
        First an empty class must be instantiated,
        then either a file must be loaded for profiling or the file of already profiled solar cell.
    """

    def __init__(self, model="N/A", producer="N/A", width=1e-50, length=1e-50, VOC=1e-50, ISC=1e-50, VMPP=1e-50,
                 IMPP=1e-50, nominal_PMPP=1e-50,
                 nominal_fill_factor=1e-50,
                 efficiency=1e-50, Ns=1, n=1e-50, Il=1e-50, I0=1e-50, rs=1e-50, rsh=1e-50, t=1e-50, nNsVth=1e-50,
                 current_multiplier=1,
                 VOC_temp_coefficient=1e-50,
                 ISC_temp_coefficient=1e-50):
        self.model = model  # str = based on datasheet
        self.producer = producer  # str = base on datasheet
        self.width = width  # float = based on datasheet in meter
        self.length = length  # float = based on datasheet in meter
        self.VOC = VOC  # float = based on datasheet in volt
        self.ISC = ISC  # float = based on datasheet in milli amper
        self.VMPP = VMPP  # float = based on datasheet in volt
        self.IMPP = IMPP  # float = based on datasheet in milli amper
        self.nominal_PMPP = nominal_PMPP  # float = based on datasheet in milli watt
        self.nominal_fill_factor = nominal_fill_factor  # float = based on datasheet in percent
        self.nominal_efficiency = efficiency  # float = based on datasheet in percent
        self.Ns = Ns  # integer = number of Diods / Cells
        self.n = n  # float = ideality Factor
        self.Il = Il  # float = light iluminatuon
        self.I0 = I0  # float = dark Saturation
        self.rs = rs  # float = R Series
        self.rsh = rsh  # float = R Shunt
        self.t = t  # integer = temperature
        self.q = 1.6e-19  # float = elementary electric charge constant
        self.k = 1.38e-23  # float = boltzmann constant
        self.nNsVth = nNsVth  # float = to use in lambertw function
        self.current_multiplier = current_multiplier  # int = to use for exponential function
        self.VOC_temp_coefficient = VOC_temp_coefficient  # float = based on datasheet in volt/kelvin
        self.ISC_temp_coefficient = ISC_temp_coefficient  # float = based on datasheet in milli amper/kelvin per cm²
        self.PMPP = VMPP * IMPP  # float = unit in milli watt
        self.fill_factor = round(((VMPP * IMPP) / (VOC * ISC)), 2)  # float = unit in percent
        self.max_efficiency = round((self.fill_factor * self.ISC * self.VOC), 4)  # float = unit in milli watt

    pass

    def __repr__(self):
        return "Model: {}, Producer: {}, Width: {}, Length: {}, VOC: {}, ISC: {}, VMPP:{}, IMPP: {}, Nominal PMPP: {}," \
               " Nominal Fill Factor: {}, Nominal Efficiency: {}, Number of Cells: {}, Ideality Facor: {}, Iluminatuion Saturation: {}," \
               " Dark Saturation: {}, R Series: {}, R Shunt: {}, STC Temprature: {}, nNsVth: {}, Current multiplier: {}," \
               " Open circuit voltage temp. coefficient: {}, Short circuit current temp. coefficient: {}," \
               " PMPP: {}, Fill Factor: {}, Maximum Effciency: {}".format(
            self.model, self.producer, self.width, self.length, self.VOC, self.ISC, self.VMPP, self.IMPP,
            self.nominal_PMPP,
            self.nominal_fill_factor, self.nominal_efficiency, self.Ns, self.n, self.Il, self.I0, self.rs, self.rsh,
            self.t, self.nNsVth, self.current_multiplier,
            self.VOC_temp_coefficient,
            self.ISC_temp_coefficient, self.PMPP, self.fill_factor, self.max_efficiency)

    def i0_check(self):
        """
        This function calculates the dark saturation based.
        :return:
        Returns the dark saturation based on unit of current. For example, on micro amperes.
        """
        i0 = (self.ISC - (self.VOC / self.rsh)) / np.exp((self.q * self.VOC) / (self.Ns * self.n * self.k * self.t))
        return i0

    def imp_check(self):
        """
        It is a equation to check if all calculated and given parameters are correct.
        :return:
        It returns the IMP, to enable user or other function to check if the IMP is equal to the actual expected IMP.
        """
        imp_new = self.ISC - (self.I0 * np.exp(
            self.q * ((self.VMPP + self.IMPP * self.rs) / (self.Ns * self.n * self.k * self.t)))) - (
                          (self.VMPP + self.IMPP * self.rs) / (self.rsh))
        return imp_new

    def imp_check2(self):
        """
        It is a equation to check if all calculated and given parameters are correct.
        :return:
        It returns the IMP, to enable user or other function to check if the IMP is equal to the actual expected IMP.
        """
        imp_new = self.Il - self.I0 * (np.exp(self.VOC / ((self.Ns * self.n * self.k * self.t) / self.q)) - 1) - (
                self.VOC / self.rsh)
        return imp_new

    def imp_check3(self):
        """
            It is a equation to check if all calculated and given parameters are correct.
            :return:
            It returns the IMP, to enable user or other function to check if the IMP is equal to the actual expected IMP.
            """
        imp_new = self.I0 * np.exp(((self.q * self.VMPP) / (self.k * self.t)) - 1)
        return imp_new

    def imp_check4(self):
        """
        It is a equation to check if all calculated and given parameters are correct.
        :return:
        It returns the IMP, to enable user or other function to check if the IMP is equal to the actual expected IMP.
        """
        imp_new = self.Il - self.I0 * (np.exp((self.q * self.VMPP) / (self.n * self.k * self.t)) - 1)
        return imp_new

    def voc_reverse_saturation(self):
        """
        This function calculates the reverse saturation for voltage.
        :return:
        Returns the VOC of volt.
        """
        voc_0 = (self.k * self.t / self.q) * np.log((self.ISC / self.I0) + 1)
        return voc_0

    def voc_check(self):
        """
        This function enables the user to check if the parameters are correct and correspond the desired VOC.
        :return:
        Returns the VOC of volt.
        """
        voc = ((self.Ns * self.n * self.k * self.t) / self.q) * np.log((self.Il / self.I0) + 1)
        return voc

    def il_check(self, deltaT=298, g=1000, gtsc=1000):
        """
        This function calculates the Light Generated Current known as IL.
        :return:
        Returns the Il in corresponding unit of amperes.
        """
        il = self.ISC + (self.k * deltaT) * (g / gtsc)
        return il

    def SC_obj_to_csv(self, type="Original_"):
        """
        This function saves the calculated parameters of a solar cell in a CSV file.
        :param type:
        The prefix of the the file name. Default is Original_
        :return:
        """
        import pandas as pd
        solar_cell = {}
        solar_cell.update(
            {"model": self.model, "producer": self.producer, "width": self.width, "length": self.length,
             "VOC": self.VOC,
             "ISC": self.ISC, "VMPP": self.VMPP, "IMPP": self.IMPP, "nominal_PMPP": self.nominal_PMPP,
             "nominal_fill_factor": self.nominal_fill_factor, "nominal_efficiency": self.nominal_efficiency,
             "Ns": self.Ns, "n": self.n,
             "Il": self.Il, "I0": self.I0, "rs": self.rs, "rsh": self.rsh, "t": self.t, "q": self.q, "k": self.k,
             "nNsVth": self.nNsVth, "current_multiplier": self.current_multiplier,
             "VOC_temp_coefficient": self.VOC_temp_coefficient,
             "ISC_temp_coefficient": self.ISC_temp_coefficient, "PMPP": self.PMPP,
             "fill_factor": self.fill_factor, "max_efficiency": self.max_efficiency})

        df = pd.DataFrame.from_dict(solar_cell, orient='index')
        path = 'data/solar_cell_profiles/' + type + self.model + '.csv'
        df.to_csv(path)
        return path

    def SC_obj_from_csv(self, path):
        """
        This function loads the solar cell parameter from a CSV file into an empty solar cell object.
        :param path:
        The path of the CSV file.
        :return:
        """
        import pandas as pd

        df = pd.read_csv(path, index_col='Unnamed: 0')
        solar_cell = df.to_dict()['0']

        self.model = solar_cell['model']
        self.producer = solar_cell['producer']
        self.width = float(solar_cell['width'])
        self.length = float(solar_cell['length'])
        self.VOC = float(solar_cell['VOC'])
        self.ISC = float(solar_cell['ISC'])
        self.VMPP = float(solar_cell['VMPP'])
        self.IMPP = float(solar_cell['IMPP'])
        self.nominal_PMPP = float(solar_cell['nominal_PMPP'])
        self.nominal_fill_factor = float(solar_cell['nominal_fill_factor'])
        self.nominal_efficiency = float(solar_cell['nominal_efficiency'])
        self.Ns = int(solar_cell['Ns'])
        self.n = float(solar_cell['n'])
        self.Il = float(solar_cell['Il'])
        self.I0 = float(solar_cell['I0'])
        self.rs = float(solar_cell['rs'])
        self.rsh = float(solar_cell['rsh'])
        self.t = int(solar_cell['t'])
        self.q = float(solar_cell['q'])
        self.k = float(solar_cell['k'])
        self.nNsVth = float(solar_cell['nNsVth'])
        self.current_multiplier = float(solar_cell['current_multiplier'])
        self.VOC_temp_coefficient = float(solar_cell['VOC_temp_coefficient'])
        self.ISC_temp_coefficient = float(solar_cell['ISC_temp_coefficient'])
        self.PMPP = float(solar_cell['PMPP'])
        self.fill_factor = float(solar_cell['fill_factor'])
        self.max_efficiency = float(solar_cell['max_efficiency'])
        return "solar cell profile loaded"

    def generate_I_V_curve(self, plt_show=False):
        """
        This function generates a curve from the given VMPP,VOC,IMPP,ISC
        :param plt_show:
        If it is True a simple plot of curve will be shown. Default is False.
        :return:
        Returns the exponential values for the curve as X and Y.
        """
        import matplotlib.pyplot as plt

        def exponential_fit(v_orig, i_orig):
            import numpy as np
            from scipy.optimize import curve_fit
            v = []
            i = []

            if len(v_orig) > 5:
                x_orig = v_orig[2:]
                i_orig = i_orig[2:]

            for j in range(0, len(v_orig), 1):
                v.append(v_orig[j])
                i.append((i_orig[j]))

            for j in range(len(v)):
                v[j] = v[j] * 100

            def func(x, a, b, c):
                return a * np.exp(-b * x) + c

            popt, pcov = curve_fit(func, v, i, p0=(1, 1e-6, 1), maxfev=100000)
            v_exponential = np.linspace(min(v) - (min(v) * 0.5), max(v), 100000)
            i_exponential = func(v_exponential, *popt)

            for j in range(len(v)):
                v[j] = v[j] / 100
            for j in range(len(v_exponential)):
                v_exponential[j] = v_exponential[j] / 100

            def check_validity():
                i_hist = np.histogram(i_exponential)
                successful_exponential = True
                if (min(i_hist[0]) == max(i_hist[0])):
                    successful_exponential = False

                diff = np.diff(i_exponential)[::-1]
                check_diff = diff[0]
                len_diff = int(len(diff) / 2)
                invalid = 0
                for j in range(1, len_diff, 1):
                    if abs(diff[j] - check_diff) < 1:
                        invalid = invalid + 1
                    else:
                        invalid = 0
                    check_diff = diff[j]
                if invalid > 10:
                    successful_exponential = False
                return successful_exponential

            # print("X: ", list(x)[::-1])
            # print(20 * "--")
            # print("Y: ", list(y))
            # print(20 * "==")
            # print("v_exponential: ", list(v_exponential))
            # print(20 * "--")
            # print("i_exponential: ", list(i_exponential)[::-1])
            # print(20 * "XXX")

            successful_exponential = check_validity()

            return v_exponential, i_exponential, successful_exponential

        dev = self.current_multiplier * 1000

        v = [0, self.VMPP, self.VOC]
        i = [self.ISC * dev, self.IMPP * dev, 0]

        v_exponential, i_exponential, check = exponential_fit(v, i)

        if plt_show:
            plt.plot(v_exponential, i_exponential, '.')
            plt.plot(v, i, '*')
            plt.show()

        return v_exponential, i_exponential

    def find_r_shunt_from_I_V_curve(self, plt_show=False, point=67585):
        """
        This function calculates geometricaly the R_Shunt which have a big impact of the curve points between ISC and PMP.
        :param point:
        The point which should be considerd as the end of the line.
        :return:
        The Value of R_Shunt in Ohm.
        """
        import matplotlib.pyplot as plt
        v_exponential, i_exponential = self.generate_I_V_curve()
        import numpy as np
        k = point

        dev = self.current_multiplier * 1000
        i_exp_new = []
        for j in range(len(i_exponential)):
            i_exp_new.append(i_exponential[j] / dev)
            # if abs(i_exponential[j] - IMP) < 10000:
            #     print(i,i_exponential[j], "True")
        rsh = - (np.mean(np.diff(i_exp_new[:k])) / np.mean(np.diff(v_exponential[:k])))
        if plt_show:
            plt.plot(v_exponential, i_exp_new, label="STC_Cruve")
            plt.plot(v_exponential[k], i_exp_new[k], marker='o', color="red", label="Slope Point 1")
            plt.plot(v_exponential[0], i_exp_new[0], marker='o', color="red", label="Slope Point 2")
            plt.xlabel("Solar Cell Voltage (V)")
            plt.ylabel("Solar Cell Current (mA)")
            plt.title("Finding the Shunt Resistance for " + self.model, fontsize=10, y=0.99)
            plt.suptitle(
                "The point at 67585 between 100000 points results to a value close to real value of R Shunt. \n"
                "This point is found by performing many tests and is highly empirical.", fontsize=8, y=0.99)
            plt.grid(True)
            plt.legend(loc='best', fancybox=True, shadow=True)
            plt.savefig("Find_RSH_" + self.model + ".png", bbox_inches="tight", pad_inches=1,
                        transparent=True, facecolor="w", edgecolor='w',
                        orientation='landscape', dpi=300)
            plt.show()
        r_shunt = 1 / rsh
        return r_shunt

    def curve(self, case, plt_show=False, save_figure=False):
        """
        If all parameters are calculated, curves can be generated, plotted/saved.
        :param case:
        It should be a list of Tuple of(Irradiation, Temperature) like: [(1000,25)]
        :param plt_show:
        If the curve needs to be plotted, it must be True.
        :param save_figure:
        If the curve needs to be saved, it must be True.
        :return:
        """

        import pandas as pd
        from pvlib import pvsystem

        parameters = {
            'Name': self.model,
            'Date': time.ctime(time.time()),
            'T_NOCT': 25,  # Normal Operating Cell Temperature
            'N_s': 72,  # number of cells
            'I_sc_ref': self.ISC,
            'V_oc_ref': self.VOC,
            'I_mp_ref': self.IMPP,
            'V_mp_ref': self.VMPP,
            'alpha_sc': self.ISC_temp_coefficient,
            'beta_oc': self.VOC_temp_coefficient,
            'a_ref': self.n,  # ** this parameter needed
            'I_L_ref': self.Il,  # ** this parameter needed
            'I_o_ref': self.I0,  # ** this parameter needed
            'R_s': self.rs,  # ** this parameter needed
            'R_sh_ref': self.rsh,  # ** this parameter needed
            'nNsVth': self.nNsVth,
            'current_multiplier': self.current_multiplier,
            'Technology': 'Mono-c-Si',
        }

        conditions = pd.DataFrame(case, columns=['Irradiation', 'Temperature'])

        IL, I0, Rs, Rsh, nNsVth = pvsystem.calcparams_desoto(
            conditions['Irradiation'],
            conditions['Temperature'],
            alpha_sc=parameters['alpha_sc'],
            a_ref=parameters['nNsVth'],
            I_L_ref=parameters['I_L_ref'],
            I_o_ref=parameters['I_o_ref'],
            R_sh_ref=parameters['R_sh_ref'],
            R_s=parameters['R_s'],
            EgRef=1.121,
            dEgdT=-0.0002677
        )

        curve_info = pvsystem.singlediode(
            photocurrent=IL,
            saturation_current=I0,
            resistance_series=Rs,
            resistance_shunt=Rsh,
            nNsVth=nNsVth,
            ivcurve_pnts=40,
            method='lambertw'
        )
        if plt_show:
            import matplotlib.pyplot as plt
            plt.figure()
            for j, case in conditions.iterrows():
                label = (
                        "$Irradiation$ " + f"{case['Irradiation']} $W/m^2$\n"
                                           "$Tempertature$ " + f"{case['Temperature']} $C$"
                )
                plt.plot(curve_info['v'][j], curve_info['i'][j], label=label)
                v_mp = curve_info['v_mp'][j]
                i_mp = curve_info['i_mp'][j]
                plt.plot([v_mp], [i_mp], ls='', marker='o', c='k')

            plt.legend(loc=(1.0, 0))
            plt.grid()
            plt.xlabel('Module voltage [V]')
            if self.current_multiplier == 1000:
                plt.ylabel('Module current [mA]')
            elif self.current_multiplier == 1000000:
                plt.ylabel('Module current [A]')
            elif self.current_multiplier == 1:
                plt.ylabel('Module current [uA]')
            plt.title(parameters['Name'])
            if save_figure:
                plt.savefig("LambertW_" + parameters['Name'] + ".png", bbox_inches="tight", pad_inches=1,
                            transparent=True,
                            facecolor="w", edgecolor='w',
                            orientation='landscape', dpi=300)
            plt.show()

            # print(pd.DataFrame({
            #     'i_sc': curve_info['i_sc'],
            #     'v_oc': curve_info['v_oc'],
            #     'i_mp': curve_info['i_mp'],
            #     'v_mp': curve_info['v_mp'],
            #     'p_mp': curve_info['p_mp'],
            # }))

        return curve_info

    def start_profiling(self, plt_show=False):
        """
        After instantiation of empty solar cell object and loading the original profile, the profiling can be started.
        :param plt_show:
        If it is True a visualisation will be shown, to enable the user see tha values and status of a curve.
        It take much more time.
        :return:
        A message which let know that the process finished successfully.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from general_functions import max_power_point, exponential_fit
        import time
        import warnings

        warnings.filterwarnings("ignore", module='scipy')

        self.Il = self.il_check()
        print("IL finished: ", self.Il)
        self.rsh = self.find_r_shunt_from_I_V_curve(point=67585)
        print("Rshunt finished: ", self.rsh)

        def original():
            # This functions processes a curve based on 5 important values
            # Specially it is important for solar cell profiling and plotting the generated curve
            def generate_curve_from_VOC_ISC_VMP_IMP(ISC, IMP, VOC, VMP, PMP):
                if abs(IMP * VMP - PMP) > 1:
                    v = [0, VMP, VOC]
                    i = [ISC, IMP, 0]
                else:
                    VSlope = (VMP * 100) / VOC
                    ISlope = (IMP * 100) / ISC

                    P1I = IMP + (IMP * ISlope)
                    P1V = 0

                    P2I = 0
                    P2V = VMP + (VMP * VSlope)

                    v = [0, VMP, P2V]
                    i = [P1I, IMP, 0]

                v_exponential, i_exponential, check = exponential_fit(v, i)

                new_VMP, new_IMP, new_PMP, new_power_points = max_power_point(v_exponential, i_exponential)

                result = {"PMP": PMP, "IMP": IMP, "VMP": VMP, "new_PMP": new_PMP, "new_IMP": new_IMP,
                          "new_VMP": new_VMP,
                          "v_exponential": v_exponential, "i_exponential": i_exponential}

                return result

            result = generate_curve_from_VOC_ISC_VMP_IMP(self.ISC * self.current_multiplier,
                                                         self.IMPP * self.current_multiplier, self.VOC,
                                                         self.VMPP,
                                                         self.PMPP)
            VMP, IMP, PMP, power_points = max_power_point(result['v_exponential'], result['i_exponential'])

            if plt_show:
                plt.plot(result['v_exponential'], result['i_exponential'], label="Orig.")
                plt.plot(VMP, IMP, 'o', label="Orig.-PMP")

            return VMP, IMP, PMP, result['v_exponential'], result['i_exponential']

        def from_obj():
            curve_new = self.curve([(1000, 25)])
            v = list(curve_new['v'][0])
            i = list(curve_new['i'][0])
            for j in range(len(i)):
                i[j] = i[j] * self.current_multiplier
            VMP, IMP, PMP, power_points = max_power_point(v, i)
            if plt_show:
                plt.plot(v, i, label="LambertW")
                plt.plot(VMP, IMP, 'o', label="LambertW-PMP")

            # v1, i1, check = exponential_fit(v, i)
            # VMP, IMP, PMP, power_points = max_power_point
            # if plt_show:
            #     plt.plot(v1, i1, label="LambertW_exp.")
            #     plt.plot(VMP, IMP, 'o', label="LambertW_exp. PMP")
            return VMP, IMP, PMP, v, i

        def ref_n_i0_nNsVth():
            n = 1.0
            result = []
            while n < 2.4:
                n = round(n + 0.00001, 5)
                I0 = (self.ISC - (self.VOC / self.rsh)) / np.exp(
                    (self.q * self.VOC) / (self.Ns * n * self.k * self.t))

                nNsVth = (self.Ns * self.n * self.k * self.t) / self.q
                result.append({'n': n, 'I0': I0, "nNsVth": nNsVth})
                # print("n: {}, I0: {}, nNsVth: {}".format(a, I0, nNsVth))
            return result

        def ref_rs():
            rs = 0
            unknown = True

            while unknown:
                rs = rs + 0.00001
                imp_new = self.ISC - (self.I0 * np.exp(
                    self.q * ((self.VMPP + self.IMPP * rs) / (
                            self.Ns * self.n * self.k * self.t)))) - (
                                  (self.VMPP + self.IMPP * rs) / (self.rsh))
                # print("Found rs: ", rs, round(imp_new, 2))
                if round(imp_new, 2) == self.IMPP:
                    unknown = False
                    self.rs = rs

        start = time.time()
        j = 0
        result = ref_n_i0_nNsVth()

        while True:
            self.n = result[j]['n']
            self.I0 = result[j]['I0']
            try:
                self.nNsVth = (self.Ns * self.n * self.k * self.t) / self.q
            except RuntimeWarning:
                print(
                    "Overflow occured at n: {}, I0: {}, nNsVth: {}, Rs: {}".format(self.n, self.I0,
                                                                                   self.nNsVth,
                                                                                   self.rs))
                break
            ref_rs()

            VMPO, IMPO, PMPO, v_exponentialO, i_exponentialO = original()
            VMP, IMP, PMP, v, i = from_obj()
            print("n: {}, I0: {}, nNsVth: {}, Rs: {}, IMP-Diff: {}, VMP-Diff: {}, "
                  "PMP-Diff: {}".format(self.n, self.I0, self.nNsVth,
                                        self.rs, (IMP / self.current_multiplier - IMPO / self.current_multiplier),
                                        (VMP - VMP), PMP / self.current_multiplier - PMPO / self.current_multiplier))

            if (round(IMP / self.current_multiplier, 5) >= round(IMPO / self.current_multiplier, 5)) and \
                    (round(VMP, 2) == round(VMPO, 2)) or \
                    (round(PMP / self.current_multiplier, 5) >= round((PMPO / self.current_multiplier) + 0.2, 5)):
                plt.clf()
                plt.suptitle(
                    "Datasheet parameters -> VMP: {}, IMP: {}, PMP: {}"
                    .format(round(VMPO, 5), round(IMPO, 5), round(PMPO / self.current_multiplier, 5)) +
                    "\n Calculated parameters -> VMP: {}, IMP: {}, PMP: {}"
                    .format(round(VMP, 5), round(IMP, 5), round(PMP / self.current_multiplier, 5)) +
                    "\n Parameters: n: {},|| I0: {},|| Rsh: {},|| Rs: {},|| Il: {},|| nNsVth: {}".format(
                        round(self.n, 5), round(self.I0, 8), round(self.rsh, 8), round(self.rs, 8),
                        round(self.Il, 8), round(self.nNsVth, 8)) + "\n Start: {}, Time elapsed: {}".format(
                        time.ctime(start), time.strftime("%H:%M:%S",
                                                         time.gmtime(time.time() - start))), fontsize=8, y=0.99)

                plt.plot(v_exponentialO, i_exponentialO, label="Orig.")
                plt.plot(VMPO, IMPO, 'o', label="Orig.-PMP")

                plt.plot(v, i, label="LambertW")
                plt.plot(VMP, IMP, 'o', label="LambertW-PMP")

                plt.xlabel("Solar Cell Voltage (V)")
                plt.ylabel("Solar Cell Current (uA)")
                plt.grid(True)
                plt.legend(loc='best', fancybox=True, shadow=True)
                plt.savefig("Profiled_solar_cell_" + self.model + ".png", bbox_inches="tight", pad_inches=1,
                            transparent=True, facecolor="w", edgecolor='w',
                            orientation='landscape', dpi=300)
                plt.savefig("Profiled_solar_cell_" + self.model + ".png", bbox_inches="tight", pad_inches=1,
                            transparent=True, facecolor="w", edgecolor='w',
                            orientation='landscape', dpi=300)
                plt.pause(0.01)

                break
            if plt_show:
                plt.suptitle(
                    "Datasheet parameters -> VMP: {}, IMP: {}, PMP: {}".
                    format(round(VMPO, 5), round(IMPO, 5), round(PMPO / self.current_multiplier, 5)) +
                    "\n Calculated parameters -> VMP: {}, IMP: {}, PMP: {}"
                    .format(round(VMP, 5), round(IMP, 5), round(PMP / self.current_multiplier, 5)) +
                    "\n Parameters: a: {},|| I0: {},|| Rsh: {},|| Rs: {},|| Il: {},|| nNsVth: {}".format(
                        round(self.n, 5), round(self.I0, 8), round(self.rsh, 8), round(self.rs, 8),
                        round(self.Il, 8), round(self.nNsVth, 8)) + "\n Start: {}, Time elapsed: {}"
                    .format(time.ctime(start), time.strftime("%H:%M:%S",
                                                             time.gmtime(time.time() - start))), fontsize=8, y=0.99)
                plt.xlabel("Solar Cell Voltage (V)")
                plt.ylabel("Solar Cell Current (uA)")
                plt.grid(True)
                plt.legend(loc='best', fancybox=True, shadow=True)
                plt.draw()
                plt.pause(0.01)
                plt.clf()
            j = j + 5

        print(30 * "-")
        print("Start: ", time.ctime(start))
        print("End: ", time.ctime(time.time()))
        print(time.strftime("%H:%M:%S", time.gmtime(time.time() - start)))

        self.curve(case=[(1000, 25), (800, 25), (600, 25), (400, 25), (200, 25)], plt_show=True)
        self.SC_obj_to_csv(type='Calculated_')

        print("a and I0 and nNsVth finished: ", self.n, self.I0, self.nNsVth)
        print("R Series finished: ", self.rs)
        self.SC_obj_to_csv("Calculated_")
        print(self)
        return "Profiling Done"
