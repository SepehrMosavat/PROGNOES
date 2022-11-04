def max_power_point(v, i):
    """
    This function finds the maximum power point and product of v and i points.
    :param v:
    A list of all volt (v) values.
    :param i:
    A list of all current (i) values.
    :return:
    Returns the VMP,IMP,PMP, power_points(a list of product of all v * i points)
    For example 4.26,49.9,212.575,[]
    """
    power_points = []
    for value in range(len(v)):
        power_points.append(v[value] * i[value])
    VMP = v[power_points.index(max(power_points))]
    IMP = i[power_points.index(max(power_points))]
    PMP = max(power_points)
    return VMP, IMP, PMP, power_points


def sorter(keylist):
    """
    HFD5 library don't provide a function to returnes the dataset in the given order.
    Therefore, the curves in a HDF5 file will be in a ordered and sorted sequence.
    For example there are 12 curves. 0,1,10,11,12,2,3,4,5,6,7,8,9.
    This function get a list of all Dataset keys in a HDF5 and sorts the numbers.
    :param keylist:
    A list of all keys in a Dataset.
    :return:
    Returns a list of the numbers of curve names in a Dataset. Like [0,1,2,3,4]
    """
    temp = []
    sortedlist = []
    for item in keylist:
        if item != "harvesting conditions" and item != "weather_conditions" and \
                item != "irradiation_weather_conditions" and item != "prognoes_conditions":
            temp.append(int(item[5:]))
    temp.sort(key=int)
    for item in temp:
        sortedlist.append('curve' + str(item))
    return sortedlist


def float_rounder(liste):
    """
    This function is used when reading a HDF5 file. It accepts a list of float numbers that may have a multi-digit decimal.
    The list is processed so that there are only two decimal numbers in the list.
    By reading the v and i values of a curve in a dataset, the values should be rounded.
    :param liste:
    A list of values containing float numbers.
    :return:
    The numbers will be rounded to two digit. For example: 1.1460 to 1.15
    """
    import math
    new_list = []
    for item in liste:
        new_list.append(math.ceil(item * 100) / 100)
    return new_list


def interp1d_curve_fit_function(v, i):
    """
    This function interpolates the values of 'i' as y and 'v' as x with scipy.
    This function will be used in if the curves recorded in a very low light conditions which lead to noisy curves.
    :param v:
    A list of 'v' values.
    :param i:
    A list of 'i' values.
    :return:
    Returns sorted interpolated values of v[] and i[].
    """
    from scipy.interpolate import interp1d
    from scipy.optimize import curve_fit
    import numpy as np

    cubic_interploation_model = interp1d(v, i, kind="linear")
    v_interp1d = np.linspace(min(v), max(v), 40)
    i_interp1d = cubic_interploation_model(v_interp1d)

    def func(x, a, b, c):
        return a * x ** 3 + b * x ** 2 + c

    params, covs = curve_fit(func, v_interp1d, i_interp1d)
    a, b, c = params[0], params[1], params[2]
    ifited = a * v_interp1d ** 3 + b * v_interp1d ** 2 + c

    return np.sort(v_interp1d), np.sort(ifited)[::-1]


def exponential_fit(v_orig, i_orig):
    """
    This function is used for an exponential fit of a curve
    If it is a line with properties of an exponential line at all.
    This function checks if the generated exponential curve has the properties of an exponential curve.
    :param v:
    A list of 'v' values.
    :param i:
    A list of 'i' values.
    :return:
    Returns a list of 'v', 'i' in exponential forms and True/Flase if exponential fit is successful or not.
    """
    import numpy as np
    from scipy.optimize import curve_fit
    x = []
    y = []

    if len(v_orig) > 5:
        x_orig = v_orig[2:]
        y_orig = i_orig[2:]

    for i in range(0, len(v_orig), 1):
        x.append(v_orig[i])
        y.append((i_orig[i]))

    for i in range(len(x)):
        x[i] = x[i] * 100

    def func(x, a, b, c):
        return a * np.exp(-b * x) + c

    popt, pcov = curve_fit(func, x, y, p0=(1, 1e-6, 1), maxfev=100000)
    v_exponential = np.linspace(min(x) - (min(x) * 0.5), max(x), 40)
    i_exponential = func(v_exponential, *popt)

    for i in range(len(x)):
        x[i] = x[i] / 100
    for i in range(len(v_exponential)):
        v_exponential[i] = v_exponential[i] / 100

    def check_validity():
        y_hist = np.histogram(i_exponential)
        successful_exponential = True
        if (min(y_hist[0]) == max(y_hist[0])):
            successful_exponential = False

        diff = np.diff(i_exponential)[::-1]
        check_diff = diff[0]
        len_diff = int(len(diff) / 2)
        invalid = 0
        for i in range(1, len_diff, 1):
            if abs(diff[i] - check_diff) < 1:
                invalid = invalid + 1
            else:
                invalid = 0
            check_diff = diff[i]
        if invalid > 10:
            successful_exponential = False
        return successful_exponential

    successful_exponential = check_validity()
    i_exponential[-1] = 0

    return v_exponential, i_exponential, successful_exponential


def generate_curve_from_noisy_data_points(v, i):
    """
    This function accepts a curve from a very intoxicated curve,whose IMP is less than 300 and looks more like a
    straight line, and does not contain any properties of an exponential curve.
    :param v:
    A list of 'v' values.
    :param i:
    A list of 'i' values.
    :return:
    Returns a exponential fited curve if possible, where as X[] contains the 'v' values and Y[] contains the 'i' values
    and True/False whether the exponential fit was successful or not.
    """
    fitted_v, fitted_i = interp1d_curve_fit_function(v, i)
    VMP, IMP, PMP, power_points = max_power_point(fitted_v, fitted_i)

    multipicator = 1

    while IMP < 300:
        if IMP < 1:
            IMP = IMP * 1000
            multipicator = multipicator * 1000
        elif IMP < 100:
            IMP = IMP * 10
            multipicator = multipicator * 10

        if IMP < 301:
            IMP = IMP * 10
            multipicator = multipicator * 10

    P2V = VMP - (VMP * 0.1)
    P2I = IMP + (IMP * 0.1)

    P3V = VMP + (VMP * 0.1)
    P3I = IMP - (IMP * 0.1)

    P1V = 0
    P1I = IMP + (IMP * 0.25)

    P4V = VMP + (VMP * 0.25)
    P4I = 0

    v = [P1V, P2V, VMP, P3V, P4V]
    i = [P1I, P2I, IMP, P3I, P4I]

    v_exponential, i_exponential, check = exponential_fit(v, i)

    for item in range(len(v_exponential)):
        v_exponential[item] = v_exponential[item] / 1
        i_exponential[item] = i_exponential[item] / multipicator

    return v_exponential, i_exponential, check


def fit_the_curve(v, i):
    """
    This function decides whether a curve should rather be fitted directly with exponential function
    after interpolation or not. This can be extended to exit the process if the maximum 'i' value is under 300.
    :param v:
    A list of 'v' values.
    :param i:
    A list of 'i' values.
    :return:
    Returns a exponential fited curve if possible, where as v contains the 'v' values and i contains the 'i' values
    and True/False whether the exponential fit was successful or not.
    """
    fitted_v, fitted_i, exponential = exponential_fit(v, i)
    if not exponential:
        fitted_v, fitted_i, exponential = generate_curve_from_noisy_data_points(v, i)
        exponential = True
    return fitted_v, fitted_i, exponential


def solar_info(prognoes_conditions, solar_cell_model):
    """
    This function processes the irradiation, based on HDF5 Metadata and given parameters,
    which will be used by generating plot subtitle.
    :param prognoes_conditions:
    It is dict containing prognoes conditions. To see example of it, see PROGNOES.Definitions()
    :param solar_cell_model:
    The solar cell model. Like: 'SM141K08LV'
    :param prediction_method:
    It can be 'cn' for Campbell-Norman or
    It can be 'cs' for ClearSky
    :param cloud_source:
    It can be 'GFS' for GFS as cloud data source or
    It can be 'mowesta' for mowesta as cloud data source
    :return:
    Returns a dictionary contating the irradiation for GHIs from GFS, IandP and Pysolar.
    The first number is the irradiation for one sun and the secound one is the calculated value for the solar cell.
    for Example: {"wm2_pysolar_mowesta_cs": 749, "wm2c_pysolar_mowesta_cs":255,
                  "wm2_mowesta_cs": 525 , "wm2c_mowesta_cs": 345,
                  "wm2_mowesta_cn": 799, "wm2c_mowesta_cn": 540,
                  "wm2_GFS_GFS": 586, "wm2c_GFS_GFS": 320,
                  "wm2_GFS_cs": 804, "wm2c_GFS_cs" : 525,
                  "wm2_GFS_cn": 294, "wm2c_GFS_cn": 89}
    """
    from pvlib_functions import PMP_calculator
    import datetime

    timestamp = float(prognoes_conditions['start_time'])

    wm2_pysolar_mowesta_cs, wm2c_pysolar_mowesta_cs = PMP_calculator(prognoes_conditions=prognoes_conditions,
                                                                     solar_cell_model=solar_cell_model,
                                                                     address=prognoes_conditions['address'],
                                                                     timestamp=datetime.datetime.fromtimestamp(
                                                                         (int(timestamp))),
                                                                     irradiation='pysolar',
                                                                     cloud_source='mowesta', prediction_method='cs')

    wm2_mowesta_cs, wm2c_mowesta_cs = PMP_calculator(prognoes_conditions=prognoes_conditions,
                                                     solar_cell_model=solar_cell_model,
                                                     address=prognoes_conditions['address'],
                                                     timestamp=datetime.datetime.fromtimestamp((int(timestamp))),
                                                     irradiation='IandP',
                                                     cloud_source='mowesta', prediction_method='cs')

    wm2_mowesta_cn, wm2c_mowesta_cn = PMP_calculator(prognoes_conditions=prognoes_conditions,
                                                     solar_cell_model=solar_cell_model,
                                                     address=prognoes_conditions['address'],
                                                     timestamp=datetime.datetime.fromtimestamp((int(timestamp))),
                                                     irradiation='IandP',
                                                     cloud_source='mowesta', prediction_method='cn')

    wm2_GFS_GFS, wm2c_GFS_GFS = PMP_calculator(prognoes_conditions=prognoes_conditions,
                                               solar_cell_model=solar_cell_model,
                                               address=prognoes_conditions['address'],
                                               timestamp=datetime.datetime.fromtimestamp((int(timestamp))),
                                               irradiation='GFS',
                                               cloud_source='GFS', prediction_method='cn')

    wm2_GFS_cs, wm2c_GFS_cs = PMP_calculator(prognoes_conditions=prognoes_conditions,
                                             solar_cell_model=solar_cell_model,
                                             address=prognoes_conditions['address'],
                                             timestamp=datetime.datetime.fromtimestamp((int(timestamp))),
                                             irradiation='IandP',
                                             cloud_source='GFS', prediction_method='cs')

    wm2_GFS_cn, wm2c_GFS_cn = PMP_calculator(prognoes_conditions=prognoes_conditions,
                                             solar_cell_model=solar_cell_model,
                                             address=prognoes_conditions['address'],
                                             timestamp=datetime.datetime.fromtimestamp((int(timestamp))),
                                             irradiation='IandP',
                                             cloud_source='GFS', prediction_method='cn')

    result = {"wm2_pysolar_mowesta_cs": round(float(wm2_pysolar_mowesta_cs), 2),
              "wm2c_pysolar_mowesta_cs": round(float(wm2c_pysolar_mowesta_cs), 2),
              "wm2_mowesta_cs": round(float(wm2_mowesta_cs), 2),
              "wm2c_mowesta_cs": round(float(wm2c_mowesta_cs), 2),
              "wm2_mowesta_cn": round(float(wm2_mowesta_cn), 2),
              "wm2c_mowesta_cn": round(float(wm2c_mowesta_cn), 2),
              "wm2_GFS_cs": round(float(wm2_GFS_cs), 2),
              "wm2c_GFS_cs": round(float(wm2c_GFS_cs), 2),
              "wm2_GFS_cn": round(float(wm2_GFS_cn), 2),
              "wm2c_GFS_cn": round(float(wm2c_GFS_cn), 2),
              "wm2_GFS_GFS": round(float(wm2_GFS_GFS), 2),
              "wm2c_GFS_GFS": round(float(wm2c_GFS_GFS), 2)}
    return result


def get_metadata(path):
    """
    This method converts the meta data into a meaningful text for the subtitle of plots.
    This method used everywhere which a HDF5 file needs to be reed.
    :param path:
    The path of the HDF5 file containing the metadata.
    :param prediction_method:
    It can be 'cn' for Campbell-Norman or
    It can be 'cs' for ClearSky.
    This will be passed to solar_info function.
    :param cloud_source:
    It can be 'GFS' for GFS as cloud data source or
    It can be 'mowesta' for mowesta as cloud data source
    This will be passed to solar_info function
    :return:
    Returns harvesting conditions as a dict for the case the beginning and end of recording must be considered.
    See PROGNOES.Definitions for the structure of the dict.
    Returns a string which used by plotting.
    """
    import numpy as np
    import h5py
    import datetime
    from solar_cell_class import solar_cell

    f = h5py.File(path, 'a')

    harvesting_conditions = {"available": True}
    weather_conditions = {"available": False}
    irradiation_weather_condition = {"available": False}
    prognoes_conditions = {"available": False}

    for label, value in zip(np.array(f.get("harvesting conditions")[0]), np.array(f.get("harvesting conditions")[1])):
        harvesting_conditions.update({label.decode('utf-8'): value.decode('utf-8')})

    prognoes_conditions['available'] = True
    for label, value in zip(np.array(f.get("prognoes_conditions")[0]), np.array(f.get("prognoes_conditions")[1])):
        prognoes_conditions.update({label.decode('utf-8'): value.decode('utf-8')})

    solar_cell = solar_cell()
    solar_cell.SC_obj_from_csv(
        'data/solar_cell_profiles/Calculated_' + prognoes_conditions['solar_cell'] + '.csv')

    irradiation_weather_condition['available'] = True
    for label, value in zip(np.array(f.get("irradiation_weather_conditions")[0]),
                            np.array(f.get("irradiation_weather_conditions")[1])):
        irradiation_weather_condition.update({label.decode('utf-8'): value.decode('utf-8')})

    weather_conditions['available'] = True
    for label, value in zip(np.array(f.get("weather_conditions")[0]),
                            np.array(f.get("weather_conditions")[1])):
        weather_conditions.update({label.decode('utf-8'): value.decode('utf-8')})

    def plt_subtitle_maker(harvesting_conditions, weather_conditions, irradiation_weather_conditions,
                           prognoes_conditions, solar_info):
        begin_time = datetime.datetime.strptime(
            harvesting_conditions['Date'] + ', ' + harvesting_conditions['Start Time (Local Timezone)'],
            "%d.%m.%Y, %H:%M:%S.%f")
        end_time = datetime.datetime.strptime(
            harvesting_conditions['Date'] + ', ' + harvesting_conditions['End Time (Local Timezone)'],
            "%d.%m.%Y, %H:%M:%S.%f")

        start = str(begin_time.hour) + ":" + str(begin_time.minute) + ":" + str(begin_time.second)
        end = str(end_time.hour) + ":" + str(end_time.minute) + ":" + str(end_time.second)
        plt_subtitle = ""
        if irradiation_weather_conditions['available']:
            plt_subtitle = 'MoWeSta : -> ' + " | Clouds: " + str(
                round(float(weather_conditions['total_clouds']), 1)) + " | Wind Speed: " + str(
                round(float(weather_conditions['windSpeed']), 1)) + " | Temperature: " + str(
                round(float(weather_conditions["temperature"]), 1)) + " | Rain: " + str(
                round(float(weather_conditions['precipitation']), 1)) + " | Snow: " + str(
                round(float(weather_conditions["snowDepth"]), 1)) + " | Humidity: " + str(
                round(float(weather_conditions["humidity"]), 1)) + " | Pressure: " + str(
                round(float(weather_conditions["pressure"]), 1)) + '\n' + 'GFS: -> ' + \
                           " | Temperature: " + str(round(float(irradiation_weather_conditions["temp_air"]), 1)) + \
                           " | Total Clouds: " + str(round(float(irradiation_weather_conditions['total_clouds']), 1)) \
                           + " | Address: " + prognoes_conditions['address'] + ' | Start: ' + start + ' | End: ' + end + \
                           ' | Date: ' + harvesting_conditions['Date'] + '\n' + \
                           "| Pysolar_M_CS: " + str(solar_info['wm2_pysolar_mowesta_cs']) + " -> " + str(
                solar_info['wm2c_pysolar_mowesta_cs']) + " | MoWeSta_CS: " + str(
                solar_info['wm2_mowesta_cs']) + " -> " + str(
                solar_info['wm2c_mowesta_cs']) + " | MoWeSta_CN: " + str(
                solar_info['wm2_mowesta_cn']) + " -> " + str(
                solar_info['wm2c_mowesta_cn']) + " | GFS_CS: " + str(
                solar_info['wm2_GFS_cs']) + " -> " + str(
                solar_info['wm2c_GFS_cs']) + '\n' + " | GFS_CN: " + str(
                solar_info['wm2_GFS_cn']) + " -> " + str(
                solar_info['wm2c_GFS_cn']) + "| GFS_GFS: " + str(
                solar_info['wm2_GFS_GFS']) + " -> " + str(solar_info['wm2c_GFS_GFS'])
        return plt_subtitle

    title = "Solar Cell IV Characteristics"
    ylabel = 'Time (s)'
    xlabel = 'Solar Cell Voltage (V)'
    zlabel = 'Solar Cell Current (uA)'
    info = plt_subtitle_maker(harvesting_conditions=harvesting_conditions, weather_conditions=weather_conditions,
                              irradiation_weather_conditions=irradiation_weather_condition,
                              prognoes_conditions=prognoes_conditions,
                              solar_info=solar_info(prognoes_conditions=prognoes_conditions,
                                                    solar_cell_model=prognoes_conditions['solar_cell']))

    plt_meta_data = {"title": title, "ylabel": ylabel, "xlabel": xlabel, "zlabel": zlabel, "info": info}

    return harvesting_conditions, plt_meta_data


def extract_z_from_hdf5(path):
    """
    This function get a HDF5 file and extract the curves. Each point of the curve has a third dimmension too.
    The third dimension (z) of the point corresponds the captured time.
    All point of a curve have the same value for z, since the capturing happens very fast.
    :param path:
    The Path of the HDF5 file which contains the curves with metadata and 2 dimensional curves.
    :return:
    Returns a data frame, which contains the curves and 3 dimensional values.
    The columns of this dataframe are: ['Solar Cell Voltage (V)', 'Solar Cell Current (uA)', 'Time (s)', 'Curve']
    """
    import numpy as np
    import pandas as pd
    import h5py
    import datetime
    f = h5py.File(path, 'a')
    harvesting_conditions = {}
    for label, value in zip(np.array(f.get("harvesting conditions")[0]), np.array(f.get("harvesting conditions")[1])):
        harvesting_conditions.update({label.decode('utf-8'): value.decode('utf-8')})

    begin = datetime.datetime.timestamp(datetime.datetime.strptime(
        harvesting_conditions['Date'] + ', ' + harvesting_conditions['Start Time (Local Timezone)'],
        "%d.%m.%Y, %H:%M:%S.%f"))
    end = datetime.datetime.timestamp(datetime.datetime.strptime(
        harvesting_conditions['Date'] + ', ' + harvesting_conditions['End Time (Local Timezone)'],
        "%d.%m.%Y, %H:%M:%S.%f"))

    dataset = sorter(list(f.keys()))
    timedelta = np.ceil(end - begin) / len(dataset)
    curveTime = 0
    df = pd.DataFrame([[72, 67, 91, 'test']],
                      columns=['Solar Cell Voltage (V)', 'Solar Cell Current (uA)', 'Time (s)', 'Curve'])
    for item in dataset:
        zList = []
        curveName = []
        temp = curveTime
        for value in range(len(list(f[item][0]))):
            temp = temp
            zList.append(temp)
            curveName.append(item)

        v, i, check = fit_the_curve(float_rounder(list(f[item][0])), float_rounder(list(f[item][1])))
        df_temp = pd.DataFrame(list(zip(v, i, zList, curveName)),
                               columns=['Solar Cell Voltage (V)', 'Solar Cell Current (uA)', 'Time (s)', 'Curve'])

        curveTime = curveTime + timedelta
        df = df.append(df_temp, ignore_index=True)

    df = df.iloc[1:]
    return df
