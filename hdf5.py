import sys
import warnings
import datetime

warnings.filterwarnings("ignore", module='pvlib')
warnings.filterwarnings("ignore", module='pysolar')


def hdf5_reader(path='data/examples/SOCRAETES_SM141K08LV_211028_130753.hdf5', dot_original=False, plt_original=True,
                curve_fit=False, draw_power_points=True):
    """
    This function get a HDF5 file, recorded with SOCRAETES recorder and processed it to plot the curves in the file.
    :param path:
    Path of the HDF5 file.
    :param dot_original:
    The original curve will plotted not as a continues line, but each point will be plotted as dot. Default is False.
    This will be passed to iv_curve_plotting function.
    :param plt_original:
    The original curve will plotted as a continues line. Default is True.
    This will be passed to iv_curve_plotting function.
    :param curve_fit:
    Fitted the original curve and plot it separately for comparing with the original one. Default is False.
    This will be passed to iv_curve_plotting function.
    :param draw_power_points:
    All products of the point namely i x v will be plotted. Default is False.
    This will be passed to iv_curve_plotting function.
    :return:
    This function has no return and will be run endless, until the user interrupt the processing.
    """
    import numpy as np
    import h5py
    import datetime
    from general_functions import sorter, float_rounder, get_metadata
    from plotting import iv_curve_plotting

    f = h5py.File(path, 'a')
    if ("harvesting conditions" not in f) or ("weather_conditions" not in f) or (
            "irradiation_weather_conditions" not in f) or ("prognoes_conditions" not in f):
        print("You are using an old format which is not compatible to PROGNOES, a copy of your file will be processed.")
        print(
            "If the GFS and MoWeSta datasets are available, irradiance can be calculated based on the selected methods.")
        print(
            "The following questions are very important in order to update file version. If you don't know press enter:")

        solar_cell_val = input("What type is the solar cell? ") or "SM141K08LV"
        thermometer_temperature_val = input('Solar cell temperature with measured with thermometer? ') or 25
        address_val = input('What was the address of the recording place? ') or "Kuglerstraße, Essen,45144"

        path = dataset_version_converter(path=path, solar_cell=solar_cell_val,
                                         thermometer_temperature=thermometer_temperature_val, address=address_val)
        f = h5py.File(path, 'a')

    harvesting_conditions, plt_meta_data = get_metadata(path=path)

    dataset = sorter(list(f.keys()))

    begin = datetime.datetime.timestamp(datetime.datetime.strptime(
        harvesting_conditions['Date'] + ', ' + harvesting_conditions['Start Time (Local Timezone)'],
        "%d.%m.%Y, %H:%M:%S.%f"))
    end = datetime.datetime.timestamp(datetime.datetime.strptime(
        harvesting_conditions['Date'] + ', ' + harvesting_conditions['End Time (Local Timezone)'],
        "%d.%m.%Y, %H:%M:%S.%f"))
    timedelta = np.ceil(end - begin) / len(dataset)

    while True:

        harvesting_conditions['Start Time (Local Timezone)'] = begin
        harvesting_conditions['End Time (Local Timezone)'] = end
        for item in dataset:
            harvesting_conditions['End Time (Local Timezone)'] = harvesting_conditions[
                                                                     'Start Time (Local Timezone)'] + timedelta
            iv_curve_plotting(float_rounder(list(f[item][0])), float_rounder(list(f[item][1])), item,
                              plt_meta_data['info'], dot_original=dot_original, plt_original=plt_original,
                              curve_fit=curve_fit, draw_power_points=draw_power_points)
            harvesting_conditions['Start Time (Local Timezone)'] = harvesting_conditions['End Time (Local Timezone)']


def dataset_to_hdf5(solar_cell_model="SM141K08LV", address="Kuglerstraße, Essen,45144", latitude=51.455643,
                    longitude=7.011555, start_time=datetime.datetime.now(),
                    end_time=datetime.datetime.now() + datetime.timedelta(hours=2),
                    metadata_time=datetime.datetime.now(),
                    seconds_intervals=600.0, curve_pause=0.1,
                    verbose_show=False, plt_show=True, export_to_hdf=False, original_dataset=None):
    """
    This function creates Dataset based on the parameters given. Maximum of seven datasets could be generated.
    The methods should be set in the methods dict.
    :param solar_cell_model:
    The type of solar_cell. Default is 'SM141K08LV'
    :param address:
    The address of the recording place. Default is 'Kuglerstraße, Essen,45144'.
    :param latitude:
    The latitude of the given address. Default is: '51.455643'
    Antarctica has no address.
    :param longitude:
    the longitude of the given address. Default is: '7.011555'
    Antarctica has no address.
    :param start_time:
    The time since when the recording started / should start. Default is running time.
    :param end_time:
    The time when the recording is/should be finished. Default is runnig time + two hours.
    :param metadata_time:
    The time at them the metadata should be saved as reference in HDF5. Default is running time.
    :param seconds_intervals:
    The periods between the curves are captured. Default is 600 seconds (each 10 minutes).
    :param curve_pause:
    The period between the curves by plotting. Default is like SOCRAETES standard 0.1 seconds.
    :param verbose_show:
    Show the calculated values for the methods. Default is False.
    :param plt_show:
    Shows plotting during the generating the curves. Default is True.
    :param export_to_hdf:
    Save the curves and metadata to a HDF5. Default is False.
    :param original_dataset:
    If given, by plotting will the original curves reorded with SOCRAETES recorder shown. Default is None.
    :return:
    Function returns a message if the process done successfully.
    """
    from solar_cell_class import solar_cell
    from dateutil import tz
    import urllib
    import requests
    from pvlib_functions import GFS_day_minute
    from mowesta import mowesta_day_minute
    import matplotlib.pyplot as plt
    import os
    import h5py
    import pandas as pd
    from general_functions import max_power_point

    try:
        if address != "Kuglerstraße, Essen,45144":
            url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote(address) + '?format=json'
            response = requests.get(url).json()
            latitude = round(float(response[0]["lat"]), 7)
            longitude = round(float(response[0]["lon"]), 7)
    except:
        print(
            "An error occurred while connecting to the openstreetmap.org. Please check your connection and try again.")
        sys.exit()

    from timezonefinder import TimezoneFinder
    tf = TimezoneFinder()
    timezone = tf.timezone_at(lng=longitude, lat=latitude)

    for i in range((end_time - start_time).days + 1):
        GFS_day_minute(address=address, date=start_time + datetime.timedelta(days=i), sys_exit=True)
        mowesta_day_minute(address=address, date=start_time + datetime.timedelta(days=i), sys_exit=True)

    solar_cell = solar_cell()
    solar_cell.SC_obj_from_csv('data/solar_cell_profiles/Calculated_' + solar_cell_model + '.csv')

    methods = {"pysolar_mowesta_cs": True, "IandP_mowesta_cs": True, "mowesta_cs": True,
               "mowesta_cn": True, "gfs_ghi": True, "gfs_cs": True, "gfs_cn": True}

    def gen_curve_from_ghi(method="Pysolar", ghi=1000, temperature=27):

        def ghi_to_pmp():
            radiation_value = ghi * 1000  # * 1000 to get values in milli watt
            radiation_value_rounded_one_sun = round(radiation_value, 2)  # 617000

            radiation_temperature_coefficient = float(
                solar_cell.curve([(round(radiation_value / 1000, 2), float(temperature))])['p_mp'])
            if verbose_show:
                print(
                    "Method: {} -> PMP : W/M² = {}mW , {} with temperatur coefficient = {}mW  ".format(
                        method, radiation_value_rounded_one_sun, solar_cell_model,
                        round(radiation_temperature_coefficient, 2)))

            return radiation_value_rounded_one_sun / 1000

        def lambert_w_curve(irradiation):
            i_v_curve = solar_cell.curve([(irradiation, float(temperature))])
            v = list(i_v_curve['v'][0])
            i = list(i_v_curve['i'][0])
            i[-1] = 0

            for j in range(len(i)):
                i[j] = i[j] * solar_cell.current_multiplier
            data = {'v': v, 'i': i}
            return data

        irradiation = ghi_to_pmp()
        curve = lambert_w_curve(irradiation)
        return ghi, irradiation, curve

    def collect_ghis(date=datetime.datetime.now()):
        def dataset_minute_to_dict(dataset):
            start_loc = datetime.datetime(year=date.year, month=date.month, day=date.day, hour=date.hour,
                                          minute=date.minute,
                                          second=0, tzinfo=tz.gettz(timezone))
            filtered_df = dataset.loc[start_loc:start_loc + datetime.timedelta(minutes=1)]
            result_minute = filtered_df.head(1)
            result_minute = result_minute.reset_index()
            result = {}
            for key, values in result_minute.to_dict().items():
                for key_value, values_value in values.items():
                    result.update({key: values_value})

            result.update({'timestamp': str(start_loc)})
            return result

        mowesta_str_name = "PROGNOES_MOWESTA_" + str(int(address.split(",")[2])) + "_" + str(date.year)[2:] + (
            '{:02d}'.format(date.month)) + '{:02d}'.format(date.day) + ".csv"

        gfs_str_name = "PROGNOES_GFS_" + str(int(address.split(",")[2])) + "_" + str(date.year)[2:] + (
            '{:02d}'.format(date.month)) + '{:02d}'.format(date.day) + ".csv"

        if os.path.exists('data/mowesta_archive/' + mowesta_str_name):
            mowesta_dataset = pd.read_csv('data/mowesta_archive/' + mowesta_str_name, parse_dates=True,
                                          index_col='timestamp')
        else:
            print("MoWesta Dataset does not exist. Try to Download it!")
            mowesta_day_minute(address=address, latitude=latitude, longitude=longitude, date=date, sys_exit=True)
            mowesta_dataset = pd.read_csv('data/mowesta_archive/' + mowesta_str_name, parse_dates=True,
                                          index_col='timestamp')

        if os.path.exists('data/GFS_archive/' + gfs_str_name):
            gfs_dataset = pd.read_csv('data/GFS_archive/' + gfs_str_name, parse_dates=True,
                                      index_col='timestamp')
        else:
            print("GFS Dataset does not exist. Try to Download it!")
            GFS_day_minute(address=address, latitude=latitude, longitude=longitude, date=date, sys_exit=True)
            gfs_dataset = pd.read_csv('data/GFS_archive/' + gfs_str_name, parse_dates=True,
                                      index_col='timestamp')

        mowesta_result = dataset_minute_to_dict(mowesta_dataset)
        gfs_result = dataset_minute_to_dict(gfs_dataset)

        pysolar_mowesta_cs = [1000, 1000, {'v': 0, 'i': 0}]
        IandP_mowesta_cs = [1000, 1000, {'v': 0, 'i': 0}]
        mowesta_cs = [1000, 1000, {'v': 0, 'i': 0}]
        mowesta_cn = [1000, 1000, {'v': 0, 'i': 0}]
        gfs_ghi = [1000, 1000, {'v': 0, 'i': 0}]
        gfs_cs = [1000, 1000, {'v': 0, 'i': 0}]
        gfs_cn = [1000, 1000, {'v': 0, 'i': 0}]

        if methods['pysolar_mowesta_cs']:
            pysolar_mowesta_cs = gen_curve_from_ghi(method='pysolar_mowesta_cs', ghi=mowesta_result['pysolar_cs_ghi'])
        if methods['IandP_mowesta_cs']:
            IandP_mowesta_cs = gen_curve_from_ghi(method='IandP_mowesta_cs', ghi=mowesta_result['IandP_cs_ghi'])

        if methods['mowesta_cs']:
            mowesta_cs = gen_curve_from_ghi(method='mowesta_cs', ghi=mowesta_result['cs_ghi'])
        if methods['mowesta_cn']:
            mowesta_cn = gen_curve_from_ghi(method='mowesta_cn', ghi=mowesta_result['cn_ghi'])

        if methods['gfs_ghi']:
            gfs_ghi = gen_curve_from_ghi(method='gfs_ghi', ghi=gfs_result['GFS_ghi'])

        if methods['gfs_cn']:
            gfs_cn = gen_curve_from_ghi(method='gfs_cn', ghi=gfs_result['cs_ghi'])

        if methods['gfs_cs']:
            gfs_cs = gen_curve_from_ghi(method='gfs_cs', ghi=gfs_result['cn_ghi'])

        result = {"pysolar_mowesta_cs": pysolar_mowesta_cs, "IandP_mowesta_cs": IandP_mowesta_cs,
                  "mowesta_cs": mowesta_cs, "mowesta_cn": mowesta_cn, "gfs_ghi": gfs_ghi, "gfs_cs": gfs_cs,
                  "gfs_cn": gfs_cn}

        return result

    def plot_curve(result, start_time):
        for key, value in methods.items():
            if value:
                if max(result[key][2]['v']) > 0.5 and max(result[key][2]['i']) > 300:
                    plt.plot(result[key][2]['v'], result[key][2]['i'],
                             label=key)
                    VMP, IMP, PMP, power_points = max_power_point(result[key][2]['v'], result[key][2]['i'])
                    if verbose_show:
                        # print("Method: {}, ISC: {}, VOC: {}, IMP: {}, VMP:{}, PMP:{}"
                        #       .format(key, max(result[key][2]['i']), max(result[key][2]['v']), IMP, VMP, PMP))

                        print("Method: {}, & {} & {} & {} & {} & {}"
                              .format(key, round(max(result[key][2]['v']), 3),
                                      round(max(result[key][2]['i']) / 1000, 3), round(VMP, 3), round(IMP / 1000, 3),
                                      round(PMP / 1000), 3))

                    plt.plot(VMP, IMP, color='black', marker='o', markersize=2)

        harvesting_conditions = "This plot shows the behavior of the solar cell under the specific weather conditions on" \
                                " \n " + str(
            start_time) + ". The legend shows the methods used to create the curves. \n " \
                          "In compairing mode the original curve corresponds the curves recorded with SOCRAETES. "

        plt.suptitle(harvesting_conditions, fontsize=8, y=0.99)
        plt.legend(loc=(1.001, 0.535), fancybox=True, shadow=True)
        # plt.legend(loc='best', fancybox=True, shadow=True)
        plt.grid(True)
        plt.title("Solar Cell '" + str(solar_cell_model) + "' IV Characteristics", fontsize=9, y=0.99)
        plt.xlabel("Solar Cell Voltage (V)")
        plt.ylabel("Solar Cell Current (uA)")
        plt.savefig("Compare.png", bbox_inches="tight", pad_inches=1, transparent=True, facecolor="w", edgecolor='w',
                    orientation='landscape', dpi=300)
        plt.draw()
        plt.pause(curve_pause)
        plt.clf()

    if export_to_hdf:
        def metadata(method='pysolar_mowesta_cs'):
            path = 'data/PROGNOES/' + start_time.strftime("%m_%d_%Y__%H_%M_%S") + "/PROGNOES_" + method + '_' + str(
                start_time.strftime("%m_%d_%Y__%H_%M_%S")) + ".hdf5"

            if not os.path.isdir('data/PROGNOES/' + start_time.strftime("%m_%d_%Y__%H_%M_%S")):
                os.mkdir('data/PROGNOES/' + start_time.strftime("%m_%d_%Y__%H_%M_%S"))

            f = h5py.File(path, 'a')

            GFS_meta = GFS_day_minute(address=address, date=metadata_time, sys_exit=True)[1]
            mowesta_meta = mowesta_day_minute(address=address, date=metadata_time, sys_exit=True)[1]
            start_time2 = datetime.datetime.strptime(str(GFS_meta['timestamp'][0]), "%Y-%m-%d %H:%M:%S%z")

            def create_harvesting_conditions():
                temp = {"harvesting_conditions": {'Date': str(start_time2.date().strftime('%d.%m.%Y')),
                                                  'Start Time (Local Timezone)': str(
                                                      start_time2.time().strftime('%H:%M:%S.%f')),
                                                  'End Time (Local Timezone)': str(
                                                      start_time2.time().strftime('%H:%M:%S.%f')),
                                                  'Indoor/Outdoor': "indoor_outdoor",
                                                  'Light Intensity (Lux)': str(50),
                                                  'Weather Condition': "Not Available", 'Country': "Not Available",
                                                  'City': str(address.split(',')[1])}}

                dt = h5py.string_dtype(encoding='utf-8')
                dsetx = f.create_dataset("harvesting conditions", (2, 8), dtype=dt)
                dsetx[0] = list(temp["harvesting_conditions"].keys())
                dsetx[1] = list(temp["harvesting_conditions"].values())

            def create_prognoes_conditions():
                temp = {"prognoes_conditions": {'start_time': str(start_time2.timestamp()),
                                                'end_time': str(start_time2.timestamp()),
                                                'thermometer_temperature': str(mowesta_meta['temperature'][0]),
                                                'address': str(address),
                                                'latitude': str(latitude),
                                                'longitude': str(longitude), 'solar_cell': str(solar_cell_model)}}
                dt = h5py.string_dtype(encoding='utf-8')
                dsetx = f.create_dataset("prognoes_conditions", (2, 7), dtype=dt)
                dsetx[0] = list(temp["prognoes_conditions"].keys())
                dsetx[1] = list(temp["prognoes_conditions"].values())

            def create_weather_conditions():
                temp = {"weather_conditions": {"timestamp": str(start_time2.timestamp()),
                                               "temperature": str(mowesta_meta['temperature'][0]),
                                               "total_clouds": str(mowesta_meta['total_clouds'][0]),
                                               "precipitation": str(mowesta_meta['precipitation'][0]),
                                               "snowDepth": str(mowesta_meta['snowDepth'][0]),
                                               "windSpeed": str(mowesta_meta['windSpeed'][0]),
                                               "humidity": str(mowesta_meta['humidity'][0]),
                                               "pressure": str(mowesta_meta['pressure'][0]),
                                               "pysolar_cs_ghi": str(mowesta_meta['pysolar_cs_ghi'][0]),
                                               "IandP_cs_ghi": str(mowesta_meta['IandP_cs_ghi'][0]),
                                               "cs_ghi": str(mowesta_meta['cs_ghi'][0]),
                                               "cs_dni": str(mowesta_meta['cs_dni'][0]),
                                               "cs_dhi": str(mowesta_meta['cs_dhi'][0]),
                                               "cn_ghi": str(mowesta_meta['cn_ghi'][0]),
                                               "cn_dni": str(mowesta_meta['cn_dni'][0]),
                                               "cn_dhi": str(mowesta_meta['cn_dhi'][0])
                                               }}

                dt = h5py.string_dtype(encoding='utf-8')
                dsetx = f.create_dataset("weather_conditions", (2, 16), dtype=dt)
                dsetx[0] = list(temp["weather_conditions"].keys())
                dsetx[1] = list(temp["weather_conditions"].values())

            def create_weather_irradiation_conditions():
                temp = {
                    "irradiation_weather_conditions": {'timestamp': str(start_time2.strftime("%Y-%m-%d %H:%M:%S%z")),
                                                       'temp_air': str(GFS_meta['temp_air'][0]),
                                                       'wind_speed': str(GFS_meta['wind_speed'][0]),
                                                       'total_clouds': str(GFS_meta['total_clouds'][0]),
                                                       'low_clouds': str(GFS_meta['low_clouds'][0]),
                                                       'mid_clouds': str(GFS_meta['mid_clouds'][0]),
                                                       'high_clouds': str(GFS_meta['high_clouds'][0]),
                                                       'GFS_ghi': str(GFS_meta['GFS_ghi'][0]),
                                                       'cs_ghi': str(GFS_meta['cs_ghi'][0]),
                                                       'cs_dni': str(GFS_meta['cs_dni'][0]),
                                                       'cs_dhi': str(GFS_meta['cs_dhi'][0]),
                                                       'cn_ghi': str(GFS_meta['cn_ghi'][0]),
                                                       'cn_dni': str(GFS_meta['cn_dni'][0]),
                                                       'cn_dhi': str(GFS_meta['cn_dhi'][0])}}
                dt = h5py.string_dtype(encoding='utf-8')
                dsetx = f.create_dataset("irradiation_weather_conditions", (2, 14), dtype=dt)
                dsetx[0] = list(temp["irradiation_weather_conditions"].keys())
                dsetx[1] = list(temp["irradiation_weather_conditions"].values())

            create_prognoes_conditions()
            create_harvesting_conditions()
            create_weather_conditions()
            create_weather_irradiation_conditions()

        for key, value in methods.items():
            if value:
                metadata(key)

    def create_dataset_curves(path, name, v, i):
        f = h5py.File(path, 'a')
        dsetx = f.create_dataset('curve' + name, (2, 40))
        dsetx[0] = v
        dsetx[1] = i
        return "Curve Saved in HDF5"

    curve_number = 0
    start_time_counter = start_time

    while (start_time_counter <= end_time):
        result = collect_ghis(date=start_time_counter)
        if export_to_hdf:

            for key, value in methods.items():
                if value == True:
                    path = 'data/PROGNOES/' + start_time.strftime(
                        "%m_%d_%Y__%H_%M_%S") + "/PROGNOES_" + key + '_' + str(
                        start_time.strftime("%m_%d_%Y__%H_%M_%S")) + ".hdf5"
                    create_dataset_curves(path, str(curve_number), result[key][2]['v'],
                                          result[key][2]['i'])

        if plt_show:
            if original_dataset is not None:
                methods['original_curve_SOCRAETES'] = True
                if curve_number < len(original_dataset):
                    result.update(
                        {'original_curve_SOCRAETES': [0, 0, {'v': original_dataset['curve' + str(curve_number)]['v'],
                                                             'i': original_dataset['curve' + str(curve_number)]['i']}]})
                else:
                    result.update(
                        {'original_curve_SOCRAETES': [0, 0, {
                            'v': original_dataset['curve' + str(len(original_dataset) - 1)]['v'],
                            'i': original_dataset['curve' + str(len(original_dataset) - 1)][
                                'i']}]})

            plot_curve(result, start_time_counter)

        if verbose_show:
            print("##########  " + str(start_time_counter) + "  ##########")

        start_time_counter = start_time_counter + datetime.timedelta(seconds=seconds_intervals)

        curve_number = curve_number + 1

    return "Process finished successfully."


def compare_dataset(path='data/examples/SOCRAETES_SM141K08LV_211028_130753.hdf5'):
    """
    This function loads a HFD5 file recorded with SOCRAETES and calls dataset_to_hdf5 for
    comparing the real recorded curves with the predicted curves.
    :param path:
    Path of the HDF5 file.
    :return:
    Returns a message, that process was successful.
    """
    from general_functions import extract_z_from_hdf5
    import h5py
    import numpy as np

    prognoes_conditions = {'available': False}
    f = h5py.File(path, 'a')

    if ("harvesting conditions" not in f) or ("weather_conditions" not in f) or (
            "irradiation_weather_conditions" not in f):
        print("You are using an old format which is not compatible to PROGNOES, a copy of your file will be processed.")
        print("The comparing of such old file makes no sense, because many information is missing."
              "To plot the curves in this file use hdf5_reader function. Program will be exit.")
        sys.exit()

    if "prognoes_conditions" in f:
        prognoes_conditions['available'] = True
        for label, value in zip(np.array(f.get("prognoes_conditions")[0]),
                                np.array(f.get("prognoes_conditions")[1])):
            prognoes_conditions.update({label.decode('utf-8'): value.decode('utf-8')})

    df = extract_z_from_hdf5(path)
    df.columns = ["X", "Y", "Z", "Curve"]
    lst = df.values.tolist()

    number_of_curves = len(lst) / 40
    start = prognoes_conditions['start_time']
    end = prognoes_conditions['end_time']
    curves_intervals = (float(end) - float(start)) / number_of_curves

    lastcurve = 0

    v = []
    i = []
    curves = {}
    for item in lst:
        if str(item[3]).split('curve')[1] != lastcurve:
            lastcurve = str(item[3]).split('curve')[1]
            v = []
            i = []
            v.append(item[0])
            i.append(item[1])
            curves.update({"curve" + str(lastcurve): {'v': v, 'i': i}})

        else:
            v.append(item[0])
            i.append(item[1])
            curves.update({"curve" + str(lastcurve): {'v': v, 'i': i}})
    print("Start: ", datetime.datetime.fromtimestamp(float(start)))
    print("end: ", datetime.datetime.fromtimestamp(float(end)))
    dataset_to_hdf5(solar_cell_model=prognoes_conditions['solar_cell'],
                    start_time=datetime.datetime.fromtimestamp(float(start)),
                    end_time=datetime.datetime.fromtimestamp(float(end)),
                    metadata_time=datetime.datetime.fromtimestamp(float(start)),
                    seconds_intervals=curves_intervals, original_dataset=curves, verbose_show=True)

    return "Done"


def dataset_version_converter(path='data/examples/OLD_VERSION_WITH_DATASET_SOCRAETES_SM111K04L_211028_130511.hdf5',
                              solar_cell="SM141K08LV", thermometer_temperature=25, address="Kuglerstraße, Essen,45144",
                              latitude=51.455643, longitude=7.011555):
    """
    This function converts the old version of HDF5 File recorded with SOCRAETES to enable the user using some functions
    like diverse model of plotting or creating the datasets for the desired time manually.
    The source file will not be changed.
    :param path:
    The path of the HDF5 file that should be converted.
    :param solar_cell:
    The type of solar cell that used at that time as the old file created. Default is 'SM141K08LV'.
    :param thermometer_temperature:
    The measured temperature of solar cell if given. Default is '25'.
    :param address:
    The address of the place, which recording took place. Default is 'Kuglerstraße, Essen,45144'
    :param latitude:
    The latitude of the place, which recording took place. Default is '51.455643'
    :param longitude:
    The longitude of the place, which recording took place. Default is '7.011555'.
    :return:
    Returns the path of converted file.
    """
    import h5py
    import numpy as np
    import datetime
    from mowesta import mowesta_day_minute
    from pvlib_functions import GFS_day_minute
    import requests
    import urllib
    import sys
    import shutil
    from dateutil import tz

    print("An attempt is being made to update the file version.")

    try:
        if address != "Kuglerstraße, Essen,45144":
            url = 'https://nominatim.openstreetmap.org/search/' + urllib.parse.quote(address) + '?format=json'
            response = requests.get(url).json()
            latitude = round(float(response[0]["lat"]), 7)
            longitude = round(float(response[0]["lon"]), 7)
    except:
        print(
            "An error occurred while connecting to the openstreetmap.org. Please check your connection and try again.")
        sys.exit()

    from timezonefinder import TimezoneFinder
    tf = TimezoneFinder()
    timezone = tf.timezone_at(lng=longitude, lat=latitude)

    dict_temp = {}

    src = path
    dst = 'data/temporary_datasets/temp_' + path.split('/')[-1]
    shutil.copyfile(src, dst)
    path = dst
    f = h5py.File(path, 'a')

    harvesting_conditions = {'Date': "Not Available",
                             'Start Time (Local Timezone)': "Not Available",
                             'End Time (Local Timezone)': "Not Available",
                             'Indoor/Outdoor': "Not Available",
                             'Light Intensity (Lux)': "Not Available",
                             'Weather Condition': "Not Available", 'Country': "Not Available",
                             'City': "Not Available"}

    if "harvesting conditions" in f:
        for label, value in zip(np.array(f.get("harvesting conditions")[0]),
                                np.array(f.get("harvesting conditions")[1])):
            harvesting_conditions.update({label.decode('utf-8'): value.decode('utf-8')})

    prognoes_conditions = {"start_time": "Not Available",
                           "end_time": "Not Available",
                           "thermometer_temperature": "Not Available", "address": "Not Available",
                           "latitude": "Not Available", "longitude": "Not Available",
                           "solar_cell": "Not Available"}

    weather_conditions = {"timestamp": "Not Available", "temperature": "Not Available",
                          "total_clouds": "Not Available",
                          "precipitation": "Not Available", "snowDepth": "Not Available",
                          "windSpeed": "Not Available",
                          "humidity": "Not Available", "pressure": "Not Available", "pysolar_cs_ghi": "Not Available",
                          "IandP_cs_ghi": "Not Available", "cs_ghi": "Not Available",
                          "cs_dni": "Not Available", "cs_dhi": "Not Available", "cn_ghi": "Not Available",
                          "cn_dni": "Not Available", "cn_dhi": "Not Available"}

    irradiation_weather_conditions = {'timestamp': "Not Available", 'temp_air': "Not Available",
                                      'wind_speed': "Not Available",
                                      'total_clouds': "Not Available",
                                      'low_clouds': "Not Available", 'mid_clouds': "Not Available",
                                      'high_clouds': "Not Available", 'GFS_ghi': "Not Available",
                                      'cs_ghi': "Not Available",
                                      'cs_dni': "Not Available",
                                      'cs_dhi': "Not Available", 'cn_ghi': "Not Available",
                                      'cn_dni': "Not Available", 'cn_dhi': "Not Available"}

    start_date = harvesting_conditions['Date'] + ' ' + harvesting_conditions[
        'Start Time (Local Timezone)']
    start_date = datetime.datetime.strptime(start_date, "%d.%m.%Y %H:%M:%S.%f").timestamp()
    start_date_datetime = (datetime.datetime.fromtimestamp(start_date))
    start_date_datetime = datetime.datetime(year=start_date_datetime.year, month=start_date_datetime.month,
                                            day=start_date_datetime.day, hour=start_date_datetime.hour,
                                            minute=start_date_datetime.minute, second=start_date_datetime.second,
                                            tzinfo=tz.gettz(timezone))

    end_date = harvesting_conditions['Date'] + ' ' + harvesting_conditions[
        'End Time (Local Timezone)']
    end_date = datetime.datetime.strptime(end_date, "%d.%m.%Y %H:%M:%S.%f").timestamp()

    GFS_result_day, GFS_result_minute, GFS_result_minute_dict = GFS_day_minute(address=address,
                                                                               date=datetime.datetime.fromtimestamp(
                                                                                   start_date), sys_exit=False)

    mowesta_result_day, mowesta_result_minute, mowesta_result_minute_dict = mowesta_day_minute(address=address,
                                                                                               date=datetime.datetime.fromtimestamp(
                                                                                                   start_date),
                                                                                               sys_exit=False)

    if "weather_conditions" not in f:
        if mowesta_result_minute_dict is not None:
            print("MoWeSta Dataset for the given address and datetime is available.")
            weather_conditions = {"timestamp": str(mowesta_result_minute_dict['timestamp']),
                                  "temperature": str(mowesta_result_minute_dict['temperature']),
                                  "total_clouds": str(mowesta_result_minute_dict['total_clouds']),
                                  "precipitation": str(mowesta_result_minute_dict['precipitation']),
                                  "snowDepth": str(mowesta_result_minute_dict['snowDepth']),
                                  "windSpeed": str(mowesta_result_minute_dict['windSpeed']),
                                  "humidity": str(mowesta_result_minute_dict['humidity']),
                                  "pressure": str(mowesta_result_minute_dict['pressure']),
                                  "pysolar_cs_ghi": str(mowesta_result_minute_dict['pysolar_cs_ghi']),
                                  "IandP_cs_ghi": str(mowesta_result_minute_dict['IandP_cs_ghi']),
                                  "cs_ghi": str(mowesta_result_minute_dict['cs_ghi']),
                                  "cs_dni": str(mowesta_result_minute_dict['cs_dni']),
                                  "cs_dhi": str(mowesta_result_minute_dict['cs_dhi']),
                                  "cn_ghi": str(mowesta_result_minute_dict['cn_ghi']),
                                  "cn_dni": str(mowesta_result_minute_dict['cn_dni']),
                                  "cn_dhi": str(mowesta_result_minute_dict['cn_dhi'])}
        else:
            print("MoWeSta Dataset for the given address and datetime is not available.")
            weather_conditions = {"timestamp": str(start_date), "temperature": str(25),
                                  "total_clouds": str(0),
                                  "precipitation": str(0), "snowDepth": str(0),
                                  "windSpeed": str(0),
                                  "humidity": str(0), "pressure": str(0), "pysolar_cs_ghi": str(0),
                                  "IandP_cs_ghi": str(0), "cs_ghi": str(0),
                                  "cs_dni": str(0), "cs_dhi": str(0), "cn_ghi": str(0),
                                  "cn_dni": str(0), "cn_dhi": str(0)}

    if "irradiation_weather_conditions" not in f:
        if GFS_result_minute_dict is not None:
            print("GFS Dataset for the given address and datetime is available.")
            irradiation_weather_conditions = {'timestamp': str(GFS_result_minute_dict['timestamp']),
                                              'temp_air': str(GFS_result_minute_dict['temp_air']),
                                              'wind_speed': str(GFS_result_minute_dict['wind_speed']),
                                              'total_clouds': str(GFS_result_minute_dict['total_clouds']),
                                              'low_clouds': str(GFS_result_minute_dict['low_clouds']),
                                              'mid_clouds': str(GFS_result_minute_dict['mid_clouds']),
                                              'high_clouds': str(GFS_result_minute_dict['high_clouds']),
                                              'GFS_ghi': str(GFS_result_minute_dict['GFS_ghi']),
                                              'cs_ghi': str(GFS_result_minute_dict['cs_ghi']),
                                              'cs_dni': str(GFS_result_minute_dict['cs_dni']),
                                              'cs_dhi': str(GFS_result_minute_dict['cs_dhi']),
                                              'cn_ghi': str(GFS_result_minute_dict['cn_ghi']),
                                              'cn_dni': str(GFS_result_minute_dict['cn_dni']),
                                              'cn_dhi': str(GFS_result_minute_dict['cn_dhi'])}

        else:
            print("GFS Dataset for the given address and datetime is not available.")
            print(start_date_datetime)
            irradiation_weather_conditions = {'timestamp': start_date_datetime.strftime("%Y-%m-%d %H:%M:%S%z"),
                                              'temp_air': str(25),
                                              'wind_speed': str(0), 'total_clouds': str(0),
                                              'low_clouds': str(0), 'mid_clouds': str(0),
                                              'high_clouds': str(0), 'GFS_ghi': str(0),
                                              'cs_ghi': str(0), 'cs_dni': str(0), 'cs_dhi': str(0),
                                              'cn_ghi': str(0), 'cn_dni': str(0), 'cn_dhi': str(0)}

    if "prognoes_conditions" not in f:
        prognoes_conditions = {"start_time": str(start_date),
                               "end_time": str(end_date),
                               "thermometer_temperature": str(thermometer_temperature), "address": address,
                               "latitude": str(latitude), "longitude": str(longitude),
                               "solar_cell": solar_cell}

    dict_temp.update({"harvesting conditions": harvesting_conditions})
    dict_temp.update({"prognoes_conditions": prognoes_conditions})
    dict_temp.update({"weather_conditions": weather_conditions})
    dict_temp.update({"irradiation_weather_conditions": irradiation_weather_conditions})

    def change_harvesting_conditions():
        f = h5py.File(path, 'a')
        if 'harvesting conditions' in f:
            del f['harvesting conditions']
        dt = h5py.string_dtype(encoding='utf-8')
        dsetx = f.create_dataset("harvesting conditions", (2, 8), dtype=dt)
        dsetx[0] = list(harvesting_conditions.keys())
        dsetx[1] = list(harvesting_conditions.values())

    def change_weather_conditions():
        f = h5py.File(path, 'a')
        if 'weather_conditions' in f:
            del f['weather_conditions']
        if 'mowesta_conditions' in f:
            del f['mowesta_conditions']
        dt = h5py.string_dtype(encoding='utf-8')
        dsetx = f.create_dataset("weather_conditions", (2, 16), dtype=dt)
        dsetx[0] = list(weather_conditions.keys())
        dsetx[1] = list(weather_conditions.values())

    def change_irradiation_weather_conditions():
        f = h5py.File(path, 'a')
        if 'irradiation_weather_conditions' in f:
            del f['irradiation_weather_conditions']
        dt = h5py.string_dtype(encoding='utf-8')
        dsetx = f.create_dataset("irradiation_weather_conditions", (2, 14), dtype=dt)
        dsetx[0] = list(irradiation_weather_conditions.keys())
        dsetx[1] = list(irradiation_weather_conditions.values())

    def change_prognoes_conditions():
        f = h5py.File(path, 'a')
        if 'prognoes_conditions' in f:
            del f['prognoes_conditions']
        dt = h5py.string_dtype(encoding='utf-8')
        dsetx = f.create_dataset("prognoes_conditions", (2, 7), dtype=dt)
        dsetx[0] = list(prognoes_conditions.keys())
        dsetx[1] = list(prognoes_conditions.values())

    change_prognoes_conditions()
    change_harvesting_conditions()
    change_weather_conditions()
    change_irradiation_weather_conditions()
    print("Converting to new version finished successfully.")
    return dst


def convert_for_emulator(path='data/examples/SOCRAETES_SM141K08LV_211028_130753.hdf5'):
    '''
    This function get a HDF5 file created with PROGNOES and removes datasets which is not available in SOCRAETES.
    :param path:
    The Path of the file
    :return:
    A Message about successful conversion.
    '''
    import h5py
    import shutil

    src = path
    dst = 'data/temporary_datasets/emulator_' + path.split('/')[-1]
    shutil.copyfile(src, dst)
    path = dst
    f = h5py.File(path, 'a')

    del f['prognoes_conditions']
    del f['weather_conditions']
    del f['irradiation_weather_conditions']
    return "The file is converted and stored in temporary_datasets"
