import datetime
import warnings

warnings.filterwarnings("ignore", module='pvlib')
warnings.filterwarnings("ignore", module='pysolar')


def IandP_day_minute(latitude_deg=51.455643, longitude_deg=7.011555, altitude=0, date=datetime.datetime.now()):
    """
    This function calculates GHI,DNI and DHI using the Ineichen and Prez method for the given parameters.
    :param latitude_deg:
    The latitude of desired location. Default is '51.455643'.
    :param longitude_deg:
    The longitude of desired location. Default is '7.011555'.
    :param altitude:
    The longitude of desired location. Default is 0.
    :param date:
    The datetime as the calculation starts.
    :return:
    Returns irradiation in a dataframe for the whole day, a dataframe for the queried minute and a dict for the
    queried time. The columns are : ['ghi', 'dni', 'dhi']
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    from pvlib.location import Location
    from dateutil import tz

    from timezonefinder import TimezoneFinder
    tf = TimezoneFinder()
    timezone = tf.timezone_at(lng=longitude_deg, lat=latitude_deg)

    plt_show = False

    location = Location(latitude_deg, longitude_deg, timezone, altitude, timezone)
    start = str(date.year) + '-' + str(date.month) + '-' + str(date.day)
    end_tt = date + datetime.timedelta(days=1)
    end = str(end_tt.year) + '-' + str(end_tt.month) + '-' + str(end_tt.day)
    times = pd.date_range(start=start, end=end, freq='1min', tz=location.tz)
    result_day = location.get_clearsky(times)
    ghi = result_day['ghi']
    dni = result_day['dni']
    dhi = result_day['dhi']

    if plt_show:
        result_day.plot()
        plt.ylabel('Irradiance $W/m^2$')
        plt.title("Generated maximum power points for the desired location.", fontsize=9, y=0.99)
        plt.legend(loc='best', fancybox=True, shadow=True)
        plt.suptitle(
            "The following plot shows radiation for the latitude:{} and longitude {} in {} timezone. \n "
            "Max GHI: {}, DNI: {}, DHI: {}. \n "
            "The query contains values from for {}.{}.{} ".format(
                latitude_deg, longitude_deg, timezone, max(ghi), max(dni), max(dhi), date.year, date.month, date.day),
            fontsize=7, y=0.99)
        plt.show()

    start_loc = datetime.datetime(year=date.year, month=date.month, day=date.day, hour=date.hour, minute=date.minute,
                                  second=0, tzinfo=tz.gettz(timezone))

    filtered_df = result_day.loc[start_loc:start_loc + datetime.timedelta(minutes=1)]
    result_minute = filtered_df.first('1s')
    result_minute = result_minute.reset_index()

    result_minute_dict = {}
    for key in list(result_minute.columns):
        result_minute_dict.update({key: result_minute[key][0]})

    return result_day, result_minute, result_minute_dict


def GFS_day_minute(address="Kuglerstraße, Essen,45144", latitude=51.455643, longitude=7.011555,
                   date=datetime.datetime.now(), sys_exit=False):
    """
    This function create a dataset for the given day and saves it as csv data. During the processing the GHI, DNI, DHI
    considering the the impact of clouds will be calculated with campbell norman and clearsky methods.
    The Dataset will be extrapolated using linear methods.
    At this time MoWeSta provides Forecast upto 7 days, so the GFS will be limited to 7 days too.
    param address:
    The address of desired location. Default is 'Kuglerstraße, Essen, 45144'.
    :param latitude:
    The latitude of desired location. Default is '51.455643'.
    :param longitude:
    The longitude of desired location. Default is '7.011555'.
    :param date:
    The time at which the process starts. Default is running time.
    :param sys_exit:
    The function dataset_version_converter should continue, even if the dataset are not available.
    :return:
    Returns the whole day dataset as dataframe, the minute corresponds the nearest time to given date as dataframe, and
    the minute corresponds the nearest time to given date as dict.
    The columns are: [timestamp, temp_air, wind_speed, total_clouds, low_clouds, mid_clouds, high_clouds, GFS_ghi,
                    cs_ghi, cs_dni, cs_dhi, cn_ghi, cn_dni, cn_dhi]
    """
    import pandas as pd
    import numpy as np
    from dateutil import tz
    import os
    import requests
    import urllib
    import time
    import sys

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

    plt_show = False
    dataset_availablity = False

    time_now = datetime.date(year=date.year, month=date.month, day=date.day)
    start = pd.Timestamp(time_now, tz=timezone)
    end = start + pd.Timedelta(days=1)

    str_name = "PROGNOES_GFS_" + str(int(address.split(",")[2])) + "_" + str(start.year)[2:] + (
        '{:02d}'.format(start.month)) + '{:02d}'.format(start.day) + ".csv"

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(ROOT_DIR, 'data/GFS_archive/' + str_name)

    def download_dataset_from_GFS():
        from pvlib.forecast import GFS

        def df_interpolate(df, date):

            cl_name = ['temp_air', 'wind_speed_u', 'wind_speed_v', 'wind_speed_gust',
                       'total_clouds', 'high_clouds', 'mid_clouds', 'low_clouds', 'boundary_clouds', 'convect_clouds',
                       'ghi_raw']

            df_n = pd.DataFrame(columns=cl_name)
            for column in cl_name:
                df_n[column] = df[column].values

            df = df_n

            start = datetime.datetime(year=date.year, month=date.month, day=date.day, hour=0, minute=0,
                                      second=0)
            end = datetime.datetime(year=date.year, month=date.month, day=date.day, hour=0, minute=0,
                                    second=0) + datetime.timedelta(days=1)

            date_time = []
            start_t = datetime.datetime(year=start.year, month=start.month, day=start.day, hour=0, minute=0, second=0,
                                        tzinfo=tz.gettz(timezone))

            for i in range((end - start).days * 24 * 60):
                date_time.append(start_t)
                start_t = start_t + datetime.timedelta(minutes=1)

            df1 = pd.DataFrame(date_time)
            df1.columns = ['timestamp']
            df_cols = list(df.columns)

            for i in df_cols:
                df1[i] = np.nan

            df1.set_index('timestamp', inplace=True)
            df2 = df.values.tolist()
            time_temp = start

            for i in range(len(df2)):
                df2[i].insert(0, time_temp)
                time_temp = time_temp + datetime.timedelta(hours=3)

            df = df2

            for item in range(len(df)):
                date = df[item][0]
                start_loc = datetime.datetime(year=date.year, month=date.month, day=date.day, hour=date.hour,
                                              minute=date.minute,
                                              second=0, tzinfo=tz.gettz(timezone))

                df1.loc[start_loc:start_loc, 'temp_air'] = df[item][1]
                df1.loc[start_loc:start_loc, 'wind_speed_u'] = df[item][2]
                df1.loc[start_loc:start_loc, 'wind_speed_v'] = df[item][3]
                df1.loc[start_loc:start_loc, 'wind_speed_gust'] = df[item][4]
                df1.loc[start_loc:start_loc, 'total_clouds'] = df[item][5]
                df1.loc[start_loc:start_loc, 'high_clouds'] = df[item][6]
                df1.loc[start_loc:start_loc, 'mid_clouds'] = df[item][7]
                df1.loc[start_loc:start_loc, 'low_clouds'] = df[item][8]
                df1.loc[start_loc:start_loc, 'boundary_clouds'] = df[item][9]
                df1.loc[start_loc:start_loc, 'convect_clouds'] = df[item][10]
                df1.loc[start_loc:start_loc, 'ghi_raw'] = df[item][11]

            df1 = df1.interpolate(method='linear', limit_direction='forward', axis=0)
            return df1

        model = GFS(resolution='Quarter')
        raw_data = model.get_data(latitude, longitude, start, end)

        data = raw_data
        data = model.rename(data)

        data = df_interpolate(data, date)

        ghi_raw_col = pd.DataFrame(data['ghi_raw'])
        ghi_raw_col.columns = ["GFS_ghi"]
        result_day = model.process_data(data)

        del result_day['ghi']
        del result_day['dhi']
        del result_day['dni']

        result_day = result_day.join(ghi_raw_col, how="outer")

        irrads = model.cloud_cover_to_irradiance(data['total_clouds'], how='clearsky_scaling')
        irrads.columns = ["cs_ghi", "cs_dni", "cs_dhi"]
        result_day = result_day.join(irrads, how='outer')

        irrads = model.cloud_cover_to_irradiance(data['total_clouds'], how='campbell_norman')
        irrads.columns = ["cn_ghi", "cn_dni", "cn_dhi"]
        result_day = result_day.join(irrads, how='outer')

        result_day.to_csv(file_path)

        if plt_show:
            datasets_from_csv(file_path, time)

        return result_day

    if (os.path.exists(file_path)):
        print("GFS dataset exists already.")
        dataset_availablity = True
        if (time.time() - os.path.getmtime(file_path) < 86400) and (
                time.time() - os.path.getmtime(file_path) > 3600) and (time.time() - start.timestamp() < 86400) and (
                time.time() - end.timestamp() < 86400):
            print("Dataset is older than 3600 secs. A new one could be available, try download last recent dataset ")
            download_dataset_from_GFS()

        result_day = pd.read_csv(file_path, parse_dates=True,
                                 index_col='timestamp')
        if plt_show:
            datasets_from_csv(file_path, date, latitude, longitude, timezone)
    else:
        if abs((datetime.datetime.now() - date).days) < 8:
            print("Try to download GFS dataset.")
            dataset_availablity = True
            result_day = download_dataset_from_GFS()
        else:
            print("The requested GFS Dataset for the date is neither online or offline available.")
            if sys_exit:
                sys.exit()

    if dataset_availablity:
        start_loc = datetime.datetime(year=date.year, month=date.month, day=date.day, hour=date.hour,
                                      minute=date.minute,
                                      second=0, tzinfo=tz.gettz(timezone))
        filtered_df = result_day.loc[start_loc:start_loc + datetime.timedelta(minutes=1)]
        result_minute = filtered_df.head(1)
        result_minute = result_minute.reset_index()
        result_minute_dict = {}
        for key in list(result_minute.columns):
            result_minute_dict.update({key: result_minute[key][0]})
    else:
        result_day, result_minute, result_minute_dict = None, None, None

    return result_day, result_minute, result_minute_dict


def datasets_from_csv(path_GFS, path_mowesta, date=datetime.datetime.utcnow(), latitude=51.455643, longitude=7.011555,
                      timezone="Europe/Berlin"):
    """
    This function plots the GFS wetter information for the gve location. This information covers the whole day.
    The plots contains cloud layers and GHI, DNI, DHI values using Campbell Norman and Clearsky Method.
    :param path:
    Path of the GFS_Dataset.
    :param date:
    The datetime for the desired GFS weather information.
    :param latitude:
    The latitude of the given address. Default is: '51.455643'
    Antarctica has no address.
    :param longitude:
    the longitude of the given address. Default is: '7.011555'
    :param timezone:
    The timezone of the given location. Default is 'Europe/Berlin'.
    :return:
    Return a message that process ended successfully.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import datetime
    import numpy as np

    result_day_GFS = pd.read_csv(path_GFS, parse_dates=True, index_col='timestamp')
    result_day_mowesta = pd.read_csv(path_mowesta, parse_dates=True, index_col='timestamp')
    date = datetime.datetime(year=date.year, month=date.month, day=date.day)
    plt_show = True
    data_GFS = result_day_GFS
    data_mowesta = result_day_mowesta

    if plt_show:
        data_GFS.rename(
            columns={'total_clouds': 'GFS_Total_Clouds', 'low_clouds': 'GFS_Low_Clouds', 'mid_clouds': 'GFS_Mid_Clouds',
                     'high_clouds': 'GFS_High_Clouds'}, inplace=True)

        data_mowesta.rename(columns={'total_clouds': 'MoWeSta_Total_Clouds'}, inplace=True)
        moewsta_clouds = data_mowesta['MoWeSta_Total_Clouds'].values
        data_GFS['MoWeSta_Total_Clouds'] = np.nan
        i = 0
        for index, row in data_GFS.iterrows():
            row['MoWeSta_Total_Clouds'] = moewsta_clouds[i]
            i = i + 1

        cloud_vars = ['GFS_Total_Clouds', 'GFS_Mid_Clouds', 'GFS_Low_Clouds', 'MoWeSta_Total_Clouds']
        # data_GFS.to_csv('new.csv')
        data_GFS[cloud_vars].plot()
        # plt.tight_layout()
        plt.ylabel('Cloud cover %')
        plt.xticks([])
        plt.xlabel('GFS Forecast Time ({})'.format(timezone))
        plt.title('GFS and MoWeSta Cloud Depth Forecast', fontsize=9, y=0.99)
        plt.suptitle("GFS and MoWeSta Forecast for Cloud Depth and Different Types of Clouds.\n "
                     "For the Desired Latitude: {} and Longitude: {}. on {}.{}.{}"
                     "\n Dataset Sources are https://thredds.ucar.edu and https://www.mowesta.com/api/"
                     .format(latitude, longitude, date.day, date.month, date.year), fontsize=7, y=0.99)
        plt.legend(loc=(1.001, 0.535), fancybox=True, shadow=True)
        plt.savefig("clouds.png", bbox_inches="tight", pad_inches=1, transparent=True, facecolor="w", edgecolor='w',
                    orientation='landscape', dpi=300)
        plt.show()

    if plt_show:
        data_GFS.rename(
            columns={'cs_ghi': 'CS_GHI', 'cs_dni': 'CS_DNI', 'cs_dhi': 'CS_DHI', 'cn_ghi': 'CN_GHI', 'cn_dni': 'CN_DNI',
                     'cn_dhi': 'CN_DHI'}, inplace=True)
        cloud_vars = ['CS_GHI', 'CS_DNI', 'CS_DHI', 'CN_GHI', 'CN_DNI', 'CN_DHI']
        data_GFS[cloud_vars].plot(cmap='Paired')
        plt.ylabel('Irradiance ($W/m^2$)')
        plt.xticks([])
        plt.xlabel('GFS Forecast Time ({})'.format(timezone))
        plt.title('GFS Forecast With Campbell Norman and Clearsky Scaling for Available Irradiance', fontsize=8, y=0.99)
        plt.suptitle(
            "Calculating the Retrieved Irradiance with Clearsky Scaling (CS) and Cambell Norman (CN) Method.\n "
            "For the Queried Coordination with the Latitude: {} and Longitude: {}. on {}.{}.{}"
            "\n GFS Dataset Retrieved from https://thredds.ucar.edu"
                .format(latitude, longitude, date.day, date.month, date.year), fontsize=7, y=0.99)
        plt.legend()
        plt.savefig("gfs_irrads.png", bbox_inches="tight", pad_inches=1, transparent=True, facecolor="w", edgecolor='w',
                    orientation='landscape', dpi=300)
        plt.show()

    if plt_show:
        data_mowesta.rename(
            columns={'pysolar_cs_ghi': 'Pysolar_CS_GHI', 'IandP_cs_ghi': 'IandP_CS_GHI', 'cs_ghi': 'CS_GHI',
                     'cs_dni': 'CS_DNI', 'cs_dhi': 'CS_DHI', 'cn_ghi': 'CN_GHI', 'cn_dni': 'CN_DNI',
                     'cn_dhi': 'CN_DHI'}, inplace=True)
        cloud_vars = ['Pysolar_CS_GHI', 'IandP_CS_GHI', 'CS_GHI', 'CS_DNI', 'CS_DHI', 'CN_GHI', 'CN_DNI', 'CN_DHI']
        data_mowesta[cloud_vars].plot(cmap='Dark2')
        plt.ylabel('Irradiance ($W/m^2$)')
        plt.xticks([])
        plt.xlabel('MoWeSta Forecast Time ({})'.format(timezone))
        plt.title('MoWeSta Forecast With Campbell Norman and Clearsky Scaling for Available Irradiance', fontsize=8,
                  y=0.99)
        plt.suptitle(
            "Calculating the Retrieved Irradiance with Clearsky Scaling (CS) and Cambell Norman (CN) Method.\n "
            "For the Queried Coordination with the Latitude: {} and Longitude: {}. on {}.{}.{}"
            "\n GFS Dataset Retrieved from https://thredds.ucar.edu"
                .format(latitude, longitude, date.day, date.month, date.year), fontsize=7, y=0.99)
        plt.legend()
        plt.savefig("mowesta_irrads.png", bbox_inches="tight", pad_inches=1, transparent=True, facecolor="w",
                    edgecolor='w',
                    orientation='landscape', dpi=300)
        plt.show()

    return "MoWeSta plotted successfully."


def PMP_calculator(prognoes_conditions, solar_cell_model='SM141K08LV', address="Kuglerstraße, Essen, 45144",
                   timestamp=datetime.datetime.now(), irradiation='IandP', cloud_source="mowesta",
                   prediction_method="cs"):
    """
    This function calculates the irradiation to maximum power point for the given solar cell at given time.
    It calculates the impact of temperature coefficient of power point too.
    The GHI of this calculation comes from IandP, the cloud data comes from MoWeSta,
    the methods for calculation of the impact of clouds of irradiation is done by Campbell Norman and ClearSky methods.
    :param prognoes_conditions:
    A dict that contains metadata. See more PROGNOES.Definition().
    :param solar_cell_model:
    The model of the solar cell. Default value is 'SM141K08LV'.
    :param address:
    The address of the location for which pysolar should calculate the irradiation.
    Default is 'Kuglerstraße, Essen,45144'.
    :param timestamp:
    The time, for that the PMP should calculated. Default is running time.
    :param timedelta:
    The time, that calculation of PMP should continue after starting the process. Default is one second.
    :param cloud_source:
    It can be 'GFS' for GFS as cloud data source or
    It can be 'mowesta' for mowesta as cloud data source
    :param prediction_method:
    It can be 'cn' for Campbell-Norman or
    It can be 'cs' for ClearSky.
    :return:
    Returns the first item of the list containing the irradiation for one sun, and calculated temperature coefficient.
    If timedelta is longer than one second the last item would be returned.
    """

    from solar_cell_class import solar_cell
    from mowesta import mowesta_day_minute

    solar_cell = solar_cell()
    solar_cell.SC_obj_from_csv('data/solar_cell_profiles/Calculated_' + solar_cell_model + '.csv')

    start = datetime.datetime(year=timestamp.year, month=timestamp.month, day=timestamp.day, hour=timestamp.hour,
                              second=timestamp.second)

    if irradiation == 'IandP':

        if cloud_source == "mowesta":
            result_day, result_minute, result_minute_dict = mowesta_day_minute(address=address, date=timestamp)
        elif cloud_source == "GFS":
            result_day, result_minute, result_minute_dict = GFS_day_minute(address=address, date=start)

        if result_minute_dict is not None:
            if prediction_method == "cn":
                radiation_value = result_minute_dict['cn_ghi'] * 1000  # * 1000 to get values in milli watt
            elif prediction_method == "cs":
                radiation_value = result_minute_dict['cs_ghi'] * 1000  # * 1000 to get values in milli watt
    elif irradiation == 'GFS':
        result_day, result_minute, result_minute_dict = GFS_day_minute(address=address, date=start)
        radiation_value = result_minute_dict['GFS_ghi'] * 1000  # * 1000 to get values in milli watt
    elif irradiation == 'pysolar':
        result_day, result_minute, result_minute_dict = mowesta_day_minute(address=address, date=timestamp)
        radiation_value = result_minute_dict['pysolar_cs_ghi'] * 1000  # * 1000 to get values in milli watt
    else:
        radiation_value = 0 * 1000  # * 1000 to get values in milli watt

    watt_one_square_meter = round(radiation_value, 2)
    watt_custom_square_meter = float(
        solar_cell.curve([(round(radiation_value / 1000, 2), float(prognoes_conditions['thermometer_temperature']))])[
            'p_mp'])

    print(irradiation + '_' + cloud_source + '_' + prediction_method +
          " -> PMP of Curve : W/M² = {}mW , {} with temperatur coefficient = {}mW  ".format(
              watt_one_square_meter, solar_cell_model, round(watt_custom_square_meter, 2)))

    return watt_one_square_meter, watt_custom_square_meter
