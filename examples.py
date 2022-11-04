'''
This file contains ready-to-use examples with recorded datasets.
'''

import math
import pandas as pd
import datetime
import sys


def pysolar_query(latitude=51.455643, longitude=7.011555, date_time=1, timezone="Europe/Berlin"):
    """
    This function returns the irradiation for the given time using the parameters.
    :param latitude:
    The latitude of the given address. Default is: '51.455643'
    :param longitude:
    the longitude of the given address. Default is: '7.011555'
    :param date_time:
    The time that Pysolar should start calculate the GHI. Default value is running time.
    :param timezone:
    The timezone of the given location. Default is 'Europe/Berlin'.
    :return:
    Returns the radiation value for one sun in W/M² for the given time
    """

    import datetime

    from pysolar.solar import get_altitude
    from pysolar import radiation

    import warnings

    warnings.filterwarnings("ignore", module='pvlib')
    warnings.filterwarnings("ignore", module='pysolar')
    import pytz
    import pysolar
    if date_time == 1:
        date_time = datetime.datetime.now()

    def timezone_converter(date_time, source_timezone, destination_timezone):
        source_timezone = pytz.timezone(source_timezone)
        destination_timezone = pytz.timezone(destination_timezone)
        date_time = source_timezone.localize(date_time)
        date_time = date_time.astimezone(destination_timezone)

        return date_time

    date_time = date_time + datetime.timedelta(hours=-1)
    date_time = datetime.datetime(year=date_time.year, month=date_time.month, day=date_time.day, hour=date_time.hour,
                                  minute=date_time.minute, second=date_time.second)
    date_time = date_time + datetime.timedelta(hours=1)
    date_time = timezone_converter(date_time, timezone, "UTC")
    altitude_deg = get_altitude(round(latitude, 7), round(longitude, 7), date_time)
    dni = radiation.get_radiation_direct(date_time, altitude_deg)  # * 1000 to get values in milli
    dhi = pysolar.util.diffuse_underclear(latitude_deg=latitude, longitude_deg=longitude, when=date_time)
    from pvlib import solarposition
    press = 100 * ((44331.514 - altitude_deg) / 11880.516) ** (1 / 0.1902632)
    solarposition = solarposition.get_solarposition(latitude=latitude, longitude=longitude, time=date_time,
                                                    altitude=altitude_deg,
                                                    pressure=press)
    import numpy as np
    ghi = (dni * np.cos(math.radians((solarposition['apparent_zenith'][0]))) + dhi)
    return ghi, dni, dhi


# print(pysolar_query(date_time=datetime.datetime(year=date.year,month=date.month,day=date.day,hour=date.hour+1,minute=date.minute,second=date.second)))

def read_MoWeSta_CSV(address='Kuglerstraße, Essen, 45144', latitude=51.455643, longitude=7.011555,
                     date=datetime.datetime.now(), sys_exit=False,
                     path="data/mowesta_archive/PROGNOES_MOWESTA_45144_220818.csv"):
    """
    This function create a dataset for the given day and saves it as csv data. During the processing the GHI, DNI, DHI
    of diverse soruces namely Pysolar and IandP will be queried and the the impact of clouds will be calculated with
    campbell norman and clearsky methods.The Dataset will be extrapolated using linear methods.
    At this time MoWeSta provides Forecast upto 7 days.
    :param address:
    The address of desired location. Default is 'Kuglerstraße, Essen, 45144'.
    :param latitude:
    The latitude of desired location. Default is '51.455643'.
    :param longitude:
    The longitude of desired location. Default is '7.011555'.
    :param date:
    The time at which the process starts. Default is running time.
    :param sys_exit:
    The function dataset_version_converter should continue, even if the dataset are not available.
    :param path
    the path of existing mowesta_dataset
    :return:
    Returns the whole day dataset as dataframe, the minute corresponds the nearest time to given date as dataframe, and
    the minute corresponds the nearest time to given date as dict.
    The columns are: ['timestamp', 'temperature', 'cloudCoverage', 'cloudDepth', 'precipitation', 'snowDepth',
                    'humidity', 'windSpeed', 'pressure', 'wind_speed_u', 'wind_speed_v', 'pysolar_cs_ghi', 'IandP_cs_ghi']
    """
    import requests
    import urllib
    import json
    import datetime
    import os
    from dateutil import tz
    import pandas as pd
    import pvlib.location
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

    try:
        resp = requests.get(
            'https://www.mowesta.com/api/forecasts/resolve?longitude=' + str(longitude) + '&latitude=' + str(
                latitude))
        response = resp.json()
    except:
        print("An error occurred while connecting to the mowesta.com. Please check your connection and try again.")
        if sys_exit:
            sys.exit()

    json.dumps(response, indent=True)

    str_name = "PROGNOES_MOWESTA_" + str(int(address.split(",")[2])) + "_" + str(date.year)[2:] + (
        '{:02d}'.format(date.month)) + '{:02d}'.format(date.day) + ".csv"
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(ROOT_DIR, 'data/temporary_datasets/mowesta_archive/' + str_name)

    from timezonefinder import TimezoneFinder
    tf = TimezoneFinder()
    timezone = tf.timezone_at(lng=float(longitude), lat=float(latitude))

    dataset_availablity = True

    def clearsky_scaling(cloud_cover, ghi_clear, offset=35):
        """
            for more information see pvlib.forecast.py : cloud_cover_to_ghi_linear
        """

        offset = offset / 100.
        cloud_cover = cloud_cover / 100.
        ghi = (offset + (1 - offset) * (1 - cloud_cover)) * ghi_clear
        return ghi

    def calculate_irradiations():

        df = pd.read_csv(path, parse_dates=True)
        start_date = df.iloc[0]['timestamp']
        start = datetime.datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S%z")
        end_date = (df.tail(1)).iloc[0]['timestamp']
        df = pd.read_csv(path, parse_dates=True, index_col='timestamp')

        del df['pysolar_cs_ghi']
        del df['IandP_cs_ghi']
        del df['cs_ghi']
        del df['cs_dni']
        del df['cs_dhi']
        del df['cn_ghi']
        del df['cn_dni']
        del df['cn_dhi']

        from pvlib.forecast import GFS
        model = GFS(resolution='Quarter')
        data = model.rename(df)
        latitude = 51.455643
        longitude = 7.011555
        model.location = pvlib.location.Location(latitude, longitude, timezone)
        print("GFS model created.")
        data['zero_clouds'] = data.loc[:, 'total_clouds']
        data['zero_clouds'] = data['zero_clouds'].fillna(0)
        irrad_data = model.cloud_cover_to_irradiance(data['total_clouds'], how='clearsky_scaling')
        irrad_data.columns = ["Iandp_MoWeSta_cs_ghi", "Iandp_MoWeSta_cs_dni", "Iandp_MoWeSta_cs_dhi"]
        data = data.join(irrad_data, how='outer')
        print("Irradiance with clearsky_scaling model calculated.")

        irrad_data = model.cloud_cover_to_irradiance(data['total_clouds'], how='campbell_norman')
        irrad_data.columns = ["CambellNorman_MoWeSta_cn_ghi", "CambellNorman_MoWeSta_cn_dni",
                              "CambellNorman_MoWeSta_cn_dhi"]
        data = data.join(irrad_data, how='outer')
        print("Irradiance with campbell_norman model calculated.")

        start_time = (datetime.datetime(year=start.year, month=start.month, day=start.day))
        end_time = ((datetime.datetime(year=start.year, month=start.month, day=start.day)) + datetime.timedelta(
            days=1)) + datetime.timedelta(minutes=-1)

        tz, altitude, name = 'US/Arizona', 0, 'Tucson'
        times = pd.date_range(start=start_time, end=end_time, freq='1Min', tz="Europe/Berlin")
        solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude)
        apparent_zenith = solpos['apparent_zenith']
        airmass = pvlib.atmosphere.get_relative_airmass(apparent_zenith)
        pressure = pvlib.atmosphere.alt2pres(altitude)
        airmass = pvlib.atmosphere.get_absolute_airmass(airmass, pressure)
        linke_turbidity = pvlib.clearsky.lookup_linke_turbidity(times, latitude, longitude)
        dni_extra = pvlib.irradiance.get_extra_radiation(times)

        ineichen = pvlib.clearsky.ineichen(apparent_zenith, airmass, linke_turbidity, altitude, dni_extra)
        ineichen = ineichen.rename(columns={"ghi": "IandP_ghi", "dni": "IandP_dni", "dhi": "IandP_dhi"})
        ineichen = ineichen.reset_index()

        result_day = data

        i = 0
        result_day = result_day.reset_index()
        pysolar_mowesta_cs_ghi = []
        pysolar_mowesta_cs_dni = []
        pysolar_mowesta_cs_dhi = []
        pysolar_mowesta_ghi = []
        pysolar_mowesta_dni = []
        pysolar_mowesta_dhi = []
        IandP_ghi = []
        IandP_dhi = []
        IandP_dni = []

        while (i < len(result_day) - 1):
            i = i + 1
            date_pysolar = result_day.iloc[i]['timestamp']
            IandP_ghi.append(ineichen.iloc[i]['IandP_ghi'])
            IandP_dhi.append(ineichen.iloc[i]['IandP_dhi'])
            IandP_dni.append(ineichen.iloc[i]['IandP_dni'])
            pysolar_ghi, pysolar_dni, pysolar_dhi = pysolar_query(date_time=date_pysolar)
            pysolar_mowesta_ghi.append(pysolar_ghi)
            pysolar_mowesta_dni.append(pysolar_dni)
            pysolar_mowesta_dhi.append(pysolar_dhi)
            pysolar_mowesta_cs_ghi.append(float(clearsky_scaling(result_day.iloc[i]['total_clouds'], pysolar_ghi)))
            pysolar_mowesta_cs_dni.append(float(clearsky_scaling(result_day.iloc[i]['total_clouds'], pysolar_dni)))
            pysolar_mowesta_cs_dhi.append(float(clearsky_scaling(result_day.iloc[i]['total_clouds'], pysolar_dhi)))

        del result_day['zero_clouds']

        df2 = pd.DataFrame(IandP_ghi)
        df2.columns = ['IandP_ghi']
        result_day = result_day.join(df2, how='outer')

        df2 = pd.DataFrame(IandP_dhi)
        df2.columns = ['IandP_dhi']
        result_day = result_day.join(df2, how='outer')

        df2 = pd.DataFrame(IandP_dni)
        df2.columns = ['IandP_dni']
        result_day = result_day.join(df2, how='outer')

        df2 = pd.DataFrame(pysolar_mowesta_cs_ghi)
        df2.columns = ['pysolar_mowesta_cs_ghi']
        result_day = result_day.join(df2, how='outer')

        df2 = pd.DataFrame(pysolar_mowesta_cs_dni)
        df2.columns = ['pysolar_mowesta_cs_dni']
        result_day = result_day.join(df2, how='outer')

        df2 = pd.DataFrame(pysolar_mowesta_cs_dhi)
        df2.columns = ['pysolar_mowesta_cs_dhi']
        result_day = result_day.join(df2, how='outer')

        df3 = pd.DataFrame(pysolar_mowesta_ghi)
        df3.columns = ['pysolar_ghi']
        result_day = result_day.join(df3, how='outer')

        df3 = pd.DataFrame(pysolar_mowesta_dni)
        df3.columns = ['pysolar_dni']
        result_day = result_day.join(df3, how='outer')

        df3 = pd.DataFrame(pysolar_mowesta_dhi)
        df3.columns = ['pysolar_dhi']
        result_day = result_day.join(df3, how='outer')

        result_day = result_day.set_index('timestamp')
        from dateutil import tz
        start_loc = datetime.datetime(year=date.year, month=date.month, day=date.day, hour=0,
                                      minute=0,
                                      second=0, tzinfo=tz.gettz(timezone))
        result_day = result_day.loc[start_loc:start_loc + datetime.timedelta(days=1) - datetime.timedelta(minutes=1)]
        result_day.to_csv(file_path)
        print("Dataset saved.")
        return result_day

    result_day = calculate_irradiations()

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
    file_path = 'data/temporary_datasets/mowesta_archive/' + str_name
    return result_day, result_minute, result_minute_dict, file_path


# print(read_MoWeSta_CSV(date=datetime.datetime(year=date.year, month=date.month, day=date.day, hour=date.hour, minute=date.minute,second=date.second), path=mowesta_path))

def read_GFS_CSV(address="Kuglerstraße, Essen,45144", latitude=51.455643, longitude=7.011555,
                 date=datetime.datetime.now(), sys_exit=False, path='data/GFS_archive/PROGNOES_GFS_45144_210921.csv'):
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
    import sys
    import pvlib.location
    import pysolar
    import pytz

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

    dataset_availablity = True

    time_now = datetime.date(year=date.year, month=date.month, day=date.day)
    start = pd.Timestamp(time_now, tz=timezone)

    str_name = "PROGNOES_GFS_" + str(int(address.split(",")[2])) + "_" + str(start.year)[2:] + (
        '{:02d}'.format(start.month)) + '{:02d}'.format(start.day) + ".csv"

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(ROOT_DIR, 'data/temporary_datasets/GFS_archive/' + str_name)

    def clearsky_scaling(cloud_cover, ghi_clear, offset=35):
        """
            for more information see pvlib.forecast.py : cloud_cover_to_ghi_linear
        """

        def timezone_converter(date_time, source_timezone, destination_timezone):
            source_timezone = pytz.timezone(source_timezone)
            destination_timezone = pytz.timezone(destination_timezone)
            date_time = source_timezone.localize(date_time)
            date_time = date_time.astimezone(destination_timezone)

            return date_time

        date_time = datetime.datetime(year=date.year, month=date.month, day=date.day,
                                      hour=date.hour,
                                      minute=date.minute, second=date.second)
        date_time = date_time + datetime.timedelta(hours=1)
        date_time = timezone_converter(date_time, timezone, "UTC")

        offset = offset / 100.
        cloud_cover = cloud_cover / 100.
        ghi = (offset + (1 - offset) * (1 - cloud_cover)) * ghi_clear

        from pvlib import solarposition
        altitude_deg = pysolar.solar.get_altitude(round(latitude, 7), round(longitude, 7), date_time)
        press = 100 * ((44331.514 - altitude_deg) / 11880.516) ** (1 / 0.1902632)
        solarposition = solarposition.get_solarposition(latitude=latitude, longitude=longitude, time=date_time,
                                                        altitude=altitude_deg,
                                                        pressure=press)

        dni = pvlib.forecast.disc(ghi, solarposition['zenith'], cloud_cover)['dni'][0]
        dhi = ghi - dni * np.cos(np.radians(solarposition['zenith']))[0]
        # print("Hallllllllllllllo   ",ghi,dni,dhi)
        return ghi, dni, dhi

    def download_dataset_from_GFS():
        df = pd.read_csv(path, parse_dates=True, index_col='timestamp')

        del df['cs_ghi']
        del df['cs_dni']
        del df['cs_dhi']
        del df['cn_ghi']
        del df['cn_dni']
        del df['cn_dhi']

        from pvlib.forecast import GFS
        model = GFS(resolution='Quarter')
        data = model.rename(df)
        model.location = pvlib.location.Location(latitude, longitude, timezone)

        result_day = data

        irrads = model.cloud_cover_to_irradiance(data['total_clouds'], how='clearsky_scaling')
        irrads.columns = ["IandP_GFS_cs_ghi", "IandP_GFS_cs_dni", "IandP_GFS_cs_dhi"]
        result_day = result_day.join(irrads, how='outer')

        irrads = model.cloud_cover_to_irradiance(data['total_clouds'], how='campbell_norman')
        irrads.columns = ["CampbellNorman_GFS_cn_ghi", "CampbellNorman_GFS_cn_dni", "CampbellNorman_GFS_cn_dhi"]
        result_day = result_day.join(irrads, how='outer')

        # pysolar gfs clear sky scailing
        i = 0
        result_day = result_day.reset_index()
        pysolar_GFS_cs_dni = []
        pysolar_GFS_cs_dhi = []
        pysolar_GFS_cs_ghi = []

        while (i < len(result_day) - 1):
            i = i + 1
            date_pysolar = result_day.iloc[i]['timestamp']
            pysolar_ghi, pysolar_dni, pysolar_dhi = pysolar_query(date_time=date_pysolar)
            irrads = clearsky_scaling(result_day.iloc[i]['total_clouds'], pysolar_ghi)
            # print(irrads)

            pysolar_GFS_cs_ghi.append(float(irrads[0]))
            pysolar_GFS_cs_dni.append(float(irrads[1]))
            pysolar_GFS_cs_dhi.append(float(irrads[2]))

        df2 = pd.DataFrame(pysolar_GFS_cs_ghi)
        df2.columns = ['pysolar_GFS_cs_ghi']
        result_day = result_day.join(df2, how='outer')

        df2 = pd.DataFrame(pysolar_GFS_cs_dni)
        df2.columns = ['pysolar_GFS_cs_dni']
        result_day = result_day.join(df2, how='outer')

        df2 = pd.DataFrame(pysolar_GFS_cs_dhi)
        df2.columns = ['pysolar_GFS_cs_dhi']
        result_day = result_day.join(df2, how='outer')

        result_day = result_day.set_index('timestamp')

        result_day.to_csv(file_path)

        return result_day

    result_day = download_dataset_from_GFS()

    if dataset_availablity:
        start_loc = datetime.datetime(year=date.year, month=date.month, day=date.day, hour=date.hour,
                                      minute=date.minute,
                                      second=0, tzinfo=tz.gettz(timezone))
        print(start_loc)
        filtered_df = result_day.loc[start_loc:start_loc + datetime.timedelta(minutes=1)]
        result_minute = filtered_df.head(1)
        result_minute = result_minute.reset_index()
        result_minute_dict = {}
        for key in list(result_minute.columns):
            result_minute_dict.update({key: result_minute[key][0]})
    else:
        result_day, result_minute, result_minute_dict = None, None, None
    file_path = 'data/temporary_datasets/GFS_archive/' + str_name
    return result_day, result_minute, result_minute_dict, file_path


# print(read_GFS_CSV(date=datetime.datetime(year=date.year,month=date.month,day=date.day,hour=date.hour,minute=date.minute,second=date.second), path=gfs_path))

def create_main_CSV(mowesta_path='data/temporary_datasets/mowesta_archive/PROGNOES_MOWESTA_45144_211227.csv',
                    GFS_path='data/temporary_datasets/GFS_archive/PROGNOES_GFS_45144_211227.csv',
                    SOCRAETES_Path='data/SOCRAETES/12_27_2021/SOCRAETES_SM141K07L_211227_140356.hdf5'):
    '''
    this fuction creates a main csv file that contains all required weather and irradation parameter avaialbale
    from existing mowesta nd GFS file.
    :param mowesta_path: the path of the existing mowesta File
    :param GFS_path: the path of existing GFS file
    :param SOCRAETES_Path: the path of recorded irradiation with SOCRAETES Recorder
    :return: Function returns the values of main CSV for the requested minute.
    '''

    from general_functions import extract_z_from_hdf5, get_metadata
    from timezonefinder import TimezoneFinder
    from dateutil import tz

    mowesta_dataset = pd.read_csv(mowesta_path)
    mowesta_dataset['mowesta_timestamp'] = mowesta_dataset.loc[:, 'timestamp']
    mowesta_dataset = mowesta_dataset.rename(
        columns={'temperature': 'MoWeSta_temperature', 'total_clouds': 'MoWeSta_total_clouds',
                 'precipitation': 'MoWeSta_precipitation', 'snowDepth': 'MoWeSta_snowDepth',
                 'humidity': 'MoWeSta_humidity', 'windSpeed': 'MoWeSta_windSpeed', 'pressure': 'MoWeSta_pressure'})
    mowesta_dataset = mowesta_dataset.set_index('timestamp')
    GFS_dataset = pd.read_csv(GFS_path)

    GFS_dataset['GFS_timestamp'] = GFS_dataset.loc[:, 'timestamp']
    GFS_dataset = GFS_dataset.rename(
        columns={'temp_air': 'GFS_temp_air', 'wind_speed': 'GFS_wind_speed',
                 'total_clouds': 'GFS_total_clouds', 'low_clouds': 'GFS_low_clouds', 'mid_clouds': 'GFS_mid_clouds',
                 'high_clouds': 'GFS_high_clouds'})
    GFS_dataset = GFS_dataset.set_index('timestamp')
    mowesta_dataset = mowesta_dataset.join(GFS_dataset, how='outer')

    mowesta_dataset.to_csv('data/temporary_datasets/Main_CSV.csv')

    df = extract_z_from_hdf5(SOCRAETES_Path)
    df.columns = ["X", "Y", "Z", "Curve"]
    metadata = dict(get_metadata(SOCRAETES_Path)[0])
    Date = metadata['Date']
    Start = Date + " " + metadata['Start Time (Local Timezone)']
    Start_time = datetime.datetime.strptime(Start, "%d.%m.%Y %H:%M:%S.%f")

    mowesta_dataset = pd.read_csv('data/temporary_datasets/Main_CSV.csv', parse_dates=True,
                                  index_col='timestamp')

    tf = TimezoneFinder()
    latitude = 51.455643
    longitude = 7.011555
    timezone = tf.timezone_at(lng=longitude, lat=latitude)

    start_loc = datetime.datetime(year=Start_time.year, month=Start_time.month, day=Start_time.day,
                                  hour=Start_time.hour,
                                  minute=Start_time.minute,
                                  second=0, tzinfo=tz.gettz(timezone))

    filtered_df = mowesta_dataset.loc[start_loc:start_loc + datetime.timedelta(minutes=1)]
    filtered_df = filtered_df.reset_index()
    result = {}
    for key, value in filtered_df.items():
        for keys, values in value.items():
            result.update({key: str(values)})
    return result


def dataset_version_converter(SOCRAETES_Path='data/SOCRAETES/12_27_2021/SOCRAETES_SM141K07L_211227_140356.hdf5',
                              solar_cell="SM111K04L", thermometer_temperature=25, address="Kuglerstraße, Essen,45144",
                              latitude=51.455643, longitude=7.011555):
    """
    This function converts the old version of HDF5 File recorded with SOCRAETES to enable the user using some functions
    like diverse model of plotting or creating the datasets for the desired time manually.
    The source file will not be changed.
    :param SOCRAETES_Path:
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

    src = SOCRAETES_Path
    dst = 'data/temporary_datasets/temp_' + SOCRAETES_Path.split('/')[-1]
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

    Main_CSV = create_main_CSV(mowesta_path=Paper_mowesta_path, GFS_path=Paper_GFS_path, SOCRAETES_Path=SOCRAETES_Path)
    prognoes_conditions = {"start_time": "Not Available",
                           "end_time": "Not Available",
                           "thermometer_temperature": "Not Available", "address": "Not Available",
                           "latitude": "Not Available", "longitude": "Not Available",
                           "solar_cell": "Not Available"}

    weather_conditions = {"timestamp": "Not Available", "temperature": "Not Available", "total_clouds": "Not Available",
                          "precipitation": "Not Available", "snowDepth": "Not Available", "humidity": "Not Available",
                          "windSpeed": "Not Available", "pressure": "Not Available",
                          "Iandp_MoWeSta_cs_ghi": "Not Available", "Iandp_MoWeSta_cs_dni": "Not Available",
                          "Iandp_MoWeSta_cs_dhi": "Not Available", "CambellNorman_MoWeSta_cn_ghi": "Not Available",
                          "CambellNorman_MoWeSta_cn_dni": "Not Available",
                          "CambellNorman_MoWeSta_cn_dhi": "Not Available", "pysolar_mowesta_cs_ghi": "Not Available",
                          "pysolar_mowesta_cs_dni": "Not Available", "pysolar_mowesta_cs_dhi": "Not Available",
                          "pysolar_ghi": "Not Available", "pysolar_dni": "Not Available",
                          "pysolar_dhi": "Not Available"}

    irradiation_weather_conditions = {"timestamp": "Not Available", "temp_air": "Not Available",
                                      "wind_speed": "Not Available", "total_clouds": "Not Available",
                                      "low_clouds": "Not Available", "mid_clouds": "Not Available",
                                      "high_clouds": "Not Available", "GFS_ghi": "Not Available",
                                      "IandP_GFS_cs_ghi": "Not Available", "IandP_GFS_cs_dni": "Not Available",
                                      "IandP_GFS_cs_dhi": "Not Available", "CampbellNorman_GFS_cn_ghi": "Not Available",
                                      "CampbellNorman_GFS_cn_dni": "Not Available",
                                      "CampbellNorman_GFS_cn_dhi": "Not Available",
                                      "pysolar_GFS_cs_ghi": "Not Available", "pysolar_GFS_cs_dni": "Not Available",
                                      "pysolar_GFS_cs_dhi": "Not Available"}

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

    if "weather_conditions" in f:
        if Main_CSV is not None:
            print("MoWeSta Dataset for the given address and datetime is available.")
            mowesta_timestamp = Main_CSV['mowesta_timestamp']
            mowesta_timestamp = mowesta_timestamp[:-6]
            mowesta_timestamp = (datetime.datetime.strptime(mowesta_timestamp, "%Y-%m-%d %H:%M:%S")).timestamp()

            weather_conditions = {"timestamp": str(mowesta_timestamp),
                                  "temperature": str(Main_CSV['MoWeSta_temperature']),
                                  "total_clouds": str(Main_CSV['MoWeSta_total_clouds']),
                                  "precipitation": str(Main_CSV['MoWeSta_precipitation']),
                                  "snowDepth": str(Main_CSV['MoWeSta_snowDepth']),
                                  "humidity": str(Main_CSV['MoWeSta_humidity']),
                                  "windSpeed": str(Main_CSV['MoWeSta_windSpeed']),
                                  "pressure": str(Main_CSV['MoWeSta_pressure']),
                                  "Iandp_MoWeSta_cs_ghi": str(Main_CSV['Iandp_MoWeSta_cs_ghi']),
                                  "Iandp_MoWeSta_cs_dni": str(Main_CSV['Iandp_MoWeSta_cs_dni']),
                                  "Iandp_MoWeSta_cs_dhi": str(Main_CSV['Iandp_MoWeSta_cs_dhi']),
                                  "CambellNorman_MoWeSta_cn_ghi": str(Main_CSV['CambellNorman_MoWeSta_cn_ghi']),
                                  "CambellNorman_MoWeSta_cn_dni": str(Main_CSV['CambellNorman_MoWeSta_cn_dni']),
                                  "CambellNorman_MoWeSta_cn_dhi": str(Main_CSV['CambellNorman_MoWeSta_cn_dhi']),
                                  "pysolar_mowesta_cs_ghi": str(Main_CSV['pysolar_mowesta_cs_ghi']),
                                  "pysolar_mowesta_cs_dni": str(Main_CSV['pysolar_mowesta_cs_dni']),
                                  "pysolar_mowesta_cs_dhi": str(Main_CSV['pysolar_mowesta_cs_dhi']),
                                  "IandP_ghi": str(Main_CSV['IandP_ghi']),
                                  "IandP_dhi": str(Main_CSV['IandP_dhi']),
                                  "IandP_dni": str(Main_CSV['IandP_dni']),
                                  "pysolar_ghi": str(Main_CSV['pysolar_ghi']),
                                  "pysolar_dni": str(Main_CSV['pysolar_dni']),
                                  "pysolar_dhi": str(Main_CSV['pysolar_dhi'])}
        else:
            print("MoWeSta Dataset for the given address and datetime is not available.")
            weather_conditions = {"timestamp": str(start_date), "temperature": str(25),
                                  "total_clouds": str(0),
                                  "precipitation": str(0), "snowDepth": str(0),
                                  "windSpeed": str(0),
                                  "humidity": str(0), "pressure": str(0), "pysolar_ghi": str(0),
                                  "IandP_cs_ghi": str(0), "cs_ghi": str(0),
                                  "cs_dni": str(0), "cs_dhi": str(0), "cn_ghi": str(0),
                                  "cn_dni": str(0), "cn_dhi": str(0)}

    if "irradiation_weather_conditions" in f:
        if Main_CSV is not None:
            print("GFS Dataset for the given address and datetime is available.")
            irradiation_weather_conditions = {"timestamp": str(Main_CSV['GFS_timestamp']),
                                              "temp_air": str(Main_CSV['GFS_temp_air']),
                                              "wind_speed": str(Main_CSV['GFS_wind_speed']),
                                              "total_clouds": str(Main_CSV['GFS_total_clouds']),
                                              "low_clouds": str(Main_CSV['GFS_low_clouds']),
                                              "mid_clouds": str(Main_CSV['GFS_mid_clouds']),
                                              "high_clouds": str(Main_CSV['GFS_high_clouds']),
                                              "GFS_ghi": str(Main_CSV['GFS_ghi']),
                                              "IandP_GFS_cs_ghi": str(Main_CSV['IandP_GFS_cs_ghi']),
                                              "IandP_GFS_cs_dni": str(Main_CSV['IandP_GFS_cs_dni']),
                                              "IandP_GFS_cs_dhi": str(Main_CSV['IandP_GFS_cs_dhi']),
                                              "CampbellNorman_GFS_cn_ghi": str(Main_CSV['CampbellNorman_GFS_cn_ghi']),
                                              "CampbellNorman_GFS_cn_dni": str(Main_CSV['CampbellNorman_GFS_cn_dni']),
                                              "CampbellNorman_GFS_cn_dhi": str(Main_CSV['CampbellNorman_GFS_cn_dhi']),
                                              "pysolar_GFS_cs_ghi": str(Main_CSV['pysolar_GFS_cs_ghi']),
                                              "pysolar_GFS_cs_dni": str(Main_CSV['pysolar_GFS_cs_dni']),
                                              "pysolar_GFS_cs_dhi": str(Main_CSV['pysolar_GFS_cs_dhi'])}

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

    if "prognoes_conditions" in f:
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
        dsetx = f.create_dataset("weather_conditions", (2, 23), dtype=dt)
        dsetx[0] = list(weather_conditions.keys())
        dsetx[1] = list(weather_conditions.values())

    def change_irradiation_weather_conditions():
        f = h5py.File(path, 'a')
        if 'irradiation_weather_conditions' in f:
            del f['irradiation_weather_conditions']
        dt = h5py.string_dtype(encoding='utf-8')
        dsetx = f.create_dataset("irradiation_weather_conditions", (2, 17), dtype=dt)
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


def dataset_to_hdf5(solar_cell_model="SM111K04L", address="Kuglerstraße, Essen,45144", latitude=51.455643,
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

    solar_cell = solar_cell()
    solar_cell.SC_obj_from_csv('data/solar_cell_profiles/Calculated_' + solar_cell_model + '.csv')

    methods = {"Iandp_MoWeSta_cs_ghi": True, "CambellNorman_MoWeSta_cn_ghi": True, "pysolar_mowesta_cs_ghi": True,
               "pysolar_ghi": True, "IandP_ghi": True, "GFS_ghi": True,
               "IandP_GFS_cs_ghi": True, "CampbellNorman_GFS_cn_ghi": True,
               "pysolar_GFS_cs_ghi": True, "original_curve_SOCRAETES": True}

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
            print("Daaaaaaaaaaaaaaaaaate:", date, (start_loc + datetime.timedelta(minutes=1)))
            filtered_df = dataset.loc[start_loc:start_loc + datetime.timedelta(minutes=1)]
            result_minute = filtered_df.head(1)
            result_minute = result_minute.reset_index()
            result = {}
            for key, values in result_minute.to_dict().items():
                for key_value, values_value in values.items():
                    result.update({key: values_value})

            result.update({'timestamp': str(start_loc)})
            return result

        Weather_Datasets_all = pd.read_csv('data/temporary_datasets/Main_CSV.csv', parse_dates=True,
                                           index_col='timestamp')
        Main_CSV = dataset_minute_to_dict(Weather_Datasets_all)

        Iandp_MoWeSta_cs_ghi = [1000, 1000, {'v': 0, 'i': 0}]
        CambellNorman_MoWeSta_cn_ghi = [1000, 1000, {'v': 0, 'i': 0}]
        IandP_ghi = [1000, 1000, {'v': 0, 'i': 0}]
        pysolar_mowesta_cs_ghi = [1000, 1000, {'v': 0, 'i': 0}]
        pysolar_ghi = [1000, 1000, {'v': 0, 'i': 0}]
        GFS_ghi = [1000, 1000, {'v': 0, 'i': 0}]
        IandP_GFS_cs_ghi = [1000, 1000, {'v': 0, 'i': 0}]
        CampbellNorman_GFS_cn_ghi = [1000, 1000, {'v': 0, 'i': 0}]
        pysolar_GFS_cs_ghi = [1000, 1000, {'v': 0, 'i': 0}]

        if methods['Iandp_MoWeSta_cs_ghi']:
            Iandp_MoWeSta_cs_ghi = gen_curve_from_ghi(method='Iandp_MoWeSta_cs_ghi',
                                                      ghi=Main_CSV['Iandp_MoWeSta_cs_ghi'])
        if methods['CambellNorman_MoWeSta_cn_ghi']:
            CambellNorman_MoWeSta_cn_ghi = gen_curve_from_ghi(method='CambellNorman_MoWeSta_cn_ghi',
                                                              ghi=Main_CSV['CambellNorman_MoWeSta_cn_ghi'])
        if methods['IandP_ghi']:
            IandP_ghi = gen_curve_from_ghi(method='IandP_ghi', ghi=Main_CSV['IandP_ghi'])

        if methods['pysolar_mowesta_cs_ghi']:
            pysolar_mowesta_cs_ghi = gen_curve_from_ghi(method='pysolar_mowesta_cs_ghi',
                                                        ghi=Main_CSV['pysolar_mowesta_cs_ghi'])
        if methods['pysolar_ghi']:
            pysolar_ghi = gen_curve_from_ghi(method='pysolar_ghi', ghi=Main_CSV['pysolar_ghi'])
        if methods['GFS_ghi']:
            GFS_ghi = gen_curve_from_ghi(method='GFS_ghi', ghi=Main_CSV['GFS_ghi'])
        if methods['IandP_GFS_cs_ghi']:
            IandP_GFS_cs_ghi = gen_curve_from_ghi(method='IandP_GFS_cs_ghi', ghi=Main_CSV['IandP_GFS_cs_ghi'])
        if methods['CampbellNorman_GFS_cn_ghi']:
            CampbellNorman_GFS_cn_ghi = gen_curve_from_ghi(method='CampbellNorman_GFS_cn_ghi',
                                                           ghi=Main_CSV['CampbellNorman_GFS_cn_ghi'])
        if methods['pysolar_GFS_cs_ghi']:
            pysolar_GFS_cs_ghi = gen_curve_from_ghi(method='pysolar_GFS_cs_ghi', ghi=Main_CSV['pysolar_GFS_cs_ghi'])

        result = {"Iandp_MoWeSta_cs_ghi": Iandp_MoWeSta_cs_ghi,
                  "CambellNorman_MoWeSta_cn_ghi": CambellNorman_MoWeSta_cn_ghi, "IandP_ghi": IandP_ghi,
                  "pysolar_mowesta_cs_ghi": pysolar_mowesta_cs_ghi, "pysolar_ghi": pysolar_ghi,
                  "GFS_ghi": GFS_ghi,
                  "IandP_GFS_cs_ghi": IandP_GFS_cs_ghi,
                  "CampbellNorman_GFS_cn_ghi": CampbellNorman_GFS_cn_ghi, "pysolar_GFS_cs_ghi": pysolar_GFS_cs_ghi}

        return result

    def plot_curve(result, start_time):
        for key, value in methods.items():
            if value:
                if max(result[key][2]['v']) > 0.5 and max(result[key][2]['i']) > 300:
                    plt.plot(result[key][2]['v'], result[key][2]['i'],
                             label=key)
                    VMP, IMP, PMP, power_points = max_power_point(result[key][2]['v'], result[key][2]['i'])
                    if verbose_show:
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
        plt.grid(True)
        plt.title("Solar Cell '" + str(solar_cell_model) + "' IV Characteristics", fontsize=9, y=0.99)
        plt.xlabel("Solar Cell Voltage (V)")
        plt.ylabel("Solar Cell Current (uA)")

        FileName = 'data/temporary_datasets/Files/' + (str(start_time)).replace(":", ".") + ".png"
        #plt.savefig(FileName, bbox_inches="tight", pad_inches=1, transparent=True, facecolor="w", edgecolor='w',
                    #orientation='landscape', dpi=300)
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


def compare_dataset(path='data/SOCRAETES/12_25_2021/SOCRAETES_SM141K07L_211225_125009.hdf5'):
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
    print(prognoes_conditions)
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


day_one = {"date": datetime.datetime(year=2021, month=10, day=28, hour=12, minute=10, second=10),
           "mowesta_path": "data/mowesta_archive/PROGNOES_MOWESTA_45144_211028.csv",
           "gfs_path": 'data/GFS_archive/PROGNOES_GFS_45144_211028.csv',
           "SOCRAETES_Path": 'data/SOCRAETES/10_28_2021/SOCRAETES_SM141K07L_211028_130328.hdf5'}

day_two = {"date": datetime.datetime(year=2021, month=11, day=1, hour=12, minute=10, second=10),
           "mowesta_path": "data/mowesta_archive/PROGNOES_MOWESTA_45144_211101.csv",
           "gfs_path": 'data/GFS_archive/PROGNOES_GFS_45144_211101.csv',
           "SOCRAETES_Path": 'data/SOCRAETES/11_01_2021/SOCRAETES_SM141K07L_211101_152809.hdf5'}

day_three = {"date": datetime.datetime(year=2021, month=11, day=6, hour=12, minute=10, second=10),
             "mowesta_path": "data/mowesta_archive/PROGNOES_MOWESTA_45144_211106.csv",
             "gfs_path": 'data/GFS_archive/PROGNOES_GFS_45144_211106.csv',
             "SOCRAETES_Path": 'data/SOCRAETES/11_06_2021/SOCRAETES_SM141K07L_211106_135119.hdf5'}

day_four = {"date": datetime.datetime(year=2021, month=12, day=23, hour=12, minute=10, second=10),
            "mowesta_path": "data/mowesta_archive/PROGNOES_MOWESTA_45144_211223.csv",
            "gfs_path": 'data/GFS_archive/PROGNOES_GFS_45144_211223.csv',
            "SOCRAETES_Path": 'data/SOCRAETES/12_23_2021/SOCRAETES_SM141K07L_211223_133213.hdf5'}

day_five = {"date": datetime.datetime(year=2021, month=12, day=25, hour=12, minute=10, second=10),
            "mowesta_path": "data/mowesta_archive/PROGNOES_MOWESTA_45144_211225.csv",
            "gfs_path": 'data/GFS_archive/PROGNOES_GFS_45144_211225.csv',
            "SOCRAETES_Path": 'data/SOCRAETES/12_25_2021/SOCRAETES_SM141K07L_211225_130146.hdf5'}

day_six = {"date": datetime.datetime(year=2021, month=12, day=26, hour=12, minute=10, second=10),
           "mowesta_path": "data/mowesta_archive/PROGNOES_MOWESTA_45144_211226.csv",
           "gfs_path": 'data/GFS_archive/PROGNOES_GFS_45144_211226.csv',
           "SOCRAETES_Path": 'data/SOCRAETES/12_26_2021/SOCRAETES_SM141K07L_211226_134336.hdf5'}

day_seven = {"date": datetime.datetime(year=2021, month=12, day=27, hour=12, minute=10, second=10),
             "mowesta_path": "data/mowesta_archive/PROGNOES_MOWESTA_45144_211227.csv",
             "gfs_path": 'data/GFS_archive/PROGNOES_GFS_45144_211227.csv',
             "SOCRAETES_Path": 'data/SOCRAETES/12_27_2021/SOCRAETES_SM141K07L_211227_140749.hdf5'}

SolarCell = 'SM141K07L'
day = day_one
date = day.get('date')

Paper_mowesta_path = (read_MoWeSta_CSV(
    date=datetime.datetime(year=date.year, month=date.month, day=date.day, hour=date.hour, minute=date.minute,
                           second=date.second), path=day.get('mowesta_path')))[3]
Paper_GFS_path = (read_GFS_CSV(
    date=datetime.datetime(year=date.year, month=date.month, day=date.day, hour=date.hour, minute=date.minute,
                           second=date.second), path=day.get('gfs_path')))[3]

create_main_CSV(mowesta_path=Paper_mowesta_path, GFS_path=Paper_GFS_path, SOCRAETES_Path=day.get('SOCRAETES_Path'))
New_Path = dataset_version_converter(SOCRAETES_Path=day.get('SOCRAETES_Path'), solar_cell=SolarCell)
compare_dataset(New_Path)
