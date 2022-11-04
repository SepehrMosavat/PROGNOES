import datetime


def mowesta_forecast_all(address='Kuglerstraße, Essen, 45144', latitude=51.455643, longitude=7.011555):
    """
    This function downloads the weather information from MoWeSta Website
    and provide the raw GHI of that location from pysolar.
    :param address:
    The address of desired location. Default is 'Kuglerstraße, Essen, 45144'.
    :param latitude:
    The latitude of desired location. Default is '51.455643'.
    :param longitude:
    The longitude of desired location. Default is '7.011555'.
    :return:
    Returns a dict containing all data received from the MoWeSta.
    """
    import requests
    import urllib
    import json
    import datetime
    from pysolar_functions import pysolar_query
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
        result = resp.json()
    except:
        print("An error occurred while connecting to the mowesta.com. Please check your connection and try again.")
        sys.exit()

    json.dumps(result, indent=True)
    j = 0
    for i in range(0, 100):
        print("--------------------------------------")
        print("Time: ", datetime.datetime.utcfromtimestamp(float(str(result["conditions"][i]['time'])[:-3])))
        print("Temperature: ", result["conditions"][i]['temperature'])
        print("Coordinate Of City: ", result["coordinate"])
        print("Cloud Coverage: ", result["conditions"][i]['cloudCoverage'])
        print("Cloud Depth: ", result["conditions"][i]['cloudDepth'])
        print("Precipitation", result["conditions"][i]['precipitation'])
        print("Snow Depth: ", result["conditions"][i]['snowDepth'])
        print("Pysolar:",
              pysolar_query(date_time=datetime.datetime.fromtimestamp(
                  float(str(result["conditions"][i]['time'])[:-3])).timestamp(),
                            latitude=round(latitude, 7),
                            longitude=round(longitude, 7)))

        if j == len(result['conditions']):
            j = len(result['conditions'])
        else:
            j = j + 1
    return result


def mowesta_day_minute(address='Kuglerstraße, Essen, 45144', latitude=51.455643, longitude=7.011555,
                       date=datetime.datetime.now(), sys_exit=False):
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
    from pysolar_functions import pysolar_query
    from pvlib_functions import IandP_day_minute
    import os
    from dateutil import tz
    import pandas as pd
    import numpy as np
    import time
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

    from timezonefinder import TimezoneFinder
    tf = TimezoneFinder()
    timezone = tf.timezone_at(lng=float(longitude), lat=float(latitude))

    dataset_availablity = False
    str_name = "PROGNOES_MOWESTA_" + str(int(address.split(",")[2])) + "_" + str(date.year)[2:] + (
        '{:02d}'.format(date.month)) + '{:02d}'.format(date.day) + ".csv"
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(ROOT_DIR, 'data/mowesta_archive/' + str_name)

    def df_interpolate_all(df):

        start_t = df[0][0]
        start = datetime.datetime(year=start_t.year, month=start_t.month, day=start_t.day, hour=0, minute=0, second=0)
        end_t = df[len(df) - 1][0]
        end = datetime.datetime(year=end_t.year, month=end_t.month, day=end_t.day, hour=0, minute=0,
                                second=0) + datetime.timedelta(days=1)

        date_time = []
        start_t = datetime.datetime(year=start.year, month=start.month, day=start.day, hour=0, minute=0, second=0,
                                    tzinfo=tz.gettz(timezone))

        for i in range((end - start).days * 24 * 60):
            date_time.append(start_t)
            start_t = start_t + datetime.timedelta(minutes=1)

        df1 = pd.DataFrame(date_time)

        df1.columns = ['timestamp']

        df_cols = ['temperature', 'cloudCoverage', 'cloudDepth', 'precipitation', 'snowDepth', 'humidity',
                   'windSpeed', 'pressure', 'wind_speed_u', 'wind_speed_v',
                   'pysolar_cs_ghi', 'IandP_cs_ghi']

        for i in df_cols:
            df1[i] = np.nan

        df1.set_index('timestamp', inplace=True)

        for item in range(len(df)):
            if item == 0:
                start_loc = datetime.datetime(year=start.year, month=start.month, day=start.day, hour=start.hour,
                                              minute=start.minute,
                                              second=0, tzinfo=tz.gettz(timezone))

                df1.loc[start_loc:start_loc, 'temperature'] = df[item][1]
                df1.loc[start_loc:start_loc, 'cloudCoverage'] = df[item][2]
                df1.loc[start_loc:start_loc, 'cloudDepth'] = df[item][3]
                df1.loc[start_loc:start_loc, 'precipitation'] = df[item][4]
                df1.loc[start_loc:start_loc, 'snowDepth'] = df[item][5]
                df1.loc[start_loc:start_loc, 'humidity'] = df[item][6]
                df1.loc[start_loc:start_loc, 'windSpeed'] = df[item][7]
                df1.loc[start_loc:start_loc, 'pressure'] = df[item][8]
                df1.loc[start_loc:start_loc, 'wind_speed_u'] = df[item][9]
                df1.loc[start_loc:start_loc, 'wind_speed_v'] = df[item][10]
                df1.loc[start_loc:start_loc, 'pysolar_cs_ghi'] = df[item][11]
                df1.loc[start_loc:start_loc, 'IandP_cs_ghi'] = df[item][12]

            if item == len(df) - 1:
                start_loc = datetime.datetime(year=end.year, month=end.month, day=end.day, hour=end.hour,
                                              minute=end.minute,
                                              second=0, tzinfo=tz.gettz(timezone)) - datetime.timedelta(minutes=1)

                df1.loc[start_loc:start_loc, 'temperature'] = df[item][1]
                df1.loc[start_loc:start_loc, 'cloudCoverage'] = df[item][2]
                df1.loc[start_loc:start_loc, 'cloudDepth'] = df[item][3] * 100
                df1.loc[start_loc:start_loc, 'precipitation'] = df[item][4]
                df1.loc[start_loc:start_loc, 'snowDepth'] = df[item][5]
                df1.loc[start_loc:start_loc, 'humidity'] = df[item][6]
                df1.loc[start_loc:start_loc, 'windSpeed'] = df[item][7]
                df1.loc[start_loc:start_loc, 'pressure'] = df[item][8]
                df1.loc[start_loc:start_loc, 'wind_speed_u'] = df[item][9]
                df1.loc[start_loc:start_loc, 'wind_speed_v'] = df[item][10]
                df1.loc[start_loc:start_loc, 'pysolar_cs_ghi'] = df[item][11]
                df1.loc[start_loc:start_loc, 'IandP_cs_ghi'] = df[item][12]

            date = df[item][0]
            start_loc = datetime.datetime(year=date.year, month=date.month, day=date.day, hour=date.hour,
                                          minute=date.minute,
                                          second=0, tzinfo=tz.gettz(timezone))
            df1.loc[start_loc:start_loc, 'temperature'] = df[item][1]
            df1.loc[start_loc:start_loc, 'cloudCoverage'] = df[item][2]
            df1.loc[start_loc:start_loc, 'cloudDepth'] = df[item][3] * 100
            df1.loc[start_loc:start_loc, 'precipitation'] = df[item][4]
            df1.loc[start_loc:start_loc, 'snowDepth'] = df[item][5]
            df1.loc[start_loc:start_loc, 'humidity'] = df[item][6]
            df1.loc[start_loc:start_loc, 'windSpeed'] = df[item][7]
            df1.loc[start_loc:start_loc, 'pressure'] = df[item][8]
            df1.loc[start_loc:start_loc, 'wind_speed_u'] = df[item][9]
            df1.loc[start_loc:start_loc, 'wind_speed_v'] = df[item][10]
            df1.loc[start_loc:start_loc, 'pysolar_cs_ghi'] = df[item][11]
            df1.loc[start_loc:start_loc, 'IandP_cs_ghi'] = df[item][12]

        df1 = df1.interpolate(method='linear', limit_direction='forward', axis=0)
        df1 = df1.rename(columns={'cloudDepth': "total_clouds"})

        return df1, start, end

    def clearsky_scaling(cloud_cover, ghi_clear, offset=35):
        """
            for more information see pvlib.forecast.py : cloud_cover_to_ghi_linear
        """

        offset = offset / 100.
        cloud_cover = cloud_cover / 100.
        ghi = (offset + (1 - offset) * (1 - cloud_cover)) * ghi_clear
        return ghi

    def download_all_data():
        i = 0
        result = []
        while i < len(response['conditions']):

            wind_speed_u = 0
            wind_speed_v = 0
            pysolar_cs_ghi = pysolar_query(latitude, longitude, int(response["conditions"][i]['time'] / 1000) + 3600,
                                           timezone) / 1000
            pysolar_cs_ghi = float(clearsky_scaling(float(response["conditions"][i]['cloudDepth']), pysolar_cs_ghi))

            IandP_cs_ghi = float(IandP_day_minute(latitude, longitude, 0,
                                                  date=datetime.datetime.fromtimestamp(
                                                      int(response["conditions"][i]['time'] / 1000)))[1]['ghi'])
            IandP_cs_ghi = float(clearsky_scaling(float(response["conditions"][i]['cloudDepth']), IandP_cs_ghi))
            if response["conditions"][i]['windDirection'] == 'NORTH':
                wind_speed_u = 0
                wind_speed_v = -float(response["conditions"][i]['windSpeed'])
            elif response["conditions"][i]['windDirection'] == "SOUTH":
                wind_speed_u = 0
                wind_speed_v = float(response["conditions"][i]['windSpeed'])
            elif response["conditions"][i]['windDirection'] == "WEST":
                wind_speed_u = float(response["conditions"][i]['windSpeed'])
                wind_speed_v = 0
            elif response["conditions"][i]['windDirection'] == "EAST":
                wind_speed_u = -float(response["conditions"][i]['windSpeed'])
                wind_speed_v = 0
            elif response["conditions"][i]['windDirection'] == 'NORTH_WEST':
                wind_speed_u = float(response["conditions"][i]['windSpeed'])
                wind_speed_v = -float(response["conditions"][i]['windSpeed'])
            elif response["conditions"][i]['windDirection'] == "SOUTH_WEST":
                wind_speed_u = float(response["conditions"][i]['windSpeed'])
                wind_speed_v = float(response["conditions"][i]['windSpeed'])
            elif response["conditions"][i]['windDirection'] == 'NORTH_EAST':
                wind_speed_u = -float(response["conditions"][i]['windSpeed'])
                wind_speed_v = -float(response["conditions"][i]['windSpeed'])
            elif response["conditions"][i]['windDirection'] == "SOUTH_EAST":
                wind_speed_u = -float(response["conditions"][i]['windSpeed'])
                wind_speed_v = float(response["conditions"][i]['windSpeed'])

            result.append([datetime.datetime.fromtimestamp(int(response["conditions"][i]['time'] / 1000)),
                           float(response["conditions"][i]['temperature']),
                           float(response["conditions"][i]['cloudCoverage']),
                           float(response["conditions"][i]['cloudDepth']),
                           float(response["conditions"][i]['precipitation']),
                           float(response["conditions"][i]['snowDepth']),
                           float(response["conditions"][i]['humidity']),
                           float(response["conditions"][i]['windSpeed']),
                           float(response["conditions"][i]['pressure']), float(wind_speed_u), float(wind_speed_v),
                           float(pysolar_cs_ghi), float(IandP_cs_ghi)])

            i = i + 1

        return result

    def calculate_irradiations():
        result = download_all_data()

        df2, start, end = df_interpolate_all(result)

        from pvlib.forecast import GFS
        model = GFS(resolution='Quarter')
        data = model.rename(df2)
        model.location = pvlib.location.Location(latitude, longitude, timezone)
        # print("GFS model created.")

        irrad_data = model.cloud_cover_to_irradiance(data['total_clouds'], how='clearsky_scaling')
        irrad_data.columns = ["cs_ghi", "cs_dni", "cs_dhi"]
        data = data.join(irrad_data, how='outer')
        # print("Irradiance with clearsky_scaling model calculated.")

        irrad_data = model.cloud_cover_to_irradiance(data['total_clouds'], how='campbell_norman')
        irrad_data.columns = ["cn_ghi", "cn_dni", "cn_dhi"]
        data = data.join(irrad_data, how='outer')
        # print("Irradiance with campbell_norman model calculated.")

        del data["wind_speed_u"]
        del data["wind_speed_v"]
        del data["cloudCoverage"]

        start_loc = datetime.datetime(year=date.year, month=date.month, day=date.day, hour=0,
                                      minute=0,
                                      second=0, tzinfo=tz.gettz(timezone))
        result_day = data.loc[start_loc:start_loc + datetime.timedelta(days=1) - datetime.timedelta(minutes=1)]
        result_day.to_csv(file_path)
        # print("Dataset saved.")
        return result_day

    if (os.path.exists(file_path)):
        print("MoWeSta dataset exists already.")
        dataset_availablity = True
        if (time.time() - os.path.getmtime(file_path) < 86400) and (time.time() - os.path.getmtime(file_path) > 3600):
            print("Dataset is older than 3600 secs. A new one could be available, try download last recent dataset ")
            calculate_irradiations()

        result_day = pd.read_csv(file_path, parse_dates=True,
                                 index_col='timestamp')

    else:
        if abs((datetime.datetime.now() - date).days) < 8:
            print("Try to download MoWeSta dataset.")
            dataset_availablity = True
            result_day = calculate_irradiations()
        else:
            print("The requested date for MoWeSta Dataset is neither online or offline available.")
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
