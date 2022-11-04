import logging
import os
import queue
import threading
import datetime

import h5py
import numpy as np

import warnings

warnings.filterwarnings("ignore", module='pvlib')
warnings.filterwarnings("ignore", module='pysolar')

from iv_curves_definitions import HarvestingCondition

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_log_handler = logging.StreamHandler()
console_log_formatter = logging.Formatter('%(levelname)s - %(message)s')
console_log_handler.setFormatter(console_log_formatter)
logger.addHandler(console_log_handler)


def create_conditions(path, thermometer_temperature, solar_cell_model, address="Kuglerstraße, Essen,45144",
                      latitude=51.455643, longitude=7.011555, start_time=datetime.datetime.now(),
                      end_time=datetime.datetime.now() + datetime.timedelta(seconds=30)):
    import requests
    import urllib
    from dateutil import tz
    import time
    import datetime

    import sys

    date = str(start_time.strftime("%d.%m.%Y"))
    start_time = str(start_time.strftime("%H:%M:%S.%f"))
    end_time = str(end_time.strftime("%H:%M:%S.%f"))

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

    def create_prognoes_conditions():
        dict_temp = {"start_time": str(
            datetime.datetime.strptime(date + ' ' + start_time, "%d.%m.%Y %H:%M:%S.%f").timestamp()),
            "end_time": str(
                datetime.datetime.strptime(date + ' ' + end_time, "%d.%m.%Y %H:%M:%S.%f").timestamp()),
            "thermometer_temperature": thermometer_temperature, "address": address,
            "latitude": str(latitude), "longitude": str(longitude),
            "solar_cell": solar_cell_model}

        f = h5py.File(path, 'a')
        dt = h5py.string_dtype(encoding='utf-8')
        dsetx = f.create_dataset("prognoes_conditions", (2, 7), dtype=dt)
        dsetx[0] = list(dict_temp.keys())
        dsetx[1] = list(dict_temp.values())

    def create_weather_conditions():
        def get_nearest_station_mowesta_forecast_current_hour():
            import sys
            import os
            sys.path.append(os.path.abspath('../'))
            from mowesta import mowesta_day_minute

            result = {"timestamp": str(time.time()), "temperature": "Not Available", "total_clouds": "Not Available",
                      "precipitation": "Not Available", "snowDepth": "Not Available", "windSpeed": "Not Available",
                      "humidity": "Not Available", "pressure": "Not Available", "pysolar_cs_ghi": "Not Available",
                      "IandP_cs_ghi": "Not Available", "cs_ghi": "Not Available",
                      "cs_dni": "Not Available", "cs_dhi": "Not Available", "cn_ghi": "Not Available",
                      "cn_dni": "Not Available", "cn_dhi": "Not Available"
                      }

            result_day, result_minute, result_minute_dict = mowesta_day_minute(address=address,
                                                                               date=datetime.datetime.now())

            if not result_minute.empty:
                dict_temp = result_minute.to_dict()
                dict_temp['timestamp'] = {0: str(datetime.datetime.utcnow().timestamp())}

                for key, value in dict_temp.items():
                    for keys, values in value.items():
                        result.update({key: str(values)})
            return result

        dict_temp = get_nearest_station_mowesta_forecast_current_hour()
        f = h5py.File(path, 'a')
        dt = h5py.string_dtype(encoding='utf-8')
        dsetx = f.create_dataset("weather_conditions", (2, 16), dtype=dt)
        dsetx[0] = list(dict_temp.keys())
        dsetx[1] = list(dict_temp.values())

    def create_weather_irradiation_conditions():
        import sys
        import os
        sys.path.append(os.path.abspath('../'))
        from pvlib_functions import GFS_day_minute
        result = {'timestamp': str(time.time()), 'temp_air': "Not Available", 'wind_speed': "Not Available",
                  'total_clouds': "Not Available",
                  'low_clouds': "Not Available", 'mid_clouds': "Not Available",
                  'high_clouds': "Not Available", 'GFS_ghi': "Not Available", 'cs_ghi': "Not Available",
                  'cs_dni': "Not Available",
                  'cs_dhi': "Not Available", 'cn_ghi': "Not Available",
                  'cn_dni': "Not Available", 'cn_dhi': "Not Available"}

        date = datetime.datetime.now()
        result_day, result_minute, result_minute_dict = GFS_day_minute(address, timezone, date)

        if not result_minute.empty:
            dict_temp = result_minute.to_dict()
            dict_temp['timestamp'] = {0: str(datetime.datetime.now(tz=tz.gettz(timezone)))}

            for key, value in dict_temp.items():
                for keys, values in value.items():
                    result.update({key: str(values)})
        import h5py
        f = h5py.File(path, 'a')
        dt = h5py.string_dtype(encoding='utf-8')
        dsetx = f.create_dataset("irradiation_weather_conditions", (2, 14), dtype=dt)
        dsetx[0] = list(result.keys())
        dsetx[1] = list(result.values())

        return result_day, result_minute

    create_prognoes_conditions()
    create_weather_conditions()
    create_weather_irradiation_conditions()
    return True


def generate_filename(_file_name) -> str:
    if _file_name == 'AUTO-GENERATE':
        for files in os.walk('data'):
            number_of_files_in_directory = len(files[2])
            if number_of_files_in_directory == 0:
                return 'trace_0.hdf5'
                # TODO Is this too hacky?
            list_of_file_names = files[2]
            highest_file_index = 0
            for file_name_without_extension in list_of_file_names:
                file_name_without_extension = file_name_without_extension.split('.')
                index_of_trace_file = file_name_without_extension[0].split('_')
                if int(index_of_trace_file[1]) > highest_file_index:
                    highest_file_index = int(index_of_trace_file[1])
            new_filename = 'trace_' + str(highest_file_index + 1) + '.hdf5'
            # TODO Add more error handling for the files already present in the directory
            return new_filename
    else:
        return _file_name


def write_iv_curves_to_disk(_iv_curves_queue: queue.Queue, _file_name, address, temperature, solar_cell_model,
                            _harvesting_condition: HarvestingCondition,
                            _stop_thread_event: threading.Event):
    curve_counter = 0
    data_array_buffer = []

    new_filename = '../data/SOCRAETES/' + generate_filename(_file_name)
    # new_filename = 'data/' + str(datetime.datetime.now().strftime("%m-%d-%Y--%H-%M-%S")) + ".hdf5"

    start_time = datetime.datetime.now()
    start_time_string = str(start_time.hour) + ':' + str(start_time.minute) + ':' + str(start_time.second) + '.' + \
                        str(start_time.microsecond)
    start_date_string = str(start_time.day) + '.' + str(start_time.month) + '.' + str(start_time.year)

    while True:
        if _stop_thread_event.isSet():
            end_time = datetime.datetime.now()
            end_time_string = str(end_time.hour) + ':' + str(end_time.minute) + ':' + str(end_time.second) + '.' + \
                              str(end_time.microsecond)

            logger.info("Committing curve data to the hard disk...")
            with h5py.File(new_filename, 'a') as f:
                harvesting_condition_list = [(np.string_('Date'),
                                              np.string_('Start Time (Local Timezone)'),
                                              np.string_('End Time (Local Timezone)'),
                                              np.string_('Indoor/Outdoor'),
                                              np.string_('Light Intensity (Lux)'),
                                              np.string_('Weather Condition'),
                                              np.string_('Country'),
                                              np.string_('City')),
                                             (np.string_(start_date_string),
                                              np.string_(start_time_string),
                                              np.string_(end_time_string),
                                              np.string_(_harvesting_condition.indoor_or_outdoor),
                                              np.string_(_harvesting_condition.light_intensity),
                                              np.string_(_harvesting_condition.weather_condition),
                                              np.string_(_harvesting_condition.country),
                                              np.string_(_harvesting_condition.city))]
                dataset = f.create_dataset('harvesting conditions', data=harvesting_condition_list)
                create_conditions(path=new_filename, address=address, thermometer_temperature=temperature,
                                  solar_cell_model=solar_cell_model, end_time=end_time)
            for arr in data_array_buffer:
                with h5py.File(new_filename, 'a') as f:
                    dataset = f.create_dataset('curve' + str(curve_counter), data=arr, dtype='f')
                curve_counter += 1
            break

        if not _iv_curves_queue.empty():
            iv_curve = _iv_curves_queue.get()
            x_temp = []
            y_temp = []
            for c in iv_curve.curve_points_list:
                x_temp.append(c.x)
                y_temp.append(c.y)

            data_array = np.array([x_temp, y_temp], dtype=float)
            data_array_buffer.append(data_array)
