import datetime

import pysolar.util
from pysolar.solar import *
from pysolar import radiation

import warnings

warnings.filterwarnings("ignore", module='pvlib')
warnings.filterwarnings("ignore", module='pysolar')


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
    import pytz
    if date_time == 1:
        date_time = datetime.datetime.fromtimestamp(datetime.datetime.now().timestamp())
    else:
        date_time = datetime.datetime.fromtimestamp(date_time)

    def timezone_converter(date_time, source_timezone, destination_timezone):
        source_timezone = pytz.timezone(source_timezone)
        destination_timezone = pytz.timezone(destination_timezone)
        date_time = source_timezone.localize(date_time)
        date_time = date_time.astimezone(destination_timezone)

        return date_time

    date_time = datetime.datetime(year=date_time.year, month=date_time.month, day=date_time.day, hour=date_time.hour,
                                  minute=date_time.minute, second=date_time.second)
    date_time = timezone_converter(date_time, timezone, "UTC")
    # print(date_time, datetime.datetime.utcnow(),datetime.datetime.now())
    altitude_deg = get_altitude(round(latitude, 7), round(longitude, 7), date_time)
    radiation_value = radiation.get_radiation_direct(date_time,
                                                     altitude_deg) * 1000  # * 1000 to get values in milli
    return radiation_value

