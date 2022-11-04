import datetime
import time
import warnings

warnings.filterwarnings("ignore", module='pvlib')


def Definitions():
    '''
    This is just for reference, to make understanding the values easier.
    :return: no return
    '''
    harvesting_conditions = {'Date': '28.10.2021', 'Start Time (Local Timezone)': '13:5:12.113325',
                             'End Time (Local Timezone)': '13:5:44.136401', 'Indoor/Outdoor': 'indoor',
                             'Light Intensity (Lux)': '50', 'Weather Condition': 'sunny', 'Country': 'Germany',
                             'City': 'Essen'}

    prognoes_conditions = {'start_time': '1635419112.113325', 'end_time': '1635419144.136401',
                           'thermometer_temperature': '25', 'address': 'Kuglerstraße, Essen,45144',
                           'latitude': '51.455643', 'longitude': '7.011555', 'solar_cell': 'SM141K08LV'}

    weather_conditions = {'timestamp': '2021-10-28 13:05:00+02:00', 'temperature': '14.89166158040367',
                          'total_clouds': '0.0', 'precipitation': '0.0', 'snowDepth': '0.0',
                          'windSpeed': '3.3372151031093105', 'humidity': '60.86004396279653',
                          'pressure': '1009.7519661458332', 'pysolar_cs_ghi': '812.6986785351758',
                          'IandP_cs_ghi': '377.076855191444', 'cs_ghi': '378.9005272810658',
                          'cs_dni': '681.4576729375241', 'cs_dhi': '88.41075233828263', 'cn_ghi': '388.0931582970756',
                          'cn_dni': '705.590014943638', 'cn_dhi': '86.92211119081557'}

    irradiation_weather_conditions = {'timestamp': '2021-10-28 13:05:00+02:00', 'temp_air': '16.942823452419702',
                                      'wind_speed': '6.558002063954822', 'total_clouds': '9.241666893164316',
                                      'low_clouds': '0.0', 'mid_clouds': '0.0', 'high_clouds': '8.841666579246521',
                                      'GFS_ghi': '260.13366911146375', 'cs_ghi': '356.13965629902225',
                                      'cs_dni': '552.7575823824292', 'cs_dhi': '120.51177788358638',
                                      'cn_ghi': '345.2492839668864', 'cn_dni': '562.1963700396528',
                                      'cn_dhi': '105.2837716180395'}

    pass


def coordinates():
    '''
        Some coordinates of the places used in the project.
    '''
    latitude, longitude, city, hour, minute, PVGIS_source, timezone, altitude = 51.455643, 7.011555, "Essen", -2, 0, "PVGIS-SARAH", "Europe/Berlin", 0
    # latitude, longitude, city, hour, minute, PVGIS_source, timezone, altitude = 52.1948, 13.180, "Berlin", -2, 0, "PVGIS-SARAH", "Europe/Berlin", 0
    # latitude, longitude, city ,hour, minute, PVGIS_source, timezone, altitude = 36.723557, -120.059879, "Kerman-California", 8,0,"PVGIS-NSRDB","America/Tijuana", 0
    # latitude, longitude, city, hour, minute , PVGIS_source, timezone, altitude= 71.030223, 27.831297, "Mehamn Lufthaven", -3, 0, "PVGIS-COSMO","Europe/Vilnius", 0
    # latitude, longitude, city, hour, minute, PVGIS_source, timezone, altitude = 27.2012, 60.686582, "Iranshahr", -4, 30,"PVGIS-SARAH","Asia/Tehran", 0
    # latitude, longitude, city, hour, minute, PVGIS_source, timezone, altitude = -33.86882, 151.209295, "Sydney", -10, 0, "PVGIS-SARAH","Australia/Sydney", 0
    # latitude, longitude, city, hour, minute, PVGIS_source, timezone, altitude = 52.1948, 13.180, "Berlin", -2, 0, "PVGIS-SARAH", "Europe/Berlin", 0


def pysolar_funcs():
    '''
        An example of the use of pysolar fucntion.
    '''
    import datetime
    from pysolar_functions import pysolar_query
    print(pysolar_query(latitude=51.455643, longitude=7.011555, date_time=datetime.datetime.now().timestamp(),
                        timezone="Europe/Berlin"))


def mowesta_funcs():
    '''
        An Example to use mowesta fucntions.
    '''
    from mowesta import mowesta_forecast_all, mowesta_day_minute
    mowesta_forecast_all()
    print(mowesta_day_minute("Kuglerstr, Essen, 45144")[1])


def hdf5_funcs():
    """
    This function gives an examples of functions implemented in 'hdf5.py'.
    Please make sure, that you copied the folder 'solar_cell_profiles' from /data/examples to /data
    Please just run only one of the functions bellow. Some functions like hdf5_reader runs endless.
    To understand the functionality of functions refer to th respected function description.
    :return:
    Function provides no return
    """
    from hdf5 import hdf5_reader, dataset_to_hdf5, compare_dataset

    path = 'data/SOCRAETES/12_23_2021/SOCRAETES_SM141K07L_211223_133213.hdf5'
    hdf5_reader(path, dot_original=False, plt_original=True, curve_fit=True, draw_power_points=False)

    start = datetime.datetime(year=2021, month=10, day=28, hour=12, minute=30, second=10)
    dataset_to_hdf5(solar_cell_model="SM141K08LV", address="Kuglerstraße, Essen,45144", latitude=51.455643,
                    longitude=7.011555,
                    start_time=start,
                    end_time=start + datetime.timedelta(minutes=4),
                    metadata_time=start, seconds_intervals=1, curve_pause=0.1,
                    verbose_show=False, plt_show=True, export_to_hdf=False, original_dataset=None)

    compare_dataset(path=path)



def plotting_funcs():
    '''
        Some examples of teh provided functions in plotting.py
    '''
    from plotting import plotting_3d_plot, plotting_3d_plot_surface, plotting_3d_interactive
    from pvlib_functions import datasets_from_csv

    path = 'data/SOCRAETES/11_06_2021/SOCRAETES_SM141K08LV_211106_134510.hdf5'
    plotting_3d_plot(path)
    plotting_3d_plot_surface(path)
    plotting_3d_interactive(path)
    datasets_from_csv("data/GFS_archive/PROGNOES_GFS_45144_211225.csv", 'data/mowesta_archive/PROGNOES_MOWESTA_45144_211225.csv')


class evaluation():
    '''
    This class is used to generate various example plots and data.
    '''

    def profile_sc_temperature_coefficient(self):
        '''
            This functions shows the effect of temperature of solar cell surface of the generated curves.
        '''
        from solar_cell_class import solar_cell
        solar_cell = solar_cell()
        solar_cell.SC_obj_from_csv('data/solar_cell_profiles/Calculated_SM141K07L.csv')
        # solar_cell.SC_obj_from_csv('data/solar_cell_profiles/Clear_SM141K07L.csv')
        # solar_cell.start_profiling(plt_show=True)
        cases = [(364.9, 35), (364.9, 30), (346.9, 26.5), (346.9, 20), (346.9, 15)]
        solar_cell.curve(case=cases, plt_show=True, save_figure=True)

    def eval_only_irradiation_data(self):  #######checkkkk?!
        '''
            This function calculate a minimal approximation of available solar energy
            It doesn't need any weather information.
        '''
        from hdf5 import compare_dataset
        path = 'data/SOCRAETES/10_28_2021/SOCRAETES_SM141K08LV_211028_130753.hdf5'  # ISC can be 58.6mA
        path = 'data/SOCRAETES/10_28_2021/SOCRAETES_SM141K07L_211028_130328.hdf5'  # ISC can be 58.6ma
        # path = 'data/SOCRAETES/10_28_2021/SOCRAETES_SM111K04L_211028_131239.hdf5'  # ISC can be 46.7mA

        compare_dataset(path=path)

    '''
    # This function is not available anymore, 
    # because the format of provided dataframes form thredds is changed
    
    def eval_download_dataset(self, next_days=7): #not available
        from hdf5 import dataset_to_hdf5

        dataset_to_hdf5(solar_cell_model="SM141K08LV", address="Kuglerstraße, Essen,45144", latitude=51.455643,
                        longitude=7.011555, start_time=datetime.datetime.now(),
                        end_time=datetime.datetime.now() + datetime.timedelta(days=next_days),
                        metadata_time=datetime.datetime.now(), seconds_intervals=600.0, curve_pause=0.1,
                        verbose_show=False, plt_show=False, export_to_hdf=False, original_dataset=None)
    '''


hdf5_funcs()

#evaluation_examples = evaluation()
#evaluation_examples.profile_sc_temperature_coefficient()
# evaluation_examples.eval_only_irradiation_data()

