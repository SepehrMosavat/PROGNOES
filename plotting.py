import numpy as np
import time
import matplotlib.pyplot as plt
import math
from general_functions import max_power_point, interp1d_curve_fit_function, get_metadata, \
    extract_z_from_hdf5, fit_the_curve
import warnings

warnings.filterwarnings("ignore", module='pvlib')
warnings.filterwarnings("ignore", module='pysolar')


def plotting_3d_plot(path):
    """
    This function plots the curves in a HDF5 file in 3D for better analysis.
    :param path:
    Path of the HDF5 File.
    :return:
    Return a message that process ended successfully.
    """
    harvesting_conditions, plt_meta_data = get_metadata(path)
    from collections import OrderedDict

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.grid(True)

    plt.title(plt_meta_data['title'], fontsize=9, y=0.99)
    ax.set_ylabel(plt_meta_data['ylabel'])
    ax.set_xlabel(plt_meta_data['xlabel'])
    ax.set_zlabel(plt_meta_data['zlabel'])
    plt.suptitle(plt_meta_data['info'], fontsize=5.5, y=0.99)

    df = extract_z_from_hdf5(path)
    df.columns = ["X", "Y", "Z", "Curve"]

    for item in list(OrderedDict.fromkeys(df['Curve'])):
        new_df = df[df['Curve'] == item]
        ax.plot(list(new_df['X']), list(new_df['Z']), list(new_df['Y']))

    fileName = (path.split('/'))[-1] + '---' + harvesting_conditions['Date'] + '---Simple' + '.png'
    plt.savefig(fileName, bbox_inches="tight", pad_inches=1, transparent=True, facecolor="w", edgecolor='w',
                orientation='landscape', dpi=300)
    plt.show()

    return "Process finished successfully"


def plotting_3d_plot_surface(path):
    """
        This function plots the curves in a HDF5 file in 3D Surface for better analysis.
        :param path:
        Path of the HDF5 File.
        :return:
        Return a message that process ended successfully.
        """
    harvesting_conditions, plt_meta_data = get_metadata(path)

    df = extract_z_from_hdf5(path)
    df.columns = ["X", "Y", "Z", "Curve"]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    subplot = ax.plot_trisurf(df['X'], df['Z'], df['Y'], cmap=plt.cm.jet, linewidth=1)
    fig.colorbar(subplot, pad=0.15)

    plt.title(plt_meta_data['title'], fontsize=9, y=0.99)
    ax.set_ylabel(plt_meta_data['ylabel'])
    ax.set_xlabel(plt_meta_data['xlabel'])
    ax.set_zlabel(plt_meta_data['zlabel'])
    plt.suptitle(plt_meta_data['info'], fontsize=5.5, y=0.99)

    fileName = (path.split('/'))[-1] + '---' + harvesting_conditions['Date'] + '---Surface' + '.png'
    plt.savefig(fileName, bbox_inches="tight", pad_inches=1, transparent=True, facecolor="w", edgecolor='w',
                orientation='landscape', dpi=300)

    plt.show()
    return "Process finished successfully"


def plotting_3d_interactive(path):
    """
    This function plots the curves in a HDF5 file in 3D Mode in HTML using plotly for better analysis.
    :param path:
    Path of the HDF5 File.
    :return:
    Return a message that process ended successfully.
    """
    import plotly.express as px
    import plotly.graph_objects as go
    harvesting_conditions, plt_meta_data = get_metadata(path)

    df = extract_z_from_hdf5(path)

    caption = plt_meta_data['info']
    caption = caption.replace("\n", "<br>")
    caption = '<b>' + caption + '</b>'

    fig = px.line_3d(df, x="Time (s)", y="Solar Cell Voltage (V)", z="Solar Cell Current (uA)", color='Curve',
                     title=plt_meta_data['title'])
    fig.update_layout(
        title=go.layout.Title(text=caption, font=dict(family="Arial Black", size=18, color="Black")),
        legend=dict(font=dict(family="Arial Black", size=18)), font=dict(family="Arial Black", size=10, color="Black"))
    fig.update_yaxes(tickfont_family="Arial Black")
    fig.update_xaxes(tickfont_family="Arial Black")
    fig.show()
    return "Process finished successfully"


def iv_curve_plotting(v_value_list, i_value_list, title, harvesting_conditions, dot_original=False, plt_original=True,
                      curve_fit=False, draw_power_points=False, save_figure=False):
    """
    This functions plots a curve in 2D mode using the given parameters.
    :param v_value_list:
    List of the values of 'v' in the curve.
    :param i_value_list:
    List of the values of 'i' in the curve.
    :param title:
    The curve number in format curve#. For example curve17
    :param harvesting_conditions:
    This is created with get_metadata function and the strings contains useful information which shown in above of plot.
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
    :param save_figure:
    If it is True the plot will be saved in a PNG file. Default is False.
    :return:
    Return a message that process ended successfully.
    """
    v = np.array(v_value_list)
    i = np.array(i_value_list)

    if dot_original:
        plt.plot(v, i, 'bo', label="y-original")

    if plt_original:
        plt.plot(v_value_list, i_value_list, color='dodgerblue', label="y-original")
        VMP, IMP, PMP, power_points = max_power_point(v_value_list, i_value_list)
        plt.plot(VMP, IMP, color='dodgerblue', marker='.',
                 label='PMP: ' + str(round(VMP, 2)) + ' x ' + str(round(IMP, 2)) + ' = ' + str(round(PMP, 2)))

        if curve_fit:
            vv, ii, check = fit_the_curve(v, i)

            if check:
                plt.plot(vv, ii, color='orange', label="exponential")
                VMP, IMP, PMP, power_points = max_power_point(vv, ii)
                plt.plot(VMP, IMP, color='red', marker='o',
                         label='MAX(' + str(round(VMP, 3)) + ',' + str(round(IMP / 1000, 3)) + '): ' + str(
                             round(round(VMP, 3) * round(IMP / 1000, 3), 5)))
                if draw_power_points:
                    plt.plot(vv, power_points, color='peru', label='Power_Points')
                    plt.plot(vv[np.argmax(power_points)], PMP, color='sienna', marker='o',
                             label='PMP: ' + str(math.ceil((vv[np.argmax(power_points)] * PMP) * 100) / 100))
            else:
                X_interp1d, yfited = interp1d_curve_fit_function(v, i)
                plt.plot(X_interp1d, np.sort(yfited)[::-1], color='royalblue', label="interp1d_curve_fit")
                # plt.plot(X_interp1d, np.sort(yfited)[::-1], marker='o', color='royalblue', label="dots")
                VMP, IMP, PMP, power_points = max_power_point(X_interp1d, yfited)
                plt.plot(VMP, IMP, color='red', marker='o',
                         label='MAX(' + str(round(VMP, 3)) + ',' + str(round(IMP / 1000, 3)) + '): ' + str(
                             round(round(VMP, 3) * round(IMP / 1000, 3), 5)))

                if draw_power_points:
                    plt.plot(X_interp1d, power_points, color='peru', label='Power_Points')
                plt.plot(X_interp1d[np.argmax(power_points)], PMP, color='sienna', marker='o',
                         label='PMP: ' + str(math.ceil((X_interp1d[np.argmax(power_points)] * PMP) * 100) / 100))

    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(True)
    plt.title("Solar Cell IV Characteristics: " + title, fontsize=9, y=0.99)
    plt.xlabel("Solar Cell Voltage (V)")
    plt.ylabel("Solar Cell Current (uA)")
    plt.suptitle(harvesting_conditions, fontsize=5.5, y=0.99)
    plt.draw()
    plt.pause(0.1)
    plt.clf()
    time.sleep(0.1)
    if save_figure:
        plt.savefig("Current_Curve_2D_" + str(time.time()) + ".png", bbox_inches="tight", pad_inches=1,
                    transparent=True,
                    facecolor="w", edgecolor='w', orientation='landscape', dpi=300)

    return "Curve plotted successfully."
