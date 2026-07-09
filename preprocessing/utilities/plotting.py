import os
import numpy as np
import pandas as pd
from warnings import warn
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from numpy.polynomial.polynomial import polyfit
from plotly.subplots import make_subplots
from scipy.stats import kde
from scipy.stats.stats import pearsonr
from scipy.stats import sem as SEM
from scipy.ndimage import center_of_mass, median_filter
from scipy.sparse import issparse, spdiags, coo_matrix, csc_matrix
from skimage.measure import find_contours
import sys
from tempfile import NamedTemporaryFile
from typing import Any, Optional


#### spatial roi functions ##### taken from CaImAn visualization
def com(A: np.ndarray, d1: int, d2: int, d3: Optional[int] = None) -> np.array:
    """Calculation of the center of mass for spatial components

     Args:
         A:   np.ndarray
              matrix of spatial components (d x K)

         d1:  int
              number of pixels in x-direction

         d2:  int
              number of pixels in y-direction

         d3:  int
              number of pixels in z-direction

     Returns:
         cm:  np.ndarray
              center of mass for spatial components (K x 2 or 3)
    """

    if 'csc_matrix' not in str(type(A)):
        A = scipy.sparse.csc_matrix(A)

    if d3 is None:
        Coor = np.matrix([np.outer(np.ones(d2), np.arange(d1)).ravel(),
                          np.outer(np.arange(d2), np.ones(d1)).ravel()],
                         dtype=A.dtype)
    else:
        Coor = np.matrix([
            np.outer(np.ones(d3),
                     np.outer(np.ones(d2), np.arange(d1)).ravel()).ravel(),
            np.outer(np.ones(d3),
                     np.outer(np.arange(d2), np.ones(d1)).ravel()).ravel(),
            np.outer(np.arange(d3),
                     np.outer(np.ones(d2), np.ones(d1)).ravel()).ravel()
        ],
                         dtype=A.dtype)

    cm = (Coor * A / A.sum(axis=0)).T
    return np.array(cm)

def get_contours(A, dims, thr=0.9, thr_method='nrg', swap_dim=False):
    """Gets contour of spatial components and returns their coordinates

     Args:
         A:   np.ndarray or sparse matrix
                   Matrix of Spatial components (d x K)

             dims: tuple of ints
                   Spatial dimensions of movie (x, y[, z])

             thr: scalar between 0 and 1
                   Energy threshold for computing contours (default 0.9)

             thr_method: [optional] string
                  Method of thresholding:
                      'max' sets to zero pixels that have value less than a fraction of the max value
                      'nrg' keeps the pixels that contribute up to a specified fraction of the energy

     Returns:
         Coor: list of coordinates with center of mass and
                contour plot coordinates (per layer) for each component
    """

    if 'csc_matrix' not in str(type(A)):
        A = csc_matrix(A)
    d, nr = np.shape(A)
    # if we are on a 3D video
    if len(dims) == 3:
        d1, d2, d3 = dims
        x, y = np.mgrid[0:d2:1, 0:d3:1]
    else:
        d1, d2 = dims
        x, y = np.mgrid[0:d1:1, 0:d2:1]

    coordinates = []

    # get the center of mass of neurons( patches )
    cm = com(A, *dims)

    # for each patches
    for i in range(nr):
        pars:dict = dict()
        # we compute the cumulative sum of the energy of the Ath component that has been ordered from least to highest
        patch_data = A.data[A.indptr[i]:A.indptr[i + 1]]
        indx = np.argsort(patch_data)[::-1]
        if thr_method == 'nrg':
            cumEn = np.cumsum(patch_data[indx]**2)
            if len(cumEn) == 0:
                pars = dict(
                    coordinates=np.array([]),
                    CoM=np.array([np.NaN, np.NaN]),
                    neuron_id=i + 1,
                )
                coordinates.append(pars)
                continue
            else:
                # we work with normalized values
                cumEn /= cumEn[-1]
                Bvec = np.ones(d)
                # we put it in a similar matrix
                Bvec[A.indices[A.indptr[i]:A.indptr[i + 1]][indx]] = cumEn
        else:
            if thr_method != 'max':
                warn("Unknown threshold method. Choosing max")
            Bvec = np.zeros(d)
            Bvec[A.indices[A.indptr[i]:A.indptr[i + 1]]] = patch_data / patch_data.max()

        if swap_dim:
            Bmat = np.reshape(Bvec, dims, order='C')
        else:
            Bmat = np.reshape(Bvec, dims, order='F')
        pars['coordinates'] = []
        # for each dimensions we draw the contour
        for B in (Bmat if len(dims) == 3 else [Bmat]):
            vertices = find_contours(B.T, thr)
            # this fix is necessary for having disjoint figures and borders plotted correctly
            v = np.atleast_2d([np.nan, np.nan])
            for _, vtx in enumerate(vertices):
                num_close_coords = np.sum(np.isclose(vtx[0, :], vtx[-1, :]))
                if num_close_coords < 2:
                    if num_close_coords == 0:
                        # case angle
                        newpt = np.round(vtx[-1, :] / [d2, d1]) * [d2, d1]
                        vtx = np.concatenate((vtx, newpt[np.newaxis, :]), axis=0)
                    else:
                        # case one is border
                        vtx = np.concatenate((vtx, vtx[0, np.newaxis]), axis=0)
                v = np.concatenate(
                    (v, vtx, np.atleast_2d([np.nan, np.nan])), axis=0)

            pars['coordinates'] = v if len(
                dims) == 2 else (pars['coordinates'] + [v])
        pars['CoM'] = np.squeeze(cm[i, :])
        pars['neuron_id'] = i + 1
        coordinates.append(pars)
    return coordinates


##### plotly figure functions ######
def plotMeanData(
    agg_data,
    groupby,
    plot_var,
    datapoint_var='Mouse',
    colors=["steelblue", "darkred"],
    plot_mode='bar',
    mean_line_color='black',
    marker_pattern_shape='',
    plot_datapoints=False,
    plot_datalines=False,
    y_range=None,
    y_title=None,
    text_size=15,
    font_family='Arial',
    plot_title=None,
    opacity=0.8,
    tick_angle=0,
    plot_width=3,
    plot_height=3,
    save_path=None,
    format='svg',
    dpi=300,
    plot_scale=1,
    datapoint_settings=dict(color="black", opacity=0.4, size=10,width= 2,edgecolor='black'),
    dataline_settings=dict(width=2,color="black"), 
    xlabel_font_size=8,
    ylabel_font_size=8,
    tick_label_fontsize=8,
    title_font_size=10,
    axis_linewidth=1):
    """
    Parameters
    ==========
    agg_data : pandas dataframe
        aggregated pandas data frame where each mouse occupies one row
    groupby : str
        either 'ExpGroup' or 'Context' for what you'd like the data split up by
    plot_var : str
        column name to plot
    datapoint_var : str
        column name of individual datapoint variable. Default is 'Mouse'
    colors : list
        list of colors you would like used in plotting
    plot_mode : str
        one of 'bar' or 'point' where means are represented as bars or as points
    plot_datapoints, plot_datalines : boolean
        whether or not to plot individual datapoints, or individual datalines. Defaults are False.
    y_range : tuple
        tuple or min and max values of plot
    y_title : str
        y-axis title. Default is None.
    text_size : int
        size of text to use in the plot. Default is 20.
    font_family : str
        font to use for all plot labels
    plot_title : str
        master title for the graph. Default is None.
    opacity : float
        opacity value for bars. Default is 0.8.
    tick_angle : int
        angle that text on x-axis is displayed at. Default is 45.
    plot_width, plot_height : int
        width and height of the entire plot. Defaults are 350 & 500, respectively.
    save_path : boolean
        an optional file path to save the plot. If save_path=None, plot will not be saved. Default is None.
    plot_scale : int
         how high of a resolution to save the plot as. Default is 5.
    datapoint_marker_settings : dict
        dictionary of marker settings for datapoints, e.g., {'color': 'red', 'size': 10, 'line_width': 2}
    dataline_marker_settings : dict
        dictionary of marker settings for datalines, e.g., {'color': 'blue', 'line_width': 3}
    xlabel_font_size : int or None
        font size for x-axis labels
    ylabel_font_size : int or None
        font size for y-axis labels
    title_font_size : int or None
        font size for the plot title
    axis_linewidth : int or None
        width of the axes lines
    """

    if plot_title == None:
        plot_title = groupby

    means = agg_data[[groupby, plot_var]].groupby(groupby).mean()[plot_var].sort_index()
    sems = agg_data[[groupby, plot_var]].groupby(groupby).sem()[plot_var].sort_index()
    names = means.index.values

    fig = go.Figure()
    if plot_mode == 'bar':
        fig.add_trace(
            go.Bar(
                x=names,
                y=means.values,
                error_y=dict(type="data", array=sems.values, visible=True,width=10),
                marker_color=colors,
                marker=dict(line=dict(width=1, color="black"), opacity=opacity),
                marker_pattern_shape=marker_pattern_shape
            )
        )
    elif (plot_mode == 'point') & (plot_datalines):
        fig.add_trace(
            go.Scatter(
                x=names,
                y=means.values,
                error_y=dict(type="data", array=sems.values, visible=True,width=1),
                mode='lines+markers',
                marker_color=colors,
                marker=dict(size=15, line=dict(width=1, color="black"), opacity=opacity),
                line=dict(color=mean_line_color, width=3)
            )
        )
    elif plot_mode == 'point':
        fig.add_trace(
            go.Scatter(
                x=names,
                y=means.values,
                error_y=dict(type="data", array=sems.values, visible=True,width=1),
                mode='markers',
                marker_color=colors,
                marker=dict(size=15, line=dict(width=1, color=mean_line_color), opacity=opacity),
            )
        )
    else:
        raise Exception("Invalid plot_mode. Must be one of 'bar' or 'point'.")

    if plot_datapoints:
        for sub in agg_data[datapoint_var].unique():
            sub_data = agg_data[agg_data[datapoint_var] == sub]
            marker_settings = datapoint_settings if datapoint_settings is not None else {}
            fig.add_trace(
                go.Scatter(
                    x=sub_data[groupby].values,
                    y=sub_data[plot_var].values,
                    mode="markers",
                    marker=dict(
                        color=marker_settings.get('color', 'black'),
                        size=marker_settings.get('size', 6),
                        line=dict(width=marker_settings.get('width', 0),color=marker_settings.get('edge_color','black')),
                        opacity=marker_settings.get('opacity', 1)
                    ),
                    name=str(sub),
                )
            )

    if plot_datalines:
        for line in agg_data[datapoint_var].unique():
            line_data = agg_data[agg_data[datapoint_var] == line]
            line_data = line_data.iloc[line_data[groupby].argsort(), :]
            line_settings = dataline_settings if dataline_settings is not None else {}
            marker_settings = datapoint_settings if datapoint_settings is not None else {}

            fig.add_trace(
                go.Scatter(
                    x=line_data[groupby].values,
                    y=line_data[plot_var].values,
                    mode="lines+markers",
                    line=dict(
                        width=line_settings.get('width', 1),
                        color=line_settings.get('color', 'black')),
                    marker=dict(
                            line=dict(width=marker_settings.get('width', 1),  # Border width for markers
                            color=marker_settings.get('edgecolor', 'black')),
                    color=marker_settings.get('color', 'black'),
                    size=marker_settings.get('size', 6),
                    opacity=marker_settings.get('opacity', 0.4)),
                    name=str(line),
                )
            )

    fig.update_layout(
        dragmode="pan",
        yaxis_title=y_title,
        font=dict(size=text_size, family=font_family),
        title_text=plot_title,
        autosize=False,
        width=plot_width*dpi,
        height=plot_height*dpi,
        template="simple_white",
        showlegend=False,
    )

    fig.update_xaxes(title_font=dict(size=xlabel_font_size), linewidth=axis_linewidth or 2,tickangle=tick_angle,tickfont=dict(size=tick_label_fontsize))
    fig.update_yaxes(title_font=dict(size=ylabel_font_size), linewidth=axis_linewidth or 2,range=y_range)
    fig.update_layout(title_font=dict(size=title_font_size))

    if tick_label_fontsize is None:
        tick_label_fontsize=text_size
    #fig.update_xaxes(tickangle=tick_angle,tickfont=dict(size=tick_label_fontsize))
    #fig.update_yaxes(range=y_range)
    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.mkdir(os.path.dirname(save_path))
        if save_path.split('.')[-1] == 'html':
            fig.write_html(save_path)
        elif save_path.split('.')[-1] != 'eps':
            fig.write_image(save_path, scale=plot_scale)
        else:
            fig.write_image(save_path, format=save_path.split('.')[-1])

    config = {
        'scrollZoom': True,
        'toImageButtonOptions': {
            'format': format,
            'filename': 'custom_image',
            'height': plot_height*dpi,
            'width': plot_width*dpi,
            'scale': plot_scale
        }
    }
    #fig.show(config=config)
    return fig

def plotAcrossGroupsJoe(
    agg_data,
    groupby,
    separateby,
    plot_var,
    colors,
    title,
    datapoint_var="Mouse",
    y_range=None,
    plot_mode='bar',
    mean_line_color='black',
    marker_pattern_shape='',
    plot_datapoints=False,
    plot_datalines=False,
    y_title=None,
    text_size=20,
    font_family='Arial',
    opacity=0.8,
    plot_width=600,
    plot_height=600,
    tick_angle=45,
    scale_y=True,
    h_spacing=0.2,
    save_path=None,
    plot_scale=5):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import os

    # Ensure proper categorical ordering for separateby
    if not isinstance(agg_data[separateby].dtype, pd.CategoricalDtype):
        agg_data[separateby] = pd.Categorical(agg_data[separateby], ordered=True)

    subplot_titles = agg_data[separateby].cat.categories
    fig = make_subplots(
        cols=len(subplot_titles), subplot_titles=subplot_titles, horizontal_spacing=h_spacing, shared_yaxes=scale_y
    )

    for i, val in enumerate(subplot_titles):
        sub_data = agg_data[agg_data[separateby] == val]
        group_means = sub_data[[groupby, plot_var]].groupby(groupby).mean()[plot_var]
        group_sems = sub_data[[groupby, plot_var]].groupby(groupby).sem()[plot_var]
        xlabels = group_means.index.values

        for j, group in enumerate(xlabels):
            if plot_mode == 'bar':
                fig.add_trace(
                    go.Bar(
                        x=[group],
                        y=[group_means.loc[group]],
                        error_y=dict(type="data", array=[group_sems.loc[group]], visible=True),
                        marker_color=colors[j % len(colors)],
                        marker=dict(line=dict(width=1, color="black"), opacity=opacity),
                        marker_pattern_shape=marker_pattern_shape,
                        name=str(group)
                    ),
                    row=1,
                    col=i + 1,
                )
            elif plot_mode == 'point':
                fig.add_trace(
                    go.Scatter(
                        x=[group],
                        y=[group_means.loc[group]],
                        error_y=dict(type="data", array=[group_sems.loc[group]], visible=True),
                        mode='markers',
                        marker=dict(size=15, color=colors[j % len(colors)], line=dict(width=1, color="black"), opacity=opacity),
                        name=str(group)
                    ),
                    row=1,
                    col=i + 1,
                )

        if plot_datapoints:
            for point in sub_data[datapoint_var].unique():
                point_data = sub_data[sub_data[datapoint_var] == point]
                fig.add_trace(
                    go.Scattergl(
                        x=point_data[groupby].values,
                        y=point_data[plot_var].values,
                        mode="markers",
                        marker=dict(color="black", opacity=0.4),
                        name=str(point),
                    ),
                    row=1,
                    col=i + 1,
                )
        if plot_datalines:
            for line in sub_data[datapoint_var].unique():
                line_data = sub_data[sub_data[datapoint_var] == line]
                line_data = line_data.iloc[line_data[groupby].argsort(), :]
                fig.add_trace(
                    go.Scatter(
                        x=line_data[groupby].values,
                        y=line_data[plot_var].values,
                        mode="lines+markers",
                        line=dict(width=1),
                        marker=dict(color="black", opacity=0.4),
                        name=str(line),
                    ),
                    row=1,
                    col=i + 1,
                )

    fig.add_hline(y=0, row=1, col='all', line_width=1, opacity=1, line_color='black')
    fig.update_layout(
        dragmode="pan",
        yaxis_title=y_title,
        font=dict(size=text_size, family=font_family),
        title_text=title,
        autosize=False,
        width=plot_width,
        height=plot_height,
        template="simple_white",
        showlegend=False,
    )
    if tick_angle is not None:
        fig.update_xaxes(tickangle=tick_angle)
    if y_range is not None:
        fig.update_yaxes(range=y_range)
    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.mkdir(os.path.dirname(save_path))
        if save_path.split('.')[-1] == 'html':
            fig.write_html(save_path)
        elif save_path.split('.')[-1] != 'eps':
            fig.write_image(save_path, scale=plot_scale)
        else:
            fig.write_image(save_path, format=save_path.split('.')[-1])
    config = {
        'scrollZoom': True,
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'custom_image',
            'height': plot_height,
            'width': plot_width,
            'scale': plot_scale
        }
    }
    fig.show(config=config)

def plotMeanDataJoe(
    agg_data,
    groupby,
    plot_var,
    datapoint_var='Mouse',
    colors=["steelblue", "darkred"],
    plot_mode='bar',
    mean_line_color='black',
    marker_pattern_shape='',
    plot_datapoints=False,
    plot_datalines=False,
    y_range=None,
    y_title=None,
    text_size=20,
    font_family='Arial',
    plot_title=None,
    opacity=0.8,
    tick_angle=45,
    plot_width=350,
    plot_height=500,
    save_path=None,
    plot_scale=5
):
    """
    Parameters
    ==========
    agg_data : pandas dataframe
        aggregated pandas data frame where each mouse occupies one row
    groupby : str
        either 'ExpGroup' or 'Context' for what you'd like the data split up by
    plot_var : str
        column name to plot
    datapoint_var : str
        column name of individual datapoint variable. Default is 'Mouse'
    colors : list
        list of colors you would like used in plotting
    plot_mode : str
        one of 'bar' or 'point' where means are represented as bars or as points
    plot_datapoints, plot_datalines : boolean
        whether or not to plot individual datapoints, or individual datalines. Defaults are False.
    y_range : tuple
        tuple or min and max values of plot
    y_title : str
        y-axis title. Default is None.
    text_size : int
        size of text to use in the plot. Default is 20.
    font_family : str
        font to use for all plot labels
    plot_title : str
        master title for the graph. Default is None.
    opacity : float
        opacity value for bars. Default is 0.8.
    tick_angle : int
        angle that text on x-axis is displayed at. Default is 45.
    plot_width, plot_height : int
        width and height of the entire plot. Defaults are 350 & 500, respectively.
    save_path : boolean
        an optional file path to save the plot. If save_path=None, plot will not be saved. Default is None.
    plot_scale : int
         how high of a resolution to save the plot as. Default is 5.
    """

    if plot_title == None:
        plot_title = groupby

    means = agg_data[[groupby, plot_var]].groupby(groupby).mean()[plot_var].sort_index()
    sems = agg_data[[groupby, plot_var]].groupby(groupby).sem()[plot_var].sort_index()
    names = means.index.values

    fig = go.Figure()
    if plot_mode == 'bar':
        fig.add_trace(
            go.Bar(
                x=names,
                y=means.values,
                error_y=dict(type="data", array=sems.values, visible=True),
                marker_color=colors,
                marker=dict(line=dict(width=1, color="black"), opacity=opacity),
                marker_pattern_shape=marker_pattern_shape
            )
        )
    elif (plot_mode == 'point') & (plot_datalines):
        fig.add_trace(
            go.Scatter(
                x=names,
                y=means.values,
                error_y=dict(type="data", array=sems.values, visible=True),
                mode='lines+markers',
                marker_color=colors,
                marker=dict(size=15, line=dict(width=1, color="black"), opacity=opacity),
                line=dict(color=mean_line_color, width=3)
            )
        )
    elif plot_mode == 'point':
        fig.add_trace(
            go.Scatter(
                x=names,
                y=means.values,
                error_y=dict(type="data", array=sems.values, visible=True),
                mode='markers',
                marker_color=colors,
                marker=dict(size=15, line=dict(width=1, color=mean_line_color), opacity=opacity),
            )
        )
    else:
        raise Exception("Invalid plot_mode. Must be one of 'bar' or 'point'.")
    if plot_datapoints:
        for sub in agg_data[datapoint_var].unique():
            sub_data = agg_data[agg_data[datapoint_var] == sub]
            fig.add_trace(
                go.Scatter(
                    x=sub_data[groupby].values,
                    y=sub_data[plot_var].values,
                    mode="markers",
                    marker=dict(color="black", opacity=0.4),
                    line = dict(color="black",width=5),
                    name=str(sub),
                )
            )
    if plot_datalines:
        for line in agg_data[datapoint_var].unique():
            line_data = agg_data[agg_data[datapoint_var] == line]
            line_data = line_data.iloc[line_data[groupby].argsort(),:]
            fig.add_trace(
                go.Scatter(
                    x=line_data[groupby].values,
                    y=line_data[plot_var].values,
                    mode="lines+markers",
                    line=dict(width=1),
                    marker=dict(color="black", opacity=0.4),
                    name=str(line),
                )
            )
    fig.update_layout(
        dragmode="pan",
        yaxis_title=y_title,
        font=dict(size=text_size, family=font_family),
        title_text=plot_title,
        autosize=False,
        width=plot_width,
        height=plot_height,
        template="simple_white",
        showlegend=False,
    )
    if tick_angle is not None:
        fig.update_xaxes(tickangle=tick_angle)
    fig.update_yaxes(range=y_range)
    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.mkdir(os.path.dirname(save_path))
        if save_path.split('.')[-1] == 'html':
            fig.write_html(save_path)
        elif save_path.split('.')[-1] != 'eps':
            fig.write_image(save_path, scale=plot_scale)
        else:
            fig.write_image(save_path, format=save_path.split('.')[-1])
    config = {
        'scrollZoom':True,
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'custom_image',
            'height': plot_height,
            'width': plot_width,
            'scale':plot_scale
            }
            }
    fig.show(config=config)
    return fig
