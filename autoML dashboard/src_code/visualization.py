
## packages

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_score_target(score, target):
    """ a plot of a score as a funtion of target for a binary classification problem
    parameters:
    -----------
        score: [pd.Series / list/ np.array, shape(n_individuals)] a score vector
        target: [pd.Series / list/ np.array, shape(n_individuals)] a target score
    return:
    -------
        fig: [plotly.graph_objs._figure.Figure] the figure of the distribution
    """
    if isinstance(score, pd.Series):
        score = list(score)
    if not isinstance(target, pd.Series):
        target = list(target)

    ## create a dataframe to ease manipulation
    df = pd.DataFrame({'score': score,
                       'class': target})

    df_1 = df.loc[df.loc[:, 'class'] == 1, :].groupby('score').count()
    df_0 = df.loc[df.loc[:, 'class'] == 0, :].groupby('score').count()

    ## find intersection of 2 proba vectors df_1 and df_2
    set_1 = set(df_1.index)
    set_0 = set(df_0.index)
    set_res = set_0.intersection(set_1)

    ## positive rate vecotr
    df_pos = (df_1.loc[set_res,] / df_0.loc[set_res,])

    trace_1 = go.Bar(x=list(df_1.index), y=list(df_1['class']), name='y_1')
    trace_0 = go.Bar(x=list(df_0.index), y=list(df_0['class']), name='y_0')
    trace_p = go.Scatter(x=list(df_pos.index), y=df_pos['class'], name='positive rate',
                         mode='markers', marker={'size': 4,
                                                 'symbol': 'circle',
                                                 'color': '#2F4F4F'})

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(trace_1, secondary_y=False)
    fig.add_trace(trace_0, secondary_y=False)
    fig.add_trace(trace_p, secondary_y=True)

    fig.update_layout(title={
        'text': "Distribution of target classes as a function of scores",
        'y': 0.90,
        'x': 0.45,
        'font': {'size': 18}},
        #width = 200,
        #height = 400,
        xaxis={'title': "Scores",
               'showline': False},
        yaxis={'title': "occurence",
               'showline': False},
        barmode='stack')

    fig.update_yaxes(title_text="Count", secondary_y=False)
    fig.update_yaxes(title_text="Positive rate", secondary_y=True)

    return fig

def plot_feature_cat(feature_serie, target):
    """ a plot of a categorical feature as a function of target
    parameters:
    -----------
        feature_serie: [pd.Series , shape(n_individuals)] feature vales
        target: [pd.Series , shape(n_individuals)] target values
    return:
    -------
        fig: [plotly.graph_objs._figure.Figure] the figure of the distribution
    """
    unique_values = list(feature_serie.unique())

    positive_weight_list = []

    for i in range(len(unique_values)):
        temp = (target[feature_serie == unique_values[i]] == 1).sum() / (
                    target[feature_serie == unique_values[i]] == 0).sum()
        positive_weight_list.append(temp)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    trace_1 = go.Histogram(x=feature_serie.loc[list(target == 1)],
                           name='y_1 count',
                           opacity=0.4)

    trace_0 = go.Histogram(x=feature_serie.loc[list(target == 0)],
                           name='y_0 count',
                           opacity=0.4)

    trace_2 = go.Scatter(x=unique_values,
                         y=positive_weight_list,
                         name='positive rate',
                         mode='markers',
                         marker={'size': 5,
                                 'symbol': 'circle',
                                 'color': 'red'})

    fig.add_trace(trace_1,  # , name="yaxis data"
                  secondary_y=False)

    fig.add_trace(trace_0,
                  secondary_y=False)

    fig.add_trace(trace_2,
                  secondary_y=True)

    fig.update_layout(
        title={'text': "La Distibution de la variable " + feature_serie.name,
               'y': 0.90,
               'x': 0.5,
               'font': {'family': "Arial",
                        'size': 22}
               },
        xaxis_title_text='Value',
        bargap=0.2,  # gap between bars of adjacent location coordinates
        bargroupgap=0.1  # gap between bars of the same location coordinates ))
    )

    fig.update_yaxes(title_text="Count", secondary_y=False)
    fig.update_yaxes(title_text="Positive ratio", secondary_y=True)

    return fig

def plot_feature_dist(feature_serie, limit_a=None, limit_b=None, autobinx=True, xbins_size=0.5):
    """ a plot of the feature's distribution
    parameters:
    -----------
        feature_serie: [pd.Series , shape(n_individuals)] feature vales
        limit_a: [float] lower limit
        limit_b: [float] upper limit
        autobinx: [boolean] activate the autobin size
        xbins_size: bin size
    return:
    -------
        fig: [plotly.graph_objs._figure.Figure] the figure of the distribution
    """
    if not (limit_a is None) and not (limit_b is None):
        perc_a = (feature_serie <= limit_a).sum() / len(feature_serie)
        perc_b = (feature_serie >= limit_b).sum() / len(feature_serie)

        feature_serie = feature_serie[(feature_serie > limit_a) & (feature_serie < limit_b)]
        title_plot = "La Distibution de la variable {} ( les valeur <= {} ({}%) et les valeurs >= {} ({}%) sont négligées) ".format(
            feature_serie.name, limit_a, round(perc_a * 100, 2), limit_b, round(perc_b * 100, 2))

    elif not (limit_a is None):
        print("a")
        perc_a = (feature_serie <= limit_a).sum() / len(feature_serie)
        print(perc_a)
        feature_serie = feature_serie[feature_serie > limit_a]
        title_plot = "La Distibution de la variable {} (les valeurs > {} ({}%) sont négligées)".format(feature_serie.name, limit_a,
                                                                                                       round(perc_a * 100, 3))
    else:
        title_plot = "La Distibution de la variable {}".format(feature_serie.name)

    trace = go.Histogram(x=feature_serie,
                         xbins={'start': feature_serie.min(),
                                'end': feature_serie.max(),
                                'size': xbins_size},
                         autobinx=autobinx,
                         opacity=0.5)

    fig = go.Figure(data=[trace])

    fig.update_layout(title={'text': title_plot,
                             'y': 0.90,
                             'x': 0.5,
                             'font': {'family': "Arial",
                                      'size': 18}},
                      xaxis_title_text='Values',
                      yaxis_title_text='Count')
    return fig

def plot_feature_dist_target(feature_serie, target, limit_a=None, limit_b=None, autobinx=True, xbins_size=0.5):
    """ a plot of the feature's distribution in regards to the target classes
    parameters:
    -----------
        feature_serie: [pd.Series , shape(n_individuals)] feature vales
        target: [pd.Series , shape(n_individuals)] target values
        limit_a: [float] lower limit
        limit_b: [float] upper limit
        autobinx: [boolean] activate the autobin size
        xbins_size: bin size
    return:
    -------
        fig: [plotly.graph_objs._figure.Figure] the figure of the distribution
    """
    if not (limit_a is None) and not (limit_b is None):
        perc_a = (feature_serie <= limit_a).sum() / len(feature_serie)
        perc_b = (feature_serie >= limit_b).sum() / len(feature_serie)

        variable_serie_0 = feature_serie[(feature_serie > limit_a) & (feature_serie < limit_b) & (target == 0)]
        variable_serie_1 = feature_serie[(feature_serie > limit_a) & (feature_serie < limit_b) & (target == 1)]

        title_plot = "La Distibution de la variable {} ( les valeur <= {} ({}%) et les valeurs >= {} ({}%) sont négligées) ".format(
            feature_serie.name, limit_a, round(perc_a * 100, 2), limit_b, round(perc_b * 100, 2))

    elif not (limit_a is None):

        perc_a = (feature_serie <= limit_a).sum() / len(feature_serie)
        variable_serie_0 = feature_serie[(feature_serie > limit_a) & (target == 0)]
        variable_serie_1 = feature_serie[(feature_serie > limit_a) & (target == 1)]

        title_plot = "La Distibution de la variable {} (les valeurs > {} ({}%) sont négligées)".format(feature_serie.name, limit_a,
                                                                                                       round(perc_a * 100, 3))
    else:
        variable_serie_0 = feature_serie[target == 0]
        variable_serie_1 = feature_serie[target == 1]
        title_plot = "La Distibution de la variable {}".format(feature_serie.name)

    trace_0 = go.Histogram(x=variable_serie_0,
                           name='y_0',
                           xbins={'start': feature_serie.min(),
                                  'end': feature_serie.max(),
                                  'size': xbins_size},
                           autobinx=autobinx,
                           opacity=0.7)

    trace_1 = go.Histogram(x=variable_serie_1,
                           name='y_1',
                           xbins={'start': feature_serie.min(),
                                  'end': feature_serie.max(),
                                  'size': xbins_size},
                           autobinx=autobinx,
                           opacity=0.7)

    fig = go.Figure(data=[trace_0, trace_1])

    fig.update_layout(title={'text': title_plot,
                             'y': 0.90,
                             'x': 0.5,
                             'font': {'family': "Arial",
                                      'size': 18}},
                      xaxis_title_text='Values',
                      yaxis_title_text='Count',
                      barmode='overlay')
    return fig

def boxplot_label(feature_serie, target, limit_a=None, limit_b=None):
    """ a plot of the boxplot of feature in regards to the target classes
    parameters:
    -----------
        feature_serie: [pd.Series , shape(n_individuals)] feature vales
        limit_a: [float] lower limit
        limit_b: [float] upper limit
        autobinx: [boolean] activate the autobin size
        xbins_size: bin size
    return:
    -------
        fig: [plotly.graph_objs._figure.Figure] the figure of the distribution
    """

    if not (limit_a is None) and not (limit_b is None):
        perc_a = (feature_serie <= limit_a).sum() / len(feature_serie)
        perc_b = (feature_serie >= limit_b).sum() / len(feature_serie)

        variable_serie_0 = feature_serie[(feature_serie > limit_a) & (feature_serie < limit_b) & (target == 0)]
        variable_serie_1 = feature_serie[(feature_serie > limit_a) & (feature_serie < limit_b) & (target == 1)]

        title_plot = "La Distibution de la variable {} ( les valeur <= {} ({}%) et les valeurs >= {} ({}%) sont négligées) ".format(
            feature_serie.name, limit_a, round(perc_a * 100, 2), limit_b, round(perc_b * 100, 2))

    elif not (limit_a is None):

        perc_a = (feature_serie <= limit_a).sum() / len(feature_serie)
        variable_serie_0 = feature_serie[(feature_serie > limit_a) & (target == 0)]
        variable_serie_1 = feature_serie[(feature_serie > limit_a) & (target == 1)]

        title_plot = "La Distibution de la variable {} (les valeurs > {} ({}%) sont négligées)".format(feature_serie.name, limit_a,
                                                                                                       round(perc_a * 100, 3))
    else:
        variable_serie_0 = feature_serie[target == 0]
        variable_serie_1 = feature_serie[target == 1]
        title_plot = "La Distibution de la variable {}".format(feature_serie.name)


    trace_0 = go.Box(y=variable_serie_0, x=[feature_serie.name] * len(variable_serie_0), marker_size=1.5, name='y_0', boxpoints ='all', jitter = 0.1, whiskerwidth = 0.2)# line_width = 1
    trace_1 = go.Box(y=variable_serie_1, x=[feature_serie.name] * len(variable_serie_0), marker_size=1.5, name='y_1', boxpoints ='all', jitter = 0.1, whiskerwidth = 0.2)

    fig = go.Figure(data = [trace_0, trace_1])


    fig.update_layout(title={'text': title_plot,
                             'y': 0.90,
                             'x': 0.5,
                             'font': {'family': "Arial",
                                      'size': 18}},
                      #width = 500,
                      #height = 800,
                      #xaxis_title_text='Values',
                      yaxis_title_text='Count',
                      boxmode='group')
    return fig

def scatterplot_matrix(data, target):
    """ a plot of the scatter matrix of numeric features
    parameters:
    -----------
        data: [pd.DataFrame , shape(n_individuals, n_features)] DataFrame
        target: [pd.Series , shape(n_individuals)] target values
    return:
    -------
        fig: [plotly.graph_objs._figure.Figure] the figure of the distribution
    """
    textd = ['y_0' if cl == 0 else 'y_1' for cl in target]
    dimensions = [{'label': col, 'values': data.loc[:, col]} for col in data.columns]

    fig = go.Figure(data=go.Splom(
        dimensions=dimensions,
        marker=dict(color=target,
                    size=5,
                    colorscale='Bluered',
                    line=dict(width=0.5,
                              color='rgb(230,230,230)')),

        text=textd,
        diagonal=dict(visible=False)))

    fig.update_layout(
        title={'text': "nuage de points",
               'y': 0.995,
               'x': 0.5,
               'font': {'family': "Arial",
                        'size': 23,
                        'color': "black"}
               },
        dragmode='select',
        width=4000,
        height=4000,
        # hovermode='closest'
        font=dict(
            family="Arial",
            size=10,
            color="black"))

    return fig

def corr_matrix_heatmap(corr_matrix):
    trace = go.Heatmap(z=corr_matrix,
                       x=list(corr_matrix.index),
                       y=list(corr_matrix.columns))

    fig = go.Figure(data=trace)
    fig.update_layout(title={'text': "Correlation Matrix",
                             'y': 0.93,
                             'x': 0.5,
                             'font': {'family': "Arial",
                                      'size': 23,
                                      'color': "black"}},
                          width=900,
                          height=900)
    return fig