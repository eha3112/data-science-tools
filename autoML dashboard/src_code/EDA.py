

import pandas as pd
import plotly.graph_objects as go


def plot_dist(serie, target=None):
    if target is None:
        trace = go.Histogram(x=serie,
                             autobinx=True,
                             name=serie.name,
                             opacity=0.5)
        fig = go.Figure(data=[trace])

    else:
        list_trace = []
        for label in target.unique():
            temp_trace = go.Histogram(x=serie[target == label],
                                      autobinx=True,
                                      name='label_'+str(label),
                                      opacity=0.5)
            list_trace.append(temp_trace)

        fig = go.Figure(data=list_trace)

    fig.update_layout(title={'text': 'distribution ' + serie.name,
                             'y': 0.90,
                             'x': 0.5,
                             'font': {'family': "Arial",
                                      'size': 18}},
                      xaxis_title_text='Values',
                      yaxis_title_text='Count')

    return fig

def boxplot(serie, target=None):
    if target is None:
        trace = go.Box(y=serie, x=[serie.name] * len(serie), marker_size=1.5, name=serie.name,
                         boxpoints=False, jitter=0.1, whiskerwidth=0.1)  # boxpoints='all'

        fig = go.FigureWidget(data=[trace])


    else:
        list_trace = []
        for label in target.unique():
            temp_trace = go.Box(y=serie[target==label], x=[serie.name] * len(serie), marker_size=1.5, name='label_'+str(label),
                     boxpoints=False, jitter=0.1, whiskerwidth=0.1)  # boxpoints='all'

            list_trace.append(temp_trace)

        fig = go.Figure(data=list_trace)

    fig.update_layout(title={'text': 'boxplot ' +serie.name,
                             'y': 0.90,
                             'x': 0.5,
                             'font': {'family': "Arial",
                                      'size': 18}},
                      yaxis_title_text='Value',
                      boxmode='group')
    return fig