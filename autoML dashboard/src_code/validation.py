

from math import floor
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_MProba_target(proba, target, title_end=''):
    """ The distribtion of the target classes as a function of the probability
    parameters:
    -----------
        proba : [list or np.array, shape(n_individuals)] the model's probability
        target: [list or np.array, shape(n_individuals] the classes array
        title_end: [string]: string to end at the end of the figure's title
    return:
    -------
        fig: [plotly.graph_objs._figure.Figure] the figure of the distribution
    """
    ## create a dataframe to ease manipulation
    df = pd.DataFrame({'proba': proba * 100,
                            'class': target})
    df.loc[:, 'proba'] = df.loc[:, 'proba'].apply(floor)

    df_1 = df.loc[df.loc[:, 'class'] == 1, :].groupby('proba').count()
    df_0 = df.loc[df.loc[:, 'class'] == 0, :].groupby('proba').count()

    ## find intersection of 2 proba vectors df_1 and df_2
    set_1 = set(df_1.index)
    set_0 = set(df_0.index)
    set_res = set_0.intersection(set_1)
    ## positive rate vecotr
    df_pos = (df_1.loc[set_res, ] / df_0.loc[set_res, ])

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    trace_1 = go.Bar(x=list(df_1.index), y=list(df_1['class']), name='y_1')
    trace_0 = go.Bar(x=list(df_0.index), y=list(df_0['class']), name='y_0')
    trace_p = go.Scatter(x=list(df_pos.index), y=df_pos['class'], name='positive rate',
                         mode='markers', marker={'size': 5,
                                                       'symbol': 'circle',
                                                       'color': '#2F4F4F'})

    fig.add_trace(trace_1, secondary_y=False)
    fig.add_trace(trace_0, secondary_y=False)
    fig.add_trace(trace_p, secondary_y=True)

    fig.update_layout(title={'text': 'Distribtion of target classes according to the predicted proba ' + title_end,
                             'x': 0.5,
                             'y': 0.9,
                             'font': {'size': 20}},
                      yaxis={'title': 'Count'},
                      xaxis={'title': 'Proba',
                             'tickmode': 'linear'},
                      # Bar propertires
                      barmode='group',
                      # bargap=0.9,
                      bargroupgap=0.3)

    fig.update_yaxes(title_text="Count", secondary_y=False)
    fig.update_yaxes(title_text="Positive ratio", secondary_y=True)

    return fig

def plot_feature_importance(features, importance):
    """ Plot feature importance of a model
    parameters:
    -----------
        features: [list] features names
        importance : [


    """
    df_feature_importance = pd.DataFrame({'features':features,'importance':importance})

    df_feature_importance = df_feature_importance.sort_values(by='importance',ascending=False)

    trace = go.Bar(x=df_feature_importance['features'], y=df_feature_importance['importance'], name=' name_1', marker_color='indianred')

    fig = go.Figure(data=[trace])

    fig.update_layout(  title={'text': "Feature importance",
                               'y': 0.90,
                               'x': 0.45,
                               'font': {'size': 18}},
                        xaxis={'title': "Features "},
                        yaxis={'title': "Importance"},
                        xaxis_tickangle=-45)
    return fig