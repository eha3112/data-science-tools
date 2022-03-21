


import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd

def ROC_model(fpr, tpr, thresholds):
    """ plot the ROC curve
    parameters:
    -----------
        tpr: [np.array ]True Positive Rate
        fpr: [np.array] False Positive Rate
        thresholds: [np.array]
    return:
    -------
        fig: [plotly.graph_objs._figure.Figure]
    """

    trace = go.Scattergl(x=fpr, y=tpr, hovertext=thresholds, mode = 'lines+markers', marker= {'size':2})

    fig = go.Figure(data= [trace])

    fig.update_layout( title={'text': "ROC curve",
                               'y': 0.90,
                               'x': 0.5},
                        yaxis={'title': 'TPR'},
                        xaxis={'title': 'FPR'},
                        font={'size': 15},
                        showlegend=False)

    return fig

def conf_matrix_plot(df_conf_matrix):
    """ Confusion Matrix Heatmap
    :param df_conf_matrix:
    :return:
    """

    z = df_conf_matrix.to_numpy()

    x = list(df_conf_matrix.index)
    y = list(df_conf_matrix.columns)
    z_text = [[str(y) for y in x] for x in z]

    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='tempo')

    fig.update_layout(title={'text': "Correlation Matrix",
                             'y': 0.995,
                             'x': 0.5,
                             'font': {'family': "Arial",
                                      'size': 28,
                                      'color': "black"}},
                      width=600,
                      height=600,

                      xaxis={
                          "type": "category",
                          "title": "Predicted value",
                          "autorange": True,
                          "titlefont": {
                              "size": 18,
                              "color": "black",
                              "family": "Courier New, monospace"},
                          "tickfont": {
                              "size": 16,
                              "color": "black",
                              "family": "Courier New, monospace"},
                          "tickangle": 0,
                      },

                      yaxis={
                          "type": "category",
                          "title": "True Value",
                          "autorange": True,
                          "titlefont": {
                              "size": 18,
                              "color": "black",
                              "family": "Courier New, monospace"},
                          "tickfont": {
                              "size": 16,
                              "color": "black",
                              "family": "Courier New, monospace"},
                          "tickangle": 0,
                      },
                      )
    # add colorbar
    fig['data'][0]['showscale'] = False

    return fig


def plot_feature_importance(features, importance):
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