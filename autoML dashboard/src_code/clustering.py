

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
import pandas as pd

from sklearn.decomposition import PCA

#https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html

import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def within_inertia_cluster(nb_clusters_list, w_cluster_inertia):
    data_inertia = go.Scatter(x=list(nb_clusters_list), y=w_cluster_inertia, mode='lines+markers')

    fig = go.Figure([data_inertia])

    fig.update_layout(
        title={'text': "Within Cluster Inertia",
               'y': 0.90,
               'x': 0.5,
               'font': {'family': "Arial",
                        'size': 23,
                        'color': "black"}},
        yaxis={'title': 'wihin cluster inertia'},
        xaxis={'title': 'number of clusters'},
        font={'family': "Arial",
              'size': 15,
              'color': "black"},
        showlegend=False)

    return fig


def silouette_cluster(nb_clusters_list, silouette_list):
    data_inertia = go.Scatter(x=list(nb_clusters_list), y=silouette_list, mode='lines+markers')

    fig = go.Figure([data_inertia])

    fig.update_layout(
        title={'text': "Silhouette Score",
               'y': 0.90,
               'x': 0.5,
               'font': {'family': "Arial",
                        'size': 23,
                        'color': "black"}},
        yaxis={'title': 'The Silhouette Coefficient '},
        xaxis={'title': 'number of clusters'},
        font={'family': "Arial",
              'size': 15,
              'color': "black"},
        showlegend=False)

    return fig


def projection(data_N, labels):

    pca = PCA()
    individuals_PC = pca.fit_transform(data_N)

    palette = sns.color_palette("hls", len(np.unique(labels)))
    plt.figure(figsize=(20, 10))

    sns.scatterplot(x=individuals_PC[:, 0], y=individuals_PC[:, 1], hue=labels, palette=palette )
    plt.title("Visualization des clusters sur le plan factoriel 1-2")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")


def feature_correlation(data_N, var_num):
    ### PCA
    pca = PCA()
    individuals_PC = pca.fit_transform(data_N)

    tmp = np.corrcoef(data_N.T, individuals_PC.T)
    corrOldNew = tmp[0:len(var_num), len(var_num):]

    del tmp

    corrOldNew = pd.DataFrame(corrOldNew, columns=['PC_' + str(i) for i in range(1, corrOldNew.shape[1] + 1)],
                              index=var_num)

    corrOldNew = corrOldNew.sort_values(by=['PC_1'], ascending=False)

    trace_1 = go.Bar(x=list(corrOldNew.index), y=corrOldNew.iloc[:, 0], opacity=0.5)
    trace_2 = go.Bar(x=list(corrOldNew.index), y=corrOldNew.iloc[:, 1], opacity=0.5)

    fig = go.Figure(data=[trace_1, trace_2])

    fig.update_layout(
        title={
            'text': "Feature Influence on PC_1 & PC_2",
            'y': 0.90,
            'x': 0.45,
            'font': {'family': "Arial",
                     'size': 20,
                     'color': "black"}},
        xaxis={'title': "Features",
               'showline': False},
        yaxis={'title': "Correlation",
               'showline': False},
        bargap=0.2,  # gap between bars of adjacent location coordinates
        bargroupgap=0.1  # gap between bars of the same location coordinates ))
    )

    return fig


def scatter_plot(data_df, labels, feature_1, feature_2):

    palette = sns.color_palette("hls", len(np.unique(labels)))

    plt.figure(figsize=(20,10))
    ax = sns.scatterplot(x=data_df.loc[:,feature_1], y=data_df.loc[:,feature_1],
                    hue=labels, palette= palette,s=80,legend='full')
    plt.title("Scatter Plot ["+feature_1+", "+feature_2+"]")
    plt.xlabel(feature_1)
    plt.ylabel(feature_2)


def plot_target_cluster(target, model):
    # seperate the cluster labels according to the target values
    y_target_list = [model.labels_[(target == val).to_numpy().reshape(-1)] for val in target.unique()]
    x = ['class' + str(i) for i in np.unique(model.labels_)]
    trace_list = []
    for i, val in enumerate(target.unique()):
        y_cluster_class = [(y_target_list[i] == label).sum() for label in np.unique(model.labels_)]
        trace_temp = go.Bar(x=x, y=y_cluster_class, name='y_' + str(val), opacity=0.5)
        trace_list.append(trace_temp)

    fig = go.Figure(data=trace_list)

    fig.update_layout(
        title={
            'text': "Distribution of target class in function of the cluster classes",
            'y': 0.90,
            'x': 0.45,
            'font': {'family': "Arial",
                     'size': 20,
                     'color': "black"}},
        xaxis={'title': "Cluster classes",
               'showline': False},
        yaxis={'title': "occurence",
               'showline': False},
        bargap=0.2,  # gap between bars of adjacent location coordinates
        bargroupgap=0.1  # gap between bars of the same location coordinates ))
    )

    return fig


def plot_target_cluster_pos(target, model):

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # seperate the cluster labels according to the target values
    y_target_list = [model.labels_[(target == val).to_numpy().reshape(-1)] for val in target.unique()]
    x = ['class' + str(i) for i in np.unique(model.labels_)]

    ### plot data for each class
    trace_list = []
    for i, val in enumerate(target.unique()):
        y_cluster_class = [(y_target_list[i] == label).sum() for label in np.unique(model.labels_)]
        trace_temp = go.Bar(x=x, y=y_cluster_class, name='y_' + str(val), opacity=0.5)
        trace_list.append(trace_temp)

    #fig = go.Figure(data=trace_list)

    ####
    y_count_0 = [(y_target_list[0] == label).sum() for label in np.unique(model.labels_)]
    y_count_1 = [(y_target_list[1] == label).sum() for label in np.unique(model.labels_)]

    pos_rate = [y_count_1[i]/y_count_0[i] for i in range(len(y_count_0))]
    trace_pos = go.Scatter(x=x, y=pos_rate,
                       mode='markers',
                       name = 'Positve Rate',
                       marker={'size': 5,
                               'symbol': 'circle',
                               'color': 'red'})

    for trace in trace_list:
        fig.add_trace(trace,  # , name="yaxis data"
                      secondary_y=False)

    fig.add_trace(trace_pos,
                  secondary_y=True)



    fig.update_layout(
        title={'text': "Distribution of target class in function of the cluster classes",
               'y': 0.90,
               'x': 0.45,
               'font': {'family': "Arial",
                        'size': 22}
               },
        xaxis_title_text='Value',
        bargap=0.2,  # gap between bars of adjacent location coordinates
        bargroupgap=0.1  # gap between bars of the same location coordinates ))
    )

    fig.update_yaxes(title_text="Count", secondary_y=False)
    fig.update_yaxes(title_text="Positive rate", secondary_y=True)







    return fig


def plot_score_clusters(score, model):
    x = ["s_" + str(val) for val in np.sort(score.unique())]

    # seperate the score vector according to the cluster's labels
    score_cluster_list = [score.loc[model.labels_ == label] for label in np.unique(model.labels_)]

    trace_list = []

    for i, val in enumerate(np.unique(model.labels_)):
        y_cluster_score = [(score_cluster_list[i] == score_val).sum() for score_val in np.sort(score.unique())]
        trace_temp = go.Bar(x=x, y=y_cluster_score, name='label_' + str(val), opacity=0.5)
        trace_list.append(trace_temp)

    fig = go.Figure(data=trace_list)

    fig.update_layout(
        title={
            'text': "Distribution of class in function of scores",
            'y': 0.90,
            'x': 0.45,
            'font': {'family': "Arial",
                     'size': 20,
                     'color': "black"}},
        xaxis={'title': "Scores",
               'showline': False},
        yaxis={'title': "occurence",
               'showline': False},
        barmode='stack')

    return fig