

import base64
import io
import pickle

import numpy as np
import pandas as pd

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import redis

import sys
sys.path.append('src_code/')
from clustering import within_inertia_cluster

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import plotly.graph_objects as go

from app import app

### Initiate Redis client
r = redis.Redis( host='redis', port=6379, db=2 )

######### very importante to interpret the clusters
#https://towardsdatascience.com/interactive-visualization-of-decision-trees-with-jupyter-widgets-ca15dd312084

## styles
style_com_dropdown_c = {"background-color": "#696969",
                    'text-align': 'center',
                   "color": 'black',
                    'border-radius':'0.4em',
                   'margin': '0.8em 0% 0.0em 0%'}

style_com_input_c = {"background-color": "#696969",
               'text-align': 'center',
              'height': '2em',
              'width': '85%',
                'border-radius':'0.4em',
              'margin': '0.8em 10% 0em 5%'}

style_text = {'text-align': 'center', 'margin': '1em 0% 0.2em 0%'}

style_upload = {'text-align': 'center', 'background':'#D3D3D3', 'border-radius':'0.5em'}

style_titles_bg = {'background':'#D3D3D3', 'border-radius': '0.5em',
                   'font-family': "'Playfair Display SC', serif"}

style_section = {'border-radius': '0.5em', 'background': '#D3D3D3'}


## layout
layout = html.Div(style={"background-color": "#696969", "color": "black"},
                        children=[
                                # Div used to store data
                                html.Div(id='intermediate-value-cl_0', style={"display": "none"}),
                                # Div to indicate the end of models fitting and returns the number of clusters
                                html.Div(id='intermediate_div_clusters', style={"display": "none"}),


                                ## Navigation Bar
                                html.Nav(className="navbar navbar-expand bg-dark navbar-dark ", children=[
                                    html.Ul(className='navbar-nav', children=[
                                    html.Li(children=dcc.Link('Home', href='/', className="nav-link")
                                            , className="nav-item"),
                                    html.Li(children= dcc.Link('Univariate Analysis', href='/uni_analysis', className="nav-link")
                                             , className="nav-item"),
                                    html.Li(children= dcc.Link('Bivariate Analysis', href='/bi_analysis', className="nav-link")
                                             , className="nav-item"),
                                    html.Li(children=dcc.Link('PCA', href='/pca', className="nav-link")
                                            , className="nav-item"),
                                    html.Li(children=dcc.Link('Clustering', href='/clustering', className="nav-link")
                                            , className="nav-item active"),
                                        html.Li(children=dcc.Link('Modeling', href='/modeling', className="nav-link")
                                                , className="nav-item")
                                    ])
                                ]),

                                # ## Upload bar
                                # html.Div(id="div_upload", className="m-2 ", style=style_upload,
                                #          children=[
                                #             dcc.Upload(id='upload-data', className="btn btn-outline-info",
                                #                        children=html.Div(id="link-upload", children=['(CSV file) Drag and Drop or ',
                                #                                                    html.A('Select Files')]))
                                # ]),

                                ## Algo and criteria section
                                html.Div(className="m-2 d-flex justify-content-center", style=style_section,
                                         children=[

                                            # filter components
                                            html.Div(className="col-lg-2 mt-3", children=[
                                                dcc.Dropdown(style=style_com_dropdown_c,
                                                             options=[{'label': 'K-means', 'value': 'k-means'}], placeholder="Algorithm"),
                                                dcc.Input(id="id_nbr_clusters", style = style_com_input_c,
                                                          type="number", placeholder="Number of clusters", min=1, max=20, step=1),
                                                #####################
                                                html.Button(id='id_button_cluster', n_clicks=0, children='fit model',
                                                            style={  # "background-color": "#696969",
                                                                'text-align': 'center',
                                                                'height': '2em',
                                                                'width': '85%',
                                                                # 'border-radius':'0.4em',
                                                                'margin': '0.5em 10% 1em 5%'},
                                                            className="btn btn-secondary"),
                                            ]),

                                            # plot
                                            html.Div(className="col-lg-10 ", children=[
                                                dcc.Graph(id='plot_inertia')
                                            ])
                                ]),

                                ## Plot Individuals
                                html.Div(className="m-2 d-flex justify-content-center", style=style_section,
                                         children=[
                                            # filter components
                                            html.Div(className="col-lg-2 mt-3", children=[
                                                dcc.Dropdown(id="id_model_nbr", style=style_com_dropdown_c,
                                                             placeholder="Number of clusters"),
                                                dcc.Dropdown(id="x_axis", style=style_com_dropdown_c,
                                                             placeholder="Principal component 1", value=1),
                                                dcc.Dropdown(id="y_axis", style=style_com_dropdown_c,
                                                             placeholder="Principal component 2", value=2),
                                            ]),
                                            # plot
                                            html.Div(className="col-lg-10", children=[
                                                dcc.Graph(id="plot_individuals"),

                                                dash_table.DataTable(
                                                    id='table_clustering',
                                                    style_header={'backgroundColor': 'rgb(30, 30, 30)',
                                                                  'textAlign': 'center'},
                                                    style_cell={'backgroundColor': '#696969',
                                                                'color': 'white',
                                                                'textAlign': 'left'
                                                                })
                                            ])
                                ])
                        ])

#
# ## define a function for encoding and decoding the file
# def preprocess_file(contents, filename):
#     content_type, content_string = contents.split(',')
#
#     decoded = base64.b64decode(content_string)
#     try:
#         if 'csv' in filename:
#             # Assume that the user uploaded a CSV file
#             df = pd.read_csv(
#                 io.StringIO(decoded.decode('utf-8')))
#     except Exception as e:
#         print(e)
#         return html.Div([
#             'There was an error processing this file.'
#         ])
#
#     return df.to_json(date_format='iso', orient='split')
#
#
# @app.callback(Output('intermediate-value-cl_0', 'children'),
#                 [Input('upload-data', 'contents')],
#               [State('upload-data', 'filename')])
# def save_df_div(contents, filename):
#     if contents is not None:
# ############################
#         # save dataframe as JSON in redis
#         data_json = preprocess_file(contents, filename)
#
#         # save the options of principal component in redis
#         df = pd.read_json(data_json, orient='split')
#
#
#         ## seperate target from df
#         if 'target' in list(df.columns):
#             # split data & target
#             target = df.loc[:, 'target']
#             df = df.drop(['target'], axis=1)
#
#             # save target
#             target_pickle = pickle.dumps(target)
#             r.mset({'target_pickle': target_pickle})
#
#         # # split data & target
#         # target = df.loc[:, 'target']
#         # df = df.drop(['target'], axis=1)
#
#         # check datatype to determince categorical features
#         vt_dict = {v.Integer: 'int64', v.Float: 'float64', v.Boolean: 'bool', v.String: 'str'}
#
#         for var in df.columns:
#             serie_type = [vt_dict[x] for x in vt_dict.keys() if df.loc[:, var] in x][0]
#             df[var] = df[var].astype(serie_type)
#
#         ## store colum names
#         var_num = []
#         var_cat = []
#         for var in df.columns:
#             if df[var].dtype == 'object':
#                 var_cat.append(var)
#             elif df[var].dtype == 'int64' or df[var].dtype == 'float64':
#                 var_num.append(var)
#             else:
#                 pass
#
#         ## save var_num, var_cat, df, target into redis database
#         # pickling variables
#         var_num_pickle = pickle.dumps(var_num)
#         var_cat_pickle = pickle.dumps(var_cat)
#         df_pickle = pickle.dumps(df)
#         #target_pickle = pickle.dumps(target)
#
#
#         r.mset({"ind_intermediate_0": 1,
#                 "var_num_pickle": var_num_pickle,
#                 "var_cat_pickle": var_cat_pickle,
#                 "df_pickle": df_pickle,
#                 #"target_pickle": target_pickle
#                 })
#
#
#         ## since I already wrote the code using df then
#         df_num = df.loc[:, var_num]
#         data_json_num = df_num.to_json(date_format='iso', orient='split')
#
#         #
#         pc_list = [{'label': 'PC_' + str(i), 'value': i} for i in range(1, df_num.shape[1] + 1)]
#         pc_list_pickle = pickle.dumps(pc_list)
#
#         r.mset({#"data_json_num": data_json_num,
#                "pc_list_pickle": pc_list_pickle})
#
#         return 1


@app.callback([Output("x_axis", "options"),
               Output("y_axis", "options")],
              [Input('intermediate_div_clusters', 'children')])
def fill_components_1(nb_clusters):
    """  1. it fills the dropdown box for the principal components analysis
         2. compute the statistics of clasters and save them in a dictionary
             {1: df_1, 2: df_2, 3: df_3 .......}
             and then save them in redis db
    """
    if nb_clusters is None:
        raise PreventUpdate

    else:
        # 1
        pc_list_pickle = r.get('pc_list_pickle')
        pc_list = pickle.loads(pc_list_pickle)

        # 2
        df_pickle = r.get("df_pickle")
        df = pickle.loads(df_pickle)
        var_num_pickle = r.get("var_num_pickle")
        var_num = pickle.loads(var_num_pickle)

        df_num = df.loc[:, var_num]
        del df

        # data_json_num = r.get("data_json_num")
        # df_num = pd.read_json(data_json_num, orient='split')

        dict_df_stat = {}
        for i in range(2, nb_clusters+1):
            # new df for each model
            df_stat = pd.DataFrame({'features': list(df_num.columns)})

            ## load the model corresponding to this number
            model_pickle = r.get("model_{}".format(i))
            KM = pickle.loads(model_pickle)
            labels = KM.labels_              # np!


            for label in np.unique(labels):
                df_temp = df_num.loc[labels == label, :]
                df_stat['clus_' + str(label) + '_' + 'mean'] = df_temp.mean().to_numpy()
                df_stat['clus_' + str(label) + '_' + 'median'] = df_temp.median().to_numpy()

            df_stat = df_stat.round(4)
            dict_df_stat['model_'+str(i)] = df_stat


        ## save dictionary of dataframes
        dict_df_stat_pickle = pickle.dumps(dict_df_stat)
        r.mset({'dict_df_stat': dict_df_stat_pickle})

        return pc_list, pc_list






@app.callback([Output('plot_inertia', 'figure'),
               Output('id_model_nbr', 'options'),
               Output('intermediate_div_clusters', 'children')],
              [Input('id_button_cluster', 'n_clicks')],
              [State('intermediate-value-cl_0', 'children'),
               State('id_nbr_clusters', 'value')])
def plot_inertia(n_clicks, ind, nbr_cluster):
    # if ind is None or nbr_cluster is None or n_clicks==0:
    if r.get("ind_intermediate_0") is None or nbr_cluster is None or n_clicks==0:
        raise PreventUpdate
    else:
        df_pickle = r.get("df_pickle")
        df = pickle.loads(df_pickle)
        var_num_pickle = r.get("var_num_pickle")
        var_num = pickle.loads(var_num_pickle)

        df_num = df.loc[:, var_num]
        del df
        # data_json = r.get("data_json_num")
        #
        # data = pd.read_json(data_json, orient='split')

        #data_N = StandardScaler().fit_transform(data)
        data_N = StandardScaler().fit_transform(df_num)

        w_cluster_inertia = []   # within cluster inertia
        nb_clusters_list = range(2, nbr_cluster+1)

        ## Options for model
        options_model = []
        ##

        for nb_clusters in nb_clusters_list:
            print("nbr of clusters : {} ...".format(nb_clusters))
            KM = KMeans(n_clusters=nb_clusters)
            KM.fit(data_N)
            w_cluster_inertia.append(KM.inertia_)

            ## save models in Redis so as to use them in the clusters visualizations
            # dictionary to map from 1, 2, 3 ... --->  model_1, model_2 ....
            options_model.append({'label': str(nb_clusters), 'value': 'model_{}'.format(nb_clusters)})
            # save model under the key model_x, where x=1, 2 ....
            km_pickle = pickle.dumps(KM)
            r.mset({'model_{}'.format(nb_clusters): km_pickle})

        fig_1 = within_inertia_cluster(nb_clusters_list, w_cluster_inertia)

        fig_1['layout']['paper_bgcolor'] = 'rgba(0,0,0,0)'
        fig_1['layout']['plot_bgcolor'] = 'rgba(0,0,0,0)'

        ind_nbr_clusters = nbr_cluster

        return fig_1, options_model, ind_nbr_clusters

#
# @app.callback(Output("plot_individuals", "figure"),
#               [Input("id_model_nbr","value"),
#                Input("x_axis","value"),
#                Input("y_axis","value"),
#                Input('intermediate-value-cl_0','children')])
# def plot_indivduals(model_x,PC_a, PC_b, ind):
#     if ind is None or model_x is None:
#         raise PreventUpdate
#     else:
#         PC_a -=1
#         PC_b -= 1
#
#         ## Compute PCA
#         data_json = r.get("data_json_num")
#         data = pd.read_json(data_json, orient='split')
#         data_N = StandardScaler().fit_transform(data)
#
#         pca = PCA()
#         individuals_PC = pca.fit_transform(data_N)
#
#         ## labels
#         model_pickle = r.get(model_x)
#         KM = pickle.loads(model_pickle)
#         labels = KM.labels_
#
#         ## plot
#         trace_list = []
#         for i in np.unique(labels):
#             trace = go.Scattergl(x=individuals_PC[labels == i, PC_a], y=individuals_PC[labels == i, PC_b],
#                                  name='class_' + str(i), mode='markers',
#                                  marker={'size': 4}, opacity=0.7)
#             trace_list.append(trace)
#
#         fig = go.Figure(data=trace_list)
#
#         fig.update_layout(
#             title={
#                 'text': "[" + str(PC_a + 1) + " X " + str(PC_b + 1) + "]",
#                'y': 0.90,
#                'x': 0.5,
#                'font': {'family': "Arial",
#                         'size': 23,
#                         'color': "black"}},
#
#             xaxis=dict(title='PC' + str(PC_a + 1), showline=True),
#             yaxis=dict(title='PC' + str(PC_b + 1), showline=True))
#
#
#         fig['layout']['paper_bgcolor'] = 'rgba(0,0,0,0)'
#         fig['layout']['plot_bgcolor'] = 'rgba(0,0,0,0)'
#         return fig




@app.callback([Output("plot_individuals", "figure"),
                Output('table_clustering', 'columns'), Output('table_clustering', 'data')],
              [Input("id_model_nbr", "value"),
               Input("x_axis", "value"),
               Input("y_axis", "value"),
               Input('intermediate-value-cl_0', 'children')])
def plot_indivduals(model_x, PC_a, PC_b, ind):
    # if ind is None or model_x is None:
    if r.get("ind_intermediate_0") is None or model_x is None:
        raise PreventUpdate
    else:
        PC_a -=1
        PC_b -= 1

        ## Compute PCA
        df_pickle = r.get("df_pickle")
        df = pickle.loads(df_pickle)
        var_num_pickle = r.get("var_num_pickle")
        var_num = pickle.loads(var_num_pickle)

        df_num = df.loc[:, var_num]
        del df
        data_N = StandardScaler().fit_transform(df_num)


        # data_json = r.get("data_json_num")
        # data = pd.read_json(data_json, orient='split')
        # data_N = StandardScaler().fit_transform(data)

        pca = PCA()
        individuals_PC = pca.fit_transform(data_N)

        ## labels
        model_pickle = r.get(model_x)
        KM = pickle.loads(model_pickle)
        labels = KM.labels_

        ## plot
        trace_list = []
        for i in np.unique(labels):
            trace = go.Scattergl(x=individuals_PC[labels == i, PC_a], y=individuals_PC[labels == i, PC_b],
                                 name='class_' + str(i), mode='markers',
                                 marker={'size': 4}, opacity=0.7)
            trace_list.append(trace)

        fig = go.Figure(data=trace_list)

        fig.update_layout(
            title={
                'text': "[" + str(PC_a + 1) + " X " + str(PC_b + 1) + "]",
               'y': 0.90,
               'x': 0.5,
               'font': {'family': "Arial",
                        'size': 23,
                        'color': "black"}},

            xaxis=dict(title='PC' + str(PC_a + 1), showline=True),
            yaxis=dict(title='PC' + str(PC_b + 1), showline=True))


        fig['layout']['paper_bgcolor'] = 'rgba(0,0,0,0)'
        fig['layout']['plot_bgcolor'] = 'rgba(0,0,0,0)'

        ## table

        # load table corresponding to the number of cluster
        dict_df_stat_pickle = r.get("dict_df_stat")
        dict_df_stat = pickle.loads(dict_df_stat_pickle)
        print(dict_df_stat.keys())
        df_stat_curr = dict_df_stat[model_x]

        columns = [{"name": i, "id": i} for i in df_stat_curr.columns]
        data_table = df_stat_curr.to_dict('records')

        return fig, columns, data_table
