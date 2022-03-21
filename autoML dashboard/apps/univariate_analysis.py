

import base64
import io
import pickle

import pandas as pd
import visions as v

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_table

import redis

import sys
sys.path.append('src_code/')
from EDA import plot_dist, boxplot

from app import app


### Initiate Redis client
r = redis.Redis( host='redis', port=6379, db=2 )
# r = redis.Redis(host='localhost', port=1234)
# ## clear the db
# r.flushall()
#
# ## Bootsrap stylesheet and script
# external_stylesheets = [
#    {'href': "https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/css/bootstrap.min.css",
#     'rel': 'stylesheet',
#     'integrity': "sha384-9aIt2nRpC12Uk9gS9baDl411NQApFmC26EwAOH8WgZl5MYYxFfc+NcPb1dKGj7Sk",
#     'crossorigin': "anonymous"},
#     { 'href': "https://fonts.googleapis.com/css2?family=Petrona&display=swap",
#       'rel':"stylesheet"},  #  font-family: 'Petrona', serif;
#     {'href': "https://fonts.googleapis.com/css2?family=Playfair+Display+SC&display=swap",
#     'rel':"stylesheet"},   # font-family: 'Playfair Display SC', serif;
# ]
#
# external_scripts = [
#     {'src': "https://code.jquery.com/jquery-3.5.1.slim.min.js",
#      'integrity': "sha384-DfXdz2htPH0lsSSs5nCTpuj/zy4C+OGpamoFVy38MVBnE+IbbVYUew+OrCXaRkfj",
#      'crossorigin': "anonymous"
#      },
#     {"src": "https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js",
#      "integrity": "sha384-Q6E9RHvbIyZFJoft+2mJbHaEWldlvI9IOYy5n3zV9zzTtmI3UksdQRVvoxMfooAo",
#      "crossorigin": "anonymous"
#      },
#     {'src': "https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/js/bootstrap.min.js",
#      'integrity': "sha384-OgVRvuATP1z7JjHLkuOU7Xw704+h835Lr+6QL9UvYjZE3Ipu6Tp75j7Bh/kR0JKI",
#      'crossorigin': "anonymous"
#      }]
#
# meta_tags=[
#     {'name':"viewport",
#     'content':"width=device-width, initial-scale=1",
#     }]
#
# ## Initialize Dash app
# app = dash.Dash(__name__, external_scripts = external_scripts,
#                         external_stylesheets = external_stylesheets,
#                         meta_tags = meta_tags)

## Styles
style_com_dropdown = {"background-color": "#696969",
                    'text-align': 'center',
                   "color": 'black',
                    'border-radius':'0.4em',
                   'margin': '0.0em 0% 0.0em 0%'}

style_com_input = {"background-color": "#696969",
               'text-align': 'center',
              'height': '2em',
              'width': '85%',
                'border-radius':'0.4em',
              'margin': '0em 10% 0em 5%'}

style_text = {'text-align': 'center', 'margin': '1em 0% 0.2em 0%'}

style_upload = {'text-align': 'center', 'background':'#D3D3D3', 'border-radius':'0.5em'}

style_titles_bg = {'background':'#D3D3D3', 'border-radius': '0.5em',
                   'font-family': "'Playfair Display SC', serif"}

style_section = {'border-radius': '0.5em', 'background': '#D3D3D3'}

layout = html.Div(style={"background-color": "#696969", "color": "black"},
                        children=[
                                # Div used to initiate some callbacks
                                html.Div(id='intermediate-value-uni', style={"display": "none"}),

                                ## Navigation Bar
                                html.Nav(className="navbar navbar-expand bg-dark navbar-dark ", children=[
                                    html.Ul(className='navbar-nav', children=[
                                        html.Li(children=dcc.Link('Home', href='/', className="nav-link")
                                                , className="nav-item"),
                                        html.Li(children=dcc.Link('EDA', href='/uni_analysis', className="nav-link")
                                                , className="nav-item active"),
                                        html.Li(children=dcc.Link('Bivariate Analysis', href='/bi_analysis', className="nav-link")
                                                , className="nav-item"),
                                        html.Li(children=dcc.Link('PCA', href='/pca', className="nav-link")
                                                , className="nav-item "),
                                        html.Li(children=dcc.Link('Clustering', href='/clustering', className="nav-link")
                                                , className="nav-item"),
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

                                ## DataFrame
                                html.Div(className="m-2 pl-3",
                                         style=style_titles_bg,
                                         children=[
                                             html.H3(className="mx-auto", children="DataFrame")
                                         ]),

                                html.Div(className="m-2 row d-flex justify-content-center", style=style_section,
                                         children=[
                                             dash_table.DataTable(
                                                 id='id_table_EDA_df',
                                                 style_header={'backgroundColor': 'rgb(30, 30, 30)'#, 'textAlign': 'center'
                                                               },
                                                 style_cell={'backgroundColor': '#696969',
                                                             'color': 'white',
                                                             'textAlign': 'center'},
                                                 #fixed_rows={'headers': True},
                                                 style_table={'height': '300px', 'width':'1300px',
                                                              'overflowY': 'auto', 'overflowX': 'auto'},
                                                 page_size=20,
                                                 css=[{'selector': '.row', 'rule': 'margin: 0'}]   # it worked
                                                             )
                                         ]),
                            ## Numeric Features
                            html.Div(className="m-2 pl-3",
                                     style=style_titles_bg,
                                     children=[
                                         html.H3(className="mx-auto", children="Numeric Features")
                                     ]),
                            html.Div(className="m-2",  style=style_section,
                                     children=[
                                         # stat table
                                         html.Div(className='row d-flex justify-content-center',
                                                  children=[
                                                      dash_table.DataTable(
                                                          id='id_table_EDA_num',
                                                          style_header={'backgroundColor': 'rgb(30, 30, 30)'
                                                                         , 'textAlign': 'center'
                                                                        },
                                                          style_cell={'backgroundColor': '#696969',
                                                                      'color': 'white',
                                                                      'textAlign': 'center'},
                                                          style_table={'width': '1300px',
                                                                       'overflowY': 'auto', 'overflowX': 'auto'},
                                                           css=[{'selector': '.row', 'rule': 'margin: 0'}]
                                                      )
                                                  ]),

                                         # plot
                                         html.Div(className='row ', # d-flex justify-content-center
                                                  children=[
                                                      html.Div(className="col-lg-2 mt-3", children=[
                                                          html.P("Plot Typle", style=style_text),
                                                          dcc.Dropdown(id="id_eda_plot_type", style=style_com_dropdown,
                                                                       options=[{'label': 'Histogram', 'value': 'dist'},
                                                                                {'label': 'Box Plot', 'value': 'box'}],
                                                                       value='dist',
                                                                       placeholder="Plot Type"),
                                                           html.P("Feature", style=style_text),
                                                          dcc.Dropdown(id="id_eda_features", style=style_com_dropdown,
                                                                       placeholder="Feature"),
                                                      ]),
                                                      html.Div(className="col-lg-10 mt-3", children=[
                                                          dcc.Graph(id="id_eda_plot_num")
                                                      ])
                                                  ])
                                         ]),

                            ## Categorical Features
                            html.Div(className="m-2 pl-3",
                                     style=style_titles_bg,
                                     children=[
                                         html.H3(className="mx-auto", children="Categorical Features")
                                     ]),
                            html.Div(className="m-2", style=style_section,


                                     children=[
                                     # stat table
                                         html.Div(className='row d-flex justify-content-center',
                                                  children=[
                                                      dash_table.DataTable(
                                                          id='id_table_EDA_cat',
                                                          style_header={'backgroundColor': 'rgb(30, 30, 30)',
                                                               'textAlign': 'center'
                                                                        },
                                                          style_cell={'backgroundColor': '#696969',
                                                                      'color': 'white',
                                                                      'textAlign': 'center'},
                                                          style_table={'width': '1300px',
                                                                       'overflowY': 'auto', 'overflowX': 'auto'},
                                                           css=[{'selector': '.row', 'rule': 'margin: 0'}]
                                                      )
                                                  ]),

                                        # plot
                                         html.Div(className='row ',  # d-flex justify-content-center
                                                  children=[
                                                      html.Div(className="col-lg-2 mt-3", children=[
                                                          html.P("Feature", style=style_text),
                                                          dcc.Dropdown(id="id_eda_features_cat", style=style_com_dropdown,
                                                                       placeholder="Feature"),
                                                      ]),
                                                      html.Div(className="col-lg-10 mt-3", children=[
                                                          dcc.Graph(id="id_eda_plot_cat")
                                                      ])
                                                  ])
                                     ])
                        ])

##################################
######      Callbacks       ######
##################################
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
# @app.callback(Output('intermediate-value', 'children'),
#                 [Input('upload-data', 'contents')],
#               [State('upload-data', 'filename')])
# def save_df_div(contents, filename):
#     if contents is not None:
#
#         # save dataframe as JSON in redis
#         data_json = preprocess_file(contents, filename)
#
#         # save the options of principal component in redis
#         df = pd.read_json(data_json, orient='split')
#
#         # split data & target
#         target = df.loc[:, 'target']
#         df = df.drop(['target'], axis=1)
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
#         target_pickle = pickle.dumps(target)
#
#
#         r.mset({"var_num_pickle" :var_num_pickle,
#                 "var_cat_pickle" :var_cat_pickle,
#                 "df_pickle" :df_pickle,
#                 "target_pickle" :target_pickle})
#
#
#         return 1


@app.callback([Output("id_eda_features","options"),
              Output("id_eda_features_cat", "options")],
              [Input('intermediate-value-uni', 'children')])
def fill_components(ind):
    #if ind is None:
    if r.get("ind_intermediate_0") is None:
        raise PreventUpdate
    else:
        var_num_pickle = r.get("var_num_pickle")
        var_num = sorted(pickle.loads(var_num_pickle))

        var_cat_pickle = r.get("var_cat_pickle")
        var_cat = pickle.loads(var_cat_pickle)

        feature_options_num = [{'label': var, 'value': var} for var in var_num]
        feature_options_cat = [{'label': var, 'value': var} for var in var_cat]
        return feature_options_num, feature_options_cat

@app.callback([Output('id_table_EDA_df', 'columns'), Output('id_table_EDA_df', 'data')],
              [Input('intermediate-value-uni', 'children')])
def table_dataframe(ind):
    # if ind is None:
    if r.get("ind_intermediate_0") is None:
        raise PreventUpdate
    else:
        df_pickle = r.get("df_pickle")
        df = pickle.loads(df_pickle)
        df_sample = df.iloc[:100,:]

        columns = [{"name": i, "id": i} for i in df_sample.columns]
        data_table = df_sample.to_dict('records')

        return columns, data_table


@app.callback([Output('id_table_EDA_num', 'columns'), Output('id_table_EDA_num', 'data')],  # id_table_EDA_cat
              [Input('intermediate-value-uni', 'children')])
def table_dataframe(ind):
    # if ind is None:
    if r.get("ind_intermediate_0") is None:
        raise PreventUpdate
    else:
        df_pickle = r.get("df_pickle")
        df = pickle.loads(df_pickle)
        var_num_pickle = r.get("var_num_pickle")
        var_num = pickle.loads(var_num_pickle)
        df_num_desc = df.loc[:, var_num].describe()
        df_num_desc = df_num_desc.round(4)

        # to include index and rename it
        df_num_desc= df_num_desc.reset_index()
        df_num_desc = df_num_desc.rename(columns={'index': 'stats'})

        columns = [{"name": i, "id": i} for i in df_num_desc.columns]
        data_table = df_num_desc.to_dict('records')

    return columns, data_table


@app.callback([Output('id_table_EDA_cat', 'columns'), Output('id_table_EDA_cat', 'data')],  # id_table_EDA_cat
              [Input('intermediate-value-uni', 'children')])
def table_dataframe(ind):
    """ it's better to seperate the caallbacks because we might have only categorical features
    """
    # if ind is None:
    if r.get("ind_intermediate_0") is None:
        raise PreventUpdate
    else:
        df_pickle = r.get("df_pickle")
        df = pickle.loads(df_pickle)
        var_cat_pickle = r.get("var_cat_pickle")
        var_cat = pickle.loads(var_cat_pickle)

        df_cat_desc = df.loc[:,var_cat].describe()

        # to include index and rename it
        df_cat_desc = df_cat_desc.reset_index()
        df_cat_desc = df_cat_desc.rename(columns={'index': 'stats'})

        columns = [{"name": i, "id": i} for i in df_cat_desc.columns]
        data_table = df_cat_desc.to_dict('records')

        return columns, data_table

@app.callback(Output("id_eda_plot_num", "figure"),
              [Input("intermediate-value-uni", "children"),
               Input("id_eda_plot_type", "value"),
               Input("id_eda_features", "value")])
def plot_numeric(ind, plot_type, feature):  #dist # box
    # if ind is None or feature is None:
    if r.get("ind_intermediate_0") is None or feature is None:
        raise PreventUpdate
    else:
        df_pickle = r.get('df_pickle')
        df = pickle.loads(df_pickle)
        target_pickle = r.get('target_pickle')
        target = pickle.loads(target_pickle)
        # var_num_pickle = r.get('')
        # var_num =

        if plot_type=='dist':
            fig = plot_dist(df.loc[:, feature], target=target)

        else:
            fig = boxplot(df.loc[:, feature], target=target)
        fig['layout']['paper_bgcolor'] = 'rgba(0,0,0,0)'
        fig['layout']['plot_bgcolor'] = 'rgba(0,0,0,0)'

        return fig


@app.callback(Output("id_eda_plot_cat","figure"),
              [Input("intermediate-value-uni","children"),
               Input("id_eda_features_cat","value")])
def plot_numeric(ind, feature):
    # if ind is None or feature is None:
    if r.get("ind_intermediate_0") is None or feature is None:
        raise PreventUpdate
    else:
        df_pickle = r.get('df_pickle')
        df = pickle.loads(df_pickle)
        target_pickle = r.get('target_pickle')
        target = pickle.loads(target_pickle)


        fig = plot_dist(df.loc[:, feature], target=target)

        fig['layout']['paper_bgcolor'] = 'rgba(0,0,0,0)'
        fig['layout']['plot_bgcolor'] = 'rgba(0,0,0,0)'

        fig.update_layout( bargap=0.2,  # gap between bars of adjacent location coordinates
                    bargroupgap=0.1  # gap between bars of the same location coordinates )))
                    )
        return fig

