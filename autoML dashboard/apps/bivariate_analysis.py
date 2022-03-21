
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
from visualization import corr_matrix_heatmap
from stat_tests import discretize, statitical_tests, statitical_tests_target

from app import app
### Initiate Redis client
r = redis.Redis( host='redis', port=6379, db=2 )
## clear the db
r.flushall()

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



layout = html.Div( style={"background-color": "#696969", "color": "black"},
                       children=[

                           # Div used to store data
                           html.Div(id='intermediate-value-bi', style={"display": "none"}),

                           ## Navigation Bar
                           html.Nav(className="navbar navbar-expand bg-dark navbar-dark ", children=[
                               html.Ul(className='navbar-nav', children=[
                                   html.Li(children=dcc.Link('Home', href='/', className="nav-link")
                                           , className="nav-item"),
                                   html.Li(children=dcc.Link('EDA', href='/uni_analysis', className="nav-link")
                                           , className="nav-item "),
                                   html.Li(children=dcc.Link('Bivariate Analysis', href='/bi_analysis',className="nav-link ")
                                           , className="nav-item active"),
                                   html.Li(children=dcc.Link('PCA', href='/pca', className="nav-link")
                                           , className="nav-item "),
                                   html.Li(children=dcc.Link('Clustering', href='/clustering', className="nav-link")
                                           , className="nav-item"),
                                   html.Li(children=dcc.Link('Modeling', href='/modeling', className="nav-link")
                                           , className="nav-item")
                               ])
                           ]),

                           ## Upload bar
                           html.Div(id="div_upload", className="m-2 ", style=style_upload,
                                    children=[
                                        dcc.Upload(id='upload-data', className="btn btn-outline-info",
                                                   children=html.Div(id="link-upload",
                                                                     children=['(CSV file) Drag and Drop or ',
                                                                               html.A('Select Files')]))
                                    ]),

                           ## confusion matrix title
                           html.Div(className="m-2 pl-3",
                                    style=style_titles_bg,
                                    children=[
                                        html.H3(className="mx-auto", children="Correlation Matrix")
                                    ]),
                            ## confusion matrix plot
                           html.Div(className="m-2 row d-flex justify-content-center", style=style_section,
                                    children=[
                                        dcc.Graph(id='id_bi_corr_matrix')
                                    ]),

                           ## categorical features title
                           html.Div(className="m-2 pl-3",
                                    style=style_titles_bg,
                                    children=[
                                        html.H3(className="mx-auto", children="Categorical Features")
                                    ]),
                           ## categorical features plot
                           html.Div(className='m-2 row d-flex justify-content-center', style=style_section,
                                    children=[
                                        dash_table.DataTable(
                                            id='id_table_bi_cat',
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

                            ## descritization & independence title
                           html.Div(className="m-2 pl-3",
                                    style=style_titles_bg,
                                    children=[
                                        html.H3(className="mx-auto", children="Descritization & Independence")
                                    ]),
                           html.Div(className="m-2 row d-flex justify-content-center", style=style_section,
                                    children=[
                                        # components
                                        html.Div(className="col-lg-2 mt-3", children=[
                                            html.P("Classification Threshold", style=style_text),
                                            dcc.Input(id="id_no_disc", style=style_com_input,
                                                      type="number", placeholder="discretization threshold", min=0, max=100,
                                                      step=1, value=10),
                                            html.P("number of modality", style=style_text),
                                            dcc.Input(id="id_nbr_modality", style=style_com_input,
                                                      type="number", placeholder="number of modality", min=1,
                                                      max=100,
                                                      step=1, value=4),
                                        ]),
                                        # 2 features
                                        html.Div(className="col-lg-10 mt-3", children=[
                                            html.Div(className="m-2",
                                                     children=[
                                                         dash_table.DataTable(
                                                             id='id_table_bi_2_var',
                                                             style_header={'backgroundColor': 'rgb(30, 30, 30)'
                                                                            , 'textAlign': 'center'
                                                                           },
                                                             style_cell={'backgroundColor': '#696969',
                                                                         'color': 'white',
                                                                         'textAlign': 'center'},
                                                             # fixed_rows={'headers': True},
                                                             style_table={'height': '300px', #'width': '1300px',
                                                                          'overflowY': 'auto', 'overflowX': 'auto'},
                                                             page_size=20,
                                                             css=[{'selector': '.row', 'rule': 'margin: 0'}]
                                                             # it worked
                                                         )
                                            ]),
                                            # feature & target
                                            html.Div(className="m-2",
                                                     children=[
                                                         dash_table.DataTable(
                                                             id='id_table_bi_var_target',
                                                             style_header={'backgroundColor': 'rgb(30, 30, 30)'
                                                                 , 'textAlign': 'center'
                                                                           },
                                                             style_cell={'backgroundColor': '#696969',
                                                                         'color': 'white',
                                                                         'textAlign': 'center'},
                                                             # fixed_rows={'headers': True},
                                                             style_table={'height': '400px',  # 'width': '1300px',
                                                                          'overflowY': 'auto', 'overflowX': 'auto'},
                                                             page_size=20,
                                                             css=[{'selector': '.row', 'rule': 'margin: 0'}]
                                                             # it worked
                                                         )
                                                     ]),

                                        ])



                           ])
                        ])





# ##################################
# ######      Callbacks       ######
# ##################################
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
# @app.callback(Output('intermediate-value-bi', 'children'),
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
#                 "target_pickle": target_pickle})
#         return 1


@app.callback(Output("id_bi_corr_matrix","figure"),
              [Input("intermediate-value-bi","children")])
def plot_corr_matrix(ind):
    # if ind is None:
    if r.get("ind_intermediate_0") is None:
        raise PreventUpdate
    else:
        df_pickle = r.get("df_pickle")
        df = pickle.loads(df_pickle)
        var_num_pickle = r.get("var_num_pickle")
        var_num = pickle.loads(var_num_pickle)

        corr_matrix = df.loc[:, var_num].corr()
        fig = corr_matrix_heatmap(corr_matrix)

        fig['layout']['paper_bgcolor'] = 'rgba(0,0,0,0)'
        fig['layout']['plot_bgcolor'] = 'rgba(0,0,0,0)'

        return fig

@app.callback([Output('id_table_bi_cat', 'columns'), Output('id_table_bi_cat', 'data')],
              [Input("intermediate-value-bi","children")])
def table_stat_cat(ind):
    # if ind is None:
    if r.get("ind_intermediate_0") is None:
        raise PreventUpdate
    else:
        df_pickle = r.get("df_pickle")
        df = pickle.loads(df_pickle)

        var_cat_pickle = r.get("var_cat_pickle")
        var_cat = pickle.loads(var_cat_pickle)

        stat_data_cat = statitical_tests(df.loc[:, var_cat], alpha=0.05)
        stat_data_cat = stat_data_cat.sort_values(by=["Cramer's V"], ascending=False)

        columns = [{"name": i, "id": i} for i in stat_data_cat.columns]
        data_table = stat_data_cat.to_dict('records')

        return columns, data_table

@app.callback([Output('id_table_bi_2_var', 'columns'), Output('id_table_bi_2_var', 'data'),
               Output('id_table_bi_var_target', 'columns'), Output('id_table_bi_var_target', 'data')],
              [Input("intermediate-value-bi","children"),
               Input("id_no_disc","value"),
               Input("id_nbr_modality","value")])
# id_table_bi_2_var  id_table_bi_var_target
def disc_stat_tests(ind, disc_threshold, nbr_modality):
    # if ind is None:
    if r.get("ind_intermediate_0") is None:
        raise PreventUpdate
    else:
        df_pickle = r.get("df_pickle")
        df = pickle.loads(df_pickle)

        var_cat_pickle = r.get("var_cat_pickle")
        var_cat = pickle.loads(var_cat_pickle)

        var_num_pickle = r.get("var_num_pickle")
        var_num = pickle.loads(var_num_pickle)

        target_pickle = r.get("target_pickle")
        target = pickle.loads(target_pickle)

        # discretization of variables
        df_cat = pd.DataFrame()
        for col in var_num:
            if len(df.loc[:, col].unique()) < disc_threshold:
                df_cat[col] = df.loc[:, col]
            else:
                temp_cat = discretize(df.loc[:, col], target, modality=nbr_modality)
                df_cat[col] = temp_cat

        stat_2_fea = statitical_tests(df_cat, alpha=0.05)
        stat_2_fea = stat_2_fea.sort_values(by=["Cramer's V"], ascending=False)

        columns_2_fea = [{"name": i, "id": i} for i in stat_2_fea.columns]
        data_table_2_fea = stat_2_fea.to_dict('records')

        stat_target = statitical_tests_target(df_cat, target, alpha=0.05)
        stat_target = stat_target.sort_values(by=["Cramer's V"], ascending=False)

        columns_fea_target = [{"name": i, "id": i} for i in stat_target.columns]
        data_table_fea_target = stat_target.to_dict('records')

        return columns_2_fea, data_table_2_fea, columns_fea_target, data_table_fea_target
# id_no_disc    # id_nbr_modality

# if __name__=="__main__":
#     app.run_server(debug=True, port=8001)
