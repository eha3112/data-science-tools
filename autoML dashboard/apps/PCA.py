

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
from PCA_module_advanced import PC_Analysis

from app import app

### Initiate Redis client
r = redis.Redis( host='redis', port=6379, db=2 )
## clear the db
##r.flushall()

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
                                # Div used to store data
                                html.Div(id='intermediate-value-pca', style={"display": "none"}),

                                ## Navigation Bar
                                html.Nav(className="navbar navbar-expand bg-dark navbar-dark ", children=[
                                    html.Ul(className='navbar-nav', children=[
                                        html.Li(children=dcc.Link('Home', href='/', className="nav-link")
                                                , className="nav-item"),
                                        html.Li(children=dcc.Link('Univariate Analysis', href='/uni_analysis', className="nav-link")
                                                , className="nav-item"),
                                        html.Li(children=dcc.Link('Bivariate Analysis', href='/bi_analysis', className="nav-link")
                                                , className="nav-item"),
                                        html.Li(children=dcc.Link('PCA', href='/pca', className="nav-link")
                                                , className="nav-item active"),
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
                                                       children=html.Div(id="link-upload", children=['(CSV file) Drag and Drop or ',
                                                                                   html.A('Select Files')]))
                                ]),

                                ## Title Explained Variance
                                html.Div(className="m-2 pl-3",
                                         style=style_titles_bg,
                                         children= [
                                            html.H2(className="mx-auto", children="Explained Variance")
                                    ]),

                                ### Explained variance
                                html.Div(className="m-2 d-flex justify-content-center", style=style_section,
                                    id="container_var", children=[
                                        html.Div(id="graph_variance", children=[
                                                    dcc.Graph(id="plot_variance")
                                            ])
                                        ]),

                                html.Div(children=[
                                    html.H2(children="Correlation Circle")
                                ], className="m-2 pl-3", style=style_titles_bg),

                                ### Correlation circle plot
                                html.Div(className="row m-2", id="container_corr", style=style_section ,children=[
                                    ## components
                                    html.Div(className="col-lg-2", id="container_compo",children=[
                                            html.Div(children=[
                                                html.P("Principal axis 1 (X)", style=style_text),
                                                dcc.Dropdown(id='axis_x', value=1,
                                                             style=style_com_dropdown)
                                            ]),
                                            html.Div(children=[
                                                html.P("Principal axis 2 (Y)", style=style_text),
                                                dcc.Dropdown(id='axis_y', value=2,
                                                             style=style_com_dropdown)
                                            ]),
                                            html.Div(className="components",children=[
                                                html.P("Threshold axis 1", style=style_text),
                                                dcc.Input(id="threshold_1", type="number", placeholder="Threshold X",
                                                          min=0, max=1, step=0.05, value=0,
                                                          style=style_com_input)
                                            ]),

                                            html.Div(children=[
                                                html.P("Threshold axis 2", style=style_text),
                                                dcc.Input(id="threshold_2", type="number", placeholder="Threshold Y",
                                                          min=0, max=1, step=0.05, value=0,
                                                          style=style_com_input)
                                            ]),
                                        ]),

                                    ## Graph
                                    html.Div(className="col-lg-10 d-flex justify-content-center", children=[
                                        dcc.Graph(id="corr_circle")])
                                ]),

                                ## Title Projection
                                html.Div(children=[
                                    html.H2(children="Projection of Individuals on the Factor Plane")
                                ], className="m-2 pl-3", style=style_titles_bg), #className="div_h2"

                                ### Individual Projection Block
                                html.Div(className="row m-2", style=style_section, children = [
                                    html.Div(className="col-lg-2",id="container_comp_1", children= [
                                        html.Div(children= [
                                            html.P("Individuals quality (X)", style=style_text),
                                            dcc.Input(id="quali_x", type="number", placeholder="Quality ration",
                                                      min=0, max=100, step=0.5, value=0, style=style_com_input)
                                            ]),
                                        html.Div(children= [
                                            html.P("Individuals quality (Y)", style=style_text),
                                            dcc.Input(id="quali_y", type="number", placeholder="Quality ration",
                                                      min=0, max=100, step=0.5, value=0, style=style_com_input)
                                            ]),
                                        html.Div(children=[
                                            html.P("list of variables", style=style_text, id="text_var"),
                                            dcc.Dropdown(id='variables_list',  #options=variables_option,
                                                         style=style_com_dropdown, multi=True)
                                            ])
                                        ]),

                                    html.Div(className="col-lg-10 ", children=[
                                        html.Div(className="d-flex justify-content-center", children=[
                                            dcc.Graph(id="plot_projection")
                                        ]),
                                        html.Div(className="m-5 ", id="div_tab",children=[
                                                dash_table.DataTable(
                                                        id='table',
                                                        style_header={'backgroundColor': 'rgb(30, 30, 30)', 'textAlign': 'left'},
                                                        style_cell={'backgroundColor': '#696969',
                                                                'color': 'white',
                                                                'textAlign': 'left'
                                                                }
                                                )

                                            ])
                                    ])
                                ], id="container_projection"),
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
#
# @app.callback(Output('intermediate-value-pca', 'children'),
#                 [Input('upload-data', 'contents')],
#               [State('upload-data', 'filename')])
# def save_df_div(contents, filename):
#     if contents is not None:
#         data_json = preprocess_file(contents, filename)
#         df = pd.read_json(data_json, orient='split')
#
#         ## seperate target from df
#         if 'target' in list(df.columns):
#             # split data & target
#             target = df.loc[:, 'target']
#             df = df.drop(['target'], axis=1)
#
#             # save target
#             target_pickle = pickle.dumps(target)
#             r.mset({'target_pickle':target_pickle})
#
#         ## check numeric & categorical features
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
#         var_num_pickle = pickle.dumps(var_num)
#         var_cat_pickle = pickle.dumps(var_cat)
#         df_pickle = pickle.dumps(df)
#
#         r.mset({"ind_intermediate_0": 1,
#                 "var_num_pickle": var_num_pickle,
#                 "var_cat_pickle": var_cat_pickle,
#                 "df_pickle": df_pickle})
#
#
#         ###########
#         ### PCA ###
#         ###########
#         ## df to extract PC options
#         df_num = df.loc[:, var_num]
#
#         ## initiate PCA instance (compute PC ...)
#         PCA = PC_Analysis(df_num)
#         pc_list = [{'label': 'PC_' + str(i), 'value': i} for i in range(1, df_num.shape[1] + 1)]
#         pc_list_pickle = pickle.dumps(pc_list)
#
#         variables_option = [{'label': var, 'value': var} for var in list(df_num.columns)]
#         variables_option_pickle = pickle.dumps(variables_option)
#
#
#         ## pickle PCA component and save it in a redis db
#         PCA_pickle = pickle.dumps(PCA)
#
#         r.mset({"PCA_instance": PCA_pickle,
#                 "pc_list_pickle": pc_list_pickle,
#                 "variables_option_pickle": variables_option_pickle})
#
#         return 1


@app.callback([Output('axis_x', 'options'),
              Output('axis_y', 'options'),
               Output('variables_list', 'options')],
              [Input("intermediate-value-pca", "children")])
def fill_components(ind):
    # if ind is None:
    if r.get("ind_intermediate_0") is None:
        raise PreventUpdate
    else:
        pc_list_pickle = r.get('pc_list_pickle')
        pc_list = pickle.loads(pc_list_pickle)

        variables_option_pickle = r.get("variables_option_pickle")
        variables_option = pickle.loads(variables_option_pickle)
        return pc_list, pc_list, variables_option



@app.callback(Output("plot_variance", "figure"),
              [Input("intermediate-value-pca", "children")])

def plot_variance(ind):
    # if ind is None:
    if r.get("ind_intermediate_0") is None:
        raise PreventUpdate
    else:
        #df = pd.read_json(file_json, orient='split')
        PCA_pickle = r.get('PCA_instance')
        PCA = pickle.loads(PCA_pickle)
        ### explained variance plot
        plot_explained_var = PCA.plot_explained_variance()
        plot_explained_var['layout']['height'] = 450
        plot_explained_var['layout']['width'] = 1000
        plot_explained_var['layout']['paper_bgcolor'] = 'rgba(0,0,0,0)'
        plot_explained_var['layout']['plot_bgcolor'] = 'rgba(0,0,0,0)'
        return plot_explained_var


@app.callback(
     Output(component_id='corr_circle', component_property='figure'),
     [Input(component_id="axis_x", component_property="value"), Input("axis_y", "value"),
      Input("threshold_1", "value"), Input("threshold_2", "value"),
      Input("intermediate-value-pca", "children")])

def update_correlation_circle(PC_a, PC_b, ratio_a, ratio_b,ind):
    """

    """
    #if ind is None:
    if r.get("ind_intermediate_0") is None:
        raise PreventUpdate
    else:
        PCA_pickle = r.get('PCA_instance')
        PCA = pickle.loads(PCA_pickle)
        corrOldNew, fig_corr_circle = PCA.correlation_circle(PC_a, PC_b, ratio_a, ratio_b,)
        fig_corr_circle['layout']['paper_bgcolor'] = 'rgba(0,0,0,0)'
        fig_corr_circle['layout']['plot_bgcolor'] = 'rgba(0,0,0,0)'
        return fig_corr_circle




@app.callback(Output(component_id="plot_projection", component_property='figure'),
              [Input(component_id="axis_x", component_property="value"), Input("axis_y", "value"),
              Input("quali_x", "value"), Input("quali_y", "value"),
               Input("intermediate-value-pca", "children")])

def update_projection(PC_a, PC_b, quality_ratio_a, quality_ratio_b, ind):
    # if ind is None:
    if r.get("ind_intermediate_0") is None:
        raise PreventUpdate
    else:
        PCA_pickle = r.get('PCA_instance')
        PCA = pickle.loads(PCA_pickle)

        if r.get('target_pickle'):
            target_pickle = r.get('target_pickle')
            target = pickle.loads(target_pickle)
            plot_projection = PCA.projection_PC_space_mdf(PC_a, PC_b, target, quality_ratio_a, quality_ratio_b)
        else:
            plot_projection = PCA.projection_PC_space_mdf(PC_a, PC_b, target=None, quality_ratio_a=quality_ratio_a, quality_ratio_b=quality_ratio_b)
        #plot_projection = PCA.projection_PC_space(PC_a, PC_b, quality_ratio_a, quality_ratio_b)


        plot_projection['layout']['paper_bgcolor'] = 'rgba(0,0,0,0)'
        plot_projection['layout']['plot_bgcolor'] = 'rgba(0,0,0,0)'
        plot_projection['layout']['clickmode'] = 'event+select'
        return plot_projection


@app.callback([Output('table', 'columns'),Output('table', 'data')],
               [Input('plot_projection', 'selectedData'),
                Input('variables_list', 'value')])
def display_selected_data(selectedData, var_list):
    if selectedData is None or var_list is None:
        raise PreventUpdate
    else:
        selected_pts = selectedData["points"]
        # df_n = df.loc[list_index,['TotalDriveTime', 'HeavyBrakingCount', 'NightTime']]
        if len(selected_pts)==1:
            list_index = int(selected_pts[0]["pointIndex"])   #pointIndex
            # list_index = int(list_index)
        else:
            list_index = [int(selected_pts[i]["pointIndex"]) for i in range(len(selected_pts))]

        df_pickle = r.get('df_pickle')
        df = pickle.loads(df_pickle)
        var_num_pickle = r.get("var_num_pickle")
        var_num = pickle.loads(var_num_pickle)
        df = df.loc[:, var_num]

        df_n = df.loc[list_index, var_list]

        df_table = pd.DataFrame({'features': var_list,
                                 'mean': df_n.mean(axis=0).values,
                                 'median': df_n.median(axis=0).values,
                                 'min': df_n.min(axis=0).values,
                                 'max': df_n.max(axis=0).values,
                                 'std': df_n.std(axis=0).values
                                 })

        columns = [{"name": i, "id": i} for i in df_table.columns]
        data_table = df_table.to_dict('records')

        return columns, data_table


#
# if __name__=="__main__":
#
#     app.run_server(debug=True, port=8001)