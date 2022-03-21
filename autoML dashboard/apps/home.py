

import base64
import io
import pickle
import visions as v

import pandas as pd

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import sys
sys.path.append('src_code/')
from PCA_module_advanced import PC_Analysis

from app import app


import redis


## Initiate Redis client
r = redis.Redis( host='redis', port=6379, db=2 )

## clear the db
r.flushall()

## style 
style_upload = {'text-align': 'center', 'background':'#D3D3D3', 'border-radius':'0.5em'}
style_paragraph = {'text-align': 'left', 'background':'#D3D3D3', 'border-radius':'0.5em'}


## layout of home page
layout = html.Div(style={"background-color": "#696969", "color": "black"},
                    children=[
                    
                    ## Div used to indicate loading data and store it in redis db
                    html.Div(id='intermediate-value', style={"display": "none"}),

                    ## navigation bar
                    html.Nav(className="navbar navbar-expand-sm bg-dark navbar-dark", children=[
                        html.Ul(className='navbar-nav', children=[
                        html.Li(children=dcc.Link('Home', href='/', className="nav-link")
                                , className="nav-item active"),
                        html.Li(children= dcc.Link('Univariate Analysis', href='/uni_analysis', className="nav-link")
                                 , className="nav-item"),
                        html.Li(children= dcc.Link('Bivariate Analysis', href='/bi_analysis', className="nav-link")
                                 , className="nav-item"),
                        html.Li(children=dcc.Link('PCA', href='/pca', className="nav-link")
                                , className="nav-item"),
                        html.Li(children=dcc.Link('Clustering', href='/clustering', className="nav-link")
                                , className="nav-item"),
                            html.Li(children=dcc.Link('Modeling', href='/modeling', className="nav-link")
                                    , className="nav-item")
                        ])
                    ]),

                    ## note
                    html.Div(className="m-2 ", style=style_paragraph,
                             children=[ html.P(children=["   The aim of this application is to treat the binary classification, therefore the input should satisfy the following conditions :"
                                                         , html.Ul(children= [ html.Li("the target variable is called 'target' in the dataframe"),
                                                                               html.Li("the dataframe requires categorical variables, which are a string"),
                                                                               html.Li("the dataframe requires numerical features")]
                                                                   )
                                                         ]
                                               )
                                        ]
                             ),

                    ## Upload bar
                    html.Div(id="div_upload", className="m-2 ", style=style_upload,
                             children=[
                                 dcc.Upload(id='upload-data', className="btn btn-outline-info",
                                            children=html.Div(id="link-upload", children=['(CSV file) Drag and Drop or ',
                                                                                          html.A('Select Files')]))
                    ]),
        ])


##################################
######      Callbacks       ######
##################################
## define a function for encoding and decoding the file
def preprocess_file(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return df.to_json(date_format='iso', orient='split')



@app.callback(Output('intermediate-value', 'children'),
                [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])
def save_df_div(contents, filename):
    """
    inputs:
    -------
    outputs:
    --------
        intermediate-value: serves a indicator for other callbacks to fire(hidden div) 
    """
    # if the csv file is loaded using "upload-data"widget
    if contents is not None:
        
        #################################
        ### load df and its variables ###
        #################################
        
        # save dataframe as JSON in redis
        data_json = preprocess_file(contents, filename)

        # save the options of principal component in redis
        df = pd.read_json(data_json, orient='split')

        ## seperate target from df and save it in redis
        if 'target' in list(df.columns):
            # split data & target
            target = df.loc[:, 'target']
            df = df.drop(['target'], axis=1)

            # save target
            target_pickle = pickle.dumps(target)
            r.mset({'target_pickle': target_pickle})


        ## check datatype to determince categorical features
        vt_dict = {v.Integer: 'int64', v.Float: 'float64', v.Boolean: 'bool', v.String: 'str'}

        for var in df.columns:
            serie_type = [vt_dict[x] for x in vt_dict.keys() if df.loc[:, var] in x][0]
            df[var] = df[var].astype(serie_type)

        ## create lists for numerical and categorical features (column names)
        var_num = []  # numerical features
        var_cat = []  # categorical features
        
        for var in df.columns:
            if df[var].dtype == 'object':
                var_cat.append(var)
            elif df[var].dtype == 'int64' or df[var].dtype == 'float64':
                var_num.append(var)
            else:
                pass

        ## save dataframe and list of features to redis db
        var_num_pickle = pickle.dumps(var_num)
        var_cat_pickle = pickle.dumps(var_cat)
        df_pickle = pickle.dumps(df)

        r.mset({"ind_intermediate_0": 1,     # it is used as indicator to start toher callbacks 
                "var_num_pickle": var_num_pickle,
                "var_cat_pickle": var_cat_pickle,
                "df_pickle": df_pickle,
                })

        ###########
        ### PCA ###
        ###########
        ## df to extract PC options
        df_num = df.loc[:, var_num]

        ## initiate PCA instance (compute PC ...)
        PCA = PC_Analysis(df_num)
        pc_list = [{'label': 'PC_' + str(i), 'value': i} for i in range(1, df_num.shape[1] + 1)]
        pc_list_pickle = pickle.dumps(pc_list)

        variables_option = [{'label': var, 'value': var} for var in list(df_num.columns)]
        variables_option_pickle = pickle.dumps(variables_option)


        ## pickle PCA component and save it in a redis db
        PCA_pickle = pickle.dumps(PCA)

        r.mset({"PCA_instance": PCA_pickle,
                "pc_list_pickle": pc_list_pickle,
                "variables_option_pickle": variables_option_pickle})

        return 1