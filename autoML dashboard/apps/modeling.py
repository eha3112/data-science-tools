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

import visions as v

import sys

sys.path.append('src_code/')
from modeling_plot import ROC_model, conf_matrix_plot, plot_feature_importance

import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve, \
    confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import Pool, cv
from catboost import CatBoostClassifier

from app import app

### Initiate Redis client
r = redis.Redis( host='redis', port=6379, db=2 )
## clear the db
#r.flushall()


## Styles
style_com_dropdown = {"background-color": "#696969",
                      'text-align': 'center',
                      "color": 'black',
                      'border-radius': '0.4em',
                      'margin': '0.0em 0% 0.0em 0%'}

style_com_input = {"background-color": "#696969",
                   'text-align': 'center',
                   'height': '2em',
                   'width': '85%',
                   'border-radius': '0.4em',
                   'margin': '0em 10% 0em 5%'}

style_text = {'text-align': 'center', 'margin': '1em 0% 0.2em 0%'}

style_upload = {'text-align': 'center', 'background': '#D3D3D3', 'border-radius': '0.5em'}

style_details = {'border': 'solid'}

style_titles_bg = {'background': '#D3D3D3', 'border-radius': '0.5em',
                   'font-family': "'Playfair Display SC', serif"}

style_section = {'border-radius': '0.5em', 'background': '#D3D3D3'}

layout = html.Div(style={"background-color": "#696969", "color": "black"},
                      children=[
                          # Div used to store data
                          html.Div(id='intermediate-value-mod', style={"display": "none"}),
                          html.Div(id='intermediate-value-mod_1', style={"display": "none"}),
                          # used to initiate the plot of confusion matrix

                          ## Navigation Bar
                          html.Nav(className="navbar navbar-expand bg-dark navbar-dark ", children=[
                              html.Ul(className='navbar-nav', children=[
                                  html.Li(children=dcc.Link('Home', href='/', className="nav-link")
                                          , className="nav-item"),

                                  html.Li(children=dcc.Link('Univariate Analysis', href='/uni_analysis',
                                                            className="nav-link")
                                          , className="nav-item"),
                                  html.Li(children=dcc.Link('Bivariate Analysis', href='/bi_analysis',
                                                            className="nav-link")
                                          , className="nav-item"),
                                  html.Li(children=dcc.Link('PCA', href='/pca', className="nav-link")
                                          , className="nav-item"),
                                  html.Li(children=dcc.Link('Clustering', href='/clustering', className="nav-link")
                                          , className="nav-item"),
                                  html.Li(children=dcc.Link('Modeling', href='/modeling', className="nav-link")
                                          , className="nav-item active")
                              ])
                          ]),

                          ## Training Title
                          html.Div(className="m-2 pl-3",
                                   style=style_titles_bg,
                                   children=[
                                       html.H3(className="mx-auto", children="Training")
                                   ]),

                          # ## Upload bar
                          # html.Div(id="div_upload", className="m-2 ", style=style_upload,
                          #          children=[
                          #              dcc.Upload(id='upload-data', className="btn btn-outline-info",
                          #                         children=html.Div(id="link-upload",
                          #                                           children=['(CSV file) Drag and Drop or ',
                          #                                                     html.A('Select Files')]))
                          #          ]),

                          ## Training Section
                          html.Div(className="m-2 d-flex justify-content-center", style=style_section,
                                   children=[

                                       # filter components
                                       html.Div(className="col-lg-2 mt-3", children=[
                                           dcc.Dropdown(id="id_algo", style=style_com_dropdown,
                                                        options=[{'label': 'CatBoost', 'value': 'CatBoost'},
                                                                 {'label': 'Random Forest', 'value': 'random_forest'},
                                                                 {'label': 'Logistic Regression',
                                                                  'value': 'log_regression'}],
                                                        placeholder="Model"),
                                           html.P("train/validation split", style=style_text),
                                           dcc.Input(id="id_par_tr_val", style=style_com_input,
                                                     type="number", placeholder="train/validation", min=0, max=1,
                                                     step=0.01, value=0.80),
                                           ##### Catboost
                                           html.Details([
                                               html.Summary("Catboost parameters"),
                                               html.P("trees_depth", style=style_text),
                                               dcc.Input(id="id_par_depth", style=style_com_input,
                                                         type="number", placeholder="tree depth", min=1, max=300,
                                                         step=1, value=6),
                                               html.P("iterations", style=style_text),
                                               dcc.Input(id="id_par_iter", style=style_com_input,
                                                         type="number", placeholder="number iterations", min=1,
                                                         max=5000,
                                                         step=1, value=150),
                                               html.P("learning rate", style=style_text),
                                               dcc.Input(id="id_par_lr", style=style_com_input,
                                                         type="number", placeholder="learning_rate", min=0, max=100,
                                                         step=10e-10, value=0.01),
                                           ], title="specify the parameters of Catboost model", contextMenu="helllo"),
                                           ##### Logistic Regression
                                           html.Details([
                                               html.Summary("Lg regression parameters"),
                                               html.P("inverse of regularization", style=style_text),
                                               dcc.Input(id="id_C_lg", style=style_com_input,
                                                         type="number", placeholder="Inverse Regularization", min=0,
                                                         max=100,
                                                         step=0.01, value=1),
                                               html.P("max iteration", style=style_text),
                                               dcc.Input(id="id_max_iter_lg", style=style_com_input,
                                                         type="number", placeholder="max iteration", min=1, max=2000,
                                                         step=1, value=50),
                                           ]),
                                           ##### Random Forest
                                           html.Details([
                                               html.Summary("Random Forest Params"),
                                               html.P("max depth", style=style_text),
                                               dcc.Input(id="id_max_depth_rd", style=style_com_input,
                                                         type="number", placeholder="max tree depth", min=1,
                                                         max=100,
                                                         step=1, value=6),
                                               html.P("number estimators", style=style_text),
                                               dcc.Input(id="id_n_estimators_rd", style=style_com_input,
                                                         type="number", placeholder="max tree depth", min=1,
                                                         max=100,
                                                         step=1, value=30),
                                           ]),

                                           #####################
                                           html.Button(id='submit-button-state', n_clicks=0, children='Train',
                                                       style={  # "background-color": "#696969",
                                                           'text-align': 'center',
                                                           'height': '2em',
                                                           'width': '85%',
                                                           # 'border-radius':'0.4em',
                                                           'margin': '0.5em 10% 1em 5%'},
                                                       className="btn btn-secondary"),

                                           html.P("Classification Threshold", style=style_text),
                                           dcc.Input(id="id_threshold", style=style_com_input,
                                                     type="number", placeholder="learning_rate", min=0, max=1,
                                                     step=10e-10, value=0.5),

                                       ]),
                                       # plot
                                       html.Div(className="col-lg-10 ", children=[
                                           html.Div(children=[
                                               dcc.Graph(id='plot_roc')
                                           ]),
                                           html.Div(className="row m-2", children=[
                                               html.Div(className="col-lg-7", children=[
                                                   dcc.Graph(id='id_plot_conf'),
                                               ]),
                                               html.Div(className="col-lg-5 ", children=[
                                                   dash_table.DataTable(
                                                       id='table_m',
                                                       style_header={'backgroundColor': 'rgb(30, 30, 30)',
                                                                     'textAlign': 'center'},
                                                       style_cell={'backgroundColor': '#696969',
                                                                   'color': 'white',
                                                                   'textAlign': 'left'
                                                                   },
                                                       style_cell_conditional=[{'if': {'column_id': 'metrics'},
                                                                                'textAlign': 'center'}]
                                                   )
                                               ])
                                           ])
                                       ])
                                   ]),

                          ## Training Title
                          html.Div(className="m-2 pl-3",
                                   style=style_titles_bg,
                                   children=[
                                       html.H3(className="mx-auto", children="Interpretation")
                                   ]),

                          html.Div(className="m-2 ", style=style_section,
                                   children=[
                                       dcc.Graph(id="id_plot_feature")
                                   ])
                      ])


##################################
######      Callbacks       ######
##################################
## define a function for encoding and decoding the file
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
# @app.callback(Output('intermediate-value-mod', 'children'),
#               [Input('upload-data', 'contents')],
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
#         r.mset({"var_num_pickle": var_num_pickle,
#                 "var_cat_pickle": var_cat_pickle,
#                 "df_pickle": df_pickle,
#                 "target_pickle": target_pickle})
#         print("data is loaded")
#
#         return 1


@app.callback([Output('plot_roc', 'figure'),
               Output('intermediate-value-mod_1', 'children')],  # Output('id_plot_feature', 'figure')
              [Input('submit-button-state', "n_clicks")],  #
              [State('intermediate-value-mod', 'children'),
               State("id_algo", "value"),
               State("id_par_tr_val", "value"),
               State("id_par_depth", "value"), State("id_par_iter", "value"), State("id_par_lr", "value"),  # CatBoost
               State("id_max_depth_rd", "value"), State("id_n_estimators_rd", "value"),  # Random Forest
               State("id_C_lg", "value"), State("id_max_iter_lg", 'value')])  # Logistic Regression
def generate_roc(n_clicks, ind, algo, train_size, depth, iterations, learning_rate,
                 max_depth_rd, n_estimators_rd,
                 C_lg, max_iter_lg):
    if n_clicks == 0:
        raise PreventUpdate
    else:

        ## Load variables from redis
        var_num_pickle = r.get("var_num_pickle")
        var_num = pickle.loads(var_num_pickle)
        var_cat_pickle = r.get("var_cat_pickle")
        var_cat = pickle.loads(var_cat_pickle)
        df_pickle = r.get("df_pickle")
        df = pickle.loads(df_pickle)
        target_pickle = r.get("target_pickle")
        target = pickle.loads(target_pickle)

        if algo == 'CatBoost':

            ## train/validation split
            X_train, X_test, y_train, y_test = train_test_split(df.loc[:, var_cat + var_num], target,
                                                                train_size=train_size, random_state=42)

            ## Create Pool Objects
            train_pool = Pool(data=X_train, label=y_train, cat_features=var_cat)
            validation_pool = Pool(data=X_test, label=y_test, cat_features=var_cat)

            ## Training Model
            model_params = {'loss_function': 'Logloss',  # The metric to use in training
                            'custom_loss': ['AUC'],  # Metric values to output during training
                            'random_seed': 42,
                            'verbose': False,
                            'class_weights': [1, (target == 0).sum() / (target == 1).sum()],
                            'use_best_model': True,  # use_best_model requires evaluation metric
                            'eval_metric': 'AUC',  # metric used for overfitting detection & best model
                            'metric_period': None  # set metric_period  to speed the training
                            # 'early_stopping_rounds' : 20
                            }

            ## critical parameters
            model_params.update({'depth': depth, 'iterations': iterations, 'learning_rate': learning_rate})

            ## initial model
            model = CatBoostClassifier(**model_params)

            ## training parameters
            training_params = {'X': train_pool,  # y : None .........
                               'eval_set': validation_pool,
                               'verbose': 10,
                               'plot': False,  # only on CPU
                               }

            model.fit(**training_params)

            ## validation (ROC)
            y_proba = model.predict_proba(X_test)

            ## Feature importance
            df_featImp = model.get_feature_importance(data=train_pool, prettified=True, thread_count=-1,
                                                      verbose=False)
            df_featImp_pickle = pickle.dumps(df_featImp)
            r.mset({"df_featImp" + algo: df_featImp_pickle})
            # fig_feature = plot_feature_importance(list(df_featImp['Feature Id']), list(df_featImp['Importances']))
            # fig_feature['layout']['paper_bgcolor'] = 'rgba(0,0,0,0)'
            # fig_feature['layout']['plot_bgcolor'] = 'rgba(0,0,0,0)'

        elif algo == 'log_regression':
            # create dummy variables
            for var in var_cat:
                dummy_var_df = pd.get_dummies(df.loc[:, var], prefix=var)
                df = df.join(dummy_var_df)
            # remove categorical variables
            df = df.drop(var_cat, axis=1)

            X_train, X_test, y_train, y_test = train_test_split(df, target, train_size=train_size, random_state=42)

            params_log = {'C': C_lg, 'class_weight': 'balanced', 'max_iter': max_iter_lg}
            # params_log = {'C': 1, 'class_weight': 'balanced', 'max_iter': 300}
            model_log = LogisticRegression(**params_log)
            model_log.fit(X_train, y_train)

            y_proba = model_log.predict_proba(X_test)
            print("yeaaaaaaaaaaaaaaaaahhhhhhhhhhhh")


        elif algo == 'random_forest':
            # create dummy variables
            for var in var_cat:
                dummy_var_df = pd.get_dummies(df.loc[:, var], prefix=var)
                df = df.join(dummy_var_df)
            # remove categorical variables
            df = df.drop(var_cat, axis=1)

            X_train, X_test, y_train, y_test = train_test_split(df, target, train_size=train_size, random_state=42)

            # max_depth_rd, n_estimators_rd
            # params_RF = {'criterion': 'entropy', 'class_weight': "balanced", 'max_depth': 6, 'n_estimators': 40}
            params_RF = {'criterion': 'entropy', 'class_weight': "balanced", 'max_depth': max_depth_rd,
                         'n_estimators': n_estimators_rd}
            model_RF = RandomForestClassifier(**params_RF)
            model_RF.fit(X_train, y_train)

            y_proba = model_RF.predict_proba(X_test)

            ## Feature importance
            df_featImp = pd.DataFrame(
                {'Feature Id': X_train.columns, 'Importances': model_RF.feature_importances_ * 100})
            df_featImp_pickle = pickle.dumps(df_featImp)
            r.mset({"df_featImp" + algo: df_featImp_pickle})

        ## Plot ROC
        fpr_ctb, tpr_ctb, thresholds = roc_curve(y_test, y_proba[:, 1], pos_label=1)

        ## intiate traces so that to avoid None when plotting
        trace_ctb = go.Scattergl(name='Catboost')
        trace_lg = go.Scattergl(name='Logistic Regression')
        trace_rd = go.Scattergl(name='Random Forest')

        if algo == 'CatBoost':
            trace_ctb = go.Scattergl(x=fpr_ctb, y=tpr_ctb, hovertext=thresholds, mode='lines+markers',
                                     marker={'size': 2}, name='Catboost')
            ## save to db
            trace_ctb_pickle = pickle.dumps(trace_ctb)
            r.mset({'trace_ctb': trace_ctb_pickle})
            if r.get("trace_lg"):
                trace_lg_picke = r.get("trace_lg")
                trace_lg = pickle.loads(trace_lg_picke)
            if r.get("trace_rd"):
                trace_rd_picke = r.get("trace_rd")
                trace_rd = pickle.loads(trace_rd_picke)

        if algo == 'log_regression':
            trace_lg = go.Scattergl(x=fpr_ctb, y=tpr_ctb, hovertext=thresholds, mode='lines+markers',
                                    marker={'size': 2}, name='logistic regression')
            ## save to db
            trace_lg_pickle = pickle.dumps(trace_lg)
            r.mset({'trace_lg': trace_lg_pickle})

            if r.get("trace_rd"):
                trace_rd_picke = r.get("trace_rd")
                trace_rd = pickle.loads(trace_rd_picke)
            if r.get("trace_ctb"):
                trace_ctb_picke = r.get("trace_lg")
                trace_ctb = pickle.loads(trace_ctb_picke)

        if algo == 'random_forest':
            trace_rd = go.Scattergl(x=fpr_ctb, y=tpr_ctb, hovertext=thresholds, mode='lines+markers',
                                    marker={'size': 2}, name='random forest')

            ## save to db
            trace_rd_pickle = pickle.dumps(trace_rd)
            r.mset({'trace_rd': trace_rd_pickle})

            if r.get("trace_lg"):
                trace_lg_picke = r.get("trace_lg")
                trace_lg = pickle.loads(trace_lg_picke)
            if r.get("trace_ctb"):
                trace_ctb_picke = r.get("trace_ctb")
                trace_ctb = pickle.loads(trace_ctb_picke)

        fig_roc = go.Figure(data=[trace_ctb, trace_lg, trace_rd])

        fig_roc.update_layout(title={'text': "ROC curve",
                                     'y': 0.90,
                                     'x': 0.5},
                              yaxis={'title': 'TPR'},
                              xaxis={'title': 'FPR'},
                              font={'size': 15},
                              showlegend=True)

        fig_roc['layout']['paper_bgcolor'] = 'rgba(0,0,0,0)'
        fig_roc['layout']['plot_bgcolor'] = 'rgba(0,0,0,0)'

        # save model & y_proba &
        # validation_pool_pickle = pickle.dumps(validation_pool)
        y_test_pickle = pickle.dumps(y_test)
        X_test_pickle = pickle.dumps(X_test)
        y_proba_pickle = pickle.dumps(y_proba)
        r.mset({  # 'model_pickle':model_pickle,
            'y_proba_pickle': y_proba_pickle,
            'X_test_ct_pickle': X_test_pickle,
            'y_test_ct_pickle': y_test_pickle})

        ### variables to use in confusion matrix
        r.mset({"y_proba_" + algo: y_proba_pickle,
                "y_test_" + algo: y_test_pickle})

        return fig_roc, algo


@app.callback([Output('table_m', 'columns'), Output('table_m', 'data')],
              [Input('intermediate-value-mod_1', 'children'),
               Input('id_threshold', 'value')])
def perfromance(algo_div, threshold):
    if algo_div is None:
        raise PreventUpdate
    else:
        # load the needed variables to compute confusion matrix
        y_test_ct_pickle = r.get("y_test_ct_pickle")
        y_test = pickle.loads(y_test_ct_pickle)
        y_proba_pickle = r.get("y_proba_pickle")
        y_proba = pickle.loads(y_proba_pickle)

        # compute y_pred using the threshold
        threshold_ctb = threshold
        y_pred = np.where(y_proba[:, 1] > threshold_ctb, 1, 0)

        # a dictionary for having the proper names in the dataset
        algo_2_col = {'CatBoost': 'Catboost',
                      'random_forest': 'Random Forest',
                      'log_regression': 'Logistic Regression'}

        # check if performance table exists in db
        if r.get \
                    (
                    "df_perf") is None:  # create the df_perf for the first time and fill it with nan values except the alo column
            temp = {'metrics': ['auc', 'accuracy', 'precision', 'recall', 'f1-score'],
                    'Catboost': [np.nan] * 5,
                    'Random Forest': [np.nan] * 5,
                    'Logistic Regression': [np.nan] * 5}

            df_perf = pd.DataFrame(temp)

            AUC = roc_auc_score(y_test, y_proba[:, 1])
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, pos_label=1)
            recall = recall_score(y_test, y_pred, pos_label=1)
            f1 = f1_score(y_test, y_pred, pos_label=1)

            # update the modified column
            df_perf.loc[:, algo_2_col[algo_div]] = [round(x, 4) for x in [AUC, accuracy, precision, recall, f1]]

        else:
            df_perf_pickle = r.get("df_perf")
            df_perf = pickle.loads(df_perf_pickle)

            ## Evluation metric
            AUC = roc_auc_score(y_test, y_proba[:, 1])
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, pos_label=1)
            recall = recall_score(y_test, y_pred, pos_label=1)
            f1 = f1_score(y_test, y_pred, pos_label=1)

            df_perf.loc[:, algo_2_col[algo_div]] = [round(x, 4) for x in [AUC, accuracy, precision, recall, f1]]

        ## save the updated df_perf
        df_perf_pickle = pickle.dumps(df_perf)
        r.mset({'df_perf': df_perf_pickle})

        columns = [{"name": i, "id": i} for i in df_perf.columns]
        data = df_perf.to_dict('records')

        return columns, data


@app.callback(Output("id_plot_conf", "figure"),
              [Input('intermediate-value-mod_1', 'children'),
               Input('id_threshold', 'value'),
               Input("id_algo", "value")])
def conf_matrix_callback(ind, threshold, algo):
    if ind is None:
        raise PreventUpdate
    else:
        print("hhhhhheeee")
        if r.get("y_proba_" + algo):
            # Confusion_Matrix
            y_proba_pickle = r.get("y_proba_" + algo)
            y_proba = pickle.loads(y_proba_pickle)
            y_test_pickle = r.get("y_test_" + algo)
            y_test = pickle.loads(y_test_pickle)

            y_pred = np.where(y_proba[:, 1] > threshold, 1, 0)

            conf = confusion_matrix(y_test, y_pred, labels=[1, 0])

            df_cm = pd.DataFrame(conf, columns=["y_1", "y_0"], index=["y_1", "y_0"])
            df_cm.index.name = 'Actual Class'
            df_cm.columns.name = 'Predicted Class'

            fig = conf_matrix_plot(df_cm)

            fig['layout']['paper_bgcolor'] = 'rgba(0,0,0,0)'
            fig['layout']['plot_bgcolor'] = 'rgba(0,0,0,0)'
        else:
            fig = go.Figure()

    return fig


@app.callback(Output('id_plot_feature', 'figure'),
              [Input('intermediate-value-mod_1', 'children'),
               Input("id_algo", "value")])
def feature_importance_callback(ind, algo):
    if ind is None:
        raise PreventUpdate
    else:
        if r.get('df_featImp' + algo):
            df_featImp_pickle = r.get('df_featImp' + algo)
            df_featImp = pickle.loads(df_featImp_pickle)
            fig_feature = plot_feature_importance(list(df_featImp['Feature Id']), list(df_featImp['Importances']))
            fig_feature['layout']['paper_bgcolor'] = 'rgba(0,0,0,0)'
            fig_feature['layout']['plot_bgcolor'] = 'rgba(0,0,0,0)'

            return fig_feature
        else:
            return go.Figure()

#
# if __name__ == "__main__":
#     app.run_server(debug=True)
