
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
#from apps import bivariate_analysis, home, clustering, modeling,  PCA, univariate_analysis
from apps import home, univariate_analysis, bivariate_analysis, PCA, clustering, modeling

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/bi_analysis':
        return bivariate_analysis.layout
    elif pathname == '/clustering':
        return clustering.layout
    elif pathname == '/modeling':
        return modeling.layout
    elif pathname == '/pca':
        return PCA.layout
    elif pathname == '/uni_analysis':
        return univariate_analysis.layout
    else:
        return home.layout

if __name__ == '__main__':
    app.run_server(host= '0.0.0.0', port='8050', debug=True)