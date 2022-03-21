
import dash

# note: verify loacal css & js files in assets

external_stylesheets = [
   {'href': "https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/css/bootstrap.min.css",
    'rel': 'stylesheet',
    'integrity': "sha384-9aIt2nRpC12Uk9gS9baDl411NQApFmC26EwAOH8WgZl5MYYxFfc+NcPb1dKGj7Sk",
    'crossorigin': "anonymous"},
    { 'href': "https://fonts.googleapis.com/css2?family=Petrona&display=swap",
      'rel':"stylesheet"},  #  font-family: 'Petrona', serif;
    {'href': "https://fonts.googleapis.com/css2?family=Playfair+Display+SC&display=swap",
    'rel':"stylesheet"},   # font-family: 'Playfair Display SC', serif;
]

external_scripts = [
    {'src': "https://code.jquery.com/jquery-3.5.1.slim.min.js",
     'integrity': "sha384-DfXdz2htPH0lsSSs5nCTpuj/zy4C+OGpamoFVy38MVBnE+IbbVYUew+OrCXaRkfj",
     'crossorigin': "anonymous"
     },
    {"src": "https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js",
     "integrity": "sha384-Q6E9RHvbIyZFJoft+2mJbHaEWldlvI9IOYy5n3zV9zzTtmI3UksdQRVvoxMfooAo",
     "crossorigin": "anonymous"
     },
    {'src': "https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/js/bootstrap.min.js",
     'integrity': "sha384-OgVRvuATP1z7JjHLkuOU7Xw704+h835Lr+6QL9UvYjZE3Ipu6Tp75j7Bh/kR0JKI",
     'crossorigin': "anonymous"
     }]

meta_tags=[
    {'name':"viewport",
    'content':"width=device-width, initial-scale=1",
    }]

# app = dash.Dash(__name__, external_scripts = external_scripts,
#                         external_stylesheets = external_stylesheets,
#                         meta_tags = meta_tags)

app = dash.Dash(__name__, suppress_callback_exceptions=True,
                external_scripts=external_scripts, external_stylesheets=external_stylesheets,
                meta_tags=meta_tags)

server = app.server