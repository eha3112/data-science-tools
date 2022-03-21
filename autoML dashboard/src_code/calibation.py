


from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibratedClassifierCV
import plotly.graph_objects as go

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

class model_calibration:
    """ Model Calibration """

    def __init__(self, model):
        """ Initiate the model to calibrate """
        self.model = model

    def reliability_diagram(self, X, y, n_bins):
        """ Plot Reliability Diagram
        parameter:
        ----------
            X: [np.array / pd.DataFrame, shape(n_individuals, n_features)] test data
            y: [np.array / pd.Series, shape(n_individuals)] target vales
            n_bins: [int] the number of bins in the reliability diagram
        return:
        -------
            fig: [plotly.graph_objs._figure.Figure] reliability plot
        """

        # predict probabilities
        y_proba = self.model.predict_proba(X)[:, 1]
        # reliability diagram
        fop, mpv = calibration_curve(y, y_proba, n_bins=n_bins)

        trace = go.Scatter(x=mpv, y=fop)

        fig = go.Figure(data= [trace])
        fig.update_layout(title = {'text':"Calibration curve (Reliability diagram)", 'y': 0.90, 'x': 0.48,
                                   'font':{'size':23}},
                          xaxis = {'title':"Expected Positive (bin index)"},
                          yaxis = {'title':"Observed Positive in bin"},
                          shapes=[{'type': "line", 'xref': "x", 'yref': "y", 'x0': 0, 'y0': 0, 'x1': 1, 'y1': 1,
                                 'line_color': "Red", 'line':{'width':2, 'dash':"dot"}}])
        return fig

    def calibrate(self, X, y, method='sigmoid', cv=3):
        """
        parameter:
        ----------
            X: [np.array / pd.DataFrame, shape(n_individuals, n_features)] test data
            y: [np.array / pd.Series, shape(n_individuals)] target vales
            method: [string] calibration method ("sigmoid" or "isotonic")
        return:
            calibrated: [sklearn.calibration.CalibratedClassifierCV ] calibrated model
        """
        calibrated = CalibratedClassifierCV(self.model, method, cv)
        calibrated.fit(X, y)

        return calibrated


    def reliability_diagram_cal(self, model_calibrated, X, y, n_bins):
        """ plot the reliability diagram of the model and its calibrated model
        parameters:
        -----------
            model_calibrated: [sklearn.calibration.CalibratedClassifierCV] calibrated model
            X: [np.array / pd.DataFrame, shape(n_individuals, n_features)] test data
            y: [np.array / pd.Series, shape(n_individuals)] target values
            n_bins: [int] the number of bins in the reliability diagram
        return:
        -------
            fig: [plotly.graph_objs._figure.Figure] reliability plot
        """
        y_proba = self.model.predict_proba(X)[:, 1]
        # reliability diagram
        fop_org, mpv_org = calibration_curve(y, y_proba, n_bins=n_bins)


        y_proba_cal = model_calibrated.predict_proba(X)[:, 1]
        fop_cal, mpv_cal = calibration_curve(y, y_proba_cal, n_bins=n_bins)

        trace = go.Scatter(x=mpv_org, y=fop_org, name='model')
        trace_cal = go.Scatter(x=mpv_cal, y=fop_cal, name='calibrated model', line={'color':'purple'})

        fig = go.Figure(data= [trace, trace_cal])

        fig.update_layout(title = {'text':"Calibration curve (Reliability diagram)", 'y': 0.90, 'x': 0.48,
                                   'font':{'size':23}},
                          xaxis = {'title':"Expected Positive (bin index)"},
                          yaxis = {'title':"Observed Positive in bin"},
                          shapes=[{'type': "line", 'xref': "x", 'yref': "y", 'x0': 0, 'y0': 0, 'x1': 1, 'y1': 1,
                                 'line_color': "Red", 'line':{'width':2, 'dash':"dot"}}])
        return fig