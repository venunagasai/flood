"""Web app."""
import flask
from flask import Flask, render_template, request, redirect, url_for
import pickle
import base64
from training import prediction
import requests
import pandas as pd


app = flask.Flask(__name__)

with open('model/flood_landslide_models.pkl', 'rb') as f:
    models = pickle.load(f)
    flood_model = models['flood_model']
    landslide_model = models['landslide_model']


# List of features (21 total - last one is target, not included in form)
features = [
    'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement', 'Deforestation',
    'Urbanization', 'ClimateChange', 'DamsQuality', 'Siltation', 'AgriculturalPractices',
    'Encroachments', 'IneffectiveDisasterPreparedness', 'DrainageSystems', 'CoastalVulnerability',
     'Watersheds', 'DeterioratingInfrastructure', 'PopulationScore',
    'WetlandLoss', 'InadequatePlanning', 'PoliticalFactors'
]

@app.route("/")
@app.route('/index.html')
def index() -> str:
    """Base page."""
    return flask.render_template("index.html")

@app.route('/plots.html')
def plots():
    return render_template('plots.html')

@app.route('/heatmaps.html')
def heatmaps():
    return render_template('heatmaps.html')

@app.route('/chart.html')
def chart():
    return render_template('chart.html')


@app.route('/predicts.html', methods=["GET", "POST"])
def get_predicts():
    if request.method == "POST":
        try:
            input_values = [float(request.form[f]) for f in features]
            test_df = pd.DataFrame([input_values], columns=features)

            # Predict flood
            flood_prediction = flood_model.predict(test_df)[0]
            test_df['FloodProbability'] = flood_prediction

            # Predict landslides
            landslide_prediction = landslide_model.predict(test_df)[0]

            # Construct message
            if flood_prediction == 1:
                result = f"🌊 Flood predicted! Estimated landslides: {landslide_prediction:.0f}"
            else:
                result = f"✅ No flood predicted. But estimated landslides: {landslide_prediction:.0f}"

            return render_template("predicts.html", features=features, prediction=result)
        except Exception as e:
            return f"Error in prediction: {str(e)}"

    return render_template("predicts.html", features=features)

if __name__ == "__main__":
    app.run(debug=True)