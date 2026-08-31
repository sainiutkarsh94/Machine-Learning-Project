import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Load model and scaler
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

print("Scaler expects", standard_scaler.n_features_in_, "features")


@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():

    if request.method == 'POST':

        try:
            # Get values from form
            Temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            Classes = float(request.form.get('Classes'))
            Region = float(request.form.get('Region'))

            # Create feature array
            features = np.array(
                [[Temperature,
                  RH,
                  Ws,
                  Rain,
                  FFMC,
                  DMC,
                  ISI,
                  Classes,
                  Region]]
            )

            # Scale input
            scaled_data = standard_scaler.transform(features)

            # Predict
            result = ridge_model.predict(scaled_data)

            return render_template(
                'home.html',
                results=round(result[0], 2)
            )

        except Exception as e:
            return str(e)

    return render_template('home.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)