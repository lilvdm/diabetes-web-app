import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
from flask import Flask, render_template
from flask_bootstrap import Bootstrap5
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
bootstrap5 = Bootstrap5(app)

# --- GLOBAL VARIABLES ---
# Load these outside the route for better performance
try:
    data = pd.read_csv('./diabetes.csv', sep=',')
    X = data.values[:, 0:8]
    scaler = MinMaxScaler()
    scaler.fit(X)
    
    # Load the model - using the specific path for Render
    model_path = os.path.join(os.getcwd(), 'pima_model.keras')
    model = keras.models.load_model(model_path)
    print("Model and Scaler loaded successfully!")
except Exception as e:
    print(f"Error during startup: {e}")
    model = None

class LabForm(FlaskForm):
    preg = StringField('# Pregnancies', validators=[DataRequired()])
    glucose = StringField('Glucose', validators=[DataRequired()])
    blood = StringField('Blood pressure', validators=[DataRequired()])
    skin = StringField('Skin thickness', validators=[DataRequired()])
    insulin = StringField('Insulin', validators=[DataRequired()])
    bmi = StringField('BMI', validators=[DataRequired()])
    dpf = StringField('DPF Score', validators=[DataRequired()])
    age = StringField('Age', validators=[DataRequired()])
    submit = SubmitField('Submit')

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def lab():
    form = LabForm()
    if form.validate_on_submit():
        if model is None:
            return "Model not loaded. Check server logs."
            
        X_test = np.array([[float(form.preg.data),
                          float(form.glucose.data),
                          float(form.blood.data),
                          float(form.skin.data),
                          float(form.insulin.data),
                          float(form.bmi.data),
                          float(form.dpf.data),
                          float(form.age.data)]])
        
        X_test_scaled = scaler.transform(X_test)
        prediction = model.predict(X_test_scaled)
        res = float(np.round(prediction[0][0] * 100, 2))
        
        return render_template('result.html', res=res)
    
    return render_template('prediction.html', form=form)

if __name__ == '__main__':
    # This is key for Render's port detection
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)