import os
# This line tells Keras to behave if there are versioning jitters
os.environ['TF_USE_LEGACY_KERAS'] = '0' 

from sklearn.preprocessing import MinMaxScaler
import os
from flask import Flask, render_template
from flask_bootstrap import Bootstrap5
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
from tensorflow import keras

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
bootstrap5 = Bootstrap5(app)

# --- GLOBAL INITIALIZATION (Only happens once) ---
# 1. Load the model globally
model = keras.models.load_model('pima_model.keras')

# 2. Setup the scaler globally so we don't read the CSV on every click
data = pd.read_csv('./diabetes.csv', sep=',')
X = data.values[:, 0:8]
scaler = MinMaxScaler()
scaler.fit(X)

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
        # Process input
        X_test = np.array([[float(form.preg.data), float(form.glucose.data),
                          float(form.blood.data), float(form.skin.data),
                          float(form.insulin.data), float(form.bmi.data),
                          float(form.dpf.data), float(form.age.data)]])
        
        # Scale data using the global scaler
        X_test_scaled = scaler.transform(X_test)
        
        # Run prediction using the global model
        prediction = model.predict(X_test_scaled)
        res = prediction[0][0]
        res = np.round(res * 100, 2)
        
        return render_template('result.html', res=res)
    
    return render_template('prediction.html', form=form)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)