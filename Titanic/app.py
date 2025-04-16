from flask import Flask, request, render_template
import pickle
import numpy as np
model=pickle.load(open('model.pkl','rb'))

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('input.html', prediction_text='')

@app.route('/predict', methods=['post'])
def predict():
    age=float(request.form['age'])
    sibsp=float(request.form['sibsp'])
    parch=float(request.form['parch'])
    fare=float(request.form['fare'])

    input_data = np.array([[age, sibsp, parch, fare]])

    prediction = model.predict(input_data)

    prediction_text = f"survived/not: {prediction[0]}"
    if prediction_text==1:
        prediction_text='Survived'
    else:
        prediction_text='not survived'

    return render_template('input.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)




    