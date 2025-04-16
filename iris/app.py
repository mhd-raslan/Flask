from flask import Flask, request, render_template
import pickle
import numpy as np

app=Flask(__name__)
model=pickle.load(open('RF.pkl','rb'))

@app.route('/')
def home():
    return render_template('u_input.html', prediction_text='')

@app.route('/predict', methods=['post'])
def predict():
    
    # Retrieve the input values from the form
    sepal_length = float(request.form['Sepal_Length'])
    sepal_width = float(request.form['Sepal_Width'])
    petal_length = float(request.form['Petal_Length'])
    petal_width = float(request.form['Petal_Width'])

    # Prepare the input data for prediction
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    
    # Make the prediction
    prediction = model.predict(input_data)

    # Get the predicted class
    prediction_text = f"Predicted Flower Class: {prediction[0]}"

    return render_template('u_input.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
