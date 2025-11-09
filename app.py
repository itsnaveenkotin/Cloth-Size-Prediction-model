#lets import the libraries
import numpy as np
from flask import Flask,render_template,request
import pickle

#lets initialize the web app
app=Flask(__name__)

#lets load the model
model=pickle.load(open('hmodel.pkl','rb'))

#lets define the app route for the home page of the application
@app.route('/')
def homePage():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    #for rendering results on GUI
    int_features=[float(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)
    output=round(prediction[0],2)
    return render_template('index.html',
                           predicted_size='{}m'
                           .format(output))

if  __name__ == '__main__':
    app.run(debug=True)