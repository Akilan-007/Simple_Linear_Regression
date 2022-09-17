from flask import Flask, render_template, request
import pickle
import numpy as np

sv = pickle.load(open('headbrain.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    
    data1 = request.form['size']
    tot_data = [[data1]]
    arr = np.array(tot_data,dtype=int)
    pred = sv.predict(arr)
    weight = float(pred)
    html_content = f"<html><head></head><body style='background-color:blue'><center><br><br><h1> MACHINE LEARNING PREDICTION</h1><br><br><h1> SIMPLE LINEAR REGRESSION</h1><br><br><h1> The weight of brain is {weight} grams</h1></center></body></html>"
    with open("templates\prediction.html",'w') as html_file:
        html_file.write(html_content)
    return render_template('prediction.html')

if __name__ == '__main__':
    app.run(debug=True)
