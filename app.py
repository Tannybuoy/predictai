from flask import Flask, redirect, url_for, render_template, request
import numpy as np
import pickle

model=pickle.load(open('iri.pkl', 'rb'))

app=Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/trial", methods=['GET', 'POST'])
def trial():
    return render_template("trial.html")

@app.route("/predict", methods=['POST','GET'])
def man():
    if request.method=='POST':
        data=request.form['nm']
        data1 = int(request.form['a'])
        data2 = int(request.form['b'])
        data3 = int(request.form['c'])
        data4 = int(request.form['d'])
        data5 = int(request.form['e'])
        arr=np.array([[data1, data2, data3, data4, data5]])
        pred=model.predict(arr)
        return render_template("after.html", data=pred, name=data)
    else:
        return render_template("trial.html")

@app.route("/<usr>", methods=['POST','GET'])
def user(usr):
    return f"<h1>{usr}</h1>"

if __name__ =="__main__":
    app.run(debug=True)
