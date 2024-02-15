from flask import Flask ,render_template,request,jsonify,session
from flask import Flask, render_template, url_for, request
import pandas as pd
import joblib
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.naive_bayes import MultinomialNB
#import sqlite3 as sql
#import base64
#from sklearn.preprocessing import LabelEncoder
#from flask_bootstrap import Bootstrap
import numpy as np
#from sklearn.utils import shuffle
import os
from flask import Flask, render_template, request, url_for,send_from_directory
import os
#import tensorflow as tf
#from geo import getTweetLocation


app = Flask(__name__)
#app.secret_key = 'any random string'
#PEOPLE_FOLDER = os.path.join('static', 'people_photo')

data=pd.read_csv('./ttt.csv')
model=joblib.load(open('./RansomwareDetection_model','rb'))


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')




@app.route('/data', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        number = request.form['number']
        number=int(number)
    # Assuming you have a dataset 'data' with columns and values
    d1=data.iloc[number:number+1,:]
    cl=data.iloc[1:1,:]
    cl=cl.keys().to_list()
    d2=np.array(d1)

    for i in d2:
        d22=i


    # Pass the dataset to the HTML template
    return render_template('data.html', data=d22,columns=cl,number=number)

@app.route('/result',methods = ['POST'])
def result():

    if request.method == 'POST':
        Characteristics = request.form['Characteristics']
        MajorLinkerVersion = request.form['MajorLinkerVersion']
        SizeOfCode = request.form['SizeOfCode']
        SizeOfInitializedData = request.form['SizeOfInitializedData']
        ImageBase = request.form['ImageBase']
        MajorOperatingSystemVersion = request.form['MajorOperatingSystemVersion']
        MinorOperatingSystemVersion = request.form['MinorOperatingSystemVersion']
        MajorImageVersion = request.form['MajorImageVersion']
        CheckSum = request.form['CheckSum']
        DllCharacteristics = request.form['DllCharacteristics']
        SectionsNb = request.form['SectionsNb']
        SectionsMeanEntropy = request.form['SectionsMeanEntropy']
        SectionsMaxEntropy = request.form['SectionsMaxEntropy']
        ImportsNbDLL = request.form['ImportsNbDLL']
        ImportsNbOrdinal = request.form['ImportsNbOrdinal']
        ResourcesNb = request.form['ResourcesNb']
        ResourcesMeanEntropy = request.form['ResourcesMeanEntropy']
        ResourcesMinEntropy = request.form['ResourcesMinEntropy']
        ResourcesMeanSize = request.form['ResourcesMeanSize']
        LoadConfigurationSize = request.form['LoadConfigurationSize']
        VersionInformationSize = request.form['VersionInformationSize']
                  
        input_data = np.array([[Characteristics, MajorLinkerVersion, SizeOfCode,
                                SizeOfInitializedData, ImageBase, MajorOperatingSystemVersion,
                                  MinorOperatingSystemVersion, MajorImageVersion, CheckSum,
                                  DllCharacteristics, SectionsNb, SectionsMeanEntropy,
                                  SectionsMaxEntropy, ImportsNbDLL, ImportsNbOrdinal, ResourcesNb,
                                  ResourcesMeanEntropy, ResourcesMinEntropy, ResourcesMeanSize,
                                  LoadConfigurationSize, VersionInformationSize]])

        # Reshape the data to a 2D array (required by some models)
        #input_data = input_data.values.reshape(1, -1)

        # Make predictions
        result = model.predict(input_data)

        # Display the result
        print(result)
 
        return render_template('result.html',result=result)

    
@app.route('/input', )
def input():

    # Assuming you have a dataset 'data' with columns and values
    data=pd.read_csv('./testingdatas.csv')
    cl=data.iloc[1:1,:]
    cl=cl.keys().to_list()


    # Pass the dataset to the HTML template
    return render_template('input.html', columns=cl)

if __name__ == '__main__':
   app.run(debug = True )
