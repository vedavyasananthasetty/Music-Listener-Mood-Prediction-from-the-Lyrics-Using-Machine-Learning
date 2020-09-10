import pandas as pd
import matplotlib as plt
import numpy as np
from sklearn import linear_model
#from sklearn.model_selection cross_validation
from scipy.stats import norm

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from random import seed
from random import randrange
from csv import reader
import csv
import numpy as np
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer



def process(path):
	msev=[]
	maev=[]
	rsqv=[]
	rmsev=[]
	acyv=[]

	df = pd.read_csv(path,encoding="latin-1")

	x1=np.array(df['Lyrics'].values.astype('U'))
	y1=np.array(df['MoodValue'])
	print(x1)
	print(y1)

	print(x1)
	print(y1)
	X_train, X_test, y_train, y_test = train_test_split(x1, y1,test_size=0.20)
	
	count_vectorizer = CountVectorizer(stop_words='english')
	count_train = count_vectorizer.fit_transform(X_train)                  # Learn the vocabulary dictionary and return term-document matrix.
	count_test = count_vectorizer.transform(X_test)

	tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)    # This removes words which appear in more than 70% of the articles
	tfidf_train = tfidf_vectorizer.fit_transform(X_train) 
	tfidf_test = tfidf_vectorizer.transform(X_test)
	


	model2=DecisionTreeClassifier()
	model2.fit(count_train, y_train)
	y_pred = model2.predict(count_test)
	print("predicted")
	print(y_pred)
	print("test")
	print(y_test)

	result2=open("results/resultCOUNTDT.csv","w")
	result2.write("ID,Predicted Value" + "\n")
	for j in range(len(y_pred)):
	    result2.write(str(j+1) + "," + str(y_pred[j]) + "\n")
	result2.close()
	
	mse=mean_squared_error(y_test, y_pred)
	mae=mean_absolute_error(y_test, y_pred)
	r2=abs(r2_score(y_test, y_pred))
	
	
	print("---------------------------------------------------------")
	print("MSE VALUE FOR DecisionTree COUNT IS %f "  % mse)
	print("MAE VALUE FOR DecisionTree COUNT IS %f "  % mae)
	print("R-SQUARED VALUE FOR DecisionTree COUNT IS %f "  % r2)
	rms = np.sqrt(mean_squared_error(y_test, y_pred))
	print("RMSE VALUE FOR DecisionTree COUNT IS %f "  % rms)
	ac=accuracy_score(y_test,y_pred)
	print ("ACCURACY VALUE DecisionTree COUNT IS %f" % ac)
	print("---------------------------------------------------------")
	
	msev.append(mse)
	maev.append(mae)
	rsqv.append(r2)
	rmsev.append(rms)
	acyv.append(ac*100)

	result2=open('results/COUNTDTMetrics.csv', 'w')
	result2.write("Parameter,Value" + "\n")
	result2.write("MSE" + "," +str(mse) + "\n")
	result2.write("MAE" + "," +str(mae) + "\n")
	result2.write("R-SQUARED" + "," +str(r2) + "\n")
	result2.write("RMSE" + "," +str(rms) + "\n")
	result2.write("ACCURACY" + "," +str(ac) + "\n")
	result2.close()
	
	
	df =  pd.read_csv('results/COUNTDTMetrics.csv')
	acc = df["Value"]
	alc = df["Parameter"]
	colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"]
	explode = (0.1, 0, 0, 0, 0)  
	
	fig = plt.figure()
	plt.bar(alc,acc,color=colors)
	plt.xlabel('Parameter')
	plt.ylabel('Value')
	plt.title(' COUNT DecisionTree Metrics Value')
	fig.savefig('results/COUNTDTMetricsValue.png') 
	plt.pause(5)
	plt.show(block=False)
	plt.close()

	model2=DecisionTreeClassifier()
	model2.fit(tfidf_train, y_train)
	y_pred = model2.predict(tfidf_test)
	print("predicted")
	print(y_pred)
	print("test")
	print(y_test)

	result2=open("results/resultTFIDFDT.csv","w")
	result2.write("ID,Predicted Value" + "\n")
	for j in range(len(y_pred)):
	    result2.write(str(j+1) + "," + str(y_pred[j]) + "\n")
	result2.close()
	
	mse=mean_squared_error(y_test, y_pred)
	mae=mean_absolute_error(y_test, y_pred)
	r2=abs(r2_score(y_test, y_pred))
	
	
	print("---------------------------------------------------------")
	print("MSE VALUE FOR DecisionTree TFIDF IS %f "  % mse)
	print("MAE VALUE FOR DecisionTree TFIDF IS %f "  % mae)
	print("R-SQUARED VALUE FOR DecisionTree TFIDF IS %f "  % r2)
	rms = np.sqrt(mean_squared_error(y_test, y_pred))
	print("RMSE VALUE FOR DecisionTree TFIDF IS %f "  % rms)
	ac=accuracy_score(y_test,y_pred)
	print ("ACCURACY VALUE DecisionTree TFIDF IS %f" % ac)
	print("---------------------------------------------------------")

	msev.append(mse)
	maev.append(mae)
	rsqv.append(r2)
	rmsev.append(rms)
	acyv.append(ac*100)
	

	result2=open('results/TFIDFDTMetrics.csv', 'w')
	result2.write("Parameter,Value" + "\n")
	result2.write("MSE" + "," +str(mse) + "\n")
	result2.write("MAE" + "," +str(mae) + "\n")
	result2.write("R-SQUARED" + "," +str(r2) + "\n")
	result2.write("RMSE" + "," +str(rms) + "\n")
	result2.write("ACCURACY" + "," +str(ac) + "\n")
	result2.close()
	
	
	df =  pd.read_csv('results/TFIDFDTMetrics.csv')
	acc = df["Value"]
	alc = df["Parameter"]
	colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"]
	explode = (0.1, 0, 0, 0, 0)  
	
	fig = plt.figure()
	plt.bar(alc,acc,color=colors)
	plt.xlabel('Parameter')
	plt.ylabel('Value')
	plt.title(' TFIDF DecisionTree Metrics Value')
	fig.savefig('results/TFIDFCOUNTDTMetricsValue.png') 
	plt.pause(5)
	plt.show(block=False)
	plt.close()


	al = ['COUNT','TFIDF']
    
    
	result2=open('results/DTMSE.csv', 'w')
	result2.write("Vectorization,MSE" + "\n")
	for i in range(0,len(msev)):
	    result2.write(al[i] + "," +str(msev[i]) + "\n")
	result2.close()
    
	colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
	explode = (0.1, 0, 0, 0, 0)  
       
    
	#Barplot for the dependent variable
	fig = plt.figure(0)
	df =  pd.read_csv('results/DTMSE.csv')
	acc = df["MSE"]
	alc = df["Vectorization"]
	plt.bar(alc,acc,color=colors)
	plt.xlabel('Vectorization')
	plt.ylabel('MSE')
	plt.title("DecisionTree MSE Value");
	fig.savefig('results/DTMSE.png')
	plt.pause(5)
	plt.show(block=False)
	plt.close()
	    
    
    
	result2=open('results/DTMAE.csv', 'w')
	result2.write("Vectorization,MAE" + "\n")
	for i in range(0,len(maev)):
	    result2.write(al[i] + "," +str(maev[i]) + "\n")
	result2.close()
                
	fig = plt.figure(0)            
	df =  pd.read_csv('results/DTMAE.csv')
	acc = df["MAE"]
	alc = df["Vectorization"]
	plt.bar(alc,acc,color=colors)
	plt.xlabel('Vectorization')
	plt.ylabel('MAE')
	plt.title('DecisionTree MAE Value')
	fig.savefig('results/DTMAE.png')
	plt.pause(5)
	plt.show(block=False)
	plt.close()
    
	result2=open('results/DTR-SQUARED.csv', 'w')
	result2.write("Vectorization,R-SQUARED" + "\n")
	for i in range(0,len(rsqv)):
	    result2.write(al[i] + "," +str(rsqv[i]) + "\n")
	result2.close()
            
	fig = plt.figure(0)        
	df =  pd.read_csv('results/DTR-SQUARED.csv')
	acc = df["R-SQUARED"]
	alc = df["Vectorization"]


	plt.bar(alc,acc,color=colors)
	plt.xlabel('Vectorization')
	plt.ylabel('R-SQUARED')
	plt.title('DecisionTree R-SQUARED Value')
	fig.savefig('results/DTR-SQUARED.png')
	plt.pause(5)
	plt.show(block=False)
	plt.close()
	    
	result2=open('results/DTRMSE.csv', 'w')
	result2.write("Vectorization,RMSE" + "\n")
	for i in range(0,len(rmsev)):
	    result2.write(al[i] + "," +str(rmsev[i]) + "\n")
	result2.close()
      
	fig = plt.figure(0)    
	df =  pd.read_csv('results/DTRMSE.csv')
	acc = df["RMSE"]
	alc = df["Vectorization"]
	plt.bar(alc,acc,color=colors)
	plt.xlabel('Vectorization')
	plt.ylabel('RMSE')
	plt.title('DecisionTree RMSE Value')
	fig.savefig('results/DTRMSE.png')
	plt.pause(5)
	plt.show(block=False)
	plt.close()
    
	result2=open('results/DTAccuracy.csv', 'w')
	result2.write("Vectorization,Accuracy" + "\n")
	for i in range(0,len(acyv)):
	    result2.write(al[i] + "," +str(acyv[i]) + "\n")
	result2.close()
    
	fig = plt.figure(0)
	df =  pd.read_csv('results/DTAccuracy.csv')
	acc = df["Accuracy"]
	alc = df["Vectorization"]
	plt.bar(alc,acc,color=colors)
	plt.xlabel('Vectorization')
	plt.ylabel('Accuracy')
	plt.title('DecisionTree Accuracy Value')
	fig.savefig('results/DTAccuracy.png')
	plt.pause(5)
	plt.show(block=False)
	plt.close()
