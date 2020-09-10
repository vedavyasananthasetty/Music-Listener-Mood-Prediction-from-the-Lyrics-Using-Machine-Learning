import emoji
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.tree import DecisionTreeClassifier

def process(path,input):
	data = pd.read_csv(path, usecols=['Lyrics', 'MoodValue'],encoding="latin-1")
	x=np.array(data['Lyrics'].values.astype('U'))
	y=data['MoodValue']

	print(x)
	print(y)
	X_test=[]
	X_test.append(input)

	tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)    # This removes words which appear in more than 70% of the articles
	tfidf_train = tfidf_vectorizer.fit_transform(x) 
	tfidf_test = tfidf_vectorizer.transform(X_test)


	model2= DecisionTreeClassifier()
	model2.fit(tfidf_train, y)
	y_pred = model2.predict(tfidf_test)
	print("predicted")
	print(y_pred)
	result=""
	if y_pred[0]==1:
		result=emoji.emojize("This is a happy :grinning_face_with_big_eyes: song")  

	if y_pred[0]==0:
		result=emoji.emojize("This is a sad :pensive_face: song")	
	return result


