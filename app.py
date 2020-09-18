#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask,render_template,url_for,request
import pickle
import joblib

filename = 'xgb_model.pkl'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('tfidf.pkl','rb'))


# In[ ]:


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data)
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run()


# In[ ]:





# In[ ]:




