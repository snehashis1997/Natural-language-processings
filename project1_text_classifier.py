import re
import nltk
import pickle
import numpy as np
from nltk.corpus import stopwords
from sklearn.datasets import load_files
nltk.download('stopwords')

reviews=load_files('txt_sentoken/')
x,y=reviews.data,reviews.target

with open('x.pickle','wb') as f:
    pickle.dump(x,f)

with open('y.pickle','wb') as f:
    pickle.dump(y,f)

with open('x.pickle','rb') as f:
    x=pickle.load(f)

with open('y.pickle','rb') as f:
    y=pickle.load(f)
    
corpus=[]
for i in range(0,len(x)):
    review=re.sub(r'\W',' ',str(x[i]))
    review=review.lower()
    review=re.sub(r'\s+[a-z]\s+',' ',review)
    review=re.sub(r'^[a-z]\s+',' ',review)
    review=re.sub(r'\s+',' ',review)
    corpus.append(review)
    
    
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

vectorizer=CountVectorizer(max_features=3000,min_df=3,max_df=0.6,stop_words=stopwords.words('english'))
x=vectorizer.fit_transform(corpus).toarray()

transformer=TfidfTransformer()
x=transformer.fit_transform(x).toarray()

from sklearn.model_selection import train_test_split
text_train,text_test,sent_train,sent_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(text_train,sent_train)

sent_pred=classifier.predict(text_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(sent_test,sent_pred)


with open('classifier.pickle','wb') as f:
    pickle.dump(classifier,f)

with open('tfidfmodel.pickle','wb') as f:
    pickle.dump(vectorizer,f)

with open('classifier.pickle','rb') as f:
    clf=pickle.load(f)

with open('tfidfmodel.pickle','rb') as f:
    tfidf=pickle.load(f)

sample=["you are a good looking boy"]
result=tfidf.transform(sample).toarray()

print(clf.predict(result))



































