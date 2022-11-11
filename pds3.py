import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report,mean_squared_error,f1_score
from sklearn.model_selection import cross_val_score,KFold
from sklearn.model_selection import train_test_split

os.chdir(r'C:\Users\sotir\Desktop\TRAINING')
dataset=pd.read_csv('training.csv',sep='\t')

xall=dataset['Text Transcription']
yall=dataset['misogynous']

train, test = train_test_split(dataset, test_size=1000)

ytrain=train['misogynous']
ytest=test['misogynous']
xtrain=train['Text Transcription']
xtest=test['Text Transcription']

#Use the TfidVectorizer we were taught in lecture to split the verses in words, so we can create our model
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(lowercase=True,analyzer='word',stop_words='english')
X = vectorizer.fit_transform(xall)
y = yall

#Get the names of the verses
fnames = vectorizer.get_feature_names()
stopwords=vectorizer.get_stop_words()
stopwords=list(stopwords)
extrawords=fnames[:1190]+fnames[19492:]

vectorizer = TfidfVectorizer(lowercase=True,analyzer='word',stop_words=extrawords+stopwords+['going','did','memeful','ll','ve','com','imgflip','don','just','net','memecenter','make','know','look','memegenerator','quickmeme','demotivational','meme','bestdemotivationalposters','got',])
X = vectorizer.fit_transform(xtrain)
y = ytrain

#Get the names of the verses
fnames = vectorizer.get_feature_names()


from sklearn.svm import SVC
clf=SVC(C=1,kernel='rbf')
clf.fit(X,y)
ypred=clf.predict(vectorizer.transform(xtest))
print(mean_squared_error(ytest, ypred))
print(classification_report(ytest,ypred))

##############################################################################
##############################################################################
##############################################################################
##############################################################################
import matplotlib.pyplot as plt
import seaborn as sns

misogynous=dataset[dataset.misogynous==1]
misogynous=misogynous.iloc[:,2:6]


q=misogynous.sum()
q=pd.DataFrame(q)
q=q.reset_index()

fig, ax = plt.subplots(figsize=(10, 10))
plt.bar(q.iloc[:,0],height=q.iloc[:,1])
ax.set_title('Number of times each misogyny type appeared in the dataset')
plt.xlabel('Types of misogyny')
plt.ylabel('Number of times in the dataset')
plt.show()



from sklearn.feature_extraction.text import CountVectorizer

only=dataset[dataset.misogynous==1]
cv = CountVectorizer(lowercase=True,analyzer='word',stop_words=extrawords+stopwords+['going','did','memeful','ll','ve','com','imgflip','don','just','net','memecenter','make','know','look','memegenerator','quickmeme','demotivational','meme','bestdemotivationalposters','got',])   
cv_fit = cv.fit_transform(only['Text Transcription'])    
word_list = cv.get_feature_names()
word_list=np.asarray(word_list)
# Added [0] here to get a 1d-array for iteration by the zip function. 
count_list = np.asarray(cv_fit.sum(axis=0))[0]

print(dict(zip(word_list, count_list)))
fin=np.stack((word_list,count_list),axis=1)
fin=pd.DataFrame(fin)
fin[1]=fin[1].astype(int)
fin=fin.sort_values(by=[1], ascending=False)

fig, ax = plt.subplots(figsize=(10, 10))
plt.barh(fin.iloc[:30,0],width=fin.iloc[:30,1])
ax.set_title('Most frequent words in the misogyny memes')
ax.legend(labels=['Misogyny Memes'])
plt.xlabel('Number of times in the Dataset')
plt.ylabel('Words in Dataset')
plt.show()


onlyzero=dataset[dataset.misogynous==0]
cv_fit = cv.fit_transform(onlyzero['Text Transcription'])    
word_list = cv.get_feature_names()
word_list=np.asarray(word_list)
# Added [0] here to get a 1d-array for iteration by the zip function. 
count_list = np.asarray(cv_fit.sum(axis=0))[0]

print(dict(zip(word_list, count_list)))
finzero=np.stack((word_list,count_list),axis=1)
finzero=pd.DataFrame(finzero)
finzero[1]=finzero[1].astype(int)
finzero=finzero.sort_values(by=[1], ascending=False)

fig, ax = plt.subplots(figsize=(10, 10))
plt.barh(finzero.iloc[:30,0],width=finzero.iloc[:30,1])
ax.set_title('Most frequent words in memes that are not misogynistic')
ax.legend(labels=['Memes that are not misogynistic'])
plt.xlabel('Number of times in the Dataset')
plt.ylabel('Words in Dataset')
plt.xlim((0,1000))
plt.show()
##############################################################################
##############################################################################
##############################################################################
##############################################################################

from sklearn.svm import SVC
clf=SVC(C=1,kernel='rbf')
clf.fit(X,y)
ypred=clf.predict(vectorizer.transform(xtest))
print(mean_squared_error(ytest, ypred))
print(classification_report(ytest,ypred))

from sklearn.metrics import confusion_matrix
ConfMatrx = confusion_matrix(ytest, ypred)
name1 = ["True $C_1$","True $C_2$"]
pd.DataFrame(ConfMatrx,name1,["Predicted: $C_1$","$C_2$"])

from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(fit_intercept=False)
clf.fit(X,y)
ypred=clf.predict(vectorizer.transform(xtrain))
print(mean_squared_error(ytrain, ypred))
print(classification_report(ytrain,ypred))


from sklearn.naive_bayes import GaussianNB
clf=GaussianNB()
clf.fit(X.toarray(),y)
ypred=clf.predict(vectorizer.transform(xtest).toarray())
print(mean_squared_error(ytest, ypred))
print(classification_report(ytest,ypred))

from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier()
clf.fit(X,y)
ypred=clf.predict(vectorizer.transform(xtest))
print(mean_squared_error(ytest, ypred))
print(classification_report(ytest,ypred))


from sklearn.neural_network import MLPClassifier
clf=MLPClassifier(hidden_layer_sizes=(100,80,60,40,20,10,5,3), activation='tanh', solver='lbfgs', alpha=0.000000000000001)
clf.fit(X,y)
ypred=clf.predict(vectorizer.transform(xtest))
print(mean_squared_error(ytest, ypred))
print(classification_report(ytest,ypred))





##############################################################################
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib


tr=[]
for i in train['file_name']:
    img = tf.keras.utils.load_img( i, color_mode='rgb', target_size=(90,90),  interpolation='nearest')
    tr.append(img)
x=[]
for i in range(9000):
    x.append(tf.keras.preprocessing.image.img_to_array(tr[i]))
x=np.array(x)


te=[]
for i in test['file_name']:
    img = tf.keras.utils.load_img( i, color_mode='rgb', target_size=(90,90),  interpolation='nearest')
    te.append(img)
xte=[]
for i in range(1000):
    xte.append(tf.keras.preprocessing.image.img_to_array(te[i]))
xte=np.array(xte)

x=x.reshape(9000,90*90*3)
xte=xte.reshape(1000,90*90*3)

clf=LogisticRegression()
clf.fit(x,y)
ypred=clf.predict(x)
print(classification_report(ytrain,ypred))
print(mean_squared_error(ytrain, ypred))

from sklearn.metrics import confusion_matrix
ConfMatrx = confusion_matrix(ytest, ypred)
name1 = ["True $C_1$","True $C_2$"]
pd.DataFrame(ConfMatrx,name1,["Predicted: $C_1$","$C_2$"])
