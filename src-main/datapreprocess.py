import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

clean_df = pd.read_csv('/content/clean_resume_data.csv')

clean_df.head()

# feature is resume

clean_df.shape

"""Exploring categories(Visualization)"""

clean_df['Category'].value_counts()

plt.figure(figsize = (10,15))
sns.countplot(clean_df['Category'])
plt.xticks(rotation=90)
plt.show()

# imbalanced data so need to do data balancing

clean_df['Category'].unique()

clean_df['Category'].value_counts()

counts = clean_df['Category'].value_counts()
labels = clean_df['Category'].unique()

plt.figure(figsize = (15,10))
plt.pie(counts, labels = labels,autopct='%1.1f%%',shadow = True,colors=plt.cm.Pastel1(np.linspace(0,1,3)))
plt.legend(loc = 'upper left' )

plt.show()

"""Balance Dataset"""

clean_df['Category'].value_counts()

from sklearn.utils import resample

max_count = clean_df['Category'].value_counts().max()
balanced_data = []

for category in clean_df['Category'].unique():
  print(category)

for category in clean_df['Category'].unique():
  category_data = clean_df[clean_df['Category'] == category]
  if len(category_data)<max_count:
    balanced_category_data = resample(category_data, replace = True, n_samples = max_count, random_state=42)
  else:
    balanced_category_data = resample(category_data, replace = False, n_samples = max_count, random_state = 42 )
  balanced_data.append(balanced_category_data)
balanced_df = pd.concat(balanced_data)

balanced_df

balanced_df.isnull().sum()

balanced_df.dropna(inplace=True)

"""TrainTestSPlit"""

X=balanced_df['Feature']
y=balanced_df['Category']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

"""Encoding"""

X_train

# using tfidf instead of word2vec

tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
X_test_tfidf

"""Training And Evaluation"""

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_tfidf, y_train)

# eval
y_pred = rf_classifier.predict(X_test_tfidf)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

y_pred

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(15,10))
sns.heatmap(cm, annot=True, fmt='d', cmap='plasma', xticklabels=rf_classifier.classes_,yticklabels = rf_classifier.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

"""Predictive system"""

import re
def clean_resume(text):
  clean_text = re.sub('http\S+\s*', ' ', text)
  clean_text = re.sub('RT|cc', ' ', clean_text)
  clean_text = re.sub('#\S+', '', clean_text)
  clean_text = re.sub('@\S+', '', clean_text)
  clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
  clean_text = re.sub(r'[^x00-x7f]', '', clean_text)
  return clean_text

def pred_category(resume_text):
  resume_text = clean_resume(resume_text)
  resume_tfidf = tfidf_vectorizer.transform([resume_text])
  predicted_category = rf_classifier.predict(resume_tfidf)[0]
  return predicted_category

"""test"""

resume_file = '''Experienced HR Manager with over 10 years of expertise in recruitment, employee relations, and benefits administration. Proven track record in developing and implementing HR strategies that improve employee engagement and performance. Skilled in conflict resolution, training and development, and compliance with labor laws. Adept at managing HR projects and leading teams to achieve organizational goals. Strong communication and interpersonal skills, with a passion for fostering a positive workplace culture.
'''
pred_category = pred_category(resume_file)
print(pred_category)

# import pickle
# pickle.dump(rf_classifier,open('models/rf_classifier.pkl','wb'))
# pickle.dump(tfidf_vectorizer,open('models/tfidf_vectorizer.pkl','wb'))