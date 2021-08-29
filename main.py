import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import seaborn as sns
import matplotlib.pyplot as plt

#nltk.download('stopwords')

#Read and load the data
messages = pd.read_csv("Resources/spam.csv")
messages.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)
messages = messages.loc[:, ~messages.columns.str.contains('^Unnamed')]
print(messages)

#Data cleaning and preprocessing
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

print(corpus[0:10])

# Creating the Bag of Words model
cv = CountVectorizer(max_features=4000)
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(messages['label'])
y = y.iloc[:, 1].values


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 7)

# Training model using Naive bayes classifier
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred = spam_detect_model.predict(X_test)
print("\ny_test : ", y_test[0:20],"\ny_pred : ",y_pred[0:20])
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy :", accuracy)

#Plot the Confusion matrix
confMatrix = confusion_matrix(y_test,y_pred)

ax = plt.subplot()
sns.heatmap(confMatrix, annot=True, fmt='g', ax=ax)  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['Ham', 'Spam'])
ax.yaxis.set_ticklabels(['Ham', 'Spam'])
plt.show()





