import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt


def load_data(folder_path):
    data = []
    labels = []
    for filename in os.listdir(folder_path):
        with open(os.path.join(folder_path, filename), 'r', encoding='latin-1') as file:
            content = file.read()
            data.append(content)
            labels.append("spm" in filename)
    return data, labels

# antrenare
train_data = []
train_labels = []

for category in ["bare", "lemm", "lemm_stop", "stop"]:
    for i in range(1, 10):
        folder_path = f"C:\\Users\\rusu_\\Desktop\\lingspam\\lingspam_public/{category}/part{i}"
        data, labels = load_data(folder_path)
        train_data.extend(data)
        train_labels.extend(labels)

test_data, test_labels = load_data("C:\\Users\\rusu_\\Desktop\\lingspam\\lingspam_public\\bare\\part10")

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_data)
X_test = vectorizer.transform(test_data)

classifier = MultinomialNB()

classifier.fit(X_train, train_labels)

predictions = classifier.predict(X_test)

accuracy = accuracy_score(test_labels, predictions)
print(f"acuratetea pe setul de testare: {accuracy}")

loo = LeaveOneOut()
cv_results = cross_val_score(classifier, X_train, train_labels, cv=loo, n_jobs=-1)

plt.figure(figsize=(10, 5))
plt.plot(range(len(cv_results)), cv_results, marker='o')
plt.title('rezultatele validarii Leave-One-Out')
plt.xlabel('fold')
plt.ylabel('acuratețe')
plt.show()
