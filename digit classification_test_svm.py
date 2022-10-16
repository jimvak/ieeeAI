import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split


#load data
digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
#classifier = svm.SVC(gamma = 0.01)
#classifier
classifier = pickle.load(open("model.pkl", "rb"))

#split data
X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5, shuffle=False)

#train
#classifier.fit(X_train, y_train)

predicted = classifier.predict(X_test)

for i in range(-1,-600,-1):
    if(predicted[i] != y_test[i]):
        print(predicted[i]," / ",y_test[i])

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(y_test, predicted)))
