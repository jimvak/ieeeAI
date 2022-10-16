import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split


#load data
digits = datasets.load_digits()

#test code

#################

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
print(data)
#classifier
classifier = svm.SVC(gamma = 0.001)

#split data
X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5, shuffle=False)


#train
classifier.fit(X_train, y_train)
print(classifier)
predicted = classifier.predict(X_test)

print(predicted)

pickle.dump(classifier, open("model.pkl", "wb"))

for i in range(-1,-600,-1):
    if(predicted[i] != y_test[i]):
        print(predicted[i]," / ",y_test[i])

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(y_test, predicted)))

"""
print(digits.images[0])
print(data[0])
plt.gray()
plt.matshow(digits.images[0])
plt.show()
"""
