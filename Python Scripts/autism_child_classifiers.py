import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import graphviz

#data = pd.read_csv("childDataAgeOnly.csv")
data = pd.read_csv("childA3a4a7a9a10.csv")

dataX = data.drop(['class'], 1)
dataY = data['class']


"""split data into training set and test set"""
x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size = 0.2, random_state = 40)


"""decision tree"""
dcTree = tree.DecisionTreeClassifier(criterion='entropy')
dcTree = dcTree.fit(x_train, y_train)
prediction = dcTree.predict(x_test)
accuracy = dcTree.score(x_test, y_test)
dot_data = tree.export_graphviz(dcTree, out_file = None, feature_names = data.columns[0: 4], class_names = data.columns[4],
                                filled = True, rounded = True, special_characters = True)
graph = graphviz.Source(dot_data)
graph.render("autism", view = True)
#print graph


"""naive bayes"""
NB = MultinomialNB(alpha = 1, class_prior = None, fit_prior = False)
scores = cross_val_score(NB, x_train, y_train, cv = 10)
#print scores.mean(), scores.std()*2
NB.fit(x_train, y_train)
prediction = NB.predict(x_test)
accuracy = NB.score(x_test, y_test)


"""normalize data for following learning algorithms: LR, SVM, FNN, KNN"""
scaler = StandardScaler()
###compute the mean and std to be used for later scaling
scaler.fit(x_train)

###Perform standardization by centering and scaling
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


"""logistic regression"""
LR = LogisticRegression(penalty = 'l2')
scores = cross_val_score(LR, x_train, y_train, cv = 10)
#print scores.mean(), scores.std()*2
LR.fit(x_train, y_train)
prediction = LR.predict(x_test)
accuracy = LR.score(x_test, y_test)


"""svm"""
svm = SVC(C=1, kernel = 'linear', degree = 1, cache_size = 200, max_iter = 100000)
svm = SVC(C=1, kernel = 'poly', degree = 3, cache_size = 200, max_iter = 100000) ##this is for A3,A4,A7,A9,A10 only
scores = cross_val_score(svm, x_train, y_train, cv = 10)
#print scores.mean(), scores.std()*2
svm.fit(x_train, y_train)
prediction = svm.predict(x_test)
accuracy = svm.score(x_test, y_test)


"""feed_forward neural network"""
###use 1 hidden layer with 10 neurons here
NN = MLPClassifier(hidden_layer_sizes = (10), max_iter = 1000, learning_rate_init = 0.05, momentum = 0.1) 
scores = cross_val_score(NN, x_train, y_train, cv = 10)
#print scores.mean(), scores.std()*2
NN.fit(x_train, y_train)
prediction = NN.predict(x_test)
accuracy = NN.score(x_test, y_test)


"""k-NN"""
knn = KNeighborsClassifier(n_neighbors = 5) ##n_neighbors = 5 for A3,A4,A7,A9,A10 only
scores = cross_val_score(knn, x_train, y_train, cv = 10)
#print scores.mean(), scores.std()*2
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
accuracy = knn.score(x_test, y_test)



"""print out classification accuracy, report and confusion matrix"""
print accuracy
print classification_report(y_test, prediction)
print confusion_matrix(y_test, prediction)
