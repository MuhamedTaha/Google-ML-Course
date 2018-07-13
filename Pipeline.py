from sklearn import datasets 
iris = datasets.load_iris()

x = iris.data
y = iris.target


from sklearn.cross_validation import train_test_split 
x_train, x_test, Y_train, Y_test = train_test_split(x,y,test_size = 0.5)

#from sklearn import tree 
#my_classifier = tree.DecisionTreeClassifier()
from sklearn.neighbors import KNeighborsClassifier 
my_classifier = KNeighborsClassifier()

my_classifier.fit(x_train,Y_train)

predictions = my_classifier.predict(x_test)

from sklearn.metrics import accuracy_score 
print accuracy_score(Y_test, predictions)

# this program load some data "iris data" 
#then split it to data to train the classifier and data to test it 
#and print the accuracy score of the trained classifier
#it also can be used with different type of classifier 