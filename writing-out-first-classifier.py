from scipy.spatial import distance 
#import random
# a is a point from the trainind data 
# b is a point from testind data 
# this function return the distance between a & b 
def euc(a,b):
	return distance.euclidean(a,b)

# this class is a first trial of building a classifier using the nearest neighbour algorithm
# this algorithm pros and cons:
# pros: 1) relatively simple
# cons: slow as it should itirate in all the data 
#		some data is more informatics than other types of data and that is hard to implement using this algorithm
class ScrappyKNN():
	def fit(self,x_train,Y_train):
		self.x_train = x_train
		self.Y_train = Y_train

	def predict(self, x_test):
		predictions = []
		for row in x_test:
#			label = random.choice(self.Y_train)
			label = self.closest(row)
			predictions.append(label)
		return predictions 

	def closest(self,row):
		best_dist = euc(row,self.x_train[0])
		best_index = 0
		for i in range(0,len(self.x_train)):
			dist = euc(row,self.x_train[i])
			if dist < best_dist:
				best_dist = dist
				best_index = i
		return self.Y_train[best_index]		


from sklearn import datasets 
iris = datasets.load_iris()

x = iris.data
y = iris.target


from sklearn.cross_validation import train_test_split 
x_train, x_test, Y_train, Y_test = train_test_split(x,y,test_size = 0.5)

#from sklearn import tree 
#my_classifier = tree.DecisionTreeClassifier()
#from sklearn.neighbors import KNeighborsClassifier 
my_classifier = ScrappyKNN()

my_classifier.fit(x_train,Y_train)

predictions = my_classifier.predict(x_test)

from sklearn.metrics import accuracy_score 
print accuracy_score(Y_test, predictions)