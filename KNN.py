import math
#import matplotlib.pyplot as plt

#from sklearn.neighbors import KNeighborsClassifier
from collections import Counter


class KNN:
    def __init__(self,k):
        self.K = k
        self.training_data = []
        self.traning_label = []
        
    #this will take the input and just save it locally
    def fit(self, x, y):
        self.training_data = x
        self.traning_label = y
        

    #calculates the distance between points to make the prediction, this specific way is called the euclidean distance 
    def euclidean_distance(self, point1, point2):
        distance = 0
        for i in range(len(point1)):
            distance += (point1[i] - point2[i]) ** 2
            
        return math.sqrt(distance)
    
    def manhattan_distance(self, point1, point2):
        distance = 0
        for i in range(len(point1)):
            distance += abs(point1[i] - point2[i])
    
    #will take the points we would like to test and evaluate them
    def predict(self, x_test):
        predictions = []
        for test_point in x_test:
            distances = []
            for i, train_point in enumerate(self.training_data):
                #dist = self.euclidean_distance(test_point,train_point)
                dist = self.manhattan_distance(test_point, train_point)
                
                distances.append((dist,self.traning_label[i]))
            
            #tells the computer to sort by distance not by label    
            distances.sort(key = lambda x:x[0])
            
            k_nearest_neighbors = distances[:self.K]
            labels = [label for _, label in k_nearest_neighbors]
            
            
            most_common_label = Counter(labels).most_common(1)[0][0]
            
            predictions.append(most_common_label)
        return predictions
    
if __name__ == "__main__":
    x_train = [[10,9],[1,4],[10,1],[9,1],[10,2],[7,3]]
    y_train = ['F','F','F','V','P','P']
    
    file_reader = open("Data.txt", "r")
    
    
    
    x_test = [[3,5]]
    
    knn = KNN(1)
    knn.fit(x_test,y_train)
    prediction = knn.predict(x_test)
    print(';Predictions: ', prediction)
   
    str = file_reader.read()
    #print(str)