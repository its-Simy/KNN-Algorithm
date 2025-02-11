import math
import csv

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
        

    #calculates the distance between points to make the prediction 
    def manhattan_distance(self, point1, point2):
        distance = 0
        for i in range(len(point1)):
            distance += abs(point1[i] - point2[i])
            
        return distance
    
    #will take the points we would like to test and evaluate them
    def predict(self, x_test):
        predictions = []
        for test_point in x_test:
            distances = []
            for i, train_point in enumerate(self.training_data):
                dist = self.manhattan_distance(test_point, train_point)
                
                distances.append((dist,self.traning_label[i]))
            
            #tells the computer to sort by distance not by label    
            distances.sort(key = lambda x:x[0])
            
            k_nearest_neighbors = distances[:self.K]
            labels = [label for _, label in k_nearest_neighbors]
            
            
            most_common_label = Counter(labels).most_common(1)[0][0]
            
            predictions.append(most_common_label)
        return predictions
    
    
    def manipulate_data(self, data):
        data_points = []
        classification = []
        
        divider = len(data[0])-1
    
        for row in data:
            classification.append(row[divider])
            data_points.append(row[:divider])
            
            
        return data_points, classification
    
    def load_data(self, filename):
       mylist = []
       with open(filename) as numbers:
           numbers_data = csv.reader(numbers)
           next(numbers_data)
           
           
           for row in numbers_data:
               mylist.append(row)
               
           return self.manipulate_data(mylist)
    
    
        
    
if __name__ == "__main__":
    
    #The following is going to be the inputs and answer for the tester to see if classification is working
    #5.9,3,5.1,1.8,"Virginica"
    
    new_list = open('iris-dataset.csv','r')
    knn1 = KNN(1)
    x_training_data, y_training_data = knn1.load_data('iris-dataset.csv')
    knn1.fit(x_training_data,y_training_data)
    tester = [5.9,3,5.1,1.8]
    answer = knn1.predict(tester)
    print(answer)
    

    
    
    
    
    
    
    