import math
import csv

from collections import Counter


class KNN:
    def __init__(self,k):
        self.K = k
        self.training_data = []
        self.training_label = []
        
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
                
                distances.append((dist,self.training_label[i]))
            
            #tells the computer to sort by distance not by label    
            distances.sort(key = lambda x:x[0])
            
            k_nearest_neighbors = distances[:self.K]
            labels = [label for _, label in k_nearest_neighbors]
            
            
            most_common_label = Counter(labels).most_common(1)[0][0]
            
            predictions.append(most_common_label)
        return predictions
    
    
    #grabs the data from the csv file, then divides it into 2 different lists, one with the data points, the other with the answer
    #in the middle i need to manipulate the datapoints into floats to save them properly
    def manipulate_data(self, data):
        data_points = []
        classification = []
        float_values = []
        
        divider = len(data[0])-1
    
        for row in data:
            classification.append(row[divider])
            
            for value in row[:divider]:
                float_values.append(float(value))
                
            data_points.append(row[:divider])
            
            
        return data_points, classification
    
    
    #open csv file, then appends everything except the header into a list to then call another method to manipulate the data provided
    def load_data(self, filename):
       mylist = []
       with open(filename) as numbers:
           numbers_data = csv.reader(numbers)
           
           next(numbers_data) #skips the header so it can start reading the data
           
           for row in numbers_data:
               mylist.append(row)
               
           return self.manipulate_data(mylist)
    #Converts the data given that is a string into a float   
    def string_to_float(self, data):
        return [float (x) for x in data.split(",")]
    
    
        
    
if __name__ == "__main__":
    
    #The following is going to be the inputs and answer for the tester to see if classification is working
    #5.9,3,5.1,1.8,"Virginica"
    
    #new_list = open('iris-dataset.csv','r')
    knn1 = KNN(3)
    
    #loads the data
    x_training_data, y_training_data = knn1.load_data('iris-dataset.csv')
    
    #puts the data into the algorithm to use for training
    knn1.fit(x_training_data,y_training_data)
    
    #inputted information
    input = "5.9,3,5.1,1.8"
    
    #converts the string into a float and seperates data
    tester = knn1.string_to_float(input)
    
    #makes the prediction
    answer = knn1.predict(tester)
    
    #prints the prediction
    print(answer)
    

    
    
    
    
    
    
    