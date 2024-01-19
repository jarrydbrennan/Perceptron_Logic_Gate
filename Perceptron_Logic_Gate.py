import codecademylib3_seaborn
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

#Creating & Visualizing AND/OR/XOR Data
data = [[0,0],[0,1],[1,0],[1,1]]
# #AND
# labels = [0,0,0,1]
# #OR
# labels = [0,1,1,1]
#XOR
labels = [0,1,1,0]

plt.scatter([point[0] for point in data],[point[1] for point in data], c = labels)
plt.show()
plt.clf()

#Build Perceptron
classifier = Perceptron(max_iter = 40, random_state = 22)
classifier.fit(data,labels)
print(classifier.score(data,labels))

#Visualizing the Perceptron (decision boundary)
x_values = np.linspace(0,1,100)
y_values = np.linspace(0,1,100)
point_grid = list(product(x_values,y_values))
distance = classifier.decision_function(point_grid)
abs_distance = [abs(pt) for pt in distance]
distance_matrix = np.reshape(abs_distance, (100,100))

heatmap = plt.pcolormesh(x_values,y_values,distance_matrix)
plt.colorbar(heatmap)
plt.show()
plt.clf()