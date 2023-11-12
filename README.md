# Data Mining Coding Assignments

## [HW 1 - Data Preprocessing](https://github.com/arminZolfaghari/Data-Mining-Assignments/tree/main/HW%201%20-%20Data%20Preprocessing)
This assignment is about preprocessing on ```iris-dataset```. I use famous libraries such as ```pandas```, ```scikit-learn```, and ```Matplotlib``` for this purpose.
#### Steps:
1. Handle ```missing values```, find ```NaN values```, and fill them with proper values or remove them.
2. Convert categorical features to numerical features by Label Encoding and ```One Hot Encoding```.
3. Normalize data by using ```StandardScaler```.
4. Reduce dimension with ```PCA (Principal Component Analysis)```.
5. Visualize data with ```Matplotlib```.

   
## [HW 2 - Classification](https://github.com/arminZolfaghari/Data-Mining-Assignments/tree/main/HW%202%20-%20Classification)
This assignment is about creating a ```neural network``` to classify data in ```make_circles``` dataset and ```fashion_mnist``` dataset.
#### Steps in Part 1 (```make_circles``` dataset):
1. Make 1500 circles.
2. Split train and test dataset.
3. Create a ```neural network```: without an activation function, use ```binary_crossentropy``` for loss function, and use ```Adam optimizer```.
4. Create a ```neural network```: with a linear activation function, use ```binary_crossentropy``` for the loss function, and use ```Adam optimizer```.
5. Create a ```neural network```: with a nonlinear activation function, use ```mean_squared_error``` for the loss function and use ```Adam optimizer```.
6. Create a ```neural network```:  with a nonlinear activation function, use ```binary_crossentropy``` for the loss function, and use a manual learning rate instead of Adams.
7. Train the models.
8. Plot loss and accuracy.

#### Steps in Part 2 (```fashion_mnist``` dataset):
1. Load the dataset.
2. Split train and test dataset.
3. Create a ```convolutional neural network```: with ```Adam optimizer``` and ```categorical_crossentropy```.
4. Train the model.
5. Plot loss and accuracy.
6. Print ```confusion_matrix``` and ```classification_report```.


## [HW 3 - Clustering & Association Rules](https://github.com/arminZolfaghari/Data-Mining-Assignments/tree/main/HW%203%20-%20Clustering%20%26%20Association%20Rules)
#### Steps Part 1 (Clustering):
1. Working with ```KMeans``` library from ```sklearn.cluster``` and plotting the result.
2. Determining the efficient number of clusters using the ```elbow ``` method.
3. Analyzing the performance of ```KMeans``` on clustering the complex datasets. 
4. Working with ```load_digits``` dataset and clustering it.
5. Dimension reduction with the ```Isomap``` method.
6. Using the ```DBSCAN``` algorithm to classify two datasets.
7. Determining efficient value for ```Epsilon``` and ```MinPts```.

#### Part 2 (Association Rules):
1. Work with the ```Apriori``` algorithm.
2. Preprocess the dataset.
3. Find ```frequent_items``` and print them.
4. Extract association rules.

## Contact
If you have any questions, feel free to ask me: </br>
:envelope_with_arrow:		arminzolfagharid@gmail.com
