# Classification Task on 'Forest Cover Type Prediction' and then Serving ML models using FastAPI 

- final.ipynb contains code for EDA + Feature Engineering + Classification
- final_shortened.ipynb contains code (with EDA)
- app.py (FastAPI)

## Workflow for Classification

# Classification Task on 'Forest Cover Type Prediction'

This repository contains code for applying classification techniques on the 'Forest Cover Type Prediction' dataset. After performing EDA, data preprocessing, and feature engineering, I have built various classification models to predict forest cover type.

## Dataset

The data includes four wilderness areas located in the Roosevelt National Forest of northern Colorado. Each observation is a 30m x 30m patch. The training set (15120 observations) contains both features and the Cover_Type. The test set contains only the features. I have listed below the name of all the features:

* Elevation - Elevation in meters
* Aspect - Aspect in degrees azimuth
* Slope - Slope in degrees
* Horizontal_Distance_To_Hydrology - Horz Dist to nearest surface water features
* Vertical_Distance_To_Hydrology - Vert Dist to nearest surface water features
* Horizontal_Distance_To_Roadways - Horz Dist to nearest roadway
* Hillshade_9am (0 to 255 index) - Hillshade index at 9am, summer solstice
* Hillshade_Noon (0 to 255 index) - Hillshade index at noon, summer solstice
* Hillshade_3pm (0 to 255 index) - Hillshade index at 3pm, summer solstice
* Horizontal_Distance_To_Fire_Points - Horz Dist to nearest wildfire ignition points
* Wilderness_Area (4 binary columns, 0 = absence or 1 = presence) - Wilderness area designation
The wilderness areas are:

         1 - Rawah Wilderness Area
         2 - Neota Wilderness Area
         3 - Comanche Peak Wilderness Area
         4 - Cache la Poudre Wilderness Area
         
* Soil_Type (40 binary columns, 0 = absence or 1 = presence) - Soil Type designation
* Cover_Type (7 types, integers 1 to 7) - Forest Cover Type designation:
The seven cover types are:

         1 - Spruce/Fir
         2 - Lodgepole Pine
         3 - Ponderosa Pine
         4 - Cottonwood/Willow
         5 - Aspen
         6 - Douglas-fir
         7 - Krummholz

![](forest-cover.png)

1. **Data Preprocessing and EDA**: 

Both train and test dataset are loaded and being analyzed. There are 15120 data with 56 columns in the train set and 565892 data with 55 features in the test set. I have checked for missing values (no missing values are present), measured skewness (most of the features are positively skewed), presented distribution of the data, and other significant statistics. There are also some outliers on the training dataset. I used the logic of extreme outliers as this is a standard and widely used technique for outlier detection to keep as much rows as possible. I dropped data points if they satisfy the following conditions:

- x < Q1 - 3 * IQR
- x > Q3 + 3 * IQR

In EDA, I have illustrated percentage of wilderness area type and the amount of each forest cover type present in each wilderness area. Among the soil types, Soil type 10 and 29 are mostly present in the training data. I have plotted distribution of various features. I calculated and demonstrated the correlation among features. After performing thorough analysis of various plots and characteristics of data, I have done some feature engineering. I have also converted 40 soil types and 4 wilderness area into separate single column and removed the previous columns. By observing correaltion and other information, I have transformed the training data into 14988 rows and 15 columns including the target column.

2. **Splitting the dataset**:

The training dataset is split in a 80-20 ratio for training and validation. I have performed Standard Scaler operation on the data before starting to build the model.

3. **Classification Models**: 

I have used the following algorithms. I have provided brief background details of each technique I used :

* **Logistic Regression**:
Logistic Regression is a linear classification algorithm used for binary and multiclass classification tasks. It models the relationship between the input features and the probability of belonging to a particular class using the logistic function, also known as the sigmoid function. It predicts the probability of an instance belonging to each class and then assigns the instance to the class with the highest probability. In the case of multiclass problems, it uses the one-vs-rest (OvR) strategy to transform the multiclass problem into multiple binary classification tasks, where each class is treated as a separate binary classification problem.

* **Support Vector Machine (SVM)**:
SVM is a powerful algorithm for multiclass classification. It finds the optimal hyperplane that maximizes the margin between classes. SVM is effective in high-dimensional spaces and can handle complex data distributions. It also allows the use of different kernels for nonlinear classification.

* **K-Nearest Neighbors (KNN)**:
KNN is a non-parametric algorithm that classifies data points based on the majority class of their k-nearest neighbors. It's simple to understand and implement. KNN can be sensitive to noisy data and requires careful selection of the appropriate value of k. However, it can be computationally expensive for large datasets and sensitive to the choice of K.

* **Decision Tree**:
Decision Trees are non-linear classification algorithms that recursively split the data into subsets based on the feature that provides the best separation at each node. For multiclass problems, decision trees can handle multiple classes directly without the need for transformation. The class with the highest number of instances in a leaf node is assigned as the prediction for that leaf.

* **AdaBoost (Adaptive Boosting)**:
AdaBoost is an ensemble learning technique that combines weak learners to create a strong learner. It sequentially trains the weak learners, giving higher weights to misclassified samples in each iteration. AdaBoost focuses on difficult-to-classify samples and adapts to complex decision boundaries. It can be used for multiclass problems through the SAMME, SAMME.R algorithm.

* **Light Gradient Boosting Machine (LightGBM)**:
LightGBM is a gradient boosting framework that uses a tree-based learning algorithm. It is designed for efficiency and can handle large-scale datasets efficiently. LightGBM uses a leaf-wise tree growth strategy, leading to faster training times and lower memory usage. It supports multiclass classification through one-vs-rest and one-vs-one strategies.

* **Multi-Layer Perceptron (MLP)**:
MLP is a type of artificial neural network used for multiclass classification. It consists of multiple layers of interconnected neurons and uses backpropagation to learn from the data. MLP can handle multiclass problems by using a softmax activation function in the output layer, which produces a probability distribution over the classes, and the class with the highest probability is chosen as the prediction.

* **Extra Trees Classifier**:
Extra Trees is an extension of the Random Forest algorithm. It builds multiple decision trees and further randomizes the feature selection and node splitting, making it even less prone to overfitting. Extra Trees can handle multiclass problems and can be an effective choice when dealing with high-dimensional data.

* **Ensemble Classifier**:
Ensemble classifiers combine multiple individual classifiers to improve overall performance. Voting and stacking are two popular ensemble techniques used in machine learning.

Voting Ensemble: In a voting ensemble, multiple base classifiers are trained independently on the same training data. During prediction, each base classifier makes its own prediction, and the final prediction is determined by majority voting (classification).
Voting can be done using either hard voting (simple majority) or soft voting (weighted average of probabilities).
Voting ensembles work well when the base classifiers have diverse strengths and weaknesses.

Stacking Ensemble:Stacking, also known as stacked generalization, involves training multiple base classifiers and a meta-classifier (also called the blender or meta-learner). The base classifiers are trained on the same training data, and their predictions are combined to create a new feature set. The meta-classifier is then trained on the new feature set to make the final predictions. Stacking allows the meta-classifier to learn how to best combine the predictions of the base classifiers, potentially improving overall performance. Stacking is more complex than voting but can be more powerful if implemented properly.

4. **Evaluation Metrics**:

For evaluation, I have used Accuracy, Precision, Recall, F1 Score, Cross validation Accuracy, and ROC Score. I want to add about the following parameters while using those metrics.

Macro Average: The macro average calculates the metric independently for each class and then takes the unweighted mean (average) of those class-wise metrics. It treats all classes equally and does not consider class imbalances. Use the "macro" average when you want to give equal importance to each class and you have a balanced dataset.

ROC score: For multiclass classification problems (i.e., more than two classes), the interpretation of ROC-AUC is not as straightforward as in binary classification. ROC-AUC can be computed using the one-vs-rest (OvR) approach, but it may not fully capture the model's performance for all classes simultaneously. In multiclass problems, other metrics like accuracy, precision, recall, and F1-score are more commonly used. In some specific scenarios, ROC-AUC might be relevant even for multiclass problems. For example, if one is interested in evaluating the performance of a multiclass classifier in distinguishing a particular class from all other classes combined, one can use ROC-AUC with the OvR approach. In ROC, a separate binary classifier is trained for each class in One-vs-Rest (OvR) strategy. Each classifier is responsible for distinguishing one class from the rest of the classes. During prediction, the class with the highest confidence (probability) from all binary classifiers is selected as the final predicted class. OvR is computationally efficient and often works well for most multiclass problems.

## Workflow for Serving ML Models with FastAPI

**1. Understanding the background**:

**REST (Representational State Transfer)** is an architectural style and set of principles for designing networked applications. It  has become a dominant approach for building web APIs and web services.

Key Principles of REST:

- **Statelessness:** Each request from the client to the server must contain all the necessary information to understand and process the request. The server does not maintain any client state between requests.

- **Client-Server Architecture:** REST follows a client-server model, where the client is responsible for the user interface and the server handles the business logic and data storage.

- **Uniform Interface:** RESTful APIs have a standardized and consistent interface, achieved through standard HTTP methods (GET, POST, PUT, DELETE) and resource-based URLs.

        * POST: Create a new resource.
        * GET: Read/Retrieve a resource or a list of resources. 
        * PUT: Update an existing resource.
        * DELETE: Delete a resource.

- **Resource-Based:** Everything in REST is considered a resource, and each resource is identified by a unique URL. Clients interact with these resources using HTTP methods.

Benefits of REST

- **Simplicity:** REST's straightforward design makes it easy to understand and implement, reducing the complexity of building APIs.

- **Scalability:** The stateless nature of RESTful services allows them to scale effortlessly to handle a large number of clients.

- **Ease of Integration:** RESTful APIs can be easily integrated into various platforms and programming languages due to their reliance on standard HTTP methods and data formats like JSON and XML.

- **Widespread Adoption:** REST has become the dominant approach for building web APIs, leading to extensive community support and tooling.

**What is FastAPI?**

FastAPI is a modern, fast (high-performance), web framework for building RESTful APIs with Python 3.7+ based on standard Python type hints. FastAPI is based on Pydantic and uses type hints to validate, serialize, and deserialize data.

Pydantic is a Python library for data parsing and validation. It uses the type hinting mechanism of the newer versions of Python (version 3.6 onwards) and validates the types during the runtime.  It provides user friendly errors when data is invalid. Pydantic defines BaseModel class. It acts as the base class for creating user defined models. 

The main thing one need to run a FastAPI application in a remote server machine is an ASGI server program like Uvicorn. ASGI is the spiritual successor of WSGI. It processes requests asynchronously, in the opposite way of WSGI. When requests are processed asynchronously, the beauty is that they don't have to wait for the others before them to finish doing their tasks.

**Brief discussion on some common HTTP methods**

- **GET:** The GET method is used to retrieve data from the server. This is a read-only method, so it has no risk of mutating or corrupting the data. For example, if we call the get method on our API, we’ll get back a list of all to-dos.

- **POST :** The POST method sends data to the server and creates a new resource. The resource it creates is subordinate to some other parent resource. When a new resource is POSTed to the parent, the API service will automatically associate the new resource by assigning it an ID (new resource URI). In short, this method is used to create a new data entry.

- **PUT :** The PUT method is most often used to update an existing resource. If you want to update a specific resource (which comes with a specific URI), you can call the PUT method to that resource URI with the request body containing the complete new version of the resource you are trying to update.

- **PATCH :** The PATCH method is very similar to the PUT method because it also modifies an existing resource. The difference is that for the PUT method, the request body contains the complete new version, whereas for the PATCH method, the request body only needs to contain the specific changes to the resource, specifically a set of instructions describing how that resource should be changed, and the API service will create a new version according to that instruction.

- **DELETE:** The DELETE method is used to delete a resource specified by its URI.

**2. Load the Model**: I have loaded three .pkl models in app.py file. The three models are, Extra Trees Classifer (Top), LightGBM (Second best), Best Ensemble Classifier( Ensemble of top 5 models).

**3. Preprocess the data and load preprocessing steps** : When a user sends input data for prediction, we need to preprocess it using the same steps that were used during training. In this step, I have only performed scaling on the input test data. I have created a file titled 'standard_scaler.pkl' for that.

**4. Define the FastAPI app and endpoints** : I have created a FastAPI app and defined the necessary endpoints such as index, version, predict1 (top classifier), predict2 (2nd best classifier), and predict3 (ensemble classifier) for the prediction task

**5. Handle user input and make predictions**: In each prediction endpoint, receive the user's input data and preprocess it using the loaded preprocessing steps. Then, use the corresponding model to make predictions on the preprocessed data.

**6. Return the prediction result**: After making predictions, the predicted class (Cover_Type) is returned to the user as the response from the API. I ensured that the response is in the expected format (JSON).

**7.Run the FastAPI app**: I ran the FastAPI app using the uvicorn.run method, specifying the host and port number on which the app should listen for incoming requests.

**8.Test the API**: After running FastAPI app, I tested the API by sending POST requests with user input data to the prediction endpoints and Verify that the API returns the correct predictions and handles errors gracefully.

