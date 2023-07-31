# Classification Task on 'Forest Cover Type Prediction' and then Serving ML models using FastAPI 

- final.ipynb contains code for EDA + Feature Engineering + Classification
- final_shortened.ipynb contains code (with EDA)
- app.py (FastAPI)

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

