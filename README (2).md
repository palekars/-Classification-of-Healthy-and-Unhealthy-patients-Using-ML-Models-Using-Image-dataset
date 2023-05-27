
#  Classification of  Healthy and Unhealthy patients Using ML Models.


This project uses classsification technique of Machine Learning for classification of Healthy and Unhealthy Patients.This project uses various ML models to predict the healthy or unhealthy patient from the patient's blood sample image and demographic data.


## Data Set
Data used in this project is images of blood samples of total 300 patients and the demographic info of patients.The 'RGB values' obtained from these images and demographic information of patient such as Age,Hb%,rutu,prakuti,data related with Vatt, Pitta,Kapha are considered for study.

## Methodology
Preprocessing of data:

For preprocessing the data project uses the techniques like handling missing values , and encoding categorical variables.
This also includes the checking of assumptions of data.For taking RGB values from images the project uses image processing techniques.

   Data splitting:

The processed data then splitted into the trainning and testing dataset into 70 % and 30 % proportions respectively.Trainning dataset is used to train the   model and testing dataset used to test the model.
  
   Model selection
         
For classification problem the project uses four ML models i.e.KNN,NB,SVM and CART.
   
Model Trainning:

By feeding the trainning dataset to the models each of the model was trained.
   
   Model Evaluation:
   
The project uses metrics like Accuracy and R-square score as evaluation parameters for models.
## Models Used
KNN:

The KNN algorithm identifies the K nearest neighbors based on a distance metric (e.g., Euclidean distance). The distance can be calculated using all features or a subset of features, depending on the problem.

Naive Baye's :
Naive Bayes is a machine learning algorithm based on the Bayes' theorem and is commonly used for classification tasks.NB model calculate the posterior probabilities of each class label using the Bayes' theorem. The posterior probability is the probability of a class label given the observed feature values. The class label with the highest posterior probability is assigned as the predicted label for that instance.
    
SVM :

SVMs are based on the concept of finding an optimal hyperplane that separates the data points into different classes or predicts the target values for regression. The model calculates the decision function or the distance from the data points to the hyperplane and assigns the data points to the appropriate class based on the decision boundary.
     
CART:

CART (Classification and Regression Trees) is a machine learning algorithm that can be used for both classification and regression tasks. It constructs a binary tree-based model that recursively partitions the data based on input features to make predictions.Each data point traverses the decision tree by following the appropriate branches based on the feature values until it reaches a leaf node. The predicted class is determined based on the majority class of the samples in the leaf node
## Kalman Filter
Project also studies about the Kalman filter and make use of it.The Kalman filter is a mathematical algorithm that helps in predicting the state of a system based on noisy measurements. In this case, the system is the RGB values of a blood sample, and the noisy measurements are the errors that can occur during the measurement process. The Kalman filter uses a set of equations to estimate the state of the system based on the measurements.So when we applied this modified Kalman Filter the accuracy of model came out to be 100% and we also got the hold on the ranges for RGB values for healthy and unhealthy patients for the given data set.

## Platforms  and Languages Used
Language:

 The project Uses Python language for implementing the models.
     
Platforms Used

Matlab - For image processing purpose.
 
Jupyter Notebook 
## Results
When only RGB values are consider  the accuracy of of models is approximates 55-60%. Also if we consider patient's responses to a questionnaire accuracy increases upto 86-97%. The CART model performs best with accuracy 97.14%.
## Conclusion
The results demonstrate the potential for using machine learning in Ayurvedic practice to provide more accurate diagnoses and treatment recommendations.







