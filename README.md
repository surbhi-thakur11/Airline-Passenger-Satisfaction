# Airline-Passenger-Satisfaction

## Overview of the problem:
The main aim of this problem statement is to predict whether the customer is satisfied with his/her airplane experience or not. This problem comes under classification and this classification is done on the basis of passenger information and ratings given by him/her.   

### About data
The dataset for this project is attained from Kaggle which contains the data sourced from a survey conducted by airlines on the satisfaction level of passengers/customers based on various factors. The dataset consists of 25 columns such as Age, Gender, Travel class, Arrival and Departure delays and also features that influences customer satisfaction level such as On-board service, Cleanliness, Seat comfort, Baggage handling etc.
The dataset consists of a column or feature named ‘satisfaction’ which describes the overall satisfaction level of the customer. It has two values, ‘neutral or dissatisfied’ and ‘satisfied’. This satisfaction feature is considered as the label feature since it conveys the overall experience of the customer based on the ratings given for other features.

### Steps Followed
![Screenshot 2021-09-04 115903](https://user-images.githubusercontent.com/77155721/132085296-a3a857db-677a-4825-aa44-423526255661.png)

After Data pre-processing, EDA and feature selection I performed feature engineering in which I scaled my data
using Standard Scaler. It will scale my data between 0 and 1 or at least all my features will be in same range .If
we do not scale our data so as we can see the some data range lies in 100’s some in 1000 some in 50’s, while
prediction our model will automatically give importance to feature with high range so for that we perform
standard scaling. And for Categorical data I have performed Label Encoding to scale my values.
Final Dataset on which we will be applying Machine Learning Algorithms.
### Final Dataset
![Screenshot 2021-09-04 120830](https://user-images.githubusercontent.com/77155721/132085411-3457eab8-e76f-4544-ac35-a1e874e59828.png)

After applying various algorithm found out Random Forest Algorithm gave the best results. 
And then I dumped the model into pickle file for deployment and to prepare the .py file.

### final project after deployment 

![Screenshot 2021-09-04 121148](https://user-images.githubusercontent.com/77155721/132085477-e593a466-4edd-4ef7-bc55-f83dc839fd88.png)

![Screenshot 2021-09-04 121210](https://user-images.githubusercontent.com/77155721/132085479-b042f9e7-3207-4aca-b0e4-c55f7ad1661f.png)

Here is the link for my model and linkedIn video:

Deployed model: https://airline-passenger-satisfaction.herokuapp.com/

Dataset link: https://www.kaggle.com/teejmahal20/airline-passenger-satisfaction

LinkedIn video: https://www.linkedin.com/posts/surbhi-thakur11_datascience-python-internship-
activity-6804736974352633856-HS93






