# Mall Customer Segmentation
This project groups mall customers into different clusters based on their age, gender, income, and spending score.
The main goal is to help businesses understand customer behavior and plan better marketing strategies using machine learning.

## Project Overview
Customer segmentation is one of the most useful techniques in marketing analytics.
This project applies unsupervised machine learning (K-Means Clustering) to group customers with similar purchasing behavior.
It demonstrates how data-driven models can help businesses target different customer segments effectively.

## Tools and Technologies
Programming Language: Python
Libraries: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
Model Training: K-Means Clustering
Model Evaluation: Elbow Method, Silhouette Score
Deployment Frameworks: Streamlit (UI) and FastAPI (backend API)
Other Tools: Git, GitHub, VS Code

## Machine Learning Workflow

### Data Preprocessing
Loaded and explored the mall customer dataset
Checked for missing values and cleaned the data
Encoded gender (Male/Female) into numeric form
Applied log transformation on Age and Income to reduce skewness
Scaled the data before applying K-Means

### Model Training
Used the Elbow Method and Silhouette Score to find the optimal number of clusters
Trained K-Means to group customers into different segments
Visualized clusters using scatter plots
Saved the trained model as mall_customer_segmentation.joblib using joblib

### Deployment
Created a Streamlit app to visualize clusters and interact with the model
Built a FastAPI backend for making predictions through API requests

### Author
Hania Nasir
