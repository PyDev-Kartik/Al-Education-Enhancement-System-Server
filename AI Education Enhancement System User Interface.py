#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[2]:


import os
import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
import ipywidgets as widgets
from IPython.display import display

# Load the dataset
os.chdir("C:\\Users\\kusha\\OneDrive\\Desktop\\Kushang's Files\\Maverick")
os.getcwd()
data = pd.read_csv('student_data.csv')

# Preprocess the dataset
data['GradeLevel'] = data['GradeLevel'].astype('category').cat.codes
data['LearningStyle'] = data['LearningStyle'].astype('category').cat.codes

# Define features and target
X = data[['GradeLevel', 'LearningStyle', 'EngagementScore', 'HoursStudied', 'Progress']]
y = data['Subject'].apply(lambda x: 1 if x == 'Science' else 0)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest
rf = RandomForestClassifier()
rf_params = {'n_estimators': [10, 50, 100], 'max_depth': [None, 10, 20]}
rf_grid = GridSearchCV(rf, rf_params, cv=3)
rf_grid.fit(X_train, y_train)

# Train SVM
svm = SVC()
svm_params = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
svm_grid = GridSearchCV(svm, svm_params, cv=3)
svm_grid.fit(X_train, y_train)

# Best models
best_rf = rf_grid.best_estimator_
best_svm = svm_grid.best_estimator_

# Evaluate models
rf_accuracy = accuracy_score(y_test, best_rf.predict(X_test))
svm_accuracy = accuracy_score(y_test, best_svm.predict(X_test))

print(f"RandomForest Accuracy: {rf_accuracy:.2f}")
print(f"SVM Accuracy: {svm_accuracy:.2f}")

# Store previous searches and their counts
previous_searches = []
search_counts = {}

# Define function to get recommendation
def get_recommendation(student_id=None, student_name=None):
    if student_name:
        student_data = data[data['Name'] == student_name]
    elif student_id:
        student_data = data[data['StudentID'] == student_id]
    else:
        return "Please provide either Student ID or Name."

    if student_data.empty:
        return "No data found for this student."

    student_features = student_data[['GradeLevel', 'LearningStyle', 'EngagementScore', 'HoursStudied', 'Progress']]
    rf_pred = best_rf.predict(student_features)
    svm_pred = best_svm.predict(student_features)

    recommendation = "The student is performing well. Keep up the good work!"
    if rf_pred[0] == 1 and svm_pred[0] == 1:
        recommendation = "The student should focus more on Science and refer to tutorials."

    # Track searches
    if student_id:
        identifier = student_id
    else:
        identifier = student_name
    
    previous_searches.append((identifier, recommendation))
    if identifier in search_counts:
        search_counts[identifier] += 1
    else:
        search_counts[identifier] = 1

    return recommendation

# Define function to get recommendations for all students
def get_all_recommendations():
    recommendations = []
    for index, row in data.iterrows():
        recommendation = get_recommendation(student_id=row['StudentID'])
        recommendations.append((row['StudentID'], recommendation))
    return recommendations

# Define function to get previous searched recommendations
def get_previous_recommendations():
    return previous_searches[-5:]  # Get last 5 searches

# Define function to get most searched recommendations
def get_most_searched_recommendations():
    most_searched_ids = sorted(search_counts, key=search_counts.get, reverse=True)[:5]
    return [(identifier, get_recommendation(student_id=int(identifier) if identifier.isdigit() else None, student_name=None if identifier.isdigit() else identifier)) for identifier in most_searched_ids]

# Define function to get random recommendations
def get_random_recommendations():
    random_ids = random.sample(list(data['StudentID']), min(5, len(data['StudentID'])))
    return [(student_id, get_recommendation(student_id=student_id)) for student_id in random_ids]

# Create UI
student_id_input = widgets.IntText(description="Student ID:")
student_name_input = widgets.Text(description="Student Name:")
recommend_button = widgets.Button(description="Get Recommendation")
all_recommend_button = widgets.Button(description="Get All Recommendations")
previous_recommend_button = widgets.Button(description="Previous Recommendations")
most_searched_recommend_button = widgets.Button(description="Most Searched Recommendations")
random_recommend_button = widgets.Button(description="Random Recommendations")
output = widgets.Output()

def on_recommend_button_clicked(b):
    with output:
        output.clear_output()
        student_id = student_id_input.value
        student_name = student_name_input.value.strip()
        recommendation = get_recommendation(student_id=student_id if student_id else None, student_name=student_name if student_name else None)
        print(f"Recommendation: {recommendation}")

def on_all_recommend_button_clicked(b):
    with output:
        output.clear_output()
        recommendations = get_all_recommendations()
        for student_id, recommendation in recommendations:
            print(f"Recommendation for Student ID {student_id}: {recommendation}")

def on_previous_recommend_button_clicked(b):
    with output:
        output.clear_output()
        recommendations = get_previous_recommendations()
        for identifier, recommendation in recommendations:
            print(f"Previous Recommendation for {identifier}: {recommendation}")

def on_most_searched_recommend_button_clicked(b):
    with output:
        output.clear_output()
        recommendations = get_most_searched_recommendations()
        for identifier, recommendation in recommendations:
            print(f"Most Searched Recommendation for {identifier}: {recommendation}")

def on_random_recommend_button_clicked(b):
    with output:
        output.clear_output()
        recommendations = get_random_recommendations()
        for student_id, recommendation in recommendations:
            print(f"Random Recommendation for Student ID {student_id}: {recommendation}")

recommend_button.on_click(on_recommend_button_clicked)
all_recommend_button.on_click(on_all_recommend_button_clicked)
previous_recommend_button.on_click(on_previous_recommend_button_clicked)
most_searched_recommend_button.on_click(on_most_searched_recommend_button_clicked)
random_recommend_button.on_click(on_random_recommend_button_clicked)

# Display UI
display(student_id_input, student_name_input, recommend_button, all_recommend_button, previous_recommend_button, most_searched_recommend_button, random_recommend_button, output)

