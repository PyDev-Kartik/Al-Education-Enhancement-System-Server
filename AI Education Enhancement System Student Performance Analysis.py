#!/usr/bin/env python
# coding: utf-8

# In[ ]:


Student Performance Analysis


# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.stats import pearsonr


# In[2]:


# Load the dataset
os.chdir("C:\\Users\\kusha\\OneDrive\\Desktop\\Kushang's Files\\Maverick")
os.getcwd()
data = pd.read_csv('student_data.csv')


# In[3]:


# Display the first few rows of the dataset
print(data.head())


# In[4]:


# Function to analyze a particular student
def analyze_student(student_id, student_data):
    student_data_filtered = student_data[student_data['StudentID'] == student_id]
    if student_data_filtered.empty:
        print(f"No data found for Student ID: {student_id}")
        return
    print(f"Analysis for Student ID: {student_id}")
    print(student_data_filtered)

    # Plotting scores in different subjects
    plt.figure(figsize=(6, 4))
    sns.barplot(x='Subject', y='Score', data=student_data_filtered, palette='viridis')
    plt.title(f'Scores for Student ID: {student_id}')
    plt.ylabel('Score')
    plt.xlabel('Subject')
    plt.ylim(0, 100)
    plt.show()

    # Plotting learning style
    plt.figure(figsize=(6, 4))
    student_data_filtered['LearningStyle'].value_counts().plot(kind='bar')
    plt.title(f'Learning Style for Student ID: {student_id}')
    plt.xlabel('Learning Style')
    plt.ylabel('Count')
    plt.show()


# In[5]:


# Function to analyze performance by subject
def analyze_subject(subject, student_data):
    subject_data = student_data[student_data['Subject'] == subject]
    
    # Average score by grade level
    avg_scores = subject_data.groupby('GradeLevel')['Score'].mean().reset_index()

    # Plotting average scores by grade level
    plt.figure(figsize=(8, 5))
    sns.barplot(x='GradeLevel', y='Score', data=avg_scores, palette='coolwarm')
    plt.title(f'Average Scores in {subject} by Grade Level')
    plt.ylabel('Average Score')
    plt.xlabel('Grade Level')
    plt.ylim(0, 100)
    plt.show()

    # Distribution of scores
    plt.figure(figsize=(8, 5))
    sns.histplot(subject_data['Score'], bins=10, kde=True, color='blue')
    plt.title(f'Distribution of Scores in {subject}')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.xlim(0, 100)
    plt.show()


# In[6]:


# Function to analyze overall performance
def overall_performance(student_data):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Subject', y='Score', data=student_data, palette='Set2')
    plt.title('Overall Performance by Subject')
    plt.ylabel('Score')
    plt.xlabel('Subject')
    plt.ylim(0, 100)
    plt.show()

    # Calculate overall average score
    overall_avg_score = np.mean(student_data['Score'])
    print(f"Overall average score: {overall_avg_score:.2f}")


# In[7]:


# Function to analyze engagement vs. score
def engagement_analysis(student_data):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='EngagementScore', y='Score', data=student_data, hue='Subject', style='GradeLevel', palette='deep')
    plt.title('Engagement vs. Score')
    plt.xlabel('Engagement Score')
    plt.ylabel('Score')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.legend()
    plt.show()

    # Calculate correlation between engagement and score
    correlation = np.corrcoef(student_data['EngagementScore'], student_data['Score'])[0, 1]
    print(f"Correlation between engagement and score: {correlation:.2f}")


# In[8]:


# Function to analyze progress
def progress_analysis(student_data):
    plt.figure(figsize=(8, 5))
    sns.histplot(student_data['Progress'], bins=10, kde=True, color='green')
    plt.title('Distribution of Student Progress')
    plt.xlabel('Progress')
    plt.ylabel('Frequency')
    plt.xlim(0, 1)
    plt.show()

    # Calculate average progress
    avg_progress = np.mean(student_data['Progress'])
    print(f"Average student progress: {avg_progress:.2f}")


# In[9]:


# Function to create a heatmap of feature correlations
def create_heatmap(student_data):
    # Compute pairwise correlations
    corr_matrix = student_data[['Score', 'EngagementScore', 'HoursStudied', 'Progress']].corr()

    # Create a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, cmap='YlOrRd', annot=True, fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.xlabel('Features')
    plt.ylabel('Features')
    plt.show()


# In[10]:


# Function to perform K-Means clustering
def kmeans_clustering(student_data):
    # Prepare data for clustering
    X = student_data[['Score', 'EngagementScore', 'HoursStudied']].values
    kmeans = KMeans(n_clusters=4, random_state=0)
    kmeans.fit(X)
    labels = kmeans.labels_

    # Compute Pearson correlation between first feature and cluster labels
    pearson_r, p_value = pearsonr(X[:, 0], labels)
    print(f'Pearson correlation: {pearson_r:.2f}, p-value: {p_value:.4f}')


# In[11]:


# Example usage
analyze_student(1, data)  # Analyze a specific student by ID
analyze_subject('Math', data)  # Analyze performance in Math
overall_performance(data)  # Analyze overall performance
engagement_analysis(data)  # Analyze engagement vs. score
progress_analysis(data)  # Analyze student progress
create_heatmap(data)  # Create a heatmap of correlations
kmeans_clustering(data)  # Perform K-Means clustering


# In[12]:


# Analyze subject performances
subjects_to_analyze = ['Math', 'Science']

for subject in subjects_to_analyze:
    analyze_subject(subject, data)

# Analyze overall performance
overall_performance(data)

# Analyze engagement vs. score
engagement_analysis(data)

# Analyze student progress
progress_analysis(data)

# Create a heatmap of correlations
create_heatmap(data)

# Perform K-Means clustering
kmeans_clustering(data)


# In[13]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
os.chdir("C:\\Users\\kusha\\OneDrive\\Desktop\\Kushang's Files\\Maverick")
data = pd.read_csv('student_data.csv')

# Add additional features
data['PastPerformanceTrend'] = np.random.rand(len(data))  # Placeholder for actual past performance trend
data['ExtraCurricularParticipation'] = np.random.randint(0, 2, size=len(data))  # Binary feature for extra-curricular participation

# Define the target variable and features
data['Target'] = data.apply(lambda row: 1 if row['Subject'] == 'Science' and row['Score'] < np.mean(data[data['Subject'] == 'Science']['Score']) else 0, axis=1)
features = ['GradeLevel', 'Score', 'EngagementScore', 'HoursStudied', 'Progress', 'PastPerformanceTrend', 'ExtraCurricularParticipation']

X = data[features]
y = data['Target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to perform model training with GridSearchCV
def train_model(X_train, y_train):
    # Define the models and hyperparameters
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42)
    }

    params = {
        'RandomForest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30]
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }
    }

    best_models = {}
    for model_name in models:
        grid = GridSearchCV(models[model_name], params[model_name], cv=5, scoring='accuracy')
        grid.fit(X_train, y_train)
        best_models[model_name] = grid.best_estimator_
        print(f"Best parameters for {model_name}: {grid.best_params_}")
    
    return best_models

# Train models and get the best ones
best_models = train_model(X_train, y_train)

# Evaluate the models on the test set
def evaluate_models(models, X_test, y_test):
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        print(f"Evaluating {model_name}...")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))

evaluate_models(best_models, X_test, y_test)

# Function to analyze a particular student and make recommendations
def analyze_student_recommendation(student_id, student_data, models):
    student_data_filtered = student_data[student_data['StudentID'] == student_id]
    if student_data_filtered.empty:
        print(f"No data found for Student ID: {student_id}")
        return

    student_features = student_data_filtered[features]
    recommendations = {}
    for model_name, model in models.items():
        recommendation = model.predict(student_features)
        recommendations[model_name] = recommendation[0]

    print(f"Recommendations for Student ID: {student_id}")
    for model_name, recommendation in recommendations.items():
        if recommendation == 1:
            print(f"{model_name}: Focus more on Science. Refer to tutorials on the platform.")
            print("If further help is needed, contact the tutor or use the doubt section.")
        else:
            print(f"{model_name}: No specific recommendation. Keep up the good work!")

# Command-line interface for interaction
def cli_interface(student_data, models):
    while True:
        student_id = input("Enter Student ID (or 'exit' to quit): ")
        if student_id.lower() == 'exit':
            break
        try:
            student_id = int(student_id)
            analyze_student_recommendation(student_id, student_data, models)
        except ValueError:
            print("Invalid Student ID. Please enter a numeric value.")

# Example usage for more students
cli_interface(data, best_models)


# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
os.chdir("C:\\Users\\kusha\\OneDrive\\Desktop\\Kushang's Files\\Maverick")
data = pd.read_csv('student_data.csv')

# Add additional features
data['PastPerformanceTrend'] = np.random.rand(len(data))  # Placeholder for actual past performance trend
data['ExtraCurricularParticipation'] = np.random.randint(0, 2, size=len(data))  # Binary feature for extra-curricular participation

# Define the target variable and features
data['Target'] = data.apply(lambda row: 1 if row['Subject'] == 'Science' and row['Score'] < np.mean(data[data['Subject'] == 'Science']['Score']) else 0, axis=1)
features = ['GradeLevel', 'Score', 'EngagementScore', 'HoursStudied', 'Progress', 'PastPerformanceTrend', 'ExtraCurricularParticipation']

X = data[features]
y = data['Target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to perform model training with GridSearchCV
def train_model(X_train, y_train):
    # Define the models and hyperparameters
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42)
    }

    params = {
        'RandomForest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30]
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }
    }

    best_models = {}
    for model_name in models:
        grid = GridSearchCV(models[model_name], params[model_name], cv=5, scoring='accuracy')
        grid.fit(X_train, y_train)
        best_models[model_name] = grid.best_estimator_
        print(f"Best parameters for {model_name}: {grid.best_params_}")
    
    return best_models

# Train models and get the best ones
best_models = train_model(X_train, y_train)

# Evaluate the models on the test set
def evaluate_models(models, X_test, y_test):
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        print(f"Evaluating {model_name}...")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))

evaluate_models(best_models, X_test, y_test)

# Function to analyze a particular student and make recommendations
def analyze_student_recommendation(student_identifier, student_data, models):
    if isinstance(student_identifier, int):
        student_data_filtered = student_data[student_data['StudentID'] == student_identifier]
    else:
        student_data_filtered = student_data[student_data['Name'].str.contains(student_identifier, case=False, na=False)]
    
    if student_data_filtered.empty:
        print(f"No data found for Student Identifier: {student_identifier}")
        return

    student_features = student_data_filtered[features]
    recommendations = {}
    for model_name, model in models.items():
        recommendation = model.predict(student_features)
        recommendations[model_name] = recommendation[0]

    print(f"Recommendations for Student Identifier: {student_identifier}")
    for model_name, recommendation in recommendations.items():
        if recommendation == 1:
            print(f"{model_name}: Focus more on Science. Refer to tutorials on the platform.")
            print("If further help is needed, contact the tutor or use the doubt section.")
        else:
            print(f"{model_name}: No specific recommendation. Keep up the good work!")

# Print all student data
print("All Student Data:")
print(data)

# Command-line interface for interaction
def cli_interface(student_data, models):
    while True:
        student_input = input("Enter Student ID or Name (or 'exit' to quit): ")
        if student_input.lower() == 'exit':
            break
        try:
            student_identifier = int(student_input)
        except ValueError:
            student_identifier = student_input
        analyze_student_recommendation(student_identifier, student_data, models)

# Example usage for more students
cli_interface(data, best_models) 

