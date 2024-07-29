# Student Performance Analysis and Recommendation System
Empowering personalized learning with intelligent recommendations to transform the educational landscape and unlock each student's full potential.
## Overview

This project includes a set of tools for analyzing student performance data and providing recommendations based on various machine learning models. It consists of:
1. A web-based user interface (`index.html`) for interacting with the recommendation system.
2. A Python server script (`server.py`) that serves the web interface and handles backend logic.
3. Python scripts for detailed analysis of student performance data.

## Project Structure

- `index.html`: Frontend user interface.
- `server.py`: Backend server script.
- `student_performance_analysis.py`: Script for analyzing student performance with various metrics.
- `model_training_and_recommendation.py`: Script for training models and generating recommendations.

## `index.html`

The `index.html` file provides a user interface for interacting with the recommendation system. It includes:
- Input fields for entering Student ID or Name.
- Buttons for various actions: Get Recommendation, Get All Recommendations, Previous Recommendations, Most Searched Recommendations, Random Recommendations.
- Display area for showing the results of the interactions.

## `server.py`

The `server.py` script sets up a simple HTTP server to serve the `index.html` file and process user requests. It uses the `http.server` module and listens on port 8000. The server:
- Serves static files (e.g., `index.html`) to the client.
- Handles requests for recommendations and interacts with the Python backend code.

## Setup and Running the Project

### 1. Prerequisites

- Python 3.x
- Required Python libraries: `pandas`, `numpy`, `scikit-learn`, `ipywidgets`, `matplotlib`, `seaborn`, `scipy`

You can install the required libraries using `pip`:

```sh
pip install pandas numpy scikit-learn ipywidgets matplotlib seaborn scipy

2. Setting Up the Server
1.	Navigate to the directory where server.py is located using the terminal or command prompt.
cd path/to/your/directory
2.	Run the server using Python:
python server.py
or if Python 3 is required:
python3 server.py

3.	Open a web browser and go to http://localhost:8000 to interact with the user interface.

3. Running the Analysis Scripts
You can run the analysis scripts directly using Python.

•	Student Performance Analysis:
python student_performance_analysis.py

4. Data
Ensure that the student_data.csv file is located in the working directory of the scripts. The dataset should include columns such as StudentID, Name, GradeLevel, LearningStyle, EngagementScore, HoursStudied, Progress, Subject, and Score.
Code Descriptions
index.html
This file creates a user interface for the recommendation system. It includes input fields for student identification and buttons to perform various actions.
server.py
This script starts a simple HTTP server to serve the index.html file and handle requests. It does not include complex backend logic but is essential for serving the frontend.
student_performance_analysis.py
This script performs various analyses on student performance data, including:
•	Plotting scores and learning styles.
•	Analyzing performance by subject.
•	Generating visualizations for engagement vs. score and student progress.
•	Creating heatmaps and performing K-Means clustering.
•	Loads the dataset and adds additional features.
•	Defines and trains machine learning models (RandomForest and SVM) using GridSearchCV.
•	Evaluates models and provides recommendations based on the trained models.
•	Includes a command-line interface for interaction.
Contribution
Feel free to contribute to the project by submitting pull requests or reporting issues.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Contact
For any questions or feedback, please contact kushangashukla1@gmail.com or kartikhajela1312@gmail.com

You can save this content as `README.md` in the root directory of your project. This file provides a clear overview of the project, how to set it up, and descriptions of the various scripts and their functionalities.
