# Analysis of correlations between weather events / infrastructure impact and aid types

## Project Overview

This project is designed to build an end-to-end pipeline for disaster response messages. The aim is to categorize incoming messages into appropriate disaster response categories to help organizations quickly route requests to the appropriate teams. The project consists of three main components: an ETL pipeline, a machine learning pipeline, and a Flask web app for data visualization and user interaction.

## Project Components

### 1. **ETL Pipeline**
The ETL (Extract, Transform, Load) pipeline is responsible for processing and cleaning the data from two datasets: messages and categories. The cleaned data is stored in a SQLite database for further use in the machine learning pipeline.

- **Script**: `process_data.py`
- **Steps**:
  1. Load the messages and categories datasets from CSV files.
  2. Merge the two datasets.
  3. Clean the data by:
     - Splitting categories into individual columns.
     - Converting category values to binary.
     - Removing duplicates and handling missing data.
  4. Save the cleaned data to a SQLite database.

### 2. **ML Pipeline**
The machine learning pipeline builds a text classification model to categorize disaster response messages. It leverages natural language processing (NLP) and machine learning algorithms to train and evaluate the model.

- **Script**: `train_classifier.py`
- **Steps**:
  1. Load data from the SQLite database.
  2. Split the dataset into training and testing sets.
  3. Build a text processing and classification pipeline using `CountVectorizer`, `TfidfTransformer`, and `RandomForestClassifier`.
  4. Tune hyperparameters using `GridSearchCV`.
  5. Train and evaluate the model on the test set.
  6. Export the final model as a pickle file (`classifier.pkl`).

### 3. **Flask Web App**
The web application allows users to input new disaster messages and receive classification results. Additionally, the web app displays data visualizations using Plotly to provide insights into the disaster categories and message types.

- **Files**:
  - `app.py`: Flask web server
  - `templates/master.html`: HTML template for the app layout
  - `static/style.css`: Custom CSS for the app styling
- **Steps**:
  1. Modify file paths to connect the app with the SQLite database and the trained model.
  2. Display interactive data visualizations using Plotly. Examples include:
     - A bar chart showing the distribution of aid request types.
     - A heatmap showing the correlation between different aid categories.

## Data Visualizations

The web app includes several interactive data visualizations created using Plotly:

1. **Barchart: Distribution of Aid Requests**
   - This bar chart displays the frequency of different types of aid requests in the dataset. It helps identify the most commonly requested aid types (e.g., medical help, shelter, food, etc.).

2. **Barchart: Distribution of Infrastructure Impact**
   - This chart shows the distribution of impacts on various types of infrastructure (e.g., transport, electricity, hospitals, etc.). It highlights how different disasters impact critical infrastructure and which types are most affected.

3. **Barchart: Distribution of Weather Events**
   - This visualization displays the frequency of various weather events (e.g., floods, storms, fires, earthquakes). It helps in understanding which weather events are most prevalent in the dataset.

4. **Piechart: Frequency of Aid Requests by Message Type**
   - This pie chart shows the distribution of messages by genre (e.g., direct, news, social). It provides insights into how people communicate disaster-related information across different media types.

5. **Heatmap: Correlation Matrix of Aid Requests**
   - This heatmap visualizes the correlation between different types of aid requests. It helps identify potential relationships between various aid categories (e.g., if requests for shelter often coincide with requests for food).

6. **Heatmap: Correlation Between Weather Events and Aid Types**
   - This heatmap shows the correlation between specific weather events and the types of aid requested in response to those events. It provides insight into how different disasters impact aid needs.

7. **Heatmap: Correlation Between Infrastructure Impact and Aid Types**
   - This heatmap explores the relationship between infrastructure impacts (e.g., transport, electricity) and the types of aid requested. It helps identify patterns in how infrastructure damage drives specific aid requests.

## Project Setup

### Prerequisites
Make sure you have the following libraries installed:
- Python 3.x
- Flask
- Pandas
- NumPy
- Scikit-learn
- SQLAlchemy
- NLTK
- Plotly
- Joblib

You can install the required libraries by running:
```bash
pip install -r requirements.txt