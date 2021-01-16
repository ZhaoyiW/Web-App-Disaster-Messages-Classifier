# Disaster-Response [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
A web app based on NLP and supervised machine learning.   
Type in your disaster report message and get the categories immediately.

## Content

- [Project Overview](#project-overview)
  * [Background](#background)
  * [Project Goal](#project-goal)
- [Installations](#installations)
  * [Data Source](#data-source)
  * [Modules](#modules)
  * [Data Processing and Database Building](#data-processing-and-database-building)
  * [NLP and Machine Learning Pipeline](#nlp-and-machine-learning-pipeline)
  * [Run the App Locally](#run-the-app-locally)
- [Web App Overview](#web-app-overview)
- [File Description](#file-description)

## Project Overview
### Background
Following a disaster, we will get millions of communications, either directly or through social media platforms. Different organizations will need to take care of different parts of the problem. These organizations have to filter and pull out messages that are most important and relevant to respond immediately.
### Project Goal
In this project, I will build an NLP and supervised machine learning model to classify the disaster-related messages into different categories and help different organizations get the messages they need to respond to.
## Installations
### Data Source
A data set containing real messages that were sent during disaster events provided by [Figure Eight](https://appen.com/).
### Modules
:star:   **pip install these modules**
- [sys](https://docs.python.org/3/library/sys.html): system-specific parameters and functions
- [pandas](https://pandas.pydata.org/): data processing
- [numpy](https://numpy.org/): linear algebra
- [re](https://docs.python.org/3/library/re.html): regular expressions   
- [json](https://docs.python.org/3/library/json.html): JSON encoder and decoder   
- [sqlalchemy](https://www.sqlalchemy.org/): SQL toolkit   
- [nltk](https://www.nltk.org/): natural language processing   
- [scikit-learn](https://scikit-learn.org/stable/index.html): machine learning
- [pickle](https://docs.python.org/3/library/pickle.html): save the machine learning model locally
- [joblib](https://joblib.readthedocs.io/en/latest/): load the machine learning model   
- [flask](https://flask.palletsprojects.com/en/1.1.x/): web framework
- [plotly](https://plotly.com/): front-end visualizations

### Data Processing and Database Building
Run the following commands in the project's root directory to set up the database:    
```  
# Run ETL pipeline that cleans data and stores in database
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db   
```   
After running this, you will have a database file called "DisasterResponse.db" in your data folder.
### NLP and Machine Learning Pipeline
Run the following commands in the project's root directory to set up the machine learning model.
```
# Run ML pipeline that trains classifier and saves
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```   
After running this, you will have a pickle file called "classifier.pkl" in your model folder.

### Run the App Locally
Run the following command in the app's directory to run your web app.   
```
python run.py
```   
Then go to http://0.0.0.0:3001/ :arrow_lower_left:

## Web App Overview

## File Description
- app
  - template
    - master.html  # main page of web app
    - go.html  # classification result page of web app
  - run.py  # Flask file that runs app

- data
  - disaster_categories.csv  # data to process 
  - disaster_messages.csv  # data to process
  - process_data.py
  - InsertDatabaseName.db   # database to save clean data to

- models
  - train_classifier.py
  - classifier.pkl  # saved model 
  
## License
This project is licensed under [MIT License](https://github.com/git/git-scm.com/blob/master/MIT-LICENSE.txt).

## Author
[Zhaoyi Wang](https://github.com/ZhaoyiW)
