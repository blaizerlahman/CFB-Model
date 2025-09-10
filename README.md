# CFB Predictive Model

### DISCLAIMER
All results of backtesting the model and success analysis are based on faulty training data which produced skewed results, and is not to be taken seriously. I plan on updating this info once I have updated the model. Only 2024 season results are legit, as they used mid-season SP+ ratings, not post-season.
## Overview
This project is separated into 6 Jupyter notebooks that contain all the necessary procedures to grab and reformat the data needed to build a successful college football predictive model, as well as the model building and analysis process that results in a model that can correctly predict the score differential of every college football game with respect to the Las Vegas spread (prediction) at a 58% success rate. Additionally, the notebooks also contain the necessary functions to run the model during the season and display the historical success rates for each prediction based on their difference from the Vegas spread.
## Notebooks
These notebooks are meant to be followed in order if trying to build the model from scratch. If you are only interested in building the most refined and successful model, follow the data importation and wrangling steps from notebooks 1 and 4 to get the data necessary to run the model and follow notebooks 5 and 6 to be able to run and evaluate the model for yourself during the season.
1.  Data Importation and Wrangling
2.  Basic Model Building and Analysis
3.  Upgrading Modeling Techniques and Analysis
4.  Importing Advanced Features and Model Analysis
5.  Success Analysis
6.  Model Usage Functionality for Mid-Season
## Data Importation and Wrangling
Data importation from collegefootballdata.com was done with Selenium's ChromeDriver to automate the data file retrieval process. Wrangling the data into a more readable and model-friendly format was done with Pandas and NumPy.
## Basic Model Building and Analysis
The first basic model was built using Scikit-learn's pipelines and functions as a linear regression model using Sequential Feature Selection and taking into account a rolling sum of recent team statistics. Data visualization done with Matplotlib and Seaborn, model analysis done with Scikit-learn.
## Upgrading Modeling Techniques and Analysis
Model improved using implementation of Lasso regression and GridSearchCV to determine the optimal alpha level for feature use. A form of imputation was also done by using simple linear regression models to determine the relationship between score differentials and columns with missing data, and then using the distribution of predicted missing values to generate likely values for missing data. Modeling and analysis done with Scikit-learn.
## Importing Advanced Features and Model Analysis
Team talent composite ratings and SP+ ratings imported from collegefootballdata.com using Selenium. Team talent and SP+ ratings were orignally compiled by 247 Sports and Bill Connelly, respectively. Also implemented a Bagging regressor to obtain more standardized results. Model building and analysis done with Scikit-learn.
## Success Analysis
Prediction success analysis on final model done across multiple years (2018-2023, excluding 2020) by predicting a score differential for games and making the model "place a bet" on each game by choosing which side of the Las Vegas spread will "win". Las Vegas spreads essentially are a prediction by sportsbooks on how much a team will win or lose a game by, so if the model predicts a team will win by more or lose by less than that amount it says the team will "cover", or win by more or lose by less than the amount of points set by Vegas. If the model predicts a team will win by less or lose by more points than Vegas says that they will, then the model says that the team will "not cover", or win by less or lose by more points than Vegas says they will. The model "places a bet" on these outcomes, and these bets are compared against the actual outcomes of games to determine the success rate i.e. how often our model prediction matched the outcome of the game. Across all 5 years, the average success rate for a given year was 57%, with the yearly success rates ranging from 54.7% to 58.8%. Each spread set by Vegas is meant to have inherent 50/50 odds, so having an average success rate of 57% is generally considered to be very good, as the success rates of "professional" gamblers sit at around 55%. The historical success rate of bets were also analyzed by looking at the "spread differential", or the difference between the model's prediction and the Vegas spread, and binning those spread differentials and looking at the historical success rates of those bins. These will be used in the following notebook to highlight specific games that have higher historical success rates. A larger dataset to pull bins from was compiled by reconstructing the model and predicting the 2018-2023 seasons 10 times over. The visualization of these bins and their success rates can be seen below. Success analysis was done using pandas and NumPy.

<img width="758" alt="Screenshot 2024-07-31 at 12 27 20 PM" src="https://github.com/user-attachments/assets/7a84a575-0d5f-4d5c-bbc4-a239c739f369">
The red, yellow, and green dashed lines represent 50, 60, and 70 percent success rates, respectively.

## Model Usage Functionality for Mid-Season
This notebook essentially acts as an all in one data gathering, model building, and prediction making hub that contains functions necessary to run the model and update the data used to run the model during the season. It also contains functions that will display all of the predictions for the upcoming week and their respective historical success rates. These functions use all of the libraries listed so far to do their respective tasks. 

## License
License: GNU Affero General Public License v3.0

## Source
Latest version is on github: https://github.com/blaiiize/CFB-Model

Contributions and forks are welcome

## Developers
Developed by Blaize Lahman

## References
Project inspired by and initially based off of Brian Lucena's baseball predictive model: https://github.com/numeristical/resources/tree/master/BaseballPred

Team and game data imported from collegefootballdata.com

Team talent ratings compiled by 247 Sports

Team SP+ ratings compiled by Bill Connelly
