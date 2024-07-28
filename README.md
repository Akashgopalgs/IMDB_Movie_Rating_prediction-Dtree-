# Movie Rating Prediction App

This repository contains a web application that predicts movie ratings based on various features using a ***Decision Tree Regressor***. The model is trained on a dataset of Indian movies from IMDb.

## Features

- **Year:** The release year of the movie.
- **Votes:** The number of votes the movie has received.
- **Duration:** The duration of the movie in minutes.
- **Genre Mean Rating:** The average rating of the genre.
- **Director Rating:** The average rating of movies directed by the director.
- **Main Actor Rating:** The average rating of movies featuring the main actor.
- **Second Actor Rating:** The average rating of movies featuring the second actor.
- **Third Actor Rating:** The average rating of movies featuring the third actor.

## Model

The model used in this application is a **Decision Tree Regressor**. It was trained using features such as the year of release, votes, duration, and encoded mean ratings for genre, director, and actors.

Evaluation
The performance of the models was evaluated using the following metrics:
- Mean Squared Error (MSE): 0.088
- R-squared (R²): 0.952
