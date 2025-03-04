
# FPL Insights

## Overview
This project aims to predict the performance of Fantasy Premier League (FPL) players using machine learning techniques. It utilizes an RNN (Recurrent Neural Network) to model the sequential nature of player form while incorporating a feed-forward neural network (FF-NN) to account for static player and team-related data.

## Data Pipeline
### 1. Scraping Data from the FPL API
- Data is fetched directly from the **FPL API** in JSON format.
- The API provides a comprehensive dataset, including:
  - Player statistics
  - Team performance data
  - Match history
  - Upcoming fixture difficulty ratings
- The JSON response is parsed, extracting relevant information for each player.

### 2. Creating a Player Profile
- A structured **Player Profile** is created for each player, containing:
  - **Static information**: Player ID, cost, team, position.
  - **Historical match data**: Performance stats from past matches (e.g., goals, assists, influence, threat, ICT index, xG, xA, xGC).
  - **Upcoming fixture data**: Opponent difficulty, home/away status.

### 3. Splicing Data into Player Form Vectors
- The last **five** matches of each player are extracted into a **player form vector**, structured as:
  ```
  [player_id, cost, team, (match_1_rating, opponent, home/away), (match_2_rating, opponent, home/away), ..., (match_5_rating, opponent, home/away)]
  ```
- The **target variable** (y) is the expected player score for the upcoming fixture.
- Data is stored in `player_data.csv` for model training.

## Machine Learning Model
### 1. Model Architecture
The ML model consists of:
- **RNN (Recurrent Neural Network)**: Processes the sequential player form data (last five matches).
- **FF-NN (Feed-Forward Neural Network)**: Processes static player/team metadata.
- The outputs from both networks are merged and passed through a final layer to predict the player’s performance score.

### 2. Training the Model
- The dataset is normalized to improve performance.
- The model is trained using **PyTorch**, with:
  - **X_features**: Sequential data (last 5 match tuples).
  - **X_metadata**: Static data (player ID, cost, team, next match details).
  - **y_target**: Expected performance score.
- Optimizer: Adam
- Loss function: Mean Squared Error (MSE)

### 3. Inference on Future Fixtures
- The model is used to predict the player scores for the upcoming FPL gameweek.
- The last 4 matches are used as historical input, and the next fixture data is appended.
- The model outputs expected scores for each player, aiding in team selection.

## Running the Project
### Requirements
Ensure you have the following installed:
- Python 3.8+
- PyTorch
- Pandas
- Requests
- NumPy
- CMake
- gcc / C++ compiler of choice

### Steps to Run
1. **Fetch Data**: Run the scraping script to download the latest FPL data.
2. **Process Player Data**: Convert raw JSON into structured player profiles and form vectors.
3. **Train the Model**: Run the training script to train the hybrid RNN + FF-NN model.
4. **Predict Player Scores**: Use the trained model to predict scores for upcoming fixtures.

## Compile and Run Data Collector and Refiner

Update - to Run the C++ Data Processor and the python ML script in 
one go use the 'run' script:

For Linux : `./run.sh`

For Windows : `run.bat`

### Future Improvements
- Expand dataset.
- Implement multithreaded data processing
- Optimize the model architecture.