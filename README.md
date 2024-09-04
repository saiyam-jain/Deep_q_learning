# Deep Q-Learning for Stock Trading

This project implements a **Deep Q-Learning** algorithm for learning trade signals on sinusoidal data. It uses a **reinforcement learning** approach where an agent learns to buy, sell, or hold stocks based on market data.

## Project Overview

This project includes:
- A **Q-Network** for predicting actions based on state observations.
- A **Trading Environment** simulating stock prices with buy/sell/hold actions.
- A **Reinforcement Learning Agent** that learns through experience, stored in a replay buffer.

The goal is to train the agent to maximize its trading balance over time.

## Features
- **Deep Q-Network (DQN)**: A deep neural network that approximates the Q-value function to predict the best actions.
- **Replay Buffer**: Stores past experiences to break correlation between consecutive steps during training.
- **Sinusoidal Data Simulation**: The environment uses simulated stock data, based on a sinusoidal wave to mimic stock price movements.
- **Action Space**: The agent can choose to:
  - Buy
  - Sell
  - Hold
