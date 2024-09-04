import numpy as np

# The Trading Environment
class TradingEnv:
    def __init__(self, data):
        self.data = data
        self.brokerage = 0
        # self.brokerage = 0.001
        self.reset()

    def reset(self):
        self.trades = 0
        self.position = 0  # Number of stocks held
        self.look_back_window = 100
        self.current_step = self.look_back_window
        self.balance = 2 # Normalized balance (2x the max-min range of stock price)
        self.total_steps = len(self.data)
        self.done = False
        return self._get_observation()

    def _get_observation(self):
        obs = self.data[self.current_step-self.look_back_window:self.current_step]
        return np.append(obs.flatten(), [self.position, self.balance])

    def step(self, action):
        current_price = self.data[self.current_step] # normalized

        if self.current_step >= self.total_steps-1:
            self.done = True
            self.current_step -= 1
            if self.position == -1:
                self.position += 1
                self.trades += 1
                reward = -(current_price + self.brokerage)
                self.balance += reward
            elif self.position == 1:
                self.position -= 1
                self.trades += 1
                reward = current_price - self.brokerage
                self.balance += reward
            elif self.position == 0:
                reward = -(2*current_price) # 2x of current price
            print(f"Done all steps, total trades: {self.trades}, position: {self.position}, balance: {20*self.balance}")


        elif (self.balance + current_price <= 0) and self.position == 1 and action == 1:
            self.done = True
            reward = -(2 * current_price)  # 2x of current price
            print(
                f"Low balance, total trades: {self.trades}, in time steps seen: {self.current_step}, open position: {self.position} balance after close: {20 * (self.balance + current_price)}"
            )

        elif (self.balance - current_price <= 0) and self.position == -1 and action == 0:
            self.done = True
            reward = -(2 * current_price)  # 2x of current price
            print(
                f"Low balance, total trades: {self.trades}, in time steps seen: {self.current_step}, open position: {self.position} balance after close: {20 * (self.balance - current_price)}"
            )
            
        elif action == 0 and self.position == 0:  # Buy
            self.position += 1
            reward = -(current_price + self.brokerage)
            self.balance += reward
        elif action == 0 and self.position == -1: # Close short and go long
            self.position += 2
            self.trades += 1
            reward = -(2*current_price + self.brokerage)
            self.balance += reward
        elif action == 1 and self.position == 0: # Sell
            self.position -= 1
            reward = current_price - self.brokerage
            self.balance += reward
        elif action == 1 and self.position == 1:  # Close long and go short
            self.position -= 2
            self.trades += 1
            reward = 2*current_price - self.brokerage
            self.balance += reward
        elif (action == 0 and self.position == 1) or (action == 1 and self.position == -1): # discourage invalid buy/sell action
            reward = -(2*current_price) # 2x of current price
        elif action == 2: # Hold
            # reward = 0
            reward = -0.004
            self.balance += reward

        self.current_step += 1
        
        next_state = self._get_observation()
        return next_state, reward, self.done