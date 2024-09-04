import utils
from trading_env import TradingEnv
from trading_agent import TradingAgent

def main_func(num_episodes):
    data = (utils.create_sinusoidal_data()+1)*10
    normalized_data = utils.normalize_data(data, data.max(), data.min())

    # Create the environment
    env = TradingEnv(normalized_data)

    # Initialize the agent
    state_dim = env.reset().shape[0]
    action_dim = 3  # Buy, Sell, Hold
    agent = TradingAgent(state_dim, action_dim)

    # Training the agent
    num_episodes = num_episodes
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        count = 0

        while not done:
            count+=1
            action = agent.select_action(state, env.position)
            next_state, reward, done = env.step(action)
            agent.store_experience(state, action, reward, next_state, done)
            agent.update()
            total_reward += utils.unnormalize(reward, data.max(), data.min()) #unnormalized reward for reference

        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2f}, epsilon: {agent.epsilon}")

    print("Training completed.")

if __name__ == "__main__":
    main_func(num_episodes=500)