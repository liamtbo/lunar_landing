import gym
import numpy as np

env = gym.make('LunarLander-v2')

num_episodes = 10
for i_episode in range(num_episodes):
    observation = env.reset()
    total_reward = 0
    done = False
    while not done:
        env.render()
        action = env.action_space.sample()  # Random action
        observation, reward, done, info, test = env.step(action)
        # tmp = env.step(action)
        # print(tmp)
        total_reward += reward
    print(f"Episode {i_episode + 1}: Total Reward: {total_reward}")

env.close()