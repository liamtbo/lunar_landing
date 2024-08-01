import gymnasium as gym
import torch
torch.manual_seed(1)

env = gym.make('LunarLander-v2', render_mode="human")

for episode in range(2):
    state, info = env.reset(seed=1)
    for step in range(2):
        print(f"state: {state}")
        next_state, _, _, _, _= env.step(1)
        state = next_state
