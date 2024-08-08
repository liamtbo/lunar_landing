# Lunar Lander DQN
##### by Liam Bouffard

## Description

## State Space
The state space of the LunarLander-v2 environment is an 8-dimensional continuous space that 
represents the state of the lunar lander. Each state is a vector containing the following elements:
- x position: Horizontal coordinate of the lander (float).
- y position: Vertical coordinate of the lander (float).
- x velocity: Horizontal velocity of the lander (float).
- y velocity: Vertical velocity of the lander (float).
- Angle: Orientation of the lander (radians, float).
- Angular velocity: Rate of change of the lander's angle (float).
- Left leg contact: Boolean value indicating whether the left leg is in contact with the ground (0 or 1).
- Right leg contact: Boolean value indicating whether the right leg is in contact with the ground (0 or 1).


## Action Space
The action space of the LunarLander-v2 environment is discrete and consists of 4 possible actions that control the lunar lander's engines:
- Action 0: Do nothing.
- Action 1: Fire the left orientation engine (applies a force to rotate the lander counterclockwise).
- Action 2: Fire the main engine (applies a vertical force to slow the descent or lift the lander).
- Action 3: Fire the right orientation engine (applies a force to rotate the lander clockwise).

## Reward Space
- +100  Reward for a successful landing (varies based on landing quality).
- -100: Penalty for crashing.
- -0.3 per frame: Penalty for firing the main engine (discourages excessive use).
- Small positive/negative rewards are given for keeping the lander upright and minimizing velocity.
