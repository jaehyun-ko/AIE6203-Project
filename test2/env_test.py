import gym
from gym.wrappers import Monitor

from pyvirtualdisplay import Display
display = Display(visible=1, size=(160, 210))
display.start()

import gym
env = gym.make('Breakout-ram-v0')
# Uncomment following line to save video of our Agent interacting in this environment
# This can be used for debugging and studying how our agent is performing
# env = gym.wrappers.Monitor(env, './video/', force = True)
env.reset()

for _ in range(10000000):
    observation, reward, done, info = env.step(env.action_space.sample())
    env.render()
    # if done: break

env.close()