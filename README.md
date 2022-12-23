# AIE6203-Project


# Setting
## test enviroment
you can test your enviroment by this code
```python
import gym
from gym.wrappers import Monitor

from pyvirtualdisplay import Display
display = Display(visible=1, size=(160, 210))
display.start()

import gym
env = gym.make('Breakout-ram-v0')
env.reset()

for _ in range(10000000):
    observation, reward, done, info = env.step(env.action_space.sample())
    env.render()
    # if done: break

env.close()
```
#usage
## train
python main.py --train 'train_model_path'
other arguments are defined in python main.py 
inspired in 
