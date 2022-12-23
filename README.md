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

# usage

```shell
git clone https://github.com/jaehyun-ko/AIE6203-Project.git
cd AIE6203-Project
```


## DQN

```shell
python main.py --train 'model.pth' --render
```

model path:[model.pth](model.pth)

## A3C

```shell
python a3c.py --render true
```

model path : breakout-v4/model_#episode num.pth

https://youtu.be/VarnVisEbhk
