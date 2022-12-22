import torch
import numpy as np
from model import ReplayBuffer, Agent
from main import args

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu:0')

def frame2tensor(frame):
    if device.type == 'cuda':
            return torch.cuda.FloatTensor(frame)/255
    return torch.FloatTensor(frame)/255


replay_buffer = ReplayBuffer(args.replay_buffer_size, args.batch_size)
agent = Agent(128, 3, replay_buffer)
if args.render:
    from pyvirtualdisplay import Display
    display = Display(visible=1, size=(160, 210))
    display.start()
    

def train(env, render_flag):
    done = False
    score = 0

    frame = env.reset()
    frame = frame2tensor(frame)
    
    while not done:

        # Epsilon greedy
        action = agent.take_action(frame)
        # Action += 1, because we avoid 0 action (noop)
        next_frame, reward, done, info = env.step(action+1)
        
        if render_flag:

            env.render()
        
        # Torch Tensor
        next_frame = frame2tensor(next_frame)
        score += reward       

        # reward max is 1.0, useful in Bellman equation
        reward = max(min(reward, 1.0), -1.0)
        
        agent.buffer.accumulate(frame, action, reward, next_frame)
        
        frame = next_frame
        
        if agent.check_buffer():
            agent.train_step()
        
    return score, agent.epsilon
        
        
def test(env, render_flag):
    done = False
    score = 0
    agent.eval()
    nothing = 0
    # If fire is True, then
    # FIREEE !!
    fire = False
    
    with torch.no_grad():

        frame = env.reset()
        env.step(1)
        frame = frame2tensor(frame)

        while not done:

            action = agent.take_action(frame)
            # Check fire button
            action =  action if not fire else 0
            
            next_frame, reward, done, info = env.step(action+1)

            if reward == 0:
                nothing += 1
                
            else:
                # AI play the game with the ball
                nothing = 0
                
            if fire:
                fire = False
                
            if nothing >= 600:
                # If the AI, don't press fire button
                # We press it
                fire = True
            
            if render_flag:
                env.render()
            
            next_frame = frame2tensor(next_frame)
            score += reward
            
            frame = next_frame

    return score