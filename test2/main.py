
import gym
import torch
import argparse
import gym.wrappers as wrappers
from train import *
from utils import *

parser = argparse.ArgumentParser()

# Hyper Parameters
parser.add_argument('--epsilon', type=float, 
                        help='Set the start value of epsilon', default=1.0)
parser.add_argument('--min_epsilon', type=float, 
                        help='End of epsilon value', default=0.01)
parser.add_argument('--replay_buffer_size', type=int, 
                        help='replay buffer size', default=100000)
parser.add_argument('--batch_size', type=int, 
                        help='Batch size', default=32)
parser.add_argument('--gamma', type=float, 
                        help='Gamma value', default=0.99)
parser.add_argument('--learning_rate', type=float, 
                        help='Learning rate value', default=0.0003)
parser.add_argument('--episodes', type=int, 
                        help='Number of episode', default=10000)
parser.add_argument('--record', action='store_true', 
                        help='Record boolean')

parser.add_argument('--train', type=str, 
                        help='Set model to train it', default='')
parser.add_argument('--test', type=str, 
                        help='Set model to test it', default='')

parser.add_argument('--model_path', type=str, 
                        help='Name to save it', default='ckpt/model.pth')
parser.add_argument('--render', action='store_true', 
                        help='Render Boolean')

args = parser.parse_args()

if __name__ == '__main__':
    
    history = []
    highest_score = 0
    score = 0
    
    env = gym.make('Breakout-ram-v0')
    env.reset()
    
    # Save videos
    if args.record:
        env = wrappers.Monitor(env, "records/Breakout", video_callable=lambda episode_id:True, force=True)
    
    if args.test == '':
        if args.train != '':
            
            agent.load_model(args.train)
        
        # Training
        for episode in range(1, args.episodes):
            reward, epsilon = train(env, args.render)
            print(f'EPISODE {episode} REWARD [{reward}] EPSILON [{epsilon}]')
            history.append([episode, reward])
            agent.plot(history)
            agent.update(episode)
            
            # If it's the highest score, save model && clear memory
            if reward > highest_score:
                highest_score = reward
                print(f'Highest SCORE [{highest_score}]')
                agent.buffer.clear()
                agent.save_model()
    
    else:
        agent.load_model(args.test)
            
    highest_score = 119
    history = []
    
    # Eval Model
    for episode in range(1, args.episodes):
    
        reward = test(env, args.render)
        print('EPISODE {0} REWARD [{1}]'.format(episode, reward))

        history.append([episode, reward])
        agent.plot(history)
        
        # Try to beat highest score
        if reward > highest_score:
            print('HIGHEST SCORE [{0}]'.format(reward))
            break
        
    print('Closing environnement..')
    env.close()