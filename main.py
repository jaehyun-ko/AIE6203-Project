
import gym
import torch
import argparse
import gym.wrappers as wrappers
from model import *
from utils import *
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
parser = argparse.ArgumentParser()

# Hyper Parameters
parser.add_argument('--epsilon', type=float, 
                        help='Set the start value of epsilon', default=0.3)
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
    replay_buffer = ReplayBuffer(args.replay_buffer_size, args.batch_size)

    env = gym.make('Breakout-v4')
    
    observation = env.reset()
    state_space = observation.shape
    print(state_space)
    state_raw = np.zeros(state_space, dtype=np.uint8)
    state_space = (state_space[2], state_space[0], state_space[1])
    frame = Transforms.to_gray(state_raw)
    action_space = env.action_space.n

    print(frame.shape, action_space)
    agent = Agent(frame.shape, action_space, replay_buffer)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')

    replay_buffer = ReplayBuffer(args.replay_buffer_size, args.batch_size)
    # if args.render:
    from pyvirtualdisplay import Display
    # display = Display(visible=1, size=(160, 210)
    display = Display(visible=1, size=(400, 400))
    display.start()

    # Save videos
    if args.record:
        env = wrappers.Monitor(env, "records/Breakout", video_callable=lambda episode_id:True, force=True)
    
    if args.test == '':
        if args.train != '':
            agent.load_model(args.train)
        # Training loop
        
        for episode in range(1, args.episodes):
            done = False
            score = 0
            observation = env.reset()
            frame = Transforms.to_gray(observation)

            while not done:
                # Epsilon greedy
                action = agent.take_action(frame)
                # Action += 1, because we avoid 0 action (noop)
                new_observation, reward, done, info = env.step(action)                
                if args.render:
                    env.render()
                
                # Torch Tensor
                next_frame = Transforms.to_gray(observation, new_observation)
                agent.buffer.accumulate(frame, action, reward, next_frame, int(done), observation)
                
                score += reward
                frame = next_frame
                observation = new_observation
            # if agent.check_buffer():
            agent.train_step()
            writer.add_scalar('Reward', score, episode)
            writer.add_scalar('Epsilon', agent.epsilon, episode)
            writer.add_scalar('Loss', agent.loss, episode)


            print(f'EPISODE {episode} REWARD [{score}] EPSILON [{agent.epsilon}]')
            history.append([episode, score])
            if not args.render:
                agent.plot(history)
            agent.update(episode)
            
            if episode%10000 == 0:
                agent.save_model(model_path ='ckpt/model_'+str(episode)+'.pth')
        
        # If it's the highest score, save model && clear memory
        if score > highest_score:
            highest_score = score
            print(f'Highest SCORE [{highest_score}]')
            agent.buffer.clear()
            agent.save_model()
    
    else:
        agent.load_model(args.test)
            
    highest_score = 120
    history = []
    
    # Eval Model
