import argparse
import random

import torch
import numpy as np
from collections import deque
from game import SnakeGame,Direction,Point
from model import Linear_QNet,QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
GAMMA = 0.9
EXPLORATION_GAMES = 80   # epsilon decays to 0 over this many games

PRETRAINED_MODEL = 'pretrained.pth'  # shipped with the repo, used by --play
TRAINED_MODEL = 'model.pth'          # written by your own training runs

class Agent:
    def __init__(self):
        self.ngames=0
        self.epsilon = 0
        self.gamma = GAMMA
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model =Linear_QNet(14,256,3)
        self.trainer = QTrainer( self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        point_le = Point(head.x - 40, head.y)
        point_re = Point(head.x + 40, head.y)
        point_ue = Point(head.x, head.y - 40)
        point_de = Point(head.x, head.y + 40)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            

            

            (dir_r and game.is_collision(point_re)) or 
            (dir_l and game.is_collision(point_le)) or 
            (dir_u and game.is_collision(point_ue)) or 
            (dir_d and game.is_collision(point_de)),

            # Danger right
            (dir_u and game.is_collision(point_re)) or 
            (dir_d and game.is_collision(point_le)) or 
            (dir_l and game.is_collision(point_ue)) or 
            (dir_r and game.is_collision(point_de)),

            # Danger left
            (dir_d and game.is_collision(point_re)) or 
            (dir_u and game.is_collision(point_le)) or 
            (dir_r and game.is_collision(point_ue)) or 
            (dir_l and game.is_collision(point_de)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)


    def remember(self,state,action,reward,next_state,done):
         self.memory.append((state, action, reward, next_state, done))

    def train_long_memo(self):
        if len(self.memory)> BATCH_SIZE:
            minisample = random.sample(self.memory,BATCH_SIZE)
        else:
            minisample = self.memory

        states,actions,rewards,next_states,dones = zip(*minisample)   
        self.trainer.train_step(states,actions,rewards,next_states,dones)

    def train_short_memo(self,state,action,reward,next_state,done):
        self.trainer.train_step(state,action,reward,next_state,done)

    def get_action(self,state):
        self.epsilon = EXPLORATION_GAMES - self.ngames
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
    
def play(n_games=None, model_file=PRETRAINED_MODEL):
    """Runs a saved model greedily (no exploration, no training)."""
    agent = Agent()
    agent.model.load(model_file)
    game = SnakeGame(speed=20)
    played, record = 0, 0
    while n_games is None or played < n_games:
        state = agent.get_state(game)
        with torch.no_grad():
            move = torch.argmax(agent.model(torch.tensor(state, dtype=torch.float))).item()
        final_move = [0, 0, 0]
        final_move[move] = 1
        _, done, score = game.play_step(final_move)
        if done:
            played += 1
            record = max(record, score)
            print('GAME', played, 'score', score, 'record', record)
            game.reset()


def train(model_file=TRAINED_MODEL):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game= SnakeGame()
    while True :
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward,done,score = game.play_step(final_move)
        state_new = agent.get_state(game)
        agent.train_short_memo(state_old,final_move,reward,state_new,done)
        agent.remember(state_old,final_move,reward,state_new,done)
        if done : 
            game.reset()
            agent.ngames +=1
            agent.train_long_memo()
            if score> record : 
                record = score
                agent.model.save(model_file)
            print('GAME',agent.ngames,'score',score,'record',record)
            plot_scores.append(score)
            total_score+= score
            mean_score= total_score/agent.ngames
            plot_mean_scores.append(mean_score)
            plot(plot_scores,plot_mean_scores)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Snake game agent trained with Deep Q-Learning')
    parser.add_argument('--play', action='store_true', help='watch the saved model play instead of training')
    parser.add_argument('--games', type=int, default=None, help='number of games to play (default: unlimited)')
    parser.add_argument('--model', default=None,
                        help=f'weights file in model/ (play default: {PRETRAINED_MODEL}, train default: {TRAINED_MODEL})')
    args = parser.parse_args()
    if args.play:
        play(args.games, args.model or PRETRAINED_MODEL)
    else:
        train(args.model or TRAINED_MODEL)
