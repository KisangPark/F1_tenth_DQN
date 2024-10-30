import os
import sys
import collections
import random
import gym
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# Hyperparameters
learning_rate = 0.0001 # 0.0001 --------------------> learning step
gamma = 0.99 #---------------------------------------> original 0.98, modified 0.99
buffer_limit = 50000 #original 50000, modified 100000
batch_size = 64 # original 32, modified 64
train_start = 5000 #5000

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)

RACETRACK = 'map_easy3'


class ReplayBuffer(): #-----------------------> reinforcement learning , replay buffer
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit) # history 저장, deque는 그냥 저장형

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n): #---------------------------------------------------> buffer history에서 n개 sampling
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch: #randomly selected minibatch의 내용을 lst에 append
            s, a, r, s_prime, done_mask = transition #state, action, reward, next state, done
            s_lst.append(s) #random으로 추출한 state action reward들을 list에 append
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
            torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
            torch.tensor(done_mask_lst) #sampling한 torch tensor들을 return

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module): 
# Q value: stae & action -> Return
# Q net: using deep Neural Network, map the state with action (almost same to policy function)
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(405, 256) #--------------------> 405 input: observation
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 5) #--------------------> modified to 7 /5 states to choose: big left, left, streight, right, big right

    def forward(self, x): #forwarding -> get the Q network result (state to action, policy)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def sample_action(self, obs, epsilon, memory_size): # epsilon greedy 반영하여 action하는 것... 얘가 실질적인 environment 내부에서의 action!
    #using self network & rofwarding learning through observation
        if memory_size < train_start:
            return random.randint(0, 4) # modified 4->6
        else:
            out = self.forward(obs) # action determined by self forwarding function & observation
            coin = random.random()
            if coin < epsilon: #epsilon greedy method!
                return random.randint(0, 4)# modified 4->6 #random choose
            else:
                return out.argmax().item() #argmax -> index 반환

    def action(self, obs): #action case
        out = self.forward(obs)
        return out.argmax().item()

    def action_to_stearing(self, a):
        steer = (a - 2) * (np.pi /30) #modified to 3 -> 30degree
        if a == 2:
            speed = 10.0 #5.0 -> 6.0
        elif a == 1 or a == 3:
            speed = 7.5 # 4.5
        else:
            speed = 6.0 #4.0
        return [steer, speed]

    def preprocess_lidar(self, ranges):
        eighth = int(len(ranges) / 8)

        return np.array(ranges[eighth:-eighth: 2])


def train(q, q_target, memory, optimizer, scheduler): # ****************************** training sequence -> get optimizer, scheduler, memory, qnet
#q, q_target 2개의 NN, memory buffer, optimizer 받아서 training 수행
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size) # 변수들 memory에서 sample해옴

        q_out = q(s) # q network에 state = observation 대입, action 취함 인줄 알았는데, state s에서의 state value function을 취함
        q_a = q_out.gather(1, a) #action value function
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1) #target network에서의 state value function
        target = r + gamma * max_q_prime * done_mask # target network의 state value update
        loss = F.smooth_l1_loss(q_a, target) #q network에서 

        optimizer.zero_grad()
        loss.backward() # backprop
        optimizer.step() # optimizer step -> gradient descent
        scheduler.step() #----------------------------------------> scheduler added


def plot_durations(laptimes):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(laptimes, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # 10개의 에피소드 평균을 가져 와서 도표 그리기
    if len(durations_t) >= 10:
        means = durations_t.unfold(0, 10, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(9), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # 도표가 업데이트되도록 잠시 멈춤
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def get_today():
    now = time.localtime()
    s = "%04d-%02d-%02d_%02d-%02d-%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    return s


def main(): # main 꼭 읽어보기
    today = get_today()
    work_dir = "./" + today
    os.makedirs(work_dir + '_' + RACETRACK)

    env = gym.make('f110_gym:f110-v0',
                   map="{}/maps/{}".format(current_dir, RACETRACK),
                   map_ext=".png", num_agents=1)
    q = Qnet()
    # q.load_state_dict(torch.load("{}\weigths\model_state_dict_easy1_fin.pt".format(current_dir)))
    q_target = Qnet()

    #q_target.load_state_dict(q.state_dict()) #여기에 pretrained model 넣고 학습하면 될듯??
    memory = ReplayBuffer()
    q.load_state_dict(torch.load("{}\\2024-06-13_22-28-32_map_easy3\\fast-model26.06_112.pt".format(current_dir)))
    q_target.load_state_dict(torch.load("{}\\2024-06-13_22-28-32_map_easy3\\fast-model26.06_112.pt".format(current_dir)))
    # poses = np.array([[0., 0., np.radians(0)]])
    # poses = np.array([[0.0702245, 0.3002981, 2.79787]]) # Oschersleben
    poses = np.array([[0.60070, -0.2753, 1.5707]])  # map easy

    print_interval = 10
    optimizer = optim.Adam(q.parameters(), lr=learning_rate) #----------------------------------------> adam optimizer
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 2000, gamma = 0.9) #Lambda # --------------------> scheduler
    

    fastlap = 10000.0
    laptimes = []

    for n_epi in range(4000): #10000
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))  # Linear annealing from 8% to 1% #n_epi/200 ------> use exponential?
        obs, r, done, info = env.reset(poses=poses)
        s = q.preprocess_lidar(obs['scans'][0])
        done = False

        #env.render()
        laptime = 0.0

        while not done: # ----------------------------------------> learning starts
            actions = []
            a = q.sample_action(torch.from_numpy(s).float(), epsilon, memory.size())
            action = q.action_to_stearing(a)
            actions.append(action)
            actions = np.array(actions) #actions -> from sample... replay buffer

            obs, r, done, info = env.step(actions) #environment에 action 대입 -> 다음 step의 reward & observation ( = state)가 출력

            s_prime = q.preprocess_lidar(obs['scans'][0]) #next state!
            #print(len(obs['scans'][0])) # added
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r / 100, s_prime, done_mask)) #memory에 ㅔ새롭게 대입
            s = s_prime

            laptime += r
            #env.render(mode='human_fast')

            if done:
                laptimes.append(laptime)
                plot_durations(laptimes)
                lap = round(obs['lap_times'][0], 3)
                if int(obs['lap_counts'][0]) == 2 and fastlap > lap:
                    torch.save(q.state_dict(), work_dir + '_' + RACETRACK + '/fast-model' + str(
                        round(obs['lap_times'][0], 3)) + '_' + str(n_epi) + '.pt')
                    fastlap = lap
                    break

        if memory.size() > train_start: # memory 충분히 차면 with random action
            train(q, q_target, memory, optimizer, scheduler) # scheduler added

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%"
                  .format(n_epi, laptime / print_interval, memory.size(), epsilon * 100))

    print('train finish')
    env.close()


def eval():
    env = gym.make('f110_gym:f110-v0',
                   map="{}/maps/{}".format(current_dir, RACETRACK),
                   map_ext=".png", num_agents=1)

    q = Qnet()
    q.load_state_dict(torch.load("{}\\fastest-model\\fast-model25.61_1903.pt".format(current_dir)))
    #q.load_state_dict(torch.load("{}\weigths\model_state_dict_easy1_fin.pt".format(current_dir))) # pretrained!
    poses = np.array([[0., 0., np.radians(90)]])
    speed = 3.0
    for t in range(5):
        obs, r, done, info = env.reset(poses=poses)
        s = q.preprocess_lidar(obs['scans'][0])

        env.render()
        done = False

        laptime = 0.0

        while not done:
            actions = []

            a = q.action(torch.from_numpy(s).float())
            action = q.action_to_stearing(a)
            actions.append(action)
            actions = np.array(actions)

            obs, r, done, info = env.step(actions)
            s_prime = q.preprocess_lidar(obs['scans'][0])

            s = s_prime

            laptime += r
            env.render(mode='human_fast')

            if done:
                break
    env.close()


if __name__ == '__main__':
    #main()
    eval()
