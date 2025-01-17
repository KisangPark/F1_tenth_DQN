# # MIT License
#
# # Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng
#
# # Permission is hereby granted, free of charge, to any person obtaining a copy
# # of this software and associated documentation files (the "Software"), to deal
# # in the Software without restriction, including without limitation the rights
# # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# # copies of the Software, and to permit persons to whom the Software is
# # furnished to do so, subject to the following conditions:
#
# # The above copyright notice and this permission notice shall be included in all
# # copies or substantial portions of the Software.
#
# # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# # SOFTWARE.
#
# '''
# Author: Hongrui Zheng
# '''
#
# # gym imports
# import gym
# from gym import error, spaces, utils
# from gym.utils import seeding
#
# # base classes
# from f110_gym.envs.base_classes import Simulator
#
# # others
# import numpy as np
# import os
# import time
#
# # gl
# import pyglet
# pyglet.options['debug_gl'] = False
# from pyglet import gl
#
# # constants
#
# # rendering
# VIDEO_W = 600
# VIDEO_H = 400
# WINDOW_W = 1000
# WINDOW_H = 800
#
# class F110Env(gym.Env, utils.EzPickle):
#     """
#     OpenAI gym environment for F1TENTH
#
#     Env should be initialized by calling gym.make('f110_gym:f110-v0', **kwargs)
#
#     Args:
#         kwargs:
#             seed (int, default=12345): seed for random state and reproducibility
#
#             map (str, default='vegas'): name of the map used for the environment. Currently, available environments include: 'berlin', 'vegas', 'skirk'. You could use a string of the absolute path to the yaml file of your custom map.
#
#             map_ext (str, default='png'): image extension of the map image file. For example 'png', 'pgm'
#
#             params (dict, default={'mu': 1.0489, 'C_Sf':, 'C_Sr':, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch':7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}): dictionary of vehicle parameters.
#             mu: surface friction coefficient
#             C_Sf: Cornering stiffness coefficient, front
#             C_Sr: Cornering stiffness coefficient, rear
#             lf: Distance from center of gravity to front axle
#             lr: Distance from center of gravity to rear axle
#             h: Height of center of gravity
#             m: Total mass of the vehicle
#             I: Moment of inertial of the entire vehicle about the z axis
#             s_min: Minimum steering angle constraint
#             s_max: Maximum steering angle constraint
#             sv_min: Minimum steering velocity constraint
#             sv_max: Maximum steering velocity constraint
#             v_switch: Switching velocity (velocity at which the acceleration is no longer able to create wheel spin)
#             a_max: Maximum longitudinal acceleration
#             v_min: Minimum longitudinal velocity
#             v_max: Maximum longitudinal velocity
#             width: width of the vehicle in meters
#             length: length of the vehicle in meters
#
#             num_agents (int, default=2): number of agents in the environment
#
#             timestep (float, default=0.01): physics timestep
#
#             ego_idx (int, default=0): ego's index in list of agents
#     """
#     metadata = {'render.modes': ['human', 'human_fast']}
#
#     def __init__(self, **kwargs):
#         # kwargs extraction
#         try:
#             self.seed = kwargs['seed']
#         except:
#             self.seed = 12345
#         try:
#             self.map_name = kwargs['map']
#             # different default maps
#             if self.map_name == 'berlin':
#                 self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/berlin.yaml'
#             elif self.map_name == 'skirk':
#                 self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/skirk.yaml'
#             elif self.map_name == 'levine':
#                 self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/levine.yaml'
#             else:
#                 self.map_path = self.map_name + '.yaml'
#         except:
#             self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/vegas.yaml'
#
#         try:
#             self.map_ext = kwargs['map_ext']
#         except:
#             self.map_ext = '.png'
#
#         try:
#             self.params = kwargs['params']
#         except:
#             self.params = {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch': 7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}
#
#         # simulation parameters
#         try:
#             self.num_agents = kwargs['num_agents']
#         except:
#             self.num_agents = 2
#
#         try:
#             self.timestep = kwargs['timestep']
#         except:
#             self.timestep = 0.01
#
#         # default ego index
#         try:
#             self.ego_idx = kwargs['ego_idx']
#         except:
#             self.ego_idx = 0
#
#         # radius to consider done
#         self.start_thresh = 0.5  # 10cm
#
#         # env states
#         self.poses_x = []
#         self.poses_y = []
#         self.poses_theta = []
#         self.collisions = np.zeros((self.num_agents, ))
#         self.collision_idx = np.zeros((self.num_agents, ))
#
#         # loop completion
#         self.near_start = True
#         self.num_toggles = 0
#
#         # race info
#         self.lap_times = np.zeros((self.num_agents, ))
#         self.lap_counts = np.zeros((self.num_agents, ))
#         self.current_time = 0.0
#
#         # finish line info
#         self.num_toggles = 0
#         self.near_start = True
#         self.near_starts = np.array([True]*self.num_agents)
#         self.toggle_list = np.zeros((self.num_agents,))
#         self.start_xs = np.zeros((self.num_agents, ))
#         self.start_ys = np.zeros((self.num_agents, ))
#         self.start_thetas = np.zeros((self.num_agents, ))
#         self.start_rot = np.eye(2)
#
#         # initiate stuff
#         self.sim = Simulator(self.params, self.num_agents, self.seed)
#         self.sim.set_map(self.map_path, self.map_ext)
#
#         # rendering
#         self.renderer = None
#         self.current_obs = None
#
#     def __del__(self):
#         """
#         Finalizer, does cleanup
#         """
#         pass
#
#     def _check_done(self):
#         """
#         Check if the current rollout is done
#
#         Args:
#             None
#
#         Returns:
#             done (bool): whether the rollout is done
#             toggle_list (list[int]): each agent's toggle list for crossing the finish zone
#         """
#
#         # this is assuming 2 agents
#         # TODO: switch to maybe s-based
#         left_t = 2
#         right_t = 2
#
#         poses_x = np.array(self.poses_x)-self.start_xs
#         poses_y = np.array(self.poses_y)-self.start_ys
#         delta_pt = np.dot(self.start_rot, np.stack((poses_x, poses_y), axis=0))
#         temp_y = delta_pt[1,:]
#         idx1 = temp_y > left_t
#         idx2 = temp_y < -right_t
#         temp_y[idx1] -= left_t
#         temp_y[idx2] = -right_t - temp_y[idx2]
#         temp_y[np.invert(np.logical_or(idx1, idx2))] = 0
#
#         dist2 = delta_pt[0,:]**2 + temp_y**2
#         closes = dist2 <= 0.1
#         for i in range(self.num_agents):
#             if closes[i] and not self.near_starts[i]:
#                 self.near_starts[i] = True
#                 self.toggle_list[i] += 1
#             elif not closes[i] and self.near_starts[i]:
#                 self.near_starts[i] = False
#                 self.toggle_list[i] += 1
#             self.lap_counts[i] = self.toggle_list[i] // 2
#             if self.toggle_list[i] < 4:
#                 self.lap_times[i] = self.current_time
#
#         done = np.all(self.collisions > 0) or np.all(self.toggle_list >= 4)
#
#         return done, self.toggle_list >= 4
#
#     def _update_state(self, obs_dict):
#         """
#         Update the env's states according to observations
#
#         Args:
#             obs_dict (dict): dictionary of observation
#
#         Returns:
#             None
#         """
#         self.poses_x = obs_dict['poses_x']
#         self.poses_y = obs_dict['poses_y']
#         self.poses_theta = obs_dict['poses_theta']
#         self.collisions = obs_dict['collisions']
#
#     def step(self, action):
#         """
#         Step function for the gym env
#
#         Args:
#             action (np.ndarray(num_agents, 2))
#
#         Returns:
#             obs (dict): observation of the current step
#             reward (float, default=self.timestep): step reward, currently is physics timestep
#             done (bool): if the simulation is done
#             info (dict): auxillary information dictionary
#         """
#
#         # call simulation step
#         obs = self.sim.step(action)
#         obs['lap_times'] = self.lap_times
#         obs['lap_counts'] = self.lap_counts
#
#         self.current_obs = obs
#
#         # times
#         reward = self.timestep
#         self.current_time = self.current_time + self.timestep
#
#         # update data member
#         self._update_state(obs)
#
#         # check done
#         done, toggle_list = self._check_done()
#         info = {'checkpoint_done': toggle_list}
#
#         return obs, reward, done, info
#
#     def reset(self, poses):
#         """
#         Reset the gym environment by given poses
#
#         Args:
#             poses (np.ndarray (num_agents, 3)): poses to reset agents to
#
#         Returns:
#             obs (dict): observation of the current step
#             reward (float, default=self.timestep): step reward, currently is physics timestep
#             done (bool): if the simulation is done
#             info (dict): auxillary information dictionary
#         """
#         # reset counters and data members
#         self.current_time = 0.0
#         self.collisions = np.zeros((self.num_agents, ))
#         self.num_toggles = 0
#         self.near_start = True
#         self.near_starts = np.array([True]*self.num_agents)
#         self.toggle_list = np.zeros((self.num_agents,))
#
#         # states after reset
#         self.start_xs = poses[:, 0]
#         self.start_ys = poses[:, 1]
#         self.start_thetas = poses[:, 2]
#         self.start_rot = np.array([[np.cos(-self.start_thetas[self.ego_idx]), -np.sin(-self.start_thetas[self.ego_idx])], [np.sin(-self.start_thetas[self.ego_idx]), np.cos(-self.start_thetas[self.ego_idx])]])
#
#         # call reset to simulator
#         self.sim.reset(poses)
#
#         # get no input observations
#         action = np.zeros((self.num_agents, 2))
#         obs, reward, done, info = self.step(action)
#         return obs, reward, done, info
#
#     def update_map(self, map_path, map_ext):
#         """
#         Updates the map used by simulation
#
#         Args:
#             map_path (str): absolute path to the map yaml file
#             map_ext (str): extension of the map image file
#
#         Returns:
#             None
#         """
#         self.sim.set_map(map_path, map_ext)
#
#     def update_params(self, params, index=-1):
#         """
#         Updates the parameters used by simulation for vehicles
#
#         Args:
#             params (dict): dictionary of parameters
#             index (int, default=-1): if >= 0 then only update a specific agent's params
#
#         Returns:
#             None
#         """
#         self.sim.update_params(params, agent_idx=index)
#
#     def render(self, mode='human'):
#         """
#         Renders the environment with pyglet. Use mouse scroll in the window to zoom in/out, use mouse click drag to pan. Shows the agents, the map, current fps (bottom left corner), and the race information near as text.
#
#         Args:
#             mode (str, default='human'): rendering mode, currently supports:
#                 'human': slowed down rendering such that the env is rendered in a way that sim time elapsed is close to real time elapsed
#                 'human_fast': render as fast as possible
#
#         Returns:
#             None
#         """
#         assert mode in ['human', 'human_fast']
#         if self.renderer is None:
#             # first call, initialize everything
#             from f110_gym.envs.rendering import EnvRenderer
#             self.renderer = EnvRenderer(WINDOW_W, WINDOW_H)
#             self.renderer.update_map(self.map_name, self.map_ext)
#         self.renderer.update_obs(self.current_obs)
#         self.renderer.dispatch_events()
#         self.renderer.on_draw()
#         self.renderer.flip()
#         if mode == 'human':
#             time.sleep(0.005)
#         elif mode == 'human_fast':
#             pass

# MIT License
# MIT License

# Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''
Author: Hongrui Zheng
'''

# gym imports
import gym
from gym import error, spaces, utils
from gym.utils import seeding

# base classes
from f110_gym.envs.base_classes import Simulator

# others
import numpy as np
import os
import time

# gl
import pyglet

pyglet.options['debug_gl'] = False
from pyglet import gl

# constants

# rendering
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800


class F110Env(gym.Env, utils.EzPickle):
    """
    OpenAI gym environment for F1TENTH

    Env should be initialized by calling gym.make('f110_gym:f110-v0', **kwargs)

    Args:
        kwargs:
            seed (int, default=12345): seed for random state and reproducibility

            map (str, default='vegas'): name of the map used for the environment. Currently, available environments include: 'berlin', 'vegas', 'skirk'. You could use a string of the absolute path to the yaml file of your custom map.

            map_ext (str, default='png'): image extension of the map image file. For example 'png', 'pgm'

            params (dict, default={'mu': 1.0489, 'C_Sf':, 'C_Sr':, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch':7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}): dictionary of vehicle parameters.
            mu: surface friction coefficient
            C_Sf: Cornering stiffness coefficient, front
            C_Sr: Cornering stiffness coefficient, rear
            lf: Distance from center of gravity to front axle
            lr: Distance from center of gravity to rear axle
            h: Height of center of gravity
            m: Total mass of the vehicle
            I: Moment of inertial of the entire vehicle about the z axis
            s_min: Minimum steering angle constraint
            s_max: Maximum steering angle constraint
            sv_min: Minimum steering velocity constraint
            sv_max: Maximum steering velocity constraint
            v_switch: Switching velocity (velocity at which the acceleration is no longer able to create wheel spin)
            a_max: Maximum longitudinal acceleration
            v_min: Minimum longitudinal velocity
            v_max: Maximum longitudinal velocity
            width: width of the vehicle in meters
            length: length of the vehicle in meters

            num_agents (int, default=2): number of agents in the environment

            timestep (float, default=0.01): physics timestep

            ego_idx (int, default=0): ego's index in list of agents
    """
    metadata = {'render.modes': ['human', 'human_fast']}

    def __init__(self, **kwargs):
        # kwargs extraction
        try:
            self.seed = kwargs['seed']
        except:
            self.seed = 12345
        try:
            self.map_name = kwargs['map']
            # different default maps
            if self.map_name == 'berlin':
                self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/berlin.yaml'
            elif self.map_name == 'skirk':
                self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/skirk.yaml'
            elif self.map_name == 'levine':
                self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/levine.yaml'
            else:
                self.map_path = self.map_name + '.yaml'
        except:
            self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/vegas.yaml'

        try:
            self.map_ext = kwargs['map_ext']
        except:
            self.map_ext = '.png'

        try:
            self.params = kwargs['params']
        except:
            self.params = {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074,
                           'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2,
                           'v_switch': 7.319, 'a_max': 9.51, 'v_min': -5.0, 'v_max': 20.0, 'width': 0.31,
                           'length': 0.58}

        # simulation parameters
        try:
            self.num_agents = kwargs['num_agents']
        except:
            self.num_agents = 2

        try:
            self.timestep = kwargs['timestep']
        except:
            self.timestep = 0.01

        # default ego index
        try:
            self.ego_idx = kwargs['ego_idx']
        except:
            self.ego_idx = 0

        # radius to consider done
        self.start_thresh = 0.5  # 10cm

        # env states
        self.poses_x = []
        self.poses_y = []
        self.poses_theta = []
        self.collisions = np.zeros((self.num_agents,))
        # TODO: collision_idx not used yet
        # self.collision_idx = -1 * np.ones((self.num_agents, ))

        # loop completion
        self.near_start = True
        self.num_toggles = 0

        # race info
        self.lap_times = np.zeros((self.num_agents,))
        self.lap_counts = np.zeros((self.num_agents,))
        self.current_time = 0.0

        # finish line info
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True] * self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,))
        self.start_xs = np.zeros((self.num_agents,))
        self.start_ys = np.zeros((self.num_agents,))
        self.start_thetas = np.zeros((self.num_agents,))
        self.start_rot = np.eye(2)

        # initiate stuff
        self.sim = Simulator(self.params, self.num_agents, self.seed)
        self.sim.set_map(self.map_path, self.map_ext)

        # rendering
        self.renderer = None
        self.current_obs = None

        # map checkpoint
        self.checklist = np.zeros((15))  # 추가

    def __del__(self):
        """
        Finalizer, does cleanup
        """
        pass

    def _check_done(self):
        """
        Check if the current rollout is done

        Args:
            None

        Returns:
            done (bool): whether the rollout is done
            toggle_list (list[int]): each agent's toggle list for crossing the finish zone
        """

        # this is assuming 2 agents
        # TODO: switch to maybe s-based
        left_t = 2
        right_t = 2

        poses_x = np.array(self.poses_x) - self.start_xs
        poses_y = np.array(self.poses_y) - self.start_ys
        delta_pt = np.dot(self.start_rot, np.stack((poses_x, poses_y), axis=0))
        temp_y = delta_pt[1, :]
        idx1 = temp_y > left_t
        idx2 = temp_y < -right_t
        temp_y[idx1] -= left_t
        temp_y[idx2] = -right_t - temp_y[idx2]
        temp_y[np.invert(np.logical_or(idx1, idx2))] = 0

        dist2 = delta_pt[0, :] ** 2 + temp_y ** 2
        closes = dist2 <= 0.1
        for i in range(self.num_agents):
            if closes[i] and not self.near_starts[i]:
                self.near_starts[i] = True
                self.toggle_list[i] += 1
            elif not closes[i] and self.near_starts[i]:
                self.near_starts[i] = False
                self.toggle_list[i] += 1
            self.lap_counts[i] = self.toggle_list[i] // 2
            if self.toggle_list[i] < 4:
                self.lap_times[i] = self.current_time

        done = (self.collisions[self.ego_idx]) or np.all(self.toggle_list >= 4)

        return done, self.toggle_list >= 4

    def _update_state(self, obs_dict):
        """
        Update the env's states according to observations

        Args:
            obs_dict (dict): dictionary of observation

        Returns:
            None
        """
        self.poses_x = obs_dict['poses_x']
        self.poses_y = obs_dict['poses_y']
        self.poses_theta = obs_dict['poses_theta']
        self.collisions = obs_dict['collisions']

    def step(self, action):
        """
        Step function for the gym env

        Args:
            action (np.ndarray(num_agents, 2))

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxillary information dictionary
        """
        # goals
        #modification!
        # map_easy1
        '''goals = [[967, 1198], [1031, 700], [967, 261], [603, 192], [230, 261], [164, 700], [260, 1208], [596, 1253]]
        for goal in goals:
            res = 0.007712
            origin = [-5.275, -1.381]S
            height = 1420
            goal[0] = goal[0] * res + origin[0]
            goal[1] = (height - goal[1]) * res + origin[1]'''

        # map_easy3
        '''goals = [[155, 281], [272, 182], [380, 230], [1361, 1335], [1322, 1365], [1235, 1369], [1184, 1354],
                 [293, 1383], [225, 1395], [167, 1357], [137, 1315]]
        for goal in goals:
            res = 0.02
            origin = [-2.7, -19.32]
            height = 1646
            goal[0] = goal[0] * res + origin[0]
            goal[1] = (height - goal[1]) * res + origin[1]'''

        # Oschersleben
        '''goals = [[16.5714257599952,11.1812245444287],[12.8416221451935,15.0974887264338],[16.5814160567464,18.9412589142763],[39.492058946279,19.9348045069869],[14.1078290454423,27.1549161072849],[7.61911291180757,23.7769210252349],[7.21711487137997,12.2406145719344],[3.60556593185953,8.69409417685197],[0.241852920484369,23.1745374181134],[8.03772101123323,32.7608876655171],[46.8365401276746,25.2124972313711],[50.7052324296386,18.2651738235072],[68.0646681857952,15.5143885046738],[73.2807835360562,2.72554645143258]]
        # excel goals...
        for goal in goals:
            res = 1 #0.02 for easy
            origin = [-47.8591, -6.19946] #-55.07650, -33.57884
            height = 1646
            goal[0] = goal[0] * res + origin[0]
            goal[1] = goal[1] * res + origin[1]'''

        # call simulation step
        obs = self.sim.step(action)
        obs['lap_times'] = self.lap_times
        obs['lap_counts'] = self.lap_counts
        self.pre_lap_counts = list(self.lap_counts)

        self.current_obs = obs
        # times
        reward = 1000 * self.timestep #----------------------------------------------> here, award = timestep만큼 부여
        
        #modified rewards
        
        if np.argmin(obs['scans'][0]) >= 480 and np.argmin(obs['scans'][0]) <= 600:
            reward -= 6000 #1
        elif np.argmin(obs['scans'][0]) >= 240 and np.argmin(obs['scans'][0]) <= 480:
            reward -= 3000
        elif np.argmin(obs['scans'][0]) >= 600 and np.argmin(obs['scans'][0]) <= 840:
            reward -= 3000
        else:
            reward += 6000
        if min(obs['scans'][0]) < 0.5: #원래: 0.5일 때 -5
            reward -= 9000 #5

        # 원래 1000
        #print(len(obs['scans'][0]))
        #for front_distance
        '''front_distance = 0
        for iter in range(279):
            i = iter+400
            front_distance += obs['scans'][0][i]/280

        right_distance = 0
        left_distance = 0
        for iter in range(399):
            right_distance += obs['scans'][0][iter]/400
            left_distance += obs['scans'][0][iter+680]/400
            side_distance = min(right_distance, left_distance)

        #direction = np.argmin(obs['scans'][0])
        #obs(['scans'][0][direction]) #가장 작을 때, 그 값
        #print (front_distance, side_distance)
        if front_distance > 2:
            reward +=200
            if side_distance < 0.6:
                reward +=100
            else:
                reward +=50
        elif front_distance > 1:
            reward -= 50
        elif front_distance < 0.7:
            reward -=200
        elif front_distance < 0.5:
            reward -= 400
        '''
        '''
        if np.argmin(obs['scans'][0]) >= 300 and np.argmin(obs['scans'][0]) <= 780: #전방 물체가 가장 가까움
            reward -= 1 #1
        elif np.argmin(obs['scans'][0]) < 300 or np.argmin(obs['scans'][0]) > 780:
            reward += 6 #2
        if min(obs['scans'][0]) < 1: #원래: 0.5일 때 -5
            reward -= 3 #5
        #추가
        elif min(obs['scans'][0]) < 0.5:
            reward -=8
        elif min(obs['scans'][0]) < 0.3:
            reward -=20
        '''

        #obs['poses_x'][0] > goal[0] - 0.5 and 
        #obs['poses_y'][0] > goal[1] - 0.5 and 
        # codes to use goal
        '''for i, goal in enumerate(goals):
            if self.checklist[i] == 1:
                continue
            if obs['poses_x'][0] < goal[0] + 0.3 and obs['poses_y'][0] < goal[1] + 0.3: # original 0.5
                print('goal pass', i)
                self.checklist[i] = 1
                reward += 10000'''

        self.current_time = self.current_time + self.timestep

        # update data member
        self._update_state(obs)

        # check done
        done, toggle_list = self._check_done()
        info = {'checkpoint_done': toggle_list}
        if self.collisions[self.ego_idx]:
            reward = 0 # if collide, reward = 0
        if self.lap_counts[0] != self.pre_lap_counts[0]:
            self.checklist = np.zeros((15))

        return obs, reward, done, info

    def reset(self, poses):
        """
        Reset the gym environment by given poses

        Args:
            poses (np.ndarray (num_agents, 3)): poses to reset agents to

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxillary information dictionary
        """
        # reset counters and data members
        self.current_time = 0.0
        self.collisions = np.zeros((self.num_agents,))
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True] * self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,))

        # states after reset
        self.start_xs = poses[:, 0]
        self.start_ys = poses[:, 1]
        self.start_thetas = poses[:, 2]
        self.start_rot = np.array(
            [[np.cos(-self.start_thetas[self.ego_idx]), -np.sin(-self.start_thetas[self.ego_idx])],
             [np.sin(-self.start_thetas[self.ego_idx]), np.cos(-self.start_thetas[self.ego_idx])]])

        # call reset to simulator
        self.sim.reset(poses)

        # get no input observations
        action = np.zeros((self.num_agents, 2))
        obs, reward, done, info = self.step(action)
        return obs, reward, done, info

    def update_map(self, map_path, map_ext):
        """
        Updates the map used by simulation

        Args:
            map_path (str): absolute path to the map yaml file
            map_ext (str): extension of the map image file

        Returns:
            None
        """
        self.sim.set_map(map_path, map_ext)

    def update_params(self, params, index=-1):
        """
        Updates the parameters used by simulation for vehicles

        Args:
            params (dict): dictionary of parameters
            index (int, default=-1): if >= 0 then only update a specific agent's params

        Returns:
            None
        """
        self.sim.update_params(params, agent_idx=index)

    def render(self, mode='human'):
        """
        Renders the environment with pyglet. Use mouse scroll in the window to zoom in/out, use mouse click drag to pan. Shows the agents, the map, current fps (bottom left corner), and the race information near as text.

        Args:
            mode (str, default='human'): rendering mode, currently supports:
                'human': slowed down rendering such that the env is rendered in a way that sim time elapsed is close to real time elapsed
                'human_fast': render as fast as possible

        Returns:
            None
        """
        assert mode in ['human', 'human_fast']
        if self.renderer is None:
            # first call, initialize everything
            from f110_gym.envs.rendering import EnvRenderer
            self.renderer = EnvRenderer(WINDOW_W, WINDOW_H)
            self.renderer.update_map(self.map_name, self.map_ext)
        self.renderer.update_obs(self.current_obs)
        self.renderer.dispatch_events()
        self.renderer.on_draw()
        self.renderer.flip()
        if mode == 'human':
            time.sleep(0.005)
        elif mode == 'human_fast':
            pass
# Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



# gym imports

