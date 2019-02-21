import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):

    def make_world(self):
        world = World()
        # create obstacles first
        self.obstacles = self.make_obstacles()
        # set any world properties
        world.dim_c = 2
        num_agents = 3
        num_landmarks = 3 + self.obstacles.shape[0]
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.1
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            if i < 3:
                landmark.name = 'landmark %d' % i
                landmark.collide = False
                landmark.movable = False
            else:
                landmark.name = 'obstacle %d' % i
                landmark.collide = True
                landmark.movable = False
                landmark.size = 0.04
        # create obstacles
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world, flip=False):
        # set properties for agents
        for i, agent in enumerate(world.agents):
            if i == 0:
                agent.color = np.array([0.35, 0.35, 0.85])
            elif i == 1:
                agent.color = np.array([0.85, 0.35, 0.35])
            elif i == 2:
                agent.color = np.array([0.35, 0.85, 0.35])

        # set properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            if i < 3:
                landmark.color = np.array([0.25, 0.25, 0.25])
            else:
                landmark.color = np.array([0.80, 0.80, 0.80])

        # set random initial states + obstacles
        for i, agent in enumerate(world.agents):
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            if True:
                if i == 0: # start anywhere
                    agent.state.p_pos = np.random.uniform(-1, 1, world.dim_p)
                elif i == 1: # start in LL room
                    agent.state.p_pos = np.array([np.random.uniform(0, -1/8), np.random.uniform(-1, 1/8)])
                elif i == 2:
                    if np.random.random() < 0.5: # start in UR room
                        agent.state.p_pos = np.array([np.random.uniform(-1/8, 1), np.random.uniform(3/4, 1)])
                    else: # start in UL room
                        agent.state.p_pos = np.array([np.random.uniform(0, -1/8), np.random.uniform(1/8, 1)])
                else:
                    raise ValueError()
            else:
                agent.state.p_pos = np.random.uniform(-1, 1, world.dim_p)
        for i, landmark in enumerate(world.landmarks):
            if i == 0:
                landmark.state.p_pos = np.array([-3/8, 7/8])
            elif i == 1:
                landmark.state.p_pos = np.array([-7/8, -7/8])
            elif i == 2:
                landmark.state.p_pos = np.array([13/16, -13/16])
            else:
                landmark.state.p_pos = self.obstacles[i-3]
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                     for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark,
        # penalized for collisions
        rew = 0
        for l in world.landmarks[:3]:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                     for a in world.agents]
            rew -= min(dists)
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
            for l in world.landmarks[3:]:
                if self.is_collision(l, agent):
                    rew -= 0.1
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks[:3]:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        entity_pos = sorted(
            entity_pos, key=lambda pos: np.arctan2(pos[1], pos[0]))
        other_pos = sorted(
            other_pos, key=lambda pos: np.arctan2(pos[1], pos[0]))
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)

    def make_obstacles(self):
        def wall_to_obstacles(x_start, x_stop, y_start, y_stop):
            assert x_stop >= x_start
            assert y_stop >= y_start

            x_dist = x_stop-x_start
            y_dist = y_stop-y_start
            dist = np.sqrt(x_dist**2 + y_dist**2)
            
            num_obstacles = dist/ 0.1
            x_coords = np.linspace(x_start, x_stop, num_obstacles+1)            
            y_coords = np.linspace(y_start, y_stop, num_obstacles+1)

            return list(zip(x_coords, y_coords))

        # set of walls
        walls = [
            [-1/8, -1/8, -1/4, 1], 
            [-1/8, -1/8, -1, -3/4],
            [-1/8, 1/4, 0, 0],
            [3/4, 1, 0, 0],
            [-9/16, -9/16, -1, -3/8],
            [-1, -5/8, 1/8, 1/8],
            [-9/16, -1/8, 5/8, 5/8],
            [7/16, 11/16, -11/16, -7/16]
            ]

        # create obstacles
        obstacles = []
        for wall in walls:
            obstacles = obstacles + wall_to_obstacles(*wall)

        return np.array(obstacles)

    def post_step_callback(self, world):
        pass
