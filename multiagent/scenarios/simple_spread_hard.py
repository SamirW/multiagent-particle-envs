import numpy as np
from multiagent.core import World, Agent, Landmark, Wall
from multiagent.scenario import BaseScenario

walls = [
    ['V', -1/8, (-1/4, 1), 0.05], 
    ['V', -1/8, (-1, -3/4), 0.05],
    ['H', 0, (-1/8, 1/4), 0.05],
    ['H', 0, (3/4, 1), 0.05],
    ['V', -9/16, (-1, -3/8), 0.05],
    ['H', 1/8, (-9/16, -1/8), 0.05],
    ['H', 5/8, (-1, -5/8), 0.05],
    ['H', -1, (-1, 1), 0.05],
    ['V', -1, (-1, 1), 0.05],
    ['H', 1, (-1, 1), 0.05],
    ['V', 1, (-1, 1), 0.05]]

class Scenario(BaseScenario):

    def make_world(self):
        world = World()
        # set any world properties
        world.dim_c = 2
        num_agents = 3
        num_landmarks = 3
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.1
            agent.max_speed = 0.85
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
        # add walls
        world.walls = [Wall(*w) for w in walls]
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
            landmark.color = np.array([0.25, 0.25, 0.25])

        # set random initial states
        for i, agent in enumerate(world.agents):
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            if flip: # everyone starts everywhere
                agent.state.p_pos = np.random.uniform(-1, 1, world.dim_p)
            else:
                if i == 0: # start anywhere
                    agent.state.p_pos = np.random.uniform(-1, 1, world.dim_p)
                elif i == 1: # start in LL room
                    agent.state.p_pos = np.array([np.random.uniform(-.95, -.1), np.random.uniform(-1, 1/8)])
                elif i == 2:
                    if np.random.random() < 0.5: # start in UR room
                        agent.state.p_pos = np.array([np.random.uniform(-1/8, 1), np.random.uniform(0, 1)])
                    else: # start in UL room
                        agent.state.p_pos = np.array([np.random.uniform(-1, -1/8), np.random.uniform(1/8, 1)])
                else:
                    raise ValueError()
        for i, landmark in enumerate(world.landmarks):
            if i == 0:
                landmark.state.p_pos = np.array([-3/4, 13/16])
            elif i == 1:
                landmark.state.p_pos = np.array([-25/32, -13/16])
            elif i == 2:
                landmark.state.p_pos = np.array([3/4, -3/4])
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

    def is_wall_collision(self, wall, agent):
        if agent.ghost and not wall.hard:
            return False  # ghost passes through soft walls
        if wall.orient == 'H':
            prll_dim = 0
            perp_dim = 1
        else:
            prll_dim = 1
            perp_dim = 0
        ent_pos = agent.state.p_pos
        if (ent_pos[prll_dim] < wall.endpoints[0] - agent.size or
            ent_pos[prll_dim] > wall.endpoints[1] + agent.size):
            return False  # agent is beyond endpoints of wall
        return True

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark,
        # penalized for collisions
        rew = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                     for a in world.agents]
            rew -= min(dists)
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
            for w in world.walls:
                if self.is_wall_collision(w, agent):
                    rew -= 0.25
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
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

    def post_step_callback(self, world):
        pass
