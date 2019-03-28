import numpy as np
from multiagent.core import World, Agent, Landmark, Wall
from multiagent.scenario import BaseScenario

walls = [
    ['V', -8, (-3.2, 3.2), 0.05],
    ['V', 8, (-3.2, 3.2), 0.05],
    ['H', -3.2, (-8, 8), 0.05],
    ['H', 3.2, (-8, 8), 0.05]]


class Scenario(BaseScenario):

    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_good_agents = 1
        num_adversaries = 4
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 0
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.65 if agent.adversary else 0.65
            agent.accel = 1.0 if agent.adversary else 2.5
            #agent.accel = 20.0 if agent.adversary else 25.0
            # agent.max_speed = 1.0 if agent.adversary else 1.5
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
        # add walls
        world.walls = [Wall(*w) for w in walls]
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world, flip=False, start_poses=None):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array(
                [0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        if start_poses is None:
            for agent in world.agents:
                agent.state.p_pos = np.array([np.random.uniform(-8, +8), np.random.uniform(-3.2, +3.2)])
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)
        else:
            for i, agent in enumerate(world.agents):
                agent.state.p_pos = np.array(start_poses[i])
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(
                    -1.5, +1.5, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

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
        if (ent_pos[perp_dim] > wall.axis_pos + agent.size or
                ent_pos[perp_dim] < wall.axis_pos - agent.size):
            return False
        return True

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.adversary_reward(
            agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        shape = True
        adversaries = self.adversaries(world)
        # reward can optionally be shaped (increased reward for increased
        # distance from adversary)
        if shape:
            for adv in adversaries:
                rew += 0.1 * \
                    np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 10
            for w in world.walls:
                if self.is_wall_collision(w, agent):
                    rew -= 0.25

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = True
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        # reward can optionally be shaped (decreased reward for increased
        # distance from agents)
        if shape:
            for adv in adversaries:
                rew -= 0.1 * \
                    min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos)))
                         for a in agents])
        if agent.collide:
            for ag in agents:
                if self.is_collision(ag, agent):
                    rew += 10
            for w in world.walls:
                if self.is_wall_collision(w, agent):
                    rew -= 0.25
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)
