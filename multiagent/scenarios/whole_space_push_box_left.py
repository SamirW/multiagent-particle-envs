import numpy as np
from multiagent.core import World, Agent, Landmark, Wall
from multiagent.scenario import BaseScenario

walls = [
    ['V', -8, (-3.2, 3.2), 0.05],
    ['V', 8, (-3.2, 3.2), 0.05],
    ['H', -3.2, (-8, 8), 0.05],
    ['H', 3.2, (-8, 8), 0.05],
    ['V', 0, (-3.2, 3.2), 0.05]]
    # ['V', -1, (-3.2, 3.2), 0.05],
    # ['V', 1, (-3.2, 3.2), 0.05]]

class Scenario(BaseScenario):

    def make_world(self, mode=0):
        """
        mode0:
        - Single goal start

        mode1:
        - Random goal start
        """
        assert mode >= 0
        assert mode <= 1
        self.mode = mode

        world = World()

        # add agents
        world.agents = [Agent() for i in range(2)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.75
            agent.i = i

        # add boxes
        n_box = 1  # two boxes
        self.boxes = [Landmark() for _ in range(n_box)]
        for i, box in enumerate(self.boxes):
            box.name = 'box %d' % i
            box.collide = True
            box.movable = True
            box.size = 1.25
            box.initial_mass = 2.
            box.index = i
            world.landmarks.append(box)

        # add targets
        self.targets = [Landmark() for _ in range(n_box)]
        for i, target in enumerate(self.targets):
            target.name = 'target %d' % i
            target.collide = False
            target.movable = False
            target.size = 0.15
            target.index = i
            world.landmarks.append(target)

        # add walls
        world.walls = [Wall(*w) for w in walls]

        # make initial conditions
        self.reset_world(world)

        self.timestep = 0.

        return world

    def reset_world(self, world, flip=False, start_poses=None, goal_poses=None, start_vels=None):
        # random properties for agents
        flip = np.random.randint(2)
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.85, 0.35, 0.35])

            if start_poses is None:
                if self.mode == 0:
                    if i == flip:
                        agent.state.p_pos = np.array([np.random.uniform(-4.25, -3.5), np.random.uniform(2, 2.25)])
                    else:
                        agent.state.p_pos = np.array([np.random.uniform(-4.25, -3.5), np.random.uniform(-2.25, -2)])
                else:
                    pass
            else:
                for i, agent in enumerate(world.agents):
                    agent.state.p_pos = np.array(start_poses[i])

            if start_vels is None:
                agent.state.p_vel = np.zeros(world.dim_p)
            else:
                agent.state.p_vel = np.array(start_vels[i])
                
            agent.state.c = np.zeros(world.dim_c)

        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
            landmark.state.p_vel = np.zeros(world.dim_p)

            if "box" in landmark.name:
                landmark.state.p_pos = np.array([-4, 0.])
            elif "target" in landmark.name:
                if goal_poses is None:
                    landmark.state.p_pos = np.array([-7, 0.])
                else:
                    landmark.state.p_pos = np.array(goal_poses[landmark.index])
            else:
                raise ValueError()

        self.timestep = 0.

    def reward(self, agent, world):
        rew = 0

        for box, target in zip(world.boxes, world.targets):
            # Move box to target (per box target pair)
            dist = np.sum(np.square(box.state.p_pos - target.state.p_pos))
            rew -= dist

        return rew

    def observation(self, agent, world):
        if agent.i == 0:
            self.timestep += 1

        # get positions of all entities
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        assert len(entity_pos) == len(self.boxes) + len(self.targets)

        # Add other agent position
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] +
        # entity_pos + other_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos)

    def post_step_callback(self):
        pass
