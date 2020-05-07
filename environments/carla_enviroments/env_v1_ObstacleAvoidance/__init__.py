from gym.envs.registration import register
from . import env_v1
register(
    id='ObstacleAvoidance-v0',
    entry_point='environments.carla_enviroments.env_v1_ObstacleAvoidance.env_v1:ObstacleAvoidanceScenario',
    trials = 10,
    reward_threshold = 1000.,
)