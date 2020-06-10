from gym.envs.registration import register
from . import env_v1
register(
    id='ObstacleAvoidance-v0',
    entry_point='environments.carla_enviroments.env_v1_ObstacleAvoidance.env_v1:ObstacleAvoidanceScenario',
    trials = 10,
    reward_threshold = 100.,
)

register(
    id='ObstacleAvoidance-v1',
    entry_point='environments.carla_enviroments.env_v1_ObstacleAvoidance.env_v1_two_eyes:ObstacleAvoidanceScenarioTwoEyes',
    trials = 10,
    reward_threshold = 100.,
)

register(
    id='ObstacleAvoidance-v2',
    entry_point='environments.carla_enviroments.env_v1_ObstacleAvoidance.env_v1_dynamic:ObstacleAvoidanceScenarioDynamic',
    trials = 10,
    reward_threshold = 100.,
)