import os

# -------- scenario settings -------- #
fix_vehicle_pos = False
vehicles_pos_file = os.path.join(os.path.dirname(__file__), 'saves/positions/6_vehicle_positions.npz')
lateral_pos_limitation = (203., 208.)
action_holding_time = 0.5

# -------- sensor settings ----------- #
collision_sensor_config = {'data_type': 'sensor.other.collision','attach_to': None}
invasion_sensor_config = {'data_type': 'sensor.other.lane_invasion', 'attach_to': None}

# --------- action settings ------- #
## action dict -- {idx: [throttle, steer, brake]}
actions = {0:[0., 0., 0.],      # Coast
           1:[0., -0.5, 0.],    # Turn Left
           2:[0., 0.5, 0.],     # Turn Right
           3:[1., 0., 0.],      # Forward
           4:[0., 0., 0.5],     # Brake
           5:[1., -0.5, 0.],    # Bear Left & accelerate
           6:[1., 0.5, 0.],     # Bear Right & accelerate
           7:[0., -0.5, 0.5],   # Bear Left & decelerate
           8:[0., 0.5, 0.5]}    # Bear Right & decelerate