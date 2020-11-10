import os

# -------- scenario settings -------- #
fix_vehicle_pos = False
vehicles_pos_file = os.path.join(os.path.dirname(__file__), 'saves/positions/12_vehicle_positions.npz')

lateral_pos_limitation = (11.4, 17.7)
action_holding_time = 1./15.    ## 20hz

# -------- sensor settings ----------- #
collision_sensor_config = {'data_type': 'sensor.other.collision','attach_to': None}
invasion_sensor_config = {'data_type': 'sensor.other.lane_invasion', 'attach_to': None}

farest_vehicle_consider = 50

# --------- action settings ------- #
## action dict -- {idx: [throttle, steer, brake]}
# actions = {0:[0., 0., 0.],      # Coast
#            1:[0., -0.5, 0.],    # Turn Left
#            2:[0., 0.5, 0.],     # Turn Right
#            3:[0.7, 0., 0.],      # Forward
#            4:[0., 0., 0.5],     # Brake
#            5:[0.5, -0.5, 0.],    # Bear Left & accelerate
#            6:[0.5, 0.5, 0.],     # Bear Right & accelerate
#            7:[0., -0.5, 0.5],   # Bear Left & decelerate
#            8:[0., 0.5, 0.5]}    # Bear Right & decelerate

actions = {0:[0.5, 0.5, 0.],
           1:[0.5, -0.5, 0.],
           2:[0.5, 0.1, 0.],
           3:[0.5, -0.1, 0.],
           4:[0.5, 0., 0.]}    # small acc