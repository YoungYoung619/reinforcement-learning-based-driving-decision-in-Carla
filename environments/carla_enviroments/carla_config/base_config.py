import socket
hostname = socket.gethostname()

egg_config = {'DESKTOP-99MRARG':'C:\my_project\CARLA_0.9.5\PythonAPI\carla\dist\carla-0.9.5-py3.7-win-amd64.egg',
              'LVANYANG-PC1':'E:\game\CARLA_0.9.4\PythonAPI\carla-0.9.4-py3.7-win-amd64.egg'}

egg_file = egg_config[hostname]

world_ops_logger = False

no_render_mode = False
synchronous_mode = True