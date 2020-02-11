"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.
Description :
a query tool for sensor data
Author：Team Li
"""
import numpy as np
import sys, glob, os
from utilities.logging import logger


try:
    sys.path.append('F:\my_project\driving-desicion-in-carla\dist/carla-0.9.4-py3.7-win-amd64.egg')
    import carla
    from carla import ColorConverter as cc
except:
    raise ImportError('Please check your carla file')

# def add_sensor_to_vehicle(vehicle, sensor):
#     """add a sensor to a vehicle
#     Args:
#         vehicle: a actor obj represents a vehicle
#         sensor: a actor_blueprint represents the sensor
#     """

class bgr_camera(object):
    """additional sensor class for convenient operation
    the raw sensor object
    Example:
        world = client.get_world()
        actor_list = world.get_actors()
        vehicles = list(actor_list.filter('vehicle*'))
        camera_config = {'data_type': 'sensor.camera.rgb', 'image_size_x': 418,
                        'image_size_y': 278, 'fov': 110, 'sensor_tick': 0.02,
                        'transform': carla.Transform(carla.Location(x=0.8, z=3)),
                        'attach_to':vehicles[0]}
        camera = bgr_camera(world, camera_config)
        while True:
            bgr = camera.get()
            ## todo
    """
    def __init__(self, world, sensor_config):
        assert sensor_config['data_type'] == 'sensor.camera.rgb'
        blueprint = world.get_blueprint_library().find(sensor_config['data_type'])
        # Modify the attributes of the blueprint to set image resolution and field of view.
        blueprint.set_attribute('image_size_x', str(sensor_config['image_size_x']))
        blueprint.set_attribute('image_size_y', str(sensor_config['image_size_y']))
        blueprint.set_attribute('fov', str(sensor_config['fov']))
        # Set the time in seconds between sensor captures
        # print(str(sensor_config['sensor_tick']))
        blueprint.set_attribute('sensor_tick', str(sensor_config['sensor_tick']))
        # Provide the position of the sensor relative to the vehicle.
        transform = sensor_config['transform']
        # Tell the world to spawn the sensor, don't forget to attach it to your vehicle actor.
        self.sensor = world.spawn_actor(blueprint, transform, attach_to=sensor_config['attach_to'])

        self.bgr = np.zeros(shape = (int(sensor_config['image_size_y']), int(sensor_config['image_size_x']), 3))
        # self.gen = self.__data_gen()
        # self.gen.send(None)

        self.sensor.listen(lambda image: self.__send_data(image))


    # def __data_gen(self):
    #     """a generator"""
    #     while True:
    #         self.bgr = yield ''
    #         pass


    def __send_data(self, image):
        """a thread would be call in a frequency"""
        raw = bytes(image.raw_data)
        raw = np.frombuffer(raw, np.uint8)
        bgra = np.reshape(raw, newshape=[image.height, image.width, -1])
        bgr = bgra[:, :, :3]
        self.bgr = bgr
        pass


    def get(self):
        """return a bgr img"""
        return self.bgr


class depth_camera(object):
    """additional sensor class for convenient operation
    the raw sensor object
    Example:
        world = client.get_world()
        actor_list = world.get_actors()
        vehicles = list(actor_list.filter('vehicle*'))
        camera_config = {'data_type': 'sensor.camera.depth', 'image_size_x': 418,
                        'image_size_y': 278, 'fov': 110, 'sensor_tick': 0.02,
                        'transform': carla.Transform(carla.Location(x=0.8, z=3)),
                        'attach_to':vehicles[0]}
        camera = depth_camera(world, camera_config)
        while True:
            bgr = camera.get()
            ## todo
    """
    def __init__(self, world, sensor_config):
        assert sensor_config['data_type'] == 'sensor.camera.depth'
        blueprint = world.get_blueprint_library().find(sensor_config['data_type'])
        # Modify the attributes of the blueprint to set image resolution and field of view.
        blueprint.set_attribute('image_size_x', str(sensor_config['image_size_x']))
        blueprint.set_attribute('image_size_y', str(sensor_config['image_size_y']))
        blueprint.set_attribute('fov', str(sensor_config['fov']))
        # Set the time in seconds between sensor captures
        # print(str(sensor_config['sensor_tick']))
        blueprint.set_attribute('sensor_tick', str(sensor_config['sensor_tick']))
        # Provide the position of the sensor relative to the vehicle.
        transform = sensor_config['transform']
        # Tell the world to spawn the sensor, don't forget to attach it to your vehicle actor.
        self.sensor = world.spawn_actor(blueprint, transform, attach_to=sensor_config['attach_to'])

        self.depth = np.zeros(shape = (int(sensor_config['image_size_y']), int(sensor_config['image_size_x']), 1))
        # self.gen = self.__data_gen()
        # self.gen.send(None)

        self.sensor.listen(lambda image: self.__send_data(image))

    # def __data_gen(self):
    #     while True:
    #         self.depth = yield ''
    #         pass

    def __send_data(self, image):
        image.convert(cc.LogarithmicDepth)
        depth = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        depth = np.reshape(depth, (image.height, image.width, 4))
        depth = depth[:, :, 0]
        depth = np.expand_dims(depth, axis=-1)
        self.depth = depth
        pass


    def get(self):
        """return a depth image"""
        return self.depth


class semantic_camera(object):
    """additional sensor class for convenient operation
    the raw sensor object
    Example:
        world = client.get_world()
        actor_list = world.get_actors()
        vehicles = list(actor_list.filter('vehicle*'))
        camera_config = {'data_type': 'sensor.camera.semantic_segmentation', 'image_size_x': 418,
                        'image_size_y': 278, 'fov': 110, 'sensor_tick': 0.02,
                        'transform': carla.Transform(carla.Location(x=0.8, z=3)),
                        'attach_to':vehicles[0]}
        camera = semantic_camera(world, camera_config)
        while True:
            bgr = camera.get()
            ## todo
    """
    def __init__(self, world, sensor_config):
        assert sensor_config['data_type'] == 'sensor.camera.semantic_segmentation'
        blueprint = world.get_blueprint_library().find(sensor_config['data_type'])
        # Modify the attributes of the blueprint to set image resolution and field of view.
        blueprint.set_attribute('image_size_x', str(sensor_config['image_size_x']))
        blueprint.set_attribute('image_size_y', str(sensor_config['image_size_y']))
        blueprint.set_attribute('fov', str(sensor_config['fov']))
        # Set the time in seconds between sensor captures
        # print(str(sensor_config['sensor_tick']))
        blueprint.set_attribute('sensor_tick', str(sensor_config['sensor_tick']))
        # Provide the position of the sensor relative to the vehicle.
        transform = sensor_config['transform']
        # Tell the world to spawn the sensor, don't forget to attach it to your vehicle actor.
        self.sensor = world.spawn_actor(blueprint, transform, attach_to=sensor_config['attach_to'])

        self.semantic = np.zeros(shape = (int(sensor_config['image_size_y']), int(sensor_config['image_size_x']), 3))
        # self.gen = self.__data_gen()
        # self.gen.send(None)

        self.sensor.listen(lambda image: self.__send_data(image))

    # def __data_gen(self):
    #     while True:
    #         self.semantic = yield ''
    #         pass

    def __send_data(self, image):
        image.convert(cc.CityScapesPalette)
        semantic = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        semantic = np.reshape(semantic, (image.height, image.width, 4))
        semantic = semantic[:, :, :3]
        # self.gen.send(semantic)
        self.semantic = semantic
        pass


    def get(self):
        """return a semantic image"""
        return self.semantic


class collision_query(object):
    """a sensor offer user to query whether collision
    Example:
        collision_sensor_config = {'data_type': 'sensor.other.collision','attach_to': vehicles[0]}
        collision_q = collision_query(world, collision_sensor_config)
    """
    def __init__(self, world, sensor_config):
        """init a collision query sensor"""
        assert sensor_config['data_type'] == 'sensor.other.collision'
        blueprint = world.get_blueprint_library().find(sensor_config['data_type'])

        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=sensor_config['attach_to'])
        self.sensor.listen(lambda event: self.__on_collision(event))

        self.whether_collision = False


    def __on_collision(self, event):
        """ would be call when ego vehicle collision"""
        #if event.other_actor.type_id != 'static.road': ## avoid the collision of static road
        self.whether_collision = True
        #logger.info('collision obj:%s'%str(event.other_actor))


    def get(self):
        """get the flag indicating whether ego vehicle collision"""
        return self.whether_collision


    def clear(self):
        """"""
        self.whether_collision = False

class lane_invasion_query(object):
    """a sensor query whether lane invasion"""
    def __init__(self, world, sensor_config):
        """init a lan invasion query sensor"""
        assert sensor_config['data_type'] == 'sensor.other.lane_invasion'
        blueprint = world.get_blueprint_library().find(sensor_config['data_type'])

        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=sensor_config['attach_to'])
        self.sensor.listen(lambda event: self.__on_invasion(event))

        self.lane_invasion = False


    def __on_invasion(self, event):
        """ would be call when ego vehicle collision"""
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        logger.info(str(text))
        if text[0] in ["'SolidBroken'", "'Solid'"]:
            self.lane_invasion = True
            # text = 'Crossed line %s' % ' and '.join(text)


    def get(self):
        """get the flag indicating whether ego vehicle collision"""
        return self.lane_invasion


    def clear(self):
        """"""
        self.lane_invasion = False