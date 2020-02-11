import random
from utilities.logging import logger

def try_spawn_random_vehicle_at(world, transform, role_name, autopilot=False, vehicle_type=None):
    if not vehicle_type:
        blueprints = world.get_blueprint_library().filter('vehicle.nissan.micra')
    else:
        blueprints = world.get_blueprint_library().filter(vehicle_type)

    blueprint = random.choice(blueprints)
    if blueprint.has_attribute('color'):
        color = random.choice(blueprint.get_attribute('color').recommended_values)
        blueprint.set_attribute('color', color)

    blueprint.set_attribute('role_name', role_name)
    vehicle = world.try_spawn_actor(blueprint, transform)

    if (vehicle is not None) and (autopilot):
        vehicle.set_autopilot(True)
        # logger.info('spawned a autopilot %r at %s' % (vehicle.type_id, transform.location))
    elif (vehicle is not None) and (not autopilot):
        vehicle.set_autopilot(False)
        # logger.info('spawned a egopilot %r at %s' % (vehicle.type_id, transform.location))
    return vehicle


def destroy_all_actors(world):
    """destroy all actors"""
    actor_list = world.get_actors()
    vehicles = list(actor_list.filter('vehicle*'))
    for vehicle in vehicles:
        vehicle.destroy()
    # logger.info('Destroy all vehicles...')

    sensors = list(actor_list.filter('sensor*'))
    for sensor in sensors:
        sensor.destroy()
    # logger.info('Destroy all sensors...')