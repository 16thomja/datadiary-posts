import time
import os
import numpy as np
import pandas as pd
import json

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

from beamngpy import BeamNGpy, Scenario, Vehicle, Road
from beamngpy.sensors import State

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--log_dir', type=str, required=True)
args = parser.parse_args()

if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)
    if not os.path.exists(args.log_dir + '/physics_logs'):
        os.mkdir(args.log_dir + '/physics_logs')

# spatial constants
STRAIGHT_LENGTH = 100
CURVE_RADIUS = 40
CURVE_RESOLUTION = 20
ROAD_WIDTH = 7
ORIGIN = (500, 0, 0) # offset to avoid map obstacles

def controller(w, Y_err, V_y, omega_err, V_err):
    weighted_sum = (
        w[0] + 
        w[6] * (np.tanh(w[1] * Y_err)) + 
        w[2] * Y_err + 
        w[3] * V_y + 
        w[4] * omega_err + 
        w[7] * (np.tanh(w[5] * omega_err))
    )
    steer = np.tanh(w[8] * weighted_sum)

    weighted_sum = (
        w[9] + 
        w[10] * abs(omega_err) + 
        w[11] * V_err + 
        w[13] * (np.tanh(w[12] * V_err))
    )
    throttle = np.tanh(w[14] * weighted_sum)

    return steer, throttle

def timestep_reward(row):
    return (
        row[0]**2 + 
        0.8 * row[1]**2 + 
        row[2]**2 + 
        0.7 * (row[3]**2 + row[4]**2)
    )

def create_scenario():
    scenario = Scenario('tech_ground', 'u_turn')

    vehicle = Vehicle('ego_vehicle', model='etk800', color='darkred', license='DATADIARY')
    scenario.add_vehicle(
        vehicle,
        pos=(ORIGIN[0] - CURVE_RADIUS, ORIGIN[1], ORIGIN[2]),
        rot_quat=(0, 0, 1, 0),
        cling=True
    )

    road = Road(
        material='track_editor_E_center',
        rid='main_road',
        interpolate=False,
        looped=False,
    )

    road.nodes.append((ORIGIN[0] - CURVE_RADIUS, ORIGIN[1] - 500, ORIGIN[2], ROAD_WIDTH))

    angle_increment = np.pi / (CURVE_RESOLUTION - 1)
    for i in range(CURVE_RESOLUTION):
        theta = i * angle_increment
        x = float(ORIGIN[0] - CURVE_RADIUS * np.cos(theta))
        y = float(ORIGIN[1] + STRAIGHT_LENGTH + CURVE_RADIUS * np.sin(theta))
        road.nodes.append((x, y, ORIGIN[2], ROAD_WIDTH))

    road.nodes.append((ORIGIN[0] + CURVE_RADIUS, ORIGIN[1] - 500, ORIGIN[2], ROAD_WIDTH))

    scenario.add_road(road)

    return scenario, vehicle

#----------------------------------------------------------------------------

# simulation params
V_target = 16
T = 25
dt = 0.25

bng = BeamNGpy(
    host='localhost',
    port=64256,
    home='/home/jared/Downloads/BeamNG.tech.v0.32.2.0',
    user='/home/jared/Projects/BeamNG_tech/beamngpy_output_files'
)
bng.open()
bng.settings.set_deterministic(60) # 60Hz temporal resolution

scenario, vehicle = create_scenario()
scenario.make(bng)

"""
trial = args.old_points # global trial counter
"""
trial = 0

def simulation_run(**kwargs):
    global bng
    global scenario
    global vehicle
    global trial
    print(f'trial: {trial}')

    weights = np.array([kwargs.get(f'w{i}', 0) for i in range(15)]).astype(np.float64)

    # reload sim every 10 trials to combat instability
    if trial % 10 == 0 and trial > 0:
        print("\033[91m!!! RELOADING SIM !!!\033[0m")
        bng.close()
        time.sleep(5)
        bng.open()
        bng.settings.set_deterministic(60) # 60Hz temporal resolution

        scenario, vehicle = create_scenario()
        scenario.make(bng)

    physics_rows = []
    rewards = []
    for episode in range(10):
        # start heading at 140 deg CCW from +x and decrement by 10 each episode
        angle_degrees = 140 - (10 * episode)
        angle_radians = np.deg2rad(angle_degrees) - np.pi / 2
        quaternion = (0, 0, np.cos(angle_radians / 2), np.sin(angle_radians / 2))

        bng.scenario.load(scenario)
        vehicle.teleport(pos=(ORIGIN[0] - CURVE_RADIUS, ORIGIN[1], ORIGIN[2]), rot_quat=quaternion)
        time.sleep(1)
        bng.scenario.start()

        tether = list(ORIGIN)

        values = []
        t = 0
        while t <= T:
            vehicle.sensors.poll()
            data = vehicle.sensors.data['state']
            
            position = np.array(data['pos'])[:2]
            velocity = np.array(data['vel'])[:2]
            speed = np.linalg.norm(velocity)
            direction = np.array(data['dir'])[:2]
            direction = direction / np.linalg.norm(direction)

            if position[1] <= STRAIGHT_LENGTH:
                tether[1] = position[1]
            else:
                tether[1] = STRAIGHT_LENGTH

            # tether-vehicle vector
            v_tv = np.array([position[0] - tether[0], position[1] - tether[1]])
            distance = np.linalg.norm(v_tv)
            # +pi/2 CW then normalize
            direction_target = np.array([v_tv[1], -v_tv[0]]) / distance

            # calc omega as arccos of projection of direction onto direction_target, 
            # add sign with cross product (CCW + or CW -)
            omega = np.arccos(np.dot(direction, direction_target))
            cross_product = direction_target[0] * direction[1] - direction_target[1] * direction[0]
            omega_err = omega * (cross_product / (abs(cross_product) + 1e-8))

            Y_err = distance - CURVE_RADIUS

            # calculate slip by projecting velocity onto direction perp
            V_y = velocity[0] * (-direction[1]) + velocity[1] * direction[0]

            V_err = V_target - speed

            steer, throttle = controller(weights, Y_err, V_y, omega_err, V_err)
            
            vehicle.control(throttle=throttle, brake=-throttle, steering=steer)

            values.append([Y_err, V_err, omega_err, steer, throttle])

            physics_rows.append({
                'episode': episode,
                't': t,
                'position': position,
                'velocity': velocity,
                'direction': direction,
                'steer': steer,
                'throttle': throttle
            })

            t += dt
            time.sleep(dt)

        values = np.array(values)

        # clip Y_err to CURVE_RADIUS to protect normalization
        #values[:, 0] = np.clip(values[:, 0], a_min=-CURVE_RADIUS, a_max=CURVE_RADIUS)
        # establish normalization bounds
        values_max = np.array([CURVE_RADIUS, np.max(np.abs(values[:,1])), np.pi, 1.0, 1.0])
        """
        # apply log normalization
        values_norm = np.sign(values) * np.log(1 + np.abs(values)) / (np.log(1 + values_max) + 1e-8)
        """
        # apply standard normalization
        values_norm = values / (values_max + 1e-8)

        # replace Y_err with tanh smoothing
        values_norm[:,0] = np.tanh((3 / CURVE_RADIUS) * values[:,0])

        reward = -np.sum(np.apply_along_axis(timestep_reward, axis=1, arr=values_norm))

        print(f'\tepisode {episode} reward: {reward}')
        
        rewards.append(reward)

    df = pd.DataFrame(physics_rows)
    df.to_pickle(f'{args.log_dir}/physics_logs/trial_{trial}.pkldf')

    trial += 1

    return np.mean(rewards)

#----------------------------------------------------------------------------

optimizer = BayesianOptimization(
    f=simulation_run,
    pbounds={f'w{i}': (0,1) for i in range(15)},
    verbose=0,
    random_state=42
)

logger = JSONLogger(path=f"{args.log_dir}/logs.log")
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

"""
optimizer.maximize(
    init_points=max(30 - args.old_points, 0),
    n_iter=100 - max(args.old_points - 30, 0)
)
"""

optimizer.maximize(
    init_points=30,
    n_iter=100
)

print('\nBest result:', optimizer.max)
