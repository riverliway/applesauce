# this file will run a specified number of simulations 
# and return the 'ticks' required to collect 95% of the apples on the orchard
from orchard import *
from simple_solver import make_simple_decision
from complex_solver import ComplexSolver

num_sims = 100
seed = 10
SIMULATION_TIMEOUT = 40000

def get_baseline_complex(height = 1600,
                width = 2000,
                num_picker_bots = 4,
                num_pusher_bots = 0,
                num_baskets = 4,
                threshold = 0.95,
                seed = 10):
    '''
    Runs a simulation for OrchardComplex2D and reports the baseline
    for a single set of initialization parameters.
    '''
    #set up the environment anew for each iteration
    environment = OrchardComplex2D(
                    width = width,
                    height = height,
                    num_picker_bots = num_picker_bots,
                    num_pusher_bots = num_pusher_bots,
                    num_baskets = num_baskets,
                    seed = seed) 

    apples = len([a for a in environment.apples if not a['collected']])
    starting_apples = len(environment.starting_apples)

    if starting_apples > 0:
        PERCENT_DONE = 1 - (apples / starting_apples)
    else:
        print('starting_apples cant be 0.')
        return
    
    solver = ComplexSolver(environment)

    while environment.time < SIMULATION_TIMEOUT and PERCENT_DONE <= threshold:
        #set up environment
        new_env = solver.make_decisions()
        apples = len([a for a in new_env.apples if not a['collected']])
        PERCENT_DONE = 1 - (apples / starting_apples)
        print(f'length of apples: {apples}')
        print(f'length of starting apples: {starting_apples}')

    print(f'Time to collect {PERCENT_DONE * 100}% of apples: {environment.time} steps ({environment.time / 10} seconds).')

    return apples, starting_apples, environment.time

get_baseline_complex()

simulation_results = []

for i in range(num_sims):
    apples, starting_apples, ticks = get_baseline_complex()
    simulation_results.append((apples, starting_apples, ticks))

avg_collection_time = simulation_results[:][2].mean()

print(f'Average collection time from {num_sims} simulations: {avg_collection_time}')