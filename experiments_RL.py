#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 11:43:56 2019

@author: juanjosealcaraz

this script iterates over different routes for a given configuration of depth and budget
"""

from route_environment import RouteEnvironment, IncidenceNormalGenerator, NormalGenerator
from rollout_agent import RolloutAgent, CostEstimator
from statistics import mean
import concurrent.futures as cf

ROUTES = [1,2,3,4,5,6]
DEPTH = 2
BUDGET = 10
RUNS = 200
PROCESSORS = 4


def simulate_route(route, depth = DEPTH, budget = BUDGET, runs = RUNS, seed_a = 1, seed_b = 1000):
    RL_samples = []
    baseline_samples = []
    RL_times = []

    for r in range(runs):
        
        print('route '+str(route)+'; run: '+str(r+1))
        
        seed_a +=1
        seed_b +=1
    
        generator_a = IncidenceNormalGenerator(seed = seed_a)
        generator_b = NormalGenerator(seed = seed_b)

        # # high uncertainty scenario
        # generator_a = IncidenceNormalGenerator(seed = seed_a, max_speed = 120, min_speed = 40, speed_dev = 10, stime_dev = 0.2)
        # generator_b = NormalGenerator(seed = seed_b, max_speed = 120, min_speed = 40, speed_dev = 10, stime_dev = 0.2)
        
        # # small uncertainty scenario
        # generator_a = NormalGenerator(seed = seed_a, max_speed = 90, min_speed = 70, speed_dev = 1, stime_dev = 0.01)
        # generator_b = NormalGenerator(seed = seed_b, max_speed = 90, min_speed = 70, speed_dev = 1, stime_dev = 0.01)

        route_sim = RouteEnvironment(route, generator_a, generator_b)
            
        weight_vector = [10.0]*route_sim.flags
        weight_vector[0] = 0.0
        n_nodes = len(route_sim.path)
        
        for i in range(n_nodes):
            if route_sim.demands[i] > 0.0:
                weight_vector.append(route_sim.demands[i])
            else:
                weight_vector.append(0.0)
            
        ce = CostEstimator(weight_vector)
        cost_criteria = ce.monetary_cost
        agent = RolloutAgent(route_sim, depth, budget, cost_criteria)
        
        trace, times = agent.run()
           
        RL_samples.append(ce.monetary_cost([trace]))
        RL_times.extend(times)
            
        state, _, trace = route_sim.reset()
        trace = route_sim.rollout(state, trace)
        
        baseline_samples.append(ce.monetary_cost([trace]))
    
    print('mean RL: ' +str(mean(RL_samples)))
    print('mean baseline: '+str(mean(baseline_samples)))
    
    return {'RL_samples': RL_samples,
            'baseline_samples': baseline_samples,
            'RL_times': RL_times}
    
if __name__== "__main__":
    import pickle

    routes = ROUTES
    
    RL_results = {}
    baseline_results = {}
    RL_times = {}

    with cf.ProcessPoolExecutor(PROCESSORS) as E:
        results = E.map(simulate_route, routes)
 
    for i, r in enumerate(results):
        RL_results[routes[i]] = r['RL_samples']
        baseline_results[routes[i]] = r['baseline_samples']
        RL_times[routes[i]] = r['RL_times']

    with open('./results/RL_results.pickle', 'wb') as f:
        pickle.dump(RL_results, f)
    
    with open('./results/baseline_results.pickle', 'wb') as f:
        pickle.dump(baseline_results, f) 

    with open('./results/RL_times.pickle', 'wb') as f:
        pickle.dump(RL_times, f) 