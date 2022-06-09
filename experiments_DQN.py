#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2021
"""

from route_environment_q import RouteEnvironmentQlearning, IncidenceNormalGenerator, NormalGenerator
from rollout_agent import CostEstimator
from ddqn_keras import DDQNAgent
from statistics import mean
import concurrent.futures as cf
import numpy as np
import pickle

ROUTES = [1, 2, 3, 4, 5, 6]
TRAIN_RUNS = 50000
PROCESSORS = 4
RUNS = 200
ACTIONS = 11
DEBUG = False

# alfa and epsilon decay rate configurations for each route
settings = [(1e-05, 0.995),
            (5e-05, 0.995),
            (5e-05, 0.99),
            (5e-05, 0.995),
            (1e-05, 0.995),
            (1e-05, 0.995)]

TRAIN_CHECKPOINT = 10000
CONTINUE_THRESHOLD = -10.0

def train_route(route, runs = TRAIN_RUNS, seed_a = 1000, seed_b = 1):

    alpha = settings[route][0]
    epsilon_dec = settings[route][1]

    fname = 'ddqn_route_'+str(route)+'_'+str(alpha)+'_'+str(epsilon_dec)+'.h5'

    ddqn_agent = DDQNAgent(alpha=alpha, gamma=0.99, n_actions=ACTIONS, epsilon=1.0,
            batch_size=64, input_dims=14, mem_size=100000, epsilon_dec=epsilon_dec, fname=fname)

    ddqn_scores = []

    max_score = -40
    threshold_attained = False
       
    for r in range(runs):

        seed_a += 1
        seed_b += 1
    
        generator_a = NormalGenerator(seed = seed_a)
        generator_b = NormalGenerator(seed = seed_b)
        route_sim = RouteEnvironmentQlearning(route, generator_a, generator_b)

        score = 0
        state, actions, trace = route_sim.reset()
        state_vector = state.vector_state
        while not state.end:
            action = ddqn_agent.choose_action(state.vector_state, actions)
            state, actions, trace, = route_sim.step(state, action, trace)
            score += state.reward
            ddqn_agent.remember(state_vector, action, state.reward, state.vector_state, int(state.end))
            state_vector = state.vector_state
            ddqn_agent.learn()

        ddqn_scores.append(score)

        avg_score = np.mean(ddqn_scores[max(0, r-100):(r+1)])

        if avg_score > CONTINUE_THRESHOLD:
            threshold_attained = True
        
        if r % TRAIN_CHECKPOINT == 0 and r > 0:
            if not threshold_attained:
                return {'score' : avg_score,
                'alpha' : alpha,
                'epsilon_dec': epsilon_dec,
                'score_history': ddqn_scores}
            else:
                threshold_attained = False
        
        if avg_score > max_score:
            ddqn_agent.save_model()
            max_score = avg_score
        
    return {'score' : avg_score,
            'alpha' : alpha,
            'epsilon_dec': epsilon_dec,
            'score_history': ddqn_scores}


def simulate_route(route, runs = RUNS, seed_a = 1, seed_b = 1000):

    alpha = settings[route][0]
    epsilon_dec = settings[route][1]

    DQN_samples = []

    filename = 'sim_log_'+str(route)
    with open(filename,'w') as f:
        f.write('Partial Results\n')

    fname = 'ddqn_route_'+str(route)+'_'+str(alpha)+'_'+str(epsilon_dec)+'.h5'

    ddqn_agent = DDQNAgent(alpha=0.0005, gamma=0.99, n_actions=ACTIONS, epsilon=0.0,
            batch_size=64, input_dims=14, mem_size=100000, fname=fname)

    ddqn_agent.load_model()

    for r in range(runs):
        
        print('route '+str(route)+'; run: '+str(r+1))
        
        seed_a +=1
        seed_b +=1
    
        generator_a = IncidenceNormalGenerator(seed = seed_a)
        generator_b = NormalGenerator(seed = seed_b)
        route_sim = RouteEnvironmentQlearning(route, generator_a, generator_b)

        weight_vector = [10.0]*route_sim.flags
        weight_vector[0] = 0.0
        n_nodes = len(route_sim.path)
        
        for i in range(n_nodes):
            if route_sim.demands[i] > 0.0:
                weight_vector.append(route_sim.demands[i])
            else:
                weight_vector.append(0.0)
            
        ce = CostEstimator(weight_vector)

        state, actions, trace = route_sim.reset()
        while not state.end:
            action = ddqn_agent.choose_action(state.vector_state, actions)
            state, actions, trace = route_sim.step(state, action, trace)

        DQN_samples.append(ce.monetary_cost([trace]))
        
        print('trace: '+str(trace))
        print('monetary_cost: '+str(ce.monetary_cost([trace])))
    
    print('mean: ' +str(mean(DQN_samples)))
    
    return {'DQN_samples': DQN_samples} 
    
if __name__== "__main__":

    with cf.ProcessPoolExecutor(PROCESSORS) as E:
        results = E.map(train_route, ROUTES)
    
    score_history = {}

    for i,r in enumerate(results):
        score_history[ROUTES[i]] =  r['score_history']

    with open('DQN_train.pickle', 'wb') as f:
        pickle.dump(score_history, f)

    with cf.ProcessPoolExecutor(PROCESSORS) as E:
        results = E.map(simulate_route, ROUTES)

    RL_results = {}
    
    for i,r in enumerate(results):
        RL_results[ROUTES[i]] = r['DQN_samples']
    
    with open('DQN_results.pickle', 'wb') as f:
        pickle.dump(RL_results, f)