#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 16:02:57 2019

@author: juanjosealcaraz

"""

from random import choice
from collections import namedtuple, defaultdict
from math import log, ceil, floor
from statistics import mean
from copy import deepcopy
import time

DEBUG = False

State = namedtuple('State',['steps','end'])

class ToyEnvironment:
    '''
    a simple environment implementing the necessary interface for the rollout agent
    '''
    def __init__(self, states, actions):
        self.N = len(states)
        self.states = states
        self.actions = actions

    def step(self, state, action, trace):
        final = False
        new_steps = state.steps + 1
        if new_steps == self.N - 1:
            final = True
        out_state = state._replace(steps = new_steps, end = final)
        trace[state.steps] = -1*action
        return out_state, self.actions, trace

    def reset(self):
        state = State(steps = 0, end = False)
        trace = [0]*self.N
        return state, self.actions, trace

    def rollout(self, state, trace):
        rollout_state = State._make(state[:])
        action = choice(self.actions)
        while rollout_state.end == False:
            rollout_state, actions, trace = self.step(rollout_state, action, trace)
            action = choice(self.actions)
        return trace
    
    def switch_generator(self):
        pass


class CostEstimator:
    '''
    auxiliary class defining diverse cost criteria
    '''
    def __init__(self, weight = []):
        self.weight = weight
    
    def average_cost(self, traces_list):
        m = len(traces_list)
        traces_sum = [sum(i)/m for i in zip(*traces_list)]
        if self.weight:
            traces_sum = [a*b for a,b in zip(traces_sum, self.weight)]
        return mean(traces_sum)
    
    def monetary_cost(self, traces_list):       
        def positive_val(n):
            if n > 0.0:
                return 1.0
            return 0.0
        result = []
        for trace in traces_list:
            processed_trace = [positive_val(e) for e in trace]
            processed_trace[0:3] = trace[0:3] # flags
            cost = 0
            for a,b in zip(processed_trace, self.weight):
                cost += a*b
            result.append(cost)
        return mean(result)
        

class RolloutAgent:
    '''
    This is the rollout agent that estimates the best action at each stage by simulating possible futures of the 
    system (model predictive control) from current stage using each action.
    The actions of current and upcoming stages are evaluated by means a Monte Carlo tree search (MCTS) strategy.
    The agent generates trajectories from the state observed at current stage.
    In the generated trajectories the actions are selected by means of a multi-armed bandit algorithm up to the lookahead horizon (depth).
    Beyond the lookahead horizon the agent uses a baseline policy (rollout) to select the actions until the end of the trajectory.
    '''
    def __init__(self, environment, depth, per_action_budget, cost_criteria):
        # route must implement
        # reset(seed), step(state, decision, trace), rollout(state, trace)
        self.environment = environment
        self.depth = depth
        self.per_action_budget = per_action_budget
        self.cost_criteria = cost_criteria
        self.total_runs = 0

    def evaluate_decision(self, state, decision, trace, depth):
        '''
        state: observed state
        decision: selected decision at current stage
        trace: rewards (or costs) received along the trajectory so far
        depth: lookahead horizon
        '''
        traces_list = []
        state, decisions, trace = self.environment.step(state, decision, trace)
        # check if we have reached the end of the route
        if state.end == True:
            traces_list.append(trace)
            return traces_list

        if depth > 0:
            # evaluates decisions and selects one
            N = len(decisions)*self.per_action_budget
            traces_list = self.find_best_decision(state, decisions, trace, N, depth-1)[1] # recursive call: for the next level we reduce the depth

        else: # rollout
            trace = self.environment.rollout(state, trace)
            traces_list.append(trace)
            self.total_runs += 1

        return traces_list # returns all the traces of the selected action

    def find_best_decision(self, state, decisions, trace, N, depth):
        '''
        state: observed state
        decisions: set of decisions we need to evaluate
        trace: rewards (or costs) received along the trajectory so far
        N: sample budget
        depth: lookahead horizon
        '''
        trace_record = defaultdict(list)
        for d in decisions:
            trace_record[d] = []
        M = len(decisions)
        A = decisions.copy()
        T = ceil(log(M,2))
        for _ in range(T):
            samples = floor(N/(ceil(log(M,2)*len(A))))
            for a in A:
                for _ in range(samples):
                    trace_copy = trace.copy()
                    state_copy = deepcopy(state)
                    traces_list = self.evaluate_decision(state_copy, a, trace_copy, depth) # recursive call: the next tree level is generated here
                    trace_record[a].extend(traces_list)
            action_cost = {a: self.cost_criteria(trace_record[a]) for a in A}
            ordered_actions = sorted(action_cost, key = action_cost.get)
            new_len_A = ceil(len(A)/2)
            A = ordered_actions[:new_len_A] # each round we remove half of the elements in A           
        a = A[0] # at the end A only contains the best estimated action
        return a, trace_record[a]

    def run(self):      
        times = [] # list with pairs (nª actions, time): one pair per decision stage

        # create initial state from route
        state, decisions, trace = self.environment.reset()
        if DEBUG:
            counter = 1
            print('run:')
            print(trace)

        # main loop
        while True:
            # check if we have reached the end of the route
            if state.end == True:
                break
            # evaluates decisions and selects one
            A = len(decisions)
            N = A*self.per_action_budget
            t1 = time.time()
            self.environment.switch_generator()
            decision = self.find_best_decision(state, decisions, trace, N, self.depth)[0] # select the best action at current step
            self.environment.switch_generator()
            t2 = time.time() - t1
            times.append((A, t2)) # list with pairs nº actions, time
            
            if DEBUG:
                counter += 1
                print('step : '+str(counter))
                print('total runs : '+str(self.total_runs)) 
                print('decision : '+str(decision))

            # step forward to the next stage
            state, decisions, trace = self.environment.step(state, decision, trace)
            
            if DEBUG:
                print('trace after decision : ')
                print(trace)
                print('end = '+str(state.end))

        return trace, times

if __name__== "__main__":
    actions = list(range(3))
    states = list(range(5))
    model = ToyEnvironment(states, actions)
    ce = CostEstimator([1]*5)
    cost_criteria = ce.average_cost
    depth = 2
    budget = 10
    agent = RolloutAgent(model, depth, budget, cost_criteria)
    trace = agent.run(1)
    print(trace)
