#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 13:22:05 2019

@author: juanjosealcaraz
"""
from serialize import Serialize
import random
from switch_function import switch
import math
import numpy as np
from scipy.stats import truncnorm

#INCIDENCES
MAX_INCIDENCES = 1

# EVENTS
START_DRIVING = 0
STOP_DRIVING = 1
ARRIVAL_TO_NODE = 2
SERVICE_AT_NODE = 3
END_SERVICE = 4

# constants related with EU-drivers regulations (EC)No.561/2006 & 2002/15/EC
DAILY_DRIVING_MAX = 9.0
EXTENDED_DRIVING_MAX = 10.0
TIME_TO_REST_INITIAL = 13.0
TIME_TO_REST_MAXIMUM = 15.0
EXTENDED_WORKTIME = 2.0
DAILY_REST = 11.0
DAILY_REDUCED_REST = 9.0
SPLIT_REST_TIME = 3.0
BREAK = 0.75
SHORT_BREAK = 0.5
PARTIAL_DRIVING_MAX = 4.5
CONTINOUS_WORK_MAX = 6.0

DAYS = ['Lunes','Martes','Miercoles','Jueves','Viernes','Sabado','Domingo']

# ACTION NAMES
MAKE_A_BREAK = 0
FINISH_WORKING_DAY = 1
START_REST = 2
START_REDUCED_REST = 3
EXTEND_DRIVE = 4
EXTEND_DRIVE_AND_REDUCE_REST = 5
MAKE_SPLIT_REST = 6
DRIVE = 7
SERVICE = 8
MAKE_A_SHORT_BREAK = 9
DEFAULT = 10

ACTION_NAMES = ['MAKE_A_BREAK', 'FINISH_WORKING_DAY', 'START_REST', 'START_REDUCED_REST',
           'EXTEND_DRIVE', 'EXTEND_DRIVE_AND_REDUCE_REST', 'MAKE_SPLIT_REST', 
           'DRIVE', 'SERVICE', 'MAKE_A_SHORT_BREAK', 'DEFAULT']
EVENT_NAMES = ['START_DRIVING','STOP_DRIVING', 'ARRIVAL_TO_NODE','SERVICE_AT_NODE','END_SERVICE']


class State:
    def __init__(self, end = False, t_simulation = 0.0, segment = 0, t_drive_day = 0.0, 
                 t_drive_max = DAILY_DRIVING_MAX, t_constant_work = 0.0, t_current_day = 0.0,
                 t_until_rest = TIME_TO_REST_INITIAL, current_day = 'Lunes',
                 d_to_next_node = 0.0, actual_node = 0, next_node = 1,
                 reduced_rest_today = False, extended_drive_today = False,
                 event_type = START_DRIVING, break_done = False, t_rest = DAILY_REST,
                 split_rest = False, av_reduced_rest = 3, av_extended_drive = 2):
        
        self.end = end
        self.t_simulation = t_simulation
        self.segment = segment
        self.t_drive_day = t_drive_day
        self.t_drive_max = t_drive_max
        self.t_constant_work = t_constant_work
        self.t_current_day = t_current_day
        self.t_until_rest = t_until_rest
        self.current_day = current_day
        self.d_to_next_node = d_to_next_node
        self.actual_node = actual_node
        self.next_node = next_node
        self.reduced_rest_today = reduced_rest_today
        self.extended_drive_today = extended_drive_today
        self.event_type = event_type
        self.break_done = break_done
        self.t_rest = t_rest
        self.split_rest = split_rest
        self.av_reduced_rest = av_reduced_rest
        self.av_extended_drive = av_extended_drive
        self.reward = 0
        self.vector_state = np.array([0]*14, dtype = np.float_)

    def vector(self):
        self.vector_state = np.array([self.segment/10, self.t_drive_day/10.0, self.t_drive_max/10.0, self.t_constant_work/6.0, self.t_simulation/120.0, \
                                    self.t_until_rest/15.0, self.d_to_next_node/1500.0, self.reduced_rest_today, self.event_type/4, self.break_done, \
                                    self.t_rest/11.0, self.split_rest, self.av_reduced_rest/3, self.av_extended_drive/2], dtype = np.float_)
        self.vector_state -= 0.5
    
    def set_reward(self, reward):
        self.reward = reward
        
class BasicGenerator:
    def __init__(self, seed = 1):
        self.initial_seed = seed
        random.seed(seed)
        self.state = random.getstate()
    
    def store_state(self):
        self.state = random.getstate()
        
    def restore_state(self):
        random.setstate(self.state)
    
    def reset(self, seed = None):
        if seed:
            random.seed(seed)
        else:
            random.seed(self.initial_seed)
            

class NormalGenerator(BasicGenerator):
    def __init__(self, seed = 1, av_speed = 80, max_speed = 100, min_speed = 60, speed_dev = 5, stime_dev = 0.1, min_serv_time = 0.5):
        BasicGenerator.__init__(self,seed)
        self.av_speed = av_speed
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.speed_dev = speed_dev
        self.stime_dev = stime_dev
        self.min_serv_time = min_serv_time
    
    def generate_velocity(self):
        velocity = random.gauss(self.av_speed, self.speed_dev)
        if velocity > self.max_speed:
            return self.max_speed
        elif velocity < self.min_speed:
            return self.min_speed
        else:
            return velocity
        
    def generate_service_time(self, av_serv_time):
        if av_serv_time > 0.0:
            serv_time = random.gauss(av_serv_time, self.stime_dev * av_serv_time)
            if serv_time < self.min_serv_time:
                serv_time = self.min_serv_time
            return serv_time
        else:
            return 0.0
            
class IncidenceNormalGenerator(BasicGenerator):
    def __init__(self, seed = 1, p_incidence = 0.1, max_incidences = MAX_INCIDENCES, av_speed = 80, max_speed = 100, min_speed = 60, speed_dev = 5, stime_dev = 0.1, min_serv_time = 0.5):
        BasicGenerator.__init__(self,seed)
        self.normal_generator = NormalGenerator(seed, av_speed, max_speed, min_speed, speed_dev, stime_dev, min_serv_time)
        self.incidence_generator = NormalGenerator(seed, av_speed = 25, max_speed = 40, min_speed = 10, speed_dev = 1, stime_dev = 0.1, min_serv_time = 1.0)
        self.av_speed = av_speed
        self.p_inicidence = p_incidence 
        self.possible_incidences = max_incidences
    
    def generate_velocity(self):
        p = random.random()
        if p < self.p_inicidence and self.possible_incidences > 0:
            self.possible_incidences -= 1
            return self.incidence_generator.generate_velocity()
        else:
            return self.normal_generator.generate_velocity()
        
    def generate_service_time(self, av_serv_time):
        p = random.random()
        if p < self.p_inicidence and self.possible_incidences > 0:
            self.possible_incidences -= 1
            return self.incidence_generator.generate_service_time(2*av_serv_time)
        else:
            return self.normal_generator.generate_service_time(av_serv_time)
    
    def reset(self, seed = None, max_incidences = MAX_INCIDENCES):
        super().reset(seed)
        self.possible_incidences = max_incidences
        
            
class TruncNormalGenerator(BasicGenerator):
    def __init__(self, seed = 1, av_speed = 80, speed_dev = 5, min_speed = 60, max_speed = 100, stime_dev = 0.1, min_serv_time = 0.5, max_serv_time = 1.5):
        BasicGenerator.__init__(self,seed)
        self.av_speed = av_speed
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.speed_dev = speed_dev
        self.a_speed = (min_speed - av_speed)/speed_dev
        self.b_speed = (max_speed - av_speed)/speed_dev
        
        self.stime_dev = stime_dev
        self.min_serv_time = min_serv_time
        self.max_serv_time = max_serv_time
    
    def generate_velocity(self):
        v_norm = truncnorm.rvs(self.a_speed, self.b_speed, size=1)
        v = v_norm[0]*self.speed_dev + self.av_speed
        return v
        
    def generate_service_time(self, av_serv_time):
        stv_dev = self.stime_dev * av_serv_time
        a = (self.min_serv_time - av_serv_time)/stv_dev
        b = (self.max_serv_time - av_serv_time)/stv_dev
        s_norm = truncnorm.rvs(a, b, size=1)
        s = s_norm[0]*stv_dev + av_serv_time
        return s
    
        
class RouteEnvironmentQlearning:
    def __init__(self, route, generator_a, generator_b):
        data = Serialize.load(route)
        self.path = data[0]
        self.tw = data[1]
        self.s_times = data[2]
        self.distances = data[3]
        self.dep_time = data[4]
        self.demands = data[5]
        self.veh_capacity = data[6]
        self.names = data[7]
        self.start_day = data[8]
        self.generator_list = [generator_a, generator_b]
        self.generator = self.generator_list[0]
        self.generator_selector = 0
        self.flags = 3 

    def switch_generator(self):
        self.generator.store_state()
        self.generator_selector = (self.generator_selector + 1)%2
        self.active_generator = self.generator_list[self.generator_selector]
        self.generator.restore_state()

    def reset(self):
            
        self.generator_list[0].reset()
        self.generator_list[1].reset()
        self.generator = self.generator_list[0]
        self.generator_selector = 0
        
        # departure time & t_until_rest
        depot_load_time = self.generator.generate_service_time(self.s_times[0])        
        deptarture_time = self.dep_time + depot_load_time
        
        t_until_rest = TIME_TO_REST_INITIAL - depot_load_time
        
        # departure day
        current_day = DAYS[self.start_day]

        # distance to next node
        d_to_next_node = self.distances[0]
        
        # initial state
        state = State(t_simulation = deptarture_time, current_day = current_day,
                      t_constant_work = depot_load_time, t_current_day = depot_load_time, 
                      d_to_next_node = d_to_next_node, t_until_rest = t_until_rest, 
                      actual_node = self.path[0], next_node = self.path[1])
        
        # initial trace
        trace = (self.flags+len(self.demands))*[0] 
        
        # advance until a decision must be made
        state, trace = self.start_driving(state, trace)
        action_set = self.create_action_set(state)

        state.vector()

        self.trace_cost = self.compute_trace_cost(trace)
        
        return state, action_set, trace

    def compute_trace_cost(self, trace):
        def positive_val(n):
            if n > 0.0:
                return 1.0
            return 0.0
        result = -1 * trace[1] -1 * trace[2]
        for t, d in zip(trace[3:], self.demands):
            result = result - positive_val(t) * d / 10.0

        return result
        
    def set_agent(self, agent):
        self.agent = agent
    
    def step(self, state, action, trace):
        while True: # continues until a decision must be made
            if action == DEFAULT:
                action = self.rollout_policy(state)

            # execute action
            state, trace = self.execute_action(state, action, trace)

            # reward given by the increment of the trace cost (negative values)
            new_trace_cost = self.compute_trace_cost(trace)
            reward = new_trace_cost - self.trace_cost
            self.trace_cost = new_trace_cost

            state.set_reward(reward)

            # compute action set
            action_set = self.create_action_set(state)

            if len(action_set) > 1 or state.end == True:
                action_set.append(DEFAULT)
                return state, action_set, trace
            elif len(action_set) == 1:
                action = action_set[0]
            else:
                action = DEFAULT
    
    def execute_action(self, state, action, trace):
        current_event = state.event_type
        for case in switch(current_event): # events where actions can be selected
            if case(START_DRIVING):
                state, trace = self.start_driving(state, trace)
                break
            if case(STOP_DRIVING):
                state, trace = self.stop_driving(state, action, trace)
                break
            if case(END_SERVICE):
                state, trace = self.end_service(state, action, trace)
                break
            if case(ARRIVAL_TO_NODE):
                state, trace = self.arrival_to_node(state, action, trace)
                break
            if case(SERVICE_AT_NODE):
                state, trace = self.service_at_node(state, action, trace)
                break
            if case():
                print(' ERROR: state.event_type not defined ')
                break
        state.vector()
        return state, trace

    
    def start_driving(self, state, trace):  
        # generate speed
        velocity = self.generator.generate_velocity() 
        
        # t available
        t_available_work = CONTINOUS_WORK_MAX - state.t_constant_work
        t_available_drive = min(PARTIAL_DRIVING_MAX, state.t_drive_max - state.t_drive_day)

        t_available = min(t_available_work, t_available_drive, state.t_until_rest)
        
        # distance covered
        d_achievable = t_available * velocity

        t_drive = 0.0
        distance = d_achievable

        if (d_achievable < state.d_to_next_node):
            state.d_to_next_node = state.d_to_next_node - d_achievable
            t_drive = t_available
            state.event_type = STOP_DRIVING
            
        else:  # arrive at destination
            distance = state.d_to_next_node
            t_drive = distance / velocity
            state.d_to_next_node = 0.0
            state.event_type = ARRIVAL_TO_NODE
        
        # advance time
        state.t_drive_day += t_drive
        state.t_simulation += t_drive
        state.t_constant_work += t_drive
        state.t_current_day += t_drive
        state.t_until_rest = max(state.t_until_rest - t_drive, 0.0)

        return state, trace    
        
    def service_at_node(self, state, action, trace):        
        state, trace = self.update_state_and_trace(state, action, trace)        
        
        # current location
        path_index = self.path.index(state.next_node)
        
        # window end
        window_end = self.tw[path_index][1]
        
        # service time
        service_time = self.generator.generate_service_time(self.s_times[path_index])
        
        # time update
        state.t_simulation += service_time
        state.t_constant_work += service_time
        state.t_current_day += service_time
        state.t_until_rest = max(0.0, state.t_until_rest - service_time)
        
        #trace update 
        margin = state.t_simulation - window_end
        if math.isinf(margin):
            margin = 0.0
        
        trace[self.flags + path_index] = margin
        
        #if (S.next_node == path[-1]):
        if (state.actual_node==self.path[-2] and state.next_node==self.path[-1]):
            state.end = True  #simulation ends

        else:
            state.segment += 1
            state.actual_node = state.next_node
            state.next_node = self.path[path_index + 1]
            state.d_to_next_node = self.distances[state.segment]
        
        state.event_type = END_SERVICE
        
        return state, trace            
    
    def end_service(self, state, action, trace):
        if action != DRIVE:
            state, trace = self.update_state_and_trace(state, action, trace)
        
        state.event_type = START_DRIVING
        
        return state, trace
        
    def stop_driving(self, state, action, trace):
        state, trace = self.update_state_and_trace(state, action, trace)
        
        state.event_type = START_DRIVING
        
        return state, trace
        
    def arrival_to_node(self, state, action, trace):
        state, trace = self.update_state_and_trace(state, action, trace)
        
        state.event_type = SERVICE_AT_NODE
        
        return state, trace
        
    def update_state_and_trace(self, state, action, trace):
        # this function updates state and trace for specific actions 
        change_day = False
        reduced_rest = 0
        extend_drive = 0
                   
        for case in switch(action):
            if case(MAKE_A_BREAK):
                t_break = (1-state.break_done)*BREAK + state.break_done*SHORT_BREAK
                state.t_simulation += t_break
                state.t_current_day += t_break
                state.t_constant_work = 0.0
                state.t_until_rest = max(state.t_until_rest - t_break, 0.0)
                state.break_done = True
                break
            if case(MAKE_SPLIT_REST):
                state.t_simulation += SPLIT_REST_TIME
                state.t_current_day += SPLIT_REST_TIME
                state.t_constant_work = 0.0
                state.t_until_rest = state.t_until_rest - SPLIT_REST_TIME + EXTENDED_WORKTIME
                state.t_rest = DAILY_REDUCED_REST
                state.split_rest = True
                break
            if case(FINISH_WORKING_DAY):
                state.t_simulation = state.t_simulation + state.t_until_rest + state.t_rest
                change_day = True
                break
            if case(START_REST):
                state.t_simulation = state.t_simulation + state.t_rest
                change_day = True
                break
            if case(START_REDUCED_REST):
                state.t_simulation = state.t_simulation + DAILY_REDUCED_REST
                state.av_reduced_rest = state.av_reduced_rest - 1
                change_day = True
                reduced_rest = 1
                break
            if case(EXTEND_DRIVE_AND_REDUCE_REST, EXTEND_DRIVE):
                if action == EXTEND_DRIVE_AND_REDUCE_REST:
                    state.reduced_rest_today = True
                    reduced_rest = 1
                    state.av_reduced_rest = state.av_reduced_rest - 1
                    state.t_until_rest = state.t_until_rest + EXTENDED_WORKTIME
                    state.t_rest = DAILY_REDUCED_REST
                
                # extend drive          
                
                t_available_work = CONTINOUS_WORK_MAX - state.t_constant_work
                
                t_drive_estimated = state.d_to_next_node / self.generator.av_speed
                
                if (state.event_type == STOP_DRIVING) or (t_available_work < min(1.0, t_drive_estimated)):
                    state.t_simulation += SHORT_BREAK
                    state.t_current_day += SHORT_BREAK  
                    state.t_constant_work = 0.0
                    state.t_until_rest = max(state.t_until_rest - SHORT_BREAK, 0.0)
                    state.break_done = True
                
                if state.t_until_rest > 0.0:
                    state.extended_drive_today = True
                    extend_drive = 1
                    state.av_extended_drive = state.av_extended_drive - 1
                    state.t_drive_max = EXTENDED_DRIVING_MAX
                    
                else:
                    state.t_simulation = state.t_simulation + state.t_rest
                    change_day = True
                    
                break
            
            if case(SERVICE):
                # actualizar hasta el momento de apertura del cliente
                path_index = self.path.index(state.next_node)
                window_start = self.tw[path_index][0]
                time_to_open = window_start - state.t_simulation
                if time_to_open > 0.0:
                    state.t_simulation = window_start
                    state.t_current_day += time_to_open
                    state.t_until_rest = max(state.t_until_rest - time_to_open, 0.0)
                if time_to_open > 0.25:
                    state.t_constant_work = 0.0
                if time_to_open > 0.75:
                    state.break_done = True    
                break
            
        # check consistency of the state variables and constraint fulfillment
        total_elapsed_time = self.dep_time + state.t_simulation
        total_elapsed_days = math.ceil(total_elapsed_time/24)
        week_day = (self.start_day + total_elapsed_days) % 7
        state.current_day = DAYS[week_day]

        violations = []
        violations.append(state.t_drive_day > state.t_drive_max)
        violations.append(state.t_constant_work > CONTINOUS_WORK_MAX * 1.1)
        violations.append(state.t_current_day > TIME_TO_REST_MAXIMUM * 1.1)
        violations.append(state.av_reduced_rest < 0)
        violations.append(state.av_extended_drive < 0)
        
        if any(violations):
            trace[0] = 1

        trace[1] += reduced_rest
        trace[2] += extend_drive
        
        if change_day:
            state.reduced_rest_today = False
            state.extended_drive_today = False
            state.break_done = False
            state.split_rest = False
            state.t_constant_work = 0.0
            state.t_drive_day = 0.0
            state.t_current_day = 0.0
            state.t_rest = DAILY_REST
            state.t_until_rest = TIME_TO_REST_INITIAL
            state.t_drive_max = DAILY_DRIVING_MAX
        
        return state, trace
            
    def create_action_set(self, state):
        
        reduce_rest_pos = ((state.av_reduced_rest > 0) and not (state.split_rest) and not(state.reduced_rest_today))
        extend_drive_pos = (state.av_extended_drive > 0) and not(state.extended_drive_today)
        split_rest_pos = not(state.split_rest) and (state.t_current_day + SPLIT_REST_TIME < TIME_TO_REST_MAXIMUM)

        action_set = set()
        
        # maximum_t_until_rest = state.t_until_rest + (1-reduce_rest_pos) * EXTENDED_WORKTIME
        if (state.t_current_day >= TIME_TO_REST_MAXIMUM):
            action_set.add(START_REST)
            return list(action_set)

        if (state.t_until_rest == 0) and not(reduce_rest_pos):
            action_set.add(START_REST)
            return list(action_set)
 
        if (state.event_type == ARRIVAL_TO_NODE):
            # window start
            path_index = self.path.index(state.next_node)
            window_start = self.tw[path_index][0]
            window_end = self.tw[path_index][1]
            current_time = state.t_simulation
            service_time = self.s_times[path_index]
            
            time_to_open = window_start - current_time
            time_to_close = window_end - current_time
            
            t_break = (1-state.break_done)*BREAK + state.break_done * SHORT_BREAK
            
            if (time_to_open <= 0.0):
                if (state.t_constant_work + service_time < CONTINOUS_WORK_MAX):
                    action_set.add(SERVICE)
                else:
                    action_set.add(MAKE_A_BREAK)
            else:   
                if (time_to_open > state.t_until_rest + state.t_rest): # free for the rest of the day
                    action_set.add(FINISH_WORKING_DAY)
                        
                if (time_to_open > state.t_rest) or (time_to_close > state.t_rest + service_time):
                    action_set.add(START_REST) # rest is technically possible
                    if not(state.split_rest):
                        action_set.add(MAKE_SPLIT_REST)
                    if reduce_rest_pos:
                        action_set.add(START_REDUCED_REST)
                
                if reduce_rest_pos:
                    if (time_to_open > DAILY_REDUCED_REST) or (time_to_close > DAILY_REDUCED_REST + service_time):
                        action_set.add(START_REDUCED_REST)
                    
                if split_rest_pos:
                    if (time_to_open > SPLIT_REST_TIME) or (time_to_close > SPLIT_REST_TIME + service_time):
                        action_set.add(MAKE_SPLIT_REST) # split rest is technically possible

                if (time_to_open > t_break): #  time for a break
                    action_set.add(MAKE_A_BREAK)
                
                elif(state.t_constant_work + service_time < CONTINOUS_WORK_MAX):  #  serve now
                    action_set.add(SERVICE)
                
                else:  #  must take a break
                    action_set.add(MAKE_A_BREAK)
        
        elif state.event_type == STOP_DRIVING:
            
            action_set.add(MAKE_A_BREAK)
            
            if split_rest_pos:
                action_set.add(MAKE_SPLIT_REST) # split rest is technically possible
                
            if state.t_until_rest < 0.5 or state.t_drive_day > state.t_drive_max * 0.9 or state.t_current_day > TIME_TO_REST_MAXIMUM * 0.9:
                action_set.update({FINISH_WORKING_DAY, START_REST}) # finish work is technically possible
            
                if extend_drive_pos:
                    action_set.add(EXTEND_DRIVE) # extend drive is technically possible
                
                if reduce_rest_pos:
                    action_set.add(START_REDUCED_REST) # reduce drive is technically possible
                    
                if extend_drive_pos and reduce_rest_pos:
                    action_set.add(EXTEND_DRIVE_AND_REDUCE_REST) # extend drive is technically possible           
            
        elif state.event_type == END_SERVICE:
            
            if not(state.break_done) or (state.t_constant_work >= CONTINOUS_WORK_MAX*0.9) or state.t_current_day > TIME_TO_REST_MAXIMUM * 0.9: # make a break
                action_set.add(MAKE_A_BREAK)
            
            if split_rest_pos:
                action_set.add(MAKE_SPLIT_REST) # split rest is technically possible
                
            if state.t_until_rest < 0.5 or state.t_drive_day > state.t_drive_max * 0.9:
                action_set.update({FINISH_WORKING_DAY, START_REST}) # finish work is technically possible
            
                if extend_drive_pos:
                    action_set.add(EXTEND_DRIVE) # extend drive is technically possible
                
                if reduce_rest_pos:
                    action_set.add(START_REDUCED_REST) # reduce drive is technically possible
                    
                if extend_drive_pos and reduce_rest_pos:
                    action_set.add(EXTEND_DRIVE_AND_REDUCE_REST) # extend drive is technically possible
            
            if (state.t_constant_work < CONTINOUS_WORK_MAX) and (state.t_drive_day < state.t_drive_max):
                action_set.add(DRIVE)
            
        elif (state.event_type == SERVICE_AT_NODE):
            action_set.add(SERVICE)

        elif (state.event_type == START_DRIVING):
            action_set.add(DRIVE)

        return list(action_set)

    def rollout_policy(self, state):
        
        t_drive_estimated = state.d_to_next_node / self.generator.av_speed
        reduce_rest_pos = ((state.av_reduced_rest > 0) and not(state.split_rest) and not(state.reduced_rest_today))
        split_rest_pos = not(state.split_rest) and (state.t_current_day + SPLIT_REST_TIME < TIME_TO_REST_MAXIMUM)
        maximum_t_until_rest = state.t_until_rest + (1-reduce_rest_pos) * EXTENDED_WORKTIME
        path_index = self.path.index(state.next_node)
        window_start = self.tw[path_index][0]
        window_end = self.tw[path_index][1]
        current_time = state.t_simulation
        service_time = self.s_times[path_index]
        time_to_open = window_start - current_time
        time_to_deadline = window_end - current_time
        
        decision = DEFAULT
        
        if (state.event_type==STOP_DRIVING):
            
            t_extended_drive = min(1.0, state.t_until_rest - SHORT_BREAK)
            extend_drive_pos = ((state.av_extended_drive > 0) and (t_extended_drive > 0.1)) and not(state.extended_drive_today)

            if(state.t_drive_day < state.t_drive_max) and (state.t_until_rest > 0.0): 
                decision = MAKE_A_BREAK
            elif((state.t_until_rest + state.t_rest + t_drive_estimated)< time_to_open):
                decision = FINISH_WORKING_DAY
            elif((state.t_rest + t_drive_estimated) < time_to_open):
                decision = START_REST
            elif(((state.t_rest + t_drive_estimated + service_time) > time_to_deadline) and (reduce_rest_pos or extend_drive_pos)): 
                if(((DAILY_REDUCED_REST + t_drive_estimated) < time_to_deadline) and reduce_rest_pos): 
                    decision = START_REDUCED_REST
                elif((t_drive_estimated < t_extended_drive) and extend_drive_pos): 
                    decision = EXTEND_DRIVE
                elif(extend_drive_pos and reduce_rest_pos):
                    decision = EXTEND_DRIVE_AND_REDUCE_REST
                else:
                    decision = START_REST
            else:
                decision = START_REST
                
        elif(state.event_type==ARRIVAL_TO_NODE):
            
            t_break = (1-state.break_done)*BREAK + state.break_done * SHORT_BREAK
            
            if((time_to_open > state.t_rest) or (time_to_open + service_time > maximum_t_until_rest)):
                decision = START_REST       
            elif((time_to_open > SPLIT_REST_TIME) and split_rest_pos): 
                decision = MAKE_SPLIT_REST
            elif(time_to_open > t_break): 
                 decision = MAKE_A_BREAK
            elif(state.t_constant_work + service_time < CONTINOUS_WORK_MAX):
                decision = SERVICE
            else: 
                decision = MAKE_A_BREAK
            
        elif (state.event_type == SERVICE_AT_NODE):
            decision = SERVICE
        
        elif(state.event_type == END_SERVICE):            
            if(state.t_until_rest > 0.0) and (state.t_drive_day < state.t_drive_max): 
                if (state.t_constant_work < CONTINOUS_WORK_MAX): 
                    decision = DRIVE
                else:
                    decision = MAKE_A_BREAK
            else:  
                decision = START_REST
        
        elif (state.event_type == START_DRIVING):
            decision = DRIVE
    
        return decision    

if __name__== "__main__":
    import sys
    from operator import itemgetter 
    route = sys.argv[1]
    generator_a = NormalGenerator()
    generator_b = NormalGenerator()
    route_sim = RouteEnvironmentQlearning(route, generator_a, generator_b)
    state, decisions, trace = route_sim.reset() 
    print('initial decisions : '+str(itemgetter(*decisions)(ACTION_NAMES)))
    for step in range(10):
        print('simtime : '+str(state.t_simulation))
        print('state event : '+EVENT_NAMES[state.event_type])
        print('decisions : '+str(decisions))
        print('decisions names : ' + str(itemgetter(*decisions)(ACTION_NAMES)))
        print('selected decision: '+ACTION_NAMES[decisions[0]])
        state, decisions, trace = route_sim.step(state, decisions[0], trace)
        print('step : ' + str(step))
        print('state_vector : ' + str(state.vector_state))
        print('trace : ' + str(trace))  
        print('reward : ' + str(state.reward))      