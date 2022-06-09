# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 16:05:11 2018

@author: Luis
"""
import pickle
import platform
from pathlib import Path
from os.path import dirname
from os.path import abspath

class Serialize:
    
    def save(route):
        data = {} #empty dictionary structure
        route_id=route.routeid
        
        dir = dirname(abspath(__file__))
        data_folder = Path(dir)
        folder_path = data_folder / 'routes'
        string='route'+str(route_id)+'.pickle'
        
        if platform.system()=='Windows':
            file=folder_path / string
        else:
            file=str(folder_path / string)
        
        
        #saving simulation useful info 
        data[0]=route.path
        data[1]=route.tw
        data[2]=route.service_t
        data[3]=route.distances
        data[4]=route.vehicle.departure_time
        data[5]=route.demands
        data[6]=route.vehicle.capacity
        #extra pretty names 
        names=[]
        for node in route.path:
            names.append(route.DB.get_name_byid(node))
        data[7]=names
        data[8]=route.DB.start_day
        
        #save in memory-file(serialize)
        with open(file,'wb') as f : 
            pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)    
        return 
    
    def load(route):
        
        dir = dirname(abspath(__file__))
        data_folder = Path(dir)
        folder_path = data_folder / 'routes'
        string='route'+str(route)+'.pickle'

        if platform.system()=='Windows':
            file=folder_path / string
        else:
            file=str(folder_path / string)
        
        with open(file,'rb') as f: 
            data = pickle.load(f) 
        return data 