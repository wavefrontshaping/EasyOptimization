# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 18:16:31 2020

@author: team popoff
"""

import numpy as np
import time
from tqdm import tqdm, trange
import functools

class ReturnDictError(Exception):
    def __init__(self, func):
        self.msg = f'Function {func.__name__} should return a dictionnary and nothing else.'
#         logger.error(self.msg)
        print(self.msg)
        super().__init__(self.msg)
        



def register_current_data(func):
    @functools.wraps(func)
    def wrapped(self, *args, **kwargs):
        data = kwargs.get('data', None).copy()
        if data:
            data.update(self._return_data)
            kwargs['data'] = data
        new_data = func(self, *args, **kwargs)
        if not isinstance(new_data, dict):
            raise ReturnDictError(func)
        self._current_data.update(new_data)
    return wrapped

def register_data(func):
    @functools.wraps(func)
    def wrapped(self, *args, **kwargs):
        if 'data' in kwargs.keys():
            kwargs['data'].update(self._return_data)
        new_data = func(self, *args, **kwargs)
        if not isinstance(new_data, dict):
            raise ReturnDictError(func)
        self._return_data.update(new_data)
    return wrapped

class EasyAlgo():
    def __init__(self, 
                 n_var,
                 min_or_max = 'min',
                 n_repeat = 1
                ):
        self.n_var = n_var
        self.n_repeat = n_repeat
        self.min_or_max = min_or_max
        self.evol_data = []
        self._return_data = {}
        self._current_data = {}
    
    def run(self, *args, x_init = None, **kwargs):
        self.current_x = self.init_x(x_init)
        self.begin()
        self.best_cost, self.best_data = self.callback(self.current_x)
        self.evol = []
        for i_full_iter in range(self.n_repeat):
            self.before_full_iter()
            kwargs['i_full_iter'] = i_full_iter
            self.run_once(*args, **kwargs)
            self.after_full_iter(i_full_iter, 
                                 self.all_time_best_cost, 
                                 self.all_time_best_x, 
                                 data = self._return_data)

        self.end()
        # update the return_data dictionnary with the best results obtained
        self._return_data.update(self.all_time_best_data)
    
        return self.all_time_best_cost, \
               self.all_time_best_x, \
               self._return_data
    
    @staticmethod
    def value_constraint(v):
        return v
    
    @register_data
    def on_best(self, best_x, best_cost, best_data):
        '''
        Called each time the algorithm find a new "all time best" value.
        '''
        return {}
    
    @register_current_data
    def after_iter(self, 
                   i_full_iter,
                   all_time_best_cost, 
                   all_time_best_x, 
                   data):
        '''
        Called after each iteration.
        '''
        return {}
    
    @register_data
    def after_full_iter(self, 
                        i_full_iter, 
                        all_time_best_cost, 
                        all_time_best_x, 
                        data):
        '''
        Called after each full iteration, i.e. after each "repeat".
        '''
        return {}
    
    @register_data
    def before_full_iter(self):
        '''
        Called before each full iteration, i.e. before each "repeat".
        '''
        return {}
    
    @register_data
    def begin(self):
        '''
        Called once at the beginning.
        ''' 
        return {}
    
    @register_data
    def end(self):
        '''
        Called once at the very end.
        '''
        return {}
    
    @staticmethod
    def log_value(name, value):
        pass
            
    def init_x(self, x_init):
        if x_init is None:
            x_init = [0.]*self.n_var
        return x_init 
    
    def register_current_data(self):
        ## Append recorded data
        # evolution of the cost function
        self.evol.append(self.best_cost)
        # now we add the data from this iteration to the data list
        # we add both the data from the callback function for the best result
        self.evol_data.append(self.best_data)
        # and the data from the helper functions (with @register_data)
        self.evol_data[-1].update(self._current_data)
    
    def register_callback(self, callback):
        self.callback = callback
       
    def run_once(self):
        pass
          
class EasyIteration(EasyAlgo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def run_once(self, values, i_full_iter = 0):
 
        coeff = 1 if self.min_or_max == 'min' else -1
        
        current_best_value = None
        x = self.current_x
        
        tr = trange(self.n_var)
        for ind_x in tr:
            # reinitialize current data 
            self._current_data = {}
            for ind_val, value in enumerate(values):
                x[ind_x] = self.value_constraint(value)
                current_cost, current_data = self.callback(x)          
                if (i_full_iter == 0 and ind_val == 0 and ind_val == 0):
                    self.all_time_best_x = x
                    self.all_time_best_data = current_data
                    self.all_time_best_cost = current_cost                    
                if ind_val == 0 or current_cost*coeff < self.best_cost*coeff:
                    self.best_cost = current_cost
                    self.best_data = current_data
                    current_best_value = value
                    
            x[ind_x] = current_best_value
            
            if self.best_cost*coeff < self.all_time_best_cost*coeff:
                self.all_time_best_x = x
                self.all_time_best_data = self.best_data
                self.all_time_best_cost = self.best_cost
                self.on_best(
                    self.all_time_best_x,
                    self.all_time_best_cost,
                    self.all_time_best_data)
            
            # Print current state of optimization in the progress bar
            msgs = [f"Repeat: {i_full_iter+1}/{self.n_repeat}",
                    f"Iter: {ind_x+1}/{self.n_var}",
                    f"Cost = {self.best_cost:.4f}"]
            tr.set_description(' | '.join(msgs))
            tr.refresh()
            self.after_iter(ind_x, 
                            self.all_time_best_cost, 
                            self.all_time_best_x,
                            data = self.all_time_best_data)
            
            self.register_current_data()
      
        self.current_x = self.all_time_best_x
        
  
class EasyPartition(EasyAlgo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run_once(self, values, fractions, i_full_iter = 0):
        
        part_sizes = (fractions*self.n_var).astype(np.int)
        coeff = 1 if self.min_or_max == 'min' else -1
        
        current_best_value = None
        x = np.array(self.current_x)
        
        pbar = tqdm(part_sizes)
            

        for ind_part, partition_size in enumerate(pbar):
            # reinitialize current data 
            self._current_data = {}
            # select a random partition of size partition_size
            partition = np.random.permutation(self.n_var)[:partition_size]
            for ind_val, value in enumerate(values):
                x[partition] = np.array(self.current_x)[partition] + value
                x[partition] = self.value_constraint(x[partition])
                current_cost, current_data = self.callback(x.tolist()) 
                current_data.update(self._current_data)
                if (i_full_iter == 0 and ind_val == 0 and ind_val == 0):
                    self.all_time_best_x = x
                    self.all_time_best_data = current_data
                    self.all_time_best_cost = current_cost                    
                if ind_val == 0 or current_cost*coeff < self.best_cost*coeff:
                    # record the best result of the input vectors tested for the current iteration
                    self.best_cost = current_cost
                    self.best_data = current_data
                    current_best_value = value
             

            x[partition] = np.array(self.current_x)[partition]+current_best_value
            self.current_x = x.tolist()
            
            # check if the best of the current iteration is the best so far
            # if so, save the data
            if self.best_cost*coeff < self.all_time_best_cost*coeff:
                self.all_time_best_x = x
                self.all_time_best_data = self.best_data
                self.all_time_best_cost = self.best_cost
                self.on_best(
                    self.all_time_best_x,
                    self.all_time_best_cost,
                    self.all_time_best_data)
            
            msgs = [f"Repeat: {i_full_iter+1}/{self.n_repeat}",
                    f"Part: {ind_part+1}/{len(part_sizes)}",
                    f"Cost = {self.best_cost:.4f}"]
            pbar.set_description(' | '.join(msgs))
            pbar.refresh()

            self.after_iter(ind_part, 
                            self.all_time_best_cost, 
                            self.all_time_best_x,
                            data = self.all_time_best_data)
            
            self.register_current_data()
            
            
        self.current_x = x.tolist()

