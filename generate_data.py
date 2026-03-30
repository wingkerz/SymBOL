import os
import time
from copy import copy

import hydra
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import pickle

import torch
import torch.nn as nn
from torch_scatter import scatter_sum

# import adamod

from data_create.NetworkSystemInstances_new import GeneralDynamics
from data_create.lib.Topo import Topo
from data_create.lib.InitCondition import InitCondition
from string_create.Creation import Creation
# import nn_models

###
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
# device = torch.device("cpu")
print("using device : ", device)


class GenerateData:
    def __init__(
            self,
            cfg,
            eq_filename=None,
    ):
        # build model
        self.max_dim = 5
        
        self.eq_filename = eq_filename
        
 
        self.state_dim = 1
        self.norm_state = True
        columns = ['no']
        columns += [f'f_eq_str_{i}' for i in range(self.state_dim)]
        columns += [f'g_eq_str_{i}' for i in range(self.state_dim)]
        columns += ['state', 'bool']
        dtype_dict = {col: str for col in columns if "eq_str" in col}
        dtype_dict.update({"no": int, "bool": bool})  

        self.dataset = pd.read_csv(
            self.eq_filename,
            header=None,
            names=columns,
            dtype=dtype_dict  
        )
        # self.dataset = pd.read_csv(
        #     self.eq_filename,
        #     header=None,
        #     names=columns
        # )
        self.repeat_num = 5
        
        self.dataset = self.dataset.loc[self.dataset.index.repeat(self.repeat_num)]
        
        self.dataset = shuffle(self.dataset)
        
        self.same_topo_num = 1
        
        #
        self.generated_data = []

        self.cfg = cfg


    def generate_data(self, ):
        state_dim = self.state_dim
        lineno = 0
        simu_ok_num = 0
        start_time = time.time()
        while lineno < len(self.dataset):
            print(self.dataset.columns)  
            print(self.dataset.iloc[lineno]) 
            f_eq_str = [self.dataset.iloc[lineno]['f_eq_str_%s' % i] for i in range(state_dim)]
            g_eq_str = [self.dataset.iloc[lineno]['g_eq_str_%s' % i] for i in range(state_dim)]
            #f_eq_str = self.dataset.iloc[lineno]['f_eq_str']
            #g_eq_str = self.dataset.iloc[lineno]['g_eq_str']
            state = self.dataset.iloc[lineno]['state']
            bool1 = self.dataset.iloc[lineno]['bool']
            self.norm_state = state
            lineno += 1

            if not bool(bool1):
                continue
                
            if simu_ok_num % self.same_topo_num == 0:
                
                print('**resample topo**[ succeed number = %s (dataset lineno %s)]' % (simu_ok_num,lineno))
                
  
                topo_index = (simu_ok_num // self.same_topo_num) % len(self.cfg.topo.type_list)
                topo_type_sampled = self.cfg.topo.type_list[topo_index]

                N_sampled = 100

  
                topo = Topo(N_sampled, topo_type_sampled)
                
                N_sampled = topo.N
            else:
                print('**keep same topo**[ succeed number = %s (dataset lineno %s)]' % (simu_ok_num,lineno))
            
            # print('lineno=', lineno)


            # generate data
            # 修改后：
            # 修改正负
            # lbs = torch.ones(state_dim) * 0.001  # 确保所有维度初始值为正
            # ubs = torch.ones(state_dim) * 2.0
            # lbs = torch.tensor([0.5, 0.5])  # 避免 x_1_0/x_1_1 过小或过大
            # ubs = torch.tensor([2.0, 2.0])  # 限制比值在 0.25 ~ 4.0 之间
            init_cond = InitCondition(N_sampled, state_dim, lbs=None, ubs=None, num_sampling=1)
            # print("初始状态")
            # print(init_cond)
            # print(init_cond.shape)
            # exit(-1)
            # print("N_sampled")
            # print(N_sampled)
            # init_cond = InitCondition(N_sampled, state_dim, lbs=torch.ones(state_dim) * (-5.),
            #                               ubs=torch.ones(state_dim) * 5., num_sampling=1)

            s_ns = GeneralDynamics(N_sampled, state_dim, topo_type_sampled, f_eq_str, g_eq_str, topo, init_cond)
            try:
                print(self.cfg.t.start)
                print(self.cfg.t.inc)
                print(self.cfg.t.end)
                s_data = s_ns.simulating_data(self.cfg.t.start,
                                              self.cfg.t.inc,
                                              self.cfg.t.end,
                                              self.cfg.resample_init_condition,
                                              norm_state=self.norm_state)
            except Exception as e:
                print(f"Simulation failed: {e}")
                print(f"Current x_1_0: {s_ns.ns.init_condition.X[:, 0]}")
                print(f"Current x_1_1: {s_ns.ns.init_condition.X[:, 1]}")
                raise
            # s_data = s_ns.simulating_data(self.cfg.t.start,
            #                                   self.cfg.t.inc,
            #                                   self.cfg.t.end,
            #                                   self.cfg.resample_init_condition,
            #                                   norm_state=self.norm_state)
            
            print(f_eq_str, g_eq_str)
            # print(s_data)
            # exit(-1)
            if s_data is None:
                print('**Simulation failed!!!**')
                continue
            
            #print('s_data[self_diff][1].min(), max()',s_data['self_diff'][1].min(),s_data['self_diff'][1].max())
            #print('s_data[interact_diff][1].min(), max()',s_data['interact_diff'][1].min(),s_data['interact_diff'][1].max())

            self.generated_data.append(s_data)
            
            simu_ok_num += 1
            
            print('**Simulation succeed!!!**[ succeed number = %s (dataset lineno %s)]**, cost=%s' % (simu_ok_num, lineno,time.time()-start_time))
            

                
    def saved_data_to_file(self, filename=None):
        if filename is None:
            print(self.norm_state)
            filename =  'data/test_data_on_%s.pickle' % str(self.norm_state)
        f = open(filename, 'wb')
        pickle.dump(self.generated_data, f)
        f.close()
        
    def load_data_from_file(self, filename):
        if os.path.isfile(filename):
            print('file exists, loading ...')
            with open(filename, 'rb') as f:
                self.generated_data = pickle.load(f)
                print('--ok')
        else:
            print('no file [%s] exists'%filename)
        

# Simulation_config,MutualisticInteraction_config, GeneRegulatory_config
@hydra.main(config_name="Simulation_config", version_base='1.2', config_path='configs')
def main(cfg):
    generate_data = GenerateData(cfg, eq_filename='data/dataset_test3.csv')
    generate_data.generate_data()
    generate_data.saved_data_to_file()
    
    #generate_data.load_data_from_file('data/dataset_5000.csv_norm_stateFalse.pkl')
    #print(generate_data.generated_data[0])
    
    #from sklearn.utils import shuffle
    #print(shuffle(generate_data.generated_data)[0])


if __name__ == "__main__":
    main()
