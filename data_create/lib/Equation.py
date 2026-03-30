import numpy as np

import pandas as pd
import sympy as sp
import numexpr as ne
import sympytorch

import torch
from torch import nn


class Equation(nn.Module):
    def __init__(self, eq_str, eq_var, eq_length):
        super(Equation, self).__init__()
        self.eq_str = eq_str
        self.eq_var = eq_var
        self.eq_length = eq_length

        self.eq_func = self.make_computable_eq_from_str(self.eq_str, self.eq_var)
    
    def make_computable_eq_from_str(self, eq_str, eq_var):
        eq_sympy = sp.sympify(eq_str)
        # eq = sp.lambdify(eq_var, eq_sympy, 'numpy')

        eq = sympytorch.SymPyModule(expressions=[eq_sympy])

        return eq

        
    """
    def make_computable_eq_from_str(self, eq_str, eq_var):
        #eq_sympy = sp.sympify(eq_str)
        #eq = sp.lambdify(eq_var, eq_sympy, 'numpy')
        #return eq
        
        infix = eq_str
        
        def wrapped_numexpr_fn(x):
            #t, x = np.array(t), np.array(x)
            local_dict = {}
            dimension = len(x)
            for d in range(dimension): 
                local_dict[eq_var[d]] = x[d]
    
            #local_dict["t"] = t[:]
            #local_dict.update(extra_local_dict)
            # predicted_dim = len(infix.split('|'))
            try:
                vals = ne.evaluate(infix, local_dict=local_dict).reshape(-1,1)   
            except Exception as e:
                # print(e)
                # print("problem with tree", infix)
                # print(traceback.format_exc())
                vals = np.array([np.nan for _ in range(x[0].shape[0])])#.reshape(-1, 1).repeat(predicted_dim, axis=1)
            return vals

        return wrapped_numexpr_fn
    """
        

    def generate_data(self, input_data):
        #print('in generate_data xx')
        #print(input_data.size())
        input_params = {}
        for eq_var_i in range(len(self.eq_var)):
            input_params[self.eq_var[eq_var_i]] = input_data[:, eq_var_i]
        res = self.eq_func(**input_params)
        # res = self.eq_func(*[xi for xi in torch.transpose(input_data, 0, 1)])
        #res = torch.from_numpy(self.eq_func([xi for xi in torch.transpose(input_data, 0, 1).numpy()]))
        #print(res)
        if type(res) is float or type(res) is int or len(res) == 1:  # dealing with constant
            return torch.Tensor([res]).repeat(input_data.size(0)).view(-1, 1)
        elif type(res) is torch.Tensor:
            return res.view(-1, 1)
        else:
            print('wrong type [%s]'%type(res))
            exit(1)


if __name__ == "__main__":
    eq_str = 'x_1**2+x_2'
    eq_var = ['x_1', 'x_2']
    eq_length = 10
    eq = Equation(eq_str, eq_var, eq_length)

    #数据，len(x)与数据变量数有关
    #x1 = torch.Tensor([[1],
    #                   [2],
    #                   [3]])
    x1_x2 = torch.Tensor([[1, 4],
                          [2, 5],
                          [3, 6]])

    print('result:',eq.generate_data(x1_x2))


