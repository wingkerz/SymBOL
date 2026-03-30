import time
import torch
from torch import nn
from torch_scatter import scatter_sum
from data_create.lib.Equation import Equation
#from Equation import Equation



class DynamicsEquation(nn.Module):
    def __init__(self, dim, f_eq_str, g_eq_str):
        super(DynamicsEquation, self).__init__()
        self.f_eq_str = f_eq_str
        self.g_eq_str = g_eq_str
        self.dim = dim

        if self.f_eq_str is None:
            self.f_eq = None
        else:
            # 
            if type(self.f_eq_str) is list:
                f_eq_var = ["x_1_%s" % dd for dd in range(dim)]

                self.f_eq = nn.ModuleList([Equation(f_eq_str_i, f_eq_var, len(f_eq_str_i)) for f_eq_str_i in self.f_eq_str])
            else:
                f_eq_var = ["x_%s" % (dd + 1) for dd in range(dim)]
                self.f_eq = Equation(f_eq_str, f_eq_var, len(f_eq_str))
        
        if self.g_eq_str is None:
            self.g_eq = None
        else:
            # g_eq_var = ["x_%s" % (dd + 1) for dd in range(dim * 2)]
            if type(self.g_eq_str) is list:
                #进行高阶的实验需要修改的地方
                g_eq_var = ["x_1_%s" % (dd) for dd in range(dim)] + ["x_2_%s" % (dd) for dd in range(dim)]

                # g_eq_var = ["x_1_%s" % (dd) for dd in range(dim)] + ["x_2_%s" % (dd) for dd in range(dim)] + ["x_3_%s" % (dd) for dd in range(dim)]
                self.g_eq = nn.ModuleList([Equation(g_eq_str_i, g_eq_var, len(g_eq_str_i)) for g_eq_str_i in self.g_eq_str])
            else:
                # 二阶的
                g_eq_var = ["x_%s" % (dd + 1) for dd in range(dim * 2)]
                # 三阶的
                # g_eq_var = ["x_%s" % (dd + 1) for dd in range(dim * 3)]

                self.g_eq = Equation(g_eq_str, g_eq_var, len(g_eq_str))

    def generate_diff_data(self, state_input, adj):
        #print('in generate_diff_data')
        #exit(1)
        # state_input : [N, dim]
        # adj : [2, #edges] or [3, #edges]
        #print(type(self.f_eq),type(self.g_eq), len(self.f_eq), len(self.g_eq))
        #exit(1)
        if len(adj) == 2:
            row, col = adj
            edge_weights = torch.ones_like(col)
        else:
            row, col, edge_weights = adj
        row = row.long()
        col = col.long()
        if self.f_eq is None:
            self_diff = torch.zeros_like(state_input)
        else:
            # [N, dim]
            #print('self.f_eq.eq_str=',self.f_eq.eq_str)
            if type(self.f_eq) is nn.ModuleList:
                self_diff = torch.cat([f_eq_i.generate_data(state_input) for f_eq_i in self.f_eq], dim=-1)
            else:
                self_diff = self.f_eq.generate_data(state_input)
                
        # [#edges, dim]
        interact_input = torch.cat([state_input[col], state_input[row]], dim=-1)
        if self.g_eq is None:
            interact_diff = torch.zeros_like(interact_input)
        else:
            if type(self.g_eq) is nn.ModuleList:
                interact_diff = torch.cat([g_eq_i.generate_data(interact_input) for g_eq_i in self.g_eq], dim=-1)
            else:
                interact_diff = self.g_eq.generate_data(interact_input)
        # [N, dim]
        total_diff_signal = self_diff + scatter_sum(interact_diff * edge_weights.view(-1, 1), col, dim=0, dim_size=state_input.size(0))
        return total_diff_signal, (state_input, self_diff), (interact_input, interact_diff)

    def generate_diff_data_1(self, state_input):
        # state_input : [N, dim]
        # adj : [2, #edges]
        if self.f_eq is None:
            self_diff = torch.zeros_like(state_input)
        else:
            # [N, dim]
            #print('self.f_eq.eq_str=',self.f_eq.eq_str)
            if type(self.f_eq) is nn.ModuleList:
                self_diff = torch.cat([f_eq_i.generate_data(state_input) for f_eq_i in self.f_eq], dim=-1)
            else:
                self_diff = self.f_eq.generate_data(state_input)
        return [state_input, self_diff]

    def generate_diff_data_2(self, state_input):
        # state_input : [N, dim]
        # adj : [2, #edges]
        interact_input = state_input
        if self.g_eq is None:
            interact_diff = torch.zeros_like(interact_input)
        else:
            if type(self.g_eq) is nn.ModuleList:
                interact_diff = torch.cat([g_eq_i.generate_data(interact_input) for g_eq_i in self.g_eq], dim=-1)
            else:
                interact_diff = self.g_eq.generate_data(interact_input)
        return [interact_input, interact_diff]


if __name__ == "__main__":
    dim = 1
    
    start_time = time.time()
    for _ in range(10000):
      # f_eq_str = '(x_1 - 1.567) / x_1'
      f_eq_str = '1.567'
      g_eq_str = 'x_1 - x_2 * (x_1 + x_2) + x_2'
      de = DynamicsEquation(dim, f_eq_str, g_eq_str)
  
      #x1 = torch.Tensor([[1],
      #                   [2],
      #                   [3]])
      x1 = torch.randn(1000,1)
  
      state_input = x1
      adj = torch.cat([torch.linspace(0,998,999).view(1,-1), torch.linspace(1,999,999).view(1,-1)], dim=0).long()
      
      
  
      res = de.generate_diff_data(state_input, adj)
    print(time.time() - start_time)
