import time
import torch
from torch import nn
from torch_scatter import scatter_sum
from data_create.lib.Equation import Equation


class DynamicsEquation(nn.Module):
    def __init__(self, dim, f_eq_str, g_eq_str, s_eq_str=None):
       
        super(DynamicsEquation, self).__init__()
        self.dim = dim

      
        f_str = f_eq_str[0] if isinstance(f_eq_str, list) else f_eq_str
        f_eq_var = ["x_1_%s" % dd for dd in range(dim)]  
        self.f_eq = Equation(f_str, f_eq_var, len(f_str)) if f_str else None

        g_str = g_eq_str[0] if isinstance(g_eq_str, list) else g_eq_str
        g_eq_var = (["x_1_%s" % (dd) for dd in range(dim)] +["x_2_%s" % (dd) for dd in range(dim)])
        
        self.g_eq = Equation(g_str, g_eq_var, len(g_str)) if g_str else None

        s_str = s_eq_str[0] if isinstance(s_eq_str, list) else s_eq_str
        s_eq_var = ["x_%s" % (dd + 1) for dd in range(dim)]
        self.s_eq = Equation(s_str, s_eq_var, len(s_str)) if s_str else None

    def generate_diff_data(self, state_input, adj):
        row, col = adj[0].long(), adj[1].long()
        edge_weights = adj[2] if adj.size(0) == 3 else torch.ones_like(col).float()

        self_diff = self.f_eq.generate_data(state_input) if self.f_eq else torch.zeros_like(state_input)

        
        
        
        interact_pair_input = torch.cat([state_input[col], state_input[row]], dim=-1)

        if self.g_eq:
            
            neighbor_processed = self.g_eq.generate_data(interact_pair_input)
        else:
            
            neighbor_processed = state_input[row]
        
        
        
        aggregated_sum = scatter_sum(neighbor_processed * edge_weights.view(-1, 1),
                                     col, dim=0, dim_size=state_input.size(0))

        
        if self.s_eq:
            interact_diff = self.s_eq.generate_data(aggregated_sum)
        else:
            interact_diff = aggregated_sum

        return self_diff + interact_diff, (state_input, self_diff), (aggregated_sum, interact_diff)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        