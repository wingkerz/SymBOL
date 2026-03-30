import torch
from torch_scatter import scatter_sum
from data_create.lib.Equation import Equation


class DynamicsEquation:
    def __init__(self, dim, f_eq_str, g_eq_str):
        self.f_eq_str = f_eq_str
        self.g_eq_str = g_eq_str
        self.dim = dim

        if self.f_eq_str is None:
            self.f_eq = None
        else:
            f_eq_var = ["x_%s" % (dd + 1) for dd in range(dim)]
            self.f_eq = Equation(f_eq_str, f_eq_var, len(f_eq_str))

        g_eq_var = ["x_%s" % (dd + 1) for dd in range(dim * 2)]
        self.g_eq = Equation(g_eq_str, g_eq_var, len(g_eq_str))

    def generate_diff_data(self, state_input, adj):
        # state_input : [N, dim]
        # adj : [2, #edges]
        row, col = adj
        if self.f_eq is None:
            self_diff = torch.zeros_like(state_input)
        else:
            # [N, dim]
            self_diff = self.f_eq.generate_data(state_input)
        # [#edges, dim]
        interact_input = torch.cat([state_input[col], state_input[row]], dim=-1)
        interact_diff = self.g_eq.generate_data(interact_input)
        # [N, dim]
        total_diff_signal = self_diff + scatter_sum(interact_diff, col, dim=0, dim_size=state_input.size(0))
        return total_diff_signal, (state_input, self_diff), (interact_input, interact_diff)


if __name__ == "__main__":
    dim = 1
    # f_eq_str = '(x_1 - 1.567) / x_1'
    f_eq_str = '1.567'
    g_eq_str = 'x_1 - x_2 * (x_1 + x_2) + x_2'
    de = DynamicsEquation(dim, f_eq_str, g_eq_str)

    x1 = torch.Tensor([[1],
                       [2],
                       [3]])

    state_input = x1
    adj = torch.Tensor([[0, 1], [1, 2]]).long()

    res = de.generate_diff_data(state_input, adj)
    print(res)
