import torch


class InitCondition:
    def __init__(self, N, dim, lbs=None, ubs=None, num_sampling=1, constraint=None, fix_init=None):
        self.N = N
        self.dim = dim
        self.constraint = constraint
        self.fix_init = fix_init
        if lbs is None:
            self.lbs = torch.ones(dim) * -10.
        else:
            self.lbs = lbs  # tensor([dim,])
        if ubs is None:
            self.ubs = torch.ones(dim) * 10.
        else:
            self.ubs = ubs  # tensor([dim,])
        self.sampled_init_condition = self.sampling_init_condition(num_sampling)

    def resampling(self, num_sampling=1):
        self.sampled_init_condition = self.sampling_init_condition(num_sampling)

    def sampling_init_condition(self, num_sampling=1):
        if self.constraint is None:
            # x0_tensor = torch.tensor(list_x0, dtype=torch.float64).view(self.N, self.dim)
            # # print(x0_tensor)
            # return x0_tensor.repeat(num_sampling, 1, 1)
            # return torch.rand(num_sampling, self.N, self.dim) * 2  # Uniform(0, 2)
            return torch.rand(num_sampling, self.N, self.dim) * (self.ubs - self.lbs) + self.lbs
        else:
            res = torch.rand(num_sampling, self.N, self.dim) * 1

            # res = torch.rand(num_sampling, self.N, self.dim) * (self.ubs - self.lbs) + self.lbs
            valid_dims, constraint_type = self.constraint
            if constraint_type == 'sum_is_one':
                res[:, :, valid_dims == 0] = 1. - torch.sum(res * valid_dims.view(1,1,self.dim), dim=-1).view(num_sampling, self.N, -1)
            elif constraint_type == 'fix':
                for ii in range(num_sampling):
                    res[ii, :, valid_dims == 0] = self.fix_init
            else:
                print("unknown constraint_type [%s]"%constraint_type)
            return res
        # return torch.randn(num_sampling, self.N, self.dim)

    def fix_init_condition(self, a):
        self.sampled_init_condition = a


if __name__ == "__main__":
    N = 80000
    dim = 3
    init_cond = InitCondition(N, dim, num_sampling=1)
    print(init_cond.sampled_init_condition)
