import torch


class InitCondition:
    def __init__(self, data_train, N, dim, num_sampling=1, constraint=None, fix_init=None):
        self.N = N
        self.dim = dim
        self.constraint = constraint
        self.fix_init = fix_init
        self.data_train = data_train
        self.sampled_init_condition = self.sampling_init_condition(num_sampling)

    def resampling(self, num_sampling=1):
        self.sampled_init_condition = self.sampling_init_condition(num_sampling)

    # def sampling_init_condition(self, num_sampling=1):
    #     # 1. 去掉第一列（假设第一列是索引或其他不需要的数据）
    #     processed_data = self.data_train[:, 1:]  # 取所有行，第1列到最后
    #
    #     # 2. 转换为 PyTorch Tensor 并调整形状
    #     tensor_data = torch.tensor(processed_data, dtype=torch.float32)  # 转换为 float32 Tensor
    #     tensor_data = tensor_data.unsqueeze(0)  # 添加 batch 维度 -> [1, N, 3]
    #     return tensor_data

    def sampling_init_condition(self, num_sampling=1):
            # 1. 分离第一列和其他列
            first_column = self.data_train[:, 0]  # 第一列数据 [N,]
            processed_data = self.data_train[:, 1:]  # 其余列数据 [N, 3]

            # 2. 转换为 PyTorch Tensor 并调整形状
            tensor_data = torch.tensor(processed_data, dtype=torch.float32)  # [N, 3]
            tensor_data = tensor_data.unsqueeze(0)  # 添加 batch 维度 -> [1, N, 3]

            # 3. 返回处理后的张量和第一列数据
            return tensor_data, first_column
        # return torch.randn(num_sampling, self.N, self.dim)

    def fix_init_condition(self, a):
        self.sampled_init_condition = a


if __name__ == "__main__":
    N = 80000
    dim = 3
    init_cond = InitCondition(N, dim, num_sampling=1)
    print(init_cond.sampled_init_condition)
