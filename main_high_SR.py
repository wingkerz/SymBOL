import heapq
import json
import math
import os
import re
import random
from datetime import datetime
import signal
import use_deepseek
import use_gpt
import use_localmodel
import ollama_test
import pynvml
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import numpy as np
import pandas as pd
import pickle
import csv
import multiprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torchensemble.utils.io import split_data_target
from torchensemble import SnapshotEnsembleRegressor, SoftGradientBoostingRegressor
from torchensemble.utils.logging import set_logger
import sympy as sp
from screen_pretrain_knowledge_eq import make_pop_from_file, evolve, multi_processing_eval_pop, \
    multi_processing_opt_constant, opt_constant_one_core, pset_f, pset_g, converter, tournamentSelection, \
    opt_constant_multi_core,multi_processing_opt_constant_discrete, get_parser
import search_2nd_phase.gp_tools as gp
import argparse
import utils
import use_gpt
from botorch.models import SingleTaskGP, gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll
from botorch.models.transforms.outcome import Standardize
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from botorch.utils.transforms import standardize, normalize, unnormalize
from gpytorch.constraints import Interval
from gpytorch.kernels import RBFKernel, LinearKernel, ScaleKernel, MaternKernel
from scipy.stats import pearsonr
import transformers.models.mistral.modeling_mistral as mistral_module
if not hasattr(mistral_module, 'MISTRAL_INPUTS_DOCSTRING'):
    mistral_module.MISTRAL_INPUTS_DOCSTRING = ""
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
device1 = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
parser = get_parser()
args, unknown = parser.parse_known_args() 
case_name = args.case_name
dim = int(args.dim)
add_str = args.add_str
init_f_eq_str_list = []
init_g_eq_str_list = []
init_fitness_list = []
now = datetime.now()
formatted_now = now.strftime("%Y-%m-%d-%H-%M")
save_dir = os.path.join("result_figure",args.case_name, str(args.dim),str(args.opt_flag), formatted_now, args.add_str)
class TimeoutException(Exception):
    pass
class SimplifyTimeoutException(Exception):
    pass
def timeout_handler(signum, frame):
    raise TimeoutException("Timeout")
def _parse_worker_new(f_str, g_str, pset_f, pset_g, queue):
    try:
        b1 = gp.PrimitiveTree.from_string_sympy_fast(f_str, pset_f)
        b2 = gp.PrimitiveTree.from_string_sympy_fast(g_str, pset_g)
        queue.put({"status": "success", "data": (b1, b2)})
    except Exception as e:
        queue.put({"status": "error", "msg": str(e)})
class Observations:
    def __init__(self):
        self.pop = []
        self.labels = []
    def append1(self, f_eq_str_list, g_eq_str_list, fitness_list):
        filtered_count = 0
        for ii in range(len(f_eq_str_list)):
            fitness = fitness_list[ii]
            if fitness > 10:
                print(f' pass fitness > 10 : [%s] %s | %s | %s' %(ii + 1, f_eq_str_list[ii], g_eq_str_list[ii], fitness_list[ii]))
                filtered_count += 1
                continue
            print('Appending to observations: [%s] %s | %s | %s'%(ii+1, f_eq_str_list[ii], g_eq_str_list[ii], fitness_list[ii]))
            f_eq_str = f_eq_str_list[ii]
            g_eq_str = g_eq_str_list[ii]
            b1 = gp.PrimitiveTree.from_string_sympy(f_eq_str, pset_f) 
            b2 = gp.PrimitiveTree.from_string_sympy(g_eq_str, pset_g)
            self.pop.append([(b1, b2), fitness_list[ii]])
            self.labels.append(fitness_list[ii])
            while len(self.pop) > 2000:
                max_index = np.argmax(self.labels)
                del self.pop[max_index]
                del self.labels[max_index]
    def append(self, f_eq_str_list, g_eq_str_list, fitness_list):
        filtered_count = 0
        count_bad_fitness = 0
        count_timeout = 0
        count_parse_error = 0
        valid_indices = []
        for i, fit in enumerate(fitness_list):
            if fit >= 10:
                count_bad_fitness += 1
            else:
                valid_indices.append(i)
        for ii in valid_indices:
            print('Appending to observations: [%s] %s | %s | %s'%(ii+1, f_eq_str_list[ii], g_eq_str_list[ii], fitness_list[ii]))
            f_eq_str = f_eq_str_list[ii]
            g_eq_str = g_eq_str_list[ii]
            fitness = fitness_list[ii]
            q = multiprocessing.Queue()
            p = multiprocessing.Process(
                target=_parse_worker_new, 
                args=(f_eq_str, g_eq_str, pset_f, pset_g, q)
            )
            try:
                p.start()
                p.join(timeout=2)
                b1, b2 = None, None
                if p.is_alive():
                    p.terminate() 
                    p.join(timeout=0.1)
                    if p.is_alive():
                        p.kill() 
                        p.join() 
                    count_timeout += 1
                    q.close()
                    q.join_thread()
                    continue 
                else:
                    if not q.empty():
                        try:
                            res = q.get_nowait()
                            if res["status"] == "success":
                                b1, b2 = res["data"]
                            else:
                                count_parse_error += 1
                                q.close()
                                q.join_thread()
                                continue
                        except Exception:
                            count_parse_error += 1
                            q.close()
                            q.join_thread()
                            continue
                    else:
                        count_parse_error += 1
                        q.close()
                        q.join_thread()
                        continue
            except Exception as e:
                if p.is_alive():
                    p.kill()
                    p.join()
            finally:
                try:
                    q.close()
                    q.join_thread()
                except:
                    pass
            self.pop.append([(b1, b2), fitness])
            self.labels.append(fitness)
            if len(self.pop) > 2500:
                max_index = np.argmax(self.labels)
                del self.pop[max_index]
                del self.labels[max_index]
    def str_from(self, ind_in_pop_f, simplify_flag=False, timeout_seconds=2):
        raw_str_obj = gp.TreeToDeadStr(ind_in_pop_f, converter)
        raw_str = str(raw_str_obj)
        if simplify_flag:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            try:
                simplified_expr = sp.simplify(raw_str)
                return str(simplified_expr)
            except TimeoutException:
                return raw_str
            except Exception as e:
                return raw_str
            finally:
                signal.alarm(0)
        else:
            return raw_str
    def get_pop_str(self, pop, simplify_flag=False):
        pop_str = []
        for ii in tqdm(range(len(pop))):
            f_expr, g_expr = pop[ii][0]
            score = pop[ii][1]
            f_eq_str = self.str_from(f_expr, simplify_flag=simplify_flag)
            g_eq_str = self.str_from(g_expr, simplify_flag=simplify_flag)
            pop_str.append([(f_eq_str, g_eq_str), score])
        return pop_str
    def return_best(self):
        return [self.pop[np.argmin(self.labels)]]
    def show(self):
        print("+++++++++++++++++++++++++++")
        print("pop:")
        print(self.get_pop_str(self.pop))
        print("labels:")
        print(self.labels)
        print("+++++++++++++++++++++++++++")
def make_passages(pop_str):
    passages = []
    for ii in range(len(pop_str)):
        f_eq_str = pop_str[ii][0][0]
        g_eq_str = pop_str[ii][0][1]
        passage = 'fx = %s, gx = %s.' % (replace_C_to_one(replace_numbers(f_eq_str)), replace_C_to_one(replace_numbers(g_eq_str)))
        passages.append(passage)
    return passages
def get_embeddings(model, passages, passage_prefix, max_length=32768, batch_size=10):   
    passage_embeddings = []
    start_idx = 0
    while start_idx < len(passages):
        passage_embeddings_ = model.encode(passages[start_idx:start_idx+batch_size], instruction=passage_prefix, max_length=max_length)
        passage_embeddings.append(passage_embeddings_.cpu())
        start_idx += batch_size
    passage_embeddings = torch.cat(passage_embeddings, dim=0)
    passage_embeddings = F.normalize(passage_embeddings, p=2, dim=1)
    passage_embeddings = passage_embeddings.detach().cpu().numpy()
    return passage_embeddings
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim1, hidden_dim2, hidden_dim3,out_dim):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.linear3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.linear4 = nn.Linear(hidden_dim3, out_dim)
        self.dropout = nn.Dropout(p=0.1)
    def forward(self, data, num_repeat=1):
        data = data.view(data.size(0), -1)
        if num_repeat > 1:
            outputs = []
            for i in range(num_repeat):
                output = F.relu(self.linear1(data))
                output = self.dropout(output)
                output = F.relu(self.linear2(output))
                output = self.dropout(output)
                output = F.relu(self.linear3(output))
                output = self.dropout(output)
                output = self.linear4(output)
                outputs.append(output.unsqueeze(0))
            outputs = torch.cat(outputs, dim=0)
            return outputs.mean(0), outputs.std(0)
        else:
            output = F.relu(self.linear1(data))
            output = self.dropout(output)
            output = F.relu(self.linear2(output))
            output = self.dropout(output)
            output = F.relu(self.linear3(output))
            output = self.dropout(output)
            output = self.linear4(output)
            return output
class OurDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.features = None
        self.labels = None
    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    def __len__(self):
        return self.features.size(0)
    def append(self, embeddings, labels):
        if self.features is None and self.labels is None:
            self.features = torch.from_numpy(embeddings).detach().to(device1) 
            self.labels = torch.tensor(labels).view(-1, 1).detach().to(device1)
        else:
            self.features = torch.cat([self.features, torch.from_numpy(embeddings).detach().to(device1)],dim=0)
            self.labels = torch.cat([self.labels, torch.tensor(labels).view(-1, 1).detach().to(device1)],dim=0)
class CosineKernel():
    def forward(self, x1, x2, diag=False, **params):
        x1_norm = F.normalize(x1, p=2, dim=-1)
        x2_norm = F.normalize(x2, p=2, dim=-1)
        if diag:
            return (x1_norm * x2_norm).sum(dim=-1)
        else:
            return x1_norm @ x2_norm.transpose(-2, -1)
class DeepSurrogate_GP3:
    def __init__(self, llm_model):
        self.y_std = 1
        self.y_mean = 0
        self.llm_model = llm_model
        embedding_dim = 4096
        self.embedding_dim = embedding_dim
        self.gp_model = None
        self.dataset = OurDataset()
        self.num_epoch = 100  
        self.batch_size = 100
        self.criterion = nn.L1Loss(reduction='mean')
        self.train_x = None
        self.train_y = None
        self.bounds = None
    def append_data(self, passages, labels, passage_prefix=""):
        embeddings = get_embeddings(self.llm_model, passages, passage_prefix)
        self.dataset.append(embeddings, labels)
    def create_new_dataset(self, passages, labels, passage_prefix=""):
        embeddings = get_embeddings(self.llm_model, passages, passage_prefix)
        dataset = OurDataset()
        dataset.append(embeddings, labels)
        return dataset
    def renew_regressor(self, train_x, train_y):
        print('Creating Gaussian Process model...')
        kernel3 = MaternKernel(nu=1.5) 
        kernel1 = RBFKernel()
        model = SingleTaskGP(train_X=train_x, train_Y=train_y, covar_module=kernel3)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        print('**done!')
        return model, mll
    def train(self, passages, labels, passage_prefix="", step_i=None):
        self.dataset = self.create_new_dataset(passages, labels, passage_prefix)
        inputs = torch.stack([sample[0] for sample in self.dataset])
        targets = torch.tensor([[sample[1]] for sample in self.dataset], dtype=torch.float32)
        self.train_x = inputs
        self.y_mean = targets.mean()  
        self.y_std = targets.std()  
        self.train_y = (targets - self.y_mean) / self.y_std  
        self.bounds = torch.stack([
            torch.min(self.train_x, dim=0)[0],
            torch.max(self.train_x, dim=0)[0]
        ])
        self.gp_model, mll = self.renew_regressor(self.train_x, self.train_y)
        fit_gpytorch_mll(mll)
        print("Gaussian Process model trained successfully!")
    def evaluate_and_predict(self, passages, labels, passage_prefix="", num_repeat=1):
        if labels is None:
            labels = [0. for i in range(len(passages))]
        dataset_eval = self.create_new_dataset(passages, labels, passage_prefix)
        inputs = torch.stack([sample[0] for sample in dataset_eval])
        targets = torch.tensor([[sample[1]] for sample in dataset_eval], dtype=torch.float32)
        self.gp_model.eval()
        with torch.no_grad():
            posterior = self.gp_model.posterior(inputs)
            output_mean = posterior.mean
            output_std = posterior.variance.sqrt()
            output_mean = output_mean * self.y_std + self.y_mean
            output_std = output_std * self.y_std
            targets = targets.to(output_mean.device)
            loss = self.criterion(output_mean, targets)
            return float(loss), output_mean, output_std, targets
    def sample_predictions(self, test_x, num_samples=10):
        labels = [0. for i in range(len(test_x))]
        dataset = self.create_new_dataset(test_x, labels)
        inputs = torch.stack([sample[0] for sample in dataset])
        posterior = self.gp_model.posterior(inputs)
        samples = posterior.sample(sample_shape=torch.Size([num_samples]))
        return samples
class AcquisitionFunction:
    def __init__(self, curr_best):
        self.curr_best = curr_best.to(device1)
        self.eta = max(0.1 * self.curr_best.item(), 1e-8) if self.curr_best.item() > 0 else 1e-9
        self.distribution = torch.distributions.Normal(torch.Tensor([0.0]).to(self.curr_best.device),
                                   torch.Tensor([1.0]).to(self.curr_best.device))
    def EI(self, mean, std, complexity=None):
        Z = (self.curr_best - mean - self.eta) / std
        ei = (self.curr_best - mean - self.eta) * self.distribution.cdf(Z) + std * torch.exp(
            self.distribution.log_prob(Z))
        return ei / complexity if complexity is not None else ei
    def UCB(self, mean, std, kappa=2.0):
        return mean + kappa * std
    def LCB(self, mean, std, train_num):
        delta = 0.05
        D_size = 100
        beta_t = 2 * math.log(D_size * train_num ** 2 * math.pi ** 2 / (6 * delta))
        kappa = math.sqrt(beta_t)  
        return mean - kappa * std
    def Maxstd(self, mean, std):
        return std
    def TS(self, model, test_x, num_samples=10, minimize=True):
        posterior_samples = model.sample_predictions(test_x, num_samples)
        selected_sample = posterior_samples[torch.randint(num_samples, (1,))].squeeze(0)
        selected_sample = selected_sample * model.y_std + model.y_mean
        return selected_sample.squeeze(-1) if minimize else -selected_sample.squeeze(-1)
    def random(self, size):
        return torch.rand(size)
class OptAcqFunc:
    def __init__(self,train_num):
        self.name = 'mixed_acquisition'
        self.curr_best = None  
        self.selection_log = []  
        self.train_num = train_num + 1
    def opt(self, surrogate,passage,mean, std, complexity, topk=10):
        device = mean.device
        num_points = mean.shape[0]
        available_mask = torch.ones(num_points, dtype=torch.bool, device=device)       
        selected_indices = []
        self.selection_log = []    
        ei_values = AcquisitionFunction(self.curr_best).EI(mean, std, complexity)
        ei_values[~available_mask] = -float('inf')
        ei_scores, ei_indices = torch.topk(ei_values, k=2, largest=True)
        for idx, score in zip(ei_indices.cpu().numpy(), ei_scores.cpu().numpy()):
            self.selection_log.append({
                'index': idx,
                'strategy': 'EI',
                'score': float(score),
                'mean': mean[idx].item(),
                'std': std[idx].item()
            })
        selected_indices.extend(ei_indices.tolist())
        available_mask[ei_indices] = False  
        ucb_values = AcquisitionFunction(self.curr_best).LCB(mean, std,self.train_num)
        ucb_values[~available_mask] = float('inf')
        ucb_scores, ucb_indices = torch.topk(ucb_values, k=2, largest=False)
        for idx, score in zip(ucb_indices.cpu().numpy(), ucb_scores.cpu().numpy()):
            self.selection_log.append({
                'index': idx,
                'strategy': 'LCB',
                'score':  float(score),
                'mean': mean[idx].item(),
                'std': std[idx].item()
            })
        selected_indices.extend(ucb_indices.tolist())
        available_mask[ucb_indices] = False
        ts_scores = AcquisitionFunction(self.curr_best).TS(
            model=surrogate,
            test_x=passage,
            num_samples=5,
            minimize=True
        )
        ts_scores[~available_mask] = float('inf')
        ts_values, ts_indices = torch.topk(ts_scores, k=2, largest=False)
        for idx, score in zip(ts_indices.cpu().numpy(), ts_values.cpu().numpy()):
            self.selection_log.append({
                'index': idx,
                'strategy': 'TS',
                'score': float(score),
                'mean': mean[idx].item(),
                'std': std[idx].item()
            })
        selected_indices.extend(ts_indices.tolist())
        available_mask[ts_indices] = False      
        maxstd_values = std.clone()  
        maxstd_values[~available_mask] = -float('inf')  
        maxstd_scores, maxstd_indices = torch.topk(maxstd_values, k=2, largest=True)
        for idx, score in zip(maxstd_indices.cpu().numpy(), maxstd_scores.cpu().numpy()):
            self.selection_log.append({
                'index': idx,
                'strategy': 'MaxStd',
                'score': float(score),
                'mean': mean[idx].item(),
                'std': std[idx].item()
            })
        selected_indices.extend(maxstd_indices.tolist())
        available_mask[maxstd_indices] = False  
        remaining_indices = torch.where(available_mask)[0]
        if len(remaining_indices) >= 2:
            random_indices = remaining_indices[torch.randperm(len(remaining_indices))[:2]]
        else:
            random_indices = remaining_indices  
        for idx in random_indices.cpu().numpy():
            self.selection_log.append({
                'index': idx,
                'strategy': 'Random',
                'score': float('1.0'),
                'mean': mean[idx].item(),
                'std': std[idx].item()
            })
        selected_indices.extend(random_indices.tolist())
        selected_indices = torch.tensor(selected_indices, device=device)
        self.selection_log.sort(key=lambda x: x['index'])
        assert len(selected_indices) == len(torch.unique(selected_indices)), "存在重复选择的点"
        assert len(selected_indices) == min(topk, num_points), f"选择的点数({len(selected_indices)})不等于要求({topk})"
        return selected_indices
    def print_selection_details(self):
        print("\n=== Selection Details ===")
        print(f"{'Index':<8}{'Strategy':<10}{'Score':<15}{'Mean':<15}{'Std':<15}")
        for item in sorted(self.selection_log, key=lambda x: x['index']):
            score = item['score']
            score_str = f"{score:.4f}" if isinstance(score, (int, float)) and not np.isnan(score) else "N/A"
            print(f"{item['index']:<8}{item['strategy']:<10}{score_str:<15}{item['mean']:<15.4f}{item['std']:<15.4f}")
def check_brackets(expression):
    stack = []
    for char in expression:
        if char == '(':
            stack.append(char)
        elif char == ')':
            if not stack:
                return False
            stack.pop()
    return not stack
allowed_operators = ['+', '-', '*', '/', 'exp', 'log', 'sin', 'cos', 'tan', 'abs']
def check_operators(expression):
    pattern = r'\b(?:' + '|'.join(re.escape(op) for op in allowed_operators) + r')\b'
    all_operators = re.findall(pattern, expression)
    illegal_operators = re.findall(r'\b\w+\b', expression)
    illegal_operators = [op for op in illegal_operators if op not in allowed_operators]
    if illegal_operators:
        return False, f"illlegenl: {', '.join(illegal_operators)}"
    return True, "lengel"
def replace_small_numbers(text):
    def is_small_number(match):
        num_str = match.group()
        try:
            num = float(num_str)
            return str(0) if abs(num) < 0.0099 else num_str
        except ValueError:
            return num_str
    pattern = r'-?\d+\.?\d*(?:[eE][-+]?\d+)?'
    obs = Observations()
    expression = obs.str_from(re.sub(pattern, is_small_number, text), simplify_flag=True)
    if expression == "nan" or expression == "zoo":
      expression ="0.01"
    return expression
def replace_numbers(expression): 
    expression = expression.replace("- 1", "+ 1").replace(" ", "")
    expression = re.sub(r'\d+\.\d+e-\d+', '0.00001', expression)
    expression = re.sub(r'\d+\.\d+e\d+', '10', expression)
    expression = re.sub(r'\d+\.\d+e+\d+', '10', expression)
    decimals = re.findall(r'\d+\.\d+', expression)
    for decimal in decimals:
        value = float(decimal)
        if value < 0.001:
            expression = expression.replace(decimal, '0')
    obs = Observations()
    expression = obs.str_from(expression.replace("^", "**"), simplify_flag=False)
    expression = "".join(expression.split())
    expression = add_parentheses_around_denominator(expression)
    result = []
    i = 0
    while i < len(expression):
        if expression[i].isdigit() or (
                expression[i] == '.' and i + 1 < len(expression) and expression[i + 1].isdigit()):
            num_start = i
            while i < len(expression) and (expression[i].isdigit() or expression[i] == '.'):
                i += 1
            if num_start > 0 and (expression[num_start - 1] == 'x'):
                result.append(expression[num_start:i])
            elif num_start > 0 and (expression[num_start - 1] == '^' or (expression[num_start - 2:num_start] == '**')):
                num_str = expression[num_start:i]
                try:
                    num_value = int(num_str)
                    if num_value in [-2, -1, 0, 1, 2, 3, 4, 5]:
                        result.append(num_str)  
                    else:
                        result.append('C')  
                except ValueError:
                    result.append(num_str)
            else:
                result.append('C')  
        else:
            result.append(expression[i])
            i += 1
    expression = ''.join(result)
    def replace_floats(match):
        num = float(match.group(1))  
        rounded_num = round(num, 1)  
        return f"**{rounded_num}"  
    expression = re.sub(r'\*\*(\d+\.\d+)', replace_floats, expression)
    expression = expression.replace("Ce", "C")
    expression = expression.replace("C**2", "C")
    expression = expression.replace("Ce-C", "0")
    expression = expression.replace("CeC", "1")
    expression = re.sub(r'([a-zA-BD-Z])(?!C)', lambda match: match.group(1).lower(), expression)
    expression = re.sub(r'(?<!C\*)\bx(\d+)', r'C*x\1', expression)
    expression = re.sub(r'(?<!C\*)\b(sin|cos|tan|log|abs|exp|tanh|sqrt)\b', r'C*\1', expression)
    if expression == "0" or expression == "1" or expression == "zoo" or expression == "nan":
       expression = "C"
    return expression
def add_parentheses_around_denominator(expr):
    expr_list = list(expr)
    i = 0
    while i < len(expr_list):
        if expr_list[i] == '/':
            start = i + 1
            if start < len(expr_list) and expr_list[start] == '(':
                i += 1
                continue
            j = start
            paren_count = 0
            while j < len(expr_list):
                if expr_list[j] == '(':
                    paren_count += 1
                elif expr_list[j] == ')':
                    paren_count -= 1
                elif paren_count == 0 and expr_list[j] in '+-':
                    break
                j += 1
            end = j
            expr_list.insert(end, ')')
            expr_list.insert(start, '(')
            i = end + 2
        else:
            i += 1
    return ''.join(expr_list)
def replace_C(input_string):
    input_string = input_string.replace("^", "**")
    input_string = input_string.replace("Ce-C", "0")
    input_string = input_string.replace("CeC", "1")
    input_string = input_string.replace("-C**2", "C")
    input_string = input_string.replace("C**2", "C")
    input_string = input_string.replace("C**-1", "C")
    input_string = input_string.replace("C**3", "C")
    input_string = input_string.replace("-C**3", "C")
    input_string = input_string.replace("C**4", "C")
    i = 0
    result = ""
    while i < len(input_string):
        char = input_string[i]
        if char == 'C':
            if i > 0 and (input_string[i - 1] == '^' or (i > 1 and input_string[i - 2:i] == '**')):
                result += str(random.uniform(-2, 2))
            else:
                result += str(random.uniform(0.1, 2.5))
        else:
            result += char
        i += 1
    result = add_parentheses_around_denominator(result)
    obs = Observations()
    result = obs.str_from(result, simplify_flag=False)
    return result
def replace_C_to_one(input_string):
    input_string = input_string.replace("Ce", "C")
    input_string = input_string.replace("Ce-C", "0")
    input_string = input_string.replace("-C**2", "C")
    input_string = input_string.replace("C**2", "C")
    input_string = input_string.replace("C**-1", "C")
    input_string = input_string.replace("C**3", "C")
    input_string = input_string.replace("C**4", "C")
    result = ""
    for char in input_string:
        if char == 'C':
            result += str(1)
        else:
            result += char
    obs = Observations()
    result = add_parentheses_around_denominator(result)
    result = obs.str_from(result, simplify_flag=False)
    result = re.sub(r'((?:\*\*|\^)\s?\(?)\s?-', r'\1N', result)
    result = result.strip()
    if result.startswith('-'):
        result = result[1:]
    result = result.replace("-", "+")
    result = result.replace("N", "-")
    result = result.replace("++", "+").replace("+ +", "+")
    result = result.replace("(+", "(")
    result = result.replace("**", "^").replace(" )", ")").replace(" ", "")
    return result
def replace_C_2_24(input_string):
    result = ""
    i = 0  
    while i < len(input_string):
        char = input_string[i]
        if char == 'C':
            rand_float = random.uniform(-2, 2)
            if rand_float < 0 < i and (input_string[i - 1] == '-' or input_string[i - 2] == '-'):
                rand_float = abs(rand_float)  
                if input_string[i - 2] == '-':
                    result = result[:-2]
                    result += '+ '
            result += str(rand_float)
        else:
            result += char
        i += 1
    return result
def replace_C_fx(input_string):
    result = ""
    i = 0  
    while i < len(input_string):
        char = input_string[i]
        if char == 'C':
            if i + 1 < len(input_string) and input_string[i + 1] == '*':
                rand_float = random.choice([-2.00001,-1.00001,1.00001, 2.00001])
            else:
                rand_float = random.choice([-2.00001,-1.00001,1.00001, 2.00001])
            if rand_float < 0:
                rand_float = abs(rand_float)  
                if i > 0 and (input_string[i - 1] == '-' or input_string[i - 2] == '-'):
                    if input_string[i - 1] == '-':
                        result = result[:-1]  
                    if input_string[i - 2] == '-':
                        result = result[:-2]
                        result += '+ '
            result += str(rand_float)
        else:
            result += char
        i += 1
    return result
def replace_C_1(input_string):
    result = ""
    for char in input_string:
        if char == 'C':
            result += str(1.000000000000000)
        else:
            result += char
    return result
def get_unique_random_samples(population, k):
   
    population = [s.replace("- C", "+ C").replace("-C", "C") for s in population]
    unique_population = list(dict.fromkeys(population))
    filtered_population = [s for s in unique_population if s != "1"]  
    if len(filtered_population) <= k:
        return filtered_population
    return unique_population[:k]
def get_unique_paired_samples(population, other_list, k):
   
    if len(population) != len(other_list):
        raise ValueError("population and other_list")
    processed_pop = [s.replace("- C", "+ C").replace("-C", "C") for s in population]
    processed_other = [s.replace("- C", "+ C").replace("-C", "C") for s in other_list]
    seen = set()
    new_pop = []
    new_other = []
    for p, o in zip(processed_pop, processed_other):
        if (p, o) not in seen:
            seen.add((p, o))
            new_pop.append(p)
            new_other.append(o)
    return new_pop[:k], new_other[:k]
def tournamentSelection(population, num_selected, tournament_size=2):
   
    selected = []
    for _ in range(num_selected):
        participants = random.choices(population, k=tournament_size)
        winner = min(participants, key=lambda x: x[1])
        selected.append(winner)
    return sorted(selected, key=lambda x: x[1])[:num_selected]

def gen_use_llm_separate(pop, list_f, list_g, suggest_list, selected_pop, context_template, add_bool):
    
    pop_new = []
    fx_list_all = []
    gx_list_all = []
    context = context_template  
    fx_list = []
    gx_list = []
    fx_list = list_f
    gx_list = list_g
    os.makedirs(save_dir, exist_ok=True)
    if suggest_list:
        suggest_text = "When generating the new equation, please refer to the following suggestions."
        formatted_suggestion_list = '\n'.join(f'- {s}' for s in suggest_list)  
    else:
        suggest_text = ""
        formatted_suggestion_list = ""
    if context[-5] == '2':
       context = context[:-5] + '4' + context[-4:]
    prompts_f = use_gpt.read_prompt_from_file(context).format(fx_list = fx_list, variables = ['x0'],suggest_text=suggest_text,suggest_list = formatted_suggestion_list)
    if context[-5] == '4':
       context = context[:-5] + '2' + context[-4:]
    if context[-5] == '1':
       context = context[:-5] + '3' + context[-4:]
    prompts_g = use_gpt.read_prompt_from_file(context).format(fx_list = gx_list, variables = ['x0','x1'],suggest_text=suggest_text,suggest_list = formatted_suggestion_list)
    num = 2
    if context[-5] == '3':
        num = 1
    for j in range(num):
        print(j)
        print(prompts_f)
        print()
        ind_one_new_f = str(use_localmodel.generate_response_separate(prompts_f))
        print(ind_one_new_f)
        save_path = os.path.join(save_dir, 'result.txt')
        os.makedirs(save_dir, exist_ok=True)
        with open(save_path, "a", encoding="utf-8") as fw:
            fw.write(prompts_f)
            fw.write("\n") 
            fw.write(ind_one_new_f)  
            fw.write("\n") 
        data_f = json.loads(ind_one_new_f)
        print(prompts_g)
        print()
        ind_one_new_g = str(use_localmodel.generate_response_separate(prompts_g))
        print(ind_one_new_g)
        save_path = os.path.join(save_dir, 'result.txt')
        with open(save_path, "a", encoding="utf-8") as fw:
            fw.write(prompts_g)
            fw.write("\n") 
            fw.write(ind_one_new_g)  
            fw.write("\n") 
        data_g = json.loads(ind_one_new_g)
        selected_pop_str = pop.get_pop_str(pop.pop)
        common_keys = set(data_f.keys()) & set(data_g.keys())
        processed_list_f = [replace_numbers(f_expr) for f_expr in list_f]
        processed_list_g = [replace_numbers(g_expr) for g_expr in list_g]
        for key in common_keys:
            try:
                fx = data_f[key]
                gx = data_g[key]
                fx = replace_numbers(fx).replace("x1", "x0").replace("x2", "x0")   
                gx = replace_numbers(gx)
                if not check_brackets(fx) or not check_brackets(gx):
                    print(f"Invalid brackets in fx: {fx} or gx: {gx}")
                    continue
                if fx not in processed_list_f:
                    fx = replace_C(fx)
                    fx_list_all.append(fx)
                if gx not in processed_list_g:
                    gx = replace_C(gx)
                    gx_list_all.append(gx)
            except Exception as e:
                print(f"Error processing key aaa {key}: fx={fx}, gx={gx}. Error: {str(e)}")
                continue
    if add_bool % 5 == 0:
        fx_list_all.extend(replace_C(replace_numbers(item)) for item in list_f)
        gx_list_all.extend(replace_C(replace_numbers(item)) for item in list_g)

    i = 0
    for fx in fx_list_all:
        for gx in gx_list_all:
            try:
                b1 = gp.PrimitiveTree.from_string_sympy(fx, pset_f)
                b2 = gp.PrimitiveTree.from_string_sympy(gx, pset_g)
                pop_new.append([(b1, b2), 1e20])
                i = i+1
            except Exception as e:
                continue
    return pop_new
def suggest_use_llm(list_f, list_g, context_template, max_retries=5):

    suggest_list = []
    fx_list = list_f
    gx_list = list_g
    prompts = use_gpt.read_prompt_from_file(context_template).format(fx_list=fx_list, gx_list=gx_list)
    save_path = os.path.join(save_dir, 'result.txt')
    os.makedirs(save_dir, exist_ok=True)
    with open(save_path, "a", encoding="utf-8") as fw:
        fw.write(prompts + "\n")
    attempt = 0
    while attempt < max_retries:
        try:
            suggestions = str(use_localmodel.generate_response_suggestion(prompts))
            clean_json = re.sub(r'^```json\s*|```$', '', suggestions.strip(), flags=re.MULTILINE)
            data = json.loads(clean_json)
            for key, value in data.items():
                suggest_list.append(value)
            save_path = os.path.join(save_dir, 'result.txt')
            os.makedirs(save_dir, exist_ok=True)
            with open(save_path, "a", encoding="utf-8") as fw:
                fw.write(prompts)
                fw.write("\n") 
                fw.write("\n".join(map(str, suggest_list)))
            print(f"Successfully parsed LLM response on attempt {attempt + 1}")
            return suggest_list
        except (json.JSONDecodeError, ValueError) as e:
            attempt += 1
            print(f"Attempt {attempt} failed: JSON decode error. Retrying...")
            if attempt >= max_retries:
                print("Max retries reached. Returning empty list or handle as error.")
                return [] 
    return suggest_list
def validate_on_training_data(surrogate_model, train_passages, train_labels):
    
    loss, pred_means, pred_stds, true_labels = surrogate_model.evaluate_and_predict(
        train_passages,
        train_labels
    )
    pred_means = pred_means.cpu().numpy().flatten()
    pred_stds = pred_stds.cpu().numpy().flatten()
    true_labels = true_labels.cpu().numpy().flatten()
    abs_errors = np.abs(pred_means - true_labels)

    rel_errors = abs_errors / (np.abs(true_labels) + 1e-8)  
    from sklearn.metrics import r2_score
    r2 = r2_score(true_labels, pred_means)
    results = {
        'mae': float(np.mean(abs_errors)),
        'mse': float(np.mean(abs_errors ** 2)),
        'r2_score': r2,
        'mean_std': float(np.mean(pred_stds)),  
        'predictions': list(zip(true_labels, pred_means, pred_stds)),
        'relative_errors': rel_errors.tolist()
    }
    print(f"Validation on Training Data Results:")
    print(f"- MAE: {results['mae']:.9f}")
    print(f"- MSE: {results['mse']:.9f}")
    print(f"- R² Score: {results['r2_score']:.9f}")
    print(f"- Mean Predictive Std: {results['mean_std']:.9f}")
    return results
def plot_validation_results(results):
    true_vals = [x[0] for x in results['predictions']]
    pred_vals = [x[1] for x in results['predictions']]
    pred_errs = [2.0*x[2] for x in results['predictions']]
    plt.figure(figsize=(12, 8))
    plt.errorbar(true_vals, pred_vals, yerr=pred_errs, fmt='o', alpha=0.5, label='Predictions ±2σ')
    plt.plot([min(true_vals), max(true_vals)], [min(true_vals), max(true_vals)], 'k--', label='Perfect Prediction')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('GP True vs Predicted Values on Training Data')
    plt.legend()
    plt.grid(True)
    os.makedirs(save_dir, exist_ok=True)
    path_error = os.path.join(save_dir, f'GP_K_RBF_test.png')
    plt.savefig(path_error)
    plt.close()
def bo_one_iter(surrogate, observations, list_f, list_g, suggest_list, selected_pop, prompts_path, train_num, train_flag=True):
    if train_flag and train_num % 4 == 0:
        print("train!")
        pop_str = observations.get_pop_str(observations.pop)
        labels = observations.labels
        total_n = len(labels)
        if total_n <= 700:
            train_passages = make_passages(pop_str)
            train_labels = labels
        else:          
            sorted_indices = sorted(range(total_n), key=lambda i: labels[i])
            best_indices = sorted_indices[:50]
            recent_indices = list(range(max(0, total_n - 100), total_n))
            remaining_indices = list(set(range(total_n)) - set(best_indices) - set(recent_indices))
            random_indices = random.sample(remaining_indices, k=min(550, len(remaining_indices)))
            selected_indices = list(set(best_indices + recent_indices + random_indices))
            train_passages = make_passages([pop_str[i] for i in selected_indices])
            train_labels = [labels[i] for i in selected_indices]
        surrogate.train(train_passages, train_labels)
        validation_results = validate_on_training_data(surrogate, train_passages, train_labels)
        plot_validation_results(validation_results)
        print("train over!")
    print('generating new pop ...')
    generated_new_pop = gen_use_llm_separate(pop=observations,list_f = list_f,list_g = list_g,suggest_list = suggest_list,selected_pop=selected_pop, context_template = prompts_path,add_bool=train_num)
    fx_list = []
    gx_list = []
    passage = make_passages(observations.get_pop_str(generated_new_pop, simplify_flag=True))
    loss, output_mean_all, output_std_all, target_true = surrogate.evaluate_and_predict(passage, None)
    complexity = torch.ones_like(output_mean_all)
    curr_best = torch.tensor([[observations.labels[np.argmin(observations.labels)]]])
    opt_acq = OptAcqFunc(train_num)
    opt_acq.curr_best = curr_best.view(-1)  
    selected_indices = opt_acq.opt(surrogate,passage,output_mean_all.view(-1),output_std_all.view(-1),complexity.view(-1),topk=10)
    opt_acq.print_selection_details()
    return [generated_new_pop[p_idx] for p_idx in selected_indices]
def plot_surrogate_prediction_vs_true(surrogate, selected_pop, selected_fitness_list, observations, step_i,
                                      save_dir=None):
    selected_pop_str = observations.get_pop_str(selected_pop, simplify_flag=True)
    passages = make_passages(selected_pop_str)
    loss, pred_mean, pred_std, true_labels = surrogate.evaluate_and_predict(passages, selected_fitness_list)
    test_pred = pred_mean.cpu().numpy().flatten()
    test_std = pred_std.cpu().numpy().flatten()
    test_true = true_labels.cpu().numpy().flatten()
    test_errors = test_pred - test_true
    pop_str = observations.get_pop_str(observations.pop)
    combined = list(zip(pop_str, observations.labels))
    combined_sorted = sorted(combined, key=lambda x: x[1])
    top_n = min(400, len(combined_sorted))
    top_combined = combined_sorted[:top_n]\
    
    top_passages = make_passages([item[0] for item in top_combined])
    top_labels = [item[1] for item in top_combined]
    loss_train, train_pred_mean, train_pred_std, train_true_labels = surrogate.evaluate_and_predict(top_passages,top_labels)
    train_pred = train_pred_mean.cpu().numpy().flatten()
    train_std = train_pred_std.cpu().numpy().flatten()
    train_true = train_true_labels.cpu().numpy().flatten()
    train_errors = train_pred - train_true
    train_corr, train_p = pearsonr(train_true, train_pred)
    test_corr, test_p = pearsonr(test_true, test_pred)
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(9, 9))
    all_pred = np.concatenate([train_pred, test_pred])
    all_std = np.concatenate([train_std, test_std])
    pred_min, pred_max = all_pred.min(), all_pred.max()
    std_min, std_max = all_std.min(), all_std.max()
    pred_range = pred_max - pred_min if pred_max != pred_min else 1
    std_range = std_max - std_min if std_max != std_min else 1
    train_pred_norm = (train_pred - pred_min) / pred_range
    test_pred_norm = (test_pred - pred_min) / pred_range
    train_std_norm = (train_std - std_min) / std_range
    test_std_norm = (test_std - std_min) / std_range
    true_min = min(train_true.min(), test_true.min())
    true_max = max(train_true.max(), test_true.max())
    true_range = true_max - true_min if true_max != true_min else 1
    train_true_norm = (train_true - true_min) / true_range
    test_true_norm = (test_true - true_min) / true_range
    plt.rcParams.update({
        'font.size': 15,          
        'axes.titlesize': 15,     
        'axes.labelsize': 15,     
        'xtick.labelsize': 15,    
        'ytick.labelsize': 15,    
        'xtick.major.size': 6,    
        'ytick.major.size': 6,    
        'xtick.major.width': 2, 
        'ytick.major.width':2, 
    })
    plt.errorbar(train_true_norm, train_pred_norm, yerr=2 * train_std_norm, fmt='o',
             color='blue', alpha=0.5, label='Training Set Predictions (mean±2σ)',
             markersize=8)
    plt.errorbar(test_true_norm, test_pred_norm, yerr=2 * test_std_norm, fmt='o',
             color='red', alpha=0.7, label='Test Set Predictions (mean±2σ)',
             markersize=8)
    min_val = min(min(train_true_norm), min(test_true_norm), min(train_pred_norm), min(test_pred_norm))
    max_val = max(max(train_true_norm), max(test_true_norm), max(train_pred_norm), max(test_pred_norm))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')
    plt.xlabel("True values")  
    plt.ylabel("Predicted values")
    plt.text(x=0.05, y=0.95, s=f"r_train : {train_corr:.3f}\nr_test: {test_corr:.3f}",
            transform=plt.gca().transAxes, 
            ha='left', va='top',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'),
            fontsize=20)  
    legend = plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0),
                        frameon=True, framealpha=0.7, edgecolor='black')
    legend.get_frame().set_linewidth(2)  
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    for axis in ['top','bottom','left','right']:
        plt.gca().spines[axis].set_linewidth(2)
    path_combined = os.path.join(save_dir, f'bo_step{step_i + 1}_combined_RBF.png')
    plt.savefig(path_combined, dpi=1000, bbox_inches='tight')  
    print(f"Saved combined plot: {path_combined}")
    plt.close()
    print(f"\n====== Surrogate Model Prediction Comparison - Step {step_i + 1} ======")
    print(f"{'Type':<8} {'Index':<5} {'Predicted':<15} {'True':<15} {'Error':<15} {'Std Dev':<15}")
    for i, (p, t, e, s) in enumerate(zip(test_pred, test_true, test_errors, test_std)):
        print(f"{'Test':<8} {i:<5} {float(p):<15.6f} {float(t):<15.6f} {float(e):<15.6f} {float(s):<15.6f}")
    print("=========================================================\n")
    save_path = os.path.join(save_dir, 'result.txt')
    with open(save_path, "a", encoding="utf-8") as fw:
        fw.write(f"\n====== Surrogate Model Prediction Comparison - Step {step_i + 1} ======\n")
        fw.write(f"{'Type':<8} {'Index':<5} {'Predicted':<15} {'True':<15} {'Error':<15} {'Std Dev':<15}\n")
        for i, (p, t, e, s) in enumerate(zip(test_pred, test_true, test_errors, test_std)):
            fw.write(f"{'Test':<8} {i:<5} {float(p):<15.6f} {float(t):<15.6f} {float(e):<15.6f} {float(s):<15.6f}\n")
        fw.write("=========================================================\n")
def perform_bo_iteration(num_iter=100):
    torch.cuda.empty_cache()
    model_name = 'nvidia/NV-Embed-v2'
    print('loading %s ... ' % model_name)
    llm_model = AutoModel.from_pretrained(model_name, trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=torch.float16).to(device)
    print(llm_model.device)
    print('**done!')
    STORED_OBSERVATIONS = Observations()
    prompts_path = "SymBOL_Code/prompts/OPRO/gen_prompts_1.txt"
    generated_init_pop = gen_use_llm_separate(pop=STORED_OBSERVATIONS, list_f=[], list_g=[], suggest_list= [],selected_pop= [],context_template=prompts_path,add_bool=0)
    if case_name.startswith(("H1N1", "COVID19", "Sars")):
        print(case_name)
        selected_pop = multi_processing_opt_constant_discrete(list(reversed(generated_init_pop)),
                                                 num_running_each_core=max(1,len(generated_init_pop) // multiprocessing.cpu_count()),
                                                 opt_const_dim=-1)
    else:
        selected_pop = multi_processing_opt_constant(list(reversed(generated_init_pop)),
                                                 num_running_each_core=max(1,len(generated_init_pop) // multiprocessing.cpu_count()),
                                                 opt_const_dim=-1)
    selected_pop_str = STORED_OBSERVATIONS.get_pop_str(selected_pop)
    selected_f_eq_str_list = [p_str[0][0] for p_str in selected_pop_str]
    selected_g_eq_str_list = [p_str[0][1] for p_str in selected_pop_str]
    selected_fitness_list = [p_str[1] for p_str in selected_pop_str]
    STORED_OBSERVATIONS.append(selected_f_eq_str_list, selected_g_eq_str_list, selected_fitness_list)
    deepsurr = DeepSurrogate_GP3(llm_model)
    suggest_list = []
    for step_i in range(num_iter):
        raw_sorted_pop = sorted(STORED_OBSERVATIONS.pop, key=lambda x: x[1])[:15]     
        sorted_pop = STORED_OBSERVATIONS.get_pop_str(raw_sorted_pop, simplify_flag=False)
        list_f = [replace_C_to_one(replace_numbers(STORED_OBSERVATIONS.str_from(p_str[0][0], simplify_flag=False))) for p_str in sorted_pop]
        list_g = [replace_C_to_one(replace_numbers(STORED_OBSERVATIONS.str_from(p_str[0][1], simplify_flag=False))) for p_str in sorted_pop]
        list_fitness = [p_str[1] for p_str in sorted_pop]
        list_f = get_unique_random_samples(list_f, 5)
        list_g = get_unique_random_samples(list_g, 5)
        if step_i > 0 and step_i % 2 ==0:      
            prompts_path = "SymBOL_Code/prompts/OPRO/gen_prompts_7.txt"
            suggest_list = suggest_use_llm(list_f = list_f,list_g = list_g, context_template = prompts_path)
        prompts_path = "SymBOL_Code/prompts/OPRO/gen_prompts_2.txt"
        if step_i > 0:
            sorted_pop = selected_pop_str
        selected_pop = bo_one_iter(deepsurr, STORED_OBSERVATIONS, list_f, list_g, suggest_list, sorted_pop,prompts_path, step_i, train_flag=True)
        if case_name.startswith(("H1N1", "COVID19", "Sars")):
            selected_pop = multi_processing_opt_constant_discrete(list(reversed(selected_pop)),
                                                 num_running_each_core=max(1,len(selected_pop) // multiprocessing.cpu_count()),
                                                 opt_const_dim=-1)
        else:
            selected_pop = multi_processing_opt_constant(list(reversed(selected_pop)),num_running_each_core=max(1,len(selected_pop) // multiprocessing.cpu_count()),opt_const_dim=-1)
        selected_pop_str = STORED_OBSERVATIONS.get_pop_str(selected_pop)
        selected_f_eq_str_list = [p_str[0][0] for p_str in selected_pop_str]
        selected_g_eq_str_list = [p_str[0][1] for p_str in selected_pop_str]
        selected_fitness_list =  [p_str[1] for p_str in selected_pop_str]
        filtered = [(pop, fit) for pop, fit in zip(selected_pop, selected_fitness_list) if 0< fit <= 1e20]
        filtered_pop, filtered_fitness_list = zip(*filtered) if filtered else ([], [])
        plot_surrogate_prediction_vs_true(
            surrogate=deepsurr,
            selected_pop=filtered_pop,
            selected_fitness_list=filtered_fitness_list,
            observations=STORED_OBSERVATIONS,
            step_i=step_i,
            save_dir=save_dir
        )
        global_best1 = STORED_OBSERVATIONS.pop[np.argsort([s_i[1] for s_i in STORED_OBSERVATIONS.pop])[0]]
        STORED_OBSERVATIONS.append(selected_f_eq_str_list, selected_g_eq_str_list, selected_fitness_list)
        fitness_all = STORED_OBSERVATIONS.labels
        if len(selected_pop_str) == 0:
            print("Warning: selected_pop_str is empty. No elements to process.")
            exit(-1)
        curr_best = selected_pop_str[np.argsort([s_i[1] for s_i in selected_pop_str])[0]]
        global_best = STORED_OBSERVATIONS.pop[np.argsort([s_i[1] for s_i in STORED_OBSERVATIONS.pop])[0]]
        if curr_best[1] < global_best1[1]:
            print("change!")
            print()
            print(f"Equations Numbers: {len(STORED_OBSERVATIONS.pop)}")
        print(add_str)
        print('Step %s: curr best is %s | %s | %s, global best is %s | %s | %s'%(step_i,
                                                                str(sp.simplify(curr_best[0][0])),
                                                                str(sp.simplify(curr_best[0][1])),
                                                                curr_best[1],
                                                                str(sp.simplify(gp.TreeToDeadStr(global_best[0][0], converter))),
                                                                str(sp.simplify(gp.TreeToDeadStr(global_best[0][1], converter))),
                                                                global_best[1]))
        save_path = os.path.join(save_dir, 'result.txt')
        with open(save_path, "a", encoding="utf-8") as fw:
            if curr_best[1] < global_best1[1]:
                fw.write("change!")
                fw.write("\n") 
                fw.write(f"Equations Numbers: {len(STORED_OBSERVATIONS.pop)}")
                fw.write("\n") 
            fw.write(f"Step {step_i} ")
            fw.write(f"curr best is {str(sp.simplify(curr_best[0][0]))} | {str(sp.simplify(curr_best[0][1]))} | {curr_best[1]} \n")
            fw.write(f"global best is {str(sp.simplify(gp.TreeToDeadStr(global_best[0][0], converter)))} | {str(sp.simplify(gp.TreeToDeadStr(global_best[0][1], converter)))} | {global_best[1]} \n")
        plt.clf()  
        plt.scatter(np.linspace(1,len(fitness_all),len(fitness_all)), fitness_all, c='k')
        plt.plot(np.linspace(1,len(fitness_all),len(fitness_all)), [np.min(fitness_all[:i+1]) for i in range(len(fitness_all))], c='r')
        plt.yscale("log")
        plt.ylim(1e-11, 1.5)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'bo_search_iter%s.png'%(step_i+1))
        plt.savefig(save_path)
        plt.close()
        with open('STORED_OBSERVATIONS.pickle', 'wb') as f:
            pickle.dump(STORED_OBSERVATIONS, f)
if __name__ == "__main__":
    torch.cuda.empty_cache()
    perform_bo_iteration(num_iter=1000)
