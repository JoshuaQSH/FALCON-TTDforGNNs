import torch
import os
from torch import Tensor    

from typing import List, Optional, Tuple, Union

from torch.cuda import memory_allocated
import time

import numpy as np
import time
import logging
from logging import handlers
import timeit
from collections import Counter
from itertools import product, permutations, combinations

import matplotlib.pyplot as plt

def max_batch_size(model, device, dummy_input, init_batch_size):
    optimal_batch_size = init_batch_size
    while(optimal_batch_size):
        try:
            _ = model(dummy_input)
            optimal_batch_size += 1
        except Exception as err:
            print("Optimal batch size is: ", optimal_batch_size)
            raise err

def give_throughput(repetitions, optimal_batch_size, total_time):
    return (repetitions*optimal_batch_size) / total_time

def memory_usage(in_tensor, is_tt=False):
    # Memory = \prob_{i=1}^{n} (d_i) * SizeOf(DataType)
    # Memory = \sum_{i=1}^{n} (r_{i-1} * d_i * r_i) * SizeOf(DataType)
    memory_count = []
    if is_tt:
        for core in in_tensor.cores:
            memory_count.append(core.element_size() * core.nelement())
        sum_memory = sum(memory_count)
    else:
        sum_memory = in_tensor.element_size() * in_tensor.nelement()
    return sum_memory

def count_nnz(x):
    nnz = torch.count_nonzero(x)
    print("The shape is: ", x.shape)
    print("The NNZ is: ", nnz)
    sparsity = 1 - (nnz / x.numel())
    print("The sparsity is: ", sparsity)

def prime_factors(n):
    factors = []
    for i in range(2, n + 1):
        while n % i == 0:
            factors.append(i)
            n = n // i
        if n == 1:
            break
    return factors


def factor_combinations_permute(n):
    primes = prime_factors(n)
    combinations_list = [[n]] # Start with the number itself

    # Function to calculate the product of elements in a list
    def product(lst):
        result = 1
        for x in lst:
            result *= x
        return result

    # Generate all combinations of the prime factors
    for i in range(1, len(primes)):
        for combo in combinations(primes, i):
            remainder = n // product(combo)
            if product(combo) * remainder == n:
                combinations_list.append(sorted(list(combo) + [remainder]))

    # Remove duplicates
    unique_combinations = set(tuple(sorted(combo)) for combo in combinations_list)

    # Generate all permutations for each combination
    all_permutations = []
    for combo in unique_combinations:
        for perm in set(permutations(combo)):
            all_permutations.append(perm)

    return all_permutations

def factor_combinations(n):
    primes = prime_factors(n)
    combinations_list = [[n]] # Start with the number itself as one of the combinations

    # Function to calculate the product of elements in a list
    def product(lst):
        result = 1
        for x in lst:
            result *= x
        return result

    # Generate all combinations of the prime factors
    for i in range(1, len(primes)):
        for combo in combinations(primes, i):
            remainder = n // product(combo)
            if product(combo) * remainder == n:
                combinations_list.append(sorted(list(combo) + [remainder]))

    # Removing duplicates and sorting each combination
    unique_combinations = []
    for combo in combinations_list:
        if combo not in unique_combinations:
            unique_combinations.append(sorted(combo))

    return unique_combinations

def generate_combinations(factors):
    factor_counts = Counter(factors)
    unique_factors = list(factor_counts.keys())

    # Generate all combinations
    all_combinations = []
    for counts in product(*(range(factor_counts[factor] + 1) for factor in unique_factors)):
        combination = []
        for factor, count in zip(unique_factors, counts):
            combination.extend([factor] * count)
        all_combinations.append(combination)

    return all_combinations

class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }

    def __init__(self, 
                 filename, 
                 level='info',
                 when='D',
                 backCount=3,
                 fmt='%(asctime)s - %(pathname)s - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))
        sh = logging.StreamHandler()
        sh.setFormatter(format_str)
        th = handlers.TimedRotatingFileHandler(filename=filename,
                                               when=when,
                                               backupCount=backCount,
                                               encoding='utf-8')

        th.setFormatter(format_str)
        self.logger.addHandler(sh) 
        self.logger.addHandler(th)

def plot_access_percentages(access_percentages, plot_name="emb_row_access.pdf"):
    plt.figure(figsize=(12, 6))
    plt.plot(torch.arange(len(access_percentages)), access_percentages)
    plt.xlabel('Embedding Index')
    plt.ylabel('Access Percentage')
    plt.title('Access Percentage of Each Embedding')
    plt.grid(True)
    # plt.show()
    plt.savefig(plot_name)

def plot_cumulative_access_percentages(cumulative_percentages, plot_name="cumulative_access.pdf"):
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_percentages)
    plt.xlabel('Number of Embeddings')
    plt.ylabel('Cumulative Access Percentage')
    plt.title('Cumulative Access Percentage of Embeddings')
    plt.grid(True)
    plt.savefig(plot_name)

def calculate_access_percentages(access_counts, plot_name="emb_row_access.pdf"):
    total_accesses = np.sum(access_counts.numpy())
    # Convert to percentage
    access_percentages = (access_counts / total_accesses) * 100
    sorted_percentages = torch.sort(access_percentages, descending=True)[0]
    cumulative_percentages = torch.cumsum(sorted_percentages, dim=0)
    plot_cumulative_access_percentages(cumulative_percentages, plot_name=plot_name)
    return access_percentages

def gpu_timing(model, device, input_, repetitions=100):
    model_ = model.to(device)
    input_ = input_.to(device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((repetitions,1))

    # GPU warm up
    for _ in range(10):
        _ = model_(input_)

    # measurement
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model_(input_)
            ender.record()

            # wait for sync
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
            
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    return mean_syn
