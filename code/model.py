# encoding: utf-8
import torch

def yield_test(n):
    # for i in range(n-1):
    yield call(n)
    print("i=", i)
    print('do something else.')

def call(i):
    return i * 2

for i in yield_test(5):
    print(i, ",")