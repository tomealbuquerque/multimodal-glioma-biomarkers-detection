# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 16:34:12 2022

@author: albu
"""

import torch 

input1=torch.rand(512)
input2=torch.rand(512)
input3=torch.rand(512)
input4=torch.rand(512)

Enc = [input1, input2, input3, input4]

jo=torch.concat(Enc)