# Copy from: https://github.com/ghliu/pytorch-ddpg/blob/master/memory.py
from __future__ import absolute_import
from collections import deque, namedtuple
import warnings
import random

import numpy as np

# [reference] https://github.com/matthiasplappert/keras-rl/blob/master/rl/memory.py

# This is to be understood as a transition: Given `state0`, performing `action`
# yields `reward` and results in `state1`, which might be `terminal`.
Experience = namedtuple('Experience', 'state0, action, reward, state1, terminal1')