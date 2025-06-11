from scipy.special import softmax
import numpy as np
import random
import tqdm as tqdm
from tqdm import trange
import matplotlib.pyplot as plt
from matplotlib import colors
from statsmodels.tsa.stattools import adfuller, kpss
import pandas as pd
import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning
from collections import deque
from collections import defaultdict