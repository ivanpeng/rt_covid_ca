import os
import requests
import pymc3 as pm
import pandas as pd
import numpy as np

from datetime import date
from datetime import datetime

from joblib import Parallel, delayed
from scipy import stats as sps
from scipy.stats import dirichlet
from scipy.interpolate import interp1d

from pygam import GammaGAM, PoissonGAM, s, l
from sklearn.utils import resample

