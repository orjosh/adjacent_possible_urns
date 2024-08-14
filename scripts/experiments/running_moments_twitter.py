import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys
sys.path.append("scripts/")
import analysisfuncs as af

datasets, filenames = af.load_all_csvs("twitter_data/", pattern="retweeted_freq*")