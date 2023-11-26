# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
# %%
url = 'https://raw.githubusercontent.com/karpathy/makemore/master/names.txt'
page = requests.get(url)
names = [page.text.strip().split()]
# %%
