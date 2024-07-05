import torch 
import os
import pandas as pd

from models import AttnRegressor
from tqdm import tqdm

data_dir = r'/Users/drew/Documents/MathModeling/MathModeling/APMCM24_cn/data'
weight_dir = r'/Users/drew/Documents/MathModeling/MathModeling/APMCM24_cn/weights'
net = AttnRegressor(nheads=8)
net.load_state_dict(torch.load(os.path.join(weight_dir, 'model.pth')))
test_data = pd.read_csv(os.path.join(data_dir, 'test.csv'))
submit_data = pd.read_csv('/Users/drew/Documents/MathModeling/MathModeling/APMCM24_cn/sample/data/sample_submission.csv')
for i in tqdm(range(len(test_data))):
    idx = test_data.iloc[i, 0]
    data = torch.tensor(test_data.iloc[i, 1:-1].values, dtype=torch.float32).view(1, 20)
    o = net(data)
    submit_data.loc[submit_data['id'] == idx, 'FloodProbability'] = o.item()
submit_data.to_csv('submit.csv', index=False)