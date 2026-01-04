import os
class Config:
    data_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/data/time_feature/'
    device = 'cuda'
    res_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/experiments/results/'
    # res_path = '/home/patel_zeel/AAAI22/experiments/results/'  # Old hardcoded path