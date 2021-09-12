import os
# model_name, optim_name, fold

model_name = 'nsgp'
optim_name = 'ad'
c_fold = '0'
node = 'gpu4'
nsgp_iters = 40
gp_iters = 50
restarts = 5
div = 4
sampling = 'uni' # cont, nn, uni
Xcols = '@'.join(['longitude', 'latitude', 'humidity', 'temperature', 'weather', 'wind_direction', 'wind_speed', 'delta_t'])
# Xcols = '@'.join(['longitude', 'latitude', 'temperature', 'humidity', 'wind_speed',
#        'weather_0.0', 'weather_1.0', 'weather_2.0', 'weather_3.0',
#        'weather_4.0', 'weather_5.0', 'weather_6.0', 'weather_7.0',
#        'weather_8.0', 'weather_9.0', 'weather_11.0', 'weather_12.0',
#        'weather_13.0', 'weather_14.0', 'weather_15.0', 'weather_16.0',
#        'wind_direction_0.0', 'wind_direction_1.0', 'wind_direction_2.0',
#        'wind_direction_3.0', 'wind_direction_4.0', 'wind_direction_9.0',
#        'wind_direction_13.0', 'wind_direction_14.0', 'wind_direction_23.0',
#        'wind_direction_24.0', 'delta_t'])
kernel = 'rbf' # Order: RBF, M32
time_kernel = 'local_per' # Order RBF, loc_per
# ['longitude', 'latitude', 'humidity', 'temperature', 'weather', 'wind_direction', 'wind_speed', 'delta_t']

sampling = 'nn'
os.system(' '.join(['python run.py', model_name, optim_name, c_fold, 'gpu1', 
                str(nsgp_iters), str(gp_iters), str(restarts), str(div), sampling, Xcols, kernel, time_kernel]))
os.system(' '.join(['python run.py', model_name, optim_name, '1', 'gpu1', 
                str(nsgp_iters), str(gp_iters), str(restarts), str(div), sampling, Xcols, kernel, time_kernel]))
os.system(' '.join(['python run.py', model_name, optim_name, '2', 'gpu2', 
                str(nsgp_iters), str(gp_iters), str(restarts), str(div), sampling, Xcols, kernel, time_kernel]))
sampling = 'uni'
os.system(' '.join(['python run.py', model_name, optim_name, c_fold, 'gpu2', 
                str(nsgp_iters), str(gp_iters), str(restarts), str(div), sampling, Xcols, kernel, time_kernel]))
os.system(' '.join(['python run.py', model_name, optim_name, '1', 'gpu3', 
                str(nsgp_iters), str(gp_iters), str(restarts), str(div), sampling, Xcols, kernel, time_kernel]))
os.system(' '.join(['python run.py', model_name, optim_name, '2', 'gpu3', 
                str(nsgp_iters), str(gp_iters), str(restarts), str(div), sampling, Xcols, kernel, time_kernel]))
### Running all
# optim_name = 'ad'
# for model_name in ['gp', 'nsgp', 'snsgp']:
#     for c_fold in range(3):
#         os.system(' '.join(['python run.py', model_name, optim_name, c_fold]))