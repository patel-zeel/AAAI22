from lib import *
import gpytorch

### GP Model

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1]))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def pprint(*args, end='\n'):
    print(*args, end=end)
    with open('../logs/'+common_path+'.txt', 'a') as f:
        f.write(' '.join(map(str, args)))
        f.write(end)

# Config.res_path = Config.res_path.replace('/workspace', '/home')
m_name = sys.argv[1]
c_fold = sys.argv[2]
sampling = sys.argv[3]
Xcols = sys.argv[4]
kernel = sys.argv[5]
timekernel = sys.argv[6]
if len(Xcols.split('@'))>15:
    common_path = 'one@hot@encoded'+c_fold
else:
    common_path = '_@_'.join([m_name, c_fold, Xcols, sampling, kernel, timekernel])
train_res = pd.read_pickle(Config.res_path+common_path+'.res')

dataloader = train_res['dataloader']

test_X, test_y = dataloader.load_test()

X, y, _ = dataloader.load_train()

# Xm = f_random(X, config.num_inducing_points, random_state=config.model_seed)
# X_bar = f_kmeans(X, config.num_latent_points, random_state=config.model_seed)

# X = X.to(config.device)
# y = y.to(config.device)
# X_bar = X_bar.to(config.device)
# Xm = Xm.to(config.device)

# model = NSGP(X, y, X_bar, random_state=config.model_seed, jitter=config.jitter)
# model.load_state_dict(torch.load(config.res_path+m_name+'_fold'+config_fold+'.model'))
# model.to(config.device)

model = torch.load(Config.res_path+common_path+'.model')
model.eval()

if m_name in ['nsgp', 'snsgp']:
    with torch.no_grad():
        pred_y, pred_var = model.predict(X.to(Config.device), y.to(Config.device), 
                                        test_X.to(Config.device))
else:
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(Config.device)
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_X.to(Config.device)))
        pred_y = observed_pred.mean
        pred_var = observed_pred.variance
        print(list(model.parameters()))
        
# dataloader.test_data['pred_mean'] = dataloader.yscaler.inverse_transform(pred_y.cpu())
# Or
dataloader.test_data['pred_mean'] = pred_y.cpu().ravel() + dataloader.y_mean

# pprint(dataloader.yscaler.var_.shape)


if m_name in ['nsgp', 'snsgp']:
    dataloader.test_data['pred_var'] = pred_var.diagonal().cpu()
else:
    dataloader.test_data['pred_var'] = pred_var.cpu()

# dataloader.test_data['pred_var'] = dataloader.test_data['pred_var'] * dataloader.yscaler.var_[0]

dataloader.test_data.to_csv(Config.res_path+common_path+'.csv')
pprint('Finished')