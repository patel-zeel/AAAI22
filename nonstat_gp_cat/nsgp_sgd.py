import torch
import gc
import numpy as np
import torch.autograd.profiler as profiler

class DimensionalityReduction(torch.nn.Sequential):
    def __init__(self, n, m):
        super(DimensionalityReduction, self).__init__()
        self.add_module('linear1', torch.nn.Linear(n, 128))
        self.add_module('relu1', torch.nn.GELU())
        self.add_module('linear2', torch.nn.Linear(128, 32))
        self.add_module('relu2', torch.nn.GELU())
        self.add_module('linear3', torch.nn.Linear(32, 8))
        self.add_module('relu3', torch.nn.GELU())
        self.add_module('linear4', torch.nn.Linear(8, m))

class NSGP(torch.nn.Module):
    def __init__(self, X_bar, cat_indicator, time_indicator,
                 jitter=10**-8, random_state=None, 
                 local_noise=True, local_std=True, debug=False, use_Trans=False, m=3, kernel='rbf', timekernel='loc_periodic'):
        super().__init__()

        # assert len(
            # X.shape) == 2, "X is expected to have shape (n, m) but has shape "+str(X.shape)
        # assert len(
            # y.shape) == 2, "y is expected to have shape (n, 1) but has shape "+str(y.shape)
        # assert y.shape[1] == 1, "y is expected to have shape (n, 1) but has shape "+str(
            # y.shape)

        # self.X = X
        # self.raw_mean = y.mean()
        # self.y = y - self.raw_mean
        self.X_bar = X_bar
        self.cat_indicator = cat_indicator
        self.time_indicator = time_indicator
        self.debug = debug
        self.kernel = kernel
        self.timekernel = timekernel
        # self.N = self.X.shape[0]
        self.input_dim = len(self.X_bar)
        self.use_Trans = use_Trans
        # self.num_latent_points = self.X_bar.shape[0]
        self.jitter = jitter
        self.local_noise = local_noise
        self.local_std = local_std
        self.random_state = random_state

        # Local params
        self.local_gp_ls = self.param((self.input_dim,))
        self.local_gp_std = self.param((self.input_dim,))
        if not self.local_std:
            self.local_gp_std.requires_grad = False
        self.local_gp_noise_std = self.param((self.input_dim,))
        if not self.local_noise:
            self.local_gp_noise_std.requires_grad = False
        self.local_ls = torch.nn.ParameterList([self.param((self.X_bar[i].shape[0],)) if self.cat_indicator[i]==0 else self.param((1,)) for i in range(len(self.X_bar))])

        # Global params
        self.global_gp_std = self.param((1,))
        self.global_gp_noise_std = self.param((1,))
        if self.timekernel in ['periodic', 'loc_periodic']:
            self.period = self.param((1,))

        # Feature reduction feature
        if self.use_Trans:
            self.F_transform = DimensionalityReduction(self.X.shape[1], m)

        # Other params to be used
        # self.eye_num_inducing_points = torch.eye(self.num_latent_points, dtype=self.X.dtype)
        # self.eye_N = torch.eye(self.N)
        self.pi = torch.tensor(np.pi)

        # Initialize model parameters
        self.initialize_params()

    def param(self, shape, requires_grad=True):
        return torch.nn.Parameter(torch.empty(shape, dtype=self.X_bar[0].dtype), requires_grad=requires_grad)

    def initialize_params(self):
        if self.random_state is None:
            self.random_state = int(torch.rand(1)*1000)
        torch.manual_seed(self.random_state)
        for param in self.parameters():
            if param.requires_grad:
                torch.nn.init.normal_(param, mean=0.0, std=1.0)
            else:
                torch.nn.init.constant_(param, 1.)

    def LocalKernel(self, x1, x2, dim):  # kernel of local gp (GP_l)
        dist = torch.square(x1 - x2.T) # RBF
        # Or
        # dist = torch.abs(x1 - x2.T) # Matern12
        # dist[dist==0] = 10**-20
        scaled_dist = dist/self.local_gp_ls[dim]**2
        # print(scaled_dist, self.local_gp_ls[dim]**2)
        return self.local_gp_std[dim]**2 * torch.exp(-0.5*scaled_dist)

    def get_LS(self, X, dim):  # Infer lengthscales for train_X (self.X)
        # print('Log: dim', dim)
        k = self.LocalKernel(
            self.Xm_bar[dim], self.Xm_bar[dim], dim)
        # print('k', k)
        # Diagonal Solution from https://stackoverflow.com/a/48170846/13330701
        dk = k.diagonal()
        dk += self.local_gp_noise_std[dim]**2 + self.jitter
        c = torch.linalg.cholesky(k)
        # print('c', c)
        # torch.clamp_(self.local_ls[:, dim], min=10**-4)
        # print(self.local_ls[dim].device, self.local_ls[dim].shape)
        alpha = torch.cholesky_solve(
            torch.log(self.local_ls[dim].reshape(-1,1)), c)
        k_star = self.LocalKernel(
            X[:, dim, None], self.Xm_bar[dim], dim)
        l = torch.exp(k_star@alpha)
        # print('L0', l[0])

        if self.training:
            k_star = self.LocalKernel(
                X[:, dim, None], self.Xm_bar[dim], dim)
            k_star_star = self.LocalKernel(
                X[:, dim, None], X[:, dim, None], dim)

            chol = torch.linalg.cholesky(k)
            v = torch.cholesky_solve(k_star.T, chol)

            k_post = k_star_star - k_star@v
            # k_post_det = torch.det(k_post)
            # k_post_det = torch.clamp(k_post_det, min=10**-20)
            # if k_post_det<=0:
            #     k_post_det = torch.tensor(10**-20)
            # B = torch.log(k_post_det).reshape(-1,1)

            dk_post = k_post.diagonal()
            dk_post += self.jitter*10
            post_chol = torch.linalg.cholesky(k_post)
            B = torch.sum(torch.log(post_chol.diagonal()))
            return l, B
        else:
            return l

    def GlobalKernel(self, X1, X2):  # global GP (GP_y)
        prefix = None
        sum_scaled_dist = None
        B_all = None
        for d in range(X1.shape[1]):
            if self.cat_indicator[d] == 1:
                dist = X1[:, None, d] != X2[None, :, d]
                scaled_dist = 0.5 * dist/self.local_ls[d]**2
            else:
                if self.training:
                    l, B = self.get_LS(X1, d)
                    if B_all is None:
                        B_all = B
                    else:
                        B_all +=  B
                    # print('l', l)
                    l1 = l
                    l2 = l
                else:
                    l1 = self.get_LS(X1, d)
                    l2 = self.get_LS(X2, d)
                if self.debug:
                    input('1:')
                lsq = torch.square(l1) + torch.square(l2.T)
                if prefix is None:
                    prefix = torch.sqrt(2 * l1@l2.T / lsq)
                else:
                    prefix = prefix * torch.sqrt(2 * l1@l2.T / lsq)
                if self.debug:
                    input('2:')
                if self.time_indicator[d]==1:
                    if self.timekernel == 'loc_periodic':
                        scaled_dist = (4 * torch.square(torch.sin(self.pi * torch.abs(X1[:, None, d] - X2[None, :, d])/self.period)) +\
                            torch.square(X1[:, None, d] - X2[None, :, d]))/lsq
                    elif self.timekernel == 'periodic':
                        scaled_dist = 4 * torch.square(torch.sin(self.pi * torch.abs(X1[:, None, d] - X2[None, :, d])/self.period))/lsq
                    else:
                        scaled_dist = torch.square(X1[:, None, d] - X2[None, :, d])/lsq
                else:
                    diff = X1[:, None, d] - X2[None, :, d]
                    if self.kernel == 'rbf':
                        scaled_dist = torch.square(diff)/lsq
                    elif self.kernel == 'm12':
                        scaled_dist = torch.abs(diff)/torch.sqrt(lsq/2)
                    elif self.kernel == 'm32':
                        d_by_r = torch.abs(diff)/torch.sqrt(lsq/2)
                        prefix = prefix * (1 + 3**0.5 * (d_by_r))
                        scaled_dist = 3**0.5 * (d_by_r)
                # except AttributeError:
                #     scaled_dist = torch.square(X1[:, None, d] - X2[None, :, d])
                # print('dist', dist)
                # print('lsq', lsq)
                # scaled_dist = dist/lsq
            if sum_scaled_dist is None:
                sum_scaled_dist = scaled_dist
            else:
                sum_scaled_dist += scaled_dist
            
            # print("sum_scaled_dist", sum_scaled_dist, 'd', d)
        K = self.global_gp_std**2 * \
            prefix * torch.exp(-sum_scaled_dist)

        if self.training:
            return K, B
        else:
            return K

    def get_Trans(self, X):
        Xm = self.F_transform(X[:, :-1])
        return torch.cat([Xm, X[:, -1:]], dim=1)

    def forward(self, X, y):
        if self.use_Trans:
            Xm = self.F_transform(X)
            self.Xm_bar = self.F_transform(self.X_bar)
        else:
            Xm = X
            self.Xm_bar = self.X_bar
        K, B = self.GlobalKernel(Xm, Xm)

        dK = K.diagonal()
        dK += self.global_gp_noise_std**2 + self.jitter
        # print(K)
        L = torch.linalg.cholesky(K)
        alpha = torch.cholesky_solve(y, L)
        
        Apart1 = y.T@alpha
        Apart2 = torch.sum(torch.log(L.diagonal()))

        A = 0.5*( Apart1 + Apart2)[0, 0]
        
        # Bpart1 = B
        # Bpart2 = 0.5*(self.num_latent_points *
        #                                    self.input_dim*torch.log(2*self.pi))
        
        # B = Bpart1# + Bpart2
        
        # print("A1", Apart1, "A2", Apart2, "B", B, "Loss", A+B, 'local var', self.local_gp_std)
        # print('A', A, 'B', B)
        return (A+B)/X.nelement()

    def predict(self, X, y, X_new):  # Predict at new locations
        if self.use_Trans:
            Xm = self.F_transform(X)
            Xm_new = self.F_transform(X_new)
            self.Xm_bar = self.F_transform(self.X_bar)
        else:
            Xm = X
            Xm_new = X_new
            self.Xm_bar = self.X_bar

        K = self.GlobalKernel(Xm, Xm)
        K_star = self.GlobalKernel(Xm_new, Xm)
        K_star_star = self.GlobalKernel(Xm_new, Xm_new)

        dK = K.diagonal()
        dK += self.global_gp_noise_std**2
        L = torch.linalg.cholesky(K)
        alpha = torch.cholesky_solve(y, L)

        pred_mean = K_star@alpha

        v = torch.cholesky_solve(K_star.T, L)
        pred_var = K_star_star - K_star@v

        dpred_var = pred_var.diagonal()
        dpred_var += self.global_gp_noise_std**2
        return pred_mean, pred_var
