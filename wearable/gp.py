import os
import torch
import gpytorch
import numpy as np

'''
train a gp on a sample of data
sample from this at given points to generate training data
'''

class GP(gpytorch.models.ExactGP):
    def __init__(self, x_train, y_train, likelihood):
        super(GP, self).__init__(x_train, y_train, likelihood)
        b = x_train.shape[1]
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([b]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([b])),
            batch_shape=torch.Size([b])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )

import time
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

def train_sample_gp(x_train, y_train, x_test, gp_epochs=25, device='cuda'):
    # gp model
    #likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=y_train.shape[1])
    model = GP(x_train, y_train, likelihood).to(device)
    #model.covar.base_kernel.lengthscale = 0.75 / np.power(x_train.shape[0], 1/dim)

    if x_train.shape[1] == 1:
        likelihood.noise = 1e-3

    # set fixed noise and lengthscale
    params = set(model.parameters())
    final_params = list(params)
        #- {model.covar.base_kernel.raw_lengthscale}
        #- {likelihood.noise_covar.raw_noise})
    optimizer = torch.optim.Adam(params, lr=0.25)

    # loss function (marginal log-likelihood) 
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # optimization setup
    model.train()
    likelihood.train()

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)

    # train loop
    for i in range(gp_epochs):
        optimizer.zero_grad()

        output = model(x_train)
        loss = -mll(output, y_train)

        loss.backward()
        optimizer.step()

    # get new points
    model.eval()
    likelihood.eval()

    # get predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(x_test))
        sample = pred.sample()

        lower, upper = pred.confidence_region()

    #plot_sample(x_train, y_train, x_test, sample, pred)
    return sample.detach().cpu()


def plot_sample(x_train, y_train, x_test, sample, pred):
    size = x_train.shape[1]
    x_train = x_train.detach().cpu(); y_train = y_train.detach().cpu()
    x_test = x_test.detach().cpu(); sample = sample.detach().cpu()

    # get first sample only
    ind = np.random.randint(sample.shape[1])
    x_train = x_train[:,ind]; y_train = y_train[:,ind]
    x_test = x_test[:,ind]; sample = sample[:,ind]

    # make plot
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)

    mean = pred.mean
    lower, upper = pred.confidence_region()
    mean = mean[:,ind].cpu(); lower = lower[:,ind].cpu(); upper = upper[:,ind].cpu()

    ax.plot(x_test, mean, 'b')
    ax.fill_between(x_test, lower, upper, alpha=0.5)
    
    ax.scatter(x_test, sample, color='r', s=10)
    ax.scatter(x_train, y_train, color='k', s=40)

    #ax.set_ylim([-3, 3])
    ax.legend(['Mean', 'Confidence', 'Samples', 'Observed Data'])
    plt.savefig(f'gp_{size}.png')

