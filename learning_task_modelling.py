"""
%load_ext autoreload
%autoreload 2
"""

import numpy as np
from DMpy import DMModel, Parameter
from DMpy.learning import metalearning_pe, rescorla_wagner, \
    dual_lr_qlearning, sk1
from DMpy.observation import softmax, softmax_ml, softmax_ml2
from DMpy.utils import load_data
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import re
import time
sns.set()

data_dir = 'C:\Users\Toby\Google Drive\PhD\Things\Laura\learning_task\data_jan18'

data_files = [f for f in os.listdir(data_dir) if re.match('learning_data.+\d{4}\.csv', f)]

dfs = []

for f in data_files:
    print f
    data = pd.read_csv(os.path.join(data_dir, f))
    print "{0} trials".format(len(data))
    if len(data) == 200:
        data['Response_binary'] = (data.Response == 'a').astype(int)  # convert keypresses to zeros and ones
        data.Response_binary[data.Response == 'Invalid'] = np.nan
        dfs.append(data[['id', 'Response_binary', 'A_reward', 'Confidence', 'trial_number']])  # add the dataframe to a list so we can combine them later
    else:
        print "Subject did not complete all trials, skipping"

data = pd.concat(dfs)
data.columns = ['Subject', 'Response', 'Outcome', 'Confidence', 'Trial_number']  # rename columns for DMpy

data = data[data.Subject != 2]

data.to_csv(os.path.join(data_dir, 'learning_data.csv'))

print "Loaded data from {0} subjects".format(len(data.Subject.unique()))

"""
Set up models
"""
import theano.tensor as T

def uncertainty_dlr(o, t, v, alpha, beta):

    pe = o - v
    value = v + alpha * pe

    alpha_m = alpha + beta * (T.pow(pe, 2) - alpha)

    return (value, alpha_m, pe)

def forgetful_beta(o, t, v, alpha, beta, f):

    alpha = T.switch(T.eq(o, 1), (1 - f) * alpha + 1, (1 - f) * alpha)
    beta = T.switch(T.eq(o, 0), (1 - f) * beta + 1, (1 - f) * beta)

    v = alpha / (alpha + beta)
    var = (alpha * beta) / (T.pow(alpha + beta, 2) * (alpha + beta + 1))

    return (v, alpha, beta, var)


# Softmax temperature parameter
b = Parameter('b', 'normal', lower_bound=1, mean=3, variance=1)

# Rescorla-Wagner
v = Parameter('value', 'fixed', mean=0.5, dynamic=True)
alpha = Parameter('alpha', 'normal', lower_bound=0, upper_bound=1, mean=0.3, variance=1)

model_rw = DMModel(rescorla_wagner, [v, alpha], softmax, [b])

# Dual learning rate

alpha_p = Parameter('alpha_p', 'normal', lower_bound=0, upper_bound=1, mean=0.3, variance=1)
alpha_n = Parameter('alpha_n', 'normal', lower_bound=0, upper_bound=1, mean=0.3, variance=1)

model_dual_lr = DMModel(dual_lr_qlearning, [v, alpha_p, alpha_n], softmax, [b])

# Uncertainty DLR
beta = Parameter('beta', 'normal', lower_bound=0, upper_bound=1, mean=0.5, variance=10)
alpha = Parameter('alpha', 'normal', lower_bound=0, upper_bound=1, mean=0.3, variance=5, dynamic=True)

model_uncertainty_dlr = DMModel(uncertainty_dlr, [v, alpha, beta], None, None)

# Sutton K1
beta = Parameter('beta', 'fixed', mean=-2, dynamic=True)
h = Parameter('h', 'normal', mean=0.005, variance=0.002, dynamic=True)
k = Parameter('k', 'fixed', mean=0.5, dynamic=True)
mu = Parameter('mu', 'normal', lower_bound=1, mean=1.65, variance=1)
rhat = Parameter('rhat', 'normal', mean=0.5, variance=1)
b = Parameter('b', 'normal', lower_bound=1, mean=3, variance=1)

model_sk1 = DMModel(sk1, [v, beta, h, mu, rhat], softmax, [b])

# FORGETFUL BETA
fb_beta = Parameter('fb_beta', 'fixed', lower_bound=0, upper_bound=1, mean=0, variance=10, dynamic=True)
fb_alpha = Parameter('fb_alpha', 'fixed', lower_bound=0, upper_bound=1, mean=0, variance=5, dynamic=True)
f = Parameter('f', 'normal', lower_bound=0, upper_bound=1, mean=0.3, variance=5, dynamic=False)

model_forgetful_beta = DMModel(forgetful_beta, [v, fb_alpha, fb_beta, f], softmax, [b])


"""
Fit models
"""

response_file = os.path.join(data_dir, 'learning_data.csv')

print "Fitting Rescorla-Wagner model"
model_rw.fit(response_file, fit_method='MAP', exclude=[2])
print "BIC = {0}".format(model_rw.BIC)

print "Fitting dual LR model"
model_dual_lr.fit(response_file, fit_method='MAP', exclude=[2])
print "BIC = {0}".format(model_dual_lr.BIC)

print "Fitting uncertainty DLR model"
model_uncertainty_dlr.fit(response_file, fit_method='MAP', exclude=[2])
print "BIC = {0}".format(model_uncertainty_dlr.BIC)

print "Fitting Sutton K1 model"
model_sk1.fit(response_file, fit_method='MAP', exclude=[2])
print "BIC = {0}".format(model_sk1.BIC)

print "Fitting forgetful beta model"
model_forgetful_beta.fit(response_file, fit_method='MAP', exclude=[2])
print "BIC = {0}".format(model_forgetful_beta.BIC)

"""
Relate modelling outputs to confidence reports
"""

sk1_parameters = model_sk1.parameter_table

outcomes = data.pivot(columns='Subject', values='Outcome', index='Trial_number').values.T

sim = model_sk1.simulate()
value = model_sk1.simulated['sim_results']['value']
est_uncertainty = model_sk1.simulated['sim_results']['phatii']


fb_sim = model_forgetful_beta.simulate()
value = model_forgetful_beta.simulated['sim_results']['v']
est_uncertainty = model_forgetful_beta.simulated['sim_results']['var']

data = pd.read_csv(response_file)
data['model_value'] = value.T.flatten()
data['irr_uncertainty'] = np.abs((value.T.flatten() - 0.5)) * 2
data['est_uncertainty'] = est_uncertainty.T.flatten()

# Plot confidence against irreducible uncertainty

sns.set_style("white")

n_cols = int(np.ceil(len(data.Subject.unique()) / 5.))

f, ax = plt.subplots(5, n_cols, figsize=(4.25 * n_cols, 10))

for n, s in enumerate(np.unique(data.Subject)):
    x = np.arange(0, len(data[data.Subject == s]))
    ax[n / n_cols, n % n_cols].plot(x, data.irr_uncertainty[data.Subject == s], color='#808080')
    confidence = data.Confidence[data.Subject == s]
    ax[n / n_cols, n % n_cols].plot(x[~confidence.isnull().values], data.Confidence[data.Subject == s][~confidence.isnull()],
                          color='#2E6AF3')
    r = np.corrcoef(confidence[~confidence.isnull().values],
                data.irr_uncertainty[data.Subject == s][~confidence.isnull()])
    ax[n / n_cols, n % n_cols].set_title("Subject {0} irreducible uncertainty, R = {1}".format(s, np.round(r[0][1], 2)),
                               fontweight='bold')

plt.tight_layout()

fname = os.path.join(data_dir, 'irreducible_uncertainty_{0}'.format(time.strftime("%d_%m_%Y")))
plt.savefig(fname + '.png')
plt.savefig(fname + '.pdf')

# Plot confidence against estimation uncertainty

f, ax = plt.subplots(5, n_cols, figsize=(4.25 * n_cols, 10))

for n, s in enumerate(np.unique(data.Subject)):
    x = np.arange(0, len(data[data.Subject == s]))
    confidence = data.Confidence[data.Subject == s]
    r = np.corrcoef(confidence[~confidence.isnull().values],
                data.est_uncertainty[data.Subject == s][~confidence.isnull()])
    est_u = data.est_uncertainty[data.Subject == s]
    est_u /= est_u.max()  # scale uncertainty to 0-1 to display it with confidence
    ax[n / n_cols, n % n_cols].plot(x, est_u, color='#808080')
    ax[n / n_cols, n % n_cols].plot(x[~confidence.isnull().values], data.Confidence[data.Subject == s][~confidence.isnull()],
                          color='#FFAA00')
    ax[n / n_cols, n % n_cols].set_title("Subject {0} estimation uncertainty, R = {1}".format(s, np.round(r[0][1], 2)),
                               fontweight='bold')

plt.tight_layout()

fname = os.path.join(data_dir, 'estimation_uncertainty_{0}'.format(time.strftime("%d_%m_%Y")))
plt.savefig(fname + '.png')
plt.savefig(fname + '.pdf')
