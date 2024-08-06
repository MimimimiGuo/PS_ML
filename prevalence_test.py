import numpy as np
import pandas as pd


n = 500 # number of observations per variable/ number of patients
a0_treat = -10 # -22 t_pre = 0.01; -13, t_pre = 0.05; -10 t_pre = 0.1; -5.3 t_pre = 0.25; -1.3 t_pre = 0.5
b0_outcome = -0.8 # -11 y_pre = 0.02; -24 y_pre = 0.001; -13.5 y_pre = 0.01; -8.2 y_pre = 0.05; -0.8 Y_PRE = 0.5
b_treat = np.log(2)
n_confound_norm = 25  # number of normal distributed confounders
n_confound_binm = 25 # number of binomial distributed confounders
# n_confound_gamm = 30 # number of gamma distributed confounders
n_instruml_binm = 5 # number of binomial distributed instrumental variables
n_risk_binm = 1  # number of binomial distributed risk factors
n_trials = 1
interaction_percent = 0.3
def simulation_first(n, a0_treat, b0_outcome, b_treat, n_confound_norm, n_confound_binm, n_instruml_binm,
                     n_risk_binm, n_trials, num_sim):
    # np.random.seed(num_sim)
    ind = 0
    while ind == 0:
        simdata = pd.DataFrame()
        logit_p = np.append(a0_treat, np.append(
            np.random.normal(0.2, 0.5, n_confound_norm + n_confound_binm +
                             int(interaction_percent*(n_confound_norm + n_confound_binm))),
            np.random.normal(0, 0.1, n_instruml_binm)))
        simdata = simdata.append(pd.Series(np.ones(n)), ignore_index=True)  # add a ‘1’s row for a0_treat, 0.2, 0.5
        mean = np.arange(0, 1, 0.1).tolist()
        matrixSize = 10
        A = np.random.rand(matrixSize, matrixSize)
        matrix = np.dot(A, A.transpose())

        multi_normal = np.random.multivariate_normal(mean, matrix, n)
        simdata = simdata.append(pd.DataFrame(multi_normal.transpose()), ignore_index=True)  # 10 normal
        for j in range(0, 5):
            sigma = 1
            mu = 0
            p = 0.3
            x = pd.Series(np.random.normal(mu, sigma, n))
            x = x ** 3  # 10% quadratic term
            simdata = simdata.append(x, ignore_index=True)  # 5 normal
            x_binm = pd.Series(np.random.binomial(n_trials, p, n))
            simdata = simdata.append(x * x_binm, ignore_index=True)  # 10% x_norm^3 and x_binom interaction
        for j in range(0, 5):
            sigma = 1
            mu = 0
            x = pd.Series(np.random.normal(mu, sigma, n))
            simdata = simdata.append(x, ignore_index=True)  # 5 normal
            x_binm = pd.Series(np.random.binomial(n_trials, p, n))
            simdata = simdata.append(x * x_binm, ignore_index=True)  # 10% x_norm and x_binom interaction

        for j in range(0, 5):
            sigma = 1
            mu = 0
            x = pd.Series(np.random.normal(mu, sigma, n))
            simdata = simdata.append(x, ignore_index=True)  # 5 normal
            x_1 = pd.Series(np.random.normal(mu, sigma, n))
            simdata = simdata.append(x * x_1, ignore_index=True)  # 10% x_norm and x_norm interaction
        for j in range(0, 25):
            p = 0.2  # np.random.uniform(0.01, 1)
            x = pd.Series(np.random.binomial(n_trials, p, n))
            simdata = simdata.append(x, ignore_index=True)  #25 binomial

        for j in range(0, n_instruml_binm):
            p = 0.4
            x = pd.Series(np.random.binomial(n_trials, p, n))
            simdata = simdata.append(x, ignore_index=True)

        # swap row and column so that each column is a variable
        simdata = simdata.T
        log_treat = simdata.dot(logit_p)
        ps_treat = np.exp(log_treat) / (1 + np.exp(log_treat))
        t = np.random.binomial(n_trials, ps_treat, n)

        simdata = simdata.T

        # add risk factor for modelling outcome Y
        for j in range(0, n_risk_binm):
            p = 0.5
            x = pd.Series(np.random.binomial(n_trials, p, n))
            simdata = simdata.append(x, ignore_index=True)

        rng = range(1, n_confound_norm + n_confound_binm + n_instruml_binm +
                    int(interaction_percent*(n_confound_norm + n_confound_binm)) + n_risk_binm + 1)
        new_cols = ['const'] + ['var_' + str(i) for i in rng]

        # add treatment
        simdata = simdata.T
        simdata.columns = new_cols
        simdata['treatment'] = pd.Series(t)
        logit_p_out = np.append(np.append(np.append(np.append(
            b0_outcome,np.random.normal(0.1, 0.3, n_confound_norm + n_confound_binm +
                                        int(interaction_percent*(n_confound_norm + n_confound_binm)))),
            np.zeros(n_instruml_binm)), np.random.normal(0.1, 0.2, n_risk_binm)), b_treat)

        log_out = simdata.dot(logit_p_out)
        drs_y = np.exp(log_out) / (1 + np.exp(log_out))  # 1 - np.exp(-np.exp(log_out))
        y = np.random.binomial(n_trials, drs_y, n)

        t_pre = (sum(t == 1) / n)
        y_pre = (sum(y == 1) / n)
        ind = 1

    pre = [t_pre,y_pre]
    return(pre)

def check_pre(n, a0_treat, b0_outcome, b_treat, n_confound_norm, n_confound_binm, n_instruml_binm, n_risk_binm, n_trials, target_pre_t, target_pre_y):
    ind_pre_t = 0
    ind_pre_y = 0
    while ind_pre_t == 0 or ind_pre_y == 0:
        all_d = []
        for i in range(0,1000):
            h = simulation_first(n, a0_treat, b0_outcome, b_treat, n_confound_norm, n_confound_binm, n_instruml_binm, n_risk_binm, n_trials, i)
            all_d.append(h)
        res = np.matrix(all_d)
        if np.mean(res, axis=0)[:,0] > target_pre_t+0.05:
            a0_treat = a0_treat-0.5
        elif np.mean(res, axis=0)[:,0] < target_pre_t-0.05:
            a0_treat = a0_treat+0.5
        else:
            ind_pre_t = 1

        if np.mean(res, axis=0)[:,1] > target_pre_y+0.05:
            b0_outcome = b0_outcome - 0.5
        elif np.mean(res, axis=0)[:,1] < target_pre_y-0.05:
            b0_outcome = b0_outcome + 0.5
        else:
            ind_pre_y = 1
    return([a0_treat, b0_outcome])



check_pre(n, a0_treat, b0_outcome, b_treat, n_confound_norm, n_confound_binm, n_instruml_binm, n_risk_binm, n_trials, target_pre_t, target_pre_y)
