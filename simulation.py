import numpy as np
import pandas as pd

from parameters import Parameters

parameters = Parameters()


class Simulation(object):

    def __init__(self,
                 iteration=None):

        self.i = iteration

    def create_x(self,
                 sigma=None,
                 mu=None
                 ):
        x = pd.Series(np.random.normal(mu,
                                       sigma,
                                       parameters.n))

    def simulation_first(self):

        # np.random.seed(num_sim)
        ind = 0
        while ind == 0:

            simdata = pd.DataFrame()
            logit_p = np.append(parameters.a0_treat,
                                np.append(np.random.normal(0.2,
                                                           0.5,
                                                           parameters.n_confound_norm + parameters.n_confound_binm +
                                                           int(parameters.interaction_percent*(
                                                               parameters.n_confound_norm + parameters.n_confound_binm))
                                                           ),
                                          np.random.normal(0,
                                                           0.1,
                                                           parameters.n_instruml_binm
                                                           )
                                          )
                                )

            # add a ‘1’s row for a0_treat, 0.2, 0.5
            simdata = simdata.append(pd.Series(np.ones(n)),
                                     ignore_index=True)
            mean = np.arange(0, 1, 0.1).tolist()
            matrixSize = 10

            A = np.random.rand(matrixSize,
                               matrixSize)
            matrix = np.dot(A,
                            A.transpose())

            multi_normal = np.random.multivariate_normal(mean,
                                                         matrix,
                                                         parameters.n)
            simdata = simdata.append(pd.DataFrame(
                multi_normal.transpose()),
                ignore_index=True)  # 10 normal

            for j in range(0, 5):

                p = 0.3
                x = self.create_x(1, 0)
                x = x**3  # 10% quadratic term

                simdata = simdata.append(x,
                                         ignore_index=True)  # 5 normal
                x_binm = pd.Series(np.random.binomial(
                    parameters.n_trials,
                    p,
                    parameters.n))
                # 10% x_norm^3 and x_binom interaction
                simdata = simdata.append(x * x_binm,
                                         ignore_index=True)

            for j in range(0, 5):

                x = self.create_x(1, 0)
                simdata = simdata.append(x,
                                         ignore_index=True)  # 5 normal
                x_binm = pd.Series(np.random.binomial(
                    parameters.n_trials,
                    p,
                    parameters.n))
                # 10% x_norm and x_binom interaction
                simdata = simdata.append(x * x_binm,
                                         ignore_index=True)

            for j in range(0, 5):
                sigma = 1
                mu = 0
                x = self.create_x(1, 0)

                simdata = simdata.append(x,
                                         ignore_index=True)  # 5 normal
                x_1 = pd.Series(np.random.normal(mu,
                                                 sigma,
                                                 parameters.n))
                # 10% x_norm and x_norm interaction
                simdata = simdata.append(x * x_1,
                                         ignore_index=True)

            for j in range(0, 25):
                p = 0.2
                x = pd.Series(np.random.binomial(
                    parameters.n_trials,
                    p,
                    parameters.n))
                simdata = simdata.append(x,
                                         ignore_index=True)  # 25 binomial

            for j in range(0, parameters.n_instruml_binm):
                p = 0.4
                x = pd.Series(np.random.binomial(
                    parameters.n_trials,
                    p,
                    parameters.n))
                simdata = simdata.append(x,
                                         ignore_index=True)

            # swap row and column so that each column is a variable
            simdata = simdata.T
            log_treat = simdata.dot(logit_p)
            ps_treat = np.exp(log_treat) / (1 + np.exp(log_treat))
            t = np.random.binomial(parameters.n_trials,
                                   ps_treat,
                                   parameters.n)

            simdata = simdata.T

            # add risk factor for modelling outcome Y
            for j in range(0, parameters.n_risk_binm):
                p = 0.5  # np.random.uniform(0.001, 1)
                x = pd.Series(np.random.binomial(
                    parameters.n_trials, p, parameters.n))
                simdata = simdata.append(x, ignore_index=True)

            rng = range(1, parameters.n_confound_norm + parameters.n_confound_binm + parameters.n_instruml_binm +
                        int(parameters.interaction_percent*(parameters.n_confound_norm + parameters.n_confound_binm)) +
                        parameters.n_risk_binm + 1)
            new_cols = ['const'] + ['var_' + str(i) for i in rng]

            # add treatment
            simdata = simdata.T
            simdata.columns = new_cols
            simdata['treatment'] = pd.Series(t)
            logit_p_out = np.append(np.append(np.append(np.append(
                parameters.b0_outcome, np.random.normal(0.1,
                                                        0.3,
                                                        parameters.n_confound_norm + parameters.n_confound_binm +
                                                        int(parameters.interaction_percent*(parameters.n_confound_norm + parameters.n_confound_binm)))),
                                                        np.zeros(parameters.n_instruml_binm)), np.random.normal(0.1, 0.2, parameters.n_risk_binm)), parameters.b_treat)

            log_out = simdata.dot(logit_p_out)
            drs_y = np.exp(log_out) / (1 + np.exp(log_out))
            y = np.random.binomial(parameters.n_trials, drs_y, parameters.n)
            if sum(t == 1)/parameters.n > 0.45 and sum(t == 1)/parameters.n <= 0.55:
                if sum(y == 1)/parameters.n > 0.4 and sum(y == 1)/parameters.n < 0.6:
                    ind = 1

        simdata['Y'] = pd.Series(y)
        simdata['DRS'] = pd.Series(drs_y)
        simdata['PS'] = pd.Series(ps_treat)
        simdata.drop(columns='const', inplace=True)
        return(simdata.to_csv(r'D:/sim_non_cov/S5/output_m{}.csv'.format(parameters.num_sim)))


for i in range(0, 1000):
    sim = Simulation(i)
    sim.simulation_first()
