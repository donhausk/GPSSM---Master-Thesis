import sys, os
sys.path.append('/home/kongi/GPT-Library/')
sys.path.append('/home/kongi/GPT-Library/GPflow/')
sys.path.append('/home/kongi/GPT-Library/GPt/')


import argparse
parser = argparse.ArgumentParser(description='Run GPT Experiment')
parser.add_argument('--name',  help='Name of the test')
parser.add_argument('--dir',help='Directory to store the results')
parser.add_argument('--largestorage', help='Directory for the large files')
parser.add_argument('--data', help='Path from which the data is loaded...')
parser.add_argument('--n_ind_pts',  type = int,default=20, help='Inducing Points')
parser.add_argument('--T', type = int, default=100,  help='Length of the time series')
parser.add_argument('--D', type = int, help='Dimensino of the latent states')
parser.add_argument('--model_class', help='The model which we would like to optimize')
parser.add_argument('--emission_model', default="gaussian", help='The emission model')
parser.add_argument('--filter_length',type = int,default=10, help='Filter length for multi step ahed predicitons')
parser.add_argument('--test_length',type = int,default=30, help='Test length for multi step ahead predictions')
parser.add_argument('--n_seq',type = int,default=1, help='Amount of parallel sequences')
parser.add_argument('--n_iter',type = int,default=int(1e5), help='Amount of parallel sequences')
parser.add_argument('--process_noise_var',type = float,default=0.25, help='Process Noise Variance')
parser.add_argument('--emission_noise_var',type = float,default=0.25, help='Emission Noise Variance')
parser.add_argument('--seed',type = int,default=1, help='Random Seed')
parser.add_argument('--train_emission', type = bool ,default=False, help='Train Emission')





from glob import glob
import numpy as np
import numpy.random as rnd
import pandas as pd
import tensorflow as tf


import gpflow as gp

from GPt.emissions import *
from GPt.transitions import *
#from GPt.bijectors import *
from GPt.ssm import *
from GPt.gpssm import *
from GPt.gpssm_multiseq import *
from GPt.gpssm_models import *
from datetime import datetime
import warnings

from gpflow.training import monitor as mon

from sacred.observers import MongoObserver, FileStorageObserver
from sacred.stflow import LogFileWriter
from sacred import Experiment

from gpflow import settings, params_as_tensors, autoflow
from gpflow.logdensities import mvn_logp
NARENDRALI_CONSTANTS = [ 0.52, 0.48]

class NarendraLiEmission(gp.likelihoods.Likelihood):
    def __init__(self, noise_var=0.1,  name=None):
        super().__init__(name=name)
        self.REQUIRE_FULL_COV = True
        self.latent_dim =2
        self.obs_dim = 1
        self.noise = tf.constant(noise_var, dtype= tf.float64)
    @params_as_tensors
    def conditional_mean(self, X):
        """
        :param X: latent state (T x E) or (n_samples x T x E)
        :return: mu(Y)|X (T x D) or (n_samples x T x D)
        """
        if X.shape.ndims == 3:
            Xvals = tf.reshape(X, [-1, tf.shape(X)[-1]])
            Ymu = tf.math.divide(Xvals[:, 0:1], 1 +NARENDRALI_CONSTANTS[0]*tf.sin(Xvals[:, 1]) + NARENDRALI_CONSTANTS[1]*tf.sin(Xvals[:,0]))
            return tf.reshape(Ymu, [tf.shape(X)[0], tf.shape(X)[1], self.obs_dim])
        else:
            return tf.math.divide(Xvals[:,0:1], 1 +NARENDRALI_CONSTANTS[0]*tf.sin(Xvals[:,1]) + NARENDRALI_CONSTANTS[1]*tf.sin(Xvals[:,0]))
    @params_as_tensors
    def conditional_variance(self, X):
        """
        :param X: latent state (T x E) or (n_samples x T x E)
        :return: cov(Y)|X (T x D x D) or (n_samples x T x D x D)
        """
        if X.shape.ndims == 3:
            return tf.fill([tf.shape(X)[0], tf.shape(X)[1], 1, 1], self.noise)
        else:
            return tf.fill([tf.shape(X)[0], 1, 1], self.noise)
    @params_as_tensors
    def logp(self, X, Y):
        """
        :param X: latent state (T x E) or (n_samples x T x E)
        :param Y: observations (T x D)
        :return: \log P(Y|X(n)) (T) or (n_samples x T)
        """
        d = Y - self.conditional_mean(X)
        dim_perm = [2, 0, 1] if X.shape.ndims == 3 else [1, 0]
        return mvn_logp(tf.transpose(d, dim_perm), tf.ones([1,1], dtype = tf.float64)*tf.sqrt(self.noise))

    def sample_conditional(self, X):
        X_in = X if X.ndim == 2 else X.reshape(-1, X.shape[-1])
        noise_samples = np.random.randn(X_in.shape[0], self.obs_dim)* tf.sqrt(self.noise)
        Y = self.conditional_mean(X) + noise_samples
        if X.ndim != 2:
            Y = Y.reshape(*X.shape[:-1], self.obs_dim)
        return Y
    @params_as_tensors
    def predict_mean_and_var(self, Xmu, Xcov):
        raise NotImplementedError

    @params_as_tensors
    def variational_expectations(self, Xmu, Xcov, Y):
        raise NotImplementedError

    @autoflow((settings.float_type,), (settings.float_type,))
    def compute_predictive_mean_and_var(self, Xmu, Xcov):
        return self.predict_mean_and_var(Xmu, Xcov)

    @autoflow((settings.float_type,), (settings.float_type,), (settings.float_type,))
    def compute_variational_expectations(self, Xmu, Xcov, Y):
        return self.variational_expectations(Xmu, Xcov, Y)












ex = Experiment('process_noise_kink_experiment')
#ex.observers.append(MongoObserver.create())
args = vars(parser.parse_args())

# This is for sacred
sys.argv = [sys.argv[0]]

ex.observers.append(FileStorageObserver.create( os.path.join(args["largestorage"], args["name"]) +'/sacred_observer'))
#ex.observers.append(FileStorageObserver.create('./sacred_observer'))

@ex.config
def ex_config():
    D = args["D"]
    T = args["T"]
    n_seq = args["n_seq"]
    #batch_size = 1
    batch_size = None
    model_class = args["model_class"]

    optimizer = 'adam'
    learning_rate = 0.001
    momentum = 0.
    use_nesterov = True
    maxiter = args["n_iter"]

    parallel_iterations = 1

    process_noise_var = args["process_noise_var"]
    emission_noise_var = args["emission_noise_var"]

    Q_init_scale = 1e-2
    R_init_scale = None
    Ucov_chol_init_scale = 1e-1
    X_chol_init_scale = 1e-1
    kern_var_init = 10.

    init_at_gp_on_Y = False


    train_Q = True
    train_kern_var = True
    train_Z = False
    train_As = None
    train_bs = None
    train_Ss = None

    n_samples = int(1e2)
    n_ind_pts = args["n_ind_pts"]

    base_save_path = args["dir"]
    init_from_disk = None

    test = True
    filter_length = args["filter_length"]
    test_length = args["test_length"]
    test_seq = 10
    test_samples = None
    args = args


@ex.automain
@LogFileWriter(ex)
def run_experiment(_seed, D, T, n_seq, batch_size, model_class,
                   optimizer, learning_rate, momentum, use_nesterov, maxiter, parallel_iterations,
                   process_noise_var, emission_noise_var,
                   Q_init_scale, R_init_scale, Ucov_chol_init_scale, X_chol_init_scale, kern_var_init, init_at_gp_on_Y,
                   train_Q, train_kern_var, train_Z, train_As, train_bs, train_Ss,
                   n_samples, n_ind_pts, base_save_path, init_from_disk,
                   test, filter_length, test_length, test_seq, test_samples, args):


    tf.set_random_seed(_seed)
    np.random.seed(_seed)

    #tr = KinkTransitions(dim=D, Q=np.ones(D) * process_noise_var, name='KinkTransitions_datagen')

    #SSM_datagen = SSM(X_init=np.zeros((T, D)), Y=np.zeros((T, D)), transitions=tr, emissions=em, name='SSM_datagen')

    if args["emission_model"] == "gaussian":
        em = GaussianEmissions(obs_dim=D, R=np.eye(D) * emission_noise_var, name='GaussianEmissions_datagen')
    elif args["emission_model"] == "narendrali":
        em = NarendraLiEmission(noise_var = emission_noise_var, name='NarendraLiEmission')





    Y, U_0 =  np.load(args["data"] + "Y_" +args["name"] +".npz"), np.load(args["data"] + "U_" +args["name"] +".npz")
    Y, U =  [np.transpose(Y[:,:,i]) for i in range(Y.shape[2])],  None if U_0.shape[0] == 0 else  [  np.transpose(U_0[:,:,i]) for i in range(U_0.shape[2])]


    Xtest, Ytest, Utest = np.load(args["data"] + "Xtest_" +args["name"] +".npz"),  np.load(args["data"] + "Ytest_" +args["name"] +".npz"), np.load(args["data"] + "Utest_" +args["name"] +".npz")
    Xtest,Ytest, Utest = None if Xtest.shape[0]== 0 else [np.transpose(Xtest[:,:,i]) for i in range(Xtest.shape[2])],  [np.transpose(Ytest[:,:,i]) for i in range(Ytest.shape[2])],     None if U_0.shape[0] == 0 else [ np.transpose(Utest[:,:,i]) for i in range(Utest.shape[2])]

    Xplot, Uplot = np.transpose(np.load(args["data"] + "Xplot_" +args["name"] +".npz")), np.transpose(np.load(args["data"] + "Uplot_" +args["name"] +".npz"))
    Uplot = None if U_0.shape[0] == 0 else Uplot
    Xplot = None if Xplot.shape[1]== 0 else Xplot # WE have already taken the transpose
    Ypred, Upred =  np.load(args["data"] + "Ypred_" +args["name"] +".npz"), np.load(args["data"] + "Upred_" +args["name"] +".npz")
    Ypred, Upred = None if Ypred.shape[0]== 0 else [np.transpose(Ypred[:,:,i]) for i in range(Ypred.shape[2])],  None if U_0.shape[0] == 0 else [ np.transpose(Upred[:,:,i]) for i in range(Upred.shape[2])]

    Ypredseqahead, Upredseqahead =  np.load(args["data"] + "Ypredseqahead_" +args["name"] +".npz"), np.load(args["data"] + "Upredseqahead_" +args["name"] +".npz")
    Ypredseqahead, Upredseqahead =  None if Ypredseqahead.shape[0]== 0 else  [np.transpose(Ypredseqahead[:,:,i]) for i in range(Ypredseqahead.shape[2])],  [None if U_0.shape[0] == 0 else  np.transpose(Upredseqahead[:,:,i]) for i in range(Upredseqahead.shape[2])]






    gaussian_GPSSM_name = 'SSM_SG_MultipleSequences'

    model_classes = [gaussian_GPSSM_name, 'PRSSM_MS_MF', 'PRSSM_MS', 'GPSSM_MS_MF', 'GPSSM_MS_SPURIOUS',
                     'GPSSM_MS_FITC', 'GPSSM_MS_CUBIC', 'GPSSM_MS_VCDT', 'GPSSM_MS_FITC_SAMPLE_F', 'GPSSM_MS_FactorizedLinear', 'GPSSM_MS_FactorizedNonLinear']

    if model_class in model_classes:
        model_class_python = eval(model_class)
    else:
        raise ValueError('Unknown model class')


    #kern = gp.kernels.Matern32(D, variance=kern_var_init, ARD=True, name='GPSSM_kern')
    #Z = np.linspace(-8., 3., n_ind_pts)[:, None]
    #mean_fn = mean_fns.Zero(D, name='GPSSM_mean_fn')
    #Q_diag = np.ones(D) * Q_init_scale
    #Ucov_chol = np.tile(np.eye(n_ind_pts)[None, ...], [D, 1, 1]) * Ucov_chol_init_scale

    if init_from_disk is not None and init_from_disk[-4:] == '.npy':
        GPSSM_params = np.load(init_from_disk).ravel()[0]
        if 'GPSSM/Q_sqrt' in GPSSM_params.keys():
            Q_diag = GPSSM_params['GPSSM/Q_sqrt'] ** 2.

    if R_init_scale is not None:
        em.Rchol = np.eye(D) * (R_init_scale ** 0.5)
        em.compile()


    extra_kwargs = {'batch_size': batch_size}
    model = model_class_python(D,
                               Y,
                               inputs=U,
                               emissions=em,
                               n_ind_pts = n_ind_pts,
                               n_samples=n_samples,
                               seed=None,
                               name='GPSSM',
                               **extra_kwargs)

    if args["emission_model"] == "gaussian":
        if args["train_emission"]:
            model.emissions.Rchol.trainable = False
            model.emissions.C.trainable = False
            model.emissions.bias.trainable = False
        else:
            model.emissions.Rchol.trainable = False
            model.emissions.C.trainable = False
            model.emissions.bias.trainable = False
    elif args["emission_model"] == "narendrali":
        model.emissions.trainable= args["train_emission"]


    model.emissions.Rchol.trainable = False
    model.emissions.C.trainable = False
    model.emissions.bias.trainable = False


    transitions = model

    if train_As is not None: model.As.trainable = train_As
    if train_bs is not None: model.bs.trainable = train_bs
    if train_Ss is not None: model.S_chols.trainable = train_Ss

    if model_class !=  'GPSSM_MS_FactorizedLinear':
        transitions.Q_sqrt.trainable = train_Q
        transitions.Z.trainable = train_Z

    try:
        transitions.kern.kern.variance.trainable = train_kern_var
    except:
        warnings.warn('Could not set trainable status of the kernel\'s variance: default is trainable')


    session = model.enquire_session()

    if init_from_disk is not None:
        if init_from_disk[-4:] != '.npy':
            matching_dirs = glob(init_from_disk)
            assert len(matching_dirs) == 1, 'More/Less than one matching run found'
            matching_dir = matching_dirs[0]
            checkpoint_loc = tf.train.latest_checkpoint(matching_dir + '/checkpoints')
            tf.train.Saver().restore(session, checkpoint_loc)
            GPSSM_params = model.read_values(session)
        if 'PRSSM' in model_class:
            offending_keys = [k for k in GPSSM_params.keys() if ('/As' in k or '/bs' in k or '/S_chols' in k)]
            for k in offending_keys: GPSSM_params.pop(k)
        model.assign(GPSSM_params)
        model.compile()
        print('================== Successfully initialised GPSSM params from: ' + init_from_disk + ' ==================')
        print('Likelihood at initialisation loaded from disk:', model.compute_log_likelihood())


    experiment_id = args["name"]
    save_path = os.path.join(args["largestorage"], experiment_id)

    if optimizer == 'adam':
        optimizer = gp.train.AdamOptimizer(learning_rate=learning_rate)
    elif optimizer == 'momentum':
        optimizer = gp.train.MomentumOptimizer(learning_rate=learning_rate,
                                               momentum=momentum, use_nesterov=use_nesterov)
    else:
        raise ValueError('Unknown optimizer')

    global_step = mon.create_global_step(session)
    print_task = mon.PrintTimingsTask() \
        .with_name('print') \
        .with_condition(mon.PeriodicIterationCondition(100))

    checkpoint_task = mon.CheckpointTask(checkpoint_dir=save_path + '/checkpoints',
                                         saver=tf.train.Saver(max_to_keep=100, save_relative_paths=True)) \
        .with_name('checkpoint') \
        .with_condition(mon.PeriodicIterationCondition(10000)) \
        .with_exit_condition(True)

    with mon.LogdirWriter(save_path + '/tensorboard') as writer:
         tensorboard_task = mon.ModelToTensorBoardTask(writer, model, only_scalars=False) \
             .with_name('tensorboard') \
             .with_condition(mon.PeriodicIterationCondition(50)) \
             .with_exit_condition(True)

         monitor_tasks = [print_task, tensorboard_task, checkpoint_task]

         with mon.Monitor(monitor_tasks, session, global_step, print_summary=True) as monitor:
             optimizer.minimize(model, step_callback=monitor, global_step=global_step, maxiter=maxiter)


    finallog = model.compute_log_likelihood()
    if model_class in ['GPSSM_MS_FactorizedLinear', gaussian_GPSSM_name]:
        #Z_Inds =  model.transitions.Z.feat.Z.value
        #Mu_Inds = model.transitions.Umu.value
        #Sigma_Inds = model.transitions.Ucov_chol.value
        X_samples = session.run(model._build_sample_qx())
    else:
        X_samples = session.run(model._build_sample()[0])
        #Z_Inds = vcdt.Z.feat.Z.value
        #Mu_Inds = model.Umu.value
        #Sigma_Inds = model.Ucov_chol.value

    #np.savez(os.path.join(save_path, 'LastState.npz'),
    #             Z_Inds=Z_Inds, Mu_Inds=Mu_Inds, Sigma_Inds=Sigma_Inds,log= finallog, X_Samples = X_samples)

    np.savez(os.path.join(save_path, 'LastState.npz'),log= finallog, X_Samples = X_samples)
    if Ypredseqahead is not None:

        if model_class in ['GPSSM_MS_FactorizedLinear', gaussian_GPSSM_name]:
            X_samples_filter = [np.transpose(xf, [1, 0, 2]) for xf in session.run(model._build_sample_qx())]
        else:
            X_samples_filter = session.run(model._build_sample()[0])
            model_to_sample = model

        X_samples, Y_samples, SE, RMSE, logp, NLPP, NLPP_mean, model_params = [], [], [], [], [], [], [], []
        for i in range(len(Ypredseqahead)):
            if model_class in ['GPSSM_MS_FactorizedLinear', gaussian_GPSSM_name]:
                model_to_sample = GPSSM_Cubic(D, Ypredseqahead[i], emissions=model.emissions,inputs=Upredseqahead[i],
                                              px1_mu=None, px1_cov=None,
                                              kern=model.transitions.kern.kern,
                                              Z=model.transitions.Z.feat.Z.value,
                                              mean_fn=model.transitions.mean_fn,
                                              Q_diag=model.transitions.Q_sqrt.value.copy() ** 2.,
                                              Umu=model.transitions.Umu.value.copy(),
                                              Ucov_chol=model.transitions.Ucov_chol.value.copy(),
                                              qx1_mu=None, qx1_cov=None, As=None, bs=None, Ss=None,
                                              n_samples=n_samples, seed=None,
                                              name='GPSSM_posterior_sampling')


            X_samples_i, Y_samples_i = model_to_sample.sample(T=test_length,
                                                              x0_samples=X_samples_filter[i][-1],
                                                              inputs=Upredseqahead[i],
                                                              cubic=False)
            Ypred_i = Ypredseqahead[i]

            SE_i = np.square(Y_samples_i.mean(1)[1:,:] - Ypred_i)
            RMSE_i = np.sqrt(np.mean(np.sum(SE_i, -1)))
            logp_i = session.run(
                model_to_sample.emissions.logp(tf.constant(X_samples_i[1:,:]), tf.constant(Ypred_i[:, None, :])))
            NLPP_i = - np.log(np.mean(np.exp(logp_i), axis=-1)) # We do not simply average over the logs, but ove the real
            NLPP_mean_i = NLPP_i.mean()
            print(RMSE_i, NLPP_mean_i)
            X_samples.append(X_samples_i)
            Y_samples.append(Y_samples_i)
            SE.append(SE_i)
            RMSE.append(RMSE_i)
            logp.append(logp_i)
            NLPP.append(NLPP_i)
            NLPP_mean.append(NLPP_mean_i)
            model_params.append(model.read_values(session=session))

        np.savez(os.path.join(save_path, 'test_results2.npz'),
                 X_samples=X_samples, Y_samples=Y_samples, Ypred=Ypredseqahead,
                 SE=SE, RMSE=RMSE, logp=logp, NLPP=NLPP, NLPP_mean=NLPP_mean,
                 model_params=model_params, Upred = Upredseqahead)





    pred_session = em.enquire_session()
    pred_mu = []
    pred_var = []
    if Xtest is not None:
        for i in range(len(Xtest)):
            if Utest is not None:
                Xin = np.concatenate([Xtest[i], Utest[i]], axis=1 )
            else:
                Xin = Xtest[i]
            Xin = tf.constant(Xin)
            if model_class == "GPSSM_MS_FactorizedLinear":

                F_mu, F_var = pred_session.run(
                    model.transitions.conditional(Xin, add_noise=False))
            else:
                F_mu, F_var = session.run(model._build_predict_f(Xin))

            pred_mu.extend(F_mu)
            pred_var.extend(F_var)
        np.save(args["dir"] + args["name"]+ "_pred", np.stack(np.stack(pred_mu), np.stack(pred_var)))

    if Xplot is not None :
        if Uplot is not None:
            Xin = np.concatenate([Xplot, Uplot], axis=1 )
        else:
            Xin = Xplot
        Xin = tf.constant(Xin)
        if model_class == "GPSSM_MS_FactorizedLinear":

            F_mu_plot, F_var_plot = pred_session.run(
                model.transitions.conditional(Xin, add_noise=False))
        else:
            F_mu_plot, F_var_plot = session.run(model._build_predict_f(Xin))
        np.save(args["dir"] + args["name"]+ "_plot", np.stack([F_mu_plot,F_var_plot]))

    if Ypred is not None:
        batch_size = None
        Y_test_filter = [Ypred[i][:filter_length, :] for i in range(len(Ypred))]
        U_test_filter = [Upred[i][:filter_length+1, :] for i in range(len(Ypred))]
        modeltest = model_class_python(D, Y_test_filter, emissions=model.emissions, inputs= U_test_filter,
                                   px1_mu=None, px1_cov=None,
                                   kern=model.kern.kern,
                                   Z=model.Z.feat.Z.value,
                                   mean_fn=model.mean_fn,
                                   Q_diag=model.Q_sqrt.value.copy() ** 2.,
                                   Umu=model.Umu.value.copy(),
                                   Ucov_chol=model.Ucov_chol.value.copy(),
                                   qx1_mu=None, qx1_cov=None, As=None, bs=None, Ss=None,
                                   n_samples=n_samples, batch_size=batch_size,
                                   seed=None,
                                   name='GPSSM_posterior')


        modeltest.trainable = False

        modeltest.qx1_mu.trainable = True
        modeltest.qx1_cov_chol.trainable = True
        if 'PRSSM' not in model_class:
            modeltest.As.trainable = True
            modeltest.bs.trainable = True
            modeltest.S_chols.trainable = True

        print('Likelihood at initialisation on filtering test sequences:', modeltest.compute_log_likelihood())


        # We use the same optimizer again

        global_step = mon.create_global_step(session)
        print_task = mon.PrintTimingsTask() \
            .with_name('print') \
            .with_condition(mon.PeriodicIterationCondition(100))

        checkpoint_task = mon.CheckpointTask(checkpoint_dir=save_path + '/checkpoints',
                                             saver=tf.train.Saver(max_to_keep=100, save_relative_paths=True)) \
            .with_name('checkpoint') \
            .with_condition(mon.PeriodicIterationCondition(10000)) \
            .with_exit_condition(True)

        with mon.LogdirWriter(save_path + '/tensorboard') as writer:
             tensorboard_task = mon.ModelToTensorBoardTask(writer, model, only_scalars=False) \
                 .with_name('tensorboard') \
                 .with_condition(mon.PeriodicIterationCondition(50)) \
                 .with_exit_condition(True)

             monitor_tasks = [print_task, tensorboard_task, checkpoint_task]

             with mon.Monitor(monitor_tasks, session, global_step, print_summary=True) as monitor:
                 optimizer.minimize(modeltest, step_callback=monitor, global_step=global_step, maxiter=maxiter)






        if test_samples is not None:
            modeltest.n_samples = test_samples
            modeltest.compile()

        if model_class  in ['GPSSM_MS_FactorizedLinear', gaussian_GPSSM_name]:
            X_samples_filter = [np.transpose(xf, [1, 0, 2]) for xf in session.run(modeltest._build_sample_qx())]
            model_to_sample = GPSSM_Cubic(D, Y_test_filter, emissions=modeltest.emissions, inputs =U_test_filter,
                                          px1_mu=None, px1_cov=None,
                                          kern=modeltest.transitions.kern.kern,
                                          Z=modeltest.transitions.Z.feat.Z.value,
                                          mean_fn=modeltest.transitions.mean_fn,
                                          Q_diag=modeltest.transitions.Q_sqrt.value.copy() ** 2.,
                                          Umu=modeltest.transitions.Umu.value.copy(),
                                          Ucov_chol=modeltest.transitions.Ucov_chol.value.copy(),
                                          qx1_mu=None, qx1_cov=None, As=None, bs=None, Ss=None,
                                          n_samples=n_samples, seed=None,
                                          name='GPSSM_posterior_sampling')
        else:
            X_samples_filter = session.run(modeltest._build_sample()[0])
            model_to_sample = modeltest

        Y_samples_filter = [modeltest.emissions.sample_conditional(xf[1:,:]) for xf in X_samples_filter]
        X_samples, Y_samples, SE, RMSE, logp, NLPP, NLPP_mean, model_params = [], [], [], [], [], [], [], []
        for i in range(len(Ypred)):

            X_samples_i, Y_samples_i = model_to_sample.sample(T=test_length,
                                                              x0_samples=X_samples_filter[i][-1],
                                                              inputs=Upred[filter_length:test_length+1, :],
                                                              cubic=False)
            Ypred_i = Ypred[i][:(filter_length + test_length), :]
            X_samples_i = np.concatenate([X_samples_filter[i], X_samples_i[1:]], 0)
            Y_samples_i = np.concatenate([Y_samples_filter[i], Y_samples_i[1:]], 0)
            SE_i = np.square(Y_samples_i.mean(1) - Ypred_i)
            RMSE_i = np.sqrt(np.mean(np.sum(SE_i, -1)))
            logp_i = session.run(
                model_to_sample.emissions.logp(tf.constant(X_samples_i), tf.constant(Ypred_i[:, None, :])))
            NLPP_i = - np.log(np.mean(np.exp(logp_i), axis=-1)) # We do not simply average over the logs, but ove the real
            NLPP_mean_i = NLPP_i.mean()
            print(RMSE_i, NLPP_mean_i)
            X_samples.append(X_samples_i)
            Y_samples.append(Y_samples_i)
            SE.append(SE_i)
            RMSE.append(RMSE_i)
            logp.append(logp_i)
            NLPP.append(NLPP_i)
            NLPP_mean.append(NLPP_mean_i)
            model_params.append(modeltest.read_values(session=session))

        np.savez(os.path.join(save_path, 'test_results.npz'),
                 X_samples=X_samples, Y_samples=Y_samples, Ypred=Ypred, Upred= Upred,
                 SE=SE, RMSE=RMSE, logp=logp, NLPP=NLPP, NLPP_mean=NLPP_mean, filter_length=filter_length,
                 model_params=model_params)
