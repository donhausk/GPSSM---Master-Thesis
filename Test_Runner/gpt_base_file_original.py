import sys, os
sys.path.append('/home/kongi/GPT-Library/')
sys.path.append('/home/kongi/GPT-Library/GPflow/')
sys.path.append('/home/kongi/GPT-Library/GPt/')




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


ex = Experiment('process_noise_kink_experiment')
# ex.observers.append(MongoObserver.create())
ex.observers.append(FileStorageObserver.create('./sacred_observer'))

@ex.config
def ex_config():
    D = 1
    T = 10
    n_seq = 1
    batch_size = 1

    model_class = 'GPSSM_MS_VCDT'

    optimizer = 'adam'
    learning_rate = 0.001
    momentum = 0.
    use_nesterov = True
    maxiter = int(2e5)

    parallel_iterations = 1

    process_noise_var = 0.25
    emission_noise_var = 0.25

    Q_init_scale = 1e-2
    R_init_scale = None
    Ucov_chol_init_scale = 1e-1
    X_chol_init_scale = 1e-1
    kern_var_init = 10.

    init_at_gp_on_Y = False

    train_R = False
    train_C = False
    train_b = False
    train_Q = True
    train_kern_var = True
    train_Z = False
    train_As = None
    train_bs = None
    train_Ss = None

    n_samples = int(1e2)
    n_ind_pts = int(1e2)

    base_save_path = os.path.expanduser('~') + '/GPT-Library/'
    init_from_disk = None

    test = False
    filter_length = 10
    test_length = 30
    test_seq = 10
    test_samples = None


@ex.automain
@LogFileWriter(ex)
def run_experiment(_seed, D, T, n_seq, batch_size, model_class,
                   optimizer, learning_rate, momentum, use_nesterov, maxiter, parallel_iterations,
                   process_noise_var, emission_noise_var,
                   Q_init_scale, R_init_scale, Ucov_chol_init_scale, X_chol_init_scale, kern_var_init, init_at_gp_on_Y,
                   train_R, train_C, train_b, train_Q, train_kern_var, train_Z, train_As, train_bs, train_Ss,
                   n_samples, n_ind_pts, base_save_path, init_from_disk,
                   test, filter_length, test_length, test_seq, test_samples):

    tf.set_random_seed(_seed)
    np.random.seed(_seed)
    tr = KinkTransitions(dim=D, Q=np.ones(D) * process_noise_var, name='KinkTransitions_datagen')
    em = GaussianEmissions(obs_dim=D, R=np.eye(D) * emission_noise_var, name='GaussianEmissions_datagen')

    SSM_datagen = SSM(X_init=np.zeros((T, D)), Y=np.zeros((T, D)), transitions=tr, emissions=em, name='SSM_datagen')

    X, Y = SSM_datagen.sample(T, N=n_seq)
    X, Y = list(X), list(Y)

    if test:
        X_test, Y_test = SSM_datagen.sample(filter_length + test_length, N=test_seq)
        Y_test_filter = list(Y_test[:, :filter_length])

    gaussian_GPSSM_name = 'SSM_nf_MULTISEQ'

    model_classes = [gaussian_GPSSM_name, 'PRSSM_MS_MF', 'PRSSM_MS', 'GPSSM_MS_MF', 'GPSSM_MS_SPURIOUS',
                     'GPSSM_MS_FITC', 'GPSSM_MS_CUBIC', 'GPSSM_MS_VCDT', 'GPSSM_MS_FITC_SAMPLE_F',
                     'GPSSM_VCDT_Stick_Land']

    if model_class in model_classes:
        model_class_python = eval(model_class)
    else:
        raise ValueError('Unknown model class')


    kern = gp.kernels.RBF(D, variance=kern_var_init, ARD=True, name='GPSSM_kern')
    Z = np.linspace(-8., 3., n_ind_pts)[:, None]
    mean_fn = mean_fns.Zero(D, name='GPSSM_mean_fn')
    Q_diag = np.ones(D) * Q_init_scale
    Ucov_chol = np.tile(np.eye(n_ind_pts)[None, ...], [D, 1, 1]) * Ucov_chol_init_scale

    if init_from_disk is not None and init_from_disk[-4:] == '.npy':
        GPSSM_params = np.load(init_from_disk).ravel()[0]
        if 'GPSSM/Q_sqrt' in GPSSM_params.keys():
            Q_diag = GPSSM_params['GPSSM/Q_sqrt'] ** 2.

    elif init_at_gp_on_Y:
        X_gp = np.concatenate([y[:-1] for y in Y])
        Y_gp = np.concatenate([y[1:] for y in Y])

        gpr = gp.models.SVGP(X_gp, Y_gp, kern,
                             gp.likelihoods.Gaussian(variance=Q_diag, name='GPR_likelihood'),
                             Z=Z, mean_function=mean_fn)
        gpr.likelihood.trainable = train_Q
        gpr.kern.variance.trainable = train_kern_var
        gpr.feature.trainable = train_Z
        opt = gp.train.ScipyOptimizer()
        opt.minimize(gpr)

    if R_init_scale is not None:
        em.Rchol = np.eye(D) * (R_init_scale ** 0.5)
        em.compile()

    if model_class == 'GPSSM_VCDT_Stick_Land':
        assert n_seq == 1
        Y = Y[0]
        extra_kwargs = {}
    else:
        extra_kwargs = {'batch_size': batch_size}
    model = model_class_python(D,
                               Y,
                               emissions=em,
                               kern=kern,
                               Z=Z,
                               mean_fn=mean_fn,
                               Q_diag=Q_diag,
                               Ucov_chol=Ucov_chol,
                               n_samples=n_samples,
                               parallel_iterations=parallel_iterations,
                               seed=None,
                               name='GPSSM',
                               **extra_kwargs)

    transitions = model

    if train_As is not None: model.As.trainable = train_As
    if train_bs is not None: model.bs.trainable = train_bs
    if train_Ss is not None: model.S_chols.trainable = train_Ss


    transitions.Q_sqrt.trainable = train_Q
    try:
        transitions.kern.kern.variance.trainable = train_kern_var
    except:
        warnings.warn('Could not set trainable status of the kernel\'s variance: default is trainable')
    transitions.Z.trainable = train_Z

    model.emissions.Rchol.trainable = train_R
    model.emissions.C.trainable = train_C
    model.emissions.bias.trainable = train_b

    session_conf = tf.ConfigProto( intra_op_parallelism_threads=4, inter_op_parallelism_threads=4)

    session = model.enquire_session()

    if test: assert init_from_disk is not None, 'Have to initialise before testing'
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

    elif init_at_gp_on_Y:
        transitions.Q_sqrt = gpr.likelihood.variance.value ** 0.5
        if model_class != gaussian_GPSSM_name:
            model.S_chols = model.S_chols.value * 0. + gpr.likelihood.variance.value ** 0.5
        transitions.Umu = gpr.q_mu.value.T
        transitions.Ucov_chol = gpr.q_sqrt.value
        transitions.kern.kern.variance = gpr.kern.variance.value
        transitions.kern.kern.lengthscales = gpr.kern.lengthscales.value
        transitions.Z.feat.Z = gpr.feature.Z.value
        transitions.mean_fn.assign(gpr.mean_function.read_values())
        model.compile()
        print('================== Successfully initialised using a GP fit on the observations ==================')
        print('Likelihood at initialisation from GP fit on the observations:', model.compute_log_likelihood())


    if test:
        batch_size = None


        model = model_class_python(D, Y_test_filter, emissions=model.emissions,
                                   px1_mu=None, px1_cov=None,
                                   kern=model.kern.kern,
                                   Z=model.Z.feat.Z.value,
                                   mean_fn=model.mean_fn,
                                   Q_diag=model.Q_sqrt.value.copy() ** 2.,
                                   Umu=model.Umu.value.copy(),
                                   Ucov_chol=model.Ucov_chol.value.copy(),
                                   qx1_mu=None, qx1_cov=None, As=None, bs=None, Ss=None,
                                   n_samples=n_samples, batch_size=batch_size,
                                   seed=None, parallel_iterations=parallel_iterations,
                                   name='GPSSM_posterior')

        model.trainable = False

        model.qx1_mu.trainable = True
        model.qx1_cov_chol.trainable = True
        if 'PRSSM' not in model_class:
            model.As.trainable = True
            model.bs.trainable = True
            model.S_chols.trainable = True

        print('Likelihood at initialisation on filtering test sequences:', model.compute_log_likelihood())


    # Monitoring:
    experiment_id = model_class \
                    + '__T_' + str(T) + '__n_seq_' + str(n_seq) \
                    + '__Q_' + str(process_noise_var) + '__R_' + str(emission_noise_var) \
                    + '__n_samples_' + str(n_samples) + '__M_' + str(n_ind_pts)

    if test:
        experiment_id += '__test'
    if init_from_disk is not None:
        experiment_id += '__initialised'

    save_path = os.path.join(base_save_path, experiment_id + '__' + str(datetime.now()))


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


    if test:
        if test_samples is not None:
            model.n_samples = test_samples
            model.compile()

        if model_class == gaussian_GPSSM_name:
            X_samples_filter = [np.transpose(xf, [1, 0, 2]) for xf in session.run(model._build_sample_qx())]
            model_to_sample = GPSSM_CUBIC(D, Y_test[0], emissions=model.emissions,
                                          px1_mu=None, px1_cov=None,
                                          kern=model.transitions.kern.kern,
                                          Z=model.transitions.Z.feat.Z.value,
                                          mean_fn=model.transitions.mean_fn,
                                          Q_diag=model.transitions.Q_sqrt.value.copy() ** 2.,
                                          Umu=model.transitions.Umu.value.copy(),
                                          Ucov_chol=model.transitions.Ucov_chol.value.copy(),
                                          qx1_mu=None, qx1_cov=None, As=None, bs=None, Ss=None,
                                          n_samples=n_samples, seed=None, parallel_iterations=parallel_iterations,
                                          name='GPSSM_posterior_sampling')
        else:
            X_samples_filter = session.run(model._build_sample()[0])
            model_to_sample = model

        Y_samples_filter = [model.emissions.sample_conditional(xf) for xf in X_samples_filter]
        X_samples, Y_samples, SE, RMSE, logp, NLPP, NLPP_mean, model_params = [], [], [], [], [], [], [], []
        for i in range(len(Y_test)):
            X_samples_i, Y_samples_i = model_to_sample.sample(T=len(Y_test[i]) - filter_length,
                                                              x0_samples=X_samples_filter[i][-1],
                                                              inputs=None,
                                                              cubic=False)
            X_samples_i = np.concatenate([X_samples_filter[i], X_samples_i[1:]], 0)
            Y_samples_i = np.concatenate([Y_samples_filter[i], Y_samples_i[1:]], 0)
            SE_i = np.square(Y_samples_i.mean(1) - Y_test[i])
            RMSE_i = np.sqrt(np.mean(np.sum(SE_i, -1)))
            logp_i = session.run(
                model_to_sample.emissions.logp(tf.constant(X_samples_i), tf.constant(Y_test[i][:, None, :])))
            NLPP_i = - log(mean(exp(logp_i), axis=-1)) # We do not simply average over the logs, but ove the real
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

        np.savez(os.path.join(matching_dir, 'test_results.npz'),
                 X_samples=X_samples, Y_samples=Y_samples, X_test=X_test, Y_test=Y_test,
                 SE=SE, RMSE=RMSE, logp=logp, NLPP=NLPP, NLPP_mean=NLPP_mean, filter_length=filter_length,
                 model_params=model_params)
