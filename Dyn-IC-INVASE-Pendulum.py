#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
# import argparse
from data_generation import generate_data
import os
import json
import pandas as pd
import time
import initpath_alg
#initpath_alg.init_sys_path()
import utilmlab
import data_loader_mlab


def array2str(a):
    s = ''
    for idx, el in enumerate(a):
        s += (' ' if idx > 0 else '') + '{:0.3f}'.format(el)
    return s


def one_hot_encoder(a):
    n_values = np.max(a) + 1
    return np.eye(n_values)[a]


def load_create_data(
        data_type,
        data_out,
        is_logging_enabled=True,
        fn_csv=None,
        label_nm=None):

    df_train, df_test, dset = None, None, None
    features = None
    if data_type in data_loader_mlab.get_available_datasets() + ['show']        or fn_csv is not None:
        if fn_csv is not None:
            rval, dset = data_loader_mlab.load_dataset_from_csv(
                logger, fn_csv, label_nm)
        else:
            rval, dset = data_loader_mlab.get_dataset(data_type)
        assert rval == 0
        data_loader_mlab.dataset_log_properties(logger, dset)
        if is_logging_enabled:
            logger.info('warning no seed')
        df = dset['df']
        features = dset['features']
        labels = dset['targets']
        nsample = len(df)
        train_ratio = 0.8
        idx = np.random.permutation(nsample)
        ntrain = int(nsample * train_ratio)
        df_train = df.iloc[idx[:ntrain]]
        df_test = df.iloc[idx[ntrain:]]

        col_drop = utilmlab.col_with_nan(df)
        if is_logging_enabled and len(col_drop):
            print('warning: dropping features {}'
                  ', contains nan'.format(col_drop))
            time.sleep(2)

        features = [el for el in features if el not in col_drop]

        x_train = df_train[features].values
        y_train = df_train[labels].values
        x_test = df_test[features].values
        y_test = df_test[labels].values

        g_train, g_test = None, None

        y_train = one_hot_encoder(np.ravel(y_train))
        y_test = one_hot_encoder(np.ravel(y_test))
        if is_logging_enabled:
            logger.info('y: train:{} test:{}'.format(
                set(np.ravel(y_train)), set(np.ravel(y_test))))
    else:
        x_train, y_train, g_train = generate_data(
            n=train_N, data_type=data_type, seed=train_seed, out=data_out, x_dim = X_DIM)
        x_test,  y_test,  g_test = generate_data(
            n=test_N,  data_type=data_type, seed=test_seed,  out=data_out, x_dim = X_DIM)
    if is_logging_enabled:
        logger.info('{} {} {} {}'.format(
            x_train.shape,
            y_train.shape,
            x_test.shape,
            y_test.shape))
    return x_train, y_train, g_train, x_test, y_test,         g_test, df_train, df_test, dset, features



# In[2]:


from IPython import embed
import torch
import torch.nn as nn
import torch.nn.functional as F

# Generator (Actor) in PyTorch
class INVASE_Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(INVASE_Actor, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 100)
        self.l2 = nn.Linear(100, 100)
        self.l3 = nn.Linear(100, action_dim)


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        
        a = F.selu(self.l1(sa))
        a = F.selu(self.l2(a))
        return torch.sigmoid(self.l3(a))
        
# Discriminator (Critic) in PyTorch    
# Critic in INVASE is a classifier that provide return signal
class INVASE_Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(INVASE_Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 200)
        #self.bn1 = nn.BatchNorm1d(num_features=200)
        self.l2 = nn.Linear(200, 200)
        #self.bn2 = nn.BatchNorm1d(num_features=200)
        self.l3 = nn.Linear(200, state_dim)


    def forward(self, state, action, mask):
        #sa = torch.cat([state, action], 1)
        sa = torch.cat([state, mask* action],1)
        
        #q1 = F.selu(self.bn1(self.l1(sa)))
        #q1 = F.selu(self.bn2(self.l2(q1)))
        q1 = F.selu(self.l1(sa))
        q1 = F.selu(self.l2(q1))
        q1 = self.l3(q1)

        return q1 # prob, actually the binary classification result with softmax activation (logits)
    
# Valuefunction (Baseline) in PyTorch   
# Valuefunction in INVASE is a classifier that provide return signal
class INVASE_Baseline(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(INVASE_Baseline, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 200)
        #self.bn1 = nn.BatchNorm1d(num_features=200)
        self.l2 = nn.Linear(200, 200)
        #self.bn2 = nn.BatchNorm1d(num_features=200)
        self.l3 = nn.Linear(200, state_dim)


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        #sa = state

        q1 = F.selu(self.l1(sa))
        q1 = F.selu(self.l2(q1))
        q1 = self.l3(q1)

        return q1 # prob, actually the binary classification result with softmax activation (logits)    


class PVS():
    # 1. Initialization
    '''
    x_train: training samples
    data_type: Syn1 to Syn 6
    '''
    def __init__(self, xs_train, data_type, nepoch, is_logging_enabled=True, thres = 0.5):
        self.is_logging_enabled = is_logging_enabled
        self.latent_dim1 = 100      # Dimension of actor (generator) network
        self.latent_dim2 = 200      # Dimension of critic (discriminator) network
        
        self.batch_size = min(1000, xs_train.shape[0])      # Batch size
        self.epochs = nepoch        # Epoch size (large epoch is needed due to the policy gradient framework)
        self.lamda = 1.0           # Hyper-parameter for the number of selected features 
        self.thres = thres
        '''lamda is number of selected features? is it the coefficient?'''
        
        
        self.input_shape_state = xs_train.shape[1]     # state dimension
        self.input_shape_action = xa_train.shape[1]    # action dimension
        logger.info('input shape: {}'.format(self.input_shape_state))
        
        # Actionvation. (For Syn1 and 2, relu, others, selu)
        self.activation = 'relu' if data_type in ['Syn1','Syn2'] else 'selu'       
        
        
        self.generator = INVASE_Actor(state_dim=self.input_shape_state, action_dim = self.input_shape_action)
        self.discriminator = INVASE_Critic(state_dim=self.input_shape_state, action_dim = self.input_shape_action)
        self.valfunction = INVASE_Baseline(state_dim=self.input_shape_state, action_dim = self.input_shape_action)
        
        
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-4)#,weight_decay=1e-3)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)#,weight_decay=1e-3)
        self.valfunction_optimizer = torch.optim.Adam(self.valfunction.parameters(), lr=1e-4)#,weight_decay=1e-3)
        
    def my_loss(self, y_true, y_pred,lmd, Thr):
        # dimension of the features
        
        '''
        sel_prob: the mask generated by bernulli sampler [bs, d]
        dis_prob: prediction of the critic               [bs, state_dim]
        val_prob: prediction of the baseline model       [bs, state_dim]
        y_batch: batch of y_train                        [bs, state_dim]
        all of those variables are 'placeholders'
        '''
        
        
        d = y_pred.shape[1]        
        
        # Put all three in y_true 
        # 1. selected probability
        sel_prob = y_true[:,:d] # bs x d
        # 2. discriminator output
        dis_prob = y_true[:,d:(d+self.input_shape_state)] # bs x 2
        # 3. valfunction output
        val_prob = y_true[:,(d+self.input_shape_state):(d+self.input_shape_state*2)] # bs x 2
        # 4. ground truth
        y_final = y_true[:,(d+self.input_shape_state*2):] # bs x 2
        
        # A1. Compute the rewards of the actor network
        #embed()
        Reward1 = torch.norm(y_final - dis_prob, p=2, dim=1)  

        # A2. Compute the rewards of the actor network
        Reward2 = torch.norm(y_final - val_prob, p=2, dim=1)  

        # Difference is the rewards
        Reward =Reward2 -  Reward1

        # B. Policy gradient loss computation. 
        loss1 = Reward * torch.sum(sel_prob * torch.log(y_pred + 1e-8) + (1-sel_prob) * torch.log(1-y_pred + 1e-8), axis = 1) - lmd *torch.mean( torch.abs(y_pred-Thr), axis = 1)
        
        # C. Maximize the loss1
        loss = torch.mean(-loss1)
        #embed()
        return loss
    
    
    def Sample_M(self, gen_prob):
        # Shape of the selection probability
        n = gen_prob.shape[0]
        d = gen_prob.shape[1]
        # Sampling
        samples = np.random.binomial(1, gen_prob, (n,d))

        return samples

    #%% Training procedure
    def train(self, xs_train, xa_train, y_train, lmd, thr):

        # For each epoch (actually iterations)
        for epoch in range(self.epochs):

            #%% Train Discriminator
            # Select a random batch of samples
            idx = np.random.randint(0, xs_train.shape[0], self.batch_size)
            xs_batch = torch.as_tensor(xs_train[idx,:]).float()
            xa_batch = torch.as_tensor(xa_train[idx,:]).float()
            y_batch = torch.as_tensor(y_train[idx,:]).float() 
            # y_batch = torch.as_tensor(np.argmax(y_train[idx,:],1)).long()
            
            # Generate a batch of probabilities of feature selection
            gen_prob = self.generator(xs_batch, xa_batch).cpu().detach().numpy()
            # Sampling the features based on the generated probability
            sel_prob = self.Sample_M(gen_prob)
            '''sel_prob is the mask'''
            
            # Compute the prediction of the critic based on the sampled features (used for generator training)
            dis_prob = self.discriminator(xs_batch, xa_batch, torch.as_tensor(sel_prob).float())
            
            # Train the discriminator
            loss_func_c = nn.MSELoss()
            self.discriminator_optimizer.zero_grad()
            critic_loss = loss_func_c(dis_prob, y_batch)
            critic_loss.backward()
            self.discriminator_optimizer.step()

            #%% Train Valud function

            # Compute the prediction of the baseline based on the sampled features (used for generator training)
            val_prob = self.valfunction(xs_batch, xa_batch)#.cpu().detach().numpy()
            
            # Train the baseline model
            #v_loss = self.valfunction.train_on_batch(x_batch, y_batch)
            loss_func_v = nn.MSELoss()
            self.valfunction_optimizer.zero_grad()
            value_loss = loss_func_v(val_prob, y_batch)
            value_loss.backward()
            self.valfunction_optimizer.step()
            
            
            #%% Train Generator
            # Use three things as the y_true: sel_prob, dis_prob, and ground truth (y_batch)
            '''
            sel_prob: the mask generated by bernulli sampler [bs, d]
            dis_prob: prediction of the critic               [bs, state_dim]
            val_prob: prediction of the baseline model       [bs, state_dim]
            y_batch: batch of y_train                        [bs, state_dim]
            all of those variables are 'placeholders'
            '''
            
            y_batch_final = torch.as_tensor(np.concatenate( (sel_prob, torch.as_tensor(dis_prob).cpu().detach().numpy(), torch.as_tensor(val_prob).cpu().detach().numpy(), y_train[idx,:]), axis = 1 ))
            # Train the generator
            
            actor_pred = self.generator(xs_batch,xa_batch)
            self.generator_optimizer.zero_grad()
            actor_loss = self.my_loss(y_batch_final,actor_pred,lmd,Thr)
            actor_loss.backward()
            self.generator_optimizer.step()
            
            #%% Plot the progress
            dialog = 'Epoch: ' + '{:6d}'.format(epoch) + ', d_loss (Acc)): '
            dialog += '{:0.3f}'.format(critic_loss) + ', v_loss (Acc): '
            dialog += '{:0.3f}'.format(value_loss) + ', g_loss: ' + '{:+6.4f}'.format(actor_loss)

            if epoch % 100 == 0:
                logger.info('{}'.format(dialog))
    
    #%% Selected Features        
    def output(self, xs_train, xa_train):
        
        gen_prob = self.generator(xs_train, xa_train).cpu().detach().numpy()
        
        return np.asarray(gen_prob)
     
    #%% Prediction Results 
    def get_prediction(self, xs, xa, m_train):
        
        val_prediction = self.valfunction(xs,xa).cpu().detach().numpy()
        
        dis_prediction = self.discriminator(xs,xa, m_train).cpu().detach().numpy()
        
        return np.asarray(val_prediction), np.asarray(dis_prediction)


# In[3]:


ENV_NAME = 'Pendulum-v1'
alias = 'Fixed_INVASE'
RED_ACTION_DIM = 100
import gym
print('\n now evaluating: \n       ', ENV_NAME)


import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
import os
import torch.nn.functional as F
import utils
import TD3_INVASE

def eval_policy(policy, eval_episodes=10):
    eval_env = gym.make(ENV_NAME)
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action[:-RED_ACTION_DIM])
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward

env = gym.make(ENV_NAME)
torch.manual_seed(0)
np.random.seed(0)

#spec = env.action_space
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0] + RED_ACTION_DIM
max_action = env.action_space.high[0]

args_policy_noise = 0.2
args_noise_clip = 0.5
args_policy_freq = 2
args_max_timesteps = 10000
args_expl_noise = 0.1
args_batch_size = 256
args_eval_freq = 1000
args_start_timesteps = 10000

kwargs = {
    "state_dim": state_dim,
    "action_dim": action_dim,
    "max_action": max_action,
    "discount": 0.99,
    "tau": 0.005
}

for repeat in range(5):
    kwargs["policy_noise"] = args_policy_noise * max_action
    kwargs["noise_clip"] = args_noise_clip * max_action
    kwargs["policy_freq"] = args_policy_freq
    policy = TD3_INVASE.TD3(**kwargs)
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    # Evaluate untrained policy
    evaluations = [eval_policy(policy)]
    
    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    counter = 0
    msk_list = []        
    temp_curve = [eval_policy(policy)]
    temp_val = []
    for t in range(int(args_max_timesteps)):
        episode_timesteps += 1
        counter += 1
        # Select action randomly or according to policy
        if t < args_start_timesteps:
            action = np.random.uniform(-max_action, max_action, action_dim)
        else:
            if np.random.uniform(0,1) < 0.0:
                action = np.random.uniform(-max_action, max_action, action_dim)
            else:
                action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action * args_expl_noise, size=action_dim)
                ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action[:-RED_ACTION_DIM])
        

        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        if t >= args_start_timesteps:
            '''TD3'''
            policy.train(replay_buffer, args_batch_size)
                    
                    
        # Train agent after collecting sufficient data
        if done:
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            msk_list = []
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 

        # Evaluate episode
        if (t + 1) % args_eval_freq == 0:
            evaluations.append(eval_policy(policy))
            print('recent Evaluation:',evaluations[-1])
            np.save('results/evaluations_alias{}_ENV{}_Repeat{}'.format(alias,ENV_NAME,repeat),evaluations)
            
            
    state_list_train = replay_buffer.state[:args_start_timesteps-5000]
    state_list_test = replay_buffer.state[args_start_timesteps-5000:args_start_timesteps]

    action_list_train = replay_buffer.action[:args_start_timesteps-5000]
    action_list_test = replay_buffer.action[args_start_timesteps-5000:args_start_timesteps]
    next_state_list_train = replay_buffer.next_state[:args_start_timesteps-5000]
    next_state_list_test = replay_buffer.next_state[args_start_timesteps-5000:args_start_timesteps]

    state_delta_train = next_state_list_train - state_list_train
    state_delta_test = next_state_list_test - state_list_test
            
            
    X_DIM = action_list_train.shape[1] # feature dimension Hyper-Param
    
    
    class init_arg(object):
        def __init__(self, it = 10000, o = 'feature_score.csv.gz', dataset = None, i= None, target = None):
            self.it = it
            self.o = o
            self.dataset = dataset
            self.i = i
            self.target = target

    args = init_arg(dataset = 'Syn5', it = 300, )
    ocsv = args.o # 'feature_score.csv.gz'
    odir = os.path.dirname(ocsv)
    odir = '.' if not len(odir) else odir
    fn_csv = args.i #'data.csv'
    label_nm = args.target # 'target'
    nepoch = args.it
    logger = utilmlab.init_logger(odir)

    dataset = args.dataset

    assert dataset is not None or fn_csv is not None
    assert fn_csv is None or label_nm is not None

    # Data output can be either binary (Y) or Probability (Prob)
    data_out_sets = ['Y', 'Prob']
    data_out = data_out_sets[0]

    logger.info('invase: {} {} {} {}'.format(dataset, nepoch, odir, data_out))

    # Number of Training and Testing samples
    train_N = 10000
    test_N = 10000

    # Seeds (different seeds for training and testing)
    train_seed = 0
    test_seed = 1

    xs_train,xa_train, y_train, xs_test, xa_test, y_test= state_list_train, action_list_train, state_delta_train, state_list_test, action_list_test, state_delta_test, 
    g_test = np.zeros((y_test.shape[0],RED_ACTION_DIM + env.action_space.shape[0]))
    g_test[:,0] = 1
    print(g_test)
    print(xs_train.shape, xa_train.shape, y_train.shape, xs_test.shape, xa_test.shape, y_test.shape, g_test.shape)
            
    '''learning INVASE'''
    
    REAL_LMD = 1.0 # 0.0 - 0.5


    import time
    elapsed_time = []

    class init_arg(object):
        def __init__(self, it = 10000, o = 'feature_score.csv.gz', dataset = None, i= None, target = None):
            self.it = it
            self.o = o
            self.dataset = dataset
            self.i = i
            self.target = target


    for DATASET in ['Syn1']:
        args = init_arg(dataset = DATASET, it = 2500,)
        ocsv = args.o # 'feature_score.csv.gz'
        odir = os.path.dirname(ocsv)
        odir = '.' if not len(odir) else odir
        fn_csv = args.i #'data.csv'
        label_nm = args.target # 'target'
        nepoch = args.it
        logger = utilmlab.init_logger(odir)

        dataset = args.dataset

        assert dataset is not None or fn_csv is not None
        assert fn_csv is None or label_nm is not None

        # Data output can be either binary (Y) or Probability (Prob)
        data_out_sets = ['Y', 'Prob']
        data_out = data_out_sets[0]

        logger.info('invase: {} {} {} {}'.format(dataset, nepoch, odir, data_out))


        start_time = time.time()
        for thres_i in [0.0]:
            Predict_Out_temp = np.zeros([3, 2])    

            PVS_Alg = PVS(xs_train, dataset, 100, thres=thres_i)

            print('start training......')

            for train_epoch in range(int(nepoch/100)):

                Lmd = 0.1 #train_epoch*100/nepoch * REAL_LMD
                Thr = 0.0 #0.5*(1 - train_epoch*100/nepoch)
                print('now at training epoch number', int(train_epoch * 100),'hyp-params: lamda %.4f prior %.4f'%(Lmd,Thr))
                PVS_Alg.train(xs_train, xa_train, y_train, lmd = Lmd , thr = Thr)
                # 3. Get the selection probability on the testing set
                #Sel_Prob_Test = PVS_Alg.output(x_test)



                '''recurssive generation'''
                input_batch_xs = xs_test * 1.0
                input_batch_xa = xa_test * 1.0

                sel_prob_tot = 1.0
                for recur_time in range(1):
                    print('rec time now',recur_time,'dataset now:',DATASET)
                    gen_prob = PVS_Alg.generator(torch.as_tensor(input_batch_xs).float(),torch.as_tensor(input_batch_xa).float())
                    #sel_prob = PVS_Alg.Sample_M(gen_prob)
                    sel_prob = 1.*(gen_prob > 0.5)
                    sel_prob_tot_0 = sel_prob_tot * 1.0
                    sel_prob_tot = sel_prob * sel_prob_tot
                    input_batch_xa = sel_prob_tot * input_batch_xa

                    score = sel_prob_tot
                    #print('score',score)



                    # 4. Selected features
                    # 5. Prediction
                    val_predict, dis_predict = PVS_Alg.get_prediction(torch.as_tensor(xs_test).float(),torch.as_tensor(xa_test).float(), score)

                    def performance_metric(score, g_truth):

                        n = len(score)
                        Temp_TPR = np.zeros([n,])
                        Temp_FDR = np.zeros([n,])

                        for i in range(n):

                            # TPR    
                            # embed()
                            TPR_Nom = np.sum((score[i,:] * g_truth[i,:]).cpu().detach().numpy())
                            TPR_Den = np.sum(g_truth[i,:])
                            Temp_TPR[i] = 100 * float(TPR_Nom)/float(TPR_Den+1e-8)

                            # FDR
                            FDR_Nom = np.sum((score[i,:] * (1-g_truth[i,:])).cpu().detach().numpy())
                            FDR_Den = np.sum(score[i,:].cpu().detach().numpy())
                            Temp_FDR[i] = 100 * float(FDR_Nom)/float(FDR_Den+1e-8)

                        return np.mean(Temp_TPR), np.mean(Temp_FDR),                            np.std(Temp_TPR), np.std(Temp_FDR)

                    #%% Output

                    TPR_mean, TPR_std = -1, 0
                    FDR_mean, FDR_std = -1, 0
                    if g_test is not None:
                        TPR_mean, FDR_mean, TPR_std, FDR_std = performance_metric(
                            score, g_test)

                        logger.info('TPR mean: {:0.1f}%  std: {:0.1f}%'.format(
                            TPR_mean, TPR_std))
                        logger.info('FDR mean: {:0.1f}%  std: {:0.1f}%'.format(
                            FDR_mean, FDR_std))
                    else:
                        logger.info('no ground truth relevance')



                    #%% Performance Metrics
                    Predict_Out_temp[0,0] = np.linalg.norm(y_test - val_predict,2).mean()
                    Predict_Out_temp[0,1] = np.linalg.norm(y_test - dis_predict,2).mean()
                    print(Predict_Out_temp)

        elapsed_time.append(time.time() - start_time)
        print('PyTorch Version: elapsed time for {}: 11 feature, 10000 sample:'.format(DATASET),np.round(elapsed_time,4),'sec.')


    '''Continue training with fixed INVASE model'''
#     PVS_Alg.generator.cuda()
    PVS_Alg.generator.eval()
    for t in range(10000, 50000):
        episode_timesteps += 1
        counter += 1
        # Select action randomly or according to policy
        if t < args_start_timesteps:
            action = np.random.uniform(-max_action, max_action, action_dim)
        else:
            if np.random.uniform(0,1) < 0.0:
                action = np.random.uniform(-max_action, max_action, action_dim)
            else:
                action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action * args_expl_noise, size=action_dim)
                ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action[:-RED_ACTION_DIM])


        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        if t >= args_start_timesteps:
            '''TD3'''
            policy.train(replay_buffer, args_batch_size, PVS_Alg)


        # Train agent after collecting sufficient data
        if done:
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            msk_list = []
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 

        # Evaluate episode
        if (t + 1) % args_eval_freq == 0:
            evaluations.append(eval_policy(policy))
            print('recent Evaluation:',evaluations[-1])
            np.save('results/evaluations_alias{}_ENV{}_Repeat{}'.format(alias,ENV_NAME,repeat),evaluations)


# In[ ]:





# In[ ]:




