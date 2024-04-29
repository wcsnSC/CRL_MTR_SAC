
import os,sys,argparse,warnings
warnings.filterwarnings("ignore")
import numpy as np
import gym
import torch
import saferl_algos
from saferl_plotter.logger import SafeLogger
import saferl_utils
from sklearn import metrics 
from queue import deque 

######################### make cartpole stabilization environment ################################################################
sys.path.append("saferl_envs")
from copy import deepcopy
from functools import partial
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.plotting import plot_from_logs
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import mkdirs, set_dir_from_config, set_device_from_config, set_seed_from_config, save_video

import numpy as np 
import matplotlib.pyplot as plt


import skimage.measure 
from sklearn import metrics 



class sample_env():
    def __init__(
        self,
        source,
        device_num,
        action_dim=1,
        max_action=1,
        M = 30,
        
        # r_weight = np.array([3, 2, 3, 2, 1, 1]), # urgency weight
        # dev_spent_RBs = np.array([1, 1, 1, 1, 5, 5]), 

        dev_spent_RBs = np.array([1, 1, 1, 1,  1, 1, 1, 1,   1, 1, 1, 1,   1, 1, 1, 1,   1, 1, 1, 1,]), 
        r_weight = np.array([5, 2, 5, 2, 3, 2, 3, 2,   3, 2, 3, 2, 5, 2, 5, 2,   1, 1,   1, 1]), # urgency weight
        
        cost_weight = int(5e1),

        delta=1,

        rew_discount=0.99,
        cost_discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        delay = 2,
        prob_pass = 0.6,
        ksi = 0.0001
    ):


        self.device_num = device_num
        self.action_dim = action_dim
        self.max_action = max_action

        self._max_episode_steps = int(5e3)
        self.tau  = tau
        
        self.prob_pass = prob_pass
        self.delay = delay
        self.ksi = ksi
        self.t = 0
        self.aoi = [1 for i in range(self.device_num)]
        self.x_ = [0 for i in range(self.device_num)]
        self.AoII = np.array([1 for i in range(self.device_num)])
        self.arrive_times =[[] for i in range(self.device_num)]
        self.send_times = [[] for i in range(self.device_num)]
        self.last_recv_t = [0 for i in range(self.device_num)]
        self.source_mean = source.mean(axis=0)
        self.source = source/self.source_mean
        self.error_indicator = [False for i in range(self.device_num)] 


        self.r_weight = np.array(r_weight) / np.sum(r_weight) 
        self.dev_spent_RBs = np.array(dev_spent_RBs) 
        self.M = M
        self.cost_weight = cost_weight

        self.done = False
        self.reset_next_state_as = [[] for i in range(self.device_num)]

        
        
    def step(self, action = [0,0]):
        all_dev_state = [[] for i in range(self.device_num)]
        for dev in range(self.device_num):        
            x = self.source[self.t, dev]
            if action[dev] > 0:
                self.send_times[dev].append(self.t)
                self.arrive_times[dev].append(self.t + self.delay)

            # loss pkg?     
            pkg_lost = np.random.uniform(0, 1) >= self.prob_pass

            gamma_t = 0 if pkg_lost else 1 
            
            error = abs(self.x_[dev] - x) 

            if dev < 16 and dev % 2 == 0:
                self.error_indicator[dev] = error / x > 1e-2     # temp   1e-1  4e-2  1e-2
            if dev < 16 and dev % 2 == 1:
                self.error_indicator[dev] = error / x >= 1e-2   # humi   1e-1    4e-2  1e-2
            elif dev >= 16:
                self.error_indicator[dev] = error > 5e-2         # traj  1e-1   3e-1  5e-2

            # self.error_indicator[dev] = error >= self.ksi 

            # AoI evoluation, only when pkg is received   
            if pkg_lost == False and self.t in self.arrive_times[dev]:
                self.aoi[dev] = self.t - self.send_times[dev][-1] # update aoi
                # del self.send_times[0]
                self.last_recv_t[dev] = self.t
                self.x_[dev] = x # update estimation result
            else: 
                self.aoi[dev] += 1

            self.AoII[dev] = self.aoi[dev] * (1 if self.error_indicator[dev] else 0)

            if self.t in self.arrive_times[dev]:
                self.arrive_times[dev].remove(self.t)
            if self.t - self.delay in self.send_times[dev]:
                self.send_times[dev].remove(self.t - self.delay)
            
            dev_state = np.array([self.aoi[dev], error,  gamma_t])
            all_dev_state[dev] = dev_state
        self.t += 1


        next_state = 1 
        reward = 1
        self.done =  0 if self.t < self._max_episode_steps  else 1
        info = "step"

        if self.t < self._max_episode_steps:
            pass
        else:
            self.reset_next_state_as = np.array(all_dev_state).reshape(3 * self.device_num)

        RB_cost = np.dot(dev_spent_RBs.T, action)
        RB_cost = (RB_cost - self.M)**2 if RB_cost > self.M else 0
        return  np.array(all_dev_state).reshape(3 * self.device_num), \
                -1000 * skimage.metrics.normalized_root_mse(source[self.t, :], \
                    np.array(self.x_) * env.source_mean[:])
                self.done, info

                # np.dot( self.r_weight ,(-1) * self.AoII.reshape(self.device_num) ), \

                # (-1 * np.max(env.AoII * env.r_weight) ), \

                # -1000 * skimage.metrics.normalized_root_mse(source[self.t, :], \
                #     np.array(self.x_) * env.source_mean[:]) + (-1 * np.max(env.AoII * env.r_weight) ),  \


                
        # np.array([self.aoi, error,  gamma_t]), -self.AoII, done, info 
        # return next_state, reward , done, info, pkg_lost, AoII, self.t, self.arrive_times, self.send_times, self.aoi, self.x_
    
    def reset(self):
        self.t = 0 
        self.arrive_times =[[] for i in range(self.device_num)]
        self.send_times = [[] for i in range(self.device_num)]
        if self.done:
            reset_state = self.reset_next_state_as
        else: reset_state = np.array([-1, 0, 0]*self.device_num)
        return [reset_state, self.t]


        

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--overrides",nargs='+',type=str)
    parser.add_argument("--exp_name",type=str)
    parser.add_argument("--env", default="Stabilization")           # Env name
    parser.add_argument("--flag", default="constraint_violation")   # c_t = info[flag]
    parser.add_argument("--base_policy", default="TD3")             # Base Policy name
    parser.add_argument("--use_td3", action="store_true")           # unconstrained RL
    parser.add_argument("--use_usl", action="store_true")           # Wether to use Unrolling Safety Layer
    parser.add_argument("--use_qpsl",action="store_true")           # Wether to use QP Safety Layer (Dalal 2018)
    parser.add_argument("--use_recovery",action="store_true")       # Wether to use Recovery RL     (Thananjeyan 2021)
    parser.add_argument("--use_lag",action="store_true")            # Wether to use Lagrangian Relaxation  (Ha 2021)
    parser.add_argument("--use_fac",action="store_true")            # Wether to use FAC (Ma 2021)
    parser.add_argument("--use_rs",action="store_true")             # Wether to use Reward Shaping
    parser.add_argument("--seed", default=1, type=int)              # Sets Gym, PyTorch and Numpy seeds
    # Hyper-parameters for all safety-aware algorithms
    parser.add_argument("--delta",default =1,type=float)         # Qc(s,a) \leq \delta
    parser.add_argument("--cost_discount", default=0.99)            # Discount factor for cost-return
    # Hyper-parameters for using USL
    parser.add_argument("--warmup_ratio", default=1/3)              # Start using USL in traing after max_timesteps*warmup_ratio steps
    parser.add_argument("--kappa",default = 5, type=float)                      # Penalized factor for Stage 1
    parser.add_argument("--early_stopping", action="store_true")    # Wether to terminate an episode upon cost > 0
    # Hyper-parameters for using Lagrangain Relaxation
    parser.add_argument("--lam_init", default = 0.)                 # Initalize lagrangian multiplier
    parser.add_argument("--lam_lr",default = 1e-5)                  # Step-size of multiplier update
    # Hyper-parameters for using Reward Shaping
    parser.add_argument("--cost_penalty",default = 0.1)               # Step-size of multiplier update
    # Other hyper-parameters for original TD3
    parser.add_argument("--start_timesteps", default=100, type=int) # o 200 Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)      # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default= 24e4, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--rew_discount", default=0.99)             # Discount factor for reward-return
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--load_reward", default = '1') 
    parser.add_argument("--mtr_replay_buffer", default = '1') 
    parser.add_argument("--irm_loss", default = '0') 

    parser.add_argument("--num_RB", default = 30, type=int) 

    args = parser.parse_args()

    device_num = 20
    action_dim = device_num


    # load dataset 
    import pandas as pd

    # source = pd.read_csv('data_0304_mote12_4dev.txt', sep=' ')
    # # source = source.drop('date:yyyy-mm-dd', axis = 1)
    # source = np.array(source)

    # source = pd.read_csv('data_6_devs_.csv', sep=' ')
    # source = np.array(source)

    source = np.loadtxt('data_dev20.csv')

    print(f'source shape:{source.shape}')

    # open env
    env = sample_env(source, device_num, action_dim, M = args.num_RB)
    eval_env = deepcopy(env) # copy the env


    idx = 0

    x__list= np.empty((env.device_num, 0))
    aoi_list = np.empty((env.device_num, 0))
    AoII_list = np.empty((env.device_num, 0))

    # device setting 
    M = env.M    # totRB , 可以脱离env修改
    print('M===', M)

    dev_spent_RBs = env.dev_spent_RBs   # device RB spent vector

    file_name = f"dev{device_num}{'_uneven_seed'}_{args.seed}_M_{M}_MTR_{args.mtr_replay_buffer}"
    
    logger = SafeLogger(exp_name='5_FAC_sammple',env_name=args.env,seed=args.seed,fieldnames=['EpRet','EpCost','CostRate'])


    if args.save_model and not os.path.exists("./models_d20_w_M"):
        os.makedirs("./models_d20_w_M")
    if args.load_reward and not os.path.exists("./reward_record_w_M"):
        os.makedirs("./reward_record_w_M")

    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.shape[0] 
    # max_action = float(env.action_space.high[0])

    # initiate DNN model
    state_dim = 3 * env.device_num
    print(f"state_dim:{state_dim}")
    action_dim = env.action_dim
    max_action = env.max_action

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "rew_discount": args.rew_discount,
        "tau": args.tau,
        "policy_noise": args.policy_noise * max_action,
        "noise_clip": args.noise_clip * max_action,
        "policy_freq": args.policy_freq,
    }

    kwargs_safe = {
        "cost_discount": args.cost_discount,
        "delta": args.delta,                     
    }

    # initiate policy model
    from saferl_algos.fac_sampler import eval_policy
    kwargs.update(kwargs_safe)
    policy = saferl_algos.fac_sampler.TD3Fac(**kwargs)

    # print('args.mtr_replay_buffer', args.mtr_replay_buffer)
    num_buffers = 4
    beta = 0.79
    
    if args.mtr_replay_buffer == '1':
        replay_buffer = saferl_utils.MultiTimescaleReplayBuffer(size = int(5e3), num_buffers= num_buffers, beta = beta, no_waste = True)
        print('replay_buffer setting:', replay_buffer._maxsize, replay_buffer.beta)
        file_name = f"dev{device_num}{'_uneven_seed'}_{args.seed}_M_{M}_MTR_{args.mtr_replay_buffer}_beta_{replay_buffer.beta}"
    else:   
    # initiate replay buffer 
        replay_buffer = saferl_utils.CostReplayBuffer(state_dim, action_dim)
        print('Here use Cost replay_buffer !!!!!!')

    if args.load_model != "":
        policy.load(f"./models_d20_w_M/{args.load_model}")
    if args.load_reward != '1':
        reward_record = np.loadtxt(f"./reward_record_w_M/{args.load_reward}").reshape(-1, 1)
        M_name = args.load_reward.split('_')[3]
        if M_name != "":
            cost_record = np.loadtxt(f'./cost_record_w_M/MTR_Cost_dev{env.device_num}_{M_name}_nochange__MTR{args.mtr_replay_buffer}_beta{beta}.txt' )
            cost_record = cost_record.reshape(-1,1)
        else:
            cost_record = np.empty((0, 1))
    else:
        reward_record = np.empty((0, 1))
        cost_record = np.empty((0, 1))
    

    reset_info, done = env.reset(), False
    # state = reset_info[0]
    state = [1, 0, 0] * device_num

    episode_reward = 0
    episode_cost = 0
    episode_timesteps = 0
    episode_num = 0
    cost_total = 0
    prev_cost = 0
    commun_queue = deque([0]*100)


    np.set_printoptions(precision=4, suppress=True) # control precision 

    """ =============== Start train  ======================
    """
    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1
        """ =============== env changes  =============
        """
        change_t_interval = int(100e4)
        convergence_bias = int(0e4)

        if  t == change_t_interval + convergence_bias:
        # change_t_interval <= t <= 2 * change_t_interval: 
            # M = 32
            env = sample_env(source, device_num, action_dim, 6)
        elif t == 2 * change_t_interval + convergence_bias:
        # 2 * change_t_interval <= t <= 3 * change_t_interval: 
            # M = 24
            env = sample_env(source, device_num, action_dim, 17)
        # else:
        #     M = 20
        #     env = sample_env(source, device_num, action_dim, M)


 
        # print('$$$$$$$ fac use else $$$$$$$')            
        if t < args.start_timesteps:
            # print(f'hhhhhhhh {t} in replay buffer hhhhhhhh')
            action = np.random.uniform(1, 1, env.device_num) 

            # action = np.random.randn(env.device_num) if np.random.uniform(0, 1) < 0.5 else -1 *(np.random.randn(env.device_num))
        else:
            # if episode_timesteps % 500 == 1:
            #     # print(f"episode_timesteps:{episode_timesteps}")
            #     print(f"state:{state}, shape{np.array(state).shape}, t = {t+1}" )
            action = policy.select_action(np.array(state), exploration=True)

            if  np.random.rand() <= 0.5:
            #     action = np.where( np.array(env.error_indicator) > 0, 1, 0) * action
            #     action = np.where( action == 0, -1, action)
                action = np.where( np.array(env.error_indicator) > 0, action, 0) 
                action = np.where( action == 0, -1, action)
                # np.where( action > 0, 1, -1)
                # action = np.where( action > 0, 1, -1)


        """Envrionment Development
        """
        
        # Perform action
        # next_state, reward, done, info = env.step(action)
        # action = 1 if action > 0.5 else 0
        next_state, reward, done, info = env.step(action)

        # collect information
        x__list =  np.append(x__list, np.array(env.x_).reshape(-1,1), axis =1)
        aoi_list =  np.append(aoi_list, np.array(env.aoi).reshape(-1,1), axis =1)
        AoII_list =  np.append(AoII_list, np.array(env.AoII).reshape(-1,1), axis =1)

        reward_record = np.append(reward_record, np.array(reward).reshape(-1,1), axis=0)
        
        # cost value

        action = np.where(action > 0, 1, 0) 

        n_RB_spent = np.dot(dev_spent_RBs.T, action) # consumed RB num
        cost = n_RB_spent - M if (n_RB_spent - M) > 0 else 0.
         



        # instant commun cost using queue
        commun_queue.appendleft(n_RB_spent)
        commun_queue.pop()
        RB_occp_rate = sum(commun_queue)/(M * len(commun_queue) ) 
        RB_q_cost = sum(commun_queue) / len(commun_queue)
        # cost = env.cost_weight * cost 



        # cost = RB_q_cost - env.M if RB_q_cost > env.M else 0
        # cost = env.M - RB_q_cost if RB_q_cost < env.M - 4 else 0
        # cost = 10 * RB_q_cost + env.cost_weight * (cost  )

        # Function modified cost:
        
        # RB_q_cost = n_RB_spent   # long-term cnstraint
        if RB_q_cost > (env.M )  :
            cost = RB_q_cost - env.M 
            cost = cost ** 2
        elif RB_q_cost < env.M -4 :
            cost = env.M - RB_q_cost 
        else:
            cost = 0
        cost =  env.cost_weight * ( cost )
        cost = (cost ) /1
        # cost = 0





        cost_record = np.append(cost_record, np.array(cost).reshape(-1,1), axis=0)

        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # set the early broken state as 'cost = 1'
        # if done and episode_timesteps < env._max_episode_steps:
        #     cost = 1

        # Store data in replay buffer
        if args.use_td3 or args.use_rs:
            replay_buffer.add(state, action, next_state, reward, done_bool)
        elif args.use_recovery:
            replay_buffer.add(state, raw_action, action, next_state, reward, cost, done_bool)
        elif args.use_qpsl:
            replay_buffer.add(state, action, next_state, reward, cost, prev_cost, done_bool)
        else: # fac use else 
            # print("buffer", state, action, next_state, reward, cost, done_bool)
            replay_buffer.add(state, action, next_state, reward, cost, done_bool)
            

        state = next_state
        prev_cost = cost
        episode_reward += reward
        episode_cost += cost

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            # if t% 1000==0:
                # print('replay_buffer shape', len(replay_buffer.buffers[0]) )
            policy.train(replay_buffer, args.batch_size, args.irm_loss)


        if done: 
            if args.use_lag:
                print(f'Lambda : {policy.lam}')
            
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total_T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} Cost: {episode_cost:.3f}")
            # MSE = np.array(source[:env._max_episode_steps]-np.array(x__list[:env._max_episode_steps])*env.source_mean).mean()
            MSE = metrics.mean_squared_error( source[:env._max_episode_steps, :], \
                  np.multiply( (x__list[:, 0:env._max_episode_steps].T), env.source_mean) )    

            print(f"Episode: MSE = {MSE:.4f}，MSE_length={env._max_episode_steps}, accum_AoII={np.sum(AoII_list, axis=1)}")
            print(f"----Action: {action}-----len(commun_queue): {len(commun_queue)}-----RB_occp_rate: {RB_occp_rate:.4f} ")
            if args.mtr_replay_buffer == '1':
                print(f"----Buffer occupied size:{replay_buffer.__len__()} \n")
            else: 
                print(f"----Buffer occupied size:{replay_buffer.size} \n")


            if args.mtr_replay_buffer == '1' or (t+1) == args.max_timesteps:
                np.savetxt(f'./reward_record_w_M/MTR_Reward_dev{env.device_num}_M{M}_nochange__MTR{args.mtr_replay_buffer}_beta{beta}_irmloss{args.irm_loss}.txt', reward_record )
                np.savetxt(f'./cost_record_w_M/MTR_Cost_dev{env.device_num}_M{M}_nochange__MTR{args.mtr_replay_buffer}_beta{beta}.txt', cost_record )
            else: 
                np.savetxt(f'./reward_record_w_M/CostBuffer_Reward_dev{env.device_num}_M{M}_nochange_compare_to_beta{beta}.txt', reward_record )
            print('rewards have been saved !!! at t = ', t+1)


            # Reset environment
            reset_info, done = env.reset(), False
            state = reset_info[0]

            episode_reward = 0
            episode_cost = 0
            episode_timesteps = 0
            episode_num += 1
            prev_cost = 0

            x__list= np.empty((env.device_num, 0))
            aoi_list = np.empty((env.device_num, 0))
            AoII_list = np.empty((env.device_num, 0))




        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:

            # print(f"go into eval_freq ******")
            # if args.use_usl:
            #     evalEpRet,evalEpCost = eval_policy(policy, eval_env, args.seed, args.flag, use_usl=True)
            # elif args.use_recovery:
            #     evalEpRet,evalEpCost = eval_policy(policy, eval_env, args.seed, args.flag, use_recovery=True)
            # elif args.use_qpsl:
            #     evalEpRet,evalEpCost = eval_policy(policy, eval_env, args.seed, args.flag, use_qpsl=True)
            # else:
            #     evalEpRet,evalEpCost = eval_policy(policy, eval_env, dev_spent_RBs, M, args.seed, args.flag)
            # logger.update([evalEpRet,evalEpCost,1.0*cost_total/t], total_steps=t+1)

            pass


            if args.save_model:
                policy.save(f"./models_d20_w_M/{file_name}")
                print(f"-----model_saved!!!-----as--\n",file_name )
            



