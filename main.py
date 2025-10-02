import time
import os
import numpy as np
from train import train_ppo, train_rpo
from test_ppo import evaluate as evaluate_ppo
from test_rpo import evaluate as evaluate_rpo
from utils import plot_metrics_per_seed, results_summary, plot_training_curves
from ppo_continuous_action import Agent as PPOAgent, make_env as ppo_make_env
from rpo_continuous_action import Agent as RPOAgent, make_env as rpo_make_env
import torch

if __name__ == "__main__":
    env_id = "Hopper-v4"    # CHOOSE EITHER     "Walker2d-v4"    OR     "HalfCheetah-v4"     OR      #"Hopper-v4"    
    seeds =  [42, 200, 73011, 86753, 690543]
    gamma = 0.99   
    algo = "ppo"  # CHOOSE EITHER "ppo" or "rpo"
    rpo_alpha = 0.5   
    capture_video = True  
    video_freq = 500    #CAN MODIFY THIS TO CAPTURE VIDEOS MORE OR LESS OFTEN
    eval_episodes = 20 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #KEEP GRAVITY AND MASS SCALE AT 1.0 FOR DEFAULT . MAKE IT 1.2, 1.5 or 1.8 FOR PERTURBED TRAINING -- GRAVITY PERTURBED BY DEFUALT
    gravity_scale = 1.2
    mass_scale = 1.0

    #STORING RETURNS FROM EACH SEED
    all_results = []
    all_training_returns = []
    all_training_timesteps = []
    all_training_entropies = []
    train_test_gaps = []
    all_training_times = []

    #TRAINING

    #IMPORTANT: MODELS HAVE ALREADY BEEN TRAINING AND SAVED IN 'LOGS' FOLDER. ONLY UNCOMMENT IF YOU WISH TO RETRAIN MODELS
    #THIS TAKES SEVERAL HOURS
    #RUNNING THE CODE WITHOUT UNCOMMENTING ANYTHING WILL PERFORM EVALUATION ON THE SAVED MODELS
    #IF YOU DO WISH TO BRIEFLY TEST IF TRAINING WORKS CORRECTLY, PLEASE DO EVALUATION FIRST TO VERIFY THAT PART WORKS AS INTENDED
    #THEREAFTER YOU MAY UNCOMMENT THE PART BELOW -- THIS IS TO AVOID ACCIDENTALLY OVERWRITING THE CURRENTLY SAVED MODELS IF YOU DONT
    #FULLY COMPLETE TRAINING
    for seed in seeds:
        '''
        start_train = time.time()
        if algo == "ppo":
            print(f"Now training PPO for seed: {seed} ")
            log_dir = train_ppo(env_id=env_id, total_timesteps=2_000_000, seed=seed, capture_video=capture_video, video_freq=video_freq)
            print(f"PPO training completed. Model saved in {log_dir}") 
        elif algo == "rpo":
            print(f"Now training RPO for seed: {seed}")

            #IMPORTANT: IF TRAINING WALKER2D MAKE total_timesteps=4_000_000 for RPO ONLY (BELOW THIS COMMENT)
            log_dir = train_rpo(env_id=env_id, total_timesteps=2_000_000, seed=seed, capture_video=capture_video, video_freq=video_freq)
            print(f"RPO training completed. Model saved in {log_dir}")
        else:
            raise ValueError("Unsupported algorithm: choose 'ppo' or 'rpo'")
        train_duration = time.time() - start_train
        print(f"Training completed in {train_duration:.2f} seconds.")
        all_training_times.append(train_duration)
        '''


        #UNCOMMENT BASED ON WHICH CASE YOU WANT TO EVALUATE -- SET TO CASE 1 BY DEFUALT.

        log_dir = f"logs/{algo.lower()}_{env_id.lower()}_seed{seed}"                #CASE 1
        #log_dir = f"logs/{algo.lower()}_{env_id.lower()}_seed{seed}_case2_mass"    #CASE 2 MASS PERTURBATION
        #log_dir = f"logs/{algo.lower()}_{env_id.lower()}_seed{seed}_case2_grav"    #CASE 2 GRAVITY PERTURBATION



        #Load and check training data
        '''UNCOMMENT BASED ON WHICH CASE YOU WANT TO EVALUATE -- SET TO CASE 1 BY DEFAULT'''

        #training_returns_path = os.path.join(log_dir, f"training_returns_{algo}_case2_mass.npy")       #CASE 2 MASS
        #training_timesteps_path = os.path.join(log_dir, f"training_timesteps_{algo}_case2_mass.npy")   #CASE 2 MASS
        #entropy_path = os.path.join(log_dir, f"entropy_{algo}_case2_mass.npy")                         #CASE 2 MASS

        #training_returns_path = os.path.join(log_dir, f"training_returns_{algo}_case2_grav.npy")       #CASE 2 GRAV
        #training_timesteps_path = os.path.join(log_dir, f"training_timesteps_{algo}_case2_grav.npy")   #CASE 2 GRAV
        #entropy_path = os.path.join(log_dir, f"entropy_{algo}_case2_grav.npy")                         #CASE 2 GRAV

        training_returns_path = os.path.join(log_dir, f"training_returns_{algo}.npy")                   #CASE 1
        training_timesteps_path = os.path.join(log_dir, f"training_timesteps_{algo}.npy")               #CASE 1
        entropy_path = os.path.join(log_dir, f"entropy_{algo}.npy")                                     #CASE 1


        #Appending training data to the earlier declared lists
        if os.path.exists(training_returns_path) and os.path.exists(training_timesteps_path) and os.path.exists(entropy_path):
            train_returns = np.load(training_returns_path)
            train_timesteps = np.load(training_timesteps_path)
            train_entropies = np.load(entropy_path)
            if train_returns.ndim > 1:
                train_returns = train_returns.flatten()
            if train_timesteps.ndim > 1:
                train_timesteps = train_timesteps.flatten()
            if train_entropies.ndim > 1:
                train_entropies = train_entropies.flatten()
            all_training_returns.append(train_returns)
            all_training_timesteps.append(train_timesteps)
            all_training_entropies.append(train_entropies)

            #compute average training returns for train-test gap
            train_avg = float(np.mean(train_returns)) if len(train_returns) > 0 else float("nan")
        else:
            #If missing data (returns, timesteps, entropies) for any particular seed, print this
            print(f"Warning: Missing training data for seed {seed}")
            continue
        
        #TESTING
        run_name = f"eval_case1_{algo.lower()}_grav_{gravity_scale}_{env_id}_seed{seed}"


        '''UNCOMMENT BASED ON WHICH CASE YOU'RE EVALUATING -- CASE 1 DEFUALT'''

        model_path = f"{log_dir}/{algo.lower()}_continuous_action_{env_id}_seed{seed}.pt"               #CASE 1
        #model_path = f"{log_dir}/{algo.lower()}_continuous_action_{env_id}_seed{seed}_case2_mass.pt"   #CASE 2 MASS
        #model_path = f"{log_dir}/{algo.lower()}_continuous_action_{env_id}_seed{seed}_case2_grav.pt"   #CASE 2 GRAV

        #Evaluating for PPO
        if algo == "ppo":            
            test_results = evaluate_ppo(
                model_path,
                ppo_make_env,
                env_id,
                eval_episodes=eval_episodes,
                run_name=run_name, 
                Model=PPOAgent,
                device=device,
                capture_video=False, 
                gamma=gamma,
                gravity_scale=gravity_scale,
                mass_scale=mass_scale,
            )
            #Evaluating for RPO
        elif algo == "rpo":
            test_results = evaluate_rpo(
                model_path,
                rpo_make_env,
                env_id,
                eval_episodes=eval_episodes,
                run_name=run_name,   
                Model=RPOAgent,
                device=device,
                capture_video=False,    
                gamma=gamma,
                gravity_scale=gravity_scale,
                mass_scale=mass_scale,
                rpo_alpha=rpo_alpha,
        )
        else:
            #Error if entered invalid algorithm
            raise ValueError("Unsupported algorithm: choose 'ppo' or 'rpo'")
    
        #Store evaluation results in a dictionary for later use
        result_dict = {
            "seed": seed,
            "algo": algo,
            "env_id": env_id,
            "returns": test_results,
            "avg_return": float(np.mean(test_results)),
            "std_return": float(np.std(test_results)),
        }

        #Append to all_results list
        all_results.append(result_dict)
        #PLOTTING
        
        #Plot metrics and compute train-test gaps
        plot_metrics_per_seed(result_dict, outdir="plots")
        test_avg = result_dict["avg_return"]
        gap = train_avg - test_avg
        train_test_gaps.append(gap)

        '''GRAVITY BY DEFUALT -- IF EVALUATING MASS, UNCOMMENT ONE BELOW IT'''
        gap_path = os.path.join(log_dir, f"train_test_gap_grav_{gravity_scale}.npy")
        #gap_path = os.path.join(log_dir, f"train_test_gap_mass_{mass_scale}.npy")  #UNCOMMENT FOR MASS  
        np.save(gap_path, np.array([gap]))
        print(f"Seed {seed}: Train avg = {train_avg:.2f}, Test avg = {test_avg:.2f}, Train-Test gap = {gap:.2f}")
    
    
    if all_results:
        avg_returns = [res["avg_return"] for res in all_results]
        std_returns = [res["std_return"] for res in all_results]
    
        overall_avg_return = np.nanmean(avg_returns) if avg_returns else np.nan
        std_across_seeds = np.std(avg_returns)            # std across seeds
        avg_within_seed_std = np.mean(std_returns)

        print("Summary Across Seeds")
        print(f"Overall average return across seeds: {overall_avg_return:.2f}")
        print(f"Overall mean return across seeds: {overall_avg_return:.2f} ± {std_across_seeds:.2f} (std across seeds)")
        print(f"Average within-seed std (noise in eval episodes): {avg_within_seed_std:.2f}")
    else:
        print("No results to summarize.")

    print(f"Avg train-test gap: {np.mean(train_test_gaps):.2f} ± {np.std(train_test_gaps):.2f}")

    '''UNCOMMENT THE CODE BELOW ONLY IF TRAINING HAS BEEN ENABLED'''
    #print(f"Average training time across {len(seeds)} seeds for {algo} = {np.mean(all_training_times):.2f} seconds.")

    '''UNCOMMENT FOR DEFUALT TRAINING -- CASE 1 (DEFUALT)'''
    #plot_training_curves(all_training_returns, all_training_timesteps, all_training_entropies, 
    #save_path=f"plots/training_return_across_seeds_{env_id}_{algo}.png", smoothing_window=10)

    '''UNCOMMENT FOR MASS PERUTRBED TRAINING -- CASE 2 MASS'''
    #plot_training_curves(all_training_returns, all_training_timesteps, all_training_entropies, 
    #save_path=f"plots/training_return_across_seeds_{env_id}_{algo}_case2_mass.png", smoothing_window=10)
    
    '''UNCOMMENT FOR GRAVITY PERTURBED TRAINING -- CASE 2 GRAVITY'''
    #plot_training_curves(all_training_returns, all_training_timesteps, all_training_entropies, 
    #save_path=f"plots/training_return_across_seeds_{env_id}_{algo}_case2_grav.png", smoothing_window=10)

    
    