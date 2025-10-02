import os
import subprocess
import sys

def train_ppo(env_id='Hopper-v4', total_timesteps=1_000_000, seed=1, capture_video = True, video_freq=100):
    #Launches PPO training with CleanRL in a subprocess.
    #Saves logs under the specified log directory for reproducibility and later analysis

    '''UNCOMMENT BASED ON WHICH CASE WE ARE TRAINING -- CASE 1 AS DEFUALT'''
    log_dir = f"logs/ppo_{env_id.lower()}_seed{seed}"
    #log_dir = f"logs/ppo_{env_id.lower()}_seed{seed}_case2_mass"       #CASE 2 MASS
    #log_dir = f"logs/ppo_{env_id.lower()}_seed{seed}_case2_grav"       #CASE 2 GRAVITY
    os.makedirs(log_dir, exist_ok=True)

    #Path to the CleanRL RPO training script
    cleanrl_script = os.path.join(os.path.dirname(__file__), "ppo_continuous_action.py")

    cmd = [
        sys.executable,                                 #Use the same Python interpreter running this script
        cleanrl_script,                                 #CleanRL PPO script
        "--env-id", env_id,                             #Environment to train on
        "--total-timesteps", str(total_timesteps),      #Number of training timesteps
        "--seed", str(seed),                            #Seed to train on
        "--track",                                      #Enable WandB tracking
        "--save-model",                                 #Save trained model
        "--wandb-project-name", "ppo-case1"
    ]
    if capture_video:
        #Optionally record environment videos at specified frequency
        cmd += ["--capture-video"]
        cmd += ["--video-freq", str(video_freq)]

    subprocess.run(cmd, check=True)
    return log_dir

def train_rpo(env_id='Hopper-v4', total_timesteps=1_000_000, seed=1, capture_video=True, video_freq=100):
    #Launches RPO training with CleanRL in a subprocess.
    #Saves logs under the specified log directory for reproducibility and later analysis

    '''UNCOMMENT BASED ON WHICH CASE WE ARE TRAINING -- CASE 1 AS DEFUALT'''
    log_dir = f"logs/rpo_{env_id.lower()}_seed{seed}"
    #log_dir = f"logs/rpo_{env_id.lower()}_seed{seed}_case2_mass"       #CASE 2 MASS
    #log_dir = f"logs/rpo_{env_id.lower()}_seed{seed}_case2_grav"       #CASE 2 GRAVITY
    os.makedirs(log_dir, exist_ok=True)

    #Path to the CleanRL RPO training script
    cleanrl_script = os.path.join(os.path.dirname(__file__), "rpo_continuous_action.py")

    cmd = [
        sys.executable,                                 #Use the same Python interpreter running this script                                    
        cleanrl_script,                                 #CleanRL RPO script
        "--env-id", env_id,                             #Environment to train on
        "--total-timesteps", str(total_timesteps),      #Number of training timesteps
        "--seed", str(seed),                            #Seed to train on
        "--track",                                      #Enable WandB tracking. No --save_modelas this is done by defualt in cleanrl_script
        "--wandb-project-name", "rpo-case1"             
    ]
    if capture_video:
        #Optionally record environment videos at specified frequency
        cmd += ["--capture-video"]
        cmd += ["--video-freq", str(video_freq)]

    subprocess.run(cmd, check=True)
    return log_dir