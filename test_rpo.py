from typing import Callable
import gymnasium as gym
import torch
import numpy as np
import os

#Customized version of CleanRL's RPO evaluation function
def evaluate(model_path: str, make_env: Callable, env_id: str, eval_episodes: int, run_name: str,
             Model: torch.nn.Module, device: torch.device = torch.device("cpu"),
             capture_video: bool = True, gamma: float = 0.99, gravity_scale: float = 1.0, mass_scale: float = 1.0, rpo_alpha: float = 0.5):
    
    #Create a vectorized environment -- single environment wrapped in SyncVectorEnv
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, capture_video, run_name, gamma)])

    # Debug block
    raw_env = envs.envs[0].unwrapped
    print("\n[DEBUG - TEST BEFORE PERTURBATION ENV INSPECTION]")
    print("env id:", env_id)
    print("gravity (sim):", raw_env.model.opt.gravity)
    print("body_mass[:10] (sim):", raw_env.model.body_mass[:10])
    print("===============================\n")

    # Applying gravity perturbation
    if gravity_scale != 1.0:
        base_gravity = np.array([0, 0, -9.81])
        raw_env.model.opt.gravity[:] = base_gravity * gravity_scale
        print(f"[PERTURBATION APPLIED] Gravity scaled by {gravity_scale}")
        print("New gravity (sim):", raw_env.model.opt.gravity)
        print("===============================================\n")
    
    # Apply mass perturbation
    if mass_scale != 1.0:
        base_mass = raw_env.model.body_mass.copy()
        raw_env.model.body_mass[:] = base_mass * mass_scale
        print(f"[PERTURBATION APPLIED] Mass scaled by {mass_scale}")
        print("New body_mass[:10] (sim):", raw_env.model.body_mass[:10])
        print("===============================================\n")

    #Load observation normalization stats -- This is not done in the original, but is critical for evaluation
    #This ensures evaluation consistency with training normalization
    rms_path = os.path.join(os.path.dirname(model_path), "obs_rms.pt")
    if os.path.exists(rms_path):
        rms_stats = torch.load(rms_path, map_location=device, weights_only=False)
        normalize_wrapper = envs.envs[0]
        #Traverse wrappers to find the NormalizeObservation wrapper
        while not isinstance(normalize_wrapper, gym.wrappers.NormalizeObservation):
            normalize_wrapper = normalize_wrapper.env
        #Restore saved running mean and variance for observations
        normalize_wrapper.obs_rms.mean = rms_stats["mean"]
        normalize_wrapper.obs_rms.var = rms_stats["var"]
        normalize_wrapper.obs_rms.count = rms_stats["count"]
        print(f"[DEBUG] Loaded normalization stats from {rms_path}")
    else:
        print(f"[WARNING] No normalization stats found at {rms_path}")

    #Load RPO agent with RPO Alpha
    agent = Model(envs, rpo_alpha).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    #Reset environment and begin evaluation loop
    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
        next_obs, _, _, _, infos = envs.step(actions.detach().cpu().numpy())
        #Check if an episode has ended and log the return
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)+1}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    try:
        envs.close()
    except Exception as e:
        print(f"Warning: Error during environment cleanup: {e}")

    #Return list of returns for all evaluation episodes
    return episodic_returns

