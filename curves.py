import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d 
import os
import glob

def load_algo_data(algo_name, env_id, log_root="logs"):
    #Load training data (returns, timesteps, entropies) for one algorithm across seeds.
    all_returns, all_timesteps, all_entropies = [], [], []

    #Each pattern corresponds to a different experimental case (default, mass perturbation, gravity perturbation)
    '''UNCOMMENT FOR FOR WHICH CASE YOU WOULD LIKE TO SEE CURVES FOR'''
    pattern = os.path.join(log_root, f"{algo_name.lower()}_{env_id.lower()}_seed*", "")                 #CASE 1 --DEFUALT
    #pattern = os.path.join(log_root, f"{algo_name.lower()}_{env_id.lower()}_seed*_case2_mass", "")     #CASE 2 MASS
    #pattern = os.path.join(log_root, f"{algo_name.lower()}_{env_id.lower()}_seed*_case2_grav", "")     #CASE 2 GRAV
    run_dirs = glob.glob(pattern)

    if not run_dirs:
        #If no runs are found, return None
        print(f"No runs found for {algo_name} in {pattern}")
        return None

    #Loop over each run directory
    for run_dir in run_dirs:
        try:
            '''CASE 1 BY DEFUALT -- UNCOMMENT BLOCKS BELOW FOR CASE 2's'''
            returns = np.load(os.path.join(run_dir, f"training_returns_{algo_name.lower()}.npy")) 
            timesteps = np.load(os.path.join(run_dir, f"training_timesteps_{algo_name.lower()}.npy"))
            entropies = np.load(os.path.join(run_dir, f"entropy_{algo_name.lower()}.npy"))

            '''UNCOMMENT FOR CASE 2 MASS'''
            #returns = np.load(os.path.join(run_dir, f"training_returns_{algo_name.lower()}_case2_mass.npy")) #ADDED case 2
            #timesteps = np.load(os.path.join(run_dir, f"training_timesteps_{algo_name.lower()}_case2_mass.npy"))
            #entropies = np.load(os.path.join(run_dir, f"entropy_{algo_name.lower()}_case2_mass.npy"))

            '''UNCOMMENT FOR CASE 2 GRAVITY'''
            #returns = np.load(os.path.join(run_dir, f"training_returns_{algo_name.lower()}_case2_grav.npy")) #ADDED case 2
            #timesteps = np.load(os.path.join(run_dir, f"training_timesteps_{algo_name.lower()}_case2_grav.npy"))
            #entropies = np.load(os.path.join(run_dir, f"entropy_{algo_name.lower()}_case2_grav.npy"))
        except Exception as e:
             #Skip any run directories that failed to load
            print(f"Skipping {run_dir}: {e}")
            continue

        if len(returns) == 0 or len(timesteps) == 0:
            #Skip if arrays are empty
            print(f"Empty arrays in {run_dir}, skipping.")
            continue

        #storing valid data
        all_returns.append(returns)
        all_timesteps.append(timesteps)
        all_entropies.append(entropies)

    if not all_returns:
        #If no valid returbs, return None
        return None

    #Return all data in a dictionary for later use
    return {
        "returns": all_returns,
        "timesteps": all_timesteps,
        "entropies": all_entropies,
    }


def plot_training_curves_combined(env_id, save_dir="plots", smoothing_window=10):
    #Plot mean ± std training return and entropy curves for PPO vs RPO across seeds
    #Loads .npy logs saved in "logs/ppo_<env>_seed*" and "logs/rpo_<env>_seed*"
    os.makedirs(save_dir, exist_ok=True)

    #Load training data
    algo_data = {
        "PPO": load_algo_data("ppo", env_id),
        "RPO": load_algo_data("rpo", env_id)
    }

    colors = {"PPO": "blue", "RPO": "red"}

    #Plot Returns
    plt.figure(figsize=(10, 6))
    entropy_curves = {}

    for algo_name, data in algo_data.items():
        if data is None:
            continue    #skip if no data available

        valid_returns, valid_timesteps, valid_entropies = (
            data["returns"], data["timesteps"], data["entropies"]
        )

        #Create a common timestep grid across seeds to allign curves
        max_common_timestep = min(ts[-1] for ts in valid_timesteps)
        timestep_grid = np.linspace(0, max_common_timestep, num=500)

        #Interpolate returns onto the common grid for averaging
        interpolated_returns = [
            np.interp(timestep_grid, timesteps, returns)
            for returns, timesteps in zip(valid_returns, valid_timesteps)
        ]
        interpolated_returns = np.array(interpolated_returns)

        #Calculate mean and standard deviation across seeds
        mean_returns = np.mean(interpolated_returns, axis=0)
        std_returns = np.std(interpolated_returns, axis=0)

        #Apply smoothing using uniform moving average
        if smoothing_window > 1:
            mean_returns = uniform_filter1d(mean_returns, size=smoothing_window)
            std_returns = uniform_filter1d(std_returns, size=smoothing_window)

        # Plot mean curve with shaded ± std region
        plt.plot(timestep_grid, mean_returns, label=f"{algo_name}", color=colors[algo_name])
        plt.fill_between(timestep_grid,
                         mean_returns - std_returns,
                         mean_returns + std_returns,
                         color=colors[algo_name], alpha=0.2)

        #Save entropy data for later
        interpolated_entropies = [
            np.interp(timestep_grid, timesteps, entropies)
            for entropies, timesteps in zip(valid_entropies, valid_timesteps)
        ]
        interpolated_entropies = np.array(interpolated_entropies)

        mean_entropies = np.mean(interpolated_entropies, axis=0)
        std_entropies = np.std(interpolated_entropies, axis=0)

        if smoothing_window > 1:
            mean_entropies = uniform_filter1d(mean_entropies, size=smoothing_window)
            std_entropies = uniform_filter1d(std_entropies, size=smoothing_window)

        entropy_curves[algo_name] = (timestep_grid, mean_entropies, std_entropies)

    #Finalise return plot
    plt.xlabel("Timesteps", fontsize=18)
    plt.ylabel("Episode Return", fontsize=18)
    plt.title(f"Training Returns Across Seeds ({env_id})", fontsize=20)
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)

    '''UNCOMMENT BASED ON WHICH CASE WE ARE PLOTTING'''
    save_path = os.path.join(save_dir, f"case1_{env_id}_ppo_vs_rpo_returns.pdf")            #CASE 1 -- DEFUALT
    #save_path = os.path.join(save_dir, f"case2_mass_{env_id}_ppo_vs_rpo_returns.pdf")      #CASE 2 MASS
    #save_path = os.path.join(save_dir, f"case2_grav_{env_id}_ppo_vs_rpo_returns.pdf")      #CASE 2 GRAVITY
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Return plot saved to {save_path}")

    #Plotting Entropies
    if entropy_curves:
        plt.figure(figsize=(10, 6))
        for algo_name, (timestep_grid, mean_entropies, std_entropies) in entropy_curves.items():
            plt.plot(timestep_grid, mean_entropies, label=f"{algo_name}", color=colors[algo_name])
            plt.fill_between(timestep_grid,
                             mean_entropies - std_entropies,
                             mean_entropies + std_entropies,
                             color=colors[algo_name], alpha=0.2)

        plt.xlabel("Timesteps", fontsize=18)
        plt.ylabel("Policy Entropy", fontsize=18)
        plt.title(f"Training Policy Entropy Across Seeds ({env_id})", fontsize=20)
        plt.legend(fontsize=16)
        plt.grid(True, alpha=0.3)

        '''UNCOMMENT BASED ON WHICH CASE WE ARE PLOTTING'''
        entropy_save_path = os.path.join(save_dir, f"case1_{env_id}_ppo_vs_rpo_entropy.pdf")            #CASE 1 --DEFUALT
        #entropy_save_path = os.path.join(save_dir, f"case2_mass_{env_id}_ppo_vs_rpo_entropy.pdf")      #CASE 2 MASS
        #entropy_save_path = os.path.join(save_dir, f"case2_grav_{env_id}_ppo_vs_rpo_entropy.pdf")      #CASE 2 GRAVITY
        plt.savefig(entropy_save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Entropy plot saved to {entropy_save_path}")

#Each call below plots curves for a specific environment
'''UNCOMMENT BASED ON WHICH AGENT TRAINING CURVES YOU'D LIKE TO SEE'''

plot_training_curves_combined(env_id="Hopper-v4", save_dir="plots")             #HOPPER
#plot_training_curves_combined(env_id="HalfCheetah-v4", save_dir="plots")       #HALFCHEETAH
#plot_training_curves_combined(env_id="Walker2d-v4", save_dir="plots")          #WALKER2D