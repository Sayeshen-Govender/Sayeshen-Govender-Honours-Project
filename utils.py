import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import uniform_filter1d

def plot_metrics_per_seed(results, outdir="plots"):

    """
    Print and log summary metrics for a single seed
    Currently outputs average return and standard deviation of return
    """

    os.makedirs(outdir, exist_ok=True)

    print(f"Avg return: {results['avg_return']:.2f}")
    print(f"Std dev return: {results['std_return']:.2f}")


def results_summary(results_list, outdir="plots"):
    """
    Summarizes experiment results across multiple seeds
    Computes mean and std of returns across seeds and prints summary
    """

    import os
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)

    if not results_list:
        print("No results to summarize (results_list is empty). Skipping plots.")
        return

    #Aggregate all returns 
    all_returns = []        #Stores all return sequences
    avg_returns = []        #Average returns per seed
    std_returns = []        #standard deviation of returns per seed

    #Collect metrics from each result dict
    for result in results_list:
        rets = result.get("returns", [])
        all_returns.extend(rets)
        avg_returns.append(result.get("avg_return", np.nan))
        std_returns.append(result.get("std_return", np.nan))

    overall_avg_return = np.nanmean(avg_returns) if len(avg_returns) > 0 else np.nan
    overall_std_return = np.nanmean(std_returns) if len(std_returns) > 0 else np.nan

    print(f"Avg return (across seeds): {overall_avg_return:.2f}")
    print(f"Std dev return (avg across seeds): {overall_std_return:.2f}")



def plot_training_curves(all_returns_per_seed, all_timesteps_per_seed, all_entropies_per_seed=None, save_path="plots/training_return_across_seeds.png", smoothing_window=10):
    """
    Plots training curves (returns and entropy) across multiple seeds
    Aligns seeds on a common timestep grid
    Interpolates to smooth differences in timestps
    Computes mean ± std shading
    Optionally smooths curves with uniform filter

    THIS ONLY CREATES CURVES FOR ONE ALGO. EITHER ONLY PPO OR ONLY RPO ON THE GRAPH. 
    curves.py PLOTS BOTH PPO AND RPO ON SAME GRAPH
    """
    
    if not all_returns_per_seed or not all_timesteps_per_seed:
        raise ValueError("Missing training data for plotting (empty returns or timesteps).")

    # Filter and validate data
    valid_returns = []
    valid_timesteps = []
    valid_entropies = [] if all_entropies_per_seed is not None else None
    for i, (returns, timesteps) in enumerate(zip(all_returns_per_seed, all_timesteps_per_seed)):
        returns = np.array(returns).flatten()
        timesteps = np.array(timesteps).flatten()
        # Skip invalid or empty data
        if len(returns) == 0 or len(timesteps) == 0:
            print(f"Warning: Empty data for seed index {i}. Skipping.")
            continue
        if len(returns) != len(timesteps):
            print(f"Warning: Mismatch in lengths for seed index {i} (returns: {len(returns)}, timesteps: {len(timesteps)}). Skipping.")
            continue
        if not np.all(np.diff(timesteps) >= 0):
            #Fix non-monotonic timesteps by sorting
            print(f"Warning: Timesteps not monotonic for seed index {i}. Sorting timesteps.")
            sorted_indices = np.argsort(timesteps)
            timesteps = timesteps[sorted_indices]
            returns = returns[sorted_indices]
        
        print(f"Seed index {i}: Returns shape: {returns.shape}, Timesteps shape: {timesteps.shape}")
        valid_returns.append(returns)
        valid_timesteps.append(timesteps)
        
        #Validate entropy if provided
        if all_entropies_per_seed is not None:
            entropies = np.array(all_entropies_per_seed[i]).flatten()
            if len(entropies) == 0:
                print(f"Warning: Empty entropy data for seed index {i}. Skipping.")
                continue
            if len(entropies) != len(timesteps):
                print(f"Warning: Mismatch in lengths for entropy in seed index {i} (entropies: {len(entropies)}, timesteps: {len(timesteps)}). Skipping.")
                continue
            valid_entropies.append(entropies)

    if not valid_returns:
        raise ValueError("No valid training data across seeds after filtering.")

    #Determine common timestep grid
    max_common_timestep = min(ts[-1] for ts in valid_timesteps)
    if max_common_timestep <= 0:
        raise ValueError("Invalid maximum timestep (≤ 0).")
    timestep_grid = np.linspace(0, max_common_timestep, num=500)

    #Interpolate returns to common grid
    interpolated_returns = []
    for returns, timesteps in zip(valid_returns, valid_timesteps):
        interp = np.interp(timestep_grid, timesteps, returns)
        interpolated_returns.append(interp)
    interpolated_returns = np.array(interpolated_returns)

    #Compute mean and std for returns
    mean_returns = np.mean(interpolated_returns, axis=0)
    std_returns = np.std(interpolated_returns, axis=0)

    #Smoothing returns with uniform moving average filter
    if smoothing_window > 1:
        mean_returns = uniform_filter1d(mean_returns, size=smoothing_window)
        std_returns = uniform_filter1d(std_returns, size=smoothing_window)

    #Plotting returns
    plt.figure(figsize=(10, 6))
    plt.plot(timestep_grid, mean_returns, color="blue", label="Mean Return")
    plt.fill_between(timestep_grid, 
                     mean_returns - std_returns, 
                     mean_returns + std_returns, 
                     color="blue", alpha=0.2, label="±1 Std. Dev.")
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Return")
    plt.title("Training Returns Across Seeds")
    plt.legend()
    plt.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Return plot saved to {save_path}")

    #Plot entropies 
    if valid_entropies is not None and len(valid_entropies) > 0:
        interpolated_entropies = []
        for entropies, timesteps in zip(valid_entropies, valid_timesteps):
            interp = np.interp(timestep_grid, timesteps, entropies)
            interpolated_entropies.append(interp)
        interpolated_entropies = np.array(interpolated_entropies)

        #Compute mean and std for entropies
        mean_entropies = np.mean(interpolated_entropies, axis=0)
        std_entropies = np.std(interpolated_entropies, axis=0)

        #Smooth entropies using uniform moving average
        if smoothing_window > 1:
            mean_entropies = uniform_filter1d(mean_entropies, size=smoothing_window)
            std_entropies = uniform_filter1d(std_entropies, size=smoothing_window)

        #Plotting entropies
        entropy_save_path = save_path.replace("return", "entropy")
        plt.figure(figsize=(10, 6))
        plt.plot(timestep_grid, mean_entropies, color="green", label="Mean Entropy")
        plt.fill_between(timestep_grid, 
                         mean_entropies - std_entropies, 
                         mean_entropies + std_entropies, 
                         color="green", alpha=0.2, label="±1 Std. Dev.")
        plt.xlabel("Timesteps")
        plt.ylabel("Policy Entropy")
        plt.title("Training Policy Entropy Across Seeds")
        plt.legend()
        plt.grid(True, alpha=0.3)

        os.makedirs(os.path.dirname(entropy_save_path), exist_ok=True)
        plt.savefig(entropy_save_path)
        plt.close()
        print(f"Entropy plot saved to {entropy_save_path}")