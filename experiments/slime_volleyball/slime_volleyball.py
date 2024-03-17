"""Runs lunar lander experiments.

Algorithms for the paper: cma_mae, cma_me, map_elites_line, map_elites, dqn_me

Usage:
    python slime_volleyball.py [ALGORITHM]

    # For CMA-MAE from the paper:
    python slime_volleyball.py [ALGORITHM] --alpha=0.01
"""
import copy
import csv
import os
import time
from pathlib import Path

import cv2
import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from alive_progress import alive_bar
from dask.distributed import Client, LocalCluster
from matplotlib.patches import Ellipse, Rectangle
#  from pyinstrument import Profiler
from torch import nn

from ribs.archives import GridArchive
from ribs.emitters import (AnnealingEmitter, DQNEmitter, GaussianEmitter,
                           ImprovementEmitter, IsoLineEmitter,
                           OptimizingEmitter)
from ribs.emitters.replay_buffer import Experience, ReplayBuffer
from ribs.optimizers import Optimizer
from ribs.visualize import _retrieve_cmap, grid_archive_heatmap
from slimevolleygym import mlp
from slimevolleygym.slimevolley import SlimeVolleyEnv


class SlimeMLP(nn.Module):
    """MLP for slime volleyball.

    Output can represent Q-values, or just be arbitrary values. Either way, the
    action is taken as the argmax of the outputs.
    """

    def __init__(self, game: mlp.Game):
        super().__init__()

        # All layers have bias.
        self.model = nn.Sequential(
            nn.Linear(game.input_size, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 6),
        )

    def forward(self, x):
        return self.model(x)

    def action(self, obs):
        """Computes action for one observation."""
        obs = torch.from_numpy(obs[None].astype(np.float32))
        return self(obs)[0].argmax().item()

    def serialize(self):
        """Returns 1D array with all parameters in the actor."""
        return np.concatenate(
            [p.data.cpu().detach().numpy().ravel() for p in self.parameters()])

    def deserialize(self, array):
        """Loads parameters from 1D array."""
        array = np.copy(array)
        arr_idx = 0
        for param in self.model.parameters():
            shape = tuple(param.data.shape)
            length = np.product(shape)
            block = array[arr_idx:arr_idx + length]
            if len(block) != length:
                raise ValueError("Array not long enough!")
            block = np.reshape(block, shape)
            arr_idx += length
            param.data = torch.from_numpy(block).float()
        return self


class DiscreteSlimeVolleyEnv(SlimeVolleyEnv):
    """Slime volleyball environment with discrete action space.

    There will be 6 discrete actions available, instead of a box with 3 binary
    values.
    """
    atari_mode = True


def simulate(model,
             seed=None,
             video_env=None,
             save_video_to=None,
             record_traj=True):
    """Simulates the slime volleyball model.

    Args:
        model (np.ndarray): The array of weights for the linear policy.
        seed (int): The seed for the environment.
        video_env (gym.Env): If passed in, this will be used instead of creating
            a new env. This is used primarily for recording video during
            evaluation.
        save_video_to (string): If not None, renders in human mode and saves
            to file path save_video_to.
    Returns:
        obj (float): The remaining points of the right player minus the
            remaining points of the left player at the end of the game.
        num_hits (int): The combined number of times both players hit the ball.
        num_volleys (float): The number of times the ball crosses x=0.
        trajectory (dict): Arrays representing (s, a, r, s', done) of both
            agents.
    """
    #  profiler = Profiler()
    #  profiler.start()

    if video_env is None:
        # Since we are using multiple processes, it is simpler if each worker
        # just creates their own copy of the environment instead of trying to
        # share the environment. This also makes the function "pure." However,
        # we should use the video_env if it is passed in.
        # env = gym.make("SlimeVolley-v0")
        env = DiscreteSlimeVolleyEnv()
    else:
        env = video_env
    env.seed(seed)
    np.random.seed(seed)

    torch.set_num_threads(1)

    policy = SlimeMLP(mlp.games['slimevolley']).deserialize(model).to("cpu")
    policy.eval()

    total_reward = 0.0
    last_x = 0.0
    num_volleys = 0
    obs_right = env.reset()
    obs_left = copy.deepcopy(obs_right)
    done = False

    max_exp = env.t_limit * 2
    if record_traj:
        trajectory = {
            "state":
                np.full((max_exp, mlp.games["slimevolley"].input_size),
                        np.nan,
                        dtype=np.float32),
            "action":
                np.full((max_exp,), np.nan, dtype=np.float32),
            "reward":
                np.full((max_exp,), np.nan, dtype=np.float32),
            "next_state":
                np.full((max_exp, mlp.games["slimevolley"].input_size),
                        np.nan,
                        dtype=np.float32),
            "done":
                np.full((max_exp,), np.nan, dtype=np.float32),
        }
    else:
        trajectory = None

    with torch.inference_mode():
        timestep = 0
        while not done:
            old_obs_right = obs_right
            old_obs_left = obs_left

            # The PyTorch model tends to be a bottleneck, likely because it is
            # not designed for single passes on a CPU.
            action_right, action_left = policy.action(obs_right), policy.action(
                obs_left)
            obs_right, reward, done, info = env.step(action_right, action_left)
            obs_left = info['otherObs']
            total_reward += reward

            if record_traj:
                trajectory["state"][2 * timestep] = old_obs_right
                trajectory["action"][2 * timestep] = action_right
                trajectory["reward"][2 * timestep] = reward
                trajectory["next_state"][2 * timestep] = obs_right
                trajectory["done"][2 * timestep] = done

                trajectory["state"][2 * timestep + 1] = old_obs_left
                trajectory["action"][2 * timestep + 1] = action_left
                trajectory["reward"][2 * timestep + 1] = -reward
                trajectory["next_state"][2 * timestep + 1] = obs_left
                trajectory["done"][2 * timestep + 1] = done

            timestep += 1

            if not video_env is None:
                if save_video_to is None:
                    # if has video_env and no save_video_to, generate video and play
                    env.render()
                else:
                    # if has video_env and save_video_to, generate video and save
                    img = env.render("rgb_array")
                    if 'video' in vars():
                        video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    else:
                        video = cv2.VideoWriter(save_video_to,
                                                cv2.VideoWriter_fourcc(*'mp4v'),
                                                40, img.shape[:2][::-1])
                        video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            cur_x = env.game.ball.x - env.game.fence.x
            if cur_x * last_x < 0:
                #print('volley!')
                num_volleys += 1
            last_x = cur_x

    num_hits = env.game.num_right_hits + env.game.num_left_hits

    if video_env is None:
        env.close()

    if not save_video_to is None:
        video.release()

    best_obj = 5
    worst_obj = -5
    obj = (total_reward - worst_obj) / (best_obj - worst_obj) * 100

    #  profiler.stop()
    #  profiler.print()

    return obj, num_hits, num_volleys, trajectory


def simulate_parallel(client, sols, seed, algorithm):
    futures = client.map(
        lambda model: simulate(model, seed, record_traj=algorithm == "dqn_me"),
        sols)
    results = client.gather(futures)

    objs, meas, trajectories = [], [], []

    # Process the results.
    for obj, num_hits, num_volleys, trajectory in results:
        objs.append(obj)
        meas.append([num_hits, num_volleys])
        trajectories.append(trajectory)

    return np.array(objs), np.array(meas), trajectories


def create_optimizer(algorithm,
                     dim,
                     replay_buffer,
                     alpha=1.0,
                     resolution=100,
                     seed=None):
    """Creates an optimizer based on the algorithm name.

    Args:
        algorithm (str): Name of the algorithm passed into sphere_main.
        dim (int): Dimensionality of the sphere function.
        alpha (float): The archive learning rate.
        resolution (int): The archive resolution (res x res).
        seed (int): Main seed or the various components.
    Returns:
        Optimizer: A ribs Optimizer for running the algorithm.
    """
    bounds = [(0.0, 200), (0.0, 50)]
    initial_sol = np.zeros((dim,))
    batch_size = 36
    num_emitters = 15
    grid_dims = (resolution, resolution)

    # Create archive.
    if algorithm in [
            "dqn_me",
            "map_elites",
            "map_elites_line",
            "cma_me",
            "cma_me_star",
            "cma_me_io",
    ]:
        archive = GridArchive(grid_dims, bounds, seed=seed)
    elif algorithm in ["cma_mae"]:
        archive = GridArchive(
            grid_dims,
            bounds,
            archive_learning_rate=alpha,
            seed=seed,
        )
    else:
        raise ValueError(f"Algorithm `{algorithm}` is not recognized")

    # Maintain a passive elitist archive
    passive_archive = GridArchive(grid_dims, bounds, seed=seed)
    passive_archive.initialize(dim)

    # Create emitters. Each emitter needs a different seed, so that they do not
    # all do the same thing.
    emitter_seeds = [None] * num_emitters if seed is None else list(
        range(seed, seed + num_emitters))
    if algorithm in ["dqn_me"]:
        emitters = [
            # Two emitters, each with half the total batch size.
            IsoLineEmitter(
                archive,
                initial_sol,
                iso_sigma=0.5,
                line_sigma=0.2,
                batch_size=(num_emitters * batch_size) // 2,
                seed=emitter_seeds[0],
            ),
            DQNEmitter(
                archive,
                initial_sol,
                sigma0=0.5,
                batch_size=(num_emitters * batch_size) // 2,
                replay_buffer=replay_buffer,
                seed=emitter_seeds[1],
                network_fn=lambda: SlimeMLP(mlp.games['slimevolley']),
                args={
                    "batch_size": 128,
                    "train_itrs": 10,
                    "target_freq": 2,
                    "gamma": 0.99,
                    "learning_rate": 2.5e-4,
                    "tau": 1.0,
                },
            ),
        ]
    elif algorithm in ["map_elites"]:
        emitters = [
            GaussianEmitter(archive,
                            initial_sol,
                            0.5,
                            batch_size=batch_size,
                            seed=s) for s in emitter_seeds
        ]
    elif algorithm in ["map_elites_line"]:
        emitters = [
            IsoLineEmitter(archive,
                           initial_sol,
                           iso_sigma=0.5,
                           line_sigma=0.2,
                           batch_size=batch_size,
                           seed=s) for s in emitter_seeds
        ]
    elif algorithm in ["cma_me"]:
        emitters = [
            ImprovementEmitter(archive,
                               initial_sol,
                               0.5,
                               restart_rule='basic',
                               selection_rule='mu',
                               batch_size=batch_size,
                               seed=s) for s in emitter_seeds
        ]
    elif algorithm in ["cma_me_io"]:
        emitters = []
        split_count = len(emitter_seeds) // 2
        emitters += [
            OptimizingEmitter(archive,
                              initial_sol,
                              0.5,
                              restart_rule='basic',
                              selection_rule='mu',
                              batch_size=batch_size,
                              seed=s) for s in emitter_seeds[:split_count]
        ]
        emitters += [
            ImprovementEmitter(archive,
                               initial_sol,
                               0.5,
                               restart_rule='basic',
                               selection_rule='mu',
                               batch_size=batch_size,
                               seed=s) for s in emitter_seeds[split_count:]
        ]
    elif algorithm in ["cma_me_star"]:
        emitters = [
            ImprovementEmitter(archive,
                               initial_sol,
                               0.5,
                               restart_rule='no_improvement',
                               selection_rule='filter',
                               batch_size=batch_size,
                               seed=s) for s in emitter_seeds
        ]
    elif algorithm in ["cma_mae"]:
        emitters = [
            AnnealingEmitter(archive,
                             initial_sol,
                             0.5,
                             restart_rule='basic',
                             batch_size=batch_size,
                             seed=s) for s in emitter_seeds
        ]

    return Optimizer(archive, emitters), passive_archive


def save_heatmap(archive, heatmap_path):
    """Saves a heatmap of the archive to the given path.

    Args:
        archive (GridArchive or CVTArchive): The archive to save.
        heatmap_path: Image path for the heatmap.
    """
    plt.figure(figsize=(8, 6))
    grid_archive_heatmap(archive, vmin=0, vmax=100, cmap='viridis')
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close(plt.gcf())


def run_experiment(algorithm,
                   trial_id,
                   total_trials,
                   client,
                   alpha=1.0,
                   arch_res_exp=False,
                   resolution=100,
                   init_pop=100,
                   itrs=2500,
                   outdir="logs",
                   log_freq=1,
                   log_arch_freq=500,
                   seed=None):

    # Create a directory for this specific trial.
    if algorithm in ["cma_mae"]:
        s_logdir = os.path.join(outdir, f"{algorithm}_{resolution}_{alpha}",
                                f"trial_{trial_id}")
    else:
        s_logdir = os.path.join(outdir, f"{algorithm}_{resolution}",
                                f"trial_{trial_id}")
    logdir = Path(s_logdir)
    if not logdir.is_dir():
        logdir.mkdir()

    # Create a new summary file
    summary_filename = os.path.join(s_logdir, "summary.csv")
    if os.path.exists(summary_filename):
        os.remove(summary_filename)
    with open(summary_filename, 'w') as summary_file:
        writer = csv.writer(summary_file)
        writer.writerow(
            ['Iteration', 'QD-Score', 'Coverage', 'Maximum', 'Average'])

    # If we are running a resolution experiment, override the resolution.
    if arch_res_exp:

        # Linearly interpolate
        index = 0.0
        if total_trials > 1:
            index = trial_id / (total_trials - 1.0)

        min_count = 50
        max_count = 500
        new_resolution = index * (max_count - min_count) + min_count
        resolution = int(new_resolution + 1e-9)
        cell_count = resolution**2

        ratio = cell_count / (100.0 * 100.0)
        alpha = 1.0 - np.power(1.0 - alpha, ratio)

        print('Running exp {} with resolution = {} and alpha = {}'.format(
            trial_id, resolution, alpha))

    is_init_pop = algorithm in [
        'map_elites',
        'map_elites_line',
    ]

    # Select the objective based on the input.
    policy = SlimeMLP(mlp.games['slimevolley'])
    dim = len(policy.serialize())

    if algorithm == "dqn_me":
        replay_buffer = ReplayBuffer(
            capacity=1_000_000,
            obs_shape=(mlp.games['slimevolley'].input_size,),
            action_shape=(1,),
        )
    else:
        replay_buffer = None

    optimizer, passive_archive = create_optimizer(
        algorithm,
        dim,
        replay_buffer,
        alpha=alpha,
        resolution=resolution,
        seed=seed,
    )
    archive = optimizer.archive

    best = 0.0
    non_logging_time = 0.0
    with alive_bar(itrs) as progress:

        if is_init_pop:
            # Sample initial population
            sols = np.array(
                [np.random.normal(size=dim) for _ in range(init_pop)])

            objs, measures, trajectories = simulate_parallel(
                client, sols, seed, algorithm)
            best = max(best, max(objs))

            # Add each solution to the archive.
            for i in range(len(sols)):
                archive.add(sols[i], objs[i], measures[i])
                passive_archive.add(sols[i], objs[i], measures[i])

        for itr in range(1, itrs + 1):
            itr_start = time.time()

            sols = optimizer.ask()
            objs, measures, trajectories = simulate_parallel(
                client, sols, seed, algorithm)
            best = max(best, max(objs))
            optimizer.tell(objs, measures)

            # Add to replay buffer.
            if algorithm == "dqn_me":
                for trajectory in trajectories:
                    for i in range(len(trajectory["reward"])):
                        if np.isnan(trajectory["reward"][i]):
                            break
                        replay_buffer.add(
                            Experience(trajectory["state"][i],
                                       trajectory["action"][i],
                                       trajectory["reward"][i],
                                       trajectory["next_state"][i],
                                       trajectory["done"][i]))

            # Update the passive elitist archive.
            for i in range(len(sols)):
                passive_archive.add(sols[i], objs[i], measures[i])

            non_logging_time += time.time() - itr_start
            progress()

            # Save the archive at the given frequency.
            # Always save on the final iteration.
            final_itr = itr == itrs
            if itr % log_arch_freq == 0 or final_itr:

                # Save a full archive for analysis.
                df = passive_archive.as_pandas(include_solutions=final_itr)
                df.to_pickle(os.path.join(s_logdir, f"archive_{itr:08d}.pkl"))

                # Save a heatmap image to observe how the trial is doing.
                save_heatmap(passive_archive,
                             os.path.join(s_logdir, f"heatmap_{itr:08d}.png"))

            # Update the summary statistics for the archive
            if (itr > 0 and itr % log_freq == 0) or final_itr:
                with open(summary_filename, 'a') as summary_file:
                    writer = csv.writer(summary_file)

                    sum_obj = 0
                    num_filled = 0
                    num_bins = passive_archive.bins
                    for sol, obj, beh, idx, meta in zip(
                            *passive_archive.data()):
                        num_filled += 1
                        sum_obj += obj
                    qd_score = sum_obj / num_bins
                    average = sum_obj / num_filled
                    coverage = 100.0 * num_filled / num_bins
                    data = [itr, qd_score, coverage, best, average]
                    writer.writerow(data)


def slime_volleyball_main(algorithm,
                          workers=4,
                          trials=20,
                          arch_res_exp=False,
                          alpha=1.0,
                          resolution=50,
                          init_pop=100,
                          itrs=2500,
                          outdir="logs",
                          log_freq=1,
                          log_arch_freq=500,
                          seed=None):
    """Experiment tool for the slime_volleyball domain from the CMA-ME paper.

    Args:
        algorithm (str): Name of the algorithm.
        workers (int): Number of parallel dask clients used for gym evaluation.
        trials (int): Number of experimental trials to run.
        arch_res_exp (bool): Runs the archive resolution experiment instead.
        alpha (float): The archive learning rate.
        resolution (int): The resolution of dimension in the archive (res x res).
        init_pop (int): Initial population size for MAP-Elites (ignored for CMA variants).
        itrs (int): Iterations to run.
        outdir (str): Directory to save output.
        log_freq (int): Number of iterations between computing QD metrics and updating logs.
        log_arch_freq (int): Number of iterations between saving an archive and generating heatmaps.
        seed (int): Seed for the algorithm. By default, there is no seed.
    """

    if arch_res_exp:
        print(
            f"Running arch res experiment on slime volleyball and alpha={alpha}."
        )
    else:
        print(
            f"Running slime volleyball, alpha={alpha}, and resolution={resolution}."
        )

    # Create a shared logging directory for the experiments for this algorithm.
    if algorithm in ["cma_mae"]:
        s_logdir = os.path.join(outdir, f"{algorithm}_{resolution}_{alpha}")
    else:
        s_logdir = os.path.join(outdir, f"{algorithm}_{resolution}")
    logdir = Path(s_logdir)
    outdir = Path(outdir)
    if not outdir.is_dir():
        outdir.mkdir()
    if not logdir.is_dir():
        logdir.mkdir()

    # Run all trials in parallel.
    cluster = LocalCluster(
        processes=True,  # Each worker is a process.
        n_workers=workers,  # Create one worker per trial (assumes >=trials cores)
        threads_per_worker=1,  # Each worker process is single-threaded.
    )
    client = Client(cluster)
    for cur_id in range(trials):
        run_experiment(
            algorithm,
            cur_id,
            trials,
            client,
            alpha=alpha,
            arch_res_exp=arch_res_exp,
            resolution=resolution,
            init_pop=init_pop,
            itrs=itrs,
            outdir=outdir,
            log_freq=log_freq,
            log_arch_freq=log_arch_freq,
            seed=seed,
        )


def play_policy(archive_path, max_mea, qcut_quantile, seed, outdir=None):
    '''
    max_mea is a list containing a boolean for each measure dimension.
        if a boolean is True, play_policy will demonstrate a high fitness policy
        with a relatively high value for this measure; if False it will prefer
        a low value for this measure.
    qcut_quantile is list containing an integer qcut quantile for each measure
        dimension. The corresponding measure will be divided into qcut_quantile bins
        for later ranking. This param is for balancing between the fineness when choosing
        each measure. For example, when qcut_quantile=[1,1], play_policy will simply choose
        the highest fitness.
    '''
    df = pd.read_pickle(archive_path)

    for i, mea in enumerate(
            df.columns[df.columns.str.startswith('behavior')].tolist()):
        df[mea + '_bin'] = pd.qcut(df[mea], qcut_quantile[i])

    df.sort_values(by=df.columns[df.columns.str.endswith('bin')].tolist() +
                   ['objective'],
                   ascending=max_mea + [True],
                   inplace=True)

    plt.figure(figsize=(8, 6))
    cmap = _retrieve_cmap('viridis')

    lower_mea_bounds = (0.0, 0.0)
    upper_mea_bounds = (200, 50)
    x_dim, y_dim = (50, 50)
    x_bounds = np.linspace(lower_mea_bounds[0], upper_mea_bounds[0], x_dim + 1)
    y_bounds = np.linspace(lower_mea_bounds[1], upper_mea_bounds[1], y_dim + 1)

    colors = np.full((y_dim, x_dim), np.nan)
    for row in df.itertuples():
        colors[row.index_1, row.index_0] = row.objective

    ax = plt.gca()
    ax.set_xlim(lower_mea_bounds[0], upper_mea_bounds[0])
    ax.set_ylim(lower_mea_bounds[1], upper_mea_bounds[1])
    plt.xlabel('number of hits')
    plt.ylabel('number of volleys')

    pcm_kwargs = {}
    vmin = 0
    vmax = 100
    t = ax.pcolormesh(x_bounds,
                      y_bounds,
                      colors,
                      cmap=cmap,
                      vmin=vmin,
                      vmax=vmax,
                      **pcm_kwargs)

    ax.figure.colorbar(t, ax=ax, pad=0.1)
    patch_coords = (x_bounds[df.tail(1).index_0] +
                    (upper_mea_bounds[0] - lower_mea_bounds[0]) / x_dim / 2,
                    y_bounds[df.tail(1).index_1] +
                    (upper_mea_bounds[1] - lower_mea_bounds[1]) / y_dim / 2)
    unit_cell_size = ((upper_mea_bounds[0] - lower_mea_bounds[0]) / x_dim,
                      (upper_mea_bounds[1] - lower_mea_bounds[1]) / y_dim)
    ax.add_patch(
        Ellipse(patch_coords,
                width=unit_cell_size[0] * 7.5,
                height=unit_cell_size[1] * 7.5,
                color='red',
                fill=False))
    ax.add_patch(
        Rectangle((x_bounds[df.tail(1).index_0], y_bounds[df.tail(1).index_1]),
                  width=unit_cell_size[0],
                  height=unit_cell_size[1],
                  color='red',
                  fill=True))

    plt.tight_layout()

    model = df.tail(
        1).loc[:, df.columns.str.startswith('solution')].to_numpy().squeeze()

    best_obj = 5
    worst_obj = -5
    restored_obj = df.tail(1).objective.iloc[0] / 100 * (best_obj -
                                                         worst_obj) + worst_obj
    print(
        f" behavior_0(num_hits):{df.tail(1).behavior_0.iloc[0]} \n behavior_1(num_volleys):{df.tail(1).behavior_1.iloc[0]} \n objective:{restored_obj}"
    )
    if outdir is None:
        plt.show()
        env = DiscreteSlimeVolleyEnv()
        simulate(model, seed=seed, video_env=env, save_video_to=None)
    else:
        heatmap_path = os.path.join(outdir, "heatmap.png")
        video_path = os.path.join(outdir, "video.mp4")
        txt_path = os.path.join(outdir, "info.txt")
        outdir = Path(outdir)
        if not outdir.is_dir():
            outdir.mkdir(parents=True)

        plt.savefig(heatmap_path)
        plt.close(plt.gcf())

        env = DiscreteSlimeVolleyEnv()
        simulate(model, seed=seed, video_env=env, save_video_to=video_path)

        with open(txt_path, 'w') as info_txt:
            info_txt.writelines([
                f"behavior_0(num_hits)   : {df.tail(1).behavior_0.iloc[0]}\n",
                f"behavior_1(num_volleys): {df.tail(1).behavior_1.iloc[0]}\n",
                f"objective              : {restored_obj}"
            ])


def collect_video_data():
    param_seq = [{
        "archive_path": "logs/cma_mae_50_0.01/trial_0/archive_00002500.pkl",
        "max_mea": [True, True],
        "qcut_quantile": [1, 1],
        "seed": 14,
        "outdir": "video_data/best"
    }, {
        "archive_path": "logs/cma_mae_50_0.01/trial_0/archive_00002500.pkl",
        "max_mea": [True, True],
        "qcut_quantile": [5, 5],
        "seed": 14,
        "outdir": "video_data/tt"
    }, {
        "archive_path": "logs/cma_mae_50_0.01/trial_0/archive_00002500.pkl",
        "max_mea": [True, False],
        "qcut_quantile": [5, 5],
        "seed": 14,
        "outdir": "video_data/tf"
    }, {
        "archive_path": "logs/cma_mae_50_0.01/trial_0/archive_00002500.pkl",
        "max_mea": [False, True],
        "qcut_quantile": [5, 5],
        "seed": 14,
        "outdir": "video_data/ft"
    }, {
        "archive_path": "logs/cma_mae_50_0.01/trial_0/archive_00002500.pkl",
        "max_mea": [False, False],
        "qcut_quantile": [5, 5],
        "seed": 14,
        "outdir": "video_data/ff"
    }]

    for param in param_seq:
        play_policy(**param)


if __name__ == '__main__':
    fire.Fire(slime_volleyball_main)

    # play_policy(
    #     "logs/cma_mae_50_0.01/trial_0/archive_00002500.pkl",
    #     max_mea=[True, True],
    #     qcut_quantile=[5, 5],
    #     seed=14,
    #     outdir=None
    # )
    # collect_video_data()
