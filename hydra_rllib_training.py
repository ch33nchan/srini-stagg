import hydra
from omegaconf import DictConfig, OmegaConf
import csv
import os
import time
from zoo_envs.env.marl_gwsh import MARL_GWSH
import ray
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
import torch
from metric_evaluation import eval_metrics

@hydra.main(version_base=None, config_path="conf", config_name="config")
def training(cfg: DictConfig):
    env_config = cfg["env"]
    env_creator = lambda config: MARL_GWSH(**config)

    ray.init()
    register_env(
        "MARL_GWSH",
        lambda config: ParallelPettingZooEnv(env_creator(env_config)),
    )

    policies = {
        f"agent_{i}" for i in range(env_config["number_of_agents"])
    }

    num_gpu = 1 if torch.cuda.is_available() else 0

    ppo_config = PPOConfig().environment(env="MARL_GWSH").env_runners(num_env_runners=3).training(
        train_batch_size=512,
        lr=2.5e-4,
        gamma=0.99,
        lambda_=0.95,
        use_gae=True,
        clip_param=0.3,
        grad_clip=None,
        entropy_coeff=0.01,
        vf_loss_coeff=0.5,
        sgd_minibatch_size=64,
        num_sgd_iter=1,
    ).debugging(log_level="ERROR").framework(framework="torch").resources(num_gpus=num_gpu).multi_agent(
        policies=policies,
        policy_mapping_fn=(lambda aid, *args, **kwargs: aid)
    ).rl_module(
        model_config_dict={
            "fcnet_hiddens": [128, 64],
            # "conv_filters": [[16, 4, 4], [32, 4, 2], [64, 3, 1]],
            # "post_fcnet_hiddens": [64],
            "use_lstm": True,
            "lstm_cell_size": 64,
            "lstm_use_prev_action": True,
            "lstm_use_prev_reward": True,
            "vf_share_layers": False,
            "soft_horizon": True,
            "no_done_at_end": False,
            "horizon": 200, },
        rl_module_spec=MultiAgentRLModuleSpec(
            module_specs={p: SingleAgentRLModuleSpec() for p in policies},
        )
    )

    ppo = ppo_config.build()
    beginning_time = time.time()
    start_time = time.time()

    checkpoints_dir = add_number_if_necessary(cfg["training"]["checkpoints_dir"])
    os.makedirs(checkpoints_dir, exist_ok=True)
    logging_file = os.path.join(checkpoints_dir, "log.csv")
    eval_file = os.path.join(checkpoints_dir, "eval.csv")

    cfg["training"]["checkpoints_dir"] = checkpoints_dir

    with open(os.path.join(checkpoints_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    with open(logging_file, "w", newline="\n") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(['agent_0', 'agent_1', 'agent_2'])

    with open(eval_file, "w", newline="\n") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(['agent_0_mean_rewards', 'agent_1_mean_rewards', 'agent_2_mean_rewards',
                         'agent_0_stag_percentage', 'agent_1_stag_percentage', 'agent_2_stag_percentage',
                         'agent_0_plant_percentage', 'agent_1_plant_percentage', 'agent_2_plant_percentage'])

    for i in range(cfg["training"]["num_iterations"] + 1):
        result = ppo.train()
        reward_list = [result["env_runners"]["policy_reward_mean"]["agent_0"],
                       result["env_runners"]["policy_reward_mean"]["agent_1"],
                       result["env_runners"]["policy_reward_mean"]["agent_2"]]
        with open(logging_file, "a", newline="\n") as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow(reward_list)
        if i % cfg["training"]["time_report_interval"] == 0:
            end_time = time.time()
            time_elapsed = end_time - start_time
            start_time = time.time()
            rounded_reward_list = [float("%.2f" % reward) for reward in reward_list]
            print(f"Time elapsed for iteration {i}: {time_elapsed:.2f}, total time elapsed: {end_time - beginning_time:.2f}, current reward: {rounded_reward_list}")
        if cfg["training"]["evaluation_interval"] != 0 and i % cfg["training"]["evaluation_interval"] == 0:
            result = eval_metrics(ppo, env_creator, env_config)
            with open(eval_file, "a", newline="\n") as f:
                writer = csv.writer(f, delimiter=",")
                writer.writerow(result)
        if i % cfg["training"]["checkpoint_interval"] == 0:
            checkpoints_dir_i = os.path.join(checkpoints_dir, f"checkpoint_{i}")
            ppo.save(checkpoints_dir_i)
            print(f"Checkpoint saved at iteration {i}")

    ppo.stop()
    ray.shutdown()

def add_number_if_necessary(path):
    if not os.path.exists(path):
        return path
    i = 1
    while True:
        new_path = path + f"_{i}"
        if not os.path.exists(new_path):
            return new_path
        i += 1

if __name__ == "__main__":
    training()