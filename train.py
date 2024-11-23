import ray
from ray.rllib.env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPO
from zoo_envs.env.marl_gwsh import MARL_GWSH
from ray.tune.registry import register_env
from omegaconf import OmegaConf

# Initialize Ray
ray.init()

# Load environment configuration
with open('conf/config.yaml', 'r') as file:
    env_config = OmegaConf.load(file)['env']

# Register environment
env_creator = lambda config: MARL_GWSH(**config)
register_env(
    "MARL_GWSH",
    lambda config: ParallelPettingZooEnv(env_creator(config)),
)

# Configure PPO algorithm
config = {
    "env": "MARL_GWSH",
    "num_workers": 1,  # Can increase for distributed training
    "train_batch_size": 4000,
    "rollout_fragment_length": 200,
    "sgd_minibatch_size": 128,
    "lr": 1e-4,
    "gamma": 0.99,  # Discount factor for rewards
    "lambda": 0.95,  # GAE lambda
    "model": {
        "fcnet_hiddens": [128, 128],
        "fcnet_activation": "tanh",
    },
    "framework": "torch",  # You can use "tf" if you prefer TensorFlow
}

# Instantiate PPO with the above configuration
ppo = PPO(config=config)

# Start training the model
for i in range(1000):  # Train for 1000 iterations, adjust as needed
    result = ppo.train()
    print(f"Iteration {i}: {result['episode_reward_mean']}")

    # Save checkpoints periodically
    if i % 50 == 0:  # Save every 50 iterations
        checkpoint = ppo.save("cloud_checkpoint/ppo_model_checkpoint")
        print(f"Checkpoint saved at {checkpoint}")

# Shutdown Ray after training
ray.shutdown()