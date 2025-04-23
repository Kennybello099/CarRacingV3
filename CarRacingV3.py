import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os
import numpy as np

class CarRacingTrainer:
    def __init__(self, env_name="CarRacing-v3", model_path="CarRacingV3/Training/Saved Models/PPO_model", 
                 total_timesteps=10000, timesteps_per_iteration=2000, 
                 initial_n_steps=1024, learning_rate=1e-4):
        """Initialize the CarRacingTrainer with environment and training parameters."""
        self.env_name = env_name
        self.model_path = os.path.join(model_path)
        self.total_timesteps = total_timesteps
        self.timesteps_per_iteration = timesteps_per_iteration
        self.iterations = total_timesteps // timesteps_per_iteration
        self.initial_n_steps = initial_n_steps
        self.learning_rate = learning_rate
        
        # Setup environment without randomization
        self.env = self._create_environment()
        self.model = None
        self.n_steps = self.initial_n_steps

    def _create_environment(self):
        """Create the Gymnasium environment."""
        env = gym.make(self.env_name, continuous=True, render_mode="human")
        return DummyVecEnv([lambda: env])

    def _initialize_model(self):
        """Initialize or load the PPO model."""
        if os.path.exists(self.model_path + ".zip"):
            try:
                self.model = PPO.load(self.model_path, env=self.env)
                print(f"Model loaded successfully from {self.model_path}.zip")
                self.n_steps = self.initial_n_steps
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                exit(1)
        else:
            print(f"No model found at {self.model_path}.zip. Training a new PPO model...")
            try:
                self.model = PPO(
                    "CnnPolicy",
                    self.env,
                    verbose=1,
                    learning_rate=self.learning_rate,
                    n_steps=self.initial_n_steps,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.98,
                )
                self.n_steps = self.initial_n_steps
            except Exception as e:
                print(f"Error during initialization: {str(e)}")
                exit(1)

    def train(self):
        """Train the model."""
        self._initialize_model()

        for i in range(self.iterations):
            try:
                self.model.learn(total_timesteps=self.timesteps_per_iteration)
                mean_reward, std_reward = evaluate_policy(self.model, self.env, n_eval_episodes=3, render=False)
                print(f"Iteration {i+1}/{self.iterations} - Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

                if mean_reward > 0:
                    self.n_steps = min(int(self.n_steps * 1.2), 2048)
                    print(f"Car on right path! Increasing n_steps to {self.n_steps}")
                else:
                    self.n_steps = max(int(self.n_steps * 0.8), 256)
                    print(f"Car off path. Reducing n_steps to {self.n_steps}")

                self.model.save(self.model_path)
                print(f"Model saved to {self.model_path}.zip")

            except Exception as e:
                print(f"Error during training iteration {i+1}: {str(e)}")
                break

    def evaluate(self):
        """Evaluate the trained model and save the final version."""
        try:
            mean_reward, std_reward = evaluate_policy(self.model, self.env, n_eval_episodes=3, render=True)
            print(f"Final Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        except Exception as e:
            print(f"Error during final evaluation: {str(e)}")

        self.model.save(self.model_path)
        print(f"Final model saved to {self.model_path}.zip")

    def run_post_training_episodes(self, n_episodes=3, epsilon=0.1):
        """Run post-training episodes with epsilon-greedy action selection."""
        try:
            print(f"Running {n_episodes} post-training episodes with epsilon={epsilon}...")
            env = gym.make(self.env_name, continuous=True, render_mode="human")
            for episode in range(n_episodes):
                obs = env.reset()[0]  # Get the observation from the reset tuple
                done = False
                total_reward = 0
                step = 0
                while not done:
                    # Generate a random number for epsilon-greedy decision
                    if np.random.random() < epsilon:
                        # Explore: select a random action from the action space
                        action = env.action_space.sample()
                    else:
                        # Exploit: use the PPO model's predicted action
                        action, _ = self.model.predict(obs, deterministic=True)
                    
                    obs, reward, terminated, truncated, _ = env.step(action)
                    total_reward += reward
                    step += 1
                    env.render()
                    done = terminated or truncated
                print(f"Episode {episode+1} - Total Reward: {total_reward:.2f}, Steps: {step}")
            env.close()
        except Exception as e:
            print(f"Error during post-training episodes: {str(e)}")

    def close(self):
        """Close the environment."""
        self.env.close()

    def run(self):
        """Run the full training and evaluation pipeline."""
        self.train()
        self.evaluate()
        self.run_post_training_episodes(n_episodes=3, epsilon=0.2)  # Run 3 episodes with epsilon=0.2
        self.close()

if __name__ == "__main__":
    trainer = CarRacingTrainer()
    trainer.run()