import gymnasium as gym
import torch

from agent import Agent

N_EPISODES = 1000

def run_episode(env: gym.Env, agent: Agent) -> float:
    obs, _ = env.reset()

    is_episode_done = False
    total_reward = 0
    while not is_episode_done:
        obs = torch.from_numpy(obs)
        action = agent.select_action(obs)
        obs, reward, is_terminated, is_truncated, _ = env.step(action)
        is_episode_done = is_terminated or is_truncated
        total_reward += reward

    return total_reward

def main() -> None:
    env = gym.make('LunarLander-v3', render_mode='human')
    agent = Agent(n_obs=8, n_actions=4)

    for episode in range(N_EPISODES):
        reward = run_episode(env, agent)
        print(f"Episode {episode + 1}\t Reward: {round(reward, 2)}\t Success: {reward>=200}")

if __name__ == '__main__':
    main()