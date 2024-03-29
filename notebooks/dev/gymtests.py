import gymnasium as gym
# env = gym.make("LunarLander-v2", render_mode="rgb_array")
env = gym.make("CarRacing-v2", render_mode="rgb_array")
observation, info = env.reset()
print(env.metadata["render_modes"])

# for _ in range(1000):
#     action = env.action_space.sample()  # agent policy that uses the observation and info
#     observation, reward, terminated, truncated, info = env.step(action)

#     if terminated or truncated:
#         observation, info = env.reset()

print("closing")
env.close()