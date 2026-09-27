import gymnasium as gym
from bioagents.gym.agent_env import register_bioagent_gym

register_bioagent_gym()

env = gym.make(
    "BioAgent-v0",
    domain="clinical_diagnosis",
    task_id="dx_pneumonia_001"
)

# 1. Reset environment
obs, info = env.reset()

print("=== INITIAL OBSERVATION ===")
print(obs)

# 2. Take the first action
action = '{"name": "get_patient_info", "arguments": {"patient_id": "P001"}}'

print("\n=== ACTION ===")
print(action)

# 3. Environment transition
obs, reward, terminated, truncated, info = env.step(action)

print("\n=== AFTER ACTION ===")
print("Observation:")
print(obs)

print("\nReward:", reward)
print("Terminated:", terminated)
print("Truncated:", truncated)
print("Info:", info)