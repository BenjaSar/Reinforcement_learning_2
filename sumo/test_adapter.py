"""Test SUMO adapter with custom observation function."""
import os, sys
sys.path.insert(0, r'F:\IA\RLII\tp_final')
os.environ['SUMO_HOME'] = r'C:\Users\PC\AppData\Local\Temp\opencode\sumo_bin\sumo-1.21.0'

import sumo_rl
from envs.sumo_adapter import SumoGridEnv

d = os.path.dirname(sumo_rl.__file__)
net_dir = os.path.join(d, 'nets', 'RESCO', 'grid4x4')

route_dir = r'F:\IA\RLII\tp_final\sumo\routes'
route_file = os.path.join(route_dir, 'grid4x4_03.rou.xml')  # our custom 0.3 demand

env = SumoGridEnv(
    net_file=os.path.join(net_dir, 'grid4x4.net.xml'),
    route_file=route_file,
    num_seconds=180,
    delta_time=5,
    use_gui=False,
)

print(f"Obs space: {env.observation_space}")
print(f"Action space: {env.action_space}")

obs, info = env.reset()
print(f"Obs shape: {obs.shape}")
print(f"Obs min/max: {obs.min():.4f}/{obs.max():.4f}")
print(f"First 18 values: {obs[:18]}")

total_r = 0.0
for s in range(10):
    action = env.action_space.sample()
    obs, r, term, trunc, info = env.step(action)
    total_r += r
    print(f"  step={s+1}, reward={r:.4f}, term={term}, trunc={trunc}")

print(f"Total reward after 10 steps: {total_r:.4f}")

# Verify obs matches 9-per-agent format
for i in range(16):
    o9 = obs[i*9:(i+1)*9]
    assert len(o9) == 9, f"Agent {i}: expected 9 obs, got {len(o9)}"
    assert abs(o9[4:8].sum() - 1.0) < 1e-5 or abs(o9[4:8].sum()) < 1e-5, \
        f"Agent {i}: phase one-hot sum = {o9[4:8].sum()}"

print("All agent obs valid!")
env.close()
