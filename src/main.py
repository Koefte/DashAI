import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import win32pipe
import win32file
import pywintypes

PIPE_NAME = r"\\.\pipe\dashai"

# Normalization constants for gamedata features
# Based on typical Geometry Dash level dimensions
NORM_CONSTANTS = {
    'playerY': (100.0, 500.0),           # min, max for player Y position
    'velocityY': (-50.0, 50.0),          # velocity range
    'nearestObstacleX': (-500.0, 500.0), # relative distance to obstacle X
    'nearestObstacleY': (-500.0, 500.0), # relative distance to obstacle Y
    'obstacleHeight': (5.0, 100.0),      # obstacle height range
    'obstacleWidth': (5.0, 100.0),       # obstacle width range
    'isGrounded': (0.0, 1.0),            # boolean
    'isShip': (0.0, 1.0),                # boolean
    'reward': (0.0, 1.0)                 # reward range
}

# PPO Agent
class PPOAgent(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)

def parse_gamedata(data_str):
    # Expects CSV: 10 floats/ints (playerY, velocityY, nearestObstacleX, nearestObstacleY, obstacleHeight, obstacleWidth, isGrounded, isShip, reward, isDead)
    raw = np.array([float(x) for x in data_str.strip().split(",")], dtype=np.float32)
    
    # Normalize each feature to range [0, 1], except reward and isDead
    normalized = np.zeros_like(raw)
    norm_keys = list(NORM_CONSTANTS.keys())
    
    for i in range(len(raw) - 2):  # Exclude last two elements (reward, isDead)
        min_val, max_val = NORM_CONSTANTS[norm_keys[i]]
        # Clamp value to [min, max] range, then normalize to [0, 1]
        clamped = np.clip(raw[i], min_val, max_val)
        normalized[i] = (clamped - min_val) / (max_val - min_val)
    
    # Keep reward as-is without clamping or normalization
    normalized[-2] = raw[-2]  # reward
    normalized[-1] = raw[-1]  # isDead
    
    return normalized

def ppo_train(agent, memory, optimizer, epochs=3, gamma=0.99, eps_clip=0.2):
    states = torch.tensor(np.array([m[0] for m in memory]), dtype=torch.float32)
    actions = torch.tensor(np.array([m[1] for m in memory]), dtype=torch.int64)
    rewards = torch.tensor(np.array([m[2] for m in memory]), dtype=torch.float32)
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    for _ in range(epochs):
        probs, values = agent(states)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        advantage = returns - values.squeeze()
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("PPO training step complete.")

def main():
    state_dim = 8  # 8 state features (playerY, velocityY, nearestObstacleX, nearestObstacleY, obstacleHeight, obstacleWidth, isGrounded, isShip)
    # 2 actions: no-jump (0), jump (1) - simplified
    action_dim = 2
    agent = PPOAgent(state_dim, action_dim)
    optimizer = optim.Adam(agent.parameters(), lr=1e-3)  # Increased learning rate
    episode_memory = []  # Current episode
    all_episodes = []    # Store completed episodes
    episode_count = 0
    total_reward_this_episode = 0
    best_episode_reward = -float('inf')
    
    print(f"Waiting for C++ client on {PIPE_NAME}...")
    pipe = win32pipe.CreateNamedPipe(
        PIPE_NAME,
        win32pipe.PIPE_ACCESS_DUPLEX,
        win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_READMODE_BYTE | win32pipe.PIPE_WAIT,
        1, 4096, 4096, 0, None)
    win32pipe.ConnectNamedPipe(pipe, None)
    print("Client connected!")
    buf = b""
    
    # Episode tracking
    episode_frame_count = 0
    max_episode_frames = 3000  # ~50 seconds at 60fps
    last_is_dead = False
    
    while True:
        try:
            chunk = win32file.ReadFile(pipe, 1024)[1]
        except pywintypes.error:
            print("Pipe closed or error.")
            break
        if not chunk:
            break
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            decoded = line.decode(errors="ignore")
            if not decoded.strip():
                continue
            
            state_full = parse_gamedata(decoded)
            state = state_full[:8]  # First 8 features (exclude reward and isDead from state)
            reward = state_full[8]  # Reward is the 9th value
            is_dead = state_full[9] > 0.5  # isDead flag is the 10th value
            
            episode_frame_count += 1
            
            # Detect episode end: death or timeout
            episode_ended = False
            if is_dead and not last_is_dead:  # Just died
                episode_ended = True
            elif episode_frame_count >= max_episode_frames:
                episode_ended = True
            
            last_is_dead = is_dead
            
            # Episode ended
            if episode_ended and len(episode_memory) > 10:  # Need at least 10 frames
                all_episodes.append(episode_memory)
                episode_count += 1
                print(f"Episode {episode_count} ended. Reward: {total_reward_this_episode:.2f} (Best: {best_episode_reward:.2f}) - {episode_frame_count} frames")
                
                if total_reward_this_episode > best_episode_reward:
                    best_episode_reward = total_reward_this_episode
                
                episode_memory = []
                total_reward_this_episode = 0
                episode_frame_count = 0
                
                # Train after every 3 episodes (more frequent training)
                if len(all_episodes) >= 3:
                    flat_memory = []
                    for ep in all_episodes:
                        flat_memory.extend(ep)
                    print(f"Training on {len(all_episodes)} episodes ({len(flat_memory)} transitions)...")
                    ppo_train(agent, flat_memory, optimizer, epochs=5, gamma=0.99)
                    all_episodes.clear()
            
            total_reward_this_episode += reward
            
            # Choose action with epsilon-greedy exploration
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                probs, _ = agent(state_tensor)
                # Add exploration noise - slower decay
                epsilon = max(0.2, 1.0 - episode_count * 0.0005)  # Decay from 1.0 to 0.2
                if np.random.random() < epsilon:
                    action = np.random.randint(0, action_dim)
                else:
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample().item()
            
            # Send action back to C++
            win32file.WriteFile(pipe, f"{action}\n".encode("ascii"))
            episode_memory.append((state, action, reward))

if __name__ == "__main__":
    main()
