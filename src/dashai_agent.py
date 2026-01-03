import math
import os
import time
import ctypes
import msvcrt
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


PIPE_PATH = r"\\.\pipe\DashAI"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = os.getenv("DASHAI_DEBUG", "0") not in ("0", "", "false", "False")


class NamedPipeClient:
    def __init__(self, path: str = PIPE_PATH, retry: float = 1.0) -> None:
        self.path = path
        self.retry = retry
        self.pipe = None
        self._buf = b""

    def connect(self) -> None:
        while self.pipe is None:
            try:
                self.pipe = open(self.path, "r+b", buffering=0)
                try:
                    # Switch to message-read mode to align with server PIPE_TYPE_MESSAGE
                    handle = msvcrt.get_osfhandle(self.pipe.fileno())
                    new_mode = ctypes.c_uint(0x00000002)  # PIPE_READMODE_MESSAGE
                    res = ctypes.windll.kernel32.SetNamedPipeHandleState(int(handle), ctypes.byref(new_mode), None, None)
                    if res == 0 and DEBUG:
                        print("[dashai] failed to set PIPE_READMODE_MESSAGE")
                except Exception as exc:
                    if DEBUG:
                        print(f"[dashai] SetNamedPipeHandleState error: {exc}")
                print(f"[dashai] connected to {self.path}")
            except OSError:
                print(f"[dashai] waiting for pipe {self.path}")
                time.sleep(self.retry)

    def readline(self) -> str:
        if self.pipe is None:
            return ""
        try:
            handle = msvcrt.get_osfhandle(self.pipe.fileno())
            avail = ctypes.wintypes.DWORD()
            ok = ctypes.windll.kernel32.PeekNamedPipe(int(handle), None, 0, None, ctypes.byref(avail), None)
            if ok == 0:
                return ""
            if avail.value == 0:
                return ""
            chunk = os.read(self.pipe.fileno(), avail.value)
            if not chunk:
                return ""
            self._buf += chunk
            if b"\n" not in self._buf:
                return ""
            line, _, rest = self._buf.partition(b"\n")
            self._buf = rest
            if DEBUG:
                print(f"[dashai] recv: {line!r}", flush=True)
            return line.decode("ascii", errors="ignore").strip()
        except Exception as exc:
            if DEBUG:
                print(f"[dashai] readline error: {exc}", flush=True)
            return ""

    def send_action(self, jump: bool, hold: bool) -> None:
        if self.pipe is None:
            return
        msg = f"action jump={int(jump)} hold={int(hold)}\n".encode("ascii")
        try:
            self.pipe.write(msg)
            self.pipe.flush()
            if DEBUG:
                print(f"[dashai] send: {msg!r}")
        except OSError:
            self.pipe = None


@dataclass
class Transition:
    obs: np.ndarray
    action: float
    logprob: float
    reward: float
    intrinsic: float
    done: float
    value: float


class RolloutBuffer:
    def __init__(self, capacity: int, obs_dim: int) -> None:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.clear()

    def add(self, transition: Transition) -> None:
        self.observations.append(transition.obs)
        self.actions.append([transition.action])
        self.logprobs.append([transition.logprob])
        self.rewards.append([transition.reward])
        self.intrinsic.append([transition.intrinsic])
        self.dones.append([transition.done])
        self.values.append([transition.value])

    def ready(self) -> bool:
        return len(self.observations) >= self.capacity

    def clear(self) -> None:
        self.observations: List[np.ndarray] = []
        self.actions: List[List[float]] = []
        self.logprobs: List[List[float]] = []
        self.rewards: List[List[float]] = []
        self.intrinsic: List[List[float]] = []
        self.dones: List[List[float]] = []
        self.values: List[List[float]] = []

    def to_tensors(self) -> Tuple[torch.Tensor, ...]:
        obs_np = np.asarray(self.observations, dtype=np.float32)
        actions_np = np.asarray(self.actions, dtype=np.float32)
        logprobs_np = np.asarray(self.logprobs, dtype=np.float32)
        rewards_np = np.asarray(self.rewards, dtype=np.float32)
        intrinsic_np = np.asarray(self.intrinsic, dtype=np.float32)
        dones_np = np.asarray(self.dones, dtype=np.float32)
        padded_values = self.values + [[0.0]]
        values_np = np.asarray(padded_values, dtype=np.float32)

        obs = torch.from_numpy(obs_np).to(DEVICE)
        actions = torch.from_numpy(actions_np).to(DEVICE)
        logprobs = torch.from_numpy(logprobs_np).to(DEVICE)
        rewards = torch.from_numpy(rewards_np).to(DEVICE)
        intrinsic = torch.from_numpy(intrinsic_np).to(DEVICE)
        dones = torch.from_numpy(dones_np).to(DEVICE)
        values = torch.from_numpy(values_np).to(DEVICE)
        return obs, actions, logprobs, rewards, intrinsic, dones, values


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.policy = nn.Linear(128, 1)
        self.value = nn.Linear(128, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.feature(obs)
        return self.policy(x), self.value(x)


class RNDModel(nn.Module):
    def __init__(self, obs_dim: int, output_dim: int = 64) -> None:
        super().__init__()
        self.target = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )
        for p in self.target.parameters():
            p.requires_grad = False
        self.predictor = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            target = self.target(obs)
        pred = self.predictor(obs)
        return pred, target


class PPOAgent:
    def __init__(self, obs_dim: int, rollout: int = 2048) -> None:
        self.model = ActorCritic(obs_dim).to(DEVICE)
        self.rnd = RNDModel(obs_dim).to(DEVICE)
        self.opt_policy = optim.Adam(self.model.parameters(), lr=3e-4)
        self.opt_rnd = optim.Adam(self.rnd.predictor.parameters(), lr=3e-4)
        self.buffer = RolloutBuffer(rollout, obs_dim)
        self.gamma = 0.99
        self.lam = 0.95
        self.clip = 0.2
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.int_coef = 0.5
        self.episode_return = 0.0
        self.episode_count = 0

    def act(self, obs: np.ndarray) -> Tuple[int, float, float, float]:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        logits, value = self.model(obs_t)
        dist = torch.distributions.Bernoulli(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        with torch.no_grad():
            pred, target = self.rnd(obs_t)
            intrinsic = torch.mean((pred - target) ** 2, dim=1)
        return int(action.item()), float(logprob.item()), float(value.item()), float(intrinsic.item())

    def update(self) -> None:
        obs, actions, logprobs, rewards, intrinsic, dones, values = self.buffer.to_tensors()
        adv = torch.zeros_like(rewards)
        gae = 0.0
        with torch.no_grad():
            for t in reversed(range(len(rewards))):
                mask = 1.0 - dones[t]
                delta = rewards[t] + self.int_coef * intrinsic[t] + self.gamma * values[t + 1] * mask - values[t]
                gae = delta + self.gamma * self.lam * mask * gae
                adv[t] = gae
        returns = adv + values[:-1]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(4):
            logits, value_pred = self.model(obs)
            dist = torch.distributions.Bernoulli(logits=logits)
            new_logprob = dist.log_prob(actions)
            ratio = torch.exp(new_logprob - logprobs)
            clip_adv = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv
            policy_loss = -(torch.min(ratio * adv, clip_adv)).mean()
            value_loss = nn.functional.mse_loss(value_pred.squeeze(-1), returns.squeeze(-1))
            entropy = dist.entropy().mean()
            self.opt_policy.zero_grad()
            (policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy).backward()
            self.opt_policy.step()

            pred, target = self.rnd(obs)
            rnd_loss = ((pred - target) ** 2).mean()
            self.opt_rnd.zero_grad()
            rnd_loss.backward()
            self.opt_rnd.step()

        self.buffer.clear()

    def store(self, transition: Transition) -> None:
        self.buffer.add(transition)


class DashEnv:
    def __init__(self, client: NamedPipeClient) -> None:
        self.client = client
        self.prev_percent = 0.0
        self.prev_alive = True
        self.last_log = 0.0
        self.prev_obstacle_x = None
        self.speed_multiplier = 1.0

    def parse_state(self, line: str) -> dict:
        parts = line.strip().split()
        data = {}
        for part in parts[1:]:
            if "=" not in part:
                continue
            key, val = part.split("=", 1)
            if key == "alive":
                data[key] = val == "1"
            else:
                try:
                    data[key] = float(val)
                except ValueError:
                    data[key] = 0.0
        # Extract speed multiplier if available
        self.speed_multiplier = data.get("speed", 1.0)
        return data

    def next_state(self) -> dict:
        deadline = time.time() + 1.0
        line = self.client.readline()
        while (not line or not line.startswith("state")) and time.time() < deadline:
            time.sleep(0.01)
            line = self.client.readline()
        if DEBUG and not line:
            print("[dashai] waiting for state...", flush=True)
        return self.parse_state(line) if line else {}

    def reward(self, state: dict) -> Tuple[float, bool]:
        percent = state.get("percent", 0.0)
        alive = state.get("alive", True)
        player_x = state.get("x", 0.0)
        
        # Increase progress reward scaling significantly
        delta = percent - self.prev_percent
        reward = delta * 10.0  # 10x increase: 0% to 100% = 1000 reward
        done = False
        
        if self.prev_alive and not alive:
            reward -= 5.0  # Proportional death penalty
            done = True

        # Reward for passing obstacles - track player position vs obstacle position
        ob_x = state.get("ob_x", -1.0)
        if ob_x >= 0 and self.prev_obstacle_x is not None and self.prev_obstacle_x >= 0:
            # If the obstacle that was ahead is now behind the player, we passed it
            if ob_x != self.prev_obstacle_x and player_x > self.prev_obstacle_x:
                reward += 2.0  # Significant obstacle passing bonus
                if DEBUG:
                    print(f"[dashai] obstacle passed! player_x={player_x:.1f} ob_x={self.prev_obstacle_x:.1f} +2.0", flush=True)
        
        # Update tracking
        self.prev_obstacle_x = ob_x if ob_x >= 0 else self.prev_obstacle_x

        self.prev_percent = 0.0 if done else percent
        self.prev_alive = True if done else alive
        return reward, done

    def obs_from_state(self, state: dict) -> np.ndarray:
        ox = state.get("ob_x", -1.0)
        oy = state.get("ob_y", -1.0)
        ow = state.get("ob_w", -1.0)
        oh = state.get("ob_h", -1.0)
        has_obstacle = 1.0 if ox >= 0 else 0.0
        return np.array(
            [
                state.get("percent", 0.0) / 100.0,
                state.get("x", 0.0) / 3000.0,
                state.get("y", 0.0) / 500.0,
                state.get("vy", 0.0) / 50.0,
                1.0 if state.get("alive", True) else 0.0,
                has_obstacle,
                ox / 3000.0 if ox >= 0 else 0.0,
                oy / 500.0 if oy >= 0 else 0.0,
                ow / 200.0 if ow >= 0 else 0.0,
                oh / 200.0 if oh >= 0 else 0.0,
            ],
            dtype=np.float32,
        )

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        jump = bool(action)
        self.client.send_action(jump, jump)
        state = self.next_state()
        obs = self.obs_from_state(state)
        reward, done = self.reward(state)
        if DEBUG:
            now = time.time()
            if now - self.last_log > 1.0:
                self.last_log = now
                print(
                    "[dashai] obs=",
                    {
                        "pct": round(state.get("percent", 0.0), 2),
                        "x": round(state.get("x", 0.0), 2),
                        "y": round(state.get("y", 0.0), 2),
                        "vy": round(state.get("vy", 0.0), 3),
                        "alive": state.get("alive", True),
                        "ob": (
                            state.get("ob_x", -1.0),
                            state.get("ob_y", -1.0),
                            state.get("ob_w", -1.0),
                            state.get("ob_h", -1.0),
                        ),
                        "a": int(jump),
                        "r": round(reward, 3),
                        "done": done,
                    },
                )
        return obs, reward, done, state

    def reset(self) -> np.ndarray:
        self.prev_percent = 0.0
        self.prev_alive = True
        state = self.next_state()
        return self.obs_from_state(state)

    def set_speed(self, multiplier: float) -> None:
        """Set the game speed multiplier (1.0 = normal, 2.0 = 2x speed, etc.)"""
        msg = f"speed={multiplier}\n".encode("ascii")
        try:
            if self.client.pipe is not None:
                self.client.pipe.write(msg)
                self.client.pipe.flush()
                if DEBUG:
                    print(f"[dashai] set speed to {multiplier}x", flush=True)
        except OSError:
            if DEBUG:
                print(f"[dashai] failed to set speed", flush=True)


def train_loop(total_steps: int = 10000, rollout: int = 128, use_speedhack: bool = True, base_speed: float = 3.0) -> None:
    """Train the agent with optional speedhack support.
    
    Args:
        total_steps: Total training steps to run
        rollout: Number of steps before policy update (default reduced to 128 for faster feedback)
        use_speedhack: Whether to use game speedup (requires compatible version)
        base_speed: Speed multiplier to use (1.0=normal, 2.0=2x, 3.0=3x, etc.)
    """
    client = NamedPipeClient()
    client.connect()
    env = DashEnv(client)
    agent = PPOAgent(obs_dim=10, rollout=rollout)  # Updated to 10 for has_obstacle flag
    
    # Set initial speed if speedhack is enabled
    if use_speedhack:
        env.set_speed(base_speed)
        print(f"[dashai] Training with {base_speed}x speedhack enabled")
    else:
        env.set_speed(1.0)
        print(f"[dashai] Training at normal speed (1.0x)")
    
    obs = env.reset()
    step = 0
    episodes_completed = 0
    episode_return = 0.0
    episode_rewards = []
    best_return = float('-inf')
    
    while step < total_steps:
        action, logprob, value, intrinsic = agent.act(obs)
        next_obs, reward, done, state = env.step(action)
        
        episode_return += reward
        
        transition = Transition(
            obs=obs,
            action=float(action),
            logprob=logprob,
            reward=reward,
            intrinsic=intrinsic,
            done=1.0 if done else 0.0,
            value=value,
        )
        agent.store(transition)
        step += 1
        
        if done:
            episodes_completed += 1
            episode_rewards.append(episode_return)
            if episode_return > best_return:
                best_return = episode_return
            
            # Log episode stats
            recent_avg = sum(episode_rewards[-10:]) / min(10, len(episode_rewards))
            percent = state.get("percent", 0.0)
            print(f"[dashai] Episode {episodes_completed}: return={episode_return:.2f}, percent={percent:.1f}%, best={best_return:.2f}, avg10={recent_avg:.2f}")
            
            episode_return = 0.0
            obs = env.reset()
        else:
            obs = next_obs
            
        if agent.buffer.ready():
            agent.update()
            current_speed = env.speed_multiplier
            avg_reward = sum(episode_rewards[-10:]) / min(10, len(episode_rewards)) if episode_rewards else 0.0
            print(f"[dashai] Policy update at step {step} | Episodes: {episodes_completed} | Avg reward (last 10): {avg_reward:.2f} | Speed: {current_speed:.1f}x")
    
    print(f"[dashai] Training complete! Episodes: {episodes_completed} | Best return: {best_return:.2f} | Final avg: {sum(episode_rewards[-10:])/min(10, len(episode_rewards)) if episode_rewards else 0.0:.2f}")


if __name__ == "__main__":
    rollout = int(os.getenv("DASHAI_ROLLOUT", "128"))  # Reduced default for faster feedback
    steps = int(os.getenv("DASHAI_STEPS", "20000"))
    use_speedhack = os.getenv("DASHAI_SPEEDHACK", "1") not in ("0", "false", "False")
    speed_multiplier = float(os.getenv("DASHAI_SPEED", "3.0"))
    
    print(f"[dashai] Starting training with rollout={rollout}, steps={steps}, speedhack={use_speedhack}, speed={speed_multiplier}x")
    print(f"[dashai] Device: {DEVICE}")
    
    train_loop(total_steps=steps, rollout=rollout, use_speedhack=use_speedhack, base_speed=speed_multiplier)
