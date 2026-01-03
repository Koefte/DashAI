# DashAI PPO + RND

## Build and install
1. Ensure the Geode SDK path is exported to `GEODE_SDK` and run CMake as usual (build produces `koefte.dashai.geode`).
2. Install the produced Geode package into Geometry Dash.

## Runtime wiring
1. Launch Geometry Dash with the mod enabled. The mod opens a named pipe at `\\.\\pipe\\DashAI` and streams `state` lines.
2. Start the Python agent (requires Python 3.9+, `pip install torch numpy`).
   ```bash
   python dashai_agent.py
   ```
3. The agent connects to the pipe, reads `state attempt=... percent=... x=... y=... vy=... alive=...` lines, and sends back `action jump=0|1 hold=0|1` lines. `jump/hold=1` holds the jump button; `0` releases it.

## Hyperparameters
- Override rollout size: `DASHAI_ROLLOUT=1024 python dashai_agent.py`
- Override total training steps: `DASHAI_STEPS=50000 python dashai_agent.py`
- Enable/disable speedhack: `DASHAI_SPEEDHACK=1 python dashai_agent.py` (default: enabled)
- Set speed multiplier: `DASHAI_SPEED=5.0 python dashai_agent.py` (default: 3.0x)
  - `1.0` = normal speed
  - `2.0` = 2x speed
  - `3.0` = 3x speed (default for faster training)
  - `5.0` = 5x speed (very fast, may be unstable)
- Enable checkpoint spawning: `DASHAI_CHECKPOINTS=1 python dashai_agent.py` (default: disabled)
- Set checkpoint positions: `DASHAI_CHECKPOINT_POSITIONS="0,500,1000,1500,2000,2500" python dashai_agent.py`

## Checkpoint Spawning (Curriculum Learning)
The agent can spawn at random positions in the level instead of always starting from the beginning:
- **Benefits**: Faster learning, better exploration, practice difficult sections more frequently
- **Enable**: Set `DASHAI_CHECKPOINTS=1`
- **Configure positions**: Comma-separated X coordinates (e.g., `"0,500,1000,1500,2000,2500"`)
- The agent will randomly spawn at one of these X positions on each episode reset
- Rewards are adjusted to track progress from the spawn point

## Speedhack Support
The mod now includes built-in speedup functionality compatible with megahack-style speedhacks:
- The C++ mod uses `CCScheduler::setTimeScale()` to accelerate game time
- The Python agent automatically detects and adapts to the current game speed
- Speed can be dynamically adjusted by sending `speed=X.X` commands through the pipe
- Training rewards and timing are speed-aware to ensure consistent learning

### Examples
```bash
# Linux/Mac (bash):
# Train at 5x speed for ultra-fast training (50000 steps)
DASHAI_SPEED=5.0 DASHAI_STEPS=50000 python dashai_agent.py

# Train at normal speed (compatible with megahack disabled)
DASHAI_SPEEDHACK=0 python dashai_agent.py

# Custom rollout with 2x speed
DASHAI_SPEED=2.0 DASHAI_ROLLOUT=1024 python dashai_agent.py

# Enable checkpoint spawning for curriculum learning
DASHAI_CHECKPOINTS=1 python dashai_agent.py

# Checkpoint spawning with custom positions and 5x speed
DASHAI_CHECKPOINTS=1 DASHAI_CHECKPOINT_POSITIONS="0,1000,2000,3000" DASHAI_SPEED=5.0 python dashai_agent.py

# Windows (PowerShell):
# Train with checkpoint spawning enabled
$env:DASHAI_CHECKPOINTS="1"; python src/dashai_agent.py

# Multiple environment variables
$env:DASHAI_CHECKPOINTS="1"; $env:DASHAI_SPEED="5.0"; python src/dashai_agent.py

# Custom checkpoint positions
$env:DASHAI_CHECKPOINTS="1"; $env:DASHAI_CHECKPOINT_POSITIONS="0,1000,2000,3000"; python src/dashai_agent.py
```

## Protocol expectations
- Messages are ASCII newline-terminated.
- The mod throttles state messages to ~30 Hz (adjusts with speed multiplier).
- State format: `state attempt=N percent=X x=X y=X vy=X alive=0|1 ob_x=X ob_y=X ob_w=X ob_h=X speed=X`
- Action format: `action jump=0|1 hold=0|1`
- Speed control: `speed=X.X` (sets game speed multiplier, range: 0.1 to 10.0)
- Restart control: `restart` (resets the level to beginning)
- Spawn control: `spawn x=X.X` (teleports player to X position, used after restart for checkpoint training)
- If the pipe disconnects, the mod will wait for a new client; the Python client will keep retrying if the connection drops.

**Note on checkpoints**: Checkpoint spawning requires the level to be restarted first (`restart`), then the player is teleported (`spawn x=X`). This ensures proper level state initialization.
