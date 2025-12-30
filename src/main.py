import argparse
import random
import time
from pathlib import Path

import win32pipe
import win32file
import pywintypes

PIPE_DEFAULT = r"\\.\pipe\dashai"


def send_status(handle, generation: int, indiv: int, best_score: float) -> bool:
    try:
        win32file.WriteFile(handle, f"STATUS,{generation},{indiv},{best_score:.2f}\n".encode("ascii"))
        return True
    except pywintypes.error:
        return False


def parse_obs(line: str):
    parts = line.strip().split(',')
    if len(parts) < 20 or parts[0] != 'OBS':
        return None, None
    obs_vals = list(map(float, parts[1:18]))
    done = float(parts[19]) > 0.5
    return obs_vals, done


def ensure_pipe(pipe_name: str):
    handle = win32pipe.CreateNamedPipe(
        pipe_name,
        win32pipe.PIPE_ACCESS_DUPLEX,
        win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_READMODE_BYTE | win32pipe.PIPE_WAIT,
        1,
        4096,
        4096,
        0,
        None,
    )
    print(f"[{time.time():.3f}] Waiting for client on {pipe_name}...", flush=True)
    win32pipe.ConnectNamedPipe(handle, None)
    print(f"[{time.time():.3f}] Client connected", flush=True)
    return handle


def mutate_actions(seq, mutate_rate):
    out = list(seq)
    for i in range(len(out)):
        if random.random() < mutate_rate:
            out[i] = 1 - out[i]
    return out


def crossover_actions(a, b):
    if not a or not b:
        return list(a)
    cut = random.randint(1, min(len(a), len(b)) - 1)
    return a[:cut] + b[cut:]


def run_episode(handle, action_seq, max_steps: int, echo: bool, print_every: int):
    buf = b""
    step = 0
    max_percent = 0.0
    actions = []
    last_print_step = -999
    n = len(action_seq)
    while step < max_steps:
        try:
            chunk = win32file.ReadFile(handle, 1024)[1]
        except pywintypes.error:
            return max_percent, True, actions
        if not chunk:
            return max_percent, True, actions
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            obs_vals, done = parse_obs(line.decode('utf-8', errors='ignore'))
            if obs_vals is None:
                continue
            max_percent = max(max_percent, float(obs_vals[5]))

            action = action_seq[step % n] if n else 0
            if echo and (step - last_print_step) >= print_every:
                print(f"step {step} action {action} percent {max_percent:.2f}", flush=True)
                last_print_step = step
            try:
                win32file.WriteFile(handle, f"{action}\n".encode('ascii'))
                win32file.FlushFileBuffers(handle)  # Force immediate send
            except pywintypes.error:
                return max_percent, True, actions
            actions.append(int(action))

            step += 1
            if done:
                return max_percent, False, actions
    return max_percent, False, actions


def evolve(args):
    random.seed(args.seed)
    pop = [[random.randint(0, 1) for _ in range(args.action_len)] for _ in range(args.pop_size)]
    best_score = -1e9
    generation = 0
    handle = ensure_pipe(args.pipe)

    try:
        while generation < args.generations:
            scores = []
            for i, genome in enumerate(pop):
                if not send_status(handle, generation, i, best_score if best_score > -1e8 else 0.0):
                    win32file.CloseHandle(handle)
                    handle = ensure_pipe(args.pipe)
                score, broken, actions = run_episode(
                    handle,
                    genome,
                    max_steps=args.max_steps,
                    echo=args.echo_actions,
                    print_every=args.print_every,
                )
                if broken:
                    win32file.CloseHandle(handle)
                    handle = ensure_pipe(args.pipe)
                    send_status(handle, generation, i, best_score if best_score > -1e8 else 0.0)
                    score, _, actions = run_episode(
                        handle,
                        genome,
                        max_steps=args.max_steps,
                        echo=args.echo_actions,
                        print_every=args.print_every,
                    )
                send_status(handle, generation, i, score)
                scores.append((score, genome))
                print(f"gen {generation} indiv {i} fitness {score:.2f} (actions taken: {len(actions)})", flush=True)

            scores.sort(key=lambda x: x[0], reverse=True)
            if scores[0][0] > best_score:
                best_score = scores[0][0]
                best_seq = list(scores[0][1])
                Path(args.checkpoint).write_text("\n".join(map(str, best_seq)))
                print(f"[best] gen {generation} score {best_score:.2f}", flush=True)
                send_status(handle, generation, -1, best_score)

            # Testing mode: carry forward only the current best genome unchanged across the whole population.
            best_genome = list(scores[0][1])
            # Mutate only the last portion (e.g., last 25%) of the action sequence
            cutoff = int(len(best_genome) * 0.75)
            new_pop = []
            for _ in range(args.pop_size):
                child = list(best_genome)
                for i in range(cutoff, len(child)):
                    if random.random() < args.mutate_rate:
                        child[i] = 1 - child[i]
                new_pop.append(child)
            pop = new_pop
            generation += 1
    finally:
        if handle is not None and handle != win32file.INVALID_HANDLE_VALUE:
            win32file.CloseHandle(handle)


def main():
    parser = argparse.ArgumentParser(description="Pure genetic search over action sequences")
    parser.add_argument("--pipe", default=PIPE_DEFAULT)
    parser.add_argument("--pop-size", type=int, default=8)
    parser.add_argument("--elite", type=int, default=2)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--action-len", type=int, default=512)
    parser.add_argument("--mutate-rate", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="best_actions.txt")
    parser.add_argument("--echo-actions", action="store_true", default=True, help="Print actions sent to the pipe (default on)")
    parser.add_argument("--no-echo", action="store_false", dest="echo_actions", help="Disable action printing")
    parser.add_argument("--print-every", type=int, default=60, help="Print once every N frames")
    args = parser.parse_args()
    evolve(args)


if __name__ == "__main__":
    main()
