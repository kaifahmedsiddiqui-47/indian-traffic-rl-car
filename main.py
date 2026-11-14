"""Train Intelli-Car through four driving scenarios, then test it on all four.

Scenarios (see config.py):
    highway        3 lanes each way
    heavy_traffic  3 lanes each way, dense and slow
    single_lane    1 lane each way (no overtaking possible)
    u_turn         2 lanes each way with a U-turn opening in the median

The agent trains on each scenario in turn (each with its own number of episodes),
keeping the same network and replay memory, and is then tested on all four
scenarios with greedy actions. The double yellow line may never be crossed (no
overtaking in any condition); the only exception is the U-turn opening.

Usage
-----
    python main.py                          # train all four scenarios, then final test
    python main.py --no-render              # same, without drawing (much faster)
    python main.py --fast                   # draw only every few frames
    python main.py --episodes 100           # 100 episodes per scenario
    python main.py --stage u_turn           # train only one scenario (then final test)
    python main.py --load checkpoints/dqn_last.pth --stage u_turn
    python main.py --eval checkpoints/dqn_final.pth                   # final test only
    python main.py --eval checkpoints/dqn_final.pth --scenario u_turn # test one scenario

Keys while running: ESC quit, R toggle rendering, F toggle fast mode.
"""

import argparse
import csv
import json
import os
import random
import time

import numpy as np
import pygame
import torch

import config
from agent.dqn_agent import DQNAgent
from env.traffic_env import TrafficEnv

TRAIN_LOG_PATH = os.path.join(config.RESULTS_DIR, "training_log.csv")
TEST_CSV_PATH = os.path.join(config.RESULTS_DIR, "final_test.csv")
TEST_JSON_PATH = os.path.join(config.RESULTS_DIR, "final_test.json")

TRAIN_LOG_FIELDS = [
    "stage", "stage_episode", "episode", "scenario", "reward", "steps", "outcome",
    "epsilon", "loss", "brake_actions", "auto_brakes", "violations", "u_turn_done", "avg_speed",
]


def parse_args():
    choices = list(config.SCENARIOS.keys())
    parser = argparse.ArgumentParser(description="Intelli-Car - DQN training on four Indian traffic scenarios")
    parser.add_argument("--episodes", type=int, default=None, help="episodes per scenario (overrides config)")
    parser.add_argument("--stage", type=str, default=None, choices=choices + ["mixed"],
                        help="train only this scenario")
    parser.add_argument("--load", type=str, default=None, help="checkpoint to continue training from")
    parser.add_argument("--eval", type=str, default=None, help="checkpoint to test (no training)")
    parser.add_argument("--scenario", type=str, default=None, choices=choices,
                        help="with --eval: test only this scenario")
    parser.add_argument("--test-episodes", type=int, default=None, help="test episodes per scenario")
    parser.add_argument("--no-test", action="store_true", help="skip the final test after training")
    parser.add_argument("--no-render", action="store_true", help="disable drawing")
    parser.add_argument("--fast", action="store_true", help="uncapped speed, draw every few frames")
    return parser.parse_args()


def handle_events(flags):
    """Process pygame events and update the shared control flags."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            flags["running"] = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                flags["running"] = False
            elif event.key == pygame.K_r:
                flags["render"] = not flags["render"]
                print("Rendering {}".format("ON" if flags["render"] else "OFF"))
            elif event.key == pygame.K_f:
                flags["fast"] = not flags["fast"]
                print("Fast mode {}".format("ON" if flags["fast"] else "OFF"))


def should_draw(flags, step):
    if not flags["render"]:
        return False
    if flags["fast"]:
        return step % config.FAST_RENDER_EVERY == 0
    return True


# ---------------------------------------------------------------------------
# One episode (used for both training and testing)
# ---------------------------------------------------------------------------
def run_episode(env, agent, scenario, flags, clock, training, hud_prefix):
    """Play one episode. The episode ends when the environment says so: the car reached the
    finish line, crashed, broke the double-line rule, missed the U-turn, or the (generous)
    time limit ran out."""
    state = env.reset(scenario)
    total_reward = 0.0
    losses = []
    overrides = 0
    brake_actions = 0
    speed_sum = 0.0
    step = 0
    info = {"outcome": "running", "violations": 0, "u_turn_done": False}

    while True:
        handle_events(flags)
        if not flags["running"]:
            return {"aborted": True}

        step += 1
        action = agent.select_action(state, training=training)
        if action in (config.ACTION_SLOW, config.ACTION_STOP):
            brake_actions += 1

        next_state, reward, done, info = env.step(action)
        if info["override"]:
            overrides += 1
        speed_sum += info["speed"]

        if training:
            # the agent's own choice is stored; the shield penalty already sits in `reward`
            agent.remember(state, action, reward, next_state, info["terminal"])
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)

        state = next_state
        total_reward += reward

        if should_draw(flags, step):
            hud = dict(hud_prefix)
            hud["Reward"] = "{:.1f}".format(total_reward)
            hud["Auto-brakes"] = str(overrides)
            env.render(hud)
            if not flags["fast"]:
                clock.tick(config.FPS)

        if done:
            break

    return {
        "aborted": False,
        "scenario": scenario,
        "reward": total_reward,
        "steps": env.steps,
        "outcome": info["outcome"],
        "loss": float(np.mean(losses)) if losses else 0.0,
        "brake_actions": brake_actions,
        "auto_brakes": overrides,
        "violations": info["violations"],
        "u_turn_done": bool(info["u_turn_done"]),
        "avg_speed": speed_sum / max(1, step),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def build_stages(args):
    """List of (stage name, episodes) to train, in order."""
    if args.stage is not None:
        if args.episodes is not None:
            episodes = args.episodes
        elif args.stage == "mixed":
            episodes = config.MIXED_EPISODES if config.MIXED_EPISODES > 0 else config.EPISODES
        else:
            episodes = config.EPISODES_PER_SCENARIO[args.stage]
        return [(args.stage, episodes)]

    stages = []
    for name in config.SCENARIO_ORDER:
        episodes = args.episodes if args.episodes is not None else config.EPISODES_PER_SCENARIO[name]
        stages.append((name, episodes))
    if config.MIXED_EPISODES > 0:
        stages.append(("mixed", args.episodes if args.episodes is not None else config.MIXED_EPISODES))
    return stages


def train_stage(env, agent, clock, flags, stage_name, episodes, stage_index, progress, log_writer, log_file):
    """Train on one scenario. Returns False if the user quit."""
    if stage_index > 0:
        agent.epsilon = max(agent.epsilon, config.STAGE_EPSILON_RESTART)

    print("")
    print("=" * 100)
    print("Stage {}: {} - {} episodes (epsilon starts at {:.3f})".format(
        stage_index + 1, stage_name, episodes, agent.epsilon))
    print("=" * 100)

    best_reward = float("-inf")
    recent_rewards = []
    stage_start = time.time()

    for episode in range(1, episodes + 1):
        scenario = random.choice(config.SCENARIO_ORDER) if stage_name == "mixed" else stage_name
        hud_prefix = {
            "Stage": "{} {}/{}".format(stage_name, episode, episodes),
            "Epsilon": "{:.3f}".format(agent.epsilon),
            "Best": "{:.1f}".format(best_reward) if best_reward > float("-inf") else "-",
        }

        result = run_episode(env, agent, scenario, flags, clock, True, hud_prefix)
        if result["aborted"]:
            return False

        progress["episode"] += 1
        agent.decay_epsilon()
        if progress["episode"] % config.TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        recent_rewards.append(result["reward"])
        average = float(np.mean(recent_rewards[-20:]))
        best_reward = max(best_reward, result["reward"])

        print(
            "{} {:3d}/{} | Reward: {:9.1f} | Avg20: {:8.1f} | Eps: {:.3f} | Steps: {:5d} | "
            "Brakes: {:4d} | AutoBrakes: {:4d} | Result: {}".format(
                stage_name, episode, episodes, result["reward"], average, agent.epsilon,
                result["steps"], result["brake_actions"], result["auto_brakes"], result["outcome"],
            )
        )

        log_writer.writerow({
            "stage": stage_name,
            "stage_episode": episode,
            "episode": progress["episode"],
            "scenario": scenario,
            "reward": round(result["reward"], 2),
            "steps": result["steps"],
            "outcome": result["outcome"],
            "epsilon": round(agent.epsilon, 4),
            "loss": round(result["loss"], 5),
            "brake_actions": result["brake_actions"],
            "auto_brakes": result["auto_brakes"],
            "violations": result["violations"],
            "u_turn_done": int(result["u_turn_done"]),
            "avg_speed": round(result["avg_speed"], 3),
        })
        log_file.flush()

        if progress["episode"] % config.SAVE_EVERY_EPISODES == 0:
            agent.save(config.LAST_MODEL_PATH, episode=progress["episode"], best_reward=best_reward)

    stage_path = os.path.join(config.CHECKPOINT_DIR, "dqn_after_{}.pth".format(stage_name))
    agent.save(stage_path, episode=progress["episode"], best_reward=best_reward)
    agent.save(config.LAST_MODEL_PATH, episode=progress["episode"], best_reward=best_reward)
    print("Stage '{}' finished in {:.0f}s. Best episode reward: {:.1f}. Saved {}".format(
        stage_name, time.time() - stage_start, best_reward, stage_path))
    return True


# ---------------------------------------------------------------------------
# Final test on all four scenarios
# ---------------------------------------------------------------------------
def summarize(scenario, results):
    count = len(results)

    def share(outcomes):
        return 100.0 * sum(1 for r in results if r["outcome"] in outcomes) / count

    summary = {
        "scenario": scenario,
        "episodes": count,
        "success_pct": share(("destination",)),
        "collision_pct": share(("collision",)),
        "double_line_pct": share(("double_line",)),
        "missed_turn_pct": share(("missed_turn",)),
        "timeout_pct": share(("stuck", "timeout")),
        "avg_reward": float(np.mean([r["reward"] for r in results])),
        "avg_steps": float(np.mean([r["steps"] for r in results])),
        "avg_speed": float(np.mean([r["avg_speed"] for r in results])),
        "avg_auto_brakes": float(np.mean([r["auto_brakes"] for r in results])),
        "u_turn_done_pct": (
            100.0 * sum(1 for r in results if r["u_turn_done"]) / count
            if config.SCENARIOS[scenario]["u_turn"] else None
        ),
    }
    return summary


def print_table(summaries):
    header = "{:<14} {:>5} {:>8} {:>10} {:>9} {:>10} {:>9} {:>10} {:>9} {:>8} {:>11}".format(
        "Scenario", "Eps", "Success%", "Collision%", "LineCross%", "MissedTurn%", "Timeout%",
        "AvgReward", "AvgSteps", "AvgSpeed", "AutoBrk/ep")
    print("")
    print("FINAL TEST RESULTS (greedy agent, no exploration, no learning)")
    print(header)
    print("-" * len(header))
    for s in summaries:
        print("{:<14} {:>5d} {:>8.1f} {:>10.1f} {:>9.1f} {:>10.1f} {:>9.1f} {:>10.1f} {:>9.0f} {:>8.2f} {:>11.1f}".format(
            s["scenario"], s["episodes"], s["success_pct"], s["collision_pct"], s["double_line_pct"],
            s["missed_turn_pct"], s["timeout_pct"], s["avg_reward"], s["avg_steps"], s["avg_speed"],
            s["avg_auto_brakes"]))
    for s in summaries:
        if s["u_turn_done_pct"] is not None:
            print("U-turn completed in {:.1f}% of '{}' test episodes.".format(s["u_turn_done_pct"], s["scenario"]))


def save_test_results(summaries, model_description, episodes):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    fields = [
        "scenario", "episodes", "success_pct", "collision_pct", "double_line_pct", "missed_turn_pct",
        "timeout_pct", "avg_reward", "avg_steps", "avg_speed", "avg_auto_brakes", "u_turn_done_pct",
    ]
    with open(TEST_CSV_PATH, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for s in summaries:
            row = {key: ("" if s[key] is None else (round(s[key], 3) if isinstance(s[key], float) else s[key]))
                   for key in fields}
            writer.writerow(row)
    with open(TEST_JSON_PATH, "w") as handle:
        json.dump({
            "model": model_description,
            "test_episodes_per_scenario": episodes,
            "test_seed": config.TEST_SEED,
            "results": summaries,
        }, handle, indent=2)
    print("Saved {} and {}".format(TEST_CSV_PATH, TEST_JSON_PATH))


def final_test(env, agent, clock, flags, scenarios, episodes, model_description):
    """Greedy test on the given scenarios. Test episode i always uses seed TEST_SEED + i,
    so every model faces exactly the same traffic."""
    print("")
    print("=" * 100)
    print("Final test: {} episodes on each of: {}".format(episodes, ", ".join(scenarios)))
    print("=" * 100)

    summaries = []
    for scenario in scenarios:
        results = []
        for i in range(episodes):
            seed = config.TEST_SEED + i
            random.seed(seed)
            np.random.seed(seed)
            hud_prefix = {"Mode": "FINAL TEST", "Test": "{} {}/{}".format(scenario, i + 1, episodes)}
            result = run_episode(env, agent, scenario, flags, clock, False, hud_prefix)
            if result["aborted"]:
                return None
            results.append(result)
            print("test {:<14} {:3d}/{} | Reward: {:9.1f} | Steps: {:5d} | AutoBrakes: {:4d} | Result: {}".format(
                scenario, i + 1, episodes, result["reward"], result["steps"], result["auto_brakes"],
                result["outcome"]))
        summaries.append(summarize(scenario, results))

    print_table(summaries)
    save_test_results(summaries, model_description, episodes)
    return summaries


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    pygame.init()
    pygame.display.set_caption("Intelli-Car - DQN, four scenarios")
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    first_scenario = args.scenario or config.SCENARIO_ORDER[0]
    env = TrafficEnv(screen=screen, scenario=first_scenario)
    agent = DQNAgent(config.STATE_SIZE, config.ACTION_SIZE)
    print("Device: {}".format(agent.device))

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    flags = {
        "running": True,
        "render": config.RENDER_TRAINING and not args.no_render,
        "fast": args.fast,
    }
    test_episodes = args.test_episodes if args.test_episodes is not None else config.FINAL_TEST_EPISODES

    try:
        if args.eval:
            agent.load(args.eval)
            agent.epsilon = 0.0
            print("Loaded {} - testing the trained agent".format(args.eval))
            scenarios = [args.scenario] if args.scenario else list(config.SCENARIO_ORDER)
            final_test(env, agent, clock, flags, scenarios, test_episodes, args.eval)
        else:
            if args.load:
                checkpoint = agent.load(args.load)
                print("Continuing from {} (epsilon {:.3f})".format(args.load, checkpoint.get("epsilon", agent.epsilon)))

            stages = build_stages(args)
            print("Training plan: " + ", ".join("{} x{}".format(name, count) for name, count in stages))

            progress = {"episode": 0}
            completed = True
            with open(TRAIN_LOG_PATH, "w", newline="") as log_file:
                log_writer = csv.DictWriter(log_file, fieldnames=TRAIN_LOG_FIELDS)
                log_writer.writeheader()
                for index, (name, count) in enumerate(stages):
                    if not train_stage(env, agent, clock, flags, name, count, index, progress,
                                       log_writer, log_file):
                        completed = False
                        break

            if completed:
                agent.save(config.FINAL_MODEL_PATH, episode=progress["episode"])
                print("")
                print("Training finished. Final model saved to {}".format(config.FINAL_MODEL_PATH))
                print("Training log saved to {}".format(TRAIN_LOG_PATH))
                if not args.no_test and flags["running"]:
                    agent.epsilon = 0.0
                    final_test(env, agent, clock, flags, list(config.SCENARIO_ORDER), test_episodes,
                               config.FINAL_MODEL_PATH)
            else:
                print("Stopped early. Latest model: {}".format(config.LAST_MODEL_PATH))
    except KeyboardInterrupt:
        print("Interrupted.")
        agent.save(config.LAST_MODEL_PATH)
        print("Saved {}".format(config.LAST_MODEL_PATH))

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()