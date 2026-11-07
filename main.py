"""Train (or watch) a DQN agent driving through Indian-style 6-lane traffic.

Usage
-----
    python main.py                       # train with live rendering
    python main.py --no-render           # train without drawing (much faster)
    python main.py --fast                # train, draw only every few frames
    python main.py --eval checkpoints/dqn_best.pth    # watch a trained agent

Keys while running: ESC quit, R toggle rendering, F toggle fast mode.
"""

import argparse
import os
import random

import numpy as np
import pygame
import torch

import config
from agent.dqn_agent import DQNAgent
from env.traffic_env import TrafficEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Indian Traffic Intelligent Car - DQN")
    parser.add_argument("--episodes", type=int, default=config.EPISODES, help="number of episodes")
    parser.add_argument("--load", type=str, default=None, help="checkpoint to continue training from")
    parser.add_argument("--eval", type=str, default=None, help="checkpoint to watch (no training)")
    parser.add_argument("--no-render", action="store_true", help="disable drawing during training")
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


def train(env, agent, clock, flags, episodes):
    best_reward = float("-inf")
    recent_rewards = []
    last_episode = 0

    for episode in range(1, episodes + 1):
        last_episode = episode
        state = env.reset()
        total_reward = 0.0
        losses = []
        overrides = 0
        brake_actions = 0
        info = {"outcome": "running"}

        for step in range(1, config.MAX_STEPS + 1):
            handle_events(flags)
            if not flags["running"]:
                break

            action = agent.select_action(state, training=True)
            if action in (config.ACTION_SLOW, config.ACTION_STOP):
                brake_actions += 1
            next_state, reward, done, info = env.step(action)
            if info["override"]:
                overrides += 1

            # the agent's own choice is stored; the shield penalty already sits in `reward`
            agent.remember(state, action, reward, next_state, info["terminal"])
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)

            state = next_state
            total_reward += reward

            if should_draw(flags, step):
                hud = {
                    "Episode": "{}/{}".format(episode, episodes),
                    "Reward": "{:.1f}".format(total_reward),
                    "Epsilon": "{:.3f}".format(agent.epsilon),
                    "Best": "{:.1f}".format(best_reward) if best_reward > float("-inf") else "-",
                    "Auto-brakes": str(overrides),
                }
                env.render(hud)
                if not flags["fast"]:
                    clock.tick(config.FPS)

            if done:
                break

        if not flags["running"]:
            break

        agent.decay_epsilon()
        if episode % config.TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()

        recent_rewards.append(total_reward)
        average = float(np.mean(recent_rewards[-20:]))
        average_loss = float(np.mean(losses)) if losses else 0.0

        print(
            "Episode {:3d}/{} | Reward: {:9.1f} | Avg20: {:8.1f} | Epsilon: {:.3f} | "
            "Steps: {:3d} | Loss: {:.4f} | Brakes: {:3d} | AutoBrakes: {:3d} | Result: {}".format(
                episode, episodes, total_reward, average, agent.epsilon,
                env.steps, average_loss, brake_actions, overrides, info["outcome"],
            )
        )

        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(config.BEST_MODEL_PATH, episode=episode, best_reward=best_reward)

        if episode % config.SAVE_EVERY_EPISODES == 0:
            agent.save(config.LAST_MODEL_PATH, episode=episode, best_reward=best_reward)

    agent.save(config.LAST_MODEL_PATH, episode=last_episode, best_reward=best_reward)
    print("Training finished. Best episode reward: {:.1f}".format(best_reward))
    print("Models saved in {}".format(config.CHECKPOINT_DIR))


def evaluate(env, agent, clock, flags, episodes):
    for episode in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        overrides = 0
        info = {"outcome": "running"}

        for step in range(1, config.MAX_STEPS + 1):
            handle_events(flags)
            if not flags["running"]:
                return

            action = agent.select_action(state, training=False)
            state, reward, done, info = env.step(action)
            total_reward += reward
            if info["override"]:
                overrides += 1

            if should_draw(flags, step):
                hud = {
                    "Mode": "EVAL",
                    "Episode": "{}/{}".format(episode, episodes),
                    "Reward": "{:.1f}".format(total_reward),
                    "Auto-brakes": str(overrides),
                }
                env.render(hud)
                if not flags["fast"]:
                    clock.tick(config.FPS)

            if done:
                break

        print("Eval episode {:3d}/{} | Reward: {:9.1f} | Steps: {:3d} | AutoBrakes: {:3d} | Result: {}".format(
            episode, episodes, total_reward, env.steps, overrides, info["outcome"]))


def main():
    args = parse_args()

    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    pygame.init()
    pygame.display.set_caption("Indian Traffic Intelligent Car - DQN")
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    env = TrafficEnv(screen=screen)
    agent = DQNAgent(config.STATE_SIZE, config.ACTION_SIZE)
    print("Device: {}".format(agent.device))

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    flags = {
        "running": True,
        "render": config.RENDER_TRAINING and not args.no_render,
        "fast": args.fast,
    }

    if args.eval:
        agent.load(args.eval)
        agent.epsilon = 0.0
        print("Loaded {} - watching the trained agent".format(args.eval))
        evaluate(env, agent, clock, flags, args.episodes)
    else:
        if args.load:
            checkpoint = agent.load(args.load)
            print("Continuing from {} (epsilon {:.3f})".format(args.load, checkpoint.get("epsilon", agent.epsilon)))
        train(env, agent, clock, flags, args.episodes)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()