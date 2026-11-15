"""Intelli-Car presentation / test run - show the trained agent as many times as you like.

No training happens here. The saved agent drives with its best actions (no exploration)
and you control the show from the keyboard. Nothing is one-shot: the demo loops forever
until you press ESC, and the scored test can be started again and again.

    python test_run.py                                  # shows the best trained model (checkpoints/dqn_best.pth)
    python test_run.py --scored                         # jumps straight into the scored test
    python test_run.py --model checkpoints/dqn_last.pth
    python test_run.py --scenario curvy_road            # start locked on one scenario
    python test_run.py --episodes 5                     # episodes per scenario in the showcase
    python test_run.py --random                         # new random traffic every time

Two modes:
    SHOWCASE     endless loop through the four scenarios (or one locked scenario). With fixed
                 seeds (default) every round shows exactly the same traffic, so you can repeat
                 a demonstration and get the same result in front of the teacher.
    SCORED TEST  runs N episodes on each of the four scenarios with fixed seeds, shows a
                 results table on screen and saves it to results/test_runs/. Press T to run
                 it again, any number of times.

Keys
    SPACE   pause / resume          N   next episode          B   previous episode
    R       replay this episode     1-4 lock to a scenario    A   cycle all scenarios
    S       fixed seeds <-> random  UP / DOWN  faster / slower
    T       start a scored test     H   help on screen        ESC quit
"""

import argparse
import datetime
import os
import random

import numpy as np
import pygame

import config
from env.traffic_env import TrafficEnv
from utils.evaluation import print_table, save_results, summarize, table_lines

SPEED_STEPS = [15, 30, 60, 120, 240, 480]       # simulation steps per second
DEFAULT_SPEED_INDEX = 2
HOLD_FRAMES = 75                                # how long the result of an episode stays on screen
OUTCOME_TEXT = {
    "destination": ("FINISH LINE REACHED", config.COLOR_GOOD),
    "collision": ("COLLISION", config.COLOR_BAD),
    "double_line": ("CROSSED THE DOUBLE LINE", config.COLOR_BAD),
    "stuck": ("STUCK - TIME OUT", config.COLOR_WARN),
    "timeout": ("STEP LIMIT REACHED", config.COLOR_WARN),
}
RUN_DIR = os.path.join(config.RESULTS_DIR, "test_runs")


def parse_args():
    parser = argparse.ArgumentParser(description="Intelli-Car presentation / repeatable test run")
    parser.add_argument("--model", type=str, default=None,
                        help="checkpoint to show (default: best model, else final, else last)")
    parser.add_argument("--scenario", type=str, default=None, choices=list(config.SCENARIOS.keys()),
                        help="start locked on this scenario")
    parser.add_argument("--episodes", type=int, default=3, help="episodes per scenario in the showcase")
    parser.add_argument("--test-episodes", type=int, default=None,
                        help="episodes per scenario in the scored test (default: config.FINAL_TEST_EPISODES)")
    parser.add_argument("--random", action="store_true", help="use new random traffic instead of fixed seeds")
    parser.add_argument("--scored", action="store_true", help="start directly with a scored test")
    parser.add_argument("--speed", type=int, default=DEFAULT_SPEED_INDEX + 1, choices=range(1, len(SPEED_STEPS) + 1),
                        help="starting speed 1 (slowest) to {}".format(len(SPEED_STEPS)))
    return parser.parse_args()


class Show:
    """All the state of the running presentation."""

    LOCKED_DURING_TEST = (
        pygame.K_n, pygame.K_b, pygame.K_r, pygame.K_a, pygame.K_s,
        pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4,
    )

    def __init__(self, env, agent, screen, args, model_path):
        self.env = env
        self.agent = agent
        self.screen = screen
        self.clock = pygame.time.Clock()
        self.model_path = model_path

        self.font = pygame.font.Font(None, 26)
        self.font_small = pygame.font.Font(None, 21)
        self.font_big = pygame.font.Font(None, 54)
        self.font_mono = pygame.font.SysFont("consolas,couriernew,monospace", 17)

        self.running = True
        self.paused = False
        self.show_help = False
        self.speed_index = max(0, min(len(SPEED_STEPS) - 1, args.speed - 1))
        self.fixed_seeds = not args.random
        self.showcase_episodes = max(1, args.episodes)
        self.test_episodes = args.test_episodes if args.test_episodes is not None else config.FINAL_TEST_EPISODES
        self.locked = args.scenario            # None = cycle through all scenarios

        # job queue: each job is (scenario, seed, label)
        self.mode = "SHOWCASE"
        self.round = 1
        self.jobs = []
        self.job_index = -1                    # index of the current job inside self.jobs
        self.history = []                      # finished episodes of the current mode: (job, result)
        self.stats = {}                        # scenario -> [episodes, wins] for the showcase scoreboard

        self.state = None
        self.job = None
        self.total_reward = 0.0
        self.overrides = 0
        self.speed_sum = 0.0
        self.steps_done = 0
        self.result = None                     # result of the finished episode, shown during the hold
        self.hold = 0
        self.report_lines = None               # on-screen test report, shown until a key is pressed
        self.report_saved = ""

        self.env.hints = ["H    help", "ESC  quit"]
        if args.scored:
            self.start_scored_test()
        else:
            self.start_showcase()

    # ------------------------------------------------------------------
    # Job lists
    # ------------------------------------------------------------------
    def scenarios(self):
        return [self.locked] if self.locked else list(config.SCENARIO_ORDER)

    def make_seed(self, i):
        return config.TEST_SEED + i if self.fixed_seeds else random.randrange(1, 10 ** 9)

    def build_showcase_round(self):
        jobs = []
        for scenario in self.scenarios():
            for i in range(self.showcase_episodes):
                jobs.append((scenario, self.make_seed(i), "{}/{}".format(i + 1, self.showcase_episodes)))
        return jobs

    def start_showcase(self):
        self.mode = "SHOWCASE"
        self.round = 1
        self.jobs = self.build_showcase_round()
        self.job_index = -1
        self.history = []
        self.stats = {}
        self.report_lines = None
        self.begin_next_episode()

    def start_scored_test(self):
        self.mode = "SCORED TEST"
        self.jobs = []
        for scenario in config.SCENARIO_ORDER:
            for i in range(self.test_episodes):
                self.jobs.append((scenario, config.TEST_SEED + i, "{}/{}".format(i + 1, self.test_episodes)))
        self.job_index = -1
        self.history = []
        self.report_lines = None
        print("")
        print("Scored test started: {} episodes on each of {}".format(self.test_episodes, ", ".join(config.SCENARIO_ORDER)))
        self.begin_next_episode()

    def restart_showcase_keep_settings(self):
        """Used after the lock, seed mode or episode count changed."""
        self.mode = "SHOWCASE"
        self.round = 1
        self.jobs = self.build_showcase_round()
        self.job_index = -1
        self.history = []
        self.begin_next_episode()

    # ------------------------------------------------------------------
    # Episodes
    # ------------------------------------------------------------------
    def begin_next_episode(self):
        self.job_index += 1
        if self.job_index >= len(self.jobs):
            if self.mode == "SHOWCASE":
                self.round += 1
                self.jobs = self.build_showcase_round()
                self.job_index = 0
            else:
                self.finish_scored_test()
                return
        self.begin_episode(self.job_index)

    def begin_episode(self, index):
        self.job_index = index
        scenario, seed, label = self.jobs[index]
        self.job = (scenario, seed, label)
        random.seed(seed)
        np.random.seed(seed)
        self.state = self.env.reset(scenario)
        self.total_reward = 0.0
        self.overrides = 0
        self.speed_sum = 0.0
        self.steps_done = 0
        self.result = None
        self.hold = 0

    def replay_episode(self):
        self.begin_episode(self.job_index)

    def step_episode(self):
        action = self.agent.select_action(self.state, training=False)
        self.state, reward, done, info = self.env.step(action)
        self.total_reward += reward
        self.steps_done += 1
        self.speed_sum += info["speed"]
        if info["override"]:
            self.overrides += 1
        if done:
            self.finish_episode(info)

    def finish_episode(self, info):
        scenario = self.job[0]
        self.result = {
            "scenario": scenario,
            "reward": self.total_reward,
            "steps": self.env.steps,
            "outcome": info["outcome"],
            "avg_speed": self.speed_sum / max(1, self.steps_done),
            "auto_brakes": self.overrides,
            "violations": info["violations"],
        }
        self.hold = HOLD_FRAMES
        self.history.append((self.job, self.result))
        if self.mode == "SHOWCASE":
            entry = self.stats.setdefault(scenario, [0, 0])
            entry[0] += 1
            entry[1] += 1 if info["outcome"] == "destination" else 0
        print("{} | {} {} | seed {} | reward {:.1f} | steps {} | {}".format(
            self.mode.lower(), scenario, self.job[2], self.job[1], self.total_reward, self.env.steps,
            info["outcome"]))

    def finish_scored_test(self):
        summaries = []
        for scenario in config.SCENARIO_ORDER:
            results = [res for job, res in self.history if job[0] == scenario]
            summaries.append(summarize(scenario, results))
        print_table(summaries, "SCORED TEST RESULTS (best actions only, same traffic every run)")

        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(RUN_DIR, "test_run_{}.csv".format(stamp))
        json_path = os.path.join(RUN_DIR, "test_run_{}.json".format(stamp))
        save_results(summaries, csv_path, json_path, self.model_path, self.test_episodes, config.TEST_SEED)
        self.report_saved = "Saved to {}".format(csv_path)
        self.report_lines = table_lines(summaries)
        self.mode = "REPORT"

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------
    def lock_to(self, scenario):
        self.locked = scenario
        self.restart_showcase_keep_settings()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                continue
            if event.type != pygame.KEYDOWN:
                continue

            if self.mode == "REPORT":
                # any key closes the report; T starts another scored test straight away
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_t:
                    self.start_scored_test()
                else:
                    self.start_showcase()
                continue

            key = event.key
            if self.mode == "SCORED TEST" and key in self.LOCKED_DURING_TEST:
                continue    # skipping, replaying or changing seeds would falsify the scored result
            if key == pygame.K_ESCAPE:
                self.running = False
            elif key == pygame.K_SPACE:
                self.paused = not self.paused
            elif key == pygame.K_h:
                self.show_help = not self.show_help
            elif key == pygame.K_UP:
                self.speed_index = min(len(SPEED_STEPS) - 1, self.speed_index + 1)
            elif key == pygame.K_DOWN:
                self.speed_index = max(0, self.speed_index - 1)
            elif key == pygame.K_r:
                self.replay_episode()
            elif key == pygame.K_n:
                self.begin_next_episode()
            elif key == pygame.K_b:
                self.begin_episode(max(0, self.job_index - 1))
            elif key == pygame.K_t:
                self.start_scored_test()
            elif key == pygame.K_a:
                self.lock_to(None)
            elif key == pygame.K_s:
                self.fixed_seeds = not self.fixed_seeds
                if self.mode == "SHOWCASE":
                    self.restart_showcase_keep_settings()
            elif pygame.K_1 <= key <= pygame.K_4:
                index = key - pygame.K_1
                if index < len(config.SCENARIO_ORDER):
                    self.lock_to(config.SCENARIO_ORDER[index])

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def hud(self):
        scenario, seed, label = self.job
        lines = {
            "Model": os.path.basename(self.model_path),
            "Mode": self.mode,
            "Episode": label,
            "Seed": "{}{}".format(seed, " fix" if self.fixed_seeds else " rnd"),
        }
        if self.mode == "SHOWCASE":
            lines["Round"] = str(self.round)
            wins = sum(v[1] for v in self.stats.values())
            total = sum(v[0] for v in self.stats.values())
            lines["Finished"] = "{}/{}".format(wins, total)
        else:
            done = len(self.history)
            wins = sum(1 for _, res in self.history if res["outcome"] == "destination")
            lines["Test"] = "{}/{} ({} ok)".format(done, len(self.jobs), wins)
        lines["Reward"] = "{:.0f}".format(self.total_reward)
        lines["Rate"] = "{}/s{}".format(SPEED_STEPS[self.speed_index], " PAUSED" if self.paused else "")
        return lines

    def draw_banner(self):
        if self.result is None:
            return
        text, color = OUTCOME_TEXT.get(self.result["outcome"], ("EPISODE OVER", config.COLOR_TEXT))
        label = self.font_big.render(text, True, color)
        width = label.get_width() + 40
        box = pygame.Surface((width, 150), pygame.SRCALPHA)
        box.fill((0, 0, 0, 190))
        x = int(config.ROAD_CENTER_X - width / 2)
        x = max(8, min(config.PANEL_X - width - 8, x))
        self.screen.blit(box, (x, 210))
        self.screen.blit(label, (x + 20, 224))
        detail = "{} steps   reward {:.0f}   seed {}".format(self.result["steps"], self.result["reward"], self.job[1])
        self.screen.blit(self.font.render(detail, True, config.COLOR_TEXT), (x + 20, 290))
        self.screen.blit(self.font_small.render("N next   R replay   B previous", True, config.COLOR_TEXT_DIM),
                         (x + 20, 322))

    def draw_overlay_box(self, lines, title, footer, font):
        line_height = font.get_height() + 4
        width = max(font.size(line)[0] for line in lines) + 40 if hasattr(font, "size") else 760
        width = min(width, config.SCREEN_WIDTH - 40)
        height = line_height * len(lines) + 120
        x = (config.SCREEN_WIDTH - width) // 2
        y = max(20, (config.SCREEN_HEIGHT - height) // 2)
        box = pygame.Surface((width, height), pygame.SRCALPHA)
        box.fill((10, 12, 18, 235))
        self.screen.blit(box, (x, y))
        self.screen.blit(self.font_big.render(title, True, config.COLOR_TEXT), (x + 20, y + 14))
        cy = y + 72
        for line in lines:
            self.screen.blit(font.render(line, True, config.COLOR_TEXT), (x + 20, cy))
            cy += line_height
        self.screen.blit(self.font_small.render(footer, True, config.COLOR_TEXT_DIM), (x + 20, y + height - 30))

    def draw_help(self):
        lines = [
            "SPACE       pause / resume",
            "N / B       next / previous episode",
            "R           replay this episode (same traffic)",
            "1 2 3 4     lock to highway / heavy traffic / single lane / curvy road",
            "A           cycle through all four scenarios",
            "S           fixed seeds (repeatable)  <->  random traffic",
            "UP / DOWN   faster / slower",
            "T           start a scored test on all four scenarios (repeatable)",
            "H           close this help        ESC  quit",
        ]
        self.draw_overlay_box(lines, "Controls", "Fixed seeds: every round shows exactly the same traffic.",
                              self.font_mono)

    def draw_report(self):
        lines = list(self.report_lines)
        lines.append("")
        lines.append(self.report_saved)
        self.draw_overlay_box(lines, "Scored test finished",
                              "T  run the scored test again    any other key  back to the showcase    ESC  quit",
                              self.font_mono)

    def render(self):
        self.env.render(self.hud(), flip=False)
        self.draw_banner()
        if self.show_help:
            self.draw_help()
        pygame.display.flip()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self):
        while self.running:
            self.handle_events()
            if not self.running:
                break

            if self.mode == "REPORT":
                self.env.render(self.hud(), flip=False)
                self.draw_report()
                pygame.display.flip()
                self.clock.tick(30)
                continue

            if self.result is not None:
                if not self.paused:
                    self.hold -= 1
                    if self.hold <= 0:
                        self.begin_next_episode()
            elif not self.paused:
                self.step_episode()

            if self.mode != "REPORT":
                self.render()
            self.clock.tick(SPEED_STEPS[self.speed_index])


def main():
    args = parse_args()

    if args.model is None:
        for candidate in (config.BEST_MODEL_PATH, config.FINAL_MODEL_PATH, config.LAST_MODEL_PATH):
            if os.path.isfile(candidate):
                args.model = candidate
                break
        else:
            args.model = config.BEST_MODEL_PATH

    if not os.path.isfile(args.model):
        print("Could not find the model file: {}".format(args.model))
        print("Train first with:  python main.py")
        print("or point to a checkpoint:  python test_run.py --model checkpoints/dqn_last.pth")
        return

    import torch  # imported late so the message above works even when torch is missing
    from agent.dqn_agent import DQNAgent

    torch.manual_seed(config.SEED)
    pygame.init()
    pygame.display.set_caption("Intelli-Car - presentation run")
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))

    agent = DQNAgent(config.STATE_SIZE, config.ACTION_SIZE)
    agent.load(args.model)
    agent.epsilon = 0.0
    print("Loaded {}  (device: {})".format(args.model, agent.device))
    print("Press H in the window for the controls.")

    env = TrafficEnv(screen=screen, scenario=args.scenario or config.SCENARIO_ORDER[0])
    show = Show(env, agent, screen, args, args.model)
    try:
        show.run()
    except KeyboardInterrupt:
        pass

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()