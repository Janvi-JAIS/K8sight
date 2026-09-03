"""
K8sight — Multi-pattern load generator for php-apache / php-apache-2

This file defines 5 distinct load "shapes" that Locust can drive against
your sample microservices. Each shape produces a different CPU usage
pattern over time, which is exactly what your forecasting model needs
to learn to distinguish (stable / spike / gradual-increase / periodic /
random-spike).

IMPORTANT: Locust only runs ONE LoadTestShape at a time. To switch
patterns, comment out all the shape classes except the one you want
active, then re-run locust. Instructions are at the bottom of this file.
"""

import math
import random
from locust import HttpUser, task, between, LoadTestShape


# ---------------------------------------------------------------------
# The "user" behavior — what each simulated user actually does.
# Locust spawns however many of these the active shape tells it to.
# ---------------------------------------------------------------------
class ApacheUser(HttpUser):
    wait_time = between(0.1, 0.5)  # small pause between requests per user

    @task
    def hit_root(self):
        self.client.get("/?value=100000")


# ---------------------------------------------------------------------
# PATTERN 1: STABLE
# Constant, low-variance load for a fixed duration. This is your
# baseline — teaches the model what "normal, unchanging" looks like.
# ---------------------------------------------------------------------
class StableShape(LoadTestShape):
    users = 10
    spawn_rate = 5
    run_time = 300  # 5 minutes

    def tick(self):
        run_time = self.get_run_time()
        if run_time > self.run_time:
            return None
        return (self.users, self.spawn_rate)


# ---------------------------------------------------------------------
# PATTERN 2: SPIKE
# Low baseline load, then a sudden short burst, then back to baseline.
# Teaches the model to recognize sharp, short-lived demand jumps.
# ---------------------------------------------------------------------
class SpikeShape(LoadTestShape):
    baseline_users = 5
    spike_users = 80
    spawn_rate = 20

    # (start_second, end_second, user_count)
    stages = [
        (0, 120, baseline_users),
        (120, 180, spike_users),   # the spike
        (180, 360, baseline_users),
    ]

    def tick(self):
        run_time = self.get_run_time()
        for start, end, users in self.stages:
            if start <= run_time < end:
                return (users, self.spawn_rate)
        return None  # stop after the last stage ends


# ---------------------------------------------------------------------
# PATTERN 3: GRADUAL INCREASE
# Load ramps up steadily over time, like organic traffic growth.
# Teaches the model to recognize trending/creeping demand.
# ---------------------------------------------------------------------
class GradualIncreaseShape(LoadTestShape):
    start_users = 5
    end_users = 100
    duration = 600  # 10 minutes to ramp fully
    spawn_rate = 5

    def tick(self):
        run_time = self.get_run_time()
        if run_time > self.duration:
            return None
        progress = run_time / self.duration
        current_users = int(self.start_users + progress * (self.end_users - self.start_users))
        return (current_users, self.spawn_rate)


# ---------------------------------------------------------------------
# PATTERN 4: PERIODIC / CYCLIC
# Load oscillates smoothly up and down (sine wave), like day/night or
# business-hours traffic compressed into minutes. Teaches the model
# to recognize recurring cycles.
# ---------------------------------------------------------------------
class PeriodicShape(LoadTestShape):
    min_users = 5
    max_users = 60
    period_seconds = 120  # one full up-down cycle every 2 minutes
    total_duration = 600  # run for 10 minutes total
    spawn_rate = 10

    def tick(self):
        run_time = self.get_run_time()
        if run_time > self.total_duration:
            return None
        amplitude = (self.max_users - self.min_users) / 2
        midpoint = self.min_users + amplitude
        users = int(midpoint + amplitude * math.sin(2 * math.pi * run_time / self.period_seconds))
        return (max(1, users), self.spawn_rate)


# ---------------------------------------------------------------------
# PATTERN 5: RANDOM SPIKE
# Baseline load with unpredictable random bursts at random times.
# Teaches the model to recognize noisy, unplanned demand — the
# hardest pattern to forecast, and a good stress test for your model.
# ---------------------------------------------------------------------
class RandomSpikeShape(LoadTestShape):
    baseline_users = 8
    spawn_rate = 15
    total_duration = 600  # 10 minutes
    check_interval = 15   # re-roll the dice every 15 seconds
    spike_chance = 0.25   # 25% chance of a spike each interval
    spike_user_range = (40, 90)

    def tick(self):
        run_time = self.get_run_time()
        if run_time > self.total_duration:
            return None

        # Re-seed deterministically per interval so repeated ticks within
        # the same interval don't flicker between values.
        interval_index = int(run_time // self.check_interval)
        rng = random.Random(interval_index)

        if rng.random() < self.spike_chance:
            users = rng.randint(*self.spike_user_range)
        else:
            users = self.baseline_users

        return (users, self.spawn_rate)


# ---------------------------------------------------------------------
# HOW TO SWITCH PATTERNS
#
# Locust auto-detects LoadTestShape subclasses in this file. If more
# than one is left active (uncommented), Locust will only use one of
# them (usually the last one defined) — so to avoid confusion, only
# keep ONE shape class active at a time. To run a given pattern:
#
#   1. Comment out the other 4 shape classes (or just rename this one
#      to leave the shape you want as `class ActiveShape(LoadTestShape):`
#      or move it to its own file, e.g. locustfile_spike.py)
#   2. Run: locust -f locustfile.py --host=http://localhost:8081
#   3. Open http://localhost:8089 and click Start — no need to enter
#      user count, since the shape class controls that automatically.
#
# RECOMMENDED WORKFLOW for K8sight:
# Run each pattern as a SEPARATE locustfile (e.g. locustfile_stable.py,
# locustfile_spike.py, etc.) so you can re-run any single pattern later
# without editing this file each time. Copy the relevant class + the
# ApacheUser class into each separate file.
# ---------------------------------------------------------------------
