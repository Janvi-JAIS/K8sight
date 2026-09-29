import random
from locust import HttpUser, task, between, LoadTestShape


class ApacheUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def hit_root(self):
        self.client.get("/?value=100000")


class RandomSpikeShape(LoadTestShape):
    baseline_users = 8
    spawn_rate = 15
    total_duration = 600
    check_interval = 15
    spike_chance = 0.25
    spike_user_range = (20,45)

    def tick(self):
        run_time = self.get_run_time()

        if run_time > self.total_duration:
            return None

        interval_index = int(run_time // self.check_interval)
        rng = random.Random(interval_index)

        if rng.random() < self.spike_chance:
            users = rng.randint(*self.spike_user_range)
        else:
            users = self.baseline_users

        return (users, self.spawn_rate)
