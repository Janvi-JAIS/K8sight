import math
from locust import HttpUser, task, between, LoadTestShape


class ApacheUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def hit_root(self):
        self.client.get("/?value=100000")


class PeriodicShape(LoadTestShape):
    min_users = 5
    max_users = 60
    period_seconds = 120
    total_duration = 600
    spawn_rate = 10

    def tick(self):
        run_time = self.get_run_time()

        if run_time > self.total_duration:
            return None

        amplitude = (self.max_users - self.min_users) / 2
        midpoint = self.min_users + amplitude

        users = int(
            midpoint
            + amplitude
            * math.sin(2 * math.pi * run_time / self.period_seconds)
        )

        return (max(1, users), self.spawn_rate)
