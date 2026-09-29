from locust import HttpUser, task, between, LoadTestShape


class ApacheUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def hit_root(self):
        self.client.get("/?value=100000")


class SpikeShape(LoadTestShape):
    baseline_users = 5
    spike_users = 80
    spawn_rate = 20

    stages = [
        (0, 120, baseline_users),
        (120, 180, spike_users),
        (180, 360, baseline_users),
    ]

    def tick(self):
        run_time = self.get_run_time()

        for start, end, users in self.stages:
            if start <= run_time < end:
                return (users, self.spawn_rate)

        return None
