from locust import HttpUser, task, between, LoadTestShape


class ApacheUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def hit_root(self):
        self.client.get("/?value=100000")


class GradualIncreaseShape(LoadTestShape):
    start_users = 5
    end_users = 100
    duration = 600  # 10 minutes
    spawn_rate = 5

    def tick(self):
        run_time = self.get_run_time()

        if run_time > self.duration:
            return None

        progress = run_time / self.duration

        current_users = int(
            self.start_users
            + progress * (self.end_users - self.start_users)
        )

        return (current_users, self.spawn_rate)
