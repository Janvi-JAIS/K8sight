from locust import HttpUser, task, between, LoadTestShape


class ApacheUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def hit_root(self):
        self.client.get("/?value=100000")


class StableShape(LoadTestShape):
    users = 10
    spawn_rate = 5
    run_time = 300

    def tick(self):
        run_time = self.get_run_time()

        if run_time > self.run_time:
            return None

        return (self.users, self.spawn_rate)
