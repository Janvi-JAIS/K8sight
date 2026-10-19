from locust import HttpUser, task, between, LoadTestShape


class ApacheUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def hit_root(self):
        self.client.get("/?value=10500")


class StableShape(LoadTestShape):
    users = 1
    spawn_rate = 5
    run_time = 300

    def tick(self):
        run_time = self.get_run_time()

        if run_time > self.run_time:
            return None

        return (self.users, self.spawn_rate)
