import time

class Profiler:
    times: dict[str, float]
    cur_stage: str
    start_time: float

    def __init__(self):
        self.times = {}
        self.cur_stage = None
    
    def enter(self, stage: str):
        t = time.time()

        if self.cur_stage is not None:
            self.stop(t)

        self.cur_stage = stage
        self.start_time = t
    
    def stop(self, t: float = None):
        if t is None:
            t = time.time()

        old_time = self.times[self.cur_stage] if self.cur_stage in self.times else 0
        delta_time = t - self.start_time
        new_time = old_time + delta_time
        self.times[self.cur_stage] = new_time
    
    def get(self) -> dict[str, float]:
        return self.times

    def clear(self) -> dict[str, float]:
        values = self.times
        self.times = {}
        return values
