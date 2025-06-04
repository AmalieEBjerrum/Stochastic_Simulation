class LCG:
    def __init__(self, seed, a=1664525, c=1013904223, m=2**32):
        self.a = a
        self.c = c
        self.m = m
        self.state = seed

    def next(self):
        self.state = (self.a * self.state + self.c) % self.m
        return self.state

    def next_uniform(self):
        """Returns a float in (0,1)"""
        return self.next() / self.m

    def generate(self, n):
        return [self.next_uniform() for _ in range(n)]

if __name__ == "__main__":
    lcg = LCG(seed=12345)
    numbers = lcg.generate(10)
    for i, x in enumerate(numbers):
        print(f"U_{i+1} = {x}")
