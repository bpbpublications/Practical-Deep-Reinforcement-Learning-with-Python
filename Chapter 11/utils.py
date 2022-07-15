class Buffer:

    def __init__(self) -> None:
        self.rewards = []
        self.actions = []
        self.states = []

    def clear(self):
        self.rewards = []
        self.actions = []
        self.states = []

    def add(self, r, a, s):
        self.rewards.append(r)
        self.actions.append(a)
        self.states.append(s)

    def unzip(self):
        return self.rewards, self.actions, self.states
