class Buffer:

    def __init__(self) -> None:
        self.rewards = []
        self.states = []
        self.actions = []
        self.dones = []

    def clear(self):
        self.rewards = []
        self.states = []
        self.actions = []
        self.dones = []

    def add(self, r, s, a, d):
        self.rewards.append(r)
        self.states.append(s)
        self.actions.append(a)
        self.dones.append(d)

    def unzip(self):
        return self.rewards, self.states, self.actions, self.dones
