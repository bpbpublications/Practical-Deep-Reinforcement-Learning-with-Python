import random


def epsilon_greedy_policy(e, Q, s):
    r = random.random()
    if r > e:
        max_q = max(Q[s])
        candidates = [i for i in range(len(Q[s, :])) if Q[s, i] == max_q]
        return random.choice(candidates)
    else:
        return random.randint(0, len(Q[s]) - 1)


def greedy_policy(Q, s):
    return epsilon_greedy_policy(0, Q, s)
