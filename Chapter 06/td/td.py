def q_learning(Q, current_s, next_s, a, r, gamma = .9, alpha = .5):
    # r + g*max[a](Q(S', a)) - Q(S, A)
    td = (r + gamma * max(Q[next_s]) - Q[current_s, a])
    Q[current_s, a] += alpha * round(td, 2)
    return Q
