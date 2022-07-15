import random
import gym
from ch12.trading_env.trading_screen import TradingScreen


class StockTradingEnv(gym.Env):

    def __init__(
            self,
            quotes,
            start_from = 30,
            state_size = 10,
            max_steps = 250,
            name = '',
            trading_screen_width = 100,
    ):
        """
        quotes - the list of daily quotes
        start_from - the day to start trading from
        state_size - number of previous days that will form the state
        max_steps - max steps per episode
        name - trading screen title
        trading_screen_width - trading screen width
        """
        self.start_from = start_from
        self.state_size = state_size
        self.trading_screen_width = trading_screen_width
        self.name = name
        self.quotes = quotes
        self.max_steps = max_steps
        self.trading_scr = TradingScreen()
        self.current_bar = start_from
        self.total_bars = len(quotes)
        self.action_history = []
        self.total_reward = 0
        self.total_steps = 0

    def get_state(self):
        return self.quotes[self.current_bar - self.state_size: self.current_bar]

    def step(self, action):
        # Transforming (0,1,2) action to (-1,0,1) order action
        # action: 2 -> buy: 1
        # action: 1 -> no action
        # action: 0 -> sell: -1
        order_action = action - 1
        self.action_history.append(order_action)
        self.current_bar += 1
        self.total_steps += 1
        info = None
        state = self.get_state()

        # Calculating reward
        diff = self.quotes[self.current_bar] - self.quotes[self.current_bar - 1]
        reward = order_action * diff
        self.total_reward += reward

        # done if no more quote left or maximum steps per episode reached
        done = (len(self.quotes) - 1 == self.current_bar) or\
               self.total_steps == self.max_steps

        return state, reward, done, info

    def reset(self):
        # Setting random trading day
        self.current_bar = random.randint(self.start_from, self.total_bars - self.max_steps - 1)
        self.action_history = [0] * self.current_bar
        self.total_reward = 0
        self.total_steps = 0
        return self.get_state()

    def render(self, mode = 'human'):
        from_pos = max(0, self.current_bar - self.trading_screen_width)
        to_pos = self.current_bar + 1
        sq = self.quotes[from_pos:to_pos]
        actions = self.action_history[from_pos:to_pos]
        sells = {i: sq[i] for i in range(len(actions)) if actions[i] == -1}
        buys = {i: sq[i] for i in range(len(actions)) if actions[i] == 1}
        self.trading_scr.update(sq, buys, sells, self.name, self.total_reward)
