import pandas as pd
from agent.execution.ExecutionAgent import ExecutionAgent
from util.util import log_print


class TWAPExecutionAgent(ExecutionAgent):
    """
    TWAP (Time Weighted Average Price) Execution Agent:
        - Aims to execute the parent order as close as possible to the TWAP average price
        - breaks up the parent order and releases dynamically smaller chunks of the order to the market using
        evenly divided time slots during the execution time horizon.
        -
        e.g. 18,000 shares - to be executed between 09:32:00 and 09:35:00
        - > 18,000 shares over 3 minutes = 100 shares per second
    """

    def __init__(self, id, name, type, symbol, starting_cash,
                 direction, quantity, execution_time_horizon, freq,
                 trade=True, log_orders=False, random_state=None):
        super().__init__(id, name, type, symbol, starting_cash,
                         direction=direction, quantity=quantity, execution_time_horizon=execution_time_horizon,
                         trade=trade, log_orders=log_orders, random_state=random_state)

        self.freq = freq
        self.schedule = self.generate_schedule()

    def generate_schedule(self):

        schedule = {}
        bins = pd.interval_range(start=self.start_time, end=self.end_time, freq=self.freq)
        child_quantity = int(self.quantity / len(self.execution_time_horizon))
        for b in bins:
            schedule[b] = child_quantity
        log_print('[---- {} {} - Schedule ----]:'.format(self.name, self.currentTime))
        log_print('[---- {} {} - Total Number of Orders ----]: {}'.format(self.name, self.currentTime, len(schedule)))
        for t, q in schedule.items():
            log_print("From: {}, To: {}, Quantity: {}".format(t.left.time(), t.right.time(), q))
        return schedule