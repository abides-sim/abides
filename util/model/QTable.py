# This utility class holds the internal state of a q-table learning agent
# to allow state preservation across multiple simulations in the same
# experiment.

# An agent's random_state attribute is required to ensure properly repeatable
# experiments.

class QTable():

  def __init__ (self, dims = (100, 2), alpha = 0.99, alpha_decay = 0.999,
                alpha_min = 0.3, epsilon = 0.99, epsilon_decay = 0.999, epsilon_min = 0.1,
                gamma = 0.98, random_state = None) :

    self.q = random_state.normal(loc = 0, scale = 1, size = dims)

    self.alpha = alpha
    self.alpha_decay = alpha_decay
    self.alpha_min = alpha_min

    self.epsilon = epsilon
    self.epsilon_decay = epsilon_decay
    self.epsilon_min = epsilon_min

    self.gamma = gamma

