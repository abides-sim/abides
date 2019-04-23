# General purpose utility functions for the simulator, attached to no particular class.
# Available to any agent or other module/utility.  Should not require references to
# any simulator object (kernel, agent, etc).

import builtins as __builtin__

# Module level variable that can be changed by config files.
silent_mode = False

# Import to overload print() to: (a) require a string parameter and (b) control printing
# levels, particularly suppressing printing for offline/batch runs or speed tests.
def print (*args, **kwargs):
  
  override = False

  if 'override' in kwargs:
    override = kwargs['override']
    del kwargs['override']

  if (not silent_mode) or override:
    return __builtin__.print (*args, **kwargs)

