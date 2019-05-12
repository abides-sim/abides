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


# This optional log_print function will call str.format(args) and print the
# result to stdout.  It will return immediately when silent mode is active.
# Use it for all permanent logging print statements to allow fastest possible
# execution when verbose flag is not set.
def log_print (str, *args):
  if not silent_mode: print (str.format(*args))


# Accessor method for the global silent_mode variable.
def be_silent ():
  return silent_mode

