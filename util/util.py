# General purpose utility functions for the simulator, attached to no particular class.
# Available to any agent or other module/utility.  Should not require references to
# any simulator object (kernel, agent, etc).

# Module level variable that can be changed by config files.
silent_mode = False


# This optional log_print function will call str.format(args) and print the
# result to stdout.  It will return immediately when silent mode is active.
# Use it for all permanent logging print statements to allow fastest possible
# execution when verbose flag is not set.  This is especially fast because
# the arguments will not even be formatted when in silent mode.
def log_print (str, *args):
  if not silent_mode: print (str.format(*args))


# Accessor method for the global silent_mode variable.
def be_silent ():
  return silent_mode


# Utility method to flatten nested lists.
def delist(list_of_lists):
    return [x for b in list_of_lists for x in b]
