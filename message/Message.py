from enum import Enum, unique

@unique
class MessageType(Enum):
  MESSAGE = 1
  WAKEUP = 2

  def __lt__(self, other):
    return self.value < other.value 


class Message:

  def __init__ (self, body = None):
    # The base Message class no longer holds envelope/header information,
    # however any desired information can be placed in the arbitrary
    # body.  Delivery metadata is now handled outside the message itself.
    # The body may be overridden by specific message type subclasses.
    # It is acceptable for WAKEUP type messages to have no body.
    self.body = body

    # The base Message class can no longer do any real error checking.
    # Subclasses are strongly encouraged to do so based on their body.


  def __lt__(self, other):
    # Required by Python3 for this object to be placed in a priority queue.

    # TODO: might consider adding a random number to message objects
    #       at creation time, or storing creation time, to provide
    #       consistent sorting of messages without biasing delivery
    #       at the same timestamp based on arbitrary body comparisons.

    return ("{}".format(self.body) <
            "{}".format(other.body))


  def __str__(self):
    # Make a printable representation of this message.
    return str(self.body)
