from enum import Enum, unique

@unique
class MessageType(Enum):
  MESSAGE = 1
  WAKEUP = 2

  def __lt__(self, other):
    return self.value < other.value 


class Message:

  uniq = 0

  def __init__ (self, body = None):
    # The base Message class no longer holds envelope/header information,
    # however any desired information can be placed in the arbitrary
    # body.  Delivery metadata is now handled outside the message itself.
    # The body may be overridden by specific message type subclasses.
    # It is acceptable for WAKEUP type messages to have no body.
    self.body = body

    # The autoincrementing variable here will ensure that, when Messages are
    # due for delivery at the same time step, the Message that was created
    # first is delivered first.  (Which is not important, but Python 3
    # requires a fully resolved chain of priority in all cases, so we need
    # something consistent.)  We might want to generate these with stochasticity,
    # but guarantee uniqueness somehow, to make delivery of orders at the same
    # exact timestamp "random" instead of "arbitrary" (FIFO among tied times)
    # as it currently is.
    self.uniq = Message.uniq
    Message.uniq += 1

    # The base Message class can no longer do any real error checking.
    # Subclasses are strongly encouraged to do so based on their body.


  def __lt__(self, other):
    # Required by Python3 for this object to be placed in a priority queue.
    # If we ever decide to place something on the queue other than Messages,
    # we will need to alter the below to not assume the other object is
    # also a Message.

    return (self.uniq < other.uniq)


  def __str__(self):
    # Make a printable representation of this message.
    return str(self.body)
