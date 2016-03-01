import pdb
eventQueue = []

class Event(object):
    def __init__(self,
                 timestamp,
                 value):
        self.timestamp = timestamp
        self.value = value
    def __lt__(self, other):
         return self.timestamp < other.timestamp

class MarketUpdate(Event):
    def __init__(self,
                 security,
                 timestamp,
                 value):
        Event.__init__(self, timestamp, value)
        self.security = security

class Prc(MarketUpdate):
    def __init__(self,
                 security,
                 timestamp,
                 value):
        MarketUpdate.__init__(self, security, timestamp, value)
        self.lastPrice = value[0]
        if len(value) > 1:
            self.lastVolume = event.value[1]
        else:
            self.lastVolume = 0

class Book(MarketUpdate):
    def __init__(self,
                 security,
                 timestamp,
                 value):
        MarketUpdate.__init__(self, security, timestamp, value)
        bids = sorted(value[0])
        asks = sorted(value[1])
        self.bidBest = bids[0][0]
        self.bidBestSize = bids[1][0]
        self.bidTotalSize = sum(bids[1])
        self.bidAvgPrc = sum([x*y for x, y in zip(bids[0], bids[1])]) / self.bidTotalSize
        self.askBest = asks[0][0]
        self.askBestSize = asks[1][0]
        self.askTotalSize = sum(asks[1])
        self.askAvgPrc = sum([x*y for x, y in zip(askss[0], asks[1])]) / self.askTotalSize
        self.mid = 0.5 * (self.bidBest + self.askBest)
        
class Bar(MarketUpdate):
    def __init__(self,
                 security,
                 timestamp,
                 value):
        MarketUpdate.__init__(self, security, timestamp, value)
        self.open = value[0]
        self.high = value[1]
        self.low = value[2]
        self.close = value[3]
        self.bodyFill = (self.close >= self.open)

class TickBar(Bar):
    def __init__(self,
                 security,
                 timestamp,
                 value,
                 numTicks):
        Bar.__init__(self, security, timestamp, value)
        self.numTicks = numTicks

class TimeBar(Bar):
    def __init__(self,
                 security,
                 timestamp,
                 value,
                 numTimeUnits,
                 timeUnit):
        Bar.__init__(self, security, timestamp, value)
        self.numTimeUnits = numTimeUnits
        self.timeUnit = timeUnit

class Order(MarketUpdate):
    def __init__(self,
                 security,
                 timestamp,
                 value,
                 orderType,
                 orderQuantity,
                 orderPrice,
                 algoInstance):
        MarketUpdate.__init__(self, security, timestamp, value)
        self.orderType = orderType
        self.orderQuantity = orderQuantity
        self.orderPrice = orderPrice
        self.algoInstance = algoInstance

class Comm(Event):
    def __init__(self,
                 originator,
                 recipients,
                 timestamp,
                 value):
        Event.__init__(self, timestamp, value)
        self.originator = originator
        self.recipients = recipients

class Transition(object):
    def __init__(self,
                 initialState='',
                 finalState='',
                 sensor=None,
                 predicate='',
                 actuator='',
                 effected=''):
        self.initialState = initialState
        self.finalState = finalState
        self.sensor = sensor
        self.predicate = predicate
        self.actuator = actuator
        self.effected = effected

def price(event, slippageFunction=None, size=0, orderType=None):
    #print("----> type(event)=%s %s" % (type(event), event.value))
    if isinstance(event, MarketUpdate):
        price = event.value
    elif isinstance(event, list) or isinstance(event, float):
        price = event
    elif isinstance(event, Comm):
        price = 0
    elif isinstance(event, Prc):
        price = event.lastPrice
    elif isinstance(event, Bar) or isinstance(event, TickBar):
        price = event.open
    elif isinstance(event, Book):
        if size == 0:
            price = event.mid
        elif size > 0:
            price = event.askBest
        else:
            price = event.bidBest
    return adjustPrice(price, slippageFunction, event, size, orderType)

def slippage(event, size, orderType):
    return 0

def adjustPrice(price, slippageFunction, event, size, orderType):
    if slippageFunction != None:
        price *= (1 + slippageFunction(event, size, orderType))
    return price

def emit(agent, emission):
    if isinstance(emission, basestring):
        # If emission is INIT, LONG or SHORT, create a Comm event
        event = Comm(agent, agent.recipientsList, agent.timestamps[-1], emission)
    else:
        event = emission
    # Append event to the queue; this then gets popped by the main loop
    # (runSimulation or run Live) and consumed by the agents.
    eventQueue.insert(0, event)
    agent.outgoingMessages.append(event)
