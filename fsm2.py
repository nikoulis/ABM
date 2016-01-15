# This is simply so that 'print' is an expression (like in Python 3)
# instead of a function and therefore can be used in the lambdas below.
from __future__ import print_function

import ystockquote
import time
import pdb
import sys
import zmq


class Agent(object):
    def __init__(self,
                 name,
                 timestamps=None,
                 revalPrices=None,
                 orders=None,
                 positions=None,
                 pls=None,
                 fitnesses=None,
                 trades=None,
                 tradestats=None,
                 incomingMessages=None,
                 outgoingMessages=None,
                 recipientsList=None):
        self.name             = name             if name != None else []
        self.timestamps       = timestamps       if timestamps != None else []
        self.revalPrices      = revalPrices      if revalPrices != None else []
        self.orders           = orders           if orders != None else []
        self.positions        = positions        if positions != None else []
        self.pls              = pls              if pls != None else []
        self.fitnesses        = fitnesses        if fitnesses != None else []
        self.trades           = trades           if trades != None else []
        self.tradestats       = tradestats       if tradestats != None else []
        self.incomingMessages = incomingMessages if incomingMessages != None else []
        self.outgoingMessages = outgoingMessages if outgoingMessages != None else []
        self.recipientsList   = recipientsList   if recipientsList != None else []

agent = Agent('MyFirstAgent')

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

def sendOrder(agent, event, orderPrice, orderQuantity, orderType, orderId):
    if orderType in ['STP', 'IOC', 'MOC', 'MOO']:
        algoType = 'AGGRESSIVE'
    elif orderType == 'LMT':
        algoType = 'PASSIVE'
    algoInstance = Sim(algoType, slippage)
    order = Order(security=event.security,
                  timestamp=event.timestamp,
                  value=orderId,
                  orderType=orderType,
                  orderQuantity=orderQuantity,
                  orderPrice=orderPrice,
                  algoInstance=algoInstance)
    agent.orders.append(order)

def modifyOrder(agent, event, orderPrice=None, orderQuantity=None, orderType=None, orderId=None):
    thisOrder = [o for o in agent.orders if o.orderId == event.value]
    otherOrders = [o for o in agent.orders if o.orderId != event.value]
    thisOrder.orderPrice = orderPrice if orderPrice != None else None
    thisOrder.orderQuantity = orderQuantity if orderQuantity != None else None
    thisOrder.orderType = orderType if orderType != None else None
    thisOrder.timestamp = event.timestamp
    agent.orders = otherOrders.append(thisOrder)

def cancelOrder(agent, orderId):
    agent.orders = [o for o in agent.orders if o.value != orderId]

def liftQuotes(quotesList, quantity, maxDepth):
    f = quotesList[0]
    q = f[1]
    result = [f]
    for i in range(1, maxDepth):
        if q <= quantity:
            levelQuantity = quotesList[i]
            q += levelQuantity[1]
            result.append(levelQuantity)
    excess = sum(result) - quantity
    if excess > 0:
        result[0][1] -= excess
    return result

def execute(algo, order, event):
    if isinstance(algo, Sim):
        trades = [Trade(timestamp=event.timestamp,
                      price=price(event,
                                  slippageFunction=algo.slippage,
                                  size=order.orderQuantity,
                                  orderType=order.orderType),
                      tradeQuantity=order.orderQuantity)]
        return (trades, None)
    elif isinstance(algo, Aggressor):
        q = orderQuantity(order)
        if q > 0:
            bookSide = event.value[1]
        else:
            bookSide = event.value[0]
        sweep = liftQuotes(bookside, q, algo.maxDepth)
        trades = map(lambda x:
                     Trade(timestamp=event.timestamp,
                           price=first(x[0]),
                           quantity=(1 if q > 0 else -1) * x[1]),
                     sweep)
        sweepSize = sum(sweep[1])
        shortfall = q - sweepSize
        if shortfall > 0:
            order = Order(timestamp=event.timestamp,
                          value=order.value,
                          security=event.security,
                          orderPrice=order.orderPrice,
                          orderQuantity=shortfall,
                          orderType=order.orderType,
                          algoInstance=algo)
        return (trades, order)

def algoCategorize(algoInstance, algoCategory):
    if algoCategory == 'ALL':
        return True
    else:
        return algoInstance == algoCategory

def oms(agent, event, algoCategory):
    categoryOrders = [o for o in agent.orders if algoCategorize(o.algoInstance, algoCategory)]
    nonCategoryOrders = [o for o in agent.orders if not algoCategorize(o.algoInstance, algoCategory)]
    newCategoryOrders = []
    for order in categoryOrders:
        if order.security == event.security:
            (executions, remainingOrder) = execute(order.algoInstance, order, event)
            if len(executions) > 0:
                agent.trades.append(executions)
                agent.tradestats.append(computeStats(agent.trades))
            if remainingOrder != None:
                newCategoryOrders.append(remainingOrder)
    agent.orders = nonCategoryOrders + newCategoryOrders

class Algo(object):
    def __init__(self, algoType):
        self.algoType = algoType   # 'Passive' or 'Aggresive'

class Sim(Algo):
    def __init__(self, algoType, slippageFunction):
        Algo.__init__(self, algoType)
        self.slippage = slippageFunction

class Aggressor(Algo):
    def __init__(self, algoType, maxDepth):
        Algo.__init__(self, algoType)
        self.maxDepth = maxDepth

def slippage(event, size, orderType):
    return 0

def adjustPrice(price, slippageFunction, event, size, orderType):
    if slippageFunction != None:
        price *= (1 + slippageFunction(event, size, orderType))
    return price

class Trade(object):
    def __init__(self, timestamp, price, tradeQuantity):
        self.timestamp = timestamp
        self.price = price
        self.tradeQuantity = tradeQuantity

def price(event, slippageFunction=slippage, size=0, orderType=None):
    #print("----> type(event)=%s %s" % (type(event), event.value))
    if isinstance(event, MarketUpdate):
        price = event.value
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

def consume(agent, event):
    if observe(agent, event):
        updateBefore(agent, event)
        update(agent, event)
        updateAfter(agent, event)

def observe(agent, event):
    if isinstance(agent, SimpleModel) and isinstance(event, MarketUpdate):
        return (agent.mkt == event.security)
    elif isinstance(agent, SimpleModel) and isinstance(event, Comm):
        return (agent in event.recipients) and (agent != event.originator)
    elif isinstance(agent, TickBarGenerator) and isinstance(event, MarketUpdate):
        return (agent.mkt == event.security) and (not isinstance(event, Bar))
    else:
        return False

def updateBefore(agent, event):
    if isinstance(event, MarketUpdate):
        if agent.timestamps == []:
            agent.pls.append(0)
            agent.fitnesses.append(0)
        agent.timestamps.append(event.timestamp)
        agent.revalPrices.append(price(event))
        oms(agent, event, 'ALL')
        preprocess(agent, event)
        print("updateBefore completed for agent %s and MktUpdate event %s" % (agent.name, event.timestamp))
    elif isinstance(event, Comm):
        agent.incomingMessages.append(event)
        preprocess(agent, event)
        print("updateBefore completed for agent %s and Comm event %s" % (agent.name, event))
    
#def update(agent, marketUpdate):
#    position = raw_input("Enter new position for T=%s and P=%s\n" % (marketUpdate.timestamp, price(marketUpdate)))
#    agent.positions.append(position)

class TradeStats:
    percentProfitable = 0
    winToLoss = 0
    avgLogRet = 0
    totPL = 0
    avgDuration = 0
    posPL = 0
    negPL = 0
    profitFactor = 0

def computeStats(trades):
    stats = TradeStats()
    stats.percentProfitable = 0
    stats.winToLoss = 0
    stats.avgLogRet = 0
    stats.totPL = 0
    stats.avgDuration = 0
    stats.posPL = 0
    stats.negPL = 0
    stats.profitFactor = 0
    return (stats)

def updateAfter(agent, event):
    if isinstance(event, MarketUpdate):
        l = len(agent.timestamps)
        lastPos = agent.positions[-1]
        if l < 2:
            prevPos = 0
        else:
            prevPos = agent.positions[-2]
        tradeQuantity = lastPos - prevPos
        lastPrice = agent.revalPrices[-1]
        if l < 2:
            prevPrice = 0
            pl = 0
        else:
            prevPrice = agent.revalPrices[-2]
            pl = prevPos * (lastPrice - prevPrice)
        agent.pls.append(pl)
        if tradeQuantity != 0:
            sendOrder(agent, event,
                      orderPrice=price(event),
                      orderQuantity=tradeQuantity,
                      orderType='STP',
                      orderId='POSCHG')
            print("Generated aggressive order for agent %s for quantity %s" % (agent.name, tradeQuantity))
#            trade = Trade(event.timestamp,
#                          price(event) + slippage(agent, event, tradeQuantity),
#                          tradeQuantity)
#            agent.trades.append(trade)
#            agent.tradestats.append(computeStats(agent.trades))
        postprocess(agent, event)
        print("updateAfter completed for agent %s and MktUpdate event %s" % (agent.name, event.timestamp))
    elif isinstance(event, Comm):
        postprocess(agent, event)
        print("updateAfter completed for agent %s and Comm event %s" % (agent.name, event))
    print()

def clusterAgents(agents, stat, numBins):
    def getStats(agent):
        ts = agent.tradestats
        if stat == 'tpl':
            return stat

class Transition(object):
    def __init__(self,
                 initialState='',
                 finalState='',
                 sensor=price,
                 predicate='',
                 actuator='',
                 effected=''):
        self.initialState = initialState
        self.finalState = finalState
        self.sensor = sensor
        self.predicate = predicate
        self.actuator = actuator
        self.effected = effected

class Comm(Event):
    def __init__(self,
                 originator,
                 recipients,
                 timestamp,
                 value):
        Event.__init__(self, timestamp, value)
        self.originator = originator
        self.recipients = recipients

agents = []
events = []    

# Socket to talk to server
context = zmq.Context()
socket = context.socket(zmq.SUB)
    

def emit(agent, msg):
    comm = Comm(agent, agent.recipientsList, agent.timestamps[-1], msg)
    # Append Comm event to the queue; this then gets popped by runSimulation
    # (prior to the next MarketUpdate event) and consumed by the agents.
    events.append(comm)
    event = Event(agent.timestamps[-1], msg)
    agent.outgoingMessages.append(event)

def perform(transition, event):
    transition.effected = transition.predicate(transition.sensor(event))
    return transition.effected

def operateFSM(fsm, event):
    applicableTransitions = [tr for tr in fsm.transitions if tr.initialState == fsm.currentState]
    effectedTransition = [tr for tr in applicableTransitions if perform(tr, event)][0]
    effectedTransition.actuator(effectedTransition.sensor(event))
    fsm.currentState = effectedTransition.finalState
    print("Transition: %s -> %s"% (effectedTransition.initialState, effectedTransition.finalState))

def update(agent, event):
    if isinstance(event, MarketUpdate):
        agent.setFSM()
        print("setFSM completed for agent %s" % agent.name)
        operateFSM(agent, event)
        print("operateFSM completed for agent %s" % agent.name)
        agent.states.append(agent.currentState)
        print("Completed work for %s and new state %s added" % (agent.name, agent.currentState))
    elif isinstance(event, Comm):
        agent.setFSM()
        print("setFSM completed for agent %s" % agent.name)
        print("Completed work for %s and comm event %s" % (agent.name, event))

class FSM(object):
    def __init__(self, currentState='', transitions=None):
        self.currentState = currentState
        self.transitions = transitions if transitions != None else []

class FSMAgent(FSM, Agent):
    def __init__(self,
                 name,
                 currentState='',
                 transitions=None,
                 states=None):
        FSM.__init__(self, currentState, transitions)
        Agent.__init__(self, name='')
        self.states = states if states != None else []

class SimpleModel(FSMAgent):
    def __init__(self, L, counter=0, ma=0):
        FSMAgent.__init__(self, name='')
        self.L = L
        self.counter = counter
        self.ma = ma
        self.states = ['INIT']
        self.name = 'SIMPLE_MODEL' + str(self.L)

class SimpleModelComm(SimpleModel):
    def __init__(self, L, mkt='', unblockShort=0, unblockLong=0):
        SimpleModel.__init__(self, L)
        self.mkt = mkt
        self.unblockLong = unblockLong
        self.unblockShort = unblockShort
        if len(self.states) == 0:
            emit(self, 'INIT')
    
    def setFSM(self):
        self.currentState = self.states[-1]
        self.transitions = [Transition(initialState='INIT',
                                       finalState='INIT',
                                       sensor=price,
                                       predicate=(lambda x:
                                                  self.counter <= self.L),
                                       actuator=(lambda x:
                                                 self.positions.append(0) or
                                                 print("%s %s -> %s" % (self.name,
                                                                        'INIT',
                                                                        'INIT')))),
                            Transition(initialState='INIT',
                                       finalState='LONG',
                                       sensor=price,
                                       predicate=(lambda x:
                                                  self.counter > self.L and
                                                  x > self.ma),
                                       actuator=(lambda x:
                                                 self.positions.append(self.unblockLong) or
                                                 emit(self, 'LONG') or
                                                 print("%s %s -> %s" % (self.name,
                                                                        'INIT',
                                                                        'LONG')))),
                            Transition(initialState='INIT',
                                       finalState='SHORT',
                                       sensor=price,
                                       predicate=(lambda x:
                                                  self.counter > self.L and
                                                  x <= self.ma),
                                       actuator=(lambda x:
                                                 self.positions.append(self.unblockShort) or
                                                 emit(self, 'SHORT') or
                                                 print("%s %s -> %s" % (self.name,
                                                                        'INIT',
                                                                        'SHORT')))),
                            Transition(initialState='LONG',
                                       finalState='INIT',
                                       sensor=price,
                                       predicate=(lambda x:
                                                  False),
                                       actuator=(lambda x:
                                                 False)),
                            Transition(initialState='LONG',
                                       finalState='LONG',
                                       sensor=price,
                                       predicate=(lambda x:
                                                  self.counter > self.L and
                                                  x > self.ma),
                                       actuator=(lambda x:
                                                 self.positions.append(self.unblockLong) or
                                                 print("%s %s -> %s" % (self.name,
                                                                        'LONG',
                                                                        'LONG')))),
                            Transition(initialState='LONG',
                                       finalState='SHORT',
                                       sensor=price,
                                       predicate=(lambda x:
                                                  self.counter > self.L and
                                                  x <= self.ma),
                                       actuator=(lambda x:
                                                 self.positions.append(self.unblockShort) or
                                                 emit(self, 'SHORT') or
                                                 print("%s %s -> %s" % (self.name,
                                                                        'LONG',
                                                                        'SHORT')))),
                            Transition(initialState='SHORT',
                                       finalState='INIT',
                                       sensor=price,
                                       predicate=(lambda x:
                                                  False),
                                       actuator=(lambda x:
                                                 False)),
                            Transition(initialState='SHORT',
                                       finalState='LONG',
                                       sensor=price,
                                       predicate=(lambda x:
                                                  self.counter > self.L and
                                                  x > self.ma),
                                       actuator=(lambda x:
                                                 self.positions.append(self.unblockLong) or
                                                 emit(self, 'LONG') or
                                                 print("%s %s -> %s" % (self.name,
                                                                        'SHORT',
                                                                        'LONG')))),
                            Transition(initialState='SHORT',
                                       finalState='SHORT',
                                       sensor=price,
                                       predicate=(lambda x:
                                                  self.counter > self.L and
                                                  x <= self.ma),
                                       actuator=(lambda x:
                                                 self.positions.append(self.unblockShort) or
                                                 print("%s %s -> %s" % (self.name,
                                                                        'SHORT',
                                                                        'SHORT'))))]

class TickBarGenerator(FSMAgent):
    def __init__(self,
                 name='',
                 currentState='',
                 transitions=None,
                 states=None,
                 mkt='',
                 numEvents=70,
                 counter=0,
                 buffer=None,
                 open=None,
                 high=None,
                 low=None,
                 close=None):
        FSMAgent.__init__(self, name, currentState, transitions, states)
        self.mkt = mkt
        self.numEvents = numEvents
        self.counter = counter
        self.buffer = buffer if buffer != None else []
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        if len(self.states) == 0:
            self.states.append('EMIT')
            self.name = 'TICKBARGENERATOR' + str(self.numEvents)

    def setFSM(self):
        self.currentState = self.states[-1]
        self.transitions = [Transition(initialState='CALC',
                                       finalState='CALC',
                                       sensor=price,
                                       predicate=(lambda x:
                                                  self.counter < self.numEvents),
                                       actuator=(lambda x:
                                                 self.actuator(x, 'CALC', 'CALC'))),
                            Transition(initialState='CALC',
                                       finalState='EMIT',
                                       sensor=price,
                                       predicate=(lambda x:
                                                  self.counter == self.numEvents),
                                       actuator=(lambda x:
                                                 self.actuator(x, 'CALC', 'EMIT'))),
                            Transition(initialState='EMIT',
                                       finalState='CALC',
                                       sensor=price,
                                       predicate=(lambda x:
                                                  True),
                                       actuator=(lambda x:
                                                 self.actuator(x, 'EMIT', 'CALC'))),
                            Transition(initialState='EMIT',
                                       finalState='EMIT',
                                       sensor=price,
                                       predicate=(lambda x:
                                                  False),
                                       actuator=(lambda x:
                                                 False))]

    def actuator(self, x, initialState, finalState):
        if initialState == 'CALC':
            if finalState == 'CALC':
                self.close = x
                self.high = max(self.high, x)
                self.low = min(self.low, x)
                self.buffer.append(x)
            elif finalState == 'EMIT':
                tickBar = TickBar(security=self.mkt,
                                  timestamp=self.timestamps[-1],
                                  value=[self.open, self.high, self.low, self.close],
                                  numTicks=self.numEvents)
                emit(self, tickBar)
                self.buffer = []
        elif initialState == 'EMIT':
            if finalState == 'CALC':
                self.open = x
                self.high = x
                self.low = x
                self.close = x
                self.buffer.append(x)
            elif finalState == 'EMIT':
                pass
        print("%s %s -> %s" % (self.name,
                               initialState,
                               finalState))
            
def preprocess(agent, event):
    if isinstance(agent, SimpleModel) and isinstance(event, MarketUpdate):
        agent.counter = len(agent.revalPrices)
        sublist = agent.revalPrices[agent.counter-agent.L:agent.counter]
        agent.ma = sum(sublist) / len(sublist)
    elif isinstance(agent, TickBarGenerator) and isinstance(event, MarketUpdate):
        agent.positions.append(0)
        agent.counter = len(agent.buffer)
    elif isinstance(agent, SimpleModel) and isinstance(event, Comm):
        if event.value == 'INIT':
            agent.unblockShort = 0
            agent.unblockLong = 0
        elif event.value == 'LONG':
            agent.unblockShort = -1
            agent.unblockLong = 0
        elif event.value == 'SHORT':
            agent.unblockShort = 0
            agent.unblockLong = 1

def postprocess(agent, event):
    print("agent=%s event=%s" % (type(agent), type(event)))
    if isinstance(agent, TickBarGenerator):
        print("Event %s %s consumed for agent %s" % (event.timestamp, price(event), agent.name))
        print("Output: Counter=%s, open=%f, high=%f, low=%f, close=%f" % (agent.counter, agent.open, agent.high, agent.low, agent.close))
    else:
        print("Event %s %s consumed for agent %s" % (event.timestamp, price(event), agent.name))
        print("Output: Counter=%s, ma=%s, state=%s, position=%s, pl=%s" % (agent.counter, agent.ma, agent.states[-1], agent.positions[-1], agent.pls[-1]))


def listen(context, socket):
    return (symbol, mm, bid, ask, bidSize, askSize)

def runSimulation(agents, events):
    while len(events) > 0:
        event = events.pop()
        for agent in agents:
            consume(agent, event)

def getEvents(security):
    events = []
    data = ystockquote.get_historical_prices(security, '2011-06-01', '2013-01-01')
    for date in data.keys():
        timestamp = date
        value = float(data[date]['Adj Close'])
        event = MarketUpdate(security, timestamp, value)
        events.append(event)
    return events

if __name__ == '__main__':
    
    # Data transmitted from C:\NxCore\Examples\NxCoreLanguages\C++\SampleApp4
    socket.connect("tcp://localhost:5556")
    symbolFilter = 'fCL.U13'
    socket.setsockopt(zmq.SUBSCRIBE, symbolFilter)

    while(1):
        string = socket.recv()
        data = string.split()
        symbol = data[0]
        mm = data[1]
        bid = float(data[2])
        ask = float(data[3])
        bidSize = float(data[4])
        askSize = float(data[5])

        print("%s %s %f %f %d %d" %(symbol, mm, bid, ask, bidSize, askSize))

    sec1 = 'AAPL'
    sec2 = 'DELL'

    print("Gathering events for %s ..." % sec1)
    events += getEvents(sec1)
    print("Gathering events for %s ..." % sec2)
    events += getEvents(sec2)
    # FSM reads from the end (top) of the queue, so...
    events.sort(reverse=True)

    print("Creating agents...")
    agent1 = SimpleModelComm(L=10, mkt=sec1)
    agent2 = SimpleModelComm(L=20, mkt=sec2)
    b1 = TickBarGenerator(mkt=sec1, numEvents=5)
    b2 = TickBarGenerator(mkt=sec2, numEvents=7)

    # We want agent1 to be potentially short when agent2 is long and vice versa
    agent1.recipientsList.append(agent2)
    agent2.recipientsList.append(agent1)
    # We want agent1 and agent2 to receive TickBar events from b1 and b2
    b1.recipientsList.append(agent1)
    b1.recipientsList.append(agent2)
    b2.recipientsList.append(agent1)
    b2.recipientsList.append(agent2)

    agents.append(agent1)
    agents.append(agent2)
    agents.append(b1)
    agents.append(b2)

    OPTIMIZE = False
    if (OPTIMIZE):
        agents = []
        for i in range(10, 111):
            agent = SimpleModel(L=10, mkt=sec1)
            agents.append(agent)
        runSimulation(agents, events)
        results = []
        for agent in agents:
            results.append(agent.tradestats[-1])

    start_time = time.time()

    print("Running...")
    runSimulation(agents, events)
    for pl in agent1.pls:
        print(pl)

    elapsed_time = time.time() - start_time
    print("Elapsed time = %.1f sec." % elapsed_time)
