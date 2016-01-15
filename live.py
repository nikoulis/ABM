# This is simply so that 'print' is an expression (like in Python 3)
# instead of a function and therefore can be used in the lambdas below.
from __future__ import print_function

import ystockquote
import time
import pdb
import sys
import zmq
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.finance import candlestick

agents = []
aggregateAgents = []

from agent import *
from event import *

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

class Trade(object):
    def __init__(self, timestamp, price, tradeQuantity):
        self.timestamp = timestamp
        self.price = price
        self.tradeQuantity = tradeQuantity

def consume(agent, event):
    if observe(agent, event):
        updateBefore(agent, event)
        update(agent, event)
        updateAfter(agent, event)

def observe(agent, event):
    if isinstance(agent, SimpleModel) and isinstance(event, list):
        return (agent.mkt == event[-1].security)
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
        #print("updateBefore completed for agent %s and MktUpdate event %s" % (agent.name, event.timestamp))
    if isinstance(event, list):
        if agent.timestamps == []:
            agent.pls.append(0)
            agent.fitnesses.append(0)
        agent.timestamps.append(event[-1].timestamp)
        agent.revalPrices.append(price(event[-1]))
        oms(agent, event[-1], 'ALL')
        #print("updateBefore completed for agent %s and list event %s" % (agent.name, event[-1].timestamp))
    elif isinstance(event, Comm):
        agent.incomingMessages.append(event)
        print("updateBefore completed for agent %s and Comm event %s" % (agent.name, event))    
    preprocess(agent, event)

def preprocess(agent, event):
    if isinstance(agent, SimpleModel) and isinstance(event, list):
        agent.counter = len(agent.revalPrices)
        sublist = agent.revalPrices[agent.counter-agent.L:agent.counter]
        arr = np.array(sublist)[:,3]
        agent.ma = sum(arr) / len(arr)
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

def updateAfter(agent, event):
    if isinstance(event, list):
        l = len(agent.timestamps)
        lastPos = agent.positions[-1]
        if l < 2:
            prevPos = 0
        else:
            prevPos = agent.positions[-2]
        tradeQuantity = lastPos - prevPos
        if len(agent.revalPrices) == 0:
            lastPrice = 0
        else:
            lastPrice = agent.revalPrices[-1][3]
        if l < 2:
            prevPrice = 0
            pl = 0
        else:
            prevPrice = agent.revalPrices[-2][3]
            pl = prevPos * (lastPrice - prevPrice)
        agent.pls.append(pl)
        if tradeQuantity != 0:
            sendOrder(agent, event[-1],
                      orderPrice=price(event[-1].value[3]),
                      orderQuantity=tradeQuantity,
                      orderType='STP',
                      orderId='POSCHG')
            print("Generated aggressive order for agent %s for quantity %s" % (agent.name, tradeQuantity))
        #print("updateAfter completed for agent %s and MktUpdate event %s" % (agent.name, event.timestamp))
    elif isinstance(event, Comm):
        print("updateAfter completed for agent %s and Comm event %s" % (agent.name, event))
    postprocess(agent, event)

def postprocess(agent, event):
    if isinstance(agent, TickBarGenerator):
        pass
    elif isinstance(agent, SimpleModel):
        print("SimpleModel Event %s %s consumed for agent %s" % (event[-1].timestamp, price(event)[-1].value, agent.name))
        print("SimpleModel Output: Counter=%s, ma=%s, state=%s, position=%s, cumpl=%s" % (agent.counter, agent.ma, agent.states[-1], agent.positions[-1], sum(agent.pls)))

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

def clusterAgents(agents, stat, numBins):
    def getStats(agent):
        ts = agent.tradestats
        if stat == 'tpl':
            return stat

def perform(transition, event):
    transition.effected = transition.predicate(transition.sensor(event))
    return transition.effected

def operateFSM(fsm, event):
    applicableTransitions = [tr for tr in fsm.transitions if tr.initialState == fsm.currentState]
    effectedTransition = [tr for tr in applicableTransitions if perform(tr, event)][0]
    effectedTransition.actuator(effectedTransition.sensor(event))
    fsm.currentState = effectedTransition.finalState
    #print("Transition: %s -> %s"% (effectedTransition.initialState, effectedTransition.finalState))

def update(agent, event):
    if isinstance(event, MarketUpdate) or isinstance(event, list):
        agent.setFSM()
        #print("setFSM completed for agent %s" % agent.name)
        operateFSM(agent, event)
        #print("operateFSM completed for agent %s" % agent.name)
        agent.states.append(agent.currentState)
        #print("Completed work for %s and new state %s added" % (agent.name, agent.currentState))
    elif isinstance(event, Comm):
        agent.setFSM()
        print("setFSM completed for agent %s" % agent.name)
        print("Completed work for %s and comm event %s" % (agent.name, event))

from tickbar import *
from simple import *

def runSimulation():
    while len(eventQueue) > 0:
        event = eventQueue.pop()
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

def getNextEvent(security):
    socket.setsockopt(zmq.SUBSCRIBE, security)
    string = socket.recv()
    data = string.split()
    symbol = data[0]
    timestamp = data[1]
    tradePrice = float(data[2])
    tradeNetChange = float(data[3])
    tradeSize = float(data[4])
#    print("%s %s %f %f %d" %(symbol, timestamp, tradePrice, tradeNetChange, tradeSize))
    event = MarketUpdate(symbol, timestamp, tradePrice)
    return event

def runLive(security):
    while len(eventQueue) > 0:
        event = eventQueue.pop()
        for agent in agents:
            consume(agent, event)
        # Consume all events produced by the agents so far (e.g. Comm or TickBar events)
        while len(eventQueue) > 0:
            event = eventQueue.pop()
            for agent in agents:
                consume(agent, event)
        # Then, consume all new events (MarketUpdate only at this point)
        event = getNextEvent(security)
        eventQueue.append(event)

#    i = 0
#    while (i < 1000):
#        event = getNextEvent(security)
#        for agent in agents:
#            consume(agent, event)

# Socket to talk to server
context = zmq.Context()
socket = context.socket(zmq.SUB)
    
if __name__ == '__main__':

    # Data transmitted from C:\NxCore\Examples\NxCoreLanguages\C++\SampleApp4
    socket.connect("tcp://localhost:5556")
    security = 'fES.U13'

    print("Creating agents...")
    # Create a 20-day moving averahe simple model using 70-tick bars
    agent1 = SimpleModelComm(L=20, mkt=security)
    b1 = TickBarGenerator(mkt=security, numEvents=70)

    # We want agent1 to receive TickBar events from b1
    b1.recipientsList.append(agent1)

    agents.append(agent1)
    agents.append(b1)

    OPTIMIZE = False
    if (OPTIMIZE):
        agents = []
        for i in range(10, 111):
            agent = SimpleModel(L=10, mkt=sec1)
            agents.append(agent)
        runSimulation(agents, eventQueue)
        results = []
        for agent in agents:
            results.append(agent.tradestats[-1])

    start_time = time.time()

    print("Running...")
    event = getNextEvent(security)
    eventQueue.append(event)
    runLive(security)
    for pl in agent1.pls:
        print(pl)

    elapsed_time = time.time() - start_time
    print("Elapsed time = %.1f sec." % elapsed_time)
