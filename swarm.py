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
from compiler.ast import flatten

agents = []
aggregateAgents = []
swarmSet = []

from agent import *
from event import *
from tickbar import *
from simple import *

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
        trade = Trade(security = event.security,
                      timestamp=event.timestamp,
                      price=price(event,
                                  slippageFunction=algo.slippage,
                                  size=order.orderQuantity,
                                  orderType=order.orderType),
                      tradeQuantity=order.orderQuantity)
        return (trade, None)
    elif isinstance(algo, Aggressor):
        q = orderQuantity(order)
        if q > 0:
            bookSide = event.value[1]
        else:
            bookSide = event.value[0]
        sweep = liftQuotes(bookside, q, algo.maxDepth)
        trades = map(lambda x:
                     Trade(security = event.security,
                           timestamp=event.timestamp,
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
    def __init__(self, security, timestamp, price, tradeQuantity):
        self.security = security
        self.timestamp = timestamp
        self.price = price
        self.tradeQuantity = tradeQuantity

def consume(agent, event):
    if observe(agent, event):
        updateBefore(agent, event)
        update(agent, event)
        updateAfter(agent, event)

#------------------------------------------------------
# Return true if agent observes event, false otherwise
#------------------------------------------------------
def observe(agent, event):
    if isinstance(agent, SimpleModel) and isinstance(event, list):
        return (agent.mkt == event[-1].security)
    elif isinstance(agent, SimpleModel) and isinstance(event, Comm):
        return (agent in event.recipients) and (agent != event.originator)
    elif isinstance(agent, TickBarGenerator) and isinstance(event, MarketUpdate):
        return (agent.mkt == event.security) and (not isinstance(event, Bar))
    elif isinstance(agent, AggregateAgent):
        return True
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
        # Calculate moving average
        agent.counter = len(agent.revalPrices)
        sublist = agent.revalPrices[agent.counter-agent.L:agent.counter]
        arr = np.array(sublist)[:,3]
        agent.mas.append(sum(arr) / len(arr))
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
    if isinstance(agent, AggregateAgent):
        pass
    elif isinstance(event, list):
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
        
        # Always append to P&L list at every tick
        agent.pls.append(pl)
        cumPl = sum(agent.pls)
        agent.cumPls.append(cumPl)

        # Only append to trade P&L list if in fact there was a trade
        # (i.e. prevPos != lastPos), not simply at every tick
        if prevPos != lastPos:
            agent.tradePls.append(pl)
            cumTradePl = sum(agent.tradePls)
            agent.cumTradePls.append(cumTradePl)

        # Send a trade if position has changed
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
        if len(agent.cumTradePls) > 0:
            cumtradepl = agent.cumTradePls[-1]
        else:
            cumtradepl = 0
        print("SimpleModel Output: Counter=%s, ma=%s, state=%s, position=%s, cumpl=%s, cumtradepl=%s" % (agent.counter, agent.mas[-1], agent.states[-1], agent.positions[-1], agent.cumPls[-1], cumtradepl))

#------------------------------------
# Rolling trade NAV fitness function
#------------------------------------
def RTNAV(agent, L, alpha, lamda):
    return 1

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
#    for trade in trades:
#        trade.price
#        trade.quantity
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
    if isinstance(agent, AggregateAgent):
        allOrders = []
        allPositions = []
        for member in agent.members:
            consume(member, event)
            allOrders = zip(allOrders, member.orders)
            allPositions = zip(allPositions, member.positions)
        agent.orders = [flatten(i) for i in allOrders]
        agent.positions = [sum(flatten(i)) for i in allPositions]
    elif isinstance(event, MarketUpdate) or isinstance(event, list):
        #agent.setFSM()
        operateFSM(agent, event)
        agent.states.append(agent.currentState)
        #print("Completed work for %s and new state %s added" % (agent.name, agent.currentState))
    elif isinstance(event, Comm):
        #agent.setFSM()
        print("setFSM completed for agent %s" % agent.name)
        print("Completed work for %s and comm event %s" % (agent.name, event))

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

def getNextMarketUpdateEvent(security):
    socket.setsockopt(zmq.SUBSCRIBE, security)
    
    try:
        string = socket.recv(zmq.NOBLOCK)
    except zmq.ZMQError as e:
        # Try again after 1 sec; if still no answer, exit
        time.sleep(1)
        try:
            string = socket.recv(zmq.NOBLOCK)
        except zmq.ZMQError as e:
            return None

    data = string.split()
    symbol = data[0]
    timestamp = data[1]
    tradePrice = float(data[2])
    tradeNetChange = float(data[3])
    tradeSize = float(data[4])
#    print("%s %s %f %f %d" %(symbol, timestamp, tradePrice, tradeNetChange, tradeSize))
    event = MarketUpdate(symbol, timestamp, tradePrice)
    return event

#------------------------------------------
# Create a swarm of SimpleModelComm agents 
#------------------------------------------
def createSwarmSet(Llist, security, fitnessFunction=RTNAV, fitnessOn=0, fitnessOff=0):
    swarm = []
    for L in Llist:
        agent = SimpleModelComm(L=L, mkt=security)
        swarm.append([agent, fitnessFunction, fitnessOn, fitnessOff])
    return swarm

#-----------------------
# Run a swarm of agents
#-----------------------
def runSwarm(securities, swarmFF, swarmType='ADD'):
    while len(eventQueue) > 0:
        # First event is a MarketUpdate event; since there is only one agent
        # (the TickBarGenerator), use this event to  generate TickBar and Comm events
        event = eventQueue.pop()
        for agent in agents:
            consume(agent, event)
        # The TickBar and Comm events are then consumed by the aggregateAgents (there is only
        # one AggregateAgent, the swarmAgent, which is itself a )
        while len(eventQueue) > 0:
            event = eventQueue.pop()
            for aggregateAgent in aggregateAgents:
                consume(aggregateAgent, event)
        # Get next MarketUpdate event and add to queue, to be consumed by the
        # TickBarGenerator in the next round
        for security in securities:
            event = getNextMarketUpdateEvent(security)
            if event != None:
                eventQueue.append(event)

# Socket to talk to server
context = zmq.Context()
socket = context.socket(zmq.SUB)

#----------------------------------------------------
# Plot the generated tickBars, the ma and the trades
#----------------------------------------------------
def plot(tickBarsPlot, ma, trades, candle=False):
    # Create a numpy array for more manipulation flexibility
    data = np.array(tickBarsPlot)
    # This replaces the first column with numbers 1..size
    data2 = np.hstack([np.arange(data[:,0].size)[:, np.newaxis], data[:,1:]])
    # Ticks are on the hour
    ticks = np.unique(np.trunc(data[:,0] / 10000), return_index=True)
    fig = plt.figure(figsize=(10, 5))
    # These numbers define margin from window edge
    ax = fig.add_axes([0.1, 0.2, 0.85, 0.7])
    ax.set_xticks(data2[ticks[1], 0])
    ax.set_xticklabels(data[ticks[1], 0], rotation=90)

    # Plot prices
    if candle:
        candlestick(ax, data2, width=0.5, colorup='g', colordown='r')
    else:
        ax.plot(data[:,4])

    # Plot moving average
    ax.plot(ma)
    
    # Plot trades
    buys = []
    sells = []
    for trade in trades:
        timestampFloat = float(trade.timestamp[0:2]) * 10000 + float(trade.timestamp[3:5]) * 100 + float(trade.timestamp[6:8])
        if trade.tradeQuantity > 0:
            buys.append(timestampFloat)
        else:
            sells.append(timestampFloat)
    # Find index of buys in the data array before plotting
    ix = np.where(np.in1d(data[:,0], buys))[0]
    ax.plot(ix, data[:,4][ix], 'g^')
    # Find index of sells in the data array before plotting
    ix = np.where(np.in1d(data[:,0], sells))[0]
    ax.plot(ix, data[:,4][ix], 'rv')
    plt.show()

if __name__ == '__main__':
    
    # Data transmitted from C:\NxCore\Examples\NxCoreLanguages\C++\SampleApp4
    socket.connect("tcp://localhost:5556")
    securities = ('fES.U13')

    print("Creating swarm of agents...")
    # Create a moving average simple model swarm using 70-tick bars
    Llist = [400] #np.arange(100, 401, 100)  # i.e. from 100 to 400 by 100
    tickBarsAgent = {}
    swarmAgent = AggregateAgent(name='SwarmAgent')
    for security in securities:
        tickBarsAgent[security] = TickBarGenerator(mkt=security, numEvents=70)
        tickBarsAgent[security].setFSM()
        agents.append(tickBarsAgent[security])
        for L in Llist:
            agent = SimpleModelComm(L=L, mkt=security)
            agent.setFSM()
            swarmAgent.members.append(agent)
            tickBarsAgent[security].recipientsList.append(swarmAgent)

    #for triple in swarmSet:
    #    swarmAgent.members.append(triple[0])
    #    swarmAgent.members[-1].setFSM()
    aggregateAgents.append(swarmAgent)

    # We want the swarm agents to receive TickBar events from this tickbarAgent (i.e. this security)
    # (each agent only has one data input; if we want the same strategy applied to a different
    # security, we should define a different agent of the same type (e.g. SimpleModelComm)
    #for triple in swarmSet:
    #    tb1.recipientsList.append(triple[0])

    # List of agents only includes TickBarGenerator, all others are in the aggregate swarm agent
    #agents.append(tb1)

    OPTIMIZE = False
    if (OPTIMIZE):
        agents = []
        for i in range(10, 111):
            agent = SimpleModel(L=10, mkt=securities[0])
            agents.append(agent)
        runSimulation(agents, eventQueue)
        results = []
        for agent in agents:
            results.append(agent.tradestats[-1])

    start_time = time.time()

    print("Running...")
    for security in securities:
        event = getNextMarketUpdateEvent(security)
        eventQueue.append(event)
    runSwarm(securities, RTNAV, 'ADD')
    print("Plotting...")
    for security in securities:
        plot(tickBarsAgent[security].tickBarsPlot, swarmSet[0][0].mas, swarmSet[0][0].trades)
    for triple in swarmSet:
        for cumpl in triple[0].cumPls:
            print("%.2f" % cumpl),

    elapsed_time = time.time() - start_time
    print("Elapsed time = %.1f sec." % elapsed_time)
