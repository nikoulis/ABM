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
from pprint import pprint
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
                      quantity=order.orderQuantity)
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
            pprint(agent.tradestats[-1].__dict__)
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
    def __init__(self, security, timestamp, price, quantity):
        self.security = security
        self.timestamp = timestamp
        self.price = price
        self.quantity = quantity

#------------------
# Consume an event
#------------------
def consume(agent, event):
    if observe(agent, event):
        preprocess(agent, event)
        update(agent, event)

#-----------------------------------
# Determine if agent observes event
#-----------------------------------
def observe(agent, event):
    if isinstance(agent, TickBarGenerator) and isinstance(event, MarketUpdate):
        # 1. TickBarGenerator agent, MarketUpdate event
        #-----------------------------------------------
        return (agent.mkt == event.security) and (not isinstance(event, Bar))
    elif isinstance(agent, SimpleModel) and isinstance(event, list):
        # 2. SimpleModel agent, TickBar event
        #-------------------------------------
        return (agent.mkt == event[-1].security)
    elif isinstance(agent, SimpleModel) and isinstance(event, Comm):
        # 3. SimpleModel agent, Comm event
        #-----------------------------------
        return (agent in event.recipients) and (agent != event.originator)
    else:
        return False

#-------------------------------------
# Update before main event processing
#-------------------------------------
def preprocess(agent, event):
    if isinstance(agent, TickBarGenerator) and isinstance(event, MarketUpdate):
        # 1. TickBarGenerator agent, MarketUpdate event
        #-----------------------------------------------
        agent.timestamps.append(event.timestamp)
        agent.revalPrices.append(price(event))
        #print("preprocess completed for agent %s and MktUpdate event %s" % (agent.name, event.timestamp))
        agent.counter = len(agent.buffer)
    if isinstance(agent, SimpleModel) and isinstance(event, list):
        # 2. SimpleModel agent, TickBar event
        #-------------------------------------
        agent.timestamps.append(event[-1].timestamp)
        agent.revalPrices.append(price(event[-1]))
        oms(agent, event[-1], 'ALL')
        #print("preprocess completed for agent %s and list event %s" % (agent.name, event[-1].timestamp))
        agent.counter = len(agent.revalPrices)
        startIndex = max(0, agent.counter - agent.L)
        sublist = agent.revalPrices[startIndex:agent.counter]
        arr = np.array(sublist)[:,3]
        agent.mas.append(sum(arr) / len(arr))
    elif isinstance(agent, SimpleModel) and isinstance(event, Comm):
        # 3. SimpleModel agent, Comm event
        #-----------------------------------
        agent.incomingMessages.append(event)
        #print("preprocess completed for agent %s and Comm event %s" % (agent.name, event))
        if event.value == 'INIT':
            agent.unblockShort = 0
            agent.unblockLong = 0
        elif event.value == 'LONG':
            agent.unblockShort = -1
            agent.unblockLong = 0
        elif event.value == 'SHORT':
            agent.unblockShort = 0
            agent.unblockLong = 1

#-----------------
# Main processing
#-----------------
def update(agent, event):
    if isinstance(agent, TickBarGenerator) and isinstance(event, MarketUpdate):
        # 1. TickBarGenerator agent, MarketUpdate event
        #-----------------------------------------------
        operateFSM(agent.fsm, event)
        agent.states.append(agent.fsm.currentState)
        #print("Completed work for %s and new state %s added" % (agent.name, agent.fsm.currentState))
    elif isinstance(agent, SimpleModel) and isinstance(event, list):
        # 2. SimpleModel agent, TickBar event
        #-------------------------------------
        operateFSM(agent.fsm, event)
        agent.states.append(agent.fsm.currentState)
        #print("Completed work for %s and new state %s added" % (agent.name, agent.fsm.currentState))
        checkAndPlaceTrade(agent, event)

#-----------------------------------
# Check if we need to place a trade
#-----------------------------------
def checkAndPlaceTrade(agent, event):
    eventLength = len(event)
    lastPos = agent.positions[-1]
    if eventLength < 2:
        prevPos = 0
        quantity = 0
    else:
        prevPos = agent.positions[-2]
        quantity = lastPos - prevPos
    if len(agent.revalPrices) == 0:
        lastPrice = 0
    else:
        lastPrice = agent.revalPrices[-1][3]
    if eventLength < 2:
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
    if quantity != 0:
        sendOrder(agent, event[-1],
                  orderPrice=price(event[-1].value[3]),
                  orderQuantity=quantity,
                  orderType='STP',
                  orderId='POSCHG')
        print("Generated aggressive order for agent %s for quantity %s" % (agent.name, quantity))
        print("SimpleModel Event %s %s consumed for agent %s" % (event[-1].timestamp, price(event)[-1].value, agent.name))
    if len(agent.cumTradePls) > 0:
        cumtradepl = agent.cumTradePls[-1]
    else:
        cumtradepl = 0
    print("SimpleModel Output: Counter=%s, ma=%s, state=%s, position=%s, cumpl=%s, cumtradepl=%s" %
          (agent.counter, agent.mas[-1], agent.states[-1], agent.positions[-1], agent.cumPls[-1], cumtradepl))

#------------------------------------
# Rolling trade NAV fitness function
#------------------------------------
def RTNAV(agent, L, alpha, lamda):
    return 1

#--------------------------------
# Convert a timestamp to seconds
#--------------------------------
def ts2sec(ts):
    return 3600 * float(ts[0:2]) + 60 * float(ts[3:5]) + float(ts[6:8])

class TradeStats:
    percentProfitable = 0
    winToLoss = 0
    avgLogRet = 0
    totPl = 0
    avgDuration = 0
    posPl = 0
    negPl = 0
    profitFactor = 0

def computeStats(trades):
    stats = TradeStats()
    # Crate trade pairs list (may contain dummy trades to correctly book P&L)
    tradeGroups = []
    tradeGroup = []
    oldPos = 0
    numTrades = len(trades)
    i = 0
    for trade in trades:
        newPos = oldPos + trade.quantity
        if newPos == 0:
            # This ends the group, e.g. if previous position was +3 and now it's flat
            tradeGroup.append(trade)
            tradeGroups.append(tradeGroup)
            tradeGroup = []
        elif oldPos * newPos < 0:
            # E.g. if oldPos=+3 and newPos=-7, then dummyTrade1=-3 and dummyTrade2=-7
            # Only create dummy trades if position switches from long to short or vice versa,
            # otherwise (e.g. if position goes from +3 to +1) wait for it to switch sign
            # before booking P&L
            dummyTrade1 = Trade(security = trade.security,
                                 timestamp=trade.timestamp,
                                 price=trade.price,
                                 quantity=-oldPos)
            dummyTrade2 = Trade(security = trade.security,
                                 timestamp=trade.timestamp,
                                 price=trade.price,
                                 quantity=newPos)
            tradeGroup.append(dummyTrade1)
            tradeGroups.append(tradeGroup)
            tradeGroup = []
            tradeGroup.append(dummyTrade2)
        else:
            # Keep adding trades to tradeGroup until position is zero (or until it switches sign)
            # E.g. trade1=+1, trade2=+3, trade3=+5, trade4=-4, trade5=-2 are all added to tradeGroup
            tradeGroup.append(trade)

        if i == numTrades - 1:
            # Close position at the end of the loop
            dummyTrade2 = Trade(security = trade.security,
                                 timestamp=trade.timestamp,
                                 price=trade.price,
                                 quantity=-newPos)
            tradeGroup.append(dummyTrade2)
            tradeGroups.append(tradeGroup)

        oldPos = newPos
        i += 1

    # Calculate trade stats
    numTradeGroups = len(tradeGroups)
    tradeStats = []
    for tradeGroup in tradeGroups:
        buyTrades = [trade for trade in tradeGroup if trade.quantity > 0]
        sellTrades = [trade for trade in tradeGroup if trade.quantity < 0]
        numBuyTrades = len(buyTrades) if len(buyTrades) > 0 else 1
        numSellTrades = len(sellTrades) if len(sellTrades) > 0 else 1
        avgBuyPrice = sum([trade.price[3] for trade in buyTrades]) / numBuyTrades
        avgSellPrice = sum([trade.price[3] for trade in sellTrades]) / numSellTrades
        avgBuyTime = sum([ts2sec(trade.timestamp) for trade in buyTrades]) / numBuyTrades
        avgSellTime = sum([ts2sec(trade.timestamp) for trade in sellTrades]) / numSellTrades
        tradeLength = abs(avgBuyTime - avgSellTime)
        if avgBuyPrice == 0:
            tradeLogRet = 1
        else:
            tradeLogRet = avgSellPrice / avgBuyPrice
        tradePl = sum([-trade.quantity * trade.price[3] for trade in tradeGroup])
        tradeStats.append([tradeLength, tradeLogRet, tradePl])

    stats.percentProfitable = sum([1.0 if t[2] >= 0 else 0.0 for t in tradeStats]) / numTradeGroups
    stats.posPl = sum([t[2] if t[2] >= 0 else 0 for t in tradeStats])
    stats.negPl = sum([t[2] if t[2] < 0 else 0 for t in tradeStats])
    if stats.negPl == 0:
        stats.winToLoss = 100
    else:
        stats.winToLoss = stats.posPl / (-stats.negPl)
    stats.avgLogRet = sum([t[1] for t in tradeStats]) / numTradeGroups
    stats.totPl = sum([t[2] for t in tradeStats])
    stats.avgDuration = sum([t[0] for t in tradeStats]) / numTradeGroups
    if stats.posPl + stats.negPl == 0:
        stats.profitFactor = 0
    else:
        # Note that negPl is a negative number
        stats.profitFactor = (stats.posPl + stats.negPl) / (stats.posPl - stats.negPl)
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
    effectedTransitions = [tr for tr in applicableTransitions if perform(tr, event)]
    if len(effectedTransitions) > 0:
        effectedTransition = effectedTransitions[0]
        effectedTransition.actuator(effectedTransition.sensor(event))
        fsm.currentState = effectedTransition.finalState
        #print("Transition: %s -> %s"% (effectedTransition.initialState, effectedTransition.finalState))

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
        events.insert(0, event)
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

#---------------------
# Display event queue
#---------------------
def displayQueue(eventQueue):
    print('Queue (' + str(len(eventQueue)) + '): ', end='')
    for event in eventQueue:
        if isinstance(event, MarketUpdate):
            print('M', end='')
        elif isinstance(event, list) or isinstance(event, float):
            print(event, end='')
        elif isinstance(event, Comm):
            print('C', end='')
        elif isinstance(event, Prc):
            pint('P', end='')
        elif isinstance(event, Bar) or isinstance(event, TickBar):
            print(event.open, end='')
        elif isinstance(event, Book):
            print('B', end='')
    print()
    
#-----------------------
# Run a swarm of agents
#-----------------------
def runSwarm(aggregateAgents, tickBarAgents, securities, swarmFF, swarmType='ADD'):
    while len(eventQueue) > 0:
        # Consume current market update events
        while len(eventQueue) > 0:
            event = eventQueue.pop()
            # TickBar agents consume MarketUpdate events and produce TickBar and Comm events
            for tickBarAgent in tickBarAgents:
                consume(tickBarAgent, event)
            # Aggregate agents consume TickBar and Comm events
            for aggregateAgent in aggregateAgents:
                for agent in aggregateAgent.members:
                    consume(agent, event)

        # Get more market update events
        for security in securities:
            event = getNextMarketUpdateEvent(security)
            if event != None:
                eventQueue.insert(0, event)

#----------------------------------------------------
# Plot the generated tickBars, the ma and the trades
#----------------------------------------------------
def plot(tickBarsPlot, ma, trades, candle=False):
    # Create a numpy array for more manipulation flexibility
    data = np.array(tickBarsPlot)
    # This replaces the first column with numbers 1..size
    data2 = np.hstack([np.arange(data[:,0].size)[:, np.newaxis], data[:,1:]])
    # Chart ticks are on the hour
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
        if trade.quantity > 0:
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
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://localhost:5556")

    securities = ['fES.U13', 'f6E.U13']

    # There are as many tickBarAgents as number of securities
    tickBarAgents = []

    # There are as many swarmAgents as number of securities;
    # each swarmAgent contains agents for each MA scenario in Llist
    swarmAgents = []

    # For each security, create a moving average simple model swarm agent using 70-tick bars
    print("Creating set of swarm agents (one swarm agent per security) ...")
    Llist = range(100, 400, 100)
    for security in securities:
        tickBarAgent = TickBarGenerator(mkt=security, numEvents=70)
        tickBarAgent.setFSM()
        tickBarAgents.append(tickBarAgent)
        swarmAgent = AggregateAgent(name='SwarmAgent')
        for L in Llist:
            agent = SimpleModelComm(L=L, mkt=security)
            agent.setFSM()
            swarmAgent.members.append(agent)
            tickBarAgent.recipientsList.append(agent)
        swarmAgents.append(swarmAgent)

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
    # Get first MarketUpdate events and add them to event queue (to kickstart the event loop)
    for security in securities:
        event = getNextMarketUpdateEvent(security)
        if event != None:
            eventQueue.insert(0, event)
    runSwarm(swarmAgents, tickBarAgents, securities, RTNAV, 'ADD')
    for swarmAgent in swarmAgents:
        for j in range(len(Llist)):
            pprint(swarmAgent.members[j].tradestats[-1].__dict__)

    print("Plotting...")
    pdb.set_trace()
    for i, tickBarAgent in enumerate(tickBarAgents):
        for j in range(len(Llist)):
            plot(tickBarAgent.tickBarsPlot, swarmAgents[i].members[j].mas, swarmAgents[i].members[j].trades)

    elapsed_time = time.time() - start_time
    print("Elapsed time = %.1f sec." % elapsed_time)
