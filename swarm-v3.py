# This is simply so that 'print' is an expression (like in Python 3)
# instead of a function and therefore can be used in the lambdas below.
#from __future__ import print_function

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
from readybar import *
from simple import *
from breakout import *

eof = False

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

#========================================
# Execute an order based on an algorithm
#========================================
def executeOrder(algo, order, event):
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

#=============================
# Return True if algoInstance
#=============================
def algoCategorize(algoInstance, algoCategory):
    if algoCategory == 'ALL':
        return True
    else:
        return algoInstance == algoCategory

#=========================
# Order management system
#=========================
def oms(agent, event, algoCategory):
    categoryOrders = [o for o in agent.orders if algoCategorize(o.algoInstance, algoCategory)]
    nonCategoryOrders = [o for o in agent.orders if not algoCategorize(o.algoInstance, algoCategory)]
    newCategoryOrders = []
    for order in categoryOrders:
        if order.security == event.security:
            (executions, remainingOrder) = executeOrder(order.algoInstance, order, event)
            agent.trades.append(executions)
            agent.tradestats.append(calcStats(agent.trades))
            #print('numTrades=', len(agent.trades))
            #pprint(agent.tradestats[-1].__dict__)
            if remainingOrder != None:
                newCategoryOrders.append(remainingOrder)
    # Replace agent.orders with non-category orders + remaining orders
    agent.orders = nonCategoryOrders + newCategoryOrders

class Algo(object):
    def __init__(self, algoType):
        self.algoType = algoType   # 'PASSIVE' or 'AGGRESSIVE'

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
        self.timestamp = timestamp   # Format: dd-hh:mm:ss
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
    if type(agent) == TickBarAgent and type(event) == MarketUpdate:
        return (agent.mkt == event.security and type(event) != Bar) or event.security == True
    elif (type(agent) == SimpleModelAgentComm or
          type(agent) == BreakoutAgent or
          type(agent) == ReadyBarAgent):
        if type(event) == list:
            return (agent.mkt == event[-1].security)
        elif type(event) == Comm:
            return (agent in event.recipients) and (agent != event.originator)
        else:
            return False
    elif event == True:
        return True
    else:
        return False

#-------------------------------------
# Update before main event processing
#-------------------------------------
def preprocess(agent, event):
    if type(agent) == TickBarAgent and type(event) == MarketUpdate:
        agent.timestamps.append(event.timestamp)
        agent.prices.append(price(event))
        agent.counter = len(agent.buffer)
    elif type(agent) == ReadyBarAgent:
        # Buy at the end of this bar, not the next bar; see update function
        pass
    elif type(agent) == SimpleModelAgentComm:
        if type(event) == list:
            agent.timestamps.append(event[-1].timestamp)
            agent.prices.append(price(event[-1]))
            oms(agent, event[-1], 'ALL')
            agent.counter = len(agent.prices)
            #startIndex = max(0, agent.counter - agent.L)
            # Assume agent.L = 100 to calculate MA; not needed for BreakoutAgent
            startIndex = max(0, agent.counter - 100)
            sublist = agent.prices[startIndex:agent.counter]
            arr = np.array(sublist)[:,3]
            agent.mas.append(sum(arr) / len(arr))
        elif type(event) == Comm:
            agent.incomingMessages.append(event)
            if event.value == 'INIT':
                agent.unblockShort = 0
                agent.unblockLong = 0
            elif event.value == 'LONG':
                agent.unblockShort = -1
                agent.unblockLong = 0
            elif event.value == 'SHORT':
                agent.unblockShort = 0
                agent.unblockLong = 1
    elif type(agent) == BreakoutAgent:
        # Updating prices and timestamps is done in update function
        pass
    
#-----------------
# Main processing
#-----------------
def update(agent, event):
    agent.execute(event)
    agent.states.append(agent.currentState)
    # For breakout, buy at the end of this bar, not next bar (see preprocess)
    if type(agent) == BreakoutAgent:
        agent.timestamps.append(event[-1].timestamp)
        agent.prices.append(event[-1].value)
    if (type(agent) == SimpleModelAgentComm or
        type(agent) == BreakoutAgent):
        checkAndPlaceOrder(agent, event)

#------------------------------------
# Check if we need to place an order
#------------------------------------
def checkAndPlaceOrder(agent, event):
    pricesLength = len(agent.prices)
    if type(agent) == BreakoutAgent:
        pricesLength += agent.yDayNumBars
    lastPos = agent.positions[-1]
    if pricesLength == 0:
        lastPrice = 0
    else:
        # Do the trade at the high for Breakout strategy, otherwise at the close
        if type(agent) == BreakoutAgent:
            lastPrice = agent.prices[-1][1]
        else:
            lastPrice = agent.prices[-1][3]
    if pricesLength < 2:
        prevPos = 0
        quantity = 0
        prevPrice = 0
        pl = 0
    else:
        prevPos = agent.positions[-2]
        quantity = lastPos - prevPos
        # Do the trade at the high for Breakout strategy, otherwise at the close
        if type(agent) == BreakoutAgent:
            if len(agent.prices) > 1:
                prevPrice = agent.prices[-2][1]
            else:
                prevPrice = agent.yDayHigh[-1]
        else:
            prevPrice = agent.prices[-2][3]
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
        # Do the trade at the high for Breakout strategy, otherwise at the close
        if type(agent) == BreakoutAgent:
            orderPrice = price(event[-1].value[1])
        else:
            orderPrice = price(event[-1].value[3])
        sendOrder(agent, event[-1],
                  orderPrice=orderPrice,
                  orderQuantity=quantity,
                  orderType='LMT',
                  orderId='')
        print '%s: %s Trade %2d at %s' % (agent.name, event[-1].timestamp, quantity, orderPrice)
        #print("SimpleModelAgent Event %s %s consumed for agent %s" % (event[-1].timestamp, price(event)[-1].value, agent.name))
    if len(agent.cumTradePls) > 0:
        cumtradepl = agent.cumTradePls[-1]
    else:
        cumtradepl = 0
    #print("%s: Counter=%s, ma=%s, state=%s, position=%s, cumpl=%s, cumtradepl=%s" %
    #      (agent.name, agent.counter, agent.mas[-1], agent.states[-1], agent.positions[-1], agent.cumPls[-1], cumtradepl))

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

#------------------
# Trade statistics
#------------------
class TradeStats:
    numTrades = 0
    percentProfitable = 0
    winToLoss = 0
    avgRet = 0
    avgDuration = 0
    totPl = 0
    posPl = 0
    negPl = 0
    profitFactor = 0

#-----------------------
# Calculate trade stats
#-----------------------
def calcStats(trades):
    stats = TradeStats()
    # Crate trade pairs list (may contain dummy trades to correctly book P&L)
    tradeGroups = []
    tradeGroup = []
    oldPos = 0
    stats.numTrades = len(trades)
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

        if i == stats.numTrades - 1:
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
            tradeRet = 0
        else:
            tradeRet = avgSellPrice / avgBuyPrice - 1
        tradePl = sum([-trade.quantity * trade.price[3] for trade in tradeGroup])
        tradeStats.append([tradeLength, tradeRet, tradePl])

    stats.percentProfitable = sum([1.0 if t[2] >= 0 else 0.0 for t in tradeStats]) / numTradeGroups
    stats.posPl = sum([t[2] if t[2] >= 0 else 0 for t in tradeStats])
    stats.negPl = sum([t[2] if t[2] < 0 else 0 for t in tradeStats])
    if stats.negPl == 0:
        stats.winToLoss = 100
    else:
        stats.winToLoss = stats.posPl / (-stats.negPl)
    stats.avgRet = sum([t[1] for t in tradeStats]) / numTradeGroups
    stats.totPl = sum([t[2] for t in tradeStats])
    stats.avgDuration = sum([t[0] for t in tradeStats]) / numTradeGroups
    if stats.posPl + stats.negPl == 0:
        stats.profitFactor = 0
    else:
        # Note that negPl is a negative number
        stats.profitFactor = (stats.posPl + stats.negPl) / (stats.posPl - stats.negPl)
    return (stats)

#==================
# Show trade stats
#==================
def showStats(stats):
    print 'Number of Trades:       %8d'     % stats.numTrades
    print 'Total P&L ($):          %8.4f'   % stats.posPl
    print 'Positive P&L ($):       %8.4f'   % stats.posPl
    print 'Negative P&L $):        %8.4f'   % stats.negPl
    print 'Win to Loss Ratio:      %8.4f'   % stats.winToLoss
    print 'Profit Factor:          %8.5f'   % stats.profitFactor
    print 'Return per Trade:      %8.4f%%' % stats.avgRet
    print 'Average Duration (Sec): %8.2f'   % stats.avgDuration

#=====================================================
# Simulate an event queue (for optimization purposes)
#=====================================================
def runSimulation():
    while len(eventQueue) > 0:
        event = eventQueue.pop()
        for agent in agents:
            consume(agent, event)

#====================================================
# Read next market update event (either tick or bar)
#====================================================
def getNextMarketUpdateEvent(security, dataSource, socket, fileHandle):
    global eof
    if dataSource == 'NXCORE':
        
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
        # Use the day from the date (20130826)
        timestamp = '26-' + data[1]
        tradePrice = float(data[2])
        tradeNetChange = float(data[3])
        tradeSize = float(data[4])
        tradeVolume = 0
        #print("%s %s %f %f %d" %(symbol, timestamp, tradePrice, tradeNetChange, tradeSize))
        event = MarketUpdate(symbol, timestamp, [tradePrice, tradeVolume])

    elif dataSource == 'DISKTRADING':
        
        line = next(fileHandle, '')
        if line[:6] == '"Date"':
            # Skip file header line
            line = next(fileHandle, '')
        if line != '':
            data = line.strip().split(',')
            # Use the day from the date
            timestamp = data[0][3:5] + '-' + data[1][:2] + ':' + data[1][-2:] + ':00'
            event = [Bar('ESM4', timestamp, map(float, data[2:6]))]
        else:
            event = None
        
    return event

#-------------------------
# Display the event queue
#-------------------------
def displayQueue(eventQueue):
    print 'Queue (Len=' + str(len(eventQueue)) + '): '
    for event in eventQueue:
        if isinstance(event, MarketUpdate):
            print 'M'
        elif isinstance(event, list) or isinstance(event, float):
            print event
        elif isinstance(event, Comm):
            print 'C'
        elif isinstance(event, Prc):
            print 'P'
        elif isinstance(event, Bar) or isinstance(event, TickBar):
            print event.open
        elif isinstance(event, Book):
            print 'B'
    print
    
#-----------------------
# Run a swarm of agents
#-----------------------
def runSwarm(aggregateAgents, barAgents, securities, swarmFF, swarmType, dataSource, socket, fileHandle):
    while len(eventQueue) > 0:
        # Consume current market update events
        while len(eventQueue) > 0:
            event = eventQueue.pop()
            # A BarAgent may be a TickBarAgent or a ReadyBarAgent
            # A TickBarAgent consumes MarketUpdate events and produces TickBar and Comm events
            # A ReadyBarAgent consumes ReadyBar events
            for barAgent in barAgents:
                consume(barAgent, event)
            # Aggregate agents consume TickBar and Comm events
            for aggregateAgent in aggregateAgents:
                for agent in aggregateAgent.members:
                    consume(agent, event)

        # Get more market update events
        for security in securities:
            event = getNextMarketUpdateEvent(security, dataSource, socket, fileHandle)
            if event != None:
                eventQueue.insert(0, event)

#----------------------------------------------------
# Plot the generated tickBars, the ma and the trades
#----------------------------------------------------
def plot(tickBarsPlot, ma, trades, plotCandle=True, plotMa=True):
    # Create a numpy array for more manipulation flexibility
    data = np.array(tickBarsPlot)
    dataSwitched = data[:, 1][np.newaxis].T                                     # Open
    dataSwitched = np.append(dataSwitched, data[:, 4][np.newaxis].T, axis=1)    # Close
    dataSwitched = np.append(dataSwitched, data[:, 2][np.newaxis].T, axis=1)    # High
    dataSwitched = np.append(dataSwitched, data[:, 3][np.newaxis].T, axis=1)    # Low
    # This replaces the first column with numbers 0..size-1
    data2 = np.append(np.arange(data[:,0].size)[:, np.newaxis], dataSwitched, axis=1)
    data2tuple = [tuple(x) for x in data2]
    # Chart ticks are on the hour
    ticks = np.unique(np.trunc(data[:,0] / 10000), return_index=True)
    fig = plt.figure(figsize=(10, 5))
    # These numbers define margin from window edge
    ax = fig.add_axes([0.1, 0.2, 0.85, 0.7])
    ax.set_xticks(data2[ticks[1], 0])
    ax.set_xticklabels(map(int, data[ticks[1], 0]), rotation=90)
    
    # Plot prices
    if plotCandle:
        candlestick(ax, data2tuple, width=0.5, colorup='g', colordown='r')
    else:
        ax.plot(data[:,4])

    # Plot moving average
    if plotMa:
        ax.plot(ma)

    # Plot trades
    buys = []
    sells = []
    for trade in trades:
        timestampFloat = (float(trade.timestamp[0:2]) * 1000000 +
                          float(trade.timestamp[3:5]) * 10000 +
                          float(trade.timestamp[6:8]) * 100 +
                          float(trade.timestamp[9:11]))
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

#===================================
# Create a bar agent for a security
#===================================
def createBarAgent(dataSource, mkt, numEvents=70):
    if dataSource == 'DISKTRADING':
        return ReadyBarAgent(mkt)
    else:
        return TickBarAgent(mkt, numEvents)

#======
# Main
#======
if __name__ == '__main__':

    baseDir = './'
    dataSource = 'NXCORE'

    if len(sys.argv) >= 2:
        symbol = sys.argv[1]
    else:
        symbol = 'COO'
        
    if len(sys.argv) >= 3:
        date = sys.argv[2]
    else:
        date = '20160602'
        
    if dataSource == 'NXCORE':
        # Data transmitted from C:\NxCore\Examples\NxCoreLanguages\C++\SampleApp4
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.connect('tcp://localhost:5556')
        fileHandle = None
        securities = ['fES.U13', 'f6E.U13']
        
    elif dataSource == 'DISKTRADING':
        # Data from DiskTrading (1-min bars)
        fileName = 'ESM4-sample.csv'
        fileHandle = open(fileName)
        socket = None
        securities = ['ESM4']

    # There are as many barAgents as number of securities
    barAgents = []

    # There are as many swarmAgents as number of securities;
    # each swarmAgent contains agents for each MA scenario in Llist
    swarmAgents = []

    # For each security, create a moving average simple model swarm agent
    #print 'Creating set of swarm agents (one swarm agent per security)...'
    Llist = range(50, 100, 50)
    for security in securities:
        barAgent = createBarAgent(dataSource, mkt=security, numEvents=100)
        barAgents.append(barAgent)
        swarmAgent = AggregateAgent(name='SwarmAgent')
        for L in Llist:
            agent = BreakoutAgent(mkt=security, numBars=3, yDayFile=baseDir+symbol+'-params.csv')
            swarmAgent.members.append(agent)
            barAgent.recipientsList.append(agent)
        swarmAgents.append(swarmAgent)

    OPTIMIZE = False
    if (OPTIMIZE):
        agents = []
        for i in range(10, 111):
            agent = SimpleModelAgentComm(L=10, mkt=securities[0])
            agents.append(agent)
        runSimulation(agents, eventQueue)
        results = []
        for agent in agents:
            results.append(agent.tradestats[-1])

    start_time = time.time()

    #print 'Running...'
    # Get first MarketUpdate events and add them to event queue (to kickstart the event loop)
    for security in securities:
        event = getNextMarketUpdateEvent(security, dataSource, socket, fileHandle)
        if event != None:
            eventQueue.insert(0, event)
            
    runSwarm(swarmAgents, barAgents, securities, RTNAV, 'ADD', dataSource, socket, fileHandle)

    for swarmAgent in swarmAgents:
        for j in range(len(Llist)):
            if len(swarmAgent.members[j].tradestats) > 0:
                showStats(swarmAgent.members[j].tradestats[-1])

    # Save yesterday's volume and last spanNumBars prices to params file
    array = np.array(agent.prices)
    prices = array[-(agent.spanNumBars + 4):]  # The +4 is for the maximum value of counter after the first loop
    totalVolume = sum(array[:, 4])
    filename = baseDir + symbol + '-params.csv'
    f = open(filename, 'w')
    f.write('%d' % totalVolume + '\n')
    np.savetxt(f, prices, delimiter=',', fmt='%.2f')
    
    #print 'Plotting...'
    #for i, barAgent in enumerate(barAgents):
    #    for j in range(len(Llist)):
    #        if swarmAgents[i].members[j] is not None:
    #            plot(barAgent.tickBarsPlot, swarmAgents[i].members[j].mas, swarmAgents[i].members[j].trades)

    #elapsed_time = time.time() - start_time
    #print 'Elapsed time = %.1f sec.' % elapsed_time
