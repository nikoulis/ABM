from __future__ import print_function
from agent import *
from event import *
import pdb
import sys

SHOW_TICKBARS = False

class TickBarAgent(Agent):
    def __init__(self,
                 mkt='',
                 numEvents=70,
                 counter=0,
                 buffer=None,
                 open=0,
                 high=0,
                 low=99999,
                 close=0,
                 volume=0,
                 tickBars=None,
                 tickBarsPlot=None):
        self.mkt = mkt
        self.numEvents = numEvents
        Agent.__init__(self, name='TICKBARAGENT_' + mkt + '_' + str(self.numEvents))
        self.counter = counter
        self.buffer = buffer if buffer != None else []
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.tickBars = tickBars if tickBars != None else []
        self.tickBarsPlot = tickBarsPlot if tickBarsPlot != None else []

        # Static states
        #---------------
        self.calcState = TickBarAgentCalcState()
        self.emitState = TickBarAgentEmitState()
        self.currentState = self.emitState

#------------
# CALC state
#------------
class TickBarAgentCalcState:
    def execute(self, agent, event):
        if event.value[0] == None:
            # File is done; need to create tick bar regardless of whether we reached numTicks or not
            price = agent.prices[-2][0]
            volume = 0
        else:
            price = event.value[0]
            volume = event.value[1]
            agent.buffer.append(price)

        agent.volume += volume
        agent.close = price
        agent.high = max(agent.high, price)
        agent.low = min(agent.low, price)
        
        if agent.counter < agent.numEvents - 1 and event.value[0] != None:
            newState = agent.currentState
        elif agent.counter == agent.numEvents - 1 or event.value[0] == None:
            tickBar = TickBar(security=agent.mkt,
                              timestamp=agent.timestamps[-agent.counter - 1],
                              value=[agent.open, agent.high, agent.low, agent.close, agent.volume],
                              numTicks=agent.numEvents)
            if SHOW_TICKBARS:
                print('============> %s %d-Tick Bar (%s): Open=%.2f, High=%.2f, Low=%.2f, Close=%.2f, Volume=%d'
                      % (agent.mkt, agent.numEvents, agent.timestamps[-agent.counter - 1],
                         agent.open, agent.high, agent.low, agent.close, agent.volume), file=sys.stderr)
            agent.tickBars.append(tickBar)
            timestampFloat = (float(agent.timestamps[-agent.counter - 1][0:8]) * 1000000 +
                              float(agent.timestamps[-agent.counter - 1][9:11]) * 10000 +
                              float(agent.timestamps[-agent.counter - 1][12:14]) * 100 +
                              float(agent.timestamps[-agent.counter - 1][15:17]))
            agent.tickBarsPlot.append([timestampFloat,
                                       agent.open,
                                       agent.high,
                                       agent.low,
                                       agent.close,
                                       agent.volume])
            emit(agent, agent.tickBars)
            agent.buffer = []
            newState = agent.emitState
        #print('%s %s -> %s' % (agent.name, agent.currentState, newState))
        agent.changeState(newState)

#------------
# EMIT state
#------------
class TickBarAgentEmitState:
    def execute(self, agent, event):
        agent.counter = 0
        price = event.value[0]
        agent.volume = event.value[1]
        agent.open = price
        agent.high = price
        agent.low = price
        agent.close = price
        if price != None:
            agent.buffer.append(price)
        newState = agent.calcState
        #print('%s %s -> %s' % (agent.name, agent.currentState, newState))
        agent.changeState(newState)
