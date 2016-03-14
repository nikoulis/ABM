from agent import *
from event import *
import pdb

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
        price = event.value
        if agent.counter < agent.numEvents:
            agent.close = price
            agent.high = max(agent.high, price)
            agent.low = min(agent.low, price)
            agent.buffer.append(price)
            newState = agent.currentState
        elif agent.counter == agent.numEvents:
            tickBar = TickBar(security=agent.mkt,
                              timestamp=agent.timestamps[-1],
                              value=[agent.open, agent.high, agent.low, agent.close],
                              numTicks=agent.numEvents)
            print("============> TickBar: Open=%0.4f High=%0.4f Low=%0.4f Close=%0.4f"
                  % (agent.open, agent.high, agent.low, agent.close))
            agent.tickBars.append(tickBar)
            timestampFloat = (float(agent.timestamps[-1][0:2]) * 10000 +
                              float(agent.timestamps[-1][3:5]) * 100 +
                              float(agent.timestamps[-1][6:8]))
            agent.tickBarsPlot.append([timestampFloat, agent.open, agent.high, agent.low, agent.close])
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
        price = event.value
        agent.open = price
        agent.high = price
        agent.low = price
        agent.close = price
        agent.buffer.append(price)
        newState = agent.calcState
        #print('%s %s -> %s' % (agent.name, agent.currentState, newState))
        agent.changeState(newState)
