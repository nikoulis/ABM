from agent import *
from event import *
import pdb

class ReadyBarAgent(Agent):
    def __init__(self,
                 mkt='',
                 open=0,
                 high=0,
                 low=99999,
                 close=0,
                 tickBars=None,
                 tickBarsPlot=None):
        self.mkt = mkt
        Agent.__init__(self, name='READYBARAGENT_' + mkt)
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.tickBars = tickBars if tickBars != None else []
        self.tickBarsPlot = tickBarsPlot if tickBarsPlot != None else []

        # Static states
        #---------------
        self.emitState = ReadyBarAgentEmitState()
        self.currentState = self.emitState

#------------
# EMIT state
#------------
class ReadyBarAgentEmitState:
    def execute(self, agent, event):
        agent.open = event[-1].open
        agent.high = event[-1].high
        agent.low = event[-1].low
        agent.close = event[-1].close

        bar = Bar(security=agent.mkt,
                  timestamp=agent.timestamps[-1],
                  value=[agent.open, agent.high, agent.low, agent.close])
        print("============> Bar: Open=%0.4f High=%0.4f Low=%0.4f Close=%0.4f"
              % (agent.open, agent.high, agent.low, agent.close))
        agent.tickBars.append(bar)
        timestampFloat = (float(agent.timestamps[-1][0:2]) * 1000000 +
                          float(agent.timestamps[-1][3:5]) * 10000 +
                          float(agent.timestamps[-1][6:8]) * 100 +
                          float(agent.timestamps[-1][9:11]))
        agent.tickBarsPlot.append([timestampFloat, agent.open, agent.high, agent.low, agent.close])
        newState = agent.emitState
        agent.changeState(newState)
