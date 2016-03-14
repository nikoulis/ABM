from __future__ import print_function
from agent import *
from event import *

#-----------------------------------------------------------
# A simple moving average model; this is simply a shell,
# most of the fuctionality is in SimpleModelAgentComm below
#-----------------------------------------------------------
class SimpleModelAgent(Agent):
    def __init__(self, L, counter=0, mas=None):
        Agent.__init__(self, name='SIMPLE_MODEL_' + str(L))
        self.L = L
        self.counter = counter
        self.mas = mas if mas != None else []

        # Static states
        #---------------
        self.initState = SimpleModelAgentInitState()
        self.longState = SimpleModelAgentLongState()
        self.shortState = SimpleModelAgentShortState()
        self.currentState = self.initState

#----------------------------------------------------------------------------------------
# A simple moving average model, with its full functionality (transitions and actuators);
# buy when price crosses above moving average, sell when it crosses below
#----------------------------------------------------------------------------------------
class SimpleModelAgentComm(SimpleModelAgent):
    # unblockShort and unblockLong can be used to influence the behavior of this agent
    # i.e. if unblockShort=0, there are no short trades and if unblockLong=0, there are no long trades;
    # normally, unblockShort=-1 and unblockLong=1
    def __init__(self, L, mkt='', unblockShort=-1, unblockLong=1):
        SimpleModelAgent.__init__(self, L)
        self.mkt = mkt
        self.unblockLong = unblockLong
        self.unblockShort = unblockShort
    
#------------
# INIT state
#------------
class SimpleModelAgentInitState:
    def execute(self, agent, event):
        price = event[-1].value[3]
        if len(event) > 1:
            prevPrice = event[-2].value[3]
        else:
            prevPrice = 0
        if (agent.counter <= agent.L or
            (price < agent.mas[-1] and prevPrice < agent.mas[-2]) or
            (price > agent.mas[-1] and prevPrice > agent.mas[-2])):
            agent.positions.append(0)
            newState = agent.currentState
        elif (agent.counter > agent.L and
              price >= agent.mas[-1] and
              prevPrice < agent.mas[-2]):
            agent.positions.append(agent.unblockLong)
            emit(agent, 'LONG')
            newState = agent.longState
        elif (agent.counter > agent.L and
              price <= agent.mas[-1] and
              prevPrice > agent.mas[-2]):
            agent.positions.append(agent.unblockShort)
            emit(agent, 'SHORT')
            newState = agent.shortState
        else:
            newState = agent.currentState
        #print('%s %s -> %s' % (agent.name, agent.currentState, newState))
        agent.changeState(newState)

#------------
# LONG state
#------------
class SimpleModelAgentLongState:
    def execute(self, agent, event):
        price = event[-1].value[3]
        if len(event) > 1:
            prevPrice = event[-2].value[3]
        else:
            prevPrice = 0
        if (agent.counter > agent.L and
            price >= agent.mas[-1]):
            agent.positions.append(agent.unblockLong)
            newState = agent.currentState
        elif (agent.counter > agent.L and
              price <= agent.mas[-1]):
            agent.positions.append(agent.unblockShort)
            emit(agent, 'SHORT')
            newState = agent.shortState
        else:
            newState = agent.currentState            
        #print('%s %s -> %s' % (agent.name, agent.currentState, newState))
        agent.changeState(newState)

#-------------
# SHORT state
#-------------
class SimpleModelAgentShortState:
    def execute(self, agent, event):
        price = event[-1].value[3]
        if len(event) > 1:
            prevPrice = event[-2].value[3]
        else:
            prevPrice = 0
        if (agent.counter > agent.L and
            price >= agent.mas[-1]):
            agent.positions.append(agent.unblockLong)
            emit(agent, 'LONG')
            newState = agent.longState
        elif (agent.counter > agent.L and
              price <= agent.mas[-1]):
            agent.positions.append(agent.unblockShort)
            newState = agent.currentState
        else:
            newState = agent.currentState
        #print('%s %s -> %s' % (agent.name, agent.currentState, newState))
        agent.changeState(newState)
