import util

#-----------------------------------------------------------------------
# An agent holds trading strategy results (P&L, orders, positions etc.)
# and communicates with other agents via incoming and outgoing messages
#-----------------------------------------------------------------------
class Agent(object):
    def __init__(self,
                 name,
                 timestamps=None,
                 revalPrices=None,
                 orders=None,
                 positions=None,
                 pls=None,
                 cumPls=None,
                 tradePls=None,
                 cumTradePls=None,
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
        self.cumPls           = cumPls           if cumPls != None else []
        self.tradePls         = tradePls         if tradePls != None else []
        self.cumTradePls      = cumTradePls      if cumTradePls != None else []
        self.fitnesses        = fitnesses        if fitnesses != None else []
        self.trades           = trades           if trades != None else []
        self.tradestats       = tradestats       if tradestats != None else []
        self.incomingMessages = incomingMessages if incomingMessages != None else []
        self.outgoingMessages = outgoingMessages if outgoingMessages != None else []
        self.recipientsList   = recipientsList   if recipientsList != None else []

#----------------------------------------------------------
# An FSM (Finite State Machine) defines a trading strategy
#----------------------------------------------------------
class FSM(object):
    def __init__(self, currentState='', transitions=None):
        self.currentState = currentState
        self.transitions = transitions if transitions != None else []

#--------------------------------------------------------------------------------------
# An FSMAgent is a trading strategy along with its performance statistics and messages
#--------------------------------------------------------------------------------------
class FSMAgent(FSM, Agent):
    def __init__(self,
                 name,
                 currentState='',
                 transitions=None,
                 states=None):
        FSM.__init__(self, currentState, transitions)
        Agent.__init__(self, name='')
        self.states = states if states != None else []

#----------------------------------------------
# An aggregate agent is a collection of agents
#----------------------------------------------
class AggregateAgent(Agent):
    def __init__(self, name, members=None):
        Agent.__init__(self, name)
        self.members = members if members != None else []

