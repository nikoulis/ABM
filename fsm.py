# This is simply so that 'print' is an expression (like in Python 3)
# and can be used in the lambdas below
from __future__ import print_function

import time
import pdb

class Agent(object):
    def __init__(self,
                 name,
                 timestamps=[],
                 revalPrices=[],
                 orders=[],
                 positions=[],
                 pls=[],
                 fitnesses=[],
                 trades=[],
                 tradestats=[],
                 incomingMessages=[],
                 outgoingMessages=[],
                 recipientList=[]):
        self.name             = name
        self.timestamps       = timestamps      
        self.revalPrices      = revalPrices     
        self.orders           = orders          
        self.positions        = positions       
        self.pls              = pls             
        self.fitnesses        = fitnesses       
        self.trades           = trades          
        self.tradestats       = tradestats      
        self.incomingMessages = incomingMessages
        self.outgoingMessages = outgoingMessages
        self.recipientList    = recipientList   

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

class Trade(object):
    def __init__(self, timestamp, price, tradeQuantity):
        self.timestamp = timestamp
        self.price = price
        self.tradeQuantity = tradeQuantity

def price(event):
    return event.value

def consume(agent, event):
    if observe(agent, event):
        updateBefore(agent, event)
        update(agent, event)
        updateAfter(agent, event)

def observe(agent, event):
    if isinstance(agent, SimpleModelComm):
        return agent.mkt == event.security
    elif isinstance(agent, SimpleModel):
        return (agent in event.recipients) and (agent != event.originator)

def updateBefore(agent, event):
    agent.timestamps.append(event.timestamp)
    agent.revalPrices.append(price(event))
    preprocess(agent, event)
    print("updateBefore completed for agent %s and event %s" % (agent.name, event.timestamp))
    if isinstance(agent, SimpleModelComm):
        agent.incomingMessages.append(event)
        preprocess(agent, event)
        print("updateBefore completed for agent %s and comm event %s" % (agent.name, event))
    
#def update(agent, marketUpdate):
#    position = raw_input("Enter new position for T=%s and P=%s\n" % (marketUpdate.timestamp, price(marketUpdate)))
#    agent.positions.append(position)

def computeStats(trades):
    return 0

def updateAfter(agent, event):
    L = len(agent.timestamps)
    lastPos = agent.positions[-1]
    if L < 2:
        prevPos = 0
    else:
        prevPos = agent.positions[-2]
    tradeQuantity = lastPos - prevPos
    lastPrice = agent.revalPrices[-1]
    if L < 2:
        prevPrice = 0
        pl = 0
    else:
        prevPrice = agent.revalPrices[-2]
        pl = prevPos * (lastPrice - prevPrice)
    agent.pls.append(pl)
    if tradeQuantity != 0:
        trade = Trade(event.timestamp,
                      price(event) + slippage(agent, event, tradeQuantity),
                      tradeQuantity)
        agent.trades.append(trade)
        agent.tradestats.append(computeStats(agent.trades))
    postprocess(agent, event)
    print("updateAfter completed for agent %s and event %s" % (agent.name, event.timestamp))
    if isinstance(agent, SimpleModelComm):
        postprocess(agent, event)
        print("updateAfter completed for agent %s and comm event %s" % (agent.name, event))

def slippage(agent, marketUpdate, tradeQuantity):
    return 0

class FSM(object):
    def __init__(self, currentState='', transitions=[]):
        self.currentState = currentState
        self.transitions = transitions

class FSMAgent(FSM, Agent):
    def __init__(self, name, currentState='', transitions=[], states=[]):
        FSM.__init__(self, currentState, transitions)
        Agent.__init__(self, name)
        self.states = states

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

eventsQueue = []

def emit(agent, msg):
    comm = Comm(agent, agent.recipients, agent.timestamps[-1], msg)
    eventsQueue.append(comm)
    event = Event(agent.timstamps[-1], msg)
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

def update(fsmAgent, marketUpdate):
    fsmAgent.setFSM()
    print("setFSM completed for fsmAgent %s" % fsmAgent.name)
    operateFSM(fsmAgent, marketUpdate)
    print("operateFSM completed for fsmAgent %s" % fsmAgent.name)
    fsmAgent.states.append(fsmAgent.currentState)
    print("Completed work for %s and new state %s added" % (fsmAgent.name, fsmAgent.currentState))

def update(fsmAgent, comm):
    fsmAgent.setFSM()
    print("setFSM completed for fsmAgent %s" % fsmAgent.name)
    print("Completed work for %s and comm event %s" % (fsmAgent.name, comm))

class SimpleModel(FSMAgent):
    def __init__(self, L, counter=0, ma=0):
        FSMAgent.__init__(self, '')
        self.L = L
        self.counter = counter
        self.ma = ma
        if len(self.states) == 0:
            self.states.append('INIT')
            self.name = 'SIMPLE_MODEL' + str(self.L)

class SimpleModelComm(SimpleModel):
    def __init__(self, L, mkt='', unblockShort=0, unblockLong=0):
        SimpleModel.__init__(self, L)
        self.mkt = mkt
        self.unblockLong = unblockLong
        self.unblockShort = unblockShort
    
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

def initialize(model):
    if model.states == []:
        model.states.append('INIT')
        model.name = 'SIMPLE_MODEL' + str(self.L)
        emit(model, 'INIT')

mod1 = SimpleModelComm(10, mkt='AAPL')

def preprocess(model, event):
    model.counter = len(model.revalPrices)
    sublist = model.revalPrices[model.counter-model.L:model.counter]
    model.ma = sum(sublist) / len(sublist)
    if isinstance(model, SimpleModelComm):
        if event.value == 'INIT':
            model.unblockShort = 0
            model.unblockLong = 0
        elif event.value == 'LONG':
            model.unblockShort = -1
            model.unblockLong = 0
        elif event.value == 'SHORT':
            model.unblockShort = 0
            model.unblockLong = 1

def postprocess(model, marketUpdate):
    print("Event %s %s consumed for agent %s" % (marketUpdate.timestamp, price(marketUpdate), model.name))
    print("Output: Counter=%s, ma=%s, state=%s, position=%s, pl=%s" % (model.counter, model.ma, model.states[-1], model.positions[-1], model.pls[-1]))

def runSimulation(events):
    for agent in agents:
        agent.initialize()
    eventsQueue = events
    for event in eventsQueue:
        for agent in agents:
            consume(agent, event)

if __name__ == '__main__':
    events = []

    f = open('aapl.csv')
    i = 0
    for line in f:
        if i > 0:
            data = line.strip().split(',')

            timestamp = data[0]
            value = float(data[6])
            event = MarketUpdate('AAPL', timestamp, value)
            events.append(event)
        i += 1

    events = sorted(events)

    start_time = time.time()

    for event in events:
        consume(mod1, event)
        print("\n")

    elapsed_time = time.time() - start_time
    print("Elapsed time = %.1f sec." % elapsed_time)
