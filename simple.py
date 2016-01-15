from __future__ import print_function
from agent import *
from event import *

#--------------------------------------------------------
# A simple moving average model; this is simply a shell,
# most of the fuctionality is in SimpleModelComm below
#--------------------------------------------------------
class SimpleModel(FSMAgent):
    def __init__(self, L, counter=0, mas=None):
        FSMAgent.__init__(self, name='')
        self.L = L
        self.counter = counter
        self.mas = mas if mas != None else []
        self.states = ['INIT']
        self.name = 'SIMPLE_MODEL_' + str(self.L)

#----------------------------------------------------------------------------------------
# A simple moving average model, with its full functionality (transitions and actuators);
# buy when price crosses above moving average, sell when it crosses below
#----------------------------------------------------------------------------------------
class SimpleModelComm(SimpleModel):
    # unblockShort and unblockLong can be used to influence the behavior of this agent
    # i.e. if unblockShort=0, there are no short trades and if unblockLong=0, there are no long trades;
    # normally, unblockShort=-1 and unblockLong=1
    def __init__(self, L, mkt='', unblockShort=-1, unblockLong=1):
        SimpleModel.__init__(self, L)
        self.mkt = mkt
        self.unblockLong = unblockLong
        self.unblockShort = unblockShort
        if len(self.states) == 0:
            emit(self, 'INIT')
    
    # Transitions
    def setFSM(self):
        self.currentState = self.states[-1]
        self.transitions = [Transition(initialState='INIT',
                                       finalState='INIT',
                                       sensor=price,
                                       predicate=(lambda x:
                                                  self.predicate(x, 'INIT', 'INIT')),
                                       actuator=(lambda x:
                                                 self.actuator(x, 'INIT', 'INIT'))),
                            Transition(initialState='INIT',
                                       finalState='LONG',
                                       sensor=price,
                                       predicate=(lambda x:
                                                  self.predicate(x, 'INIT', 'LONG')),
                                       actuator=(lambda x:
                                                 self.actuator(x, 'INIT', 'LONG'))),
                            Transition(initialState='INIT',
                                       finalState='SHORT',
                                       sensor=price,
                                       predicate=(lambda x:
                                                  self.predicate(x, 'INIT', 'SHORT')),
                                       actuator=(lambda x:
                                                 self.actuator(x, 'INIT', 'SHORT'))),
                            Transition(initialState='LONG',
                                       finalState='INIT',
                                       sensor=price,
                                       predicate=(lambda x:
                                                  self.predicate(x, 'LONG', 'INIT')),
                                       actuator=(lambda x:
                                                 False)),
                            Transition(initialState='LONG',
                                       finalState='LONG',
                                       sensor=price,
                                       predicate=(lambda x:
                                                  self.predicate(x, 'LONG', 'LONG')),
                                       actuator=(lambda x:
                                                 self.actuator(x, 'LONG', 'LONG'))),
                            Transition(initialState='LONG',
                                       finalState='SHORT',
                                       sensor=price,
                                       predicate=(lambda x:
                                                  self.predicate(x, 'LONG', 'SHORT')),
                                       actuator=(lambda x:
                                                 self.actuator(x, 'LONG', 'SHORT'))),
                            Transition(initialState='SHORT',
                                       finalState='INIT',
                                       sensor=price,
                                       predicate=(lambda x:
                                                  self.predicate(x, 'SHORT', 'INIT')),
                                       actuator=(lambda x:
                                                 False)),
                            Transition(initialState='SHORT',
                                       finalState='LONG',
                                       sensor=price,
                                       predicate=(lambda x:
                                                  self.predicate(x, 'SHORT', 'LONG')),
                                       actuator=(lambda x:
                                                 self.actuator(x, 'SHORT', 'LONG'))),
                            Transition(initialState='SHORT',
                                       finalState='SHORT',
                                       sensor=price,
                                       predicate=(lambda x:
                                                  self.predicate(x, 'SHORT', 'SHORT')),
                                       actuator=(lambda x:
                                                 self.actuator(x, 'SHORT', 'SHORT')))]

    def predicate(self, x, initialState, finalState):
        if initialState == 'INIT':
            if finalState == 'INIT':
                return (self.counter <= self.L or
                        (x[-1].value[3] < self.mas[-1] and x[-2].value[3] < self.mas[-2]) or
                        (x[-1].value[3] > self.mas[-1] and x[-2].value[3] > self.mas[-2]))
            elif finalState == 'LONG':
                return (self.counter > self.L and
                        x[-1].value[3] >= self.mas[-1] and
                        x[-2].value[3] < self.mas[-2])
            elif finalState == 'SHORT':
                return (self.counter > self.L and
                        x[-1].value[3] <= self.mas[-1] and
                        x[-2].value[3] > self.mas[-2])
        elif initialState == 'LONG':
            if finalState == 'INIT':
                return False
            elif finalState == 'LONG':
                return (self.counter > self.L and
                        x[-1].value[3] >= self.mas[-1])
            elif finalState == 'SHORT':
                return (self.counter > self.L and
                        x[-1].value[3] <= self.mas[-1])
        elif initialState == 'SHORT':
            if finalState == 'INIT':
                return False
            elif finalState == 'LONG':
                return (self.counter > self.L and
                        x[-1].value[3] >= self.mas[-1])
            elif finalState == 'SHORT':
                return (self.counter > self.L and
                        x[-1].value[3] <= self.mas[-1])

    # Actuators; these would normally be part of the transitions array,
    # if only Python allowed multi-line lambdas
    def actuator(self, x, initialState, finalState):
        print("%s %s -> %s" % (self.name,
                               initialState,
                               finalState))
        if initialState == 'INIT' and finalState == 'INIT':
            self.positions.append(0)
        elif initialState == 'INIT' and finalState == 'LONG':
            self.positions.append(self.unblockLong)
            emit(self, 'LONG')
        elif initialState == 'INIT' and finalState == 'SHORT':
            self.positions.append(self.unblockShort)
            emit(self, 'SHORT')
        elif initialState == 'LONG' and finalState == 'INIT':
            pass
        elif initialState == 'LONG' and finalState == 'LONG':
            self.positions.append(self.unblockLong)
        elif initialState == 'LONG' and finalState == 'SHORT':
            self.positions.append(self.unblockShort)
            emit(self, 'SHORT')
        elif initialState == 'SHORT' and finalState == 'INIT':
            pass
        elif initialState == 'SHORT' and finalState == 'LONG':
            self.positions.append(self.unblockLong)
            emit(self, 'LONG')
        elif initialState == 'SHORT' and finalState == 'SHORT':
            self.positions.append(self.unblockShort)
