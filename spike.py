from __future__ import print_function
from agent import *
from event import *

class SpikeAgent(FSMAgent):
    def __init__(self, mkt='', numBarsUp=5, numBarsDown=1):
        FSMAgent.__init__(self, name='')
        self.mkt = mkt
        self.numBarsUp = numBarsUp
        self.numbarsDown = numBarsDown
        if len(self.states) == 0:
            emit(self, 'INIT')
    
    def setFSM(self):
        self.currentState = self.states[-1]
        self.transitions = [Transition(initialState='INIT',
                                       finalState='INIT',
                                       sensor=price,
                                       predicate=(lambda x:
                                                  self.counter <= self.numBarsUp),
                                       actuator=(lambda x:
                                                 self.actuator(x, 'INIT', 'INIT'))),
                            Transition(initialState='INIT',
                                       finalState='LONG',
                                       sensor=price,
                                       predicate=(lambda x:
                                                  self.counter > self.numBarsUp and
                                                  x > self.ma),
                                       actuator=(lambda x:
                                                 self.actuator(x, 'INIT', 'LONG')
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

    def actuator(self, x, initialState, finalState):
        if initialState == 'INIT' and finalState == 'INIT':
            self.positions.append(0)
            self.close = x
            self.high = max(self.high, x)
            self.low = min(self.low, x)
            self.buffer.append(x)
        elif initialState == 'CALC' and finalState == 'EMIT':
            tickBar = TickBar(security=self.mkt,
                              timestamp=self.timestamps[-1],
                              value=[self.open, self.high, self.low, self.close],
                              numTicks=self.numEvents)
            print("============> TickBar: Open=%0.4f High=%0.4f Low=%0.4f Close=%0.4f"
                  % (self.open, self.high, self.low, self.close))
            tickbars.append([len(tickbars)+1] + tickBar.value)
            if CHART:
                # Don't do it, too slow
                plot.update(tickbars)
            emit(self, tickBar)
            self.buffer = []
        elif initialState == 'EMIT' and finalState == 'CALC':
            self.open = x
            self.high = x
            self.low = x
            self.close = x
            self.buffer.append(x)
        elif initialState == 'EMIT' and finalState == 'EMIT':
                pass
#        print("%s %s -> %s" % (self.name,
#                               initialState,
#                               finalState))
