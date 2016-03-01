from agent import *
from event import *
import pdb

class TickBarGenerator(Agent):
    def __init__(self,
                 mkt='',
                 numEvents=70,
                 counter=0,
                 buffer=None,
                 open=None,
                 high=None,
                 low=None,
                 close=None,
                 tickBars=None,
                 tickBarsPlot=None):
        self.mkt = mkt
        self.numEvents = numEvents
        Agent.__init__(self, name='TICKBARGENERATOR_' + mkt + '_' + str(self.numEvents))
        self.counter = counter
        self.buffer = buffer if buffer != None else []
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.tickBars = tickBars if tickBars != None else []
        self.tickBarsPlot = tickBarsPlot if tickBarsPlot != None else []
        self.fsm.currentState = 'EMIT'
        #self.states = ['EMIT']
        
    def setFSM(self):
        #self.fsm.currentState = self.states[-1]
        self.fsm.transitions = [Transition(initialState='CALC',
                                           finalState='CALC',
                                           sensor=price,
                                           predicate=(lambda x:
                                                      self.counter < self.numEvents),
                                           actuator=(lambda x:
                                                     self.actuator(x, 'CALC', 'CALC'))),
                                Transition(initialState='CALC',
                                           finalState='EMIT',
                                           sensor=price,
                                           predicate=(lambda x:
                                                      self.counter == self.numEvents),
                                           actuator=(lambda x:
                                                     self.actuator(x, 'CALC', 'EMIT'))),
                                Transition(initialState='EMIT',
                                           finalState='CALC',
                                           sensor=price,
                                           predicate=(lambda x:
                                                      True),
                                           actuator=(lambda x:
                                                     self.actuator(x, 'EMIT', 'CALC'))),
                                Transition(initialState='EMIT',
                                           finalState='EMIT',
                                           sensor=price,
                                           predicate=(lambda x:
                                                      False),
                                           actuator=(lambda x:
                                                     False))]

    def actuator(self, x, initialState, finalState):
        if initialState == 'CALC' and finalState == 'CALC':
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
            self.tickBars.append(tickBar)
            timestampFloat = float(self.timestamps[-1][0:2]) * 10000 + float(self.timestamps[-1][3:5]) * 100 + float(self.timestamps[-1][6:8])
            self.tickBarsPlot.append([timestampFloat, self.open, self.high, self.low, self.close])
            emit(self, self.tickBars)
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

