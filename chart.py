#!/usr/bin/env python
import matplotlib.pyplot as plt
from matplotlib.dates import  DateFormatter, WeekdayLocator, HourLocator, \
     DayLocator, MONDAY
from matplotlib.finance import quotes_historical_yahoo, candlestick,\
     plot_day_summary, candlestick2


# make plot interactive in order to update
plt.ion()

class Candleplot:
    def __init__(self):
        fig, self.ax = plt.subplots()
        fig.subplots_adjust(bottom=0.2)

    def update(self, quotes, clear=False):

        if clear:
            # clear old data
            self.ax.cla()

        # axis formatting
        self.ax.xaxis.set_major_locator(mondays)
        self.ax.xaxis.set_minor_locator(alldays)
        self.ax.xaxis.set_major_formatter(weekFormatter)

        # plot quotes
        candlestick(self.ax, quotes, width=0.6)

        # more formatting
        self.ax.xaxis_date()
        self.ax.autoscale_view()
        plt.setp( plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')

        # use draw() instead of show() to update the same window
        plt.draw()


# (Year, month, day) tuples suffice as args for quotes_historical_yahoo
date1 = ( 2004, 2, 1)
date2 = ( 2004, 4, 12 )
date3 = ( 2004, 5, 1 )

mondays = WeekdayLocator(MONDAY)        # major ticks on the mondays
alldays    = DayLocator()              # minor ticks on the days
weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
dayFormatter = DateFormatter('%d')      # e.g., 12

quotes = quotes_historical_yahoo('INTC', date1, date2)

plot = Candleplot()
plot.update(quotes)

raw_input('Hit return to add new data to old plot')

new_quotes = quotes_historical_yahoo('INTC', date2, date3)

plot.update(new_quotes, clear=False)

raw_input('Hit return to replace old data with new')

plot.update(new_quotes, clear=True)

raw_input('Finished')
