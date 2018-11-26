from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates

def plot_candlestick_ohlc(df, ax, width=0.002):
    df_ohlc = df.copy()
    df_ohlc['date'] = df_ohlc.index.map(mdates.date2num)
    
    #ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
    #ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1,sharex=ax1)
    ax.xaxis_date()
    candlestick_ohlc(ax,
                     df_ohlc[['date','open', 'high', 'low', 'close']].values,
                     width=width,
                     colorup='g')