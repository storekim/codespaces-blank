import os
import pandas as pd
import numpy as np
import backtrader as bt
import random as rd
import matplotlib
matplotlib.use('Agg')  # Undgå GUI ved grafgenerering i ikke-interaktiv miljø
import matplotlib.pyplot as plt

class TestStrategy(bt.Strategy): 
    params = (
        ('ma',5),('std',1),('rsi_period',14),('rsi_value',14),('rsi_overbought_or_oversold',2),('interest_series', None)
        )

    def __init__(self): 
        self.sma = bt.indicators.SMA(period=self.p.ma,plot=False)
        self.std = bt.indicators.StandardDeviation(period=self.p.ma,plot=False)
        self.rsi=bt.indicators.RSI(period=self.p.rsi_period,safediv=True,plot=False)
        self.interest_series = self.p.interest_series
        self.last_date = None
        self.account_values = []
        self.account_rates=[]

    def next(self):
        try:
            current_date = pd.Timestamp(self.datas[0].datetime.date(0))
            try:
                origin_date = pd.Timestamp(self.datas[0].log_origin_date[0])
            except (AttributeError, IndexError, KeyError):
                origin_date = current_date
            if self.last_date and self.interest_series is not None:
                for i in range((current_date - self.last_date).days):
                    day = self.last_date + pd.Timedelta(days=i + 1)
                    rate = self.interest_series.get(origin_date, None)
                    if rate is None:
                        future = self.interest_series.index[self.interest_series.index > origin_date]
                        if not future.empty:
                            rate = self.interest_series.loc[future[0]]
                    if rate is not None:
                        if not self.position:
                            daily_interest = self.broker.get_cash() * (rate / 100) / 365
                            self.broker.add_cash(daily_interest)
            
                self.account_rates.append(((current_date - self.last_date).days*(rate / 100))/365+1)
                self.account_values.append(self.broker.getvalue())
            self.last_date = current_date
            
            if not self.position:
                if self.data.close > self.sma + self.std * self.p.std and (
                    (self.rsi < self.p.rsi_value and self.p.rsi_overbought_or_oversold == 0) or
                    (self.rsi > self.p.rsi_value and self.p.rsi_overbought_or_oversold == 1) or
                    self.p.rsi_overbought_or_oversold == 2):
                    self.buy(size=self.getsizing())
            elif (self.data.close < self.sma + self.std * self.p.std and (
                (self.rsi > self.p.rsi_value and self.p.rsi_overbought_or_oversold == 0) or
                (self.rsi < self.p.rsi_value and self.p.rsi_overbought_or_oversold == 1) or
                self.p.rsi_overbought_or_oversold == 2)):
                self.close()
            
        except Exception as e:
            print(e)
        
        return
    
def make_bootstrap_data(price_data: pd.DataFrame, smoothing=0.01):
    """
    Genererer bootstrappet prisdata ud fra en dataframe med open/close priser.

    Parametre:
    price_data (pd.DataFrame): Dataframe med kolonner ['datetime', 'Open', 'close']
    smoothing (float): Sandsynlighed for at afslutte blok (geometrisk fordeling)

    Returns:
    pd.DataFrame: Bootstrappet datasæt med kolonner ['datetime', 'open', 'close', 'log_origin_date']
    """
    # Sørg for at datetime er korrekt type
    price_data = price_data.copy()
    price_data['datetime'] = pd.to_datetime(price_data['datetime'], dayfirst=True)

    # Beregn log-return (logchange)
    price_data['logchange'] = np.log10(price_data['close'] / price_data['close'].shift(1))
    price_data = price_data.dropna().reset_index(drop=True)  # Drop første række uden logchange

    reference_dates = price_data['datetime'].tolist()
    newdata, prev_price, current_index = [], 1, 0

    while current_index < len(reference_dates):
        block_start = rd.randint(0, len(price_data) - 1)
        block_size = np.random.geometric(smoothing)
        for i in range(block_size):
            if current_index >= len(reference_dates):
                break
            log_row = price_data.iloc[(block_start + i) % len(price_data)]
            log_change = log_row['logchange']
            new_price = prev_price * (10 ** log_change)
            newdata.append([
                reference_dates[current_index],
                prev_price,
                new_price,
                log_row['datetime']  # log_origin_date
            ])
            prev_price = new_price
            current_index += 1

    return pd.DataFrame(newdata, columns=['datetime', 'open', 'close', 'log_origin_date'])

if __name__ == '__main__':
    dtb = pd.read_csv(DTB3.csv', parse_dates=['observation_date'])
    dtb['observation_date'] = pd.to_datetime(dtb['observation_date'],dayfirst=True)
    dtb.set_index('observation_date', inplace=True)
    dtb['DTB3']=dtb['DTB3'].ffill()
    interest_series = dtb['DTB3']
    cerebro = bt.Cerebro()#stdstats=False)

    cerebro.addstrategy(TestStrategy, interest_series=interest_series)

    datapath = pd.DataFrame(pd.read_csv('C:/Users/Joachim Mencke/Desktop/bachelor/spx_d.csv',usecols=(0,4,1)))

    real_data = pd.read_csv('spx_d.csv', usecols=[0, 4, 1])

    real_data['datetime'] = pd.to_datetime(real_data['datetime'], dayfirst=True)

    start_date = '1954-01-01'
    end_date = '2025-02-05'

    datapath=make_bootstrap_data('spx_d')     
    datapath['datetime'] = pd.to_datetime(datapath['datetime'],dayfirst=True)
    print(datapath)
    data=datapath
    initial_price, final_price = data.iloc[0]['close'], data.iloc[-1]['close']
    buy_and_hold = (1 * data['close'] / initial_price).mean()

    # Create a Data Feed
    buy_and_hold_values = 1 * datapath['close'] / datapath.iloc[0]['close']
    avg_benchmark_value = buy_and_hold_values.mean()
    
    data = bt.feeds.PandasData(dataname=datapath,datetime=0 ,open=1 ,close=2)
    data.plotinfo.plotlog = False
    cerebro.adddata(data)
    cerebro.broker.setcash(1)

    cerebro.addsizer(bt.sizers.PercentSizer,percents=99.9)
    cerebro.broker.setcommission(commission=0.00002,leverage=1)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio_A, _name="mysharpe")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="tradeanalysers")
    cerebro.addobserver(bt.observers.DrawDown)
    cerebro.addobserver(bt.observers.Value)
    print('Starting Balance: %.2f' % cerebro.broker.getvalue())
    
    opt_runs=cerebro.run()
    run=opt_runs[0]

    account_values = run.account_values

    log_returns = np.log10(account_values[1:]) - np.log10(account_values[:-1])
    excess_returns=np.mean(log_returns)-np.mean(np.log10(run.account_rates[:-1]))
    std=np.std(log_returns)
    annual_sharpe_ratio=excess_returns/std*252**0.5

    log_returns_bench  = np.log10(datapath['close'].values[1:] / datapath['close'].values[:-1])
    excess_returns_bench=np.mean(log_returns_bench)-np.mean(np.log10(run.account_rates[:-1]))
    std_bench=np.std(log_returns_bench)
    annual_sharpe_ratio_bench=excess_returns_bench/std_bench*252**0.5
    
    return_diff=10**((np.mean(log_returns)-np.mean(log_returns_bench))*252)-1
    sharpe_diff=annual_sharpe_ratio-annual_sharpe_ratio_bench

    print(return_diff,sharpe_diff)
    print('Sharpe Ratio:', run.analyzers.mysharpe.get_analysis())
    print(annual_sharpe_ratio,annual_sharpe_ratio_bench)
    #print('analyser:', run.analyzers.tradeanalysers.get_analysis())
    print('Final Balance: %.2f' % cerebro.broker.getvalue())
    #figs_nested=cerebro.plot(style='candlestick',volume=False,voloverlay=False,plot=False)#volume=True,style='candlestick',voloverlay=False)
    cerebro.plot(style='candlestick',plot=False,volume=False)
