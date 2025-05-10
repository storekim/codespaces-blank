import backtrader as bt
import pandas as pd
import numpy as np
import random as rd
import itertools
import time
import concurrent.futures
import matplotlib.pyplot as plt

class TestStrategy(bt.Strategy):
    params = dict(
        ma=200,
        std=1,
        rsi_period=14,
        rsi_value=50,
        rsi_overbought_or_oversold=0,
        interest_series=None
    )

    def __init__(self):
        self.sma = bt.indicators.SMA(period=self.p.ma)
        self.std = bt.indicators.StandardDeviation(period=self.p.ma)
        self.rsi = bt.indicators.RSI(period=self.p.rsi_period,safediv=True)
        self.interest_series = self.p.interest_series
        self.last_date = None
        self.account_values = []

    def next(self):
        try:
            current_date = pd.Timestamp(self.datas[0].datetime.date(0))
            origin_date = getattr(self.datas[0], 'log_origin_date', [current_date])[0]

            if self.last_date and self.interest_series is not None and not self.position:
                for i in range((current_date - self.last_date).days):
                    day = self.last_date + pd.Timedelta(days=i + 1)
                    rate = self.interest_series.get(origin_date, None)
                    if rate is None:
                        future = self.interest_series.index[self.interest_series.index > origin_date]
                        if not future.empty:
                            rate = self.interest_series.loc[future[0]]
                    if rate is not None:
                        daily_interest = self.broker.get_cash() * (rate / 100) / 365
                        self.broker.add_cash(daily_interest)

            self.last_date = current_date
            self.account_values.append(self.broker.getvalue())

            if not self.position:
                if self.data.close > self.sma + self.std * self.p.std and (
                    (self.rsi < self.p.rsi_value and self.p.rsi_overbought_or_oversold == 0) or
                    (self.rsi > self.p.rsi_value and self.p.rsi_overbought_or_oversold == 1) or
                    self.p.rsi_overbought_or_oversold == 2):
                    self.buy(size=self.getsizing())
            elif (self.data.close < self.sma + self.std * self.p.std and (
                (self.rsi > self.p.rsi_value and self.p.rsi_overbought_or_oversold == 0) or
                (self.rsi < self.p.rsi_value and self.p.rsi_overbought_or_oversold == 1))):
                self.close()
        except Exception as e:
            print(e)


def make_bootstrap_data(symbol='spx_d', smoothing=0.1):
    log_data = pd.read_csv(f'C:/Users/Joachim Mencke/Desktop/bachelor/log_{symbol}.csv')
    log_data['datetime'] = pd.to_datetime(log_data['datetime'], dayfirst=True)
    reference_dates = log_data['datetime'].tolist()
    newdata, prev_price, current_index = [], 16.66, 0

    while current_index < len(reference_dates):
        block_start = rd.randint(0, len(log_data) - 1)
        block_size = np.random.geometric(smoothing)
        for i in range(block_size):
            if current_index >= len(reference_dates): break
            log_row = log_data.iloc[(block_start + i) % len(log_data)]
            log_change = log_row['logchange']
            new_price = prev_price * (10 ** log_change)
            newdata.append([reference_dates[current_index], prev_price, new_price, log_row['datetime']])
            prev_price, current_index = new_price, current_index + 1

    return pd.DataFrame(newdata, columns=['datetime', 'open', 'close', 'log_origin_date'])


def run_backtest(ma, std, rsi_period, rsi_value, rsi_overbought_or_oversold, data, interest_series):
    initial_price, final_price = data.iloc[0]['close'], data.iloc[-1]['close']
    buy_and_hold = (1 * data['close'] / initial_price).mean()

    cerebro = bt.Cerebro()
    cerebro.addstrategy(TestStrategy, ma=ma, std=std, rsi_period=rsi_period,
        rsi_value=rsi_value, rsi_overbought_or_oversold=rsi_overbought_or_oversold,
        interest_series=interest_series)
    data['datetime'] = pd.to_datetime(data['datetime'], dayfirst=True)
    feed = bt.feeds.PandasData(dataname=data, datetime=0, open=1, close=2)
    cerebro.adddata(feed)
    cerebro.broker.setcash(1)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=99.9)
    cerebro.broker.setcommission(commission=0.00002, leverage=1)
    result = cerebro.run()[0]
    account_values = result.account_values

    returns = pd.Series(np.diff(account_values) / account_values[:-1])
    rf_daily = interest_series.mean() / 100 / 365
    excess_returns = returns - rf_daily
    strategy_mean = np.mean(excess_returns)
    strategy_std = np.std(returns)
    sharpe_ratio = strategy_mean / strategy_std if strategy_std != 0 else 0

    benchmark_returns = pd.Series(np.diff(data['close']) / data['close'][:-1])
    benchmark_sharpe = (benchmark_returns - rf_daily).mean() / benchmark_returns.std()

    return np.mean(account_values) - buy_and_hold, sharpe_ratio - benchmark_sharpe, benchmark_sharpe


def evaluate_strategy(params, data, interest_series):
    ma, std, rsi_period, rsi_value, rsi_type = params
    try:
        return (str(params), run_backtest(ma, std, rsi_period, rsi_value, rsi_type, data.copy(), interest_series))
    except Exception as e:
        print(f"Error in strategy {params}: {e}")
        return (str(params), (None, None))


if __name__ == '__main__':
    n_iterations = 1000
    dtb = pd.read_csv('C:/Users/Joachim Mencke/Desktop/bachelor/DTB3.csv', parse_dates=['observation_date'])
    dtb.set_index('observation_date', inplace=True)
    interest_series = dtb['DTB3'].ffill()

    ma_period = [5, 20, 50, 200]
    std_dev = [-1, 0, 1]
    rsi_periods = [7, 14, 28]
    rsi_values = [50, 70]
    rsi_types = [0, 1, 2]

    all_strategies = list(itertools.product(ma_period, std_dev, rsi_periods, rsi_values, [0, 1]))
    all_strategies += list(itertools.product(ma_period, std_dev, [14], [14], [2]))

    real_data = pd.read_csv('C:/Users/Joachim Mencke/Desktop/bachelor/spx_d.csv', usecols=[0, 4, 1])
    real_data['datetime'] = pd.to_datetime(real_data['datetime'], dayfirst=True)

    strategy_losses, strategy_sharpes = {}, {}
    print("\nRunning strategies on real data...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(evaluate_strategy, params, real_data, interest_series): params for params in all_strategies}
        real_results = {key: val for key, val in (f.result() for f in concurrent.futures.as_completed(futures)) if val[0] is not None}

    for key, (loss, sharpe,bench_sharpe) in real_results.items():
        strategy_losses[key], strategy_sharpes[key] = [loss], [sharpe]
        print(f"{key}: Loss={loss:.6f}, Sharpe={sharpe:.4f}")
    real_max_loss = max([v[0] for v in real_results.values()])
    real_max_sharpe = max([v[1] for v in real_results.values()])
    print(f"\nMax Real Sharpe: {real_max_sharpe:.4f}, Max Real LossFunc: {real_max_loss:.4f}")

    bootstrap_loss_beats_real = bootstrap_sharpe_beats_real = 0
    list_max_loss=[]
    list_max_sharpe=[]
    for i in range(n_iterations):
        start_time=time.time()
        print(f"\n--- Bootstrap Iteration {i+1}/{n_iterations} ---")
        bdata = make_bootstrap_data('spx_d')
        bdata['datetime'] = pd.to_datetime(bdata['datetime'], dayfirst=True)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(evaluate_strategy, params, bdata, interest_series): params for params in all_strategies}
            boot_results = {key: val for key, val in (f.result() for f in concurrent.futures.as_completed(futures)) if val[0] is not None}

        max_loss = max(val[0] for val in boot_results.values())
        max_sharpe = max(val[1] for val in boot_results.values())

        for key, (loss, sharpe,bench_sharpe) in boot_results.items():
            strategy_losses[key].append(loss)
            strategy_sharpes[key].append(sharpe)
        if max_loss > real_max_loss:
            bootstrap_loss_beats_real += 1
        if max_sharpe > real_max_sharpe:
            bootstrap_sharpe_beats_real += 1

        list_max_loss.append(max_loss)
        list_max_sharpe.append(max_sharpe)
        print('Time spend on one iteration: ',time.time()-start_time)
        print('max_loss: ',max_loss,'max_sharpe: ',max_sharpe)

        

    print(f"\nBootstraps beating real max loss: {bootstrap_loss_beats_real}/{n_iterations}")
    print(f"Bootstraps beating real max sharpe: {bootstrap_sharpe_beats_real}/{n_iterations}")

    summary_df = pd.DataFrame({
        'Strategy': list(strategy_losses.keys()),
        'AvgLoss': [np.mean(v) for v in strategy_losses.values()],
        'AvgSharpe': [np.mean(v) for v in strategy_sharpes.values()]
    })
    summary_df.to_csv('C:/Users/Joachim Mencke/Desktop/bachelor/strategy_summary.csv', index=False)

    plt.figure(figsize=(10, 6))
    plt.hist(summary_df['AvgLoss'], bins=30, edgecolor='black')
    plt.xlabel('Afkast')
    plt.ylabel('Antal')
    plt.title('Gennemsnitligs afkast for hver strategi i bootstrap')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('strategy_avg_performance_histogram.png')
    #plt.show()

    plt.figure(figsize=(10, 6))
    plt.hist(list_max_loss, bins=30, edgecolor='black')
    plt.axvline(x=real_max_loss, color='red', linestyle='--', linewidth=2, label=f'Max afkast i reel data = {real_max_loss}')
    plt.xlabel('Max afkast')
    plt.ylabel('Antal')
    plt.title('Histogram over gennemsnitligt max afkast for en strategi i hver iteration')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('strategy_avg_performance_histogram.png')
    #plt.show()

    plt.figure(figsize=(10, 6))
    plt.hist(list_max_sharpe, bins=30, edgecolor='black')
    plt.axvline(x=real_max_sharpe, color='red', linestyle='--', linewidth=2, label=f'Max sharpe i reel data = {real_max_sharpe}')
    plt.xlabel('Max sharpe_ratio')
    plt.ylabel('Antal')
    plt.title('Histogram over gennemsnitligt max sharpe_ratio for en strategi i hver iteration')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('strategy_avg_performance_histogram.png')
    plt.show()