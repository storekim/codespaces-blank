import backtrader as bt
import pandas as pd
import numpy as np
import random as rd
import itertools
import time
import concurrent.futures
import matplotlib.pyplot as plt
import ast
import seaborn as sns

class TestStrategy(bt.Strategy):
    
    params = dict(ma=200,std=1,rsi_period=14,rsi_value=50,rsi_overbought_or_oversold=0,interest_series=None)

    def __init__(self):
        self.sma = bt.indicators.SMA(period=self.p.ma)
        self.std = bt.indicators.StandardDeviation(period=self.p.ma)
        self.rsi = bt.indicators.RSI(period=self.p.rsi_period,safediv=True)
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


def run_backtest(ma, std, rsi_period, rsi_value, rsi_overbought_or_oversold, data, interest_series):
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

    #Strategi afkast  og Sharpe-ratio udregnes her kan ganges med 252**0.5 for at få årlig sharpe
    log_returns = np.log10(account_values[1:]) - np.log10(account_values[:-1])
    excess_returns=np.mean(log_returns)-np.mean(np.log10(result.account_rates[:-1]))
    std=np.std(log_returns)
    annual_sharpe_ratio=excess_returns/std

    #Benchmark afkast of Sharpe-ratio udregnes her kan ganges med 252**0.5 for at få årlig sharpe
    log_returns_bench  = np.log10(data['close'].values[1:] / data['close'].values[:-1])
    excess_returns_bench=np.mean(log_returns_bench)-np.mean(np.log10(result.account_rates[:-1]))
    std_bench=np.std(log_returns_bench)
    annual_sharpe_ratio_bench=excess_returns_bench/std_bench

    #Forskellen mellem de to returnes til datahåndtering
    return_diff=10**((np.mean(log_returns)-np.mean(log_returns_bench))*252)-1
    sharpe_diff=annual_sharpe_ratio-annual_sharpe_ratio_bench

    return return_diff, sharpe_diff


def evaluate_strategy(params, data, interest_series):
    ma, std, rsi_period, rsi_value, rsi_type = params
    try:
        return (str(params), run_backtest(ma, std, rsi_period, rsi_value, rsi_type, data.copy(), interest_series))
    except Exception as e:
        print(f"Error in strategy {params}: {e}")
        return (str(params), (None, None))


if __name__ == '__main__':
    n_iterations = 500

    real_data = pd.read_csv(spx_d.csv', usecols=[0, 4, 1])
    real_data['datetime'] = pd.to_datetime(real_data['datetime'], dayfirst=True)

    start_date = '1954-01-01'
    end_date = '2025-02-05'

    real_data=real_data[(real_data['datetime'] >= start_date) & (real_data['datetime'] <= end_date)].sort_values(by='datetime').reset_index(drop=True)
    dtb = pd.read_csv('DTB3.csv', parse_dates=['observation_date'])
    dtb['observation_date'] = pd.to_datetime(dtb['observation_date'],dayfirst=True)
    dtb.set_index('observation_date', inplace=True)
    interest_series = dtb['DTB3'].ffill()
    
    ma_period = [5, 20, 50, 200]
    std_dev = [-1, 0, 1]
    rsi_periods = [7, 14, 28]
    rsi_values = [50, 70]
    rsi_types = [0, 1, 2]

    all_strategies = list(itertools.product(ma_period, std_dev, rsi_periods, rsi_values, [0, 1]))
    all_strategies += list(itertools.product(ma_period, std_dev, [14], [14], [2]))
    
    strategy_profites, strategy_sharpes = {}, {}
    real_results_list=[]

    print("\nRunning strategies on real data...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(evaluate_strategy, params, real_data, interest_series): params for params in all_strategies}
        real_results = {key: val for key, val in (f.result() for f in concurrent.futures.as_completed(futures)) if val[0] is not None}

    for key, (profit, sharpe) in real_results.items():
        strategy_profites[key], strategy_sharpes[key] = [profit], [sharpe]
        print(f"{key}: profit={profit:.6f}, Sharpe={sharpe:.4f}",)
        real_results_list.append([key,profit, sharpe])

    real_results_dataframe=(pd.DataFrame(real_results_list,columns=['strategy','profit','sharpe']))
    real_results_dataframe['strategy'] = real_results_dataframe['strategy'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    real_max_profit = max([v[0] for v in real_results.values()])
    real_max_sharpe = max([v[1] for v in real_results.values()])
    print(f"\nMax Real Sharpe: {real_max_sharpe:.4f}, Max Real profitFunc: {real_max_profit:.4f}")

    bootstrap_data = {}
    print(f"\nGenerating {n_iterations} bootstrap datasets in parallel...")
    start_bootstrap_time = time.time()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_index = {
            executor.submit(make_bootstrap_data, real_data): n for n in range(n_iterations)
        }
        bootstrap_data = {}
        for future in concurrent.futures.as_completed(future_to_index):
            n = future_to_index[future]
            try:
                bootstrap_data[f'bdata{n}'] = future.result()
            except Exception as e:
                print(f"Error generating bdata{n}: {e}")
    print(f"Finished bootstrap generation in {time.time() - start_bootstrap_time:.2f} seconds.")

    all_boot_data=[]
    for strategy in (all_strategies):
        start_time=time.time()
        current_data=[]
        print(f"\n--- Evaluating Strategy {all_strategies.index(strategy)+1}/{len(all_strategies)}: '{strategy}' over {n_iterations} Bootstraps ---")

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(evaluate_strategy, strategy, data, interest_series): name for name, data in bootstrap_data.items()}

        boot_results = []

        for future in concurrent.futures.as_completed(futures):
            name = futures[future]  # Få navn på bootstrap (f.eks., 'bdata43')
            try:
                result = future.result()
                if result is not None:
                    strategy=result[0]
                    profit=result[1][0]
                    sharpe=result[1][1]
                    boot_results.append([strategy, name,profit, sharpe]) 

            except Exception as e:
                print(f"Error processing {name}: {e}")

        all_boot_data.extend(boot_results)
        print(f"Time taken for strategy: {time.time() - start_time:.2f} seconds")

    full_boot_dataframe=(pd.DataFrame(all_boot_data,columns=['strategy','bdata','profit','sharpe']))
    full_boot_dataframe['strategy'] = full_boot_dataframe['strategy'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    full_boot_dataframe.to_csv(r'C:\Users\Joachim Mencke\Desktop\bachelor\data_folder\bootstrap_strategy_data{}.csv'.format(round(time.time())))
    real_results_dataframe.to_csv(r'C:\Users\Joachim Mencke\Desktop\bachelor\data_folder\real_strategy_data{}.csv'.format(round(time.time())))

    p_list_profit=[];p_list_sharpe=[];max_profit_list=[];max_sharpe_list=[];list_strategy_returns = [];list_strategy_sharpes = [];strategy_list=[]
    bootstrap_sharpes=[]

    for idx, strategy in enumerate(all_strategies):
        print(f"Evaluating strategy {idx + 1}/{len(all_strategies)}: {strategy}")
        #Der laves en liste med de strategier der evalueres(først evalueres 1 strategi, så 2, så 3 osv. op til antallet af strategier)
        strategy_list.append(strategy)
        boot_higher_profit=0 ; boot_higher_sharpe=0
        current_max_profit=float('-inf') ; current_max_sharpe = float('-inf')
        #Der bliver skabt et nyt datasæt med kun de strategier der bliver evalueret
        filtered_dataframe = full_boot_dataframe[full_boot_dataframe['strategy'].apply(lambda x: x in strategy_list)]
        filtered_real_dataframe=real_results_dataframe[real_results_dataframe['strategy'].apply(lambda x: x in strategy_list)]
        #Der skal sammenlignes med det bedste resultat i det oprindelige data så det findes i det nu mindre univers
        real_max_profit=filtered_real_dataframe['profit'].max()
        real_max_sharpe=filtered_real_dataframe['sharpe'].max()
        #Dette holder øje med når en nu bedste strategi filføjes til universet så dette kan plottes senere
        if real_max_profit>current_max_profit:
                current_max_profit=real_max_profit
        if real_max_sharpe > current_max_sharpe:
            current_max_sharpe = real_max_sharpe
        #Dette laver en liste over resultater for hver strategi som også bruges til at plotte senere
        list_strategy_returns.append(real_results_dataframe[real_results_dataframe['strategy'].apply(lambda x: x==strategy)]['profit'].values[0])
        list_strategy_sharpes.append(real_results_dataframe[real_results_dataframe['strategy'].apply(lambda x: x == strategy)]['sharpe'].values[0])
        
        #Denne løkke sammen med det efterfølgende regner p-værdien for kun de  strategier der evalueres
        for n in range(n_iterations):
            max_profit_bdatan = filtered_dataframe[filtered_dataframe['bdata'] == 'bdata{}'.format(n)]['profit'].max()
            max_sharpe_bdatan = filtered_dataframe[filtered_dataframe['bdata'] == 'bdata{}'.format(n)]['sharpe'].max()

            if idx==0:
                bootstrap_sharpes.append(max_sharpe_bdatan)

            if max_profit_bdatan>real_max_profit:
                boot_higher_profit+=1
            if max_sharpe_bdatan>real_max_sharpe:
                boot_higher_sharpe+=1

        p_list_profit.append(boot_higher_profit/(n_iterations))
        max_profit_list.append(current_max_profit)

        p_list_sharpe.append(boot_higher_sharpe/(n_iterations))
        max_sharpe_list.append(current_max_sharpe)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(range(len(all_strategies)), p_list_profit, label='P-værdi for profit kriteriet', color='#f87c7c')
    ax1.set_xlabel('Strategi nummer x')
    ax1.set_ylabel('P-værdi', color='#f87c7c', fontsize=13)
    ax1.tick_params(axis='y', labelcolor='black')

    ax2 = ax1.twinx()  
    ax2.plot(range(len(all_strategies)), max_profit_list, label='Maks Profit', color='#6fa3f8')
    ax2.set_ylabel('Maks profit', color='#6fa3f8', fontsize=13)
    ax2.tick_params(axis='y', labelcolor='black')

    ax2.scatter(range(len(all_strategies)),list_strategy_returns, color='black', label='Reel afkast for strategi x', zorder=5,marker='x',alpha=0.6,s=30)

    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.title('Whites Reality Check for afkast over antal af x strategier')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    plt.tight_layout() 
    plt.grid(True)

    # --- Plot for Sharpe ---
    fig, ax3 = plt.subplots(figsize=(10, 6))

    ax3.plot(range(len(all_strategies)), p_list_sharpe, label='P-værdi for Sharpe', color='#e7a87c')
    ax3.set_xlabel('Strategi nummer x')
    ax3.set_ylabel('P-værdi', color='#e7a87c', fontsize=13)
    ax3.tick_params(axis='y', labelcolor='black')

    ax4 = ax3.twinx()
    ax4.plot(range(len(all_strategies)), max_sharpe_list, label='Maks Sharpe', color='#7cb9e7')
    ax4.scatter(range(len(all_strategies)), list_strategy_sharpes, color='black', label='Reel Sharpe for strategi x', marker='x', alpha=0.6, s=30, zorder=5)
    ax4.set_ylabel('Maks Sharpe', color='#7cb9e7', fontsize=13)
    ax4.tick_params(axis='y', labelcolor='black')

    ax3.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    lines3, labels3 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax4.get_legend_handles_labels()
    ax3.legend(lines3 + lines4, labels3 + labels4, loc='upper left', fontsize=10)

    plt.title('Whites Reality Check for Sharpe-ratio over x antal af strategier')
    plt.tight_layout()
    plt.show()
