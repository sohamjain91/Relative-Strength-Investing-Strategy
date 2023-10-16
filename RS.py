import datetime
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
from jugaad_data.nse import NSELive

import zipfile
import os
import warnings

warnings.filterwarnings("ignore")


def extract_zip(zip_file_path, extracted_dir):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_dir)


def nifty_ret():
    todays_date = datetime.datetime.now().date()
    start_date = todays_date - datetime.timedelta(days=200)

    s = yf.download("^NSEI", start=start_date)
    s = s['Adj Close']
    s = s.sort_index()
    s = s.to_frame()

    s = s.iloc[[-1, -6, -22, -64]].T
    s['week_ret'] = s.iloc[:, 0] / s.iloc[:, 1]
    s['month_ret'] = s.iloc[:, 0] / s.iloc[:, 2]
    s['quarter_ret'] = s.iloc[:, 0] / s.iloc[:, 3]

    returns = ["nifty_ret", s['week_ret'].sum(), s['month_ret'].sum(), s['quarter_ret'].sum()]

    return returns


def stocks_selector(stocks_dict, market_cap_dict, sector):
    df_sectors = return_analysis(stocks_dict, market_cap_dict, sector)

    df_rets = df_sectors[1]
    df_full_df = df_sectors[0]
    indices_returns_df = df_sectors[2]

    df_nifty = nifty_ret()

    sectors = []
    stocks = []
    rs_score = []

    if (df_rets[1] > df_nifty[1]) & (df_rets[2] > df_nifty[2]) & (df_rets[3] > df_nifty[3]):

        try:
            sectors.append(df_rets[0])

            df_full_df['position'] = np.where((df_full_df['week_ret'] > df_rets[1]) &
                                              (df_full_df['month_ret'] > df_rets[2]) &
                                              (df_full_df['quarter_ret'] > df_rets[3]),
                                              1, 0)

            df_full_df = df_full_df[df_full_df['position'] == 1]

            df_full_df['rs_score'] = (df_full_df['week_ret'] / df_rets[1]) - 1 + (
                    df_full_df['month_ret'] / df_rets[2]) - 1 + (df_full_df['quarter_ret'] / df_rets[3]) - 1 + (
                                             df_rets[1] / df_nifty[1]) - 1 + (df_rets[2] / df_nifty[2]) - 1 + (
                                             df_rets[3] / df_nifty[3]) - 1

            stocks.append(list(df_full_df.index))
            rs_score.append(df_full_df['rs_score'].to_list())

            df = pd.concat([pd.DataFrame(sectors).T, pd.DataFrame(stocks).T, pd.DataFrame(rs_score).T], axis=1)
            df = df.fillna(method="ffill")

        except Exception as e:

            print('hello')

    else:
        df = 'NA'

    return df_rets, df, indices_returns_df


def return_analysis(stocks_dict, market_cap_dict, sector):
    s = yf.download(stocks_dict[sector])
    s = s['Adj Close']
    s = s.sort_index()
    f = s / s.shift(1)
    indices_df = f.T.mean().fillna(1).to_frame().rename(columns={0: 'daily_nav'})
    s = s.iloc[[-1, -6, -22, -64]].T
    s['week_ret'] = s.iloc[:, 0] / s.iloc[:, 1]
    s['month_ret'] = s.iloc[:, 0] / s.iloc[:, 2]
    s['quarter_ret'] = s.iloc[:, 0] / s.iloc[:, 3]
    market_cap_df = pd.DataFrame(market_cap_dict[sector], stocks_dict[sector])
    s = pd.concat([s, market_cap_df], axis=1)
    s = s.dropna()
    s = s.rename(columns={0: 'market_cap'})
    s['weight'] = s['market_cap'] / s['market_cap'].sum()
    s['weighted_weekly_ret'] = s['week_ret'] * s['weight']
    s['weighted_month_ret'] = s['month_ret'] * s['weight']
    s['weighted_quarter_ret'] = s['quarter_ret'] * s['weight']

    sector_index_ret = [sector, s['weighted_weekly_ret'].sum(), s['weighted_month_ret'].sum(),
                        s['weighted_quarter_ret'].sum()]

    return s, sector_index_ret, indices_df


def _run():
    n = NSELive()
    zip_file_path = "StocksAndSectors.zip"
    extracted_dir = "extracted_files"
    extract_zip(zip_file_path, extracted_dir)
    path_address = os.listdir('extracted_files/StocksAndSectors')
    path_address = ['extracted_files/StocksAndSectors/' + i for i in path_address]
    pd.read_csv(path_address[3], skiprows=5).head()
    data = pd.read_csv(path_address[1], skiprows=5)
    for i in path_address[2:]:
        temp_data = pd.read_csv(i, skiprows=5)
        data = pd.concat([data, temp_data], axis=0)
    data = data.drop_duplicates()
    data = data[data['Market Cap(Rs. Cr.)'] > 500]
    numeric_pattern = r'^\d+$'
    data = data[~data['Symbol'].str.match(numeric_pattern)]
    categories = data['Industry'].value_counts()
    categories = categories[categories > 5]
    data = data[data['Industry'].isin(categories.index)]
    data['Industry'].value_counts().tail()
    data = data.drop(columns=data.columns[data.isna().sum() > 1000])
    list(data[data['Industry'] == 'Paper & Paper Products'].Symbol.values + ".NS")
    stocks_list = {}
    market_cap = {}
    for i in categories.index:
        stocks_list[i] = list(data[data['Industry'] == i].Symbol.values + ".NS")
        market_cap[i] = list(data[data['Industry'] == i].loc[:, 'Market Cap(Rs. Cr.)'].values)
    s = yf.download(stocks_list['Paper & Paper Products'])
    s = s['Adj Close']
    s = s.sort_index()
    s = s / s.shift(1)
    sector_analysis = []
    sector_returns = []
    sector_indices = {}
    for i in stocks_list.keys():
        try:
            lists = stocks_selector(stocks_list, market_cap, i)
            sector_analysis.append(lists[1])
            sector_returns.append(lists[0])
            sector_indices[i] = lists[2]
        except Exception:
            print('empty')
    sector_returns_df = pd.DataFrame(sector_returns, columns=['sectors', 'week_ret', 'monthly_ret', 'quarterly_ret'])
    sector_returns_df = sector_returns_df.set_index('sectors')
    sector_returns_df = (sector_returns_df - 1) * 100
    filtered_sector_analysis = []
    for i in sector_analysis:
        if type(i) == str:
            pass
        else:
            filtered_sector_analysis.append(i)
    final_df = pd.concat(filtered_sector_analysis).dropna()
    final_df.columns = ['sector', 'stock_name', 'RS']
    final_df.nlargest(20, columns='RS')
    final_df.columns = pd.MultiIndex.from_tuples([('Sector Name', col) for col in final_df.columns])
    # final_df.nlargest(20, columns='RS').sector.value_counts()

    output_path = r"C:\Users\soham\relative strength strategy\final_datas"
    with open(f"sector_analysis.pkl", "wb") as f:
        pickle.dump(sector_analysis, f)

    with open(f"sector_returns.pkl", "wb") as f:
        pickle.dump(sector_returns, f)

    with open(f"sector_indices.pkl", "wb") as f:
        pickle.dump(sector_indices, f)

    with open(f"last_run.txt", "w") as f:
        f.write(str(datetime.datetime.now()))


def run():
    time_rn = datetime.datetime.now()
    if ((not os.path.exists('last_run.txt')) or
            (time_rn - datetime.datetime.strptime(open('last_run.txt').read(),
                                                  '%Y-%m-%d %H:%M:%S.%f') > datetime.timedelta(days=1)) or (
                    datetime.datetime.now().hour > 23)):
        _run()
    else:
        print('No need to run')
        pass


if __name__ == '__main__':
    run()
