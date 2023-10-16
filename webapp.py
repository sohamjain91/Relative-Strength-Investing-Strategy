import pandas as pd
import seaborn as sns
import datetime
from jugaad_data.nse import NSELive

import warnings
from RS import run
import streamlit as st
import pickle
from matplotlib.colors import Normalize
import plotly.express as px
warnings.filterwarnings("ignore")
n = NSELive()
run()
input_path = ""

with open(f"{input_path}sector_analysis.pkl", "rb") as f:
    sector_analysis = pickle.load(f)

with open(f"{input_path}sector_returns.pkl", "rb") as f:
    sector_returns = pickle.load(f)

with open(f"{input_path}sector_indices.pkl", "rb") as f:
    sector_indices = pickle.load(f)

sector_returns_df = pd.DataFrame(sector_returns, columns=['sectors', 'week_ret', 'monthly_ret', 'quarterly_ret'])
sector_returns_df = sector_returns_df.set_index('sectors')
sector_returns_df = (sector_returns_df - 1) * 100
sector_returns_df = sector_returns_df[sector_returns_df['week_ret'] != -100]

filtered_sector_analysis = []
for i in sector_analysis:
    if type(i) == str:
        pass
    else:
        filtered_sector_analysis.append(i)

final_df = pd.concat(filtered_sector_analysis).dropna()

final_df.columns = ['sector', 'stock_name', 'RS']

final_df.nlargest(20, columns='RS').sector.value_counts()

st.title('Relative Strength Strategy')
option = st.sidebar.selectbox('Task',
                              options=('Top RS Stocks', 'Sectors Analysis', 'Sector Indices'))

if option == 'Top RS Stocks':

    Analysis = st.sidebar.selectbox('Stock Analysis',
                                    options=('Stocks', 'Stocks Sector Distribution'))

    if Analysis == 'Stocks':
        st.header('Top RS Stocks')
        number_of_stock = st.sidebar.slider('Number of Stocks', value=20, max_value=50)
        st.write(final_df.nlargest(number_of_stock, columns='RS').reset_index(drop=True))

    if Analysis == 'Stocks Sector Distribution':
        st.header('Stocks Sector Distribution')
        number_of_stock = st.sidebar.slider('Number of Stocks', value=20, max_value=50)
        st.write(final_df.nlargest(number_of_stock, columns='RS').sector.value_counts())

if option == 'Sectors Analysis':
    st.header('Sector Return Analysis')
    norm = Normalize(vmin=-5, vmax=25)
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    styled_df = sector_returns_df.style.background_gradient(
        cmap=cmap)  # , subset=['Col1', 'Col2', 'Col3'], vmin=-5, vmax=25)
    st.write(styled_df)

if option == 'Sector Indices':

    st.header('Sector Indices Equity Curves')

    sectors_select = st.sidebar.multiselect('Sectors',
                                            options=list(sector_indices.keys()),
                                            default=list(final_df.sector.value_counts().index[:5]))

    start_date = st.sidebar.date_input('Start Date', value=datetime.date(2022, 1, 1))
    end_date = st.sidebar.date_input('End Date')

    df = sector_indices[sectors_select[0]].rename(columns={"daily_nav": sectors_select[0]})
    df = df.loc[pd.Timestamp(start_date):]
    df = df.cumprod()

    for i in range(1, len(sectors_select)):
        df1 = sector_indices[sectors_select[i]].rename(columns={"daily_nav": sectors_select[i]})
        df1 = df1.loc[pd.Timestamp(start_date):]
        df1 = df1.cumprod()
        df = pd.concat([df, df1], axis=1)

    fig = px.line(df, title='Indices Equity Curve')
    st.plotly_chart(fig)

    st.subheader('Top Stocks')
    number_of_stock = st.slider('Number of Stocks', value=5, max_value=20)

    df_final_stocks = final_df[final_df['sector'] == sectors_select[0]].sort_values(by='RS').nlargest(number_of_stock,
                                                                                                      columns='RS').reset_index(
        drop=True)
    df_final_stocks = df_final_stocks.iloc[:, [-2, -1]]
    df_final_stocks.columns = pd.MultiIndex.from_tuples([(sectors_select[0], col) for col in df_final_stocks.columns])

    for i, sector in enumerate(sectors_select[1:], start=1):
        df_final_stocks1 = final_df[final_df['sector'] == sector].sort_values(by='RS').nlargest(number_of_stock,
                                                                                                columns='RS').reset_index(
            drop=True)
        df_final_stocks1 = df_final_stocks1.iloc[:, [-2, -1]]
        df_final_stocks1.columns = pd.MultiIndex.from_tuples([(sector, col) for col in df_final_stocks1.columns])
        df_final_stocks = pd.concat([df_final_stocks, df_final_stocks1], axis=1)

    st.dataframe(df_final_stocks.T)
