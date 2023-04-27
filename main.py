import streamlit as st
import pandas as pd
from datetime import datetime
from read_data import volume, adj_close, economic_calendar, dji, sp500, nasdaq, stock_data, cpi_daily, cpi_monthly, cpi_daily_scaled, cpi_monthly_scaled, adj_close_reset
import plotly.graph_objs as go
from analysis import display_interest_rates_economic_calender, display_gdp_economic_calender


# Site configuration
st.set_page_config(layout="wide")
st.title("retrospective analysis of market performance in relation to economic events 2022")

index_dict = {
    "Dow Jones Industrial Average": dji,
    "S&P 500": sp500,
    "NASDAQ Composite": nasdaq
}

st.subheader("Market performance portrayed by 3 major indexes")
selected_indices = st.multiselect("Select the indices to plot", options=list(index_dict.keys()), default=list(index_dict.keys()))

st.subheader("Investor interest in 10 most bought stocks defined by volume")
selected_stocks = st.multiselect("Select stocks to display:", volume.columns.tolist(), default=volume.columns.tolist())

tab1, tab2, tab3, tab4 = st.tabs(["General", "Interest Rates", "GDP", "Inflation"])








# --------------------- Tab 1 - General Tab  ---------------------
def plot_index_chart(index_data_dict, selected_indices):
    fig = go.Figure()

    for index_name in selected_indices:
        data = index_data_dict[index_name]
        fig.add_trace(go.Scatter(x=data["date"], y=data["Adj Close"], name=index_name, mode="lines"))

    fig.update_layout(title='Market Performance of Major Indices',
                      xaxis_title='Date',
                      yaxis_title='Adjusted Close Price',
                      legend_title='Indices')
    return fig


def plot_adj_chart(adj_close_data, selected_stocks):
    fig = go.Figure()

    quarterly_dates = [
        datetime(2022, 2, 3),
        datetime(2022, 4, 28),
        datetime(2022, 7, 28),
        datetime(2022, 10, 27),
    ]

    for column in selected_stocks:
        fig.add_trace(go.Scatter(x=adj_close_data.index, y=adj_close_data[column], name=column, mode='lines'))

        for date in quarterly_dates:
          if date in adj_close_data.index:
            fig.add_shape(type='line', x0=date, x1=date, y0=0, y1=1, yref='paper', line=dict(color='black', width=1))


    fig.update_layout(title='Stock Adjusted Close Price Over Time',
                      xaxis_title='Date',
                      yaxis_title='Adjusted Close Price',
                      legend_title='Stocks')
    return fig


def plot_volume_chart(volume_data, selected_stocks):
    fig = go.Figure()

    quarterly_dates = [
        datetime(2022, 2, 3),
        datetime(2022, 4, 28),
        datetime(2022, 7, 28),
        datetime(2022, 10, 27),
    ]

    for column in selected_stocks:
        fig.add_trace(go.Scatter(x=volume_data.index, y=volume_data[column], name=column, mode='lines'))

        for date in quarterly_dates:
          if date in volume_data.index:
            fig.add_shape(type='line', x0=date, x1=date, y0=0, y1=1, yref='paper', line=dict(color='black', width=1))

    fig.update_layout(title='Stock Volume Over Time',
                      xaxis_title='Date',
                      yaxis_title='Volume',
                      legend_title='Stocks')
    return fig

def plot_correlation_chart(index_data, stock_data, selected_stock):
    fig = go.Figure()

    index_data_df = index_data.reset_index().rename(columns={'index': 'Date', 'Adj Close': 'Adj Close_index'})
    stock_data_df = stock_data.reset_index().rename(columns={'index': 'Date', 'Adj Close': 'Adj Close_stock'})

    merged_data = index_data_df.merge(stock_data_df, on='Date', suffixes=('_index', '_stock'))

    # Normalize the prices by dividing them by their initial prices
    normalized_index_data = merged_data['Adj Close_index'] / merged_data['Adj Close_index'].iloc[0]
    normalized_stock_data = merged_data[selected_stock] / merged_data[selected_stock].iloc[0]

    fig.add_trace(go.Scatter(x=merged_data['Date'], y=normalized_index_data, name="Index", mode="lines"))
    fig.add_trace(go.Scatter(x=merged_data['Date'], y=normalized_stock_data, name="Stock", mode="lines"))

    fig.update_layout(title='Normalized Index & Stock Adj Close Price Over Time',
                      xaxis_title='Date',
                      yaxis_title='Normalized Price',
                      legend_title='Series')

    return fig
    



def plot_daily_returns_scatter(index_data, stock_data, selected_stock):
    fig = go.Figure()

    index_data_df = index_data.reset_index().rename(columns={'index': 'Date', 'Adj Close': 'Adj Close_index'})
    stock_data_df = stock_data.reset_index().rename(columns={'index': 'Date', 'Adj Close': 'Adj Close_stock'})

    merged_data = index_data_df.merge(stock_data_df, on='Date', suffixes=('_index', '_stock'))

    index_returns = merged_data['Adj Close_index'].pct_change().dropna()
    stock_returns = merged_data[selected_stock].pct_change().dropna()

    fig.add_trace(go.Scatter(x=index_returns, y=stock_returns, mode='markers', marker=dict(size=5, opacity=0.75)))

    fig.update_layout(title='Scatter Plot of Daily Returns',
                      xaxis_title=f'{selected_index} Daily Returns',
                      yaxis_title=f'{selected_stock} Daily Returns',
                      legend_title='Daily Returns')
    return fig



def calculate_correlation(index_data, stock_data):
    # Calculate the percentage returns
    index_returns = index_data.pct_change().dropna()
    stock_returns = stock_data.pct_change().dropna()
    # Calculate the correlation coefficient between index_returns and stock_returns
    correlation = np.corrcoef(index_returns, stock_returns)[0, 1]

    return correlation




with tab1:
  col1, col2 = st.columns(2)
  with col1:
      index_chart = plot_index_chart(index_dict, selected_indices)
      st.plotly_chart(index_chart)
      

  with col2:
      a, b = st.tabs(["Adj Close", "Volume"])
      with a:
        adj_close_chart = plot_adj_chart(adj_close, selected_stocks)
        st.plotly_chart(adj_close_chart)
      with b:
        volume_chart = plot_volume_chart(volume, selected_stocks)
        st.plotly_chart(volume_chart)


  col13, col14 = st.columns(2)
  with col13:
    if len(selected_indices) == 1 and len(selected_stocks) == 1:
            selected_index = selected_indices[0]
            selected_stock = selected_stocks[0]
            selected_index_data = index_dict[selected_index]["Adj Close"]
            selected_stock_data = adj_close_reset[selected_stock]
            
            corr_chart = plot_correlation_chart(selected_index_data, selected_stock_data, selected_stock)
            st.plotly_chart(corr_chart)
            st.write("The normalized Adj Close of stock and index are plotted to show the correlation between them. ")
    else:
        st.write("Please select exactly one index and one stock to display the correlation chart.")


  with col14:
    if len(selected_indices) == 1 and len(selected_stocks) == 1:
        selected_index = selected_indices[0]
        selected_stock = selected_stocks[0]
        selected_index_data = index_dict[selected_index]["Adj Close"]
        selected_stock_data = adj_close_reset[selected_stock]

        daily_returns_scatter = plot_daily_returns_scatter(selected_index_data, selected_stock_data, selected_stock)
        st.plotly_chart(daily_returns_scatter)
        st.write("If the points in the scatter plot are tightly clustered around a straight line, it indicates that the stock's movement is highly correlated with the index.")


  import numpy as np



  if len(selected_indices) == 1 and len(selected_stocks) == 1:
      selected_index = selected_indices[0]
      selected_stock = selected_stocks[0]
      selected_index_data = index_dict[selected_index]["Adj Close"]
      selected_stock_data = adj_close_reset[selected_stock]

      # Calculate the correlation
      corr_coeff = calculate_correlation(selected_index_data, selected_stock_data)
      st.write("")
      st.write(f"The correlation coefficient between {selected_index} and {selected_stock} is {corr_coeff:.4f}")
      st.write("Conclusion: This proves the close alignmnet of single stocks performance in relation to overall market performance/performance.")






# ---------------------------- Tab 2 - interest rates ----------------------------

# Fed Interest Rate Decision events in the Economic Calendar

with tab2:
  col3, col4 = st.columns(2)

  with col3:
    st.subheader("Fed Interest Rate Decision events in the Economic Calendar")
    st.dataframe(display_interest_rates_economic_calender)

  with col4:
    def plot_interest_rate_chart(interest_rate_data):
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=interest_rate_data["Date"], y=interest_rate_data["Actual"], name="Interest Rates", mode="lines+markers"))

        fig.update_layout(title='Interest Rates Over Time',
                          xaxis_title='Date',
                          yaxis_title='Interest Rate',
                          legend_title='Fed Interest Rate Decision')

        return fig
    interest_rate_chart = plot_interest_rate_chart(display_interest_rates_economic_calender)
    st.plotly_chart(interest_rate_chart)


  st.subheader("Do the interest rate decisions have an impact on the stock market?")

  # Step 1: Extract the dates from the display_economic_calender DataFrame
  interest_rate_dates = display_interest_rates_economic_calender['Date'].tolist()

  # Step 2: Modify the plot_index_chart and plot_volume_chart functions
  def plot_index_chart_with_events(index_data_dict, selected_indices, event_dates):
      fig = go.Figure()

      for index_name in selected_indices:
          data = index_data_dict[index_name]
          fig.add_trace(go.Scatter(x=data["date"], y=data["Adj Close"], name=index_name, mode="lines"))

          # Add markers for interest rate decision dates
          event_y = data[data["date"].isin(event_dates)]["Adj Close"]
          fig.add_trace(go.Scatter(x=event_dates, y=event_y, mode="markers", marker=dict(color="black", size=6), name="Interest Rate Decision"))

      fig.update_layout(title='Market Performance of Major Indices with Interest Rate Decision Dates',
                        xaxis_title='Date',
                        yaxis_title='Adjusted Close Price',
                        legend_title='Indices')
      return fig


  def plot_volume_chart_with_events_interest_rate(volume_data, selected_stocks, event_dates):
      fig = go.Figure()

      for column in selected_stocks:
          fig.add_trace(go.Scatter(x=volume_data.index, y=volume_data[column], name=column, mode='lines'))

          # Add markers for interest rate decision dates
          event_y = volume_data[volume_data.index.isin(event_dates)][column]
          fig.add_trace(go.Scatter(x=event_dates, y=event_y, mode="markers", marker=dict(color="black", size=6), name="Interest Rate Decision"))

      fig.update_layout(title='Stock Volume Over Time with Interest Rate Decision Dates',
                        xaxis_title='Date',
                        yaxis_title='Volume',
                        legend_title='Stocks')
      return fig
  

  def plot_adj_chart_with_events_interest_rate(adj_close_data, selected_stocks, event_dates):
      fig = go.Figure()

      for column in selected_stocks:
          fig.add_trace(go.Scatter(x=adj_close_data.index, y=adj_close_data[column], name=column, mode='lines'))

          # Add markers for interest rate decision dates
          event_y = adj_close_data[adj_close_data.index.isin(event_dates)][column]
          fig.add_trace(go.Scatter(x=event_dates, y=event_y, mode="markers", marker=dict(color="black", size=6), name="Interest Rate Decision"))

      fig.update_layout(title='Stock Adj Close Over Time with Interest Rate Decision Dates',
                        xaxis_title='Date',
                        yaxis_title='Adj Close',
                        legend_title='Stocks')
      return fig

  # Create columns to display modified plots side by side
  col5, col6 = st.columns(2)

  with col5:
      # Display the modified index chart
      index_chart_with_events = plot_index_chart_with_events(index_dict, selected_indices, interest_rate_dates)
      st.plotly_chart(index_chart_with_events)

  with col6:
      c , d = st.tabs(["Adj Close", "Volume"])
      # Display the modified volume chart
      with c:
        adj_close_chart_with_events_interest_rates = plot_adj_chart_with_events_interest_rate(adj_close, selected_stocks, interest_rate_dates)
        st.plotly_chart(adj_close_chart_with_events_interest_rates)
      with d:
        volume_chart_with_events_interest_rates = plot_volume_chart_with_events_interest_rate(volume, selected_stocks, interest_rate_dates)
        st.plotly_chart(volume_chart_with_events_interest_rates)
         


  st.divider()
  st.subheader("Topic Conclusion")
  st.markdown("While there is no immediate connection between the reports of the interest rate decisions by the Fed and the stock market, we can see that the stock market performance troughout the year correlates with the interest rate decisions.")
  st.markdown("If interest rates move higher, stock investors become more reluctant to bid up stock prices because the value of future earnings looks less attractive versus bonds that pay more competitive yields today")
  st.markdown("Present value calculations of future earnings for stocks are tied to assumptions about interest rates or inflation. If investors anticipate higher rates in the future, it reduces the present value of future earnings for stocks. When this occurs, stock prices tend to face more pressure")
  st.markdown("The hardest hit stocks have primarily been those that are considerer 'pricey' from a valuation perspective, This included secular growth and technology companies that enjoyed extremely strong performance since the pandemic began.")
  st.markdown("The overall downward trend of the major market indeces and the top ten most bought stocks (which are mostly tech stocks) correlates with the decisions to rise the interest rates.")





# ---------------------------- Tab 3 - GDP Growth Rate ----------------------------

# GDP Growth Rate events in the Economic Calendar

with tab3:
  col11, col12 = st.columns(2)

  with col11:
      st.subheader("GDP growth in the Economic Calendar")
      st.dataframe(display_gdp_economic_calender)

  with col12:
      def plot_gdp_rate_chart(gdp_data):
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=gdp_data["Date"], y=gdp_data["Actual"], name="Interest Rates", mode="lines+markers"))

        fig.update_layout(title='GDP Growth Rate Over Time',
                          xaxis_title='Date',
                          yaxis_title='GDP Growth Rate',
                          legend_title='GDP Growth Rate Report')

        return fig
      
      gdp_chart = plot_gdp_rate_chart(display_gdp_economic_calender)
      st.plotly_chart(gdp_chart)


  st.subheader("Does the GDP growth show an impact on the stock market?")
 
  gdp_dates = display_gdp_economic_calender['Date'].tolist()


  def plot_index_chart_with_events(index_data_dict, selected_indices, event_dates):
      fig = go.Figure()

      for index_name in selected_indices:
          data = index_data_dict[index_name]
          fig.add_trace(go.Scatter(x=data["date"], y=data["Adj Close"], name=index_name, mode="lines"))

          
          event_y = data[data["date"].isin(event_dates)]["Adj Close"]
          fig.add_trace(go.Scatter(x=event_dates, y=event_y, mode="markers", marker=dict(color="black", size=6), name="GDP Growth Report"))

      fig.update_layout(title='Market Performance of Major Indices with GDP Growth Rate reports',
                        xaxis_title='Date',
                        yaxis_title='Adjusted Close Price',
                        legend_title='Indices')

      return fig


  def plot_volume_chart_with_events_gdp(volume_data, selected_stocks, event_dates):
      fig = go.Figure()

      for column in selected_stocks:
          fig.add_trace(go.Scatter(x=volume_data.index, y=volume_data[column], name=column, mode='lines'))


          event_y = volume_data[volume_data.index.isin(event_dates)][column]
          fig.add_trace(go.Scatter(x=event_dates, y=event_y, mode="markers", marker=dict(color="black", size=6), name="GDP Growth Report"))

      fig.update_layout(title='Stock Volume Over Time with GDP growth rate reports',
                        xaxis_title='Date',
                        yaxis_title='Volume',
                        legend_title='Stocks')

      return fig
  
  def plot_adj_close_chart_with_events_gdp(adj_close, selected_stocks, event_dates):
    fig = go.Figure()

    for column in selected_stocks:
        fig.add_trace(go.Scatter(x=adj_close.index, y=adj_close[column], name=column, mode='lines'))


        event_y = adj_close[adj_close.index.isin(event_dates)][column]
        fig.add_trace(go.Scatter(x=event_dates, y=event_y, mode="markers", marker=dict(color="black", size=6), name="GDP Growth Report"))

    fig.update_layout(title='Stock Adj Close Over Time with GDP growth rate reports ',
                      xaxis_title='Date',
                      yaxis_title='Adj Close',
                      legend_title='Stocks')

    return fig
  




  # Create columns to display modified plots side by side
  col7, col8 = st.columns(2)

  with col7:
      # Display the modified index chart
      index_chart_with_events = plot_index_chart_with_events(index_dict, selected_indices, gdp_dates)
      st.plotly_chart(index_chart_with_events)

  with col8:
      e, f = st.tabs(["Adj Close", "Volume"])
      with e:
        adj_close_chart_with_events_gdp = plot_adj_close_chart_with_events_gdp(adj_close, selected_stocks, gdp_dates)
        st.plotly_chart(adj_close_chart_with_events_gdp)
      with f:
        # Display the modified volume chart
        volume_chart_with_events_gdp = plot_volume_chart_with_events_gdp(volume, selected_stocks, gdp_dates)
        st.plotly_chart(volume_chart_with_events_gdp)


  st.divider()
  st.subheader("Topic Conclusion")
  st.markdown("Again there is no instant correlation between the report dates of the GDP dates and the stocks performance")
  st.markdown("However this was already expected as the GDP growth rate is a long term indicator and not a short term one.")
  st.markdown("It is interesting to oberseve the correlation between the rapid fall of the GDP in April 2022 and the plummiting of indeces and stocks in the same period.")
  st.markdown("The costly pandemic recovery as well as the Russia/Ucraine war, contributed to the shrinking GDP")


# ---------------------------- CPI / inflation ----------------------------



with tab4: 
  col9, col10 = st.columns(2)

  with col9:
    st.subheader("monthly inflation measeured by CPI")
    st.dataframe(cpi_monthly)

  with col10:
      def plot_daily_cpi_line_chart(daily_cpi_data, montly_cpi_data):
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=daily_cpi_data.index, y=daily_cpi_data["CPI"], mode="lines", name="Daily CPI"))
        fig.add_trace(go.Scatter(x=montly_cpi_data.index, y=montly_cpi_data["CPI"], mode="markers", marker=dict(size=8), name="Monthly CPI"))

        fig.add_trace(go.Scatter(x=daily_cpi_data.index,
                              y=[2] * len(daily_cpi_data),
                              mode='lines',
                              name='Desired CPI',
                              line=dict(color='rgba(255, 0, 0, 0.5)', width=2)))

        fig.update_layout(title="US CPI Daily Growth (Interpolated)", xaxis_title="Date", yaxis_title="CPI")

        return fig
      
      daily_cpi_line_chart = plot_daily_cpi_line_chart(cpi_daily_scaled, cpi_monthly_scaled)
      st.plotly_chart(daily_cpi_line_chart)


  col15, col16 = st.columns(2)

  def plot_index_chart_with_events_inflation(index_data_dict, selected_indices):
      fig = go.Figure()

      for index_name in selected_indices:
          data = index_data_dict[index_name]
          fig.add_trace(go.Scatter(x=data["date"], y=data["Adj Close"], name=index_name, mode="lines"))

      fig.update_layout(title='Market Performance of Major Indices with Inflation rates',
                        xaxis_title='Date',
                        yaxis_title='Adjusted Close Price',
                        legend_title='Indices')

      return fig

  def plot_volume_chart_with_events_inflation(volume_data, selected_stocks):
      fig = go.Figure()

      for column in selected_stocks:
          fig.add_trace(go.Scatter(x=volume_data.index, y=volume_data[column], name=column, mode='lines'))


      fig.update_layout(title='Stock Volume Over Time with Inflation rates',
                        xaxis_title='Date',
                        yaxis_title='Volume',
                        legend_title='Stocks')

      return fig
  
  def plot_adj_close_chart_with_events_inflation(adj_close, selected_stocks):
    fig = go.Figure()

    for column in selected_stocks:
        fig.add_trace(go.Scatter(x=adj_close.index, y=adj_close[column], name=column, mode='lines'))

    fig.update_layout(title='Stock Adj Close Over Time with Inflation rates',
                      xaxis_title='Date',
                      yaxis_title='Adj Close',
                      legend_title='Stocks')

    return fig

  with col15:
    # Display the modified index chart
    index_chart_with_events = plot_index_chart_with_events_inflation(index_dict, selected_indices)
    st.plotly_chart(index_chart_with_events)

  with col16:
    g, h = st.tabs(["Adj Close", "Volume"])
    with g:
      adj_close_chart = plot_adj_close_chart_with_events_inflation(adj_close, selected_stocks)
      st.plotly_chart(adj_close_chart)
    with h:
      # Display the modified volume chart
      volume_chart = plot_volume_chart_with_events_inflation(volume, selected_stocks)
      st.plotly_chart(volume_chart)


  st.divider()
  st.subheader("Topic Conclusion")
  st.markdown("The inflation rates go into the year 2022, already at a high level. A desired inflation rate of 2% is not in reach.")
  st.markdown("Beginning of February the rise above the the 8% mark occurs - where it stays until mid september.")
  st.markdown("The inflation levels peak in June 2022 reaching 9%")
  st.markdown("The rise of the inflation levels is a result of the pandemic recovery and the Russia/Ucraine war.")
  st.markdown("This behavior correlates with the rising GDP and fall of index and stocks prices")


st.divider()
st.subheader("General Conclusion")
st.markdown("2022 was a tumultuous year. War broke out between Russia and Ukraine, oil prices and inflation soared, wages remained low for many workers, interest rates rose and many feared the beginning of a recession.")
st.markdown("All of these factors combined to force stocks downward.")
st.markdown("These downturns particularly impacted tech. 2022 was the first year the NASDAQ saw four quarters of dropping values. It was the third-worst year for tech after 2008 and the bursting of the dot-com bubble in 2000.")
st.markdown("Figuring out why specific stocks rise and fall in price is often difficult, but it is easier to identify what influences trends across the market and sectors.")
st.markdown("While the immediate behavior of individual stocks is not only loosly connected to the market events, such as Fed interest rates decisions, GDP growth reports, and inflation reports, the overall market trends are heavily influenced by these events.")
st.divider()
st.markdown("Is is obvious to observe a clear relationship between the rising interest rates, shrinking GDP and and rising inflation.")
st.markdown("Additionally it is very interesting to observe that the economy was struggling the most in mid 2022, which is indicatded by a steep drop in GDP in April lasting until September.")
st.markdown("This behavior correlated with the rising of (already high) inflation rates, which experienced their yearly peak in June - staying above 8% from beginning of February until mid September.")
st.markdown("Also all of the stocks as well as the indexes, experineced a major slope beginning end of March, often resulting in a rock bottom in mid June.")
st.divider()
st.markdown("For indiviual stocks one of the key changing events are the quarterly earnings reporst, which are often followed by a significant change in the stock price.")
st.markdown("Special events, such as the for example the Amazon stock split, also have a significant impact on the stock price.")
st.markdown("The stock split resulted in short rising of the AMZN stock price, uplfiting itself from the mid 2022 market down for roughly a month in June")





