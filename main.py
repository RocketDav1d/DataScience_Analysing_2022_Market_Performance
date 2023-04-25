import streamlit as st
import pandas as pd
#from analyse_stock_data import volume, adj_close, economic_calendar, dji, sp500, nasdaq
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

def plot_volume_chart(adj_close_data, selected_stocks):
    fig = go.Figure()

    for column in selected_stocks:
        fig.add_trace(go.Scatter(x=adj_close_data.index, y=adj_close_data[column], name=column, mode='lines'))

    fig.update_layout(title='Stock Adjusted Close Price Over Time',
                      xaxis_title='Date',
                      yaxis_title='Adjusted Close Price',
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
    
    # fig = go.Figure()

    # index_data_df = index_data.reset_index().rename(columns={'index': 'Date', 'Adj Close': 'Adj Close_index'})
    # stock_data_df = stock_data.reset_index().rename(columns={'index': 'Date', 'Adj Close': 'Adj Close_stock'})

    # merged_data = index_data_df.merge(stock_data_df, on='Date', suffixes=('_index', '_stock'))

    # # Normalize the prices by dividing them by their initial prices
    # normalized_index_data = merged_data['Adj Close_index'] / merged_data['Adj Close_index'].iloc[0]
    # normalized_stock_data = merged_data[selected_stock] / merged_data[selected_stock].iloc[0]

    # fig.add_trace(go.Scatter(x=merged_data['Date'], y=normalized_index_data, name="Index", mode="lines"))
    # fig.add_trace(go.Scatter(x=merged_data['Date'], y=normalized_stock_data, name="Stock", mode="lines"))

    # index_returns = merged_data['Adj Close_index'].pct_change().dropna()
    # stock_returns = merged_data[selected_stock].pct_change().dropna()

    # # Calculate the percentage returns for the normalized data
    # index_normalized_returns = normalized_index_data.pct_change().dropna()
    # stock_normalized_returns = normalized_stock_data.pct_change().dropna()

    # # Calculate the rolling correlation using normalized returns
    # # rolling_correlation = index_normalized_returns.rolling(window=80).corr(stock_normalized_returns).dropna()
    # # fig.add_trace(go.Scatter(x=merged_data['Date'][rolling_correlation.index], y=rolling_correlation, name="Rolling Correlation", mode="lines"))

    # fig.update_layout(title='Normalized Index & Stock Adj Close Price Over Time to show Correlation',
    #                   xaxis_title='Date',
    #                   yaxis_title='Price / Correlation',
    #                   legend_title='Series')
    # return fig



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
      
  # with col2:
  #     volume_chart = plot_volume_chart(stock_data, selected_stocks)
  #     #volume_chart = plot_volume_chart(volume, selected_stocks)
  #     st.plotly_chart(volume_chart)

  with col2:
      volume_chart = plot_volume_chart(adj_close, selected_stocks)
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

  def plot_volume_chart_with_events(volume_data, selected_stocks, event_dates):
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

  # Create columns to display modified plots side by side
  col5, col6 = st.columns(2)

  with col5:
      # Display the modified index chart
      index_chart_with_events = plot_index_chart_with_events(index_dict, selected_indices, interest_rate_dates)
      st.plotly_chart(index_chart_with_events)

  with col6:
      # Display the modified volume chart
      volume_chart_with_events = plot_volume_chart_with_events(volume, selected_stocks, interest_rate_dates)
      st.plotly_chart(volume_chart_with_events)









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

  def plot_volume_chart_with_events(volume_data, selected_stocks, event_dates):
      fig = go.Figure()

      for column in selected_stocks:
          fig.add_trace(go.Scatter(x=volume_data.index, y=volume_data[column], name=column, mode='lines'))


          event_y = volume_data[volume_data.index.isin(event_dates)][column]
          fig.add_trace(go.Scatter(x=event_dates, y=event_y, mode="markers", marker=dict(color="black", size=6), name="GDP Growth Report"))

      fig.update_layout(title='Stock Volume Over Time with ',
                        xaxis_title='Date',
                        yaxis_title='Volume',
                        legend_title='Stocks')

      return fig

  # Create columns to display modified plots side by side
  col7, col8 = st.columns(2)

  with col7:
      # Display the modified index chart
      index_chart_with_events = plot_index_chart_with_events(index_dict, selected_indices, gdp_dates)
      st.plotly_chart(index_chart_with_events)

  with col8:
      # Display the modified volume chart
      volume_chart_with_events = plot_volume_chart_with_events(volume, selected_stocks, gdp_dates)
      st.plotly_chart(volume_chart_with_events)









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























































# st.title("Analysis of market performance in relation to economic events")

# # 3 Market performance portrayed by 3 major indexes
# index_dict = {
#     "Dow Jones Industrial Average": dji,
#     "S&P 500": sp500,
#     "NASDAQ Composite": nasdaq
# }
# st.subheader("Market performance portrayed by 3 major indexes ")
# selected_indices = st.multiselect("Select the indices to plot", options=list(index_dict.keys()), default=list(index_dict.keys()))


# def plot_index_chart(index_data_dict, selected_indices):
#     fig = go.Figure()

#     for index_name in selected_indices:
#         data = index_data_dict[index_name]
#         fig.add_trace(go.Scatter(x=data["date"], y=data["Adj Close"], name=index_name, mode="lines"))

#     fig.update_layout(title='Market Performance of Major Indices',
#                       xaxis_title='Date',
#                       yaxis_title='Adjusted Close Price',
#                       legend_title='Indices')

#     return fig

# index_chart = plot_index_chart(index_dict, selected_indices)
# st.plotly_chart(index_chart)


# #  investor interst in 10 most bought stocks defined by volume

# st.subheader("investor interst in stocks defined by volume")
# selected_stocks = st.multiselect("Select stocks to display:", volume.columns.tolist(), default=volume.columns.tolist())

# def plot_volume_chart(volume_data, selected_stocks):
#     fig = go.Figure()

#     for column in selected_stocks:
#         fig.add_trace(go.Scatter(x=volume_data.index, y=volume_data[column], name=column, mode='lines'))

#     fig.update_layout(title='Stock Volume Over Time',
#                       xaxis_title='Date',
#                       yaxis_title='Volume',
#                       legend_title='Stocks')

#     return fig

# volume_chart = plot_volume_chart(volume, selected_stocks)
# st.plotly_chart(volume_chart)





























# # Question 1: Were there any significant price jumps or drops during the year, potentially indicating important events or announcements?

# def plot_price_jump_drop_chart(adj_close_data, significant_changes_data, selected_stocks):
#     fig = go.Figure()

#     for column in selected_stocks:
#         fig.add_trace(go.Scatter(x=adj_close_data.index,
#                                  y=adj_close_data[column],
#                                  name=column,
#                                  mode='lines'))

#         significant_changes_for_stock = significant_changes_data[column].dropna()
#         jumps = significant_changes_for_stock > 0
#         drops = significant_changes_for_stock < 0

#         fig.add_trace(go.Scatter(x=significant_changes_for_stock[jumps].index,
#                                  y=adj_close_data.loc[significant_changes_for_stock[jumps].index, column],
#                                  name=f"{column} Price Jump",
#                                  mode='markers',
#                                  marker=dict(size=8, symbol='triangle-up', color='green')))

#         fig.add_trace(go.Scatter(x=significant_changes_for_stock[drops].index,
#                                  y=adj_close_data.loc[significant_changes_for_stock[drops].index, column],
#                                  name=f"{column} Price Drop",
#                                  mode='markers',
#                                  marker=dict(size=8, symbol='triangle-down', color='red')))

#     fig.update_layout(title='Stock Price Jumps and Drops',
#                       xaxis_title='Date',
#                       yaxis_title='Adjusted Close Price',
#                       legend_title='Stocks')

#     return fig



# price_jump_drop_chart = plot_price_jump_drop_chart(adj_close, significant_changes, selected_stocks)
# st.plotly_chart(price_jump_drop_chart)


# # Question 2: What was the overall return on investment (ROI) for each of the top 10 stocks during 2022?

# import plotly.graph_objs as go

# def plot_roi_chart(roi_data):
#     fig = go.Figure()
#     fig.add_trace(go.Bar(x=roi_data.index, y=roi_data, text=roi_data, textposition='auto'))
#     fig.update_layout(title='Return on Investment (ROI) for Top 10 Stocks in 2022',
#                       xaxis_title='Stock',
#                       yaxis_title='ROI (%)',
#                       xaxis_tickangle=-45)
#     return fig

# roi_chart = plot_roi_chart(roi)
# st.plotly_chart(roi_chart)



# # Question 3: Which stock(s) had the highest and lowest volatility during the year?

# st.write(f"The stock with the highest volatility during the year is {highest_volatility_stock} with a volatility of {volatility[highest_volatility_stock]:.4f}.")
# st.write(f"The stock with the lowest volatility during the year is {lowest_volatility_stock} with a volatility of {volatility[lowest_volatility_stock]:.4f}.")

# import plotly.graph_objs as go

# def plot_volatility_chart(volatility_data):
#     fig = go.Figure()
#     fig.add_trace(go.Bar(x=volatility_data.index, y=volatility_data, text=volatility_data, textposition='auto'))
#     fig.update_layout(title='Volatility for Top 10 Stocks in 2022',
#                       xaxis_title='Stock',
#                       yaxis_title='Volatility',
#                       xaxis_tickangle=-45)
#     return fig

# volatility_chart = plot_volatility_chart(volatility)
# st.plotly_chart(volatility_chart)
