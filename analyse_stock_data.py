from read_data import adj_close   

# Question 1: Were there any significant price jumps or drops during the year, potentially indicating important events or announcements?

daily_pct_change = adj_close.pct_change() * 100
threshold = 2 * daily_pct_change.std()
significant_changes = daily_pct_change[(daily_pct_change > threshold) | (daily_pct_change < -threshold)]
significant_changes_readable = significant_changes.dropna(how='all').stack().reset_index()
significant_changes_readable.columns = ['Date', 'Stock', 'Percentage Change']


# Cluster price jumps and price drops
price_jumps = significant_changes[significant_changes > 0]
price_drops = significant_changes[significant_changes < 0]

window_size = 365
jumps_count = price_jumps.rolling(window=window_size, min_periods=1).count()
drops_count = price_drops.rolling(window=window_size, min_periods=1).count()


cluster_threshold = 5  # Adjust this value based on your preference


jumps_cluster_dates = jumps_count[jumps_count.sum(axis=1) >= cluster_threshold * window_size].index
drops_cluster_dates = drops_count[drops_count.sum(axis=1) >= cluster_threshold * window_size].index


jumps_cluster = adj_close.loc[jumps_cluster_dates]
drops_cluster = adj_close.loc[drops_cluster_dates]


print(jumps_cluster.head())
print(drops_cluster.head())


# Question 2: What was the overall return on investment (ROI) for each of the top 10 stocks during 2022?
start_date = adj_close.index.min()
end_date = adj_close.index.max()

start_price = adj_close.loc[start_date]
end_price = adj_close.loc[end_date]

roi = (end_price - start_price) / start_price * 100


# Question 3: Which stock(s) had the highest and lowest volatility during the year?

daily_pct_change = adj_close.pct_change()
volatility = daily_pct_change.std() * 100
highest_volatility_stock = volatility.idxmax()
lowest_volatility_stock = volatility.idxmin()















# Question 4: Did any stocks have a particularly high or low volume during the year?