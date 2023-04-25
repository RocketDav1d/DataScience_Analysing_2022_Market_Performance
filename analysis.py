from read_data import adj_close, volume, economic_calendar, dji, sp500, nasdaq

columns_to_display = ["Event", "Time", "Country", "Actual", "Consensus", "Previous", "Date"]



# interest rate decision
filtered_ec = economic_calendar.query('Event == "Fed Interest Rate Decision"')
display_interest_rates_economic_calender = filtered_ec.filter(columns_to_display)
# Reset index and remove the index name
display_interest_rates_economic_calender.reset_index(drop=True, inplace=True)


# GDP growth rate
gdp_growth_rate = economic_calendar.query('Event == "GDP"')
display_gdp_economic_calender = gdp_growth_rate.filter(columns_to_display)
# Reset index and remove the index name
display_gdp_economic_calender.reset_index(drop=True, inplace=True)
