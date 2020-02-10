# Stock Price Forecasting

We use an LSTM on data from Quandl to predict the next 10 days stock prices using data on the previous 300 days. 
We use two different stocks, ORA (orange) and TFI (television francaise),
from the euronext Paris stock exchange. The datasets contain the opening, high, low and closing prices as well as volume 
of each stock for each day from Jan 1 2007 until Nov 30 2018. The data is of course fairly limited, so getting a predicted price which is in the
right range of prices was really all that could be expectd. 

![](https://github.com/collinb9/Data-Science-projects/blob/master/StockPriceForecasting/images/ORA_84_predicted_price.png "Sample ORA prediction")

![](https://github.com/collinb9/Data-Science-projects/blob/master/StockPriceForecasting/images/TFI_33_predicted_price.png "Sample TFI prediction")

The above are two example predictions, one for ORA and one for TFI. Graphs of all prediction are in the [images](https://github.com/collinb9/Data-Science-projects/tree/master/StockPriceForecasting/images) folder.
In general, the model seems to predict the price of ORA better than TFI.
