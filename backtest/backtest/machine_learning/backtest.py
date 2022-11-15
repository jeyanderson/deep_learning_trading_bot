import sys
sys.path.append('../..')

import pandas as pd
import ccxt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sys import argv

from utilities.backtesting import basic_single_asset_backtest,get_metrics
from utilities.get_data import get_historical_from_db

class lstm_strat():
    def __init__(
        self,
        df,
        use_short=True,
        strat=0,
    ):
        self.df = df
        self.use_short = use_short
        self.strat = strat
        
    def populateIndicators(self, show_log=False):
        # -- Clear dataset --
        df = self.df
        df.drop(columns=df.columns.difference(['open','high','low','close','volume', "predicted"]), inplace=True)
        
        # -- Populate indicators --
        df["diff_close"] = df["close"].diff()
        df["diff_predicted"] = df["predicted"].diff()
        df["next_predicted"] = df["predicted"].shift(-1)
        df["diff_predicted_next"] = df["next_predicted"] - df["predicted"]
        
        # -- Log --
        if(show_log):
            print(df)
        
        self.df = df    
        return self.df
    
    def populateBuySell(self, show_log=False): 
        df = self.df
        # -- Initiate populate --
        df["open_long_market"] = False
        df["close_long_market"] = False
        df["open_short_market"] = False
        df["close_short_market"] = False
        
        if self.strat == 0:
            # -- Populate open long market --
            df.loc[
                (df['diff_predicted'] > 0)
                , "open_long_market"
            ] = True
            
            # -- Populate close long market --
            df.loc[
                (df['diff_predicted'] < 0) 
                , "close_long_market"
            ] = True
            
            if self.use_short:
                # -- Populate open short market --
                df.loc[
                    (df['diff_predicted'] < 0) 
                    , "open_short_market"
                ] = True
                
                # -- Populate close short market --
                df.loc[
                    (df['diff_predicted'] > 0) 
                    , "close_short_market"
                ] = True

        elif self.strat == 1:
            # -- Populate open long market --
            df.loc[
                (df['diff_predicted_next'] > df['diff_close'])
                , "open_long_market"
            ] = True
            
            # -- Populate close long market --
            df.loc[
                (df['diff_close'] > df['diff_predicted_next']) 
                , "close_long_market"
            ] = True
            
            if self.use_short:
                # -- Populate open short market --
                df.loc[
                    (df['diff_close'] > df['diff_predicted_next']) 
                    , "open_short_market"
                ] = True
                
                # -- Populate close short market --
                df.loc[
                    (df['diff_predicted_next'] > df['diff_close']) 
                    , "close_short_market"
                ] = True
            
        
        # -- Log --
        if(show_log):
            print("Open LONG length :",len(df.loc[df["open_long_market"]==True]))
            print("Close LONG length :",len(df.loc[df["close_long_market"]==True]))
            print("Open SHORT length :",len(df.loc[df["open_short_market"]==True]))
            print("Close SHORT length :",len(df.loc[df["close_short_market"]==True]))
        
        self.df = df   
        return self.df
        
    def run_backtest(self, initial_wallet=1000, return_type="metrics"):
        dt = self.df[:]
        wallet = initial_wallet
        taker_fee = 0.0007
        trades = []
        days = []
        current_day = 0
        previous_day = 0
        current_position = None

        # print("tp",take_profit_pct,"sl",stop_loss_pct)
        
        for index, row in dt.iterrows():
            
            # -- Add daily report --
            current_day = index.day
            if previous_day != current_day:
                temp_wallet = wallet
                if current_position:
                    if current_position['side'] == "LONG":
                        close_price = row['close']
                        trade_result = (close_price - current_position['price']) / current_position['price']
                        temp_wallet += temp_wallet * trade_result
                        fee = temp_wallet * taker_fee
                        temp_wallet -= fee
                    elif current_position['side'] == "SHORT":
                        close_price = row['close']
                        trade_result = (current_position['price'] - close_price) / current_position['price']
                        temp_wallet += temp_wallet * trade_result
                        fee = temp_wallet * taker_fee
                        temp_wallet -= fee
                    
                days.append({
                    "day":str(index.year)+"-"+str(index.month)+"-"+str(index.day),
                    "wallet":temp_wallet,
                    "price":row['close']
                })
            previous_day = current_day
            if current_position:
            # -- Check for closing position --
                if current_position['side'] == "LONG":
                        
                    # -- Close LONG market --
                    if row['close_long_market']:
                        close_price = row['close']
                        trade_result = (close_price - current_position['price']) / current_position['price']
                        wallet += wallet * trade_result
                        fee = wallet * taker_fee
                        wallet -= fee
                        trades.append({
                            "open_date": current_position['date'],
                            "close_date": index,
                            "position": "LONG",
                            "open_reason": current_position['reason'],
                            "close_reason": "Market",
                            "open_price": current_position['price'],
                            "close_price": close_price,
                            "open_fee": current_position['fee'],
                            "close_fee": fee,
                            "open_trade_size":current_position['size'],
                            "close_trade_size": wallet,
                            "wallet": wallet
                        })
                        current_position = None
                        
                elif current_position['side'] == "SHORT":
                    # -- Close SHORT Market --
                    if row['close_short_market']:
                        close_price = row['close']
                        trade_result = (current_position['price'] - close_price) / current_position['price']
                        wallet += wallet * trade_result
                        fee = wallet * taker_fee
                        wallet -= fee
                        trades.append({
                            "open_date": current_position['date'],
                            "close_date": index,
                            "position": "SHORT",
                            "open_reason": current_position['reason'],
                            "close_reason": "Market",
                            "open_price": current_position['price'],
                            "close_price": close_price,
                            "open_fee": current_position['fee'],
                            "close_fee": fee,
                            "open_trade_size": current_position['size'],
                            "close_trade_size": wallet,
                            "wallet": wallet
                        })
                        current_position = None

            # -- Check for opening position --
            else:
                # Open long market
                if row['open_long_market']:
                    open_price = row['close']
                    fee = wallet * taker_fee
                    wallet -= fee
                    pos_size = wallet
                    current_position = {
                        "size": pos_size,
                        "date": index,
                        "price": open_price,
                        "fee":fee,
                        "reason": "Market",
                        "side": "LONG",
                    }
                elif row['open_short_market']:
                    open_price = row['close']
                    fee = wallet * taker_fee
                    wallet -= fee
                    pos_size = wallet
                    current_position = {
                        "size": pos_size,
                        "date": index,
                        "price": open_price,
                        "fee":fee,
                        "reason": "Market",
                        "side": "SHORT"
                    }
                    
                    
        dfDays = pd.DataFrame(days)
        dfDays['day'] = pd.to_datetime(dfDays['day'])
        dfDays = dfDays.set_index(dfDays['day'])

        dfTrades = pd.DataFrame(trades)
        dfTrades['open_date'] = pd.to_datetime(dfTrades['open_date'])
        dfTrades = dfTrades.set_index(dfTrades['open_date'])   
        
        if return_type == "metrics":
            return get_metrics(dfTrades, dfDays) | {
                "wallet": wallet,
                "trades": dfTrades,
                "days": dfDays
            }  
        else:
            return True

pair = "BTCd/USDT"
tf = "1d"

df = get_historical_from_db(
    ccxt.binance(), 
    pair,
    tf,
    path="../../database/"
)

training_set = df.copy().loc["2022-08-29":]
test_set = df.copy().loc["2022-08-29":]
learn_period = 4
epochs = int(argv[1])
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set["close"].values.reshape(-1, 1))

X_train = []
y_train = []

for i in range(learn_period, len(training_set)):
    X_train.append(training_set_scaled[i-learn_period:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_train[0][0:2]
regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, epochs, batch_size = 32)
real_stock_price = test_set.iloc[:, 1:2].values

dataset_total = pd.concat((training_set['close'], test_set['close']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(test_set) - learn_period:].values
inputs = inputs.reshape(-1,1)
inputs = sc.fit_transform(inputs)
X_test = []
for i in range(learn_period, len(test_set) + learn_period):
    X_test.append(inputs[i-learn_period:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
df["predicted"] = 0
df["predicted"].iloc[-len(predicted_stock_price.flatten()):] = predicted_stock_price.flatten()
df["diff_close"] = df["close"].diff()
df["diff_predicted"] = df["predicted"].diff()
df["next_predicted"] = df["predicted"].shift(-1)
df["next_close"] = df["close"].shift(-1)
df["diff_predicted_next"] = df["next_predicted"] - df["predicted"]
df["diff_close_next"] = df["next_close"] - df["close"]
df["mean_evol_3"] = df["close"].shift(-3).rolling(3).mean() - df["close"]
df["mean_evol_5"] = df["close"].shift(-5).rolling(5).mean() - df["close"]
df["mean_evol_10"] = df["close"].shift(-10).rolling(10).mean() - df["close"]
df["mean_evol_20"] = df["close"].shift(-20).rolling(20).mean() - df["close"]
print(len(df.loc[(df["diff_predicted_next"] > df["diff_close"]) & (df["diff_close_next"] > 0)]))
print(len(df.loc[(df["diff_predicted_next"] > df["diff_close"]) & (df["diff_close_next"] < 0)]))
print("-------------")
print(len(df.loc[(df["diff_predicted_next"] < df["diff_close"]) & (df["diff_close_next"] < 0)]))
print(len(df.loc[(df["diff_predicted_next"] < df["diff_close"]) & (df["diff_close_next"] > 0)]))
print("-------------")
print(df.loc[(df["diff_predicted_next"] > df["diff_close"])]["diff_close_next"].mean())
print(df.loc[(df["diff_predicted_next"] > df["diff_close"])]["mean_evol_3"].mean())
print(df.loc[(df["diff_predicted_next"] > df["diff_close"])]["mean_evol_5"].mean())
print(df.loc[(df["diff_predicted_next"] > df["diff_close"])]["mean_evol_10"].mean())
print(df.loc[(df["diff_predicted_next"] > df["diff_close"])]["mean_evol_20"].mean())
print("-------------")
print(df.loc[(df["diff_predicted_next"] < df["diff_close"])]["diff_close_next"].mean())
print(df.loc[(df["diff_predicted_next"] < df["diff_close"])]["mean_evol_3"].mean())
print(df.loc[(df["diff_predicted_next"] < df["diff_close"])]["mean_evol_5"].mean())
print(df.loc[(df["diff_predicted_next"] < df["diff_close"])]["mean_evol_10"].mean())
print(df.loc[(df["diff_predicted_next"] < df["diff_close"])]["mean_evol_20"].mean())
print(len(df.loc[(df["diff_predicted"] > 0) & (df["diff_close"] > 0)]))
print(len(df.loc[(df["diff_predicted"] > 0) & (df["diff_close"] < 0)]))
print("-------------")
print(len(df.loc[(df["diff_predicted"] < 0) & (df["diff_close"] < 0)]))
print(len(df.loc[(df["diff_predicted"] < 0) & (df["diff_close"] > 0)]))
print("-------------")
print(df.loc[(df["diff_predicted"] > 0)]["diff_close_next"].mean())
print(df.loc[(df["diff_predicted"] > 0)]["mean_evol_3"].mean())
print(df.loc[(df["diff_predicted"] > 0)]["mean_evol_5"].mean())
print(df.loc[(df["diff_predicted"] > 0)]["mean_evol_10"].mean())
print(df.loc[(df["diff_predicted"] > 0)]["mean_evol_20"].mean())
print("-------------")
print(df.loc[(df["diff_predicted"] < 0)]["diff_close_next"].mean())
print(df.loc[(df["diff_predicted"] < 0)]["mean_evol_3"].mean())
print(df.loc[(df["diff_predicted"] < 0)]["mean_evol_5"].mean())
print(df.loc[(df["diff_predicted"] < 0)]["mean_evol_10"].mean())
print(df.loc[(df["diff_predicted"] < 0)]["mean_evol_20"].mean())
dt = df.copy().loc["2022":]

strat = lstm_strat(
    df = dt,
    use_short=True,
    strat=0,
)

strat.populateIndicators()
strat.populateBuySell(show_log=True)
btResult = strat.run_backtest(initial_wallet=1000, return_type="metrics")
dfTrades, dfDays = basic_single_asset_backtest(trades=btResult['trades'], days = btResult['days'])
if btResult['wallet']>2900:
    regressor.save('/saved2/')