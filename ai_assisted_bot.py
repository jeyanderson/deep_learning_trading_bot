import ccxt
import pandas as pd
import time
import sys
sys.path.append('../..')
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import keras

ftx=ccxt.ftx({
            "apiKey":'apiKey',
            "secret":'secretKey',
            'headers':{
                'FTX-SUBACCOUNT':'subaccountName'
            }
        })
symbol='ETH-PERP'
tf='1d'

def get_historical_since(symbol,timeframe,startDate):
        try:
            tempData=ftx.fetch_ohlcv(symbol,timeframe,int(
                time.time()*1000)-1209600000,limit=1000)
            dtemp=pd.DataFrame(tempData)
            timeInter=int(dtemp.iloc[-1][0]-dtemp.iloc[-2][0])
        except:
            return None

        finished=False
        start=False
        allDf=[]
        startDate=ftx.parse8601(startDate)
        while(start==False):
            try:
                tempData=ftx.fetch_ohlcv(
                    symbol,timeframe,startDate,limit=1000)
                dtemp=pd.DataFrame(tempData)
                timeInter=int(dtemp.iloc[-1][0]-dtemp.iloc[-2][0])
                nextTime=int(dtemp.iloc[-1][0]+timeInter)
                allDf.append(dtemp)
                start=True
            except:
                startDate=startDate+1209600000*2

        if dtemp.shape[0]<1:
            finished=True
        while(finished==False):
            try:
                tempData=ftx.fetch_ohlcv(
                    symbol,timeframe,nextTime,limit=1000)
                dtemp=pd.DataFrame(tempData)
                nextTime=int(dtemp.iloc[-1][0]+timeInter)
                allDf.append(dtemp)
                # print(dtemp.iloc[-1][0])
                if dtemp.shape[0]<1:
                    finished=True
            except:
                finished=True
        result=pd.concat(allDf,ignore_index=True,sort=False)
        result=result.rename(
            columns={0:'timestamp',1:'open',2:'high',3:'low',4:'close',5:'volume'})
        result=result.set_index(result['timestamp'])
        result.index=pd.to_datetime(result.index,unit='ms')
        del result['timestamp']
        return result

df=get_historical_since(symbol,tf,'2022-01-01 00:00')
learn_period=4
training_set=df.copy().loc[:"2022"]
test_set=df.copy().loc["2022":]
sc=MinMaxScaler(feature_range=(0,1))
#import your own saved model
regressor=keras.models.load_model('./saved1')
real_stock_price=test_set.iloc[:,1:2].values

dataset_total=pd.concat((training_set['close'], test_set['close']), axis=0)
inputs=dataset_total[len(dataset_total) - len(test_set) - learn_period:].values
inputs=inputs.reshape(-1,1)
inputs=sc.fit_transform(inputs)
X_test=[]
for i in range(learn_period, len(test_set)+learn_period):
    X_test.append(inputs[i-learn_period:i,0])
    X_test2=np.array(X_test)
del(X_test)
X_test=np.reshape(X_test2,(X_test2.shape[0],X_test2.shape[1],1))
del(X_test2)
predicted_stock_price=regressor.predict(X_test)
predicted_stock_price=sc.inverse_transform(predicted_stock_price)
df["predicted"]=0
df["predicted"].iloc[-len(predicted_stock_price.flatten()):]=predicted_stock_price.flatten()
df["diff_close"]=df["close"].diff()
df["diff_predicted"]=df["predicted"].diff()
df["next_predicted"]=df["predicted"].shift(-1)
df["next_close"]=df["close"].shift(-1)
df["diff_predicted_next"]=df["next_predicted"]-df["predicted"]
df["diff_close_next"]=df["next_close"] - df["close"]
df["mean_evol_3"]=df["close"].shift(-3).rolling(3).mean()-df["close"]
df["mean_evol_5"]=df["close"].shift(-5).rolling(5).mean()-df["close"]
df["mean_evol_10"]=df["close"].shift(-10).rolling(10).mean()-df["close"]
df["mean_evol_20"]=df["close"].shift(-20).rolling(20).mean()-df["close"]
row=df.iloc[-2]
old_row=df.iloc[-3]
allBalance=ftx.fetchBalance()
usdBalance=allBalance['USD']
buy_limit_price=row['close']
ethBalance=allBalance['ETH']
positions=ftx.fetchPositions([symbol])
netSize=positions[0]['info']['netSize']
if row.diff_predicted_next>0 and old_row.diff_predicted_next<0:
    if float(netSize):
       ftx.create_market_buy_order(symbol,abs(float(netSize)))
    allBalance=ftx.fetchBalance()
    usdBalance=allBalance['USD']['free']
    buy_quantity=float(ftx.amount_to_precision(symbol, usdBalance/buy_limit_price))*2
    exchange_buy_quantity=buy_quantity*buy_limit_price
    print(f"Place Buy Limit Order: {buy_quantity} {'ETH-PERP'} at the price of {buy_limit_price}$ ~{round(exchange_buy_quantity, 2)}$")
    ftx.create_limit_buy_order(symbol, buy_quantity, buy_limit_price)
elif row.diff_predicted_next<0 and old_row.diff_predicted_next>0:
    if float(netSize):
        ftx.create_market_sell_order(symbol,float(netSize))
    allBalance=ftx.fetchBalance()
    usdBalance=allBalance['USD']['free']
    sell_quantity=float(ftx.amount_to_precision(symbol, usdBalance/buy_limit_price))*2
    exchange_buy_quantity=sell_quantity*buy_limit_price
    print(f"Place Sell Limit Order: {sell_quantity} {'ETH-PERP'} at the price of {buy_limit_price}$ ~{round(exchange_buy_quantity, 2)}$")
    ftx.create_limit_sell_order(symbol,sell_quantity,buy_limit_price)