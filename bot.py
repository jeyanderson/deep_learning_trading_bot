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

def getHistoricalSince(symbol,timeframe,startDate):
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

df=getHistoricalSince(symbol,tf,'2022-01-01 00:00')
learnPeriod=4
trainingSet=df.copy().loc[:"2022"]
testSet=df.copy().loc["2022":]
sc=MinMaxScaler(feature_range=(0,1))
#import your own saved model
regressor=keras.models.load_model('./saved1')
realStockPrice=testSet.iloc[:,1:2].values

datasetTotal=pd.concat((trainingSet['close'], testSet['close']), axis=0)
inputs=datasetTotal[len(datasetTotal) - len(testSet) - learnPeriod:].values
inputs=inputs.reshape(-1,1)
inputs=sc.fit_transform(inputs)
xTest=[]
for i in range(learnPeriod, len(testSet)+learnPeriod):
    xTest.append(inputs[i-learnPeriod:i,0])
    xTest2=np.array(xTest)
del(xTest)
xTest=np.reshape(xTest2,(xTest2.shape[0],xTest2.shape[1],1))
del(xTest2)
predictedStockPrice=regressor.predict(xTest)
predictedStockPrice=sc.inverse_transform(predictedStockPrice)
df["predicted"]=0
df["predicted"].iloc[-len(predictedStockPrice.flatten()):]=predictedStockPrice.flatten()
df["diffClose"]=df["close"].diff()
df["diffPredicted"]=df["predicted"].diff()
df["nextPredicted"]=df["predicted"].shift(-1)
df["nextClose"]=df["close"].shift(-1)
df["diffPredictedNext"]=df["nextPredicted"]-df["predicted"]
df["diffCloseNext"]=df["nextClose"] - df["close"]
df["meanEvol3"]=df["close"].shift(-3).rolling(3).mean()-df["close"]
df["meanEvol5"]=df["close"].shift(-5).rolling(5).mean()-df["close"]
df["meanEvol10"]=df["close"].shift(-10).rolling(10).mean()-df["close"]
df["meanEvol20"]=df["close"].shift(-20).rolling(20).mean()-df["close"]
row=df.iloc[-2]
oldRow=df.iloc[-3]
allBalance=ftx.fetchBalance()
usdBalance=allBalance['USD']
buyLimitPrice=row['close']
ethBalance=allBalance['ETH']
positions=ftx.fetchPositions([symbol])
netSize=positions[0]['info']['netSize']
if row.diffPredictedNext>0 and oldRow.diffPredictedNext<0:
    if float(netSize):
       ftx.create_market_buy_order(symbol,abs(float(netSize)))
    allBalance=ftx.fetchBalance()
    usdBalance=allBalance['USD']['free']
    buyQuantity=float(ftx.amount_to_precision(symbol, usdBalance/buyLimitPrice))*2
    exchangeBuyQuantity=buyQuantity*buyLimitPrice
    print(f"Place Buy Limit Order: {buyQuantity} {'ETH-PERP'} at the price of {buyLimitPrice}$ ~{round(exchangeBuyQuantity, 2)}$")
    ftx.create_limit_buy_order(symbol, buyQuantity, buyLimitPrice)
elif row.diffPredictedNext<0 and oldRow.diffPredictedNext>0:
    if float(netSize):
        ftx.create_market_sell_order(symbol,float(netSize))
    allBalance=ftx.fetchBalance()
    usdBalance=allBalance['USD']['free']
    sellQuantity=float(ftx.amount_to_precision(symbol, usdBalance/buyLimitPrice))*2
    exchangeBuyQuantity=sellQuantity*buyLimitPrice
    print(f"Place Sell Limit Order: {sellQuantity} {'ETH-PERP'} at the price of {buyLimitPrice}$ ~{round(exchangeBuyQuantity, 2)}$")
    ftx.create_limit_sell_order(symbol,sellQuantity,buyLimitPrice)