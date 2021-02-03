import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

class StockPrediction:
    def StockData(self, code, date):
        yf.pdr_override()
        df = pdr.get_data_yahoo(f'{code}.KS', start=f'{date}')
        df = df[['Open','High','Low','Volume','Close']]
        return df
    
    def MinMaxScaler(self, data):
        numerator = data - np.min(data, 0)
        denominator = np.max(data, 0) - np.min(data, 0)
        return numerator / (denominator + 1e-7)
    
    def DataSet(self, code, date):
        df = self.StockData(code, date)
        df_x = self.MinMaxScaler(df)
        df_y = df_x[['Close']]
        
        x = df_x.values.tolist()
        y = df_y.values.tolist()
        
        data_x = []
        data_y = []
        window_size = 10
        for i in range(len(y) - window_size):
            x2 = x[i : i + window_size]
            y2 = y[i + window_size]
            data_x.append(x2)
            data_y.append(y2)
            
        # 훈련용 데이터셋 70%
        train_size = int(len(data_y) * 0.7)
        train_x = np.array(data_x[0:train_size])
        train_y = np.array(data_y[0:train_size])
        
        # 테스트용 데이터셋 30%
        test_size = len(data_y) - train_size
        test_x = np.array(data_x[train_size:len(data_x)])
        test_y = np.array(data_y[train_size:len(data_y)])
        return df, df_y, train_x, train_y, test_x, test_y
    
    def LSTMModel(self, code, date):
        df, df_y, train_x, train_y, test_x, test_y = self.DataSet(code, date)
        
        model = Sequential()
        model.add(LSTM(units=10, activation='relu', return_sequences=True, input_shape=(10, 5)))  
        model.add(Dropout(0.1))     
        model.add(LSTM(units=10, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(units=1))
        # model.summary()

        model.compile(optimizer='adam', loss='mean_squared_error')  
        model.fit(train_x, train_y, epochs=60, batch_size=30)         
        pred_y = model.predict(test_x)
        return df, df_y, test_y, pred_y
    
    def PredictionResult(self, code, date):
        df, df_y, test_y, pred_y = self.LSTMModel(code, date)
        
        plt.figure()
        plt.plot(test_y, color='r', label='real stock price')
        plt.plot(pred_y, color='b', label='predicted stock price')
        plt.title('Real Stock Price vs Predicted Stock Price')
        plt.xlabel('time')
        plt.ylabel('stock price')
        plt.legend(loc='best')
        plt.show()
        
        print("다음 날 예측 종가 : ", df.Close[-1]*pred_y[-1]/df_y.Close[-1])
        
# if __name__ == '__main__':
#     stp = StockPrediction()
#     stp.PredictionResult()




