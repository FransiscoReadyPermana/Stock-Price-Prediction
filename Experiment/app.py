from flask import Flask, render_template, request, redirect, url_for, flash, Blueprint
import pandas as pd
import yfinance as yf
# from pandas_datareader import data as pdr
from io import BytesIO
import base64
import matplotlib
matplotlib.use('agg', force=False)
from matplotlib import pyplot as plt


class saham():
    def __init__(self, symbol):
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        self.train_test_split = train_test_split
        self.model = LinearRegression()
        self.symbol = symbol
        self.ticker = yf.Ticker(symbol)
        self.info = self.ticker.info
        self.realData = self.ticker.history(period='max')
        data = pd.DataFrame()
        self.data = data.assign(Open=self.realData['Open'], Close=self.realData['Close'], High=self.realData['High'], Low=self.realData['Low'])
        self.data = self.data.dropna()
        x = self.data.drop(['Low'], axis=1)
        y = self.data['Low']
        self.X_train, self.X_test, self.y_train, self.y_test = self.train_test_split(x, y, test_size=0.2)
    def trainingData(self, model = None):
        from sklearn.metrics import mean_squared_error
        if model == None:
            model = self.model
        self.model = model
        self.model.fit(self.X_train, self.y_train)
        # self.trainPred = self.model.predict(self.X_train)
        # self.testPred = self.model.predict(self.X_test)
        self.scoreTest = self.model.score(self.X_test, self.y_test)
        self.scoreTrain = self.model.score(self.X_train, self.y_train)
        # self.mseTrain = mean_squared_error(self.scoreTrain, self.y_train)
        # self.mseTest = mean_squared_error(self.scoreTest, self.y_test)
        return ("R2 Train\t: " , self.scoreTrain,"R2 Test\t: ", self.scoreTest)
    def predict(self, open, close, high):
        self.trainingData(model = self.model)
        data = pd.DataFrame()
        data = data.assign(Open=[open], Close=[close], High=[high])
        if isinstance(open, list):
            data = pd.DataFrame({'Open': open, 'Close': close, 'High': high})
        else:
            data = pd.DataFrame({'Open': [open], 'Close': [close], 'High': [high]})
        prediction = self.model.predict(data)
        return prediction
    def chart(self, column = ['Open', 'Close', 'High', 'Low']):
        plt.clf()
        return self.data[column].plot(figsize=(20,5)).set_title(self.symbol)

# shm = saham('^JKSE')
app = Flask(__name__)

global stock
@app.route('/<string:stock>')
def index(stock):
    global shm
    shm = saham(stock)
    return (str(shm.trainingData()))

@app.route('/<string:stock>/chart', methods=['POST'])
def chart(stock):
    shm = saham(stock)
    # Generate the plot
    if request.method == 'POST':
        column = request.form['column']
        buffer = BytesIO()
        shm.chart(column=column).get_figure().savefig(buffer, format='png')
        chart_data = base64.b64encode(buffer.getvalue()).decode()
        # Return the HTML code to embed the chart
        return f'<img src="data:image/png;base64,{chart_data}">'

@app.route('/<string:stock>/defaultChart')
def defaultChart(stock):
    index(stock=stock)
    # Generate the plot
    buffer = BytesIO()
    shm.chart(column=['Low', 'High', 'Close', 'Open']).get_figure().savefig(buffer, format='png')
    chart_data = base64.b64encode(buffer.getvalue()).decode()
    # Return the HTML code to embed the chart
    return f'<img src="data:image/png;base64,{chart_data}">'

@app.route('/<string:stock>/predict', methods=['POST'])
def predict(stock):
    index(stock=stock)
    if request.method == 'POST':
        open = request.form['open']
        close = request.form['close']
        high = request.form['high']
        prediction = shm.predict(open, close, high)
        return str(prediction)

if __name__ == '__main__':
    app.run(debug=True)