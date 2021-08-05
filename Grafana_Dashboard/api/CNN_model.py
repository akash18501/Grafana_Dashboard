from prometheus_api_client import PrometheusConnect
from influxdb import InfluxDBClient
import datetime
import numpy as np
import threading
from threading import Thread
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    # prom = PrometheusConnect(url="http://demo.robustperception.io:9090/", disable_ssl=True)
    prom = PrometheusConnect(url="http://localhost:9090/", disable_ssl=True)

except:
    print("prometheus host not connected")
    exit(0)

try:
    client = InfluxDBClient(host='localhost', port=8086)
    client.switch_database('metric_predictions_db_2')
except:
    print("InfluxDb not connected")
    exit(0)

class metric_prediction():

    #method to add the predicted data to influx db
    def push_to_influx(self, data_list, measurement_name):
        # query = "drop measurement" + measurement_name
        # client.query(query)
        for val in data_list:
            json_body = [{
                "measurement": measurement_name,
                "tags": {
                    "instance": "temp_demo",
                    "job": self.job,
                    "load": "original"
                },
                "fields": {
                    "metric": float(val[1])

                },
                "time": val[0]
            }]
            client.write_points(json_body)

    #method to add predicted from Linear Regression data to influx db
    def push_to_influx_LR(self, data_list, measurement_name, load):
        temp_end_time = self.end_time

        for val in data_list:
            json_body = [{
                "measurement": measurement_name,
                "tags": {
                    "instance": "temp_demo",
                    "job": self.job,
                    "load": load
                },
                "fields": {
                    "metric": val

                },
                "time": temp_end_time + datetime.timedelta(hours=0, minutes=1)
            }]
            client.write_points(json_body)
            temp_end_time = temp_end_time + datetime.timedelta(hours=0, minutes=1)

    #method to add accuracy data to influx db
    def push_accuracy_in_influx(self, accuracy_val, measurement_name):
        # query = "drop measurement " + measurement_name
        # client.query(query)
        json_body = [{
            "measurement": measurement_name,
            "tags": {
                "instance": "temp_demo",
                "job": self.job,
            },
            "fields": {
                "accuracy": 100 - accuracy_val*100

            },

        }]
        client.write_points(json_body)

    #Creating data frame from the raw metric data
    def create_dataframe(self, metric):
        data = []
        for y in metric[0]["values"]:
            json_body = [{
                "measurement": "NodeLoad",
                "tags": {
                    "instance": metric[0]["metric"]["instance"],
                    "job": metric[0]["metric"]["job"],
                },
                "fields": {
                    "metric": float(y[1])

                },
                "time": datetime.datetime.utcfromtimestamp(int(y[0]))
            }]
            # client.write_points(json_body)
            data.append([datetime.datetime.utcfromtimestamp(int(y[0])), float(y[1])])

        # global end_time
        self.end_time = data[-1][0]
        # +datetime.timedelta(hours=5,minutes=30)
        df = pd.DataFrame(data, columns=["time", "metric"])
        # df["time"] = [x + datetime.timedelta(hours=5,minutes=30) for x in df["time"]]
        df.set_index("time", inplace=True)

        return df, data

    #Fetching metric data from prometheus using prometheus api client object
    def get_metric(self, query):
        metric = prom.custom_query(query=query)
        return metric

    #Splitting the training and the testing data
    def training_testing_split(self, df):
        # print("df shape", df.shape)
        training_data_len = int(df.shape[0] * 0.8)  # 80% for training 20% for testing
        # print("training data length: ", training_data_len)
        train = df.iloc[:training_data_len]
        test = df.iloc[training_data_len:]

        return train, test

    #Converting the training and the testing data to a array so
    #that it can be converted into batches later
    def frame_to_list(self, train_frame, test_frame):
        train_list = train_frame['metric'].tolist()
        test_list = test_frame['metric'].tolist()

        return train_list, test_list

    #method to convert data into samples/batches
    def sequence_generator(self, sequence, no_input, no_feature):
        X, y = list(), list()
        for i in range(len(sequence)):
            # find the end index for this batch
            end_ix = i + no_input
            if end_ix > len(sequence) - 1:
                break;
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            X.append(seq_x)
            y.append(seq_y)

        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], no_feature))
        return X, y

    #Defining the CNN model
    def define_model(self, no_input, no_feature):
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import Flatten
        from keras.layers.convolutional import Conv1D
        from keras.layers.convolutional import MaxPooling1D

        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(no_input, no_feature)))
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        return model
    #Training the CNN model
    def train_model(self, model, X, y, no_of_epochs):
        model.fit(X, y, epochs=no_of_epochs, verbose=1)
        loss_per_epoch = model.history.history['loss']
        plt.plot(range(len(loss_per_epoch)), loss_per_epoch)
        # plt.show()

    #Testing the accuracy on the testing data
    def testing_on_test_data(self, model, train_list, test_list, no_input, no_feature):
        last_processed_batch = np.array(train_list[-no_input:])
        type(last_processed_batch)
        last_processed_batch = last_processed_batch.reshape(1, no_input, no_feature)

        predictions = []

        for i in range(len(test_list)):
            pred_value = model.predict(last_processed_batch)[0]

            predictions.append(pred_value[0])

            last_processed_batch = np.delete(last_processed_batch, (0), axis=1)
            last_processed_batch = np.append(last_processed_batch, [[pred_value]], axis=1)

        return predictions

    #Predicting the future data using the trained model
    def future_prediction(self, model, train_list, test_list, no_input, no_feature):
        last_processed_batch = np.array(test_list[-no_input:])
        type(last_processed_batch)
        last_processed_batch = last_processed_batch.reshape(1, no_input, no_feature)

        predictions = []
        temp_end_time = self.end_time
        for i in range(no_input):
            pred_value = model.predict(last_processed_batch)[0]

            predictions.append([temp_end_time + datetime.timedelta(hours=0, minutes=1), pred_value[0]])
            temp_end_time = temp_end_time + datetime.timedelta(hours=0, minutes=1)

            last_processed_batch = np.delete(last_processed_batch, (0), axis=1)
            last_processed_batch = np.append(last_processed_batch, [[pred_value]], axis=1)

        prediction_frame = pd.DataFrame(predictions, columns=["time", "metric"])
        prediction_frame.set_index("time", inplace=True)

        return prediction_frame, predictions

    # mean absolute error
    def mae(self, actual, pred):
        sum = 0.0
        for i in range(len(actual)):
            sum += abs(actual[i] - pred[i])
        print(sum / len(actual))
        return sum / len(actual)

    # Mean of Absolute Percentage Errors
    def mape(self, actual, pred):
        cnt = 0
        for i in range(len(actual)):
            sum = (abs(actual[i] - pred[i]) / actual[i]) * 100
            if (sum <= 8):
                cnt += 1

        return (cnt / len(actual)) * 100

    #Calculating the accuracy and visualizing the data
    def show_test_prediction(self, test_frame, prediction_on_test_data):
        test_frame['predictions'] = prediction_on_test_data
        # test_frame.plot(figsize=(12, 6))
        # plt.legend("prediction on test data")
        # plt.show()
        # test_frame=test_frame.drop['predictions']
        x = self.mae(test_frame.metric, prediction_on_test_data)
        print("Mean absolute error ", self.mae(test_frame.metric, prediction_on_test_data))

        return x

    #Visualizing the future prediction
    def show(self, frame, test_frame):
        # plt.plot(test_frame)
        # plt.plot(frame)
        # plt.legend('prediction of future')
        # plt.show()
        pass

    #Init method
    def __init__(self, query, request_query, m_name):
        threading.Thread.__init__(self)
        self.query = query
        self.m_name = m_name
        self.end_time = None
        self.job = None
        self.request_query = request_query

    #Using Linear Regression to find the Correlation between the http req
    #and the respective metric data and calculating for double and half req
    def LinearRegression(self, no_input):
        metric = prom.custom_query(query=self.query)
        request_metric = prom.custom_query(query=self.request_query)

        d = []
        requests = []
        for y in request_metric[0]['values']:
            requests.append(float(y[1]))

        for y in metric[0]["values"]:
            d.append([datetime.datetime.utcfromtimestamp(int(y[0])), float(y[1])])

        request_per_unit = []
        for i in range(len(requests)):
            if i == 0:
                request_per_unit.append(10)
                continue
            request_per_unit.append(requests[i] - requests[i - 1])
        request_per_unit[0] = request_per_unit[1]

        if(len(request_per_unit) < len(d)):
            while (len(request_per_unit) < len(d)):
                request_per_unit.append(request_per_unit[len(request_per_unit) - 1])
        elif len(d) < len(request_per_unit):
            request_per_unit = request_per_unit[:len(d)]

        # global end_time
        #       self.end_time = d[-1][0]
        # +datetime.timedelta(hours=5,minutes=30)
        dataframe = pd.DataFrame(d, columns=["time", "metric"])
        print(len(request_per_unit))
        print(dataframe.shape[0])
        dataframe['requests'] = request_per_unit
        dataframe['double_requests'] = dataframe['requests'] * 2
        dataframe['half_requests'] = dataframe['requests'] / 2

        http_request_list = []
        for i in range(len(request_per_unit)):
            http_request_list.append([d[i][0], request_per_unit[i]])
        self.push_to_influx(http_request_list, self.m_name + "_requests")

        # df["time"] = [x + datetime.timedelta(hours=5,minutes=30) for x in df["time"]]
        dataframe.set_index("time", inplace=True)
        #         print(dataframe.head(20))

        X = dataframe[['requests']]
        y = dataframe[['metric']]
        double_x = dataframe[['double_requests']]
        half_x = dataframe[['half_requests']]

        train_size = int(0.8 * dataframe['metric'].shape[0])

        from sklearn.linear_model import LinearRegression
        lm = LinearRegression()
        lm.fit(X, y)

        double_predictions = lm.predict(double_x)
        half_predictions = lm.predict(half_x)
        double_predictions = double_predictions[-no_input:]
        half_predictions = half_predictions[-no_input:]
        print(len(double_predictions))
        self.push_to_influx_LR(double_predictions, self.m_name + "_predicted", "double")
        self.push_to_influx_LR(half_predictions, self.m_name + "_predicted", "half")

        plt.plot(double_predictions)
        plt.plot(half_predictions)

    def run(self):

        # fetching the metrics from prometheus in our python application
        metric = self.get_metric(self.query)
        self.job = metric[0]['metric']['job']
        # preparing the dataset from the fetched metrics so that it can be used in ML model
        df, dataset_list = self.create_dataframe(metric)

        # splitting the data in training and testing sets
        train_frame, test_frame = self.training_testing_split(df)

        train_list, test_list = self.frame_to_list(train_frame, test_frame)
        no_input = test_frame.shape[0] - 1
        no_feature = 1
        X_train, y_train = self.sequence_generator(train_list, no_input, no_feature)

        model = self.define_model(no_input, no_feature)

        self.train_model(model, X_train, y_train,
                         300)  # Training the model on training data set to calculate accuracy with testing data set
        prediction_on_test_data = self.testing_on_test_data(model, train_list, test_list, no_input, no_feature)
        accuracy = self.show_test_prediction(test_frame, prediction_on_test_data)

        future_prediction_frame, future_prediction_list = self.future_prediction(model, train_list, test_list, no_input,
                                                                                 no_feature)
        self.show(future_prediction_frame, test_frame)

        self.push_accuracy_in_influx(accuracy, self.m_name + "_accuracy")
        self.push_to_influx(dataset_list, self.m_name + "_original")
        self.push_to_influx(future_prediction_list, self.m_name + "_predicted")

        self.LinearRegression(no_input)


