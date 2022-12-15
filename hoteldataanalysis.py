import pandas as pd
import matplotlib.pyplot as plt
import statistics as stat
import numpy as np
from sklearn.metrics import r2_score
from statistics import mean
from tensorflow import keras


hb = pd.read_csv("hotel_bookings.csv")

hb1 = hb[hb['hotel'] == 'Resort Hotel']
hb1.sort_values(['lead_time'], inplace=True)
hb2 = hb[hb['hotel'] == 'City Hotel']
hb2.sort_values(['lead_time'], inplace=True)

ltr = hb1["lead_time"].values
ltc = hb2["lead_time"].values

adrr = hb1["adr"].values
adrc = hb2["adr"].values

cltr = []
cltc = []

cadrr = []
cadrc = []

templead = ltr[0]
tempadr = [adrr[0]]
cltr.append(ltr[0])

for i in range(ltr.size - 1):
    if i == ltr.size - 2:
        cadrr.append(mean(tempadr))
    elif ltr[i+1] == templead:
        tempadr.append(adrr[i+1])
    else:
        avgadr = mean(tempadr)
        cadrr.append(avgadr)
        templead = ltr[i + 1]
        tempadr = [adrr[i+1]]
        cltr.append(templead)

templead = ltc[0]
tempadr = [adrc[0]]
cltc.append(ltc[0])

for i in range(ltc.size - 1):
    if i == ltc.size - 2:
        cadrc.append(mean(tempadr))
    elif ltc[i+1] == templead:
        tempadr.append(adrc[i+1])
    else:
        avgadr = mean(tempadr)
        cadrc.append(avgadr)
        templead = ltc[i + 1]
        tempadr = [adrc[i+1]]
        cltc.append(templead)

polyreg1 = np.poly1d(np.polyfit(cltr, cadrr, 5))
polyreg2 = np.poly1d(np.polyfit(cltc, cadrc, 5))
line = np.linspace(1, 700, 100)

fig, ax = plt.subplots()
ax = plt.scatter(cltr, cadrr, s=5, label='Resort Hotels')
ax = plt.scatter(cltc, cadrc, color='r', s=5, label='City Hotels')
ax = plt.plot(line, polyreg1(line), c='b')
ax = plt.plot(line, polyreg2(line), c='r')
ax = plt.ylim([0, 250])
ax = plt.title("Lead time vs ADR with Regression Line")
ax = plt.xlabel("Lead Time")
ax = plt.ylabel("Average Daily Rate (ADR)")
ax = plt.legend()
plt.show()

print('Resort regression equation: \n', polyreg1)
print('City regression equation: \n', polyreg2)
print('r^2 Score: ', r2_score(cltr, polyreg1(cadrr)), r2_score(cltc, polyreg2(cadrc)))

lead1 = hb1["lead_time"]
adr1 = hb1["adr"]
polyreg1 = np.poly1d(np.polyfit(adr1, lead1, 20))

lead2 = hb2["lead_time"]
adr2 = hb2["adr"]
polyreg2 = np.poly1d(np.polyfit(adr2, lead2, 20))

line = np.linspace(1, 500, 100000)

fig, ax = plt.subplots()
ax = plt.scatter(cltr, cadrr, s=5)
ax = plt.scatter(cltc, cadrc, color='r', s=5)
ax = plt.plot(line, polyreg1(line), c='black')
ax = plt.plot(line, polyreg2(line))
ax = plt.ylim([0, 200])
plt.show()

for i in range(10):
    polyreg1 = np.poly1d(np.polyfit(cltr, cadrr, i+1))
    polyreg2 = np.poly1d(np.polyfit(cltc, cadrc, i+1))
    print(i+1, "degree polynomial R2 score:", r2_score(cltr, polyreg1(cadrr)), r2_score(cltc, polyreg2(cadrc)))

# Neural Network

# Formatting Data
cancelled = ['is_canceled']
variables = ['lead_time', 'arrival_date_year', 'arrival_date_week_number', 'arrival_date_day_of_month',
             'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies',
             'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled',
             'booking_changes', 'agent', 'company', 'days_in_waiting_list', 'adr', 'required_car_parking_spaces',
             'total_of_special_requests']

x = hb[variables].values
y = hb[cancelled].values

from sklearn.preprocessing import StandardScaler
PredictorScaler = StandardScaler()

PredictorScalerFit = PredictorScaler.fit(x)

x = PredictorScalerFit.transform(x)

from sklearn.model_selection import KFold

cv = KFold(n_splits=5)

model = keras.Sequential()

neurons = [5, 10, 15, 20, 30]

num = -1

# K-Fold CV enumeration for optimal neurons and layers
for train, test in cv.split(x):
    num += 1
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for i in train:
        x_train.append(x[i])
        y_train.append(y[i])

    for i in test:
        x_test.append(x[i])
        y_test.append(y[i])

    model.add(keras.layers.Dense(20, input_dim=19, activation='relu'))
    model.add(keras.layers.Dense(neurons[num], activation='relu'))
    model.add(keras.layers.Dense(2, activation='softmax'))

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=20, epochs=4)
    print(model.evaluate(x_test, y_test))


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)


def FunctionFindBestParams(x_train, y_train):
    # Defining the list of hyper parameters to try
    TrialNumber = 0
    batch_size_list = [5, 10, 15, 20]
    epoch_list = [2, 5, 10, 15]

    SearchResultsData = pd.DataFrame(columns=['TrialNumber', 'Parameters', 'Accuracy'])

    for batch_size_trial in batch_size_list:
        for epochs_trial in epoch_list:
            TrialNumber += 1

            # Creating the classifier model
            classifier = tf.keras.Sequential()
            classifier.add(tf.keras.layers.Dense(units=20, input_dim=19, kernel_initializer='uniform', activation='relu'))
            classifier.add(tf.keras.layers.Dense(units=20, kernel_initializer='uniform', activation='relu'))
            classifier.add(tf.keras.layers.Dense(units=10, kernel_initializer='uniform', activation='relu'))
            classifier.add(tf.keras.layers.Dense(units=2, kernel_initializer='uniform', activation='softmax'))
            classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            classifier.fit(x_train, y_train, batch_size=batch_size_trial, epochs=epochs_trial,
                                               verbose=0)
            # Fetching the accuracy of the training
            Accuracy = classifier.evaluate(x_test, y_test)

            # printing the results of the current iteration
            print(TrialNumber, 'Parameters:', 'batch_size:', batch_size_trial, '-', 'epochs:', epochs_trial,
                  'Accuracy:', Accuracy)

            SearchResultsData = SearchResultsData.append(pd.DataFrame(data=[[TrialNumber,
                                                                             'batch_size' + str(
                                                                                 batch_size_trial) + '-' + 'epoch' + str(
                                                                                 epochs_trial), Accuracy]],
                                                                      columns=['TrialNumber', 'Parameters',
                                                                               'Accuracy']))
    return (SearchResultsData)


ResultsData = FunctionFindBestParams(x_train, y_train)

#Rebuild optimal Classifier
model.add(keras.layers.Dense(units=20, input_dim=19, activation='relu'))
model.add(keras.layers.Dense(units=30, activation='relu'))
model.add(keras.layers.Dense(units=15, activation='relu'))
model.add(keras.layers.Dense(units=2, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=20, epochs=4)
print(model.evaluate(x_test, y_test))






