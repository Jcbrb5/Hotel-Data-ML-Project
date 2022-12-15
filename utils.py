import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from scipy import linalg
import numpy as np
import pylab as pl
import matplotlib as mpl
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


def loadData():
    df = pd.read_csv ('hotel_bookings.csv')
    return df

def plotData(xData, yData, xLabel, yLabel, title):
    plt.scatter(xData, yData)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    plt.show()

def correlationHeatmap(df) :
    sns.heatmap(df.corr(),cmap='YlGnBu')
    plt.subplots_adjust(bottom=.3)
    plt.subplots_adjust(left=.3)
    plt.title('Correlation Between Features')
    plt.show()

def plot_ellipse(splot, mean, cov, color):
    v, w = linalg.eigh(cov)
    u = w[0] / linalg.norm(w[0])
    angle = np.arctan(u[1]/u[0])
    angle = 180 * angle / np.pi # convert to degrees
    # filled gaussian at 2 standard deviation
    ell = mpl.patches.Ellipse(mean, 2 * v[0] ** 0.5, 2 * v[1] ** 0.5,
                                            180 + angle, color=color)
    ell.set_clip_box(splot.bbox)
    ell.set_alpha(0.5)
    splot.add_artist(ell)

def LDA_isCancelled(df):
    is_cancelled = df['is_canceled']
    previous_cancellations = df['previous_cancellations']
    lead_time = df['lead_time']
    total_of_special_requests = df['total_of_special_requests']

    target_names = ['Not Cancelled', 'Cancelled']

    X = pd.concat([previous_cancellations, lead_time, total_of_special_requests], axis=1)
    y = is_cancelled

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    lda = LinearDiscriminantAnalysis(n_components=1, store_covariance=True)
    qda = QuadraticDiscriminantAnalysis(store_covariance=True)
    y_pred = lda.fit(X, y).predict(X)
    y_pred = qda.fit(X, y).predict(X)

    xx, yy = np.meshgrid(np.linspace(4, 8.5, 200), np.linspace(1.5, 4.5, 200))
    X_grid = np.c_[xx.ravel(), yy.ravel()]
    zz_lda = lda.predict_proba(X_grid)[:,1].reshape(xx.shape)
    zz_qda = qda.predict_proba(X_grid)[:,1].reshape(xx.shape)

    X = X.to_numpy()

    pl.figure()
    splot = pl.subplot(1, 2, 1)
    pl.contourf(xx, yy, zz_lda > 0.5, alpha=0.5)
    pl.scatter(X[y==1,0], X[y==1,1], c='r', label=target_names[1])
    pl.scatter(X[y==0,0], X[y==0,1], c='b', label=target_names[0])
    pl.contour(xx, yy, zz_lda, [0.5], linewidths=2., colors='k')
    plot_ellipse(splot, lda.means_[0], lda.covariance_, 'b')
    plot_ellipse(splot, lda.means_[1], lda.covariance_, 'r')
    pl.legend()
    pl.axis('tight')
    pl.title('Linear Discriminant Analysis')
    pl.show()

def plotCancellationData():
    is_cancelled = df.iloc[:, 1]
    previous_cancellations = df.iloc[:, 17]
    # total_of_special_requests = df['total_of_special_requests']
    lead_time = df.iloc[:, 2]
    target_names = ['Not Cancelled', 'Cancelled']
    X = pd.concat([previous_cancellations, lead_time], axis=1)
    X = X.to_numpy()
    y = is_cancelled
    pl.scatter(X[y==1,0], X[y==1,1], c='r', label=target_names[1])
    pl.scatter(X[y==0,0], X[y==0,1], c='b', label=target_names[0])
    pl.xlabel('Previous Cancellations')
    pl.ylabel('Lead Time (days)')
    pl.title('Lead Time vs. Previous Cancellations for Hotel Bookings')
    pl.legend()
    pl.show()

def knn_classifier(df):
    is_cancelled = df['is_canceled']
    previous_cancellations = df['previous_cancellations']
    lead_time = df['lead_time']
    total_of_special_requests = df['total_of_special_requests']
    required_car_parking_spaces = df['required_car_parking_spaces']
    booking_changes = df['booking_changes']
    days_in_waiting_list = df['days_in_waiting_list']
    adr = df['adr']



    X = pd.concat([previous_cancellations, lead_time, total_of_special_requests, required_car_parking_spaces, booking_changes, days_in_waiting_list, adr], axis=1)
    y = is_cancelled

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    range_k = range(1, 15)
    scores = {}
    scores_list = []
    for k in range_k:
        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        scores[k] = metrics.accuracy_score(y_test,y_pred)
        scores_list.append(metrics.accuracy_score(y_test,y_pred))
        result = metrics.confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(result)
        result1 = metrics.classification_report(y_test, y_pred)
        print("Classification Report:",)
        print (result1)
    plt.plot(range_k,scores_list)
    plt.xlabel("Value of K")
    plt.ylabel("Accuracy")
    plt.show()

    # classifier = KNeighborsClassifier(n_neighbors=12)
    # classifier.fit(X_train, y_train)



def separateCancellationsMarketSegments(df):
    nonCancellationDf = df[df['is_canceled'] == 0]
    cancellationDf = df[df['is_canceled'] == 1]
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    axs[0].hist(cancellationDf['market_segment'], label="cancelled")
    axs[1].hist(nonCancellationDf['market_segment'], label="non-cancelled")
    axs[0].legend()
    axs[1].legend()
    fig.suptitle('Market Segments for Cancelled and non-Cancelled Bookings')
    fig.supxlabel('Market Segment')
    fig.supylabel('Number of bookings')
    plt.show()
    #non-cancelled
    nc_onlineTa = nonCancellationDf[nonCancellationDf['market_segment'] == 'Online TA']
    print("nc_onlineTa: ", nc_onlineTa.shape[0] / nonCancellationDf.shape[0])
    nc_offline = nonCancellationDf[nonCancellationDf['market_segment'] == 'Offline TA/TO']
    print("nc_offline: ", nc_offline.shape[0] / nonCancellationDf.shape[0])
    nc_direct = nonCancellationDf[nonCancellationDf['market_segment'] == 'Direct']
    print("nc_direct: ", nc_direct.shape[0] / nonCancellationDf.shape[0])
    nc_corp = nonCancellationDf[nonCancellationDf['market_segment'] == 'Corporate']
    print("nc_corp: ", nc_corp.shape[0] / nonCancellationDf.shape[0])
    nc_group = nonCancellationDf[nonCancellationDf['market_segment'] == 'Groups']
    print("nc_group: ", nc_group.shape[0] / nonCancellationDf.shape[0])
    nc_complementary = nonCancellationDf[nonCancellationDf['market_segment'] == 'Complementary']
    print("nc_complementary: ", nc_complementary.shape[0] / nonCancellationDf.shape[0])
    nc_aviation = nonCancellationDf[nonCancellationDf['market_segment'] == 'Aviation']
    print("nc_aviation: ", nc_aviation.shape[0] / nonCancellationDf.shape[0])
    #cancelled
    c_onlineTa = cancellationDf[cancellationDf['market_segment'] == 'Online TA']
    print("c_onlineTa: ", c_onlineTa.shape[0] / cancellationDf.shape[0])
    c_offline = cancellationDf[cancellationDf['market_segment'] == 'Offline TA/TO']
    print("c_offline: ", c_offline.shape[0] / cancellationDf.shape[0])
    c_direct = cancellationDf[cancellationDf['market_segment'] == 'Direct']
    print("c_direct: ", c_direct.shape[0] / cancellationDf.shape[0])
    c_corp = cancellationDf[cancellationDf['market_segment'] == 'Corporate']
    print("c_corp: ", c_corp.shape[0] / cancellationDf.shape[0])
    c_group = cancellationDf[cancellationDf['market_segment'] == 'Groups']
    print("c_group: ", c_group.shape[0] / cancellationDf.shape[0])
    c_complementary = cancellationDf[cancellationDf['market_segment'] == 'Complementary']
    print("c_complementary: ", c_complementary.shape[0] / cancellationDf.shape[0])
    c_aviation = cancellationDf[cancellationDf['market_segment'] == 'Aviation']
    print("c_aviation: ", c_aviation.shape[0] / cancellationDf.shape[0])

def separateCancellationsDepositType(df):
    nonCancellationDf = df[df['is_canceled'] == 0]
    cancellationDf = df[df['is_canceled'] == 1]
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    axs[0].hist(cancellationDf['deposit_type'], label="cancelled")
    axs[1].hist(nonCancellationDf['deposit_type'], label="non-cancelled")
    axs[0].legend()
    axs[1].legend()
    fig.suptitle('Deposit type for Cancelled and non-Cancelled Bookings')
    fig.supxlabel('Deposit type')
    fig.supylabel('Number of bookings')
    plt.show()
     #non-cancelled
    nc_noDep = nonCancellationDf[nonCancellationDf['deposit_type'] == 'No Deposit']
    print("nc_noDep: ", nc_noDep.shape[0] / nonCancellationDf.shape[0])
    nc_nonRefund = nonCancellationDf[nonCancellationDf['deposit_type'] == 'Non Refund']
    print("nc_nonRefund: ", nc_nonRefund.shape[0] / nonCancellationDf.shape[0])
    nc_Refund = nonCancellationDf[nonCancellationDf['deposit_type'] == 'Refundable']
    print("nc_Refund: ", nc_Refund.shape[0] / nonCancellationDf.shape[0])
    #cancelled
    c_noDep = cancellationDf[cancellationDf['deposit_type'] == 'No Deposit']
    print("c_noDep: ", c_noDep.shape[0] / cancellationDf.shape[0])
    c_nonRefund = cancellationDf[cancellationDf['deposit_type'] == 'Non Refund']
    print("c_nonRefund: ", c_nonRefund.shape[0] / cancellationDf.shape[0])
    c_Refund = cancellationDf[cancellationDf['deposit_type'] == 'Refundable']
    print("c_Refund: ", c_Refund.shape[0] / cancellationDf.shape[0])
    #all
    nonRefund = df[df['deposit_type'] == 'Non Refund']
    print("Non-refund percentage: ", nonRefund.shape[0] / df.shape[0])

def separateCancellationsRepeatGuests(df):
    nonCancellationDf = df[df['is_canceled'] == 0]
    cancellationDf = df[df['is_canceled'] == 1]
    isRepeatedGuestNonCancelledDf = nonCancellationDf[nonCancellationDf['is_repeated_guest'] == 1]
    isRepeatedGuestCancelledDf = cancellationDf[cancellationDf['is_repeated_guest'] == 1]
    print('Percent of repeated guests for non cancelled: {}%', isRepeatedGuestNonCancelledDf.shape[0] / nonCancellationDf.shape[0])
    print('Percent of repeated guests for cancelled: {}%', isRepeatedGuestCancelledDf.shape[0] / cancellationDf.shape[0])

    repeated = df[df['is_repeated_guest'] == 1]
    not_repeated = df[df['is_repeated_guest'] == 0]
    repeatedWhoCancelled = repeated[repeated['is_canceled'] == 1]
    not_repeatedWhoCancelled = not_repeated[not_repeated['is_canceled'] == 1]
    print('Percent of repeated guests who cancelled: {}%', repeatedWhoCancelled.shape[0] / repeated.shape[0])
    print('Percent of not repeated guests who cancelled: {}%', not_repeatedWhoCancelled.shape[0] / not_repeated.shape[0])

def knn_classifierNew(df):
    df['market_segment'] = df['market_segment'].replace({'Online TA': 0, 'Offline TA/TO': 1, 'Direct': 2, 'Corporate': 3, 'Complementary': 4, 'Undefined': 5, 'Aviation': 6, 'Groups': 7})
    df['deposit_type'] = df['deposit_type'].replace({'No Deposit': 0, 'Non Refund': 1, 'Refundable': 2})
    df['hotel'] = df['hotel'].replace({'Resort Hotel': 0, 'City Hotel': 1})
    is_cancelled = df['is_canceled']
    market_segment = df['market_segment']
    deposit_type = df['deposit_type']
    hotel = df['hotel']
    is_repeated_guest = df['is_repeated_guest']
    previous_cancellations = df['previous_cancellations']
    lead_time = df['lead_time']
    total_of_special_requests = df['total_of_special_requests']
    required_car_parking_spaces = df['required_car_parking_spaces']
    booking_changes = df['booking_changes']
    days_in_waiting_list = df['days_in_waiting_list']
    adr = df['adr']



    X = pd.concat([hotel, previous_cancellations, lead_time, total_of_special_requests, adr, days_in_waiting_list, booking_changes, required_car_parking_spaces, is_repeated_guest, deposit_type, market_segment], axis=1)
    y = is_cancelled

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    range_k = range(1, 15)
    scores = {}
    scores_list = []
    for k in range_k:
        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        scores[k] = metrics.accuracy_score(y_test,y_pred)
        scores_list.append(metrics.accuracy_score(y_test,y_pred))
        result = metrics.confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(result)
        result1 = metrics.classification_report(y_test, y_pred)
        print("Classification Report:",)
        print (result1)
    plt.plot(range_k,scores_list)
    plt.xlabel("Value of K")
    plt.ylabel("Accuracy")
    plt.show()

    # classifier = KNeighborsClassifier(n_neighbors=12)
    # classifier.fit(X_train, y_train)

if __name__ == '__main__':
    df = loadData()
    # correlationHeatmap(df)
    # plotCancellationData()
    # LDA_isCancelled(df)
    # knn_classifier(df)
    # separateCancellationsMarketSegments(df)
    # separateCancellationsDepositType(df)
    # separateCancellationsRepeatGuests(df)
    knn_classifierNew(df)