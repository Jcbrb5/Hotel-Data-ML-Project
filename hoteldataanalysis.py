import pandas as pd
import matplotlib.pyplot as plt
import statistics as stat
import numpy as np
from sklearn.metrics import r2_score
from statistics import mean

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

# lead1 = hb1["lead_time"]
# adr1 = hb1["adr"]
# polyreg1 = np.poly1d(np.polyfit(adr1, lead1, 20))
#
# lead2 = hb2["lead_time"]
# adr2 = hb2["adr"]
# polyreg2 = np.poly1d(np.polyfit(adr2, lead2, 20))
#
# line = np.linspace(1, 500, 100000)
#
# fig, ax = plt.subplots()
# ax = plt.plot(line, polyreg1(line), c='black')
# ax = plt.scatter(adr1, lead1, s=5)
# ax = plt.xlim([0, 600])
# ax = plt.ylim([0, 700])
# ax = plt.title("ADR vs Lead time for Resort Hotels with regression")
# ax = plt.xlabel("Average Daily Rate (ADR)")
# ax = plt.ylabel("Lead Time")
#
# plt.show()
#
# fig, ax = plt.subplots()
# ax = plt.plot(line, polyreg2(line), c='black')
# ax = plt.scatter(adr2, lead2, s=5)
# ax = plt.xlim([0, 600])
# ax = plt.ylim([0, 700])
# ax = plt.title("ADR vs Lead time for City Hotels with regression")
# ax = plt.xlabel("Average Daily Rate (ADR)")
# ax = plt.ylabel("Lead Time")
# plt.show()
#
# print(r2_score(lead1, polyreg1(adr1)), r2_score(lead2, polyreg2(adr2)))
#
# lead0 = []
# adr0 = []
# lead2s = []
# adr2s = []
# lead15 = []
# adr15 = []
# lead31 = []
# adr31 = []
# lead61 = []
# adr61 = []
# lead91 = []
# adr91 = []
# lead131 = []
# adr131 = []
# lead181 = []
# adr181 = []
# lead271 = []
# adr271 = []
# lead361 = []
# adr361 = []
#
#
# for i in range(lead1.size):
#     if lead1[i] < 2:
#         lead0.append(lead1[i])
#         adr0.append(adr1[i])
#     elif lead1[i] < 15:
#         lead2s.append(lead1[i])
#         adr2s.append(adr1[i])
#     elif lead1[i] < 31:
#         lead15.append(lead1[i])
#         adr15.append(adr1[i])
#     elif lead1[i] < 61:
#         lead31.append(lead1[i])
#         adr31.append(lead1[i])
#     elif lead1[i] < 91:
#         lead61.append(lead1[i])
#         adr61.append(adr1[i])
#     elif lead1[i] < 131:
#         lead91.append(lead1[i])
#         adr91.append(adr1[i])
#     elif lead1[i] < 181:
#         lead131.append(lead1[i])
#         adr131.append(adr1[i])
#     elif lead1[i] < 271:
#         lead181.append(lead1[i])
#         adr181.append(adr1[i])
#     elif lead1[i] < 361:
#         lead271.append(lead1[i])
#         adr271.append(adr1[i])
#     else:
#         lead361.append(lead1[i])
#         adr361.append(adr1[i])
#
#
# medianadr1 = [stat.median(adr0), stat.median(adr2s), stat.median(adr15), stat.median(adr31), stat.median(adr61),
#               stat.median(adr91), stat.median(adr131), stat.median(adr181), stat.median(adr271), stat.median(adr361)]
#
# lead0 = []
# adr0 = []
# lead2s = []
# adr2s = []
# lead15 = []
# adr15 = []
# lead31 = []
# adr31 = []
# lead61 = []
# adr61 = []
# lead91 = []
# adr91 = []
# lead131 = []
# adr131 = []
# lead181 = []
# adr181 = []
# lead271 = []
# adr271 = []
# lead361 = []
# adr361 = []
#
#
# for i in np.arange(lead1.size, lead1.size + lead2.size, 1):
#     if lead2[i] < 2:
#         lead0.append(lead2[i])
#         adr0.append(adr2[i])
#     elif lead2[i] < 25:
#         lead2s.append(lead2[i])
#         adr2s.append(adr2[i])
#     elif lead2[i] < 31:
#         lead15.append(lead2[i])
#         adr15.append(adr2[i])
#     elif lead2[i] < 61:
#         lead31.append(lead2[i])
#         adr31.append(lead2[i])
#     elif lead2[i] < 91:
#         lead61.append(lead2[i])
#         adr61.append(adr2[i])
#     elif lead2[i] < 131:
#         lead91.append(lead2[i])
#         adr91.append(adr2[i])
#     elif lead2[i] < 181:
#         lead131.append(lead2[i])
#         adr131.append(adr2[i])
#     elif lead2[i] < 271:
#         lead181.append(lead2[i])
#         adr181.append(adr2[i])
#     elif lead2[i] < 361:
#         lead271.append(lead2[i])
#         adr271.append(adr2[i])
#     else:
#         lead361.append(lead2[i])
#         adr361.append(adr2[i])
#
#
# medianadr2 = [stat.median(adr0), stat.median(adr2s), stat.median(adr15), stat.median(adr31), stat.median(adr61),
#               stat.median(adr91), stat.median(adr131), stat.median(adr181), stat.median(adr271), stat.median(adr361)]
#
# adrlabel = ['0-1 days', '2-14 days', '15-30 days', '31-60 days', '61-90 days', '91-130 days', '131-180 days', '181-270 days', '271-360 days', '>360 days']
#
# x = np.arange(0, 30, 3)
#
# fig, ax = plt.subplots()
# ax.bar(x-0.4, medianadr1, label='Resort Hotels', width=.8)
# ax.bar(x+0.4, medianadr2, label='City Hotels', width=.8)
# ax.set_xticks(x, labels=adrlabel, fontsize=5)
# ax.set_ylabel('Median ADR')
# ax.set_title("Median ADR for range of lead times")
# ax.legend()
# plt.show()





