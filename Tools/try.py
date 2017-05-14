import pandas as pd
import matplotlib.pyplot as plt

#data = pd.read_clipboard(sep='\t') #read from clipboard
data = pd.read_csv("gradient_descent_example_data.csv", sep=",", header=None)
data = pd.read_clipboard(sep='\t', header=None)
print(data.columns.values) #show col names:
data.columns = ['year','month', 'traffic', 'device', 'visits', 'users', 'bouncerate', 'clickout', 'pageviews', 'unique_pageviews', 'measurement']
#change content of table to fit different sources.
set(data.traffic)
data.loc[data["traffic"]=="Referrals", ["traffic"]]="referral" #change multiple cells at once by condition
data.loc[data["traffic"]=="Direct", ["traffic"]]="direct"
data.loc[data["traffic"]=="Organic Search", ["traffic"]]="organic"
data.loc[data["traffic"]=="Paid Search", ["traffic"]]="paid"
data.loc[data["traffic"]=="Display Ads", ["traffic"]]="ads"
data.loc[data["traffic"]=="Social", ["traffic"]]="social"
data.loc[data["traffic"]=="Mail", ["traffic"]]="mail"

#adding new column:
data["device2"]=data["device"]
data.loc[data["device2"]=="tablet", ["device2"]]="mobile"

#change month string to a date object
data["date"]=datetime.date(1984,10,21)
for i in range(0, len(data)):
    m = date_conv.Month2Num(data["month"][i])
    y = data["year"][i]
    data.loc[i,"date"] = datetime.date(y, m, 1)

#plot SW against GA for organice and paid
X = data.loc[(data["traffic"]=="paid") & (data["measurement"]=="SW"), ["date"]]
Y = data.loc[(data["traffic"]=="paid") & (data["measurement"]=="SW"), ["visits"]]
Y2 = data.loc[(data["traffic"] == "paid") & (data["measurement"] == "GA") & (data["device"]=="desktop"), ["visits"]]
plt.plot (X, Y, color='lightblue', alpha=1.00, marker='o', label="SW", linewidth=2)
plt.plot (X, Y2, color='orange', alpha=1.00, marker='o', label="GA", linewidth=2)
plt.legend(loc="upper right")
plt.grid(color='gray', linestyle='-', linewidth=0.05, fillstyle="full")
plt.show()



