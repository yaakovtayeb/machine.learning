import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import date_conv

data = pd.read_clipboard(sep='\t') #read from clipboard
show_plots=True
#change month names to numbers
for i in range(0, len(data["Month"])):
    data.loc[i,"Month"] = date_conv.Month2Num(data.loc[i,"Month"])

#remove commas:
data["SW value"]=data["SW value"].str.replace("\,","").astype('float')
data["GA value"]=data["GA value"].str.replace("\,","").astype('float')

#x=list(set(data.Month))
summary=list()
local_summary={}
for sites in set(data["Site"]):
    local_summary["Site"]=sites
    for y in set(data["Year"]):
        local_summary["year"]=y
        for devices in set(data["Device"]):
            tmp_data = data.loc[(data['Site'] == sites) & (data['Device'] == devices) & (data['Year'] == y)]
            y_sw = tmp_data["SW value"]
            y_ga = tmp_data["GA value"]
            local_summary["device"]=devices
            x = tmp_data["Month"]
            #plot
            plt.figure(1, figsize = (8,4))
            plt.plot(x, y_sw, 'b-', label='SW Data', linewidth=2.0)
            plt.plot(x, y_ga, 'r-', label="GA Data", linewidth=2.0)
            plt.xlabel('Month')
            plt.ylabel('Visits')
            plt.legend(loc='upper right')
            plt.suptitle("%s (%d) - %s" % (sites, y, devices))
            ymax=int(np.max(np.concatenate((y_ga,y_sw))))
            plt.ylim([0,ymax])
            if show_plots == True: #display the plot
                #plt.clf()
                plt.show()
            else: #save the plot
                plt.savefig("output\\%s (%d) - %s.png" % (sites, y, devices)) #save the plot to file
                plt.clf()
            local_summary["delta"]=(sum(y_sw)-sum(y_ga))/float(sum(y_ga))
            local_summary["corr"]=np.corrcoef(y_sw,y_ga)[0,1]
            summary.append(dict(local_summary))
            #print details for each site, year, device
            for k,v in local_summary.iteritems():
                print "%s:\t %s" % (k, v)
            # raw_input("Press Enter to continue...") #stop for manual processing
print(summary)
#save output file
output=pd.DataFrame(summary)
output.to_csv('C:\\Users\\yaakov.tayeb\\Desktop\\\dis_data.csv', header=True, mode='a')