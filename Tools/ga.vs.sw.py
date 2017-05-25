import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
filepath=os.getcwd()+'/github/similar/Clients/india.market.gasw.tsv'
filepath="/Users/yaakovtayeb/github/similar/Clients/india.market.gasw.tsv"
data = pd.read_csv(filepath, sep="\t", header='infer')
data.columns.values
data.columns = [new.replace("ga_analysis.","") for new in  data.columns.values] #remove headers prefix

#plot and check correlation between ga and sw for each month at 16 and 17 for india sites.
for year in set(data["year"]):
    for month in set(data.loc[data["year"]==year]["month"]):
        ga=data.loc[(data["month"]==month) & (data["year"]==year)]["visitsmobile"]
        sw=data.loc[(data["month"]==month) & (data["year"]==year)]["panel_est_visits"]
        R=round(np.corrcoef(ga,sw)[0,1],2) #correlation coefficient 
        plt.figure(figsize=(9,6))
        plt.scatter(ga, sw, color=[0.36, 0.66, 0.63], marker='x')  
        plt.plot(ga,ga, color=[0.36, 0.4, 0.65])
        plt.xlabel('google analytics')  
        plt.ylabel('similarweb')  
        plottit='India Google Analytics VS. SimilarWeb. R:'+str(R)+" - "+str(year)+": "+str(month)
        
        plt.title(plottit)
        xy_lab=np.arange(5)
        xy_lab*=int((max(ga)*1.2)/5)
        plt.xticks(xy_lab)
        plt.yticks(xy_lab)
        #plt.show() #display the plot
        plt.savefig("output/%d.%d.png" % (year, month)) #save the plot to fil
        #raw_input("Press Enter to continue...") #stop for manual processing

#plt.ylim([0,1.2*ymax])
#output=pd.DataFrame(summary)
#output.to_csv('C:\\Users\\yaakov.tayeb\\Desktop\\\dis_data.csv', header=True, mode='a')