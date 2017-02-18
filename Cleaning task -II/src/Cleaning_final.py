
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import pandas as pd
from scipy.stats.kde import gaussian_kde
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDRegressor 
import sklearn.metrics as sm
import sys


# In[3]:

input_path = sys.argv[1]
output_path = sys.argv[2]

print("Reading input files from :")
print(input_path)
zone = input_path.split('/')
zone = zone[len(zone)-1]


# In[4]:

df_flow = pd.read_csv(input_path + "/flow.tsv" ,delimiter= "\t",header = None)
df_speed = pd.read_csv(input_path +"/speed.tsv",delimiter= "\t",header = None)
df_occupancy = pd.read_csv(input_path +"/occupancy.tsv",delimiter= "\t",header = None)
df_timestamp = pd.read_csv(input_path +"/timestamp.tsv",delimiter= "\t",header = None)
df_prob= pd.read_csv(input_path +"/prob.tsv",delimiter= "\t",header = None)
df_lanes = pd.read_table(input_path +"/lanes.txt",delimiter= "\t",header = None)


# In[6]:

df_timestamp['timestamp_date'], df_timestamp['timestamp_time'] = df_timestamp[0].str.split('T', 1).str
df_timestamp['Hour'] ,_= df_timestamp['timestamp_time'].str.split(':', 1).str


# In[7]:

lanes = str(df_lanes[0].values[0]).split(' ')
num_lanes = len(lanes)
lanes = list(range(0, num_lanes))


# In[9]:

predicted_flows = pd.DataFrame()


# In[11]:

for lane in lanes :

    print("processing lane "+str(lane))

    remaining_lanes = list(set(lanes).difference([lane]))
    train_sample=pd.DataFrame()
    for rem in remaining_lanes:
        train_sample[str(rem)]= df_flow[rem] 
       
    train_sample['speed']= df_speed[0]
    train_sample['occupancy']= df_occupancy[0]
    train_sample['Hour']= df_timestamp['Hour'] 
    train_sample['Prob']= df_prob[0] 
    train_sample['FinalFlow'] = df_flow[lane]
    
    Missing_sample =  pd.DataFrame()
    if (train_sample['FinalFlow'].isnull().values.any()):
        train_sample['FinalFlow'] = train_sample['FinalFlow'].replace(np.nan,' ',regex=True)
        Missing_sample = train_sample[train_sample['FinalFlow'] == ' ']
        train_sample['FinalFlow'] = train_sample['FinalFlow'].replace(' ',np.nan,regex=True)
    
    train_sample=train_sample.fillna(-1)
    
    corrupt1_sample = pd.DataFrame()
    corrupt2_sample = pd.DataFrame()
    corrupt3_sample = pd.DataFrame()
    corrupt4_sample = pd.DataFrame()
    corrupt5_sample = pd.DataFrame()
    
    for rem in remaining_lanes:
        corrupt1_sample = pd.concat([train_sample[train_sample[str(rem)]<0],corrupt1_sample])
        train_sample = train_sample[train_sample[str(rem)]>= 0]
    
    corrupt2_sample = train_sample[train_sample['speed']<0]
    train_sample = train_sample[train_sample['speed']>= 0]
    
    corrupt3_sample = pd.DataFrame(train_sample[train_sample['FinalFlow']<0])
    train_sample = train_sample[train_sample['FinalFlow']>=0]
    
    corrupt4_sample = pd.DataFrame(train_sample[train_sample['occupancy']<0])
    train_sample = train_sample[train_sample['occupancy']>=0]
    
    corrupt5_sample = pd.DataFrame(train_sample[train_sample['Prob']<=0])
    train_sample = train_sample[train_sample['Prob']>0]
    
    
    
    corrupt_sample = pd.DataFrame()
    corrupt_sample = pd.concat([corrupt1_sample,corrupt2_sample,corrupt3_sample,corrupt4_sample,corrupt5_sample])
    
    grouped = train_sample.groupby([train_sample['Hour']]).mean()
    for i in range(0,24):
       for rem in remaining_lanes:
            corrupt_sample.ix[corrupt_sample[str(rem)]<0,str(rem)] = grouped.ix[i,str(rem)]
    
       corrupt_sample.ix[corrupt_sample['speed']<0,'speed'] = grouped.ix[i,'speed']
       corrupt_sample.ix[corrupt_sample['occupancy']<0,'occupancy'] = grouped.ix[i,'occupancy']
       corrupt_sample.ix[corrupt_sample['Prob']<0,'Prob'] = grouped.ix[i,'Prob']
        
    train_sample_labels= train_sample['FinalFlow']
    corrupt_sample_labels= corrupt_sample['FinalFlow']
    
    train_sample = train_sample.iloc[:,0:len(remaining_lanes)+4]
    corrupt_sample= corrupt_sample.iloc[:,0:len(remaining_lanes)+4]
    
    model = SGDRegressor(eta0=0.00005).fit(train_sample, train_sample_labels)
    complete = pd.concat([train_sample,corrupt_sample])
    complete.sort_index(inplace=True)
    
    predictions = model.predict(complete)
  
    predicted_flows[str(lane)] = predictions
    if(len(Missing_sample)>0):
        predicted_flows.ix[Missing_sample.index,str(lane)] = np.nan


# In[ ]:

filename=zone + ".flow.txt"
print("Task Completed")
print("Writing output file to:")
print(output_path)

predicted_flows.to_csv(output_path+"/"+filename,sep = "\t",index=False,header=False)


# In[31]:




# In[ ]:




# In[ ]:



