#!/usr/bin/env python3

# Import required libraries
import pandas as pd
import nolds
from pathlib import Path


# function to only get relevant correlations
def get_redundant_pairs(df):
    
    # Get diagonal and lower triangular pairs of correlation matrix 
    
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop
            
#def entropy function to create df
def compute_features(file_name):
    
    # compute features and return them as dict
    
    raw_data = pd.read_csv(file_name, header=None)
    raw_data.columns = ["emg1","emg2","emg3","emg4","airborne", "acc_u_x" , "acc_u_y" , "acc_u_z" , "gonio_x",
                        "acc_l_x","acc_l_y" , "acc_l_z", "gonio_y","gyro_u_x", "gyro_u_y","gyro_u_z",
                        "gyro_l_x","gyro_l_y","gyro_l_z"]

    columns = ["emg1","emg2","emg3","emg4","airborne", "acc_u_x" , "acc_u_y" , "acc_u_z" , "gonio_x",
               "acc_l_x","acc_l_y" , "acc_l_z", "gonio_y","gyro_u_x", "gyro_u_y","gyro_u_z",
               "gyro_l_x","gyro_l_y","gyro_l_z"]
    
    # mean center every column
    raw_data = raw_data.subtract(raw_data.mean())
    
    #create new df for emg at 1000Hz and other sensors at 100 Hz
    emg_raw = raw_data.iloc[:,:4]
    other_raw = raw_data.iloc[::10,4:].reset_index(drop = True)
    
    # make columns
    features_list = ["min", "max", "std", "count_distinct", "sge", "iqr", "skew", "ent"]

    list_names = ["id", "n_rows"]

    # feature column names
    for feature in features_list:
        for col in columns:
            list_names.append(col+"_"+feature)
    

    # Compute the values for each column and attach to list
    new_value_set = []
    
    #id
    new_value_set.append(str(file_name)[40:])
    
    #n_rows
    new_value_set.append(len(raw_data))
    
    #min
    new_value_set.extend(emg_raw.min().tolist())
    new_value_set.extend(other_raw.min().tolist())
    
    #max
    new_value_set.extend(emg_raw.max().tolist())
    new_value_set.extend(other_raw.max().tolist())
    
    #std
    new_value_set.extend(emg_raw.std().tolist())
    new_value_set.extend(other_raw.std().tolist())
    
    #count_distinct
    new_value_set.extend(emg_raw.nunique().tolist())
    new_value_set.extend(other_raw.nunique().tolist())

    #sge
    for col in columns:
        if "emg" in str(col):
            column = raw_data[col]
        else:
            column = raw_data[col][::10]  
            
        result = column.iloc[0] - column.iloc[-1]
        if result > 0:
            new_value_set.append(1)
        else:
            new_value_set.append(0)
    
    #iqr
    for column in emg_raw.columns:
        Q1 = raw_data[column].quantile(0.25)
        Q3 = raw_data[column].quantile(0.75)
        new_value_set.append(Q3 - Q1)
    
    for column in other_raw.columns:
        Q1 = raw_data[column].quantile(0.25)
        Q3 = raw_data[column].quantile(0.75)
        new_value_set.append(Q3 - Q1)
    
    #skew
    new_value_set.extend(emg_raw.skew().tolist())
    new_value_set.extend(other_raw.skew().tolist())
    
    #ent
    for name in columns:
        if "emg" in name:
            new_value_set.append(nolds.sampen(raw_data[name]))
        else:
            column = raw_data[name]
            new_value_set.append(nolds.sampen(column[::10]))
    
    #cor
    s_e = emg_raw.corr().unstack()
    s_e.drop(labels = get_redundant_pairs(emg_raw), inplace = True)
    
    for indx1, indx2 in s_e.index:
        if indx1 != indx2:
            new_value_set.append(s_e[indx1][indx2])
    
    
    s_o = other_raw.corr().unstack()
    s_o.drop(labels = get_redundant_pairs(other_raw), inplace = True)
    
    for indx1, indx2 in s_o.index:
        if indx1 != indx2:
            new_value_set.append(s_o[indx1][indx2])
    
    #make cor names
    for indx1, indx2 in s_e.index:
        list_names.append("cor_"+str(indx1)+"_"+str(indx2))
    for indx1, indx2 in s_o.index:
        list_names.append("cor_"+str(indx1)+"_"+str(indx2))
            
    # create dict
    final_dict = {}
    for i in range(len(list_names)):
        final_dict.update({list_names[i]: new_value_set[i]})

    return final_dict

# set the variables for the parent data folder and train file
dataFolder = '/home/BBDC-GTA/bbdc_2019_Bewegungsdaten/'

# make a list of all the csv files in the parent and sub directories
result = list(Path(dataFolder).rglob("Subject*.[cC][sS][vV]"))

df = pd.DataFrame(compute_features(result[0]), index = [0])
for i in range(1, len(result)):
    df = df.append(compute_features(result[i]), ignore_index = True)

df.to_csv('new_feature_df.csv', index=False)