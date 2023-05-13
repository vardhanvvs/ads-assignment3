# -*- coding: utf-8 -*-
"""
Created on Thu May 11 11:12:38 2023

@author: Vardhan
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import scipy.optimize as opt
import cluster_tools as ct
import errors as err
import importlib


def read_data(file_name):
    '''
    

    Parameters
    ----------
    file_path : STR
        takes input as string as file name and creates a dataframe.

    Returns
    -------
    data : Pandas.DataFrame
        returns the dataframe after manipulating the data.

    '''
    df_name=pd.read_csv(file_name)
    df_name=df_name.set_index('Country Name',drop=True)
    df_name=df_name.loc[:,'1990':'2014']
    return df_name



def transpose(df_name):
    '''
    

    Parameters
    ----------
    data : STR
        takes input as csv file and transposes the given file.

    Returns
    -------
    data_tr : Pandas.DataFrame
        returns a dataframe after transposing.

    '''
    df_name_tr=df_name.transpose()
    
    return df_name_tr



def n_clusters(data,data_norm,a,b):
    
    n_clusters=[]
    cluster_score=[]
    
    
    for ncluster in range(2, 10):
        
        # set up the clusterer with the number of expected clusters
        kmeans = cluster.KMeans(n_clusters=ncluster)

        # Fit the data, results are stored in the kmeans object
        kmeans.fit(data_norm)     # fit done on x,y pairs

        labels = kmeans.labels_
        
        # extract the estimated cluster centres
        cen = kmeans.cluster_centers_

        # calculate the silhoutte score 
        print(ncluster, skmet.silhouette_score(data, labels))
        
        n_clusters.append(ncluster)
        cluster_score.append(skmet.silhouette_score(data, labels))
        
    n_clusters=np.array(n_clusters)
    cluster_score=np.array(cluster_score)
        
    best_ncluster=n_clusters[cluster_score==np.max(cluster_score)]
    print('n clusters',best_ncluster[0])
    
    return best_ncluster[0]
        
        
    
def clusters_centers(df_norm,ncluster,a,b):
    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=ncluster)

    # Fit the data, results are stored in the kmeans object
    kmeans.fit(df_norm)     # fit done on x,y pairs

    labels = kmeans.labels_
    df_norm['labels']=labels
    # extract the estimated cluster centres
    cen = kmeans.cluster_centers_

    centres = np.array(cen)
    xcentres = cen[:, 0]
    ycentres = cen[:, 1]


    # cluster by cluster
    plt.figure(figsize=(8.0, 8.0))

    cm = plt.cm.get_cmap('tab10')
    plt.scatter(df_norm[a], df_norm[b], 10, labels, marker="o", cmap=cm)
    plt.scatter(xcentres, ycentres, 45, "k", marker="d")
    plt.xlabel(f"ele per capita({a})")
    plt.ylabel(f"ele per capita({b})")
    plt.title('Clusters of Countries over electricity per capita in 1990 and 2014')
    plt.show()

    print(cen)

    
    return 
    
#----------------------------------------

#using the readfile function to read files
land_area_sq = pd.read_csv("urbanpopulation.csv")
forest_sq = pd.read_csv("electricity_per_capita.csv")
#using describe method to get meaningful insights
print(land_area_sq.describe())
print(forest_sq.describe())

# drop rows with nan's in 2020
land_area_sq = land_area_sq[land_area_sq["2010"].notna()]
print(land_area_sq.describe())
# alternative way of targetting one or more columns
forest_sq = forest_sq.dropna(subset=["2010"])
print(forest_sq.describe)

#extracting the year 2020 from each dataframe
#we used copy so that any changes to the columns will not effect the original dataframe
land_area_sq2020 = land_area_sq[["Country Name", "Country Code", "2010"]].copy()
forest_sq2020 = forest_sq[["Country Name", "Country Code", "2010"]].copy()

print(land_area_sq2020.describe())
print(forest_sq2020.describe())


#merging the required dataframes using merge method
df_2010 = pd.merge(land_area_sq2020, forest_sq2020, on="Country Name", how="outer")
print(df_2010.describe())
df_2010.to_excel("agr_for2020.xlsx")



print(df_2010.describe())
df_2010 = df_2010.dropna() # entries with one datum or less are useless.
print()
print(df_2010.describe())
# rename columns
df_2010 = df_2010.rename(columns={"2010_x":"urbanpopulation", "2010_y":"electricity_per_capita"})


pd.plotting.scatter_matrix(df_2010, figsize=(12, 12), s=5, alpha=0.8)




#------------------------------------------



df_co2 = read_data("electricity_per_capita.csv")
print(df_co2.describe())

df_co2_tr=transpose(df_co2)
print(df_co2_tr.head())

df_co3 = df_co2[['1990','1995','2000','2005','2010','2014']]
print(df_co3.describe())


column_1="1990"
column_2="2014"
df_ex = df_co3[[column_1, column_2]]  # extract the two columns for clustering


df_ex = df_ex.dropna()  # entries with one nan are useless
print(df_ex.head())

# normalise, store minimum and maximum
df_norm, df_min, df_max = ct.scaler(df_ex)

print()
print("n  score")
# loop over number of clusters
ncluster=n_clusters(df_ex,df_norm,column_1,column_2)

clusters_centers(df_norm, ncluster,column_1,column_2)

clusters_centers(df_ex, ncluster,column_1,column_2)

print(df_ex[df_ex['labels']==ncluster-2])


df_co2_tr=df_co2_tr.loc[:,'Finland']
df_co2_tr=df_co2_tr.dropna(axis=0) 
print('Transpose')
print(df_co2_tr.head())


new_ele_p_cap=pd.DataFrame()

new_ele_p_cap['Year']=pd.DataFrame(df_co2_tr.index)
new_ele_p_cap['ele_per_cap']=pd.DataFrame(df_co2_tr.values)

print(new_ele_p_cap.head())

new_ele_p_cap.plot("Year", "ele_per_cap")
plt.ylabel('electricity in kwh')
plt.title('Electricity per individual in kwh')
plt.show()


new_ele_p_cap["Year"] = pd.to_numeric(new_ele_p_cap["Year"])

def logistic(t, n0, g, t0): 
    """Calculates the logistic function with scale factor n0 
    and growth rate g"""
    
    log_fun = n0 / (1 + np.exp(-g*(t - t0)))
    
    return log_fun

importlib.reload(opt)

param, covar = opt.curve_fit(logistic, new_ele_p_cap["Year"],
                             new_ele_p_cap["ele_per_cap"], 
                             p0=(1.2e12, 0.03, 1990.0))

sigma = np.sqrt(np.diag(covar))

year = np.arange(1990, 2035)
forecast = logistic(year, *param)
low, up = err.err_ranges(year, logistic, param, sigma)
new_ele_p_cap["fit"] = logistic(new_ele_p_cap["Year"], *param)

new_ele_p_cap.plot("Year", ["ele_per_cap", "fit"])
plt.fill_between(year,low,forecast,color="yellow",alpha=0.7)
plt.fill_between(year,up,forecast,color="yellow",alpha=0.7)
plt.ylabel('electricity in kwh')
plt.title('Electricity per individual in kwh')
plt.show()

print("turning point", param[2], "+/-", sigma[2])
print("ele_per_cap at turning point", param[0]/1e9, "+/-", sigma[0]/1e9)
print("rate of growth", param[1], "+/-", sigma[1])



year = np.arange(1960, 2035)
forecast = logistic(year, *param)
low, up = err.err_ranges(year, logistic, param, sigma)
plt.figure()
plt.plot(new_ele_p_cap["Year"], new_ele_p_cap["ele_per_cap"], label="ele_per_cap",
         color='#8A307F')
plt.plot(year, forecast, label="forecast",color='k')
plt.fill_between(year, low, forecast, color="skyblue", alpha=0.7)
plt.fill_between(year, forecast, up, color="blue", alpha=0.7)
plt.xlabel("year")
plt.ylabel("ele_per_cap")
plt.legend(loc='upper left')
plt.title('electricity per capita forecast for Finland')
plt.savefig('us.png',bbox_inches='tight',dpi=300)
plt.show()

print(logistic(2020, *param)/1e9)
print(err.err_ranges(2020, logistic, param, sigma))

# assuming symmetrie estimate sigma
gdp2020 = logistic(2020, *param)/1e9

low, up = err.err_ranges(2020, logistic, param, sigma)
sig = np.abs(up-low)/(2.0 * 1e9)

print("ele_per_cap 2020", gdp2020*1e9, "+/-", sig)