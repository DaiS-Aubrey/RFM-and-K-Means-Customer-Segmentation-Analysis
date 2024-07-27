#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().system('pip install lifetimes')
get_ipython().system('pip install jcopml')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifetimes.utils import summary_data_from_transaction_data



# Load the dataset
df = pd.read_csv("Pakistan Biggest Ecommerce Analysis-Ready Dataset.csv")

# List of columns to remove from the dataframe
columns_to_remove = [
    'increment_id', 'sales_commission_code', 'discount_amount', 
    'payment_method', 'Working Date', 'BI Status', ' MV ', 
    'Year', 'Month', 'FY', 'M-Y'
]

# Drop the specified columns
df.drop(columns=columns_to_remove, inplace=True)

# Filter the dataframe to only include completed orders
df_complete = df[df['status'] == 'complete']

# Get the number of unique categories in 'category_name_1'
x = df['category_name_1'].value_counts().nunique()
print("Number of categories: {}".format(x))

# Create a dataframe with the count of completed orders by category
df_category = pd.DataFrame(df_complete['category_name_1'].value_counts()).reset_index()
df_category.columns = ['category', 'quantity']
print(df_category)

# Create a dictionary with the count of completed orders by category
barplot = dict(df_complete['category_name_1'].value_counts())
print(pd.DataFrame(barplot, index=['count']))

# Get the series of counts of completed orders by category
category_count_series = df_complete['category_name_1'].value_counts()

# Plot the counts of completed orders by category
plt.figure(figsize=(18, 6))
sns.barplot(x=category_count_series.index, y=category_count_series.values)
plt.xlabel('Category Name')  
plt.ylabel('Counts')         
plt.title('Counts of Complete Orders by Category', fontsize=15)
plt.xticks(rotation=45)  





labels = category_count_series.index
sizes = category_count_series.values

# 创建饼状图
plt.figure(figsize=(11, 11))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)

# 添加标题
plt.title('Distribution of Complete Orders by Category')

# 显示图表
plt.show()




# Convert 'created_at' to datetime format
df_complete['created_at'] = pd.to_datetime(df_complete['created_at'], format='%m/%d/%y')

# Filter data for 'Mobiles & Tablets' category
df_mobiles_tablets = df_complete[df_complete['category_name_1'] == 'Mobiles & Tablets']
df_mobiles_tablets.tail()

# Remove rows where 'Customer ID' is NaN
df_mobiles_tablets = df_mobiles_tablets[~df_mobiles_tablets['Customer ID'].isna()]

# Print the number of unique 'Customer ID's
x = df_mobiles_tablets['Customer ID'].value_counts().nunique()
print(x)

# Filter rows where 'qty_ordered' and 'price' are greater than 0
df_mobiles_tablets = df_mobiles_tablets[df_mobiles_tablets['qty_ordered'] > 0]
df_mobiles_tablets = df_mobiles_tablets[df_mobiles_tablets['price'] > 0]

# Calculate 'Revenue' for each order
df_mobiles_tablets['Revenue'] = df_mobiles_tablets['qty_ordered'] * df_mobiles_tablets['price']
df_mobiles_tablets.head()

# Create RFM summary data from transaction data
rfm = summary_data_from_transaction_data(df_mobiles_tablets, 'Customer ID', 'created_at', monetary_value_col='Revenue').reset_index()
rfm.head()

# Sort RFM data by 'monetary_value' in descending order
rfm_sorted = rfm.sort_values(by='monetary_value', ascending=False)
rfm_sorted.head()

# Filter RFM data where 'frequency' is greater than 0 and 'monetary_value' is less than or equal to 628000
rfm = rfm[rfm['frequency'] > 0]
rfm = rfm[rfm['monetary_value'] <= 628000]

# Plot histogram of 'monetary_value'
plt.hist(rfm['monetary_value'])

# Calculate quartiles for RFM data
quartiles = rfm.quantile(q=[0.25, 0.5, 0.75])
print(quartiles)

# Define functions to assign scores based on recency, frequency, and monetary value quartiles
def recency_score(data):
    if data <= 107:
        return 1
    elif data <= 212:
        return 2
    elif data <= 301:
        return 3
    else:
        return 4

def frequency_score(data):
    if data <= 1:
        return 1
    elif data <= 3:
        return 2
    elif data <= 5:
        return 3
    else:
        return 4

def monetary_value_score(data):
    if data <= 183.4475:
        return 1
    elif data <= 302.6700:
        return 2
    elif data <= 434.7200:
        return 3
    else:
        return 4

# Apply the scoring functions to RFM data
rfm['R'] = rfm['recency'].apply(recency_score)
rfm['F'] = rfm['frequency'].apply(frequency_score)
rfm['M'] = rfm['monetary_value'].apply(monetary_value_score)

rfm.head()

# Calculate RFM score and assign labels based on the score
rfm['RFM_score'] = rfm[['R', 'F', 'M']].sum(axis=1)
rfm['label'] = 'Bronze'

rfm.loc[rfm['RFM_score'] > 4, 'label'] = 'Silver'
rfm.loc[rfm['RFM_score'] > 6, 'label'] = 'Gold'
rfm.loc[rfm['RFM_score'] > 8, 'label'] = 'Platinum'
rfm.loc[rfm['RFM_score'] > 10, 'label'] = 'Diamond'

rfm.head()

# Create a dictionary with the count of each label
barplot = dict(rfm['label'].value_counts())
print(pd.DataFrame(barplot, index=['count']))

# Plot the count of each label
label_count = rfm['label'].value_counts()
sns.barplot(x=label_count.index, y=label_count.values)


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Standardize the RFM scores
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['recency', 'frequency', 'monetary_value']])

# Determine the number of clusters using the elbow method. sse: sum of Squared Errors, the sum of squared distances between data points and their respective cluster centers.
sse = []
for k in range(1, 15):
    kmeans = KMeans(n_clusters=k, random_state=10)
    kmeans.fit(rfm_scaled)
    sse.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(15, 6))
plt.plot(range(1, 15), sse, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.title('Elbow Method for Determining Optimal k')
plt.show()





# Determine the optimal number of clusters
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=10)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled) #拟合模型+生成聚类标签


# Examine the centroids of the clusters
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_centers_df = pd.DataFrame(cluster_centers, columns=['recency', 'frequency', 'monetary_value'])
print(cluster_centers_df)

# Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='recency', y='frequency', hue='Cluster', data=rfm, palette='viridis')
plt.title('Clusters of Customers Based on RFM Scores')
plt.show()

# Analyze the distribution of monetary value in each cluster
plt.figure(figsize=(10, 6))
sns.boxplot(x='Cluster', y='monetary_value', data=rfm)
plt.title('Monetary Value Distribution in Each Cluster')
plt.show()

# Analyze the distribution of frequency in each cluster
plt.figure(figsize=(10, 6))
sns.boxplot(x='Cluster', y='frequency', data=rfm)
plt.title('Frequency Distribution in Each Cluster')
plt.show()

# Analyze the distribution of recency in each cluster
plt.figure(figsize=(10, 6))
sns.boxplot(x='Cluster', y='recency', data=rfm)
plt.title('Recency Distribution in Each Cluster')
plt.show()




rfm['KMeans_label'] = 'Bronze' 
rfm.loc[rfm['Cluster'] > 0, 'KMeans_label'] = 'Silver' 
rfm.loc[rfm['Cluster'] > 1, 'KMeans_label'] = 'Gold'
rfm.loc[rfm['Cluster'] > 2, 'KMeans_label'] = 'Platinum'
rfm.loc[rfm['Cluster'] > 3, 'KMeans_label'] = 'Diamond'







barplot = dict(rfm['KMeans_label'].value_counts())
bar_names = list(barplot.keys())
bar_values = list(barplot.values())
plt.bar(bar_names,bar_values)
print(pd.DataFrame(barplot, index=[' ']))







