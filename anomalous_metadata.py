import pandas as pd
from datetime import datetime
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np 
from sklearn.cluster import KMeans

# Function to preprocess the data
def preprocess_data(df):
    # 1. Owner-Author Mismatch: Create a column where 1 if names are different, otherwise 0
    df['Owner-Author Mismatch'] = df.apply(lambda row: 1 if row['Owner Name'] != row['Author Name'] else 0, axis=1)

    # 2. Camera Details: 1 if value exists, 0 if empty
    df['Camera Details'] = df['Camera Details'].apply(lambda x: 1 if pd.notna(x) and x != "" else 0)

    # 3. Shutter Details: 1 if value exists, 0 if empty
    df['Shutter Details'] = df['Shutter Details'].apply(lambda x: 1 if pd.notna(x) and x != "" else 0)

    # 4. Lens Details: 1 if value exists, 0 if empty
    df['Lens Details'] = df['Lens Details'].apply(lambda x: 1 if pd.notna(x) and x != "" else 0)

    # 5. Last Modified - Creation Time >= 3 Months: 1 if time difference >= 3 months, otherwise 0
    df['Last Modified - Creation Time >= 3 Months'] = df.apply(
        lambda row: 1 if (datetime.strptime(row['Last Modified Time'], '%Y-%m-%d %H:%M:%S') - 
                         datetime.strptime(row['Creation Time'], '%Y-%m-%d %H:%M:%S')).days >= 90 else 0, axis=1)

    # 6. File Size Anomaly: 1 if file size >= 50MB or <= 10KB, otherwise 0
    df['File size anomaly'] = df['File Size (KB)'].apply(lambda x: 1 if x >= 50 * 1024 or x <= 10 else 0)

    # 7. Editing software:
    df['Editing Software Used'] = df['Editing Software Used'].apply(lambda x: 1 if pd.notna(x) and x != '' else 0)

    # 8. Drop the unnecessary columns (Owner Name, Author Name, Creation Time, Last Modified Time, File Size)
    df.drop(columns=['Owner Name', 'Author Name', 'Creation Time', 'Last Modified Time', 'File Size (KB)'], inplace=True)

    return df

# Example of how to use this function with your dataset
if __name__ == "__main__":
    # Assuming you have a DataFrame `df` which is read from your CSV
    df_anomalous = pd.read_csv('anomalous_metadata.csv')
    df_generic = pd.read_csv('generic_metadata.csv')
    #print(df_anomalous)
    #print(df_generic)
    # Preprocess the data
    df_anomalous.drop(columns=['Filename','Watermark or Digital Signature'],inplace=True)
    preprocessed_anomalous_df = preprocess_data(df_anomalous)
    preprocessed_anomalous_df['Anomaly']=1
    #print(df_anomalous)
    #print(preprocessed_anomalous_df)
    preprocessed_generic_df = preprocess_data(df_generic)
    preprocessed_generic_df['Anomaly']=0
    #print(len(preprocessed_generic_df))

    # Combine the datasets (700 anomalous and 700 non-anomcalous for training, 300 anomalous and 300 non-anomalous for testing)
train_data = pd.concat([preprocessed_anomalous_df.iloc[:700], preprocessed_generic_df.iloc[:700]])
test_data = pd.concat([preprocessed_anomalous_df.iloc[700:], preprocessed_generic_df.iloc[700:]])
#print("train data=",train_data)

# Shuffle the training and test datasets
train_data = train_data.sample(frac=1).reset_index(drop=True)
test_data = test_data.sample(frac=1).reset_index(drop=True)

# Split features (X) and labels (y) for training and testing
X_train = train_data.drop(columns=['Anomaly']).values  # Convert to NumPy array
y_train = train_data['Anomaly'].values  # Convert to NumPy array
#print(y_train.)
y_train = np.concatenate((y_train, [0, 0, 1, 1,1,1,1,1,0,1,1]), axis=0)
#print("X_train=",X_train)
X_train = X_train.astype(float)
#print("2",X_train.shape)
X_test = test_data.drop(columns=['Anomaly']).values  # Convert to NumPy array
y_test = test_data['Anomaly'].values  # Convert to NumPy array


# Create and fit the KMeans model
kmeans = KMeans(n_clusters=2, random_state=42,n_init=500,max_iter=1000)
kmeans.fit(X_train)

# Get cluster labels for the dataset
cluster_labels = kmeans.labels_
print("Cluster labels for the dataset:", cluster_labels)

# Function to predict the cluster for a new data point
def predict_cluster(new_data_point):
    # Standardize the new data point
    #new_data_point_scaled = scaler.transform([new_data_point])
    # Predict the cluster
    cluster = kmeans.predict(new_data_point)
    return cluster[0]

# Example new data point
new_data_point = [[1, 1, 1, 0, 0, 1, 1]]
predicted_cluster = predict_cluster(new_data_point)
print("The new data point belongs to cluster:", predicted_cluster)