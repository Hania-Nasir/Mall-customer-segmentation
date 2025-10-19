from sklearn.cluster import KMeans
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(page_title="Customer Segmentation",layout="centered")

st.title("Customer Segmentation app")
st.write("upload your dataset and explore customer segments interactively")

#Upload Dataset

uploaded=st.file_uploader("upload your csv file here",type=["csv"])

if uploaded is not None:
    df=pd.read_csv(uploaded)
    st.success("file uploaded sucessfully")
    st.subheader("dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Summary")
    st.write(f"**shape:** {df.shape[0]} rows x {df.shape[1]} columns")
    st.write("**Missing Values:**")
    st.write(df.isnull().sum())

    #Select numeric Columns only

    num_col=df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if len(num_col) < 2:
        st.warning("select atleast 2 columns for PCA and clustering")
    else:
        selected_features=st.multiselect(
            "select numeric features to use:",
            options=num_col
            )

    if len(selected_features) >= 2:
        X=df[selected_features]

        #Standardize data

        scaler=StandardScaler()
        X_scaled=scaler.fit_transform(X)

        #PCA option

        st.divider()
        st.subheader("PCA for Dimentionality Reduction")

        pca_option=st.checkbox("apply PCA for 2D visualization")
        if pca_option:
            n_comp=st.slider("Select number of PCA components",2,5,2)
            pca=PCA(n_components=n_comp)
            pca_result=pca.fit_transform(X_scaled)
            st.write(f"Explained Variance Ratio:{np.sum(pca.explained_variance_ratio_):.2%}")

        #Elbow Method    

        st.divider()
        st.subheader("Elbow Method to find Optimal K")

        if st.button("Run Elbow Method"):
            wcss=[]
            for i in range(1,11):
                kmeans=KMeans(n_clusters=i,random_state=42)
                kmeans.fit(X_scaled)
                wcss.append(kmeans.inertia_)

            fig , ax = plt.subplots()
            ax.plot(range(1,11),wcss,marker="o")
            ax.set_xlabel("Number of Clusters (k)")
            ax.set_ylabel("WCSS")
            ax.set_title("Elbow Method for Optimal k")
            st.pyplot(fig)

        #KMeans Clustering
        st.divider()
        st.subheader("Apply KMeans Clustering")

        k=st.slider("Select number of clusters(k)",2,10,3)
        if st.button("Run KMeans Clustering"):
            kmeans=KMeans(n_clusters=k,random_state=42)
            df["Clusters"]=kmeans.fit_predict(X_scaled)

            st.success("Clustering Complete")
            st.dataframe(df.head())

            #Cluster Visualization
            if X_scaled.shape[1] >= 2:
                fig, ax = plt.subplots()
                sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=df["Clusters"], palette="viridis")
                ax.set_title("Customer Clusters (2D View)")
                st.pyplot(fig)

            st.subheader("Clusters Summary")
            st.dataframe(df.groupby("Clusters")[selected_features].mean().round(2))