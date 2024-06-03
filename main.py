import model as m
import streamlit as st

def display_summary(df):
    st.write(df.describe())

def display_missing_values(df):
    st.write(df.isnull().sum())

def display_histograms(df):
    df.hist(figsize=(10, 8))
    m.plt.tight_layout()
    st.pyplot(m.plt)

st.title("K_Medoids Clustering:")
st.caption("Made by Rimsha :heart: :sparkles:")

# File uploader for CSV file
file = st.file_uploader("Select a CSV file to analyze them", type="csv")

# Numeric input for number of clusters
n_clusters = st.number_input("Enter number of clusters", min_value=2, max_value=20, step=1)

if file is not None:
    # Load the dataset
    df = m.load_dataset(file)
    
    # Perform K-Medoids clustering
    clustered_df, X_scaled, message = m.perform_kmedoids_clustering(df, n_clusters)
    st.info(message)
    
    if clustered_df is not None:
        st.write("## Data Preview")
        st.write(clustered_df.head())
        
        # Select operation for further data analysis
        operation = st.selectbox(
            "Select Operation",
            ("Display Summary", "Display Missing Values", "Display Histograms", "Display Scatter Plot")
        )
        
        if operation == "Display Summary":
            display_summary(clustered_df)
        elif operation == "Display Missing Values":
            display_missing_values(clustered_df)
        elif operation == "Display Histograms":
            display_histograms(clustered_df)
       