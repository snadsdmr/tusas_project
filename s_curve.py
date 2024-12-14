import streamlit as st
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from scipy.optimize import curve_fit

# Initialize Elasticsearch connection
def initialize_elasticsearch():
    try:
        es = Elasticsearch(
            "https://localhost:9200",
            basic_auth=("elastic", ""),
            verify_certs=False,
            timeout=60  # Increase the timeout to 60 seconds
        )
        if es.ping():
            st.success("Successfully connected to Elasticsearch!")
            return es
        else:
            st.error("Oops! Cannot connect to Elasticsearch!")
            return None
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return None

# Function to perform the search with relevancy filtering
def search_and_filter(es, input_keyword, threshold=0.7):
    model = SentenceTransformer('all-mpnet-base-v2')
    vector_of_input_keyword = model.encode(input_keyword)

    query = {
        "field": "DescriptionVector",
        "query_vector": vector_of_input_keyword,
        "k": 500,
        "num_candidates": 500,
    }

    try:
        res = es.knn_search(index="uav_patents", knn=query, source=["CPC", "_score"])
        results = [hit for hit in res["hits"]["hits"] if hit["_score"] >= threshold]
        return results
    except Exception as e:
        st.error(f"Error during search: {e}")
        return []

# Function to find the most used CPC codes
def find_most_used_cpc_codes(results):
    try:
        cpc_codes = []
        for result in results:
            cpc_field = result["_source"].get("CPC", "")
            if cpc_field:
                codes = [code[:4] for code in cpc_field.split(" | ") if code[:4].isalnum()]
                cpc_codes.extend(codes)

        # Count the occurrences of each CPC code
        cpc_counter = Counter(cpc_codes)
        most_common = cpc_counter.most_common(5)
        return most_common
    except Exception as e:
        st.error(f"Error processing CPC codes: {e}")
        return []

# Function to retrieve yearly data for a specific CPC code
def get_yearly_data(es, cpc_code):
    try:
        query = {
            "size": 10000,
            "query": {
                "wildcard": {
                    "CPC.keyword": f"{cpc_code}*"
                }
            },
            "_source": ["CPC", "Application Date"]
        }

        res = es.search(index="uav_patents", body=query)

        years = []
        for doc in res["hits"]["hits"]:
            application_date = doc["_source"].get("Application Date", None)
            if application_date:
                year = int(application_date.split("-")[0])
                years.append(year)

        return years
    except Exception as e:
        st.error(f"Error retrieving yearly data: {e}")
        return []

# Function to plot the S-curve
def plot_s_curve(df_yearly, selected_cpc):
    years = df_yearly.index - df_yearly.index[0]
    cumulative_counts = df_yearly["Cumulative"]

    initial_guesses = [cumulative_counts.max(), 1, 0.1]
    params, _ = curve_fit(logistic_growth, years, cumulative_counts.values, p0=initial_guesses, maxfev=5000)
    K, P0, r = params

    future_years = np.arange(0, 2050 - df_yearly.index[0], 1)
    forecasted_patents = logistic_growth(future_years, K, P0, r)

    plt.figure(figsize=(10, 6))
    plt.scatter(df_yearly.index, cumulative_counts, label='Historical Data', color='blue')
    plt.plot(future_years + df_yearly.index[0], forecasted_patents, label='Logistic Growth Model (Forecast)', color='orange')
    plt.title(f'Cumulative Patents for CPC Code {selected_cpc}')
    plt.xlabel('Year')
    plt.ylabel('Cumulative Patents')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    st.write(f"Logistic Model Parameters for CPC Code {selected_cpc}:")
    st.write(f"Carrying Capacity (K): {K}")
    st.write(f"Initial Population (P0): {P0}")
    st.write(f"Growth Rate (r): {r}")

# Logistic growth model
def logistic_growth(t, K, P0, r):
    return K / (1 + ((K - P0) / P0) * np.exp(-r * t))

# Main function to run the Streamlit app
def main():
    st.title("Analyze CPC Codes from UAV Patent Search")

    # Initialize Elasticsearch
    es = initialize_elasticsearch()

    # Input: User enters search query
    search_query = st.text_input("Enter your UAV search query")

    if st.button("Analyze CPC Codes"):
        if es and search_query:
            # Perform the search and filter results
            results = search_and_filter(es, search_query)

            # Find and display most used CPC codes
            st.subheader("Top 5 Most Used CPC Codes")
            most_common_cpc_codes = find_most_used_cpc_codes(results)
            if most_common_cpc_codes:
                df = pd.DataFrame(most_common_cpc_codes, columns=["CPC Code", "Count"])

                # Display as a table
                st.dataframe(df)

                # Plot the results
                plt.figure(figsize=(8, 5))
                plt.bar(df["CPC Code"], df["Count"], color="skyblue")
                plt.title("Top 5 Most Used CPC Codes")
                plt.xlabel("CPC Code")
                plt.ylabel("Count")
                plt.xticks(rotation=45, ha='right')
                plt.grid(axis="y")
                st.pyplot(plt)

                # Plot S-curves for each CPC code
                for cpc_code, _ in most_common_cpc_codes:
                    yearly_data = get_yearly_data(es, cpc_code)
                    if yearly_data:
                        df_yearly = (
                            pd.DataFrame(yearly_data, columns=["Year"])
                            .groupby("Year")
                            .size()
                            .reset_index(name="Count")
                            .sort_values("Year")
                            .set_index("Year")
                        )
                        df_yearly["Cumulative"] = df_yearly["Count"].cumsum()

                        st.subheader(f"S-Curve for CPC Code {cpc_code}")
                        plot_s_curve(df_yearly, cpc_code)
            else:
                st.write("No CPC codes found in the results.")

if __name__ == "__main__":
    main()
