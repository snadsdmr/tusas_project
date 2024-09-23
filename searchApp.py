import streamlit as st
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

# Initialize Elasticsearch connection
def initialize_elasticsearch():
    try:
        es = Elasticsearch(
            "http://localhost:9200",
            basic_auth=("ELASTIC_USERNAME", "ELASTIC_PASSWORD"),
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

# Function to perform the search for UAV patents
def search(es, input_keyword):
    model = SentenceTransformer('all-mpnet-base-v2')
    vector_of_input_keyword = model.encode(input_keyword)

    query = {
        "field": "DescriptionVector",
        "query_vector": vector_of_input_keyword,
        "k": 6,
        "num_candidates": 10000
    }
    
    try:
        res = es.knn_search(index="patents_uav", knn=query, source=["Title (Translated)(English)", "Abstract (Translated)(English)"])
        results = res["hits"]["hits"]
        return results
    except Exception as e:
        st.error(f"Error during search: {e}")
        return []

# Main function to run the Streamlit app
def main():
    st.title("Search UAV Patents")

    # Initialize Elasticsearch
    es = initialize_elasticsearch()
    
    # Input: User enters search query
    search_query = st.text_input("Enter your UAV search query")

    # Button: User triggers the search
    if st.button("Search"):
        if es and search_query:
            # Perform the search and get results
            results = search(es, search_query)

            # Display search results
            st.subheader("Search Results")
            if results:
                for result in results:
                    with st.container():
                        if '_source' in result:
                            try:
                                st.header(f"{result['_source']['Title (Translated)(English)']}")
                                st.write(f"Abstract: {result['_source']['Abstract (Translated)(English)']}")
                            except Exception as e:
                                st.error(f"Error displaying result: {e}")
                        st.divider()
            else:
                st.write("No results found.")

if __name__ == "__main__":
    main()
