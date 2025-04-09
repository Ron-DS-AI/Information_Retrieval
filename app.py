import streamlit as st
import json
from datetime import date

# Load mock data
with open("mock_documents.json") as f:
    mock_docs = json.load(f)

# Mock search function
def mock_search(query, category_filter=None, date_filter=None):
    results = []
    for doc in mock_docs:
        # Simple text matching (replace with your logic)
        if query.lower() in doc["content"].lower():
            if category_filter and doc["category"] not in category_filter:
                continue
            if date_filter and doc["date"] < str(date_filter):
                continue
            results.append(doc)
    return results

# UI Components
st.title("Search Engine Prototype")

# 1. Search Box
query = st.text_input("Enter your search query")

# 2. Filters
col1, col2 = st.columns(2)
with col1:
    date_filter = st.date_input("Filter by date", value=date(2024, 1, 1))
with col2:
    categories = list(set(doc["category"] for doc in mock_docs))
    category_filter = st.multiselect("Filter by category", categories)

# 3. Search Action
if st.button("Search"):
    results = mock_search(query, category_filter, date_filter)
    
    # Display results
    st.subheader(f"Found {len(results)} results")
    for doc in results:
        with st.expander(doc["title"]):
            st.caption(f"Category: {doc['category']} | Date: {doc['date']}")
            st.write(doc["content"][:200] + "...")