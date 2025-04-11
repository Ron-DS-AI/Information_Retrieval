import streamlit as st
import pandas as pd
import json
from datetime import date
from st_aggrid import AgGrid, GridOptionsBuilder
import zipfile
import os
from io import BytesIO
import glob

# @st.cache_data
# def load_mock_data():
#     with open("mock_documents.json") as f:
#         return pd.DataFrame(json.load(f)).assign(
#             date=lambda df: pd.to_datetime(df['date']),
#             content_preview=lambda df: df['content'].str[:200] + "..."
#         )

@st.cache_data(show_spinner=True, ttl=3600)
def load_corpus():
    # Configure paths and file patterns
    data_dir = "corpus_data"
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    zip_files = glob.glob(os.path.join(data_dir, "*.zip"))
    
    # Define efficient data types to reduce memory usage
    dtype_mapping = {
        'title': 'category',
        'content': 'string',
        'category': 'category',
        'date': 'datetime64[ns]'
    }
    
    metadata_dfs = []
    embeddings_dfs = []
    
    # Process CSV files
    
    metadata_df = pd.concat([pd.read_csv(csv_file, dtype=dtype_mapping, parse_dates=['date']) for csv_file in csv_files])
    
    # Process ZIP files containing CSVs
    for zip_file in zip_files:
        with zipfile.ZipFile(zip_file) as z:
            for file_info in z.infolist():
                if file_info.filename.endswith('.csv'):
                    with z.open(file_info) as f:
                        embeddings_df = pd.read_csv(f, dtype=dtype_mapping, parse_dates=['date'])
                        embeddings_dfs.append(embeddings_df)

    embeddings_df = pd.concat(embeddings_dfs, ignore_index=True)
    
    # Combine and optimize final dataframe
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Convert to more efficient types
    combined_df = combined_df.astype(dtype_mapping)
    
    return combined_df

def main():


    
    # Add image at the top
    st.image("logo.png", 
             use_container_width=True,
             caption="Document Search Engine")  

    # st.title("🔍 Document Search Engine")

    # df = load_mock_data()
    min_date = df['publish_time'].min().date()
    max_date = df['publish_time'].max().date()
    categories = sorted(df['category'].unique())

    # Custom CSS for styling
    st.markdown("""
    <style>
    .help-box {
        border: 1px solid #4A4A4A;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #000000;
        color: #FFFFFF;
    }
    .help-box h3 {
        color: #00FF00 !important;
        margin-top: 0;
    }
    .stMultiSelect [data-baseweb="select"] span{
        max-height: 150px;
        overflow-y: auto;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'search_performed' not in st.session_state:
        st.session_state.search_performed = False

    if 'selected_categories' not in st.session_state:
        st.session_state.selected_categories = []

    # Sidebar controls
    with st.sidebar:
        st.header("Search Parameters")
        query = st.text_input("Enter search query:", 
                            help="Type keywords or * to show all documents")
        
        # Date range picker
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "From date:",
                value=min_date,
                min_value=min_date,
                max_value=max_date
            )
        with col2:
            end_date = st.date_input(
                "To date:",
                value=max_date,
                min_value=min_date,
                max_value=max_date
            )
        
        # Enhanced category selector with checkboxes
        selected_categories = st.multiselect(
            "Select categories:",
            options=categories,
            default=st.session_state.selected_categories,
            format_func=lambda x: f"✓ {x}" if x in st.session_state.selected_categories else x,
            placeholder="Choose categories..."
        )
        st.session_state.selected_categories = selected_categories
        
        search_btn = st.button("Search", type="primary")

    # Handle search logic
    if search_btn or query:
        st.session_state.search_performed = True
        show_all = query.strip() in ("", "*")
        
        # Filter data
        filtered = df[
            (df['date'].dt.date >= start_date) & 
            (df['date'].dt.date <= end_date)
        ]
        if st.session_state.selected_categories:
            filtered = filtered[filtered['category'].isin(st.session_state.selected_categories)]
        
        if not filtered.empty:
            st.subheader(f"Found {len(filtered)} documents ({start_date} to {end_date})")
            
            # Configure AgGrid
            gb = GridOptionsBuilder.from_dataframe(filtered)
            gb.configure_pagination(paginationPageSize=10)
            gb.configure_side_bar()
            gb.configure_default_column(
                sortable=True, 
                filterable=True,
                resizable=True
            )
            gb.configure_column("date", type=["dateColumnFilter","customDateTimeFormat"], 
                              custom_format_string='yyyy-MM-dd')
            
            AgGrid(
                filtered[['title', 'category', 'date', 'content_preview']],
                gridOptions=gb.build(),
                height=400,
                theme='streamlit'
            )
        else:
            st.warning("No documents match your search criteria")

    # Initial help message
    if not st.session_state.search_performed:
        st.markdown(f"""
        <div class="help-box">
            <h3>📘 How to Search:</h3>
            <ul>
                <li>Enter <strong>*</strong> to show all documents (currently {len(df)} in total)</li>
                <li>Use date filters to focus on specific time periods</li>
                <li>Select multiple categories using the dropdown checkboxes</li>
                <li>Clear search box and click Search to reset filters</li>
            </ul>
            <p>Available categories: {', '.join(categories)}</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()