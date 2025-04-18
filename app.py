import streamlit as st
import pandas as pd
import numpy as np
import ast
import os
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from datetime import date

# Configuration
CORPUS_PATH = "full_corpus_SBERT_trained"
EMBEDDING_FILES = [
    "embeddings_part1.csv.gz",
    "embeddings_part2.csv.gz"
]
METADATA_FILES = [
    "metadata_part1_final.csv",
    "metadata_part2_final.csv"
]

@st.cache_resource
def load_model():
    return SentenceTransformer('StephKeddy/sbert-IR-covid-search-v2')    

@st.cache_data
def load_corpus():
    # Load metadata with tag processing
    meta_dfs = []
    tag_frequency = defaultdict(int)
    
    for f in METADATA_FILES:
        path = os.path.join(CORPUS_PATH, f)
        df = pd.read_csv(path, index_col='cord_uid')
        
        # Convert string lists to actual lists
        df['tags'] = df['tags'].apply(
            lambda x: ast.literal_eval(x) if pd.notnull(x) and x.startswith('[') else []
        )
        
        # Process dates
        df['publish_time'] = pd.to_datetime(
            df['publish_time'],
            format='mixed',
            dayfirst=True
        )
        
        # Count tag frequencies
        for tags in df['tags']:
            for tag in tags:
                tag_frequency[tag] += 1
                
        meta_dfs.append(df)
    
    metadata = pd.concat(meta_dfs)
    
    # Load embeddings
    embed_dfs = []
    for f in EMBEDDING_FILES:
        path = os.path.join(CORPUS_PATH, f)
        df = pd.read_csv(path, compression='gzip', index_col='cord_uid')
        df.columns = df.columns.astype(int)
        df = df.apply(pd.to_numeric, errors='coerce')
        embed_dfs.append(df)
    
    embeddings = pd.concat(embed_dfs)
    
    # Validate embeddings
    if embeddings.shape[1] != 768:
        st.error(f"Expected 768 embedding dimensions, found {embeddings.shape[1]}")
        st.stop()
    
    # Merge data
    corpus = metadata.merge(embeddings, left_index=True, right_index=True, how='inner')
    
    # Sort tags by frequency then alphabetically
    sorted_tags = sorted(
        tag_frequency.keys(),
        key=lambda x: (-tag_frequency[x], x)
    )
    
    return corpus, sorted_tags, tag_frequency

def search(query, corpus, model, top_k=50):
    query_embedding = model.encode(query, convert_to_tensor=False).reshape(1, -1)
    doc_embeddings = corpus[list(range(768))].values
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return corpus.iloc[top_indices], similarities[top_indices]

def preprocess_text(text):
    if isinstance(text, str):
        return ' '.join(text.lower().split())
    return text

def recalibrate_score(score, low=0.999, high=1.0):
    """Rescales cosine similarity to a 0-100 relevance score."""
    score = np.clip(score, low, high)
    return round(100 * (score - low) / (high - low), 1)

def main():
    st.set_page_config(layout="wide", page_title="Document Search", page_icon="🔍")
    
    # Logo and Header
    st.image("shuttle_logo_cropped.png", width=300)
    st.title("Research Document Search Engine")
    
    # Load data and model
    with st.spinner("Loading corpus and model..."):
        corpus, all_tags, tag_freq = load_corpus()
        model = load_model()
    
    # Initialize session state for query
    if 'query' not in st.session_state:
        st.session_state.query = ""
    
    # Sidebar controls
    with st.sidebar:
        st.header("Search Parameters")
        sidebar_query = st.text_input("Enter your search query:", key="sidebar_query", value=st.session_state.query)
        # Update session state when sidebar input changes
        st.session_state.query = sidebar_query
        
        top_k = st.slider("Number of results to display", 10, 100, 50)
        # Number input for minimum referenced_by_count
        max_refs = int(corpus['referenced_by_count'].max())
        min_refs = st.number_input(
            "Minimum cited by",
            min_value=0,
            max_value=max_refs,
            value=0,
            step=1,
            help=f"Enter a number between 0 and {max_refs} to filter documents by minimum citations."
        )
        
        # Date range filter with validation
        min_date = corpus['publish_time'].min().date()
        max_date = corpus['publish_time'].max().date()
        
        try:
            date_selection = st.date_input(
                "Publication Date Range (select two dates)",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
        except ValueError as e:
            st.error("Please select both start and end dates")
            st.stop()
        
        # Handle date range validation
        if isinstance(date_selection, tuple) and len(date_selection) == 2:
            start_date, end_date = date_selection
        elif isinstance(date_selection, date):
            st.warning("Showing results for single selected date")
            start_date = end_date = date_selection
        else:
            st.error("Invalid date selection. Please choose a start and end date.")
            st.stop()
        
        # Tag filtering
        st.subheader("Filter by Tags")
        tag_search = st.text_input("Search tags:", "").lower()
        filtered_tags = [t for t in all_tags if tag_search in t.lower()]
        
        tags_container = st.container()
        with tags_container:
            selected_tags = []
            for tag in filtered_tags:
                if st.checkbox(
                    f"{tag} ({tag_freq[tag]})",
                    key=f"tag_{tag}"
                ):
                    selected_tags.append(tag)

        # CSS for tag display
        st.markdown("""
        <style>
        div[role="checkbox"] label {
            white-space: normal;
            word-wrap: break-word;
            margin: 8px 0;
            line-height: 1.4;
        }
        </style>
        """, unsafe_allow_html=True)
        
    # Main interface
    query = st.session_state.query
    if query:
        # Preprocess query unless it's '*'
        processed_query = query if query.strip() == '*' else preprocess_text(query)
        with st.spinner(f"Searching for '{query}'..."):
            if query.strip() == '*':
                # Special case: '*' returns all documents
                raw_results = corpus
                scores = np.zeros(len(corpus))  # Placeholder scores
                st.info("Showing all documents matching filters (date, tags, references).")
            else:
                # Regular semantic search with preprocessed query
                raw_results, scores = search(processed_query, corpus, model, top_k=1000)
            
            # Apply date filter
            date_filtered = raw_results[
                (raw_results['publish_time'].dt.date >= start_date) &
                (raw_results['publish_time'].dt.date <= end_date)
            ]
            
            # Apply tag filter
            if selected_tags:
                tag_filtered = date_filtered[
                    date_filtered['tags'].apply(
                        lambda x: any(tag in x for tag in selected_tags)
                    )
                ]
            else:
                tag_filtered = date_filtered
            
            # Apply referenced_by_count filter
            results = tag_filtered[
                tag_filtered['referenced_by_count'] >= min_refs
            ]
            
            # Trim to top_k results after filtering
            results = results.iloc[:top_k]
            scores = scores[:len(results)]
            
            if results.empty:
                st.warning("No documents match your search criteria. Try adjusting the filters.")
                return
            
            # Prepare display dataframe
            display_df = results.reset_index()[['title', 'publish_time', 'abstract', 'referenced_by_count', 'url']]
            if query.strip() == '*':
                display_df['similarity'] = None  # No similarity for '*' query
                display_df['Rank'] = range(1, len(display_df) + 1)  # Sequential rank
            else:
                # Recalibrate scores for display
                display_df['similarity'] = np.array([recalibrate_score(score) for score in scores])
                # Rank based on original scores to preserve granularity
                display_df['Rank'] = pd.Series(scores).rank(ascending=False, method='min').astype(int)
            display_df['publish_time'] = display_df['publish_time'].dt.strftime('%d %b %Y')
            
            # Reorder columns to show Rank first
            display_df = display_df[['Rank', 'title', 'publish_time', 'abstract', 'referenced_by_count', 'url', 'similarity']]
            
            # Display results
            st.subheader(f"Showing {len(results)} Results ({start_date} to {end_date})")
            if len(results) < top_k:
                st.info(f"Only {len(results)} results found after applying filters. Adjust filters to see more.")
            
            st.dataframe(
                display_df,
                column_config={
                    "Rank": st.column_config.NumberColumn(
                        "Rank",
                        width=45
                    ),
                    "title": "Title",
                    "publish_time": st.column_config.DateColumn(
                        "Published",
                        format="DD MMM YYYY",
                    ),
                    "abstract": st.column_config.TextColumn(
                        "Abstract",
                        width="large"
                    ),
                    "referenced_by_count": st.column_config.NumberColumn(
                        "# Cited By",
                        format="%d"
                    ),
                    "url": st.column_config.LinkColumn(
                        "URL",
                        display_text="Link"
                    ),
                    "similarity": st.column_config.NumberColumn(
                        "Relevance",
                        format="%.1f"
                    )
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Download button
            csv = display_df.to_csv(index=False, na_rep='').encode('utf-8')
            st.download_button(
                "Download Results",
                data=csv,
                file_name="search_results.csv",
                mime="text/csv"
            )
    else:
        # Centered search input
        # st.markdown("<h3 style='text-align: center;'>Search for Research Documents</h3>", unsafe_allow_html=True)
        st.text_input(
            "", 
            key="main_query", 
            placeholder="Enter a search query to begin",
            value=st.session_state.query,
            on_change=lambda: st.session_state.update(query=st.session_state.main_query)
        )

if __name__ == "__main__":
    main()