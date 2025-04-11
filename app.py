# all-mpnet-base-v2


import streamlit as st
import pandas as pd
import numpy as np
import ast
import os
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
CORPUS_PATH = "corpus_data"
EMBEDDING_FILES = [
    "embeddings_final_pt1_mj.csv.gz",
    "embeddings_final_pt2_mj.csv.gz"
]
METADATA_FILES = [
    "metadata_final_pt1.csv",
    "metadata_final_pt2.csv"
]

@st.cache_resource
def load_model():
    return SentenceTransformer('all-mpnet-base-v2')

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

def main():
    st.set_page_config(layout="wide", page_title="Document Search")
    
    # Load data and model
    with st.spinner("Loading corpus and model..."):
        corpus, all_tags, tag_freq = load_corpus()
        model = load_model()
    
    # Sidebar controls
    with st.sidebar:
        st.header("Search Parameters")
        query = st.text_input("Enter your search query:")
        top_k = st.slider("Number of results", 10, 100, 50)
        
        # Date range filter
        min_date = corpus['publish_time'].min().date()
        max_date = corpus['publish_time'].max().date()
        start_date, end_date = st.date_input(
            "Publication Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Tag filtering with vertical layout
        st.subheader("Filter by Tags")
        tag_search = st.text_input("Search tags:", "").lower()
        filtered_tags = [t for t in all_tags if tag_search in t.lower()]
        
        # Create a scrollable container for tags
        tags_container = st.container()
        with tags_container:
            # Vertical checkboxes with wrapping
            selected_tags = []
            for tag in filtered_tags:
                if st.checkbox(
                    f"{tag} ({tag_freq[tag]})",
                    key=f"tag_{tag}",
                    help=f"Show documents tagged with {tag}"
                ):
                    selected_tags.append(tag)

        # Add CSS for wrapping long tag names
        st.markdown("""
        <style>
        div[data-testid="stVerticalBlock"] {
            max-height: 1000px;  /* Increased from 400px */
            overflow-y: auto;
            padding-right: 10px;
            margin-bottom: 2rem;
        }
        </style>
        """, unsafe_allow_html=True)

    # Main interface
    st.title("Research Document Search")
    
    if query:
        with st.spinner(f"Searching for '{query}'..."):
            raw_results, scores = search(query, corpus, model, top_k)
            
            # Apply date filter
            date_filtered = raw_results[
                (raw_results['publish_time'].dt.date >= start_date) &
                (raw_results['publish_time'].dt.date <= end_date)
            ]
            
            # Apply tag filter
            if selected_tags:
                results = date_filtered[
                    date_filtered['tags'].apply(
                        lambda x: any(tag in x for tag in selected_tags)
                    )
                ]
            else:
                results = date_filtered
            
            if results.empty:
                st.warning("No documents match your search criteria")
                return
            
            # Prepare display dataframe
            display_df = results.reset_index()[[
                'cord_uid', 'title', 'publish_time', 'abstract', 'referenced_by_count', 'tags'
            ]]
            display_df['similarity'] = np.round(scores[:len(results)], 3)
            display_df['publish_time'] = display_df['publish_time'].dt.strftime('%d %b %Y')
            display_df['tags'] = display_df['tags'].apply(lambda x: ', '.join(x))
            
            # Display results
            st.subheader(f"Top {len(results)} Results ({start_date} to {end_date})")
            
            st.dataframe(
                display_df,
                column_config={
                    "cord_uid": "Document ID",
                    "title": "Title",
                    "publish_time": "Published",
                    "abstract": st.column_config.TextColumn(
                        "Abstract",
                        width="large"
                    ),
                    "tags": "Categories",
                    "referenced_by_count": "Citations",
                    "similarity": st.column_config.NumberColumn(
                        "Relevance",
                        format="%.3f"
                    )
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Download button
            csv = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Results",
                data=csv,
                file_name="search_results.csv",
                mime="text/csv"
            )
    else:
        st.info("Enter a search query to begin")

if __name__ == "__main__":
    main()