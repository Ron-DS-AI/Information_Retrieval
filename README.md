# Information_Retrieval
* Information retrieval assignment - Group 7
* Date: 14/04/25
* Created by: Stephanie Edwards, Munir Jahangir, Neelam Ahmed and Ronald Almeida

## **TREC COVID2 (rnd3) Dataset**
The original dataset was downloaded from here: [https://www.kaggle.com/competitions/trec-covid-information-retrieval/data?select=docids-rnd3.txt ](https://www.kaggle.com/competitions/trec-covid-information-retrieval/data) 

Files included:
- metadata.csv
- topics-rnd3.csv
- docids-rnd3.txt
- qrels.csv

## **Preprocessing**
Step 1: Cleaning & Abstractive summary creation 
  1. [clean_and_prepoc_MJ.ipynb](https://github.com/Ron-DS-AI/Information_Retrieval/blob/main/Preprocessing/clean_and_prepoc_MJ.ipynb)
  2. [BertSUM-summarised_abstracts.ipynb](https://github.com/Ron-DS-AI/Information_Retrieval/blob/main/Preprocessing/BertSUM-summarised_abstracts.ipynb)

Step 2: Search purpose tag category creation:

  3. [Search_Purpose_pt1_SE.ipynb](https://github.com/Ron-DS-AI/Information_Retrieval/blob/main/SearchPurposeCategories/Search_Purpose_pt1_SE.ipynb) *(initial keyword matching to create labelled dataset)* 
  4. [CategoryTraining - SE.ipynb](https://github.com/Ron-DS-AI/Information_Retrieval/blob/main/SearchPurposeCategories/CategoryTraining%20-%20SE.ipynb) *(BERT classification model training on labelled dataset to create predictions on unlabelled data)* 
  5. [Search_Purpose_pt2_SE.ipynb](https://github.com/Ron-DS-AI/Information_Retrieval/blob/main/SearchPurposeCategories/Search_Purpose_pt2_SE.ipynb) *(Combining output from keyword matching and predicted categories into single dataset )* 

**FINAL PREPROCESSING OUTPUT FILES**: 
 * [preprocessed_metadata_with_summaries.csv](https://github.com/Ron-DS-AI/Information_Retrieval/blob/0fd7d6d8139ec75283087fed63419376812a68bc/Preprocessing/preprocessed_metadata_with_summaries.csv)
* [queries_rnd3_stacked.csv](https://github.com/Ron-DS-AI/Information_Retrieval/blob/0fd7d6d8139ec75283087fed63419376812a68bc/qrels/queries_rnd3_stacked.csv) (restructured to utilise all 3 variations of query topics)


## **SBERT Model Training**
6. [Fine_tuned_SBERT_hard_neg_mining_final.ipynb](https://github.com/Ron-DS-AI/Information_Retrieval/blob/95a2a13e7a47ce2e99f36fbff877bfa5d70c2f6c/Fine_tuned_SBERT_hard_neg_mining_final.ipynb) (download to view) 

**OUTPUT**:
* SBERT trained model, uploaded to huggingface: https://huggingface.co/StephKeddy/sbert-IR-covid-search-v2
* [DEMO_test_queries.csv](https://github.com/Ron-DS-AI/Information_Retrieval/blob/main/DEMO_test_queries.csv)(excludes queries used for training the model)
* [DEMO_test_qrels.csv](https://github.com/Ron-DS-AI/Information_Retrieval/blob/main/DEMO_test_qrels.csv) (qrels for the above demo test queries)

## **Indexing & Retrieval (testing)** 

7. [Indexing_and_retrieval.ipynb](https://github.com/Ron-DS-AI/Information_Retrieval/blob/main/Indexing_and_retrieval.ipynb) 

**OUTPUT**: 
* [embeddings_part1.csv.gz](https://github.com/Ron-DS-AI/Information_Retrieval/blob/main/full_corpus_SBERT_trained/embeddings_part1.csv.gz)
* [embeddings_part2.csv.gz](https://github.com/Ron-DS-AI/Information_Retrieval/blob/main/full_corpus_SBERT_trained/embeddings_part2.csv.gz)
* [metadata_part1_final.csv](https://github.com/Ron-DS-AI/Information_Retrieval/blob/main/full_corpus_SBERT_trained/metadata_part1_final.csv)
* [metadata_part2_final.csv](https://github.com/Ron-DS-AI/Information_Retrieval/blob/main/full_corpus_SBERT_trained/metadata_part2_final.csv)

## **Streamlit UI**
