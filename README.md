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
Step 1: Cleaning & Abstractive summary creation [Preprocessing Directory](https://github.com/Ron-DS-AI/Information_Retrieval/tree/0fd7d6d8139ec75283087fed63419376812a68bc/Preprocessing)
  1. **clean_and_prepoc_MJ.ipynb**
  2. **BertSUM-summarised_abstracts.ipynb**

Step 2: Search purpose tag category creation: [Search Purpose Categories Directory](https://github.com/Ron-DS-AI/Information_Retrieval/tree/0fd7d6d8139ec75283087fed63419376812a68bc/SearchPurposeCategories) 

  3. **Search_Purpose_pt1_SE.ipynb** *(initial keyword matching to create labelled dataset)* 
  4. **CategoryTraining - SE.ipynb** *(BERT classification model training on labelled dataset to create predictions on unlabelled data)* 
  5. **Search_Purpose_pt2_SE.ipynb** *(Combining output from keyword matching and predicted categories into single dataset )* 

**FINAL PREPROCESSING OUTPUT FILES**: 
 * **preprocessed_metadata_with_summaries.csv**
(https://github.com/Ron-DS-AI/Information_Retrieval/blob/0fd7d6d8139ec75283087fed63419376812a68bc/Preprocessing/preprocessed_metadata_with_summaries.csv)
* **queries_rnd3_stacked.csv** (restructured to utilise all 3 variations of query topics) (https://github.com/Ron-DS-AI/Information_Retrieval/blob/0fd7d6d8139ec75283087fed63419376812a68bc/qrels/queries_rnd3_stacked.csv) 


## **SBERT Model Training**
6. **Fine_tuned_SBERT_hard_neg_mining_final.ipynb** (download to view) (https://github.com/Ron-DS-AI/Information_Retrieval/blob/0fd7d6d8139ec75283087fed63419376812a68bc/Fine_tuned_SBERT_hard_neg_mining_final.ipynb) 

**OUTPUT**:
* SBERT trained model, uploaded to huggingface: https://huggingface.co/StephKeddy/sbert-IR-covid-search-v2
* **DEMO_test_queries.csv** (excludes queries used for training the model): https://github.com/Ron-DS-AI/Information_Retrieval/blob/0fd7d6d8139ec75283087fed63419376812a68bc/DEMO_test_queries.csv
* **DEMO_test_qrels.csv** (qrels for the above demo test queries)

## **Indexing & Retrieval (testing)** 

7. **Indexing_and_retreival.ipynb** (https://github.com/Ron-DS-AI/Information_Retrieval/blob/0fd7d6d8139ec75283087fed63419376812a68bc/Indexing_and_retrieval.ipynb)

**OUTPUT**: https://github.com/Ron-DS-AI/Information_Retrieval/tree/0fd7d6d8139ec75283087fed63419376812a68bc/full_corpus_SBERT_trained 
* embeddings_part1.csv.gz
* embeddings_part2.csv.gz
* metadata_part1_final.csv
* metadata_part2_final.csv

## **Streamlit UI**
