![contrast](https://user-images.githubusercontent.com/40825205/188257392-ecb6965a-57c7-42bf-8fa1-f533fb403fb1.png)


# Abstract
In recent years, open-domain question answering (OpenQA) draws significant research attention due to its wide range of applications. However, one concern for the existing OpenQA system is its slow inference speed for being used in practical scenarios. Aiming at this concern, the idea of indexing possible asked questions (referred to as PAQ) is proposed. The idea is two-fold: (1) pre-generating and indexing all possible asked questions and the corresponding answers and (2) processing on-line queries by retrieving similar QA pairs through index structures. In this paper, we extend the PAQ idea by considering \textit{QA pair merging} to boost the accuracy of the PAQ retriever. We find that embedding merged questions (rather than a single question) leads to better representation that improves the accuracy of the QA pair retriever. Along with the question merging idea, we propose MPAQ Retriever, a retriever for merged QA pairs, based on \textit{self-filtering} and \textit{self-supervised contrastive learning} techniques. The performance evaluation demonstrates an improvement of 7.1\% (1\%) on TriviaQA (Natural Question).

for the original data of PAQ, please refer to https://github.com/facebookresearch/PAQ
## building index of MPAQ
for more indexing information , please refer to https://github.com/facebookresearch/PAQ
### embed
    python -m paq.retrievers.embed     --model_name_or_path ./data/models/retrievers/retriever_multi_base_256     --qas_to_embed nsc385.jsonl  --output_dir XXX/my_vectors_distributed --fp16     --batch_size 128     --verbose     --memory_friendly_parsing     --n_jobs 18     --slurm_partition my_clusters_partition     --slurm_comment "my embedding job"
    
### build index
    python -m paq.retrievers.build_index     --embeddings_dir XXX/my_vectors_distributed  --output_path XXX/my_index.hnsw.faiss --hnsw     --SQ8     --store_n 32     --ef_construction 128     --ef_search 128     --verbose
    
### retrieval   
    python -m paq.retrievers.retrieve     --model_name_or_path ./data/models/retrievers/retriever_multi_base_256     --qas_to_answer data/annotated_datasets/NQ-open.test.jsonl      --qas_to_retrieve_from nsc385.jsonl   --top_k 50     --output_file XXX/my_retrieval_results.jsonl --faiss_index_path XXX/my_index.hnsw.faiss     --fp16     --memory_friendly_parsing     --verbose

### eval 
    python -m paq.evaluation.eval_retriever     --predictions XXX/my_retrieval_results.jsonl --references data/annotated_datasets/NQ-open.test.jsonl     --hits_at_k 1,10,50





## retraining
從https://github.com/facebookresearch/DPR  修改

## generate train
透過retrieve nq train&dev 和trivia train&dev來得到其positive & hard negative sample，並透過"generate_train.ipy"來使其產生train & dev data
tips:記得retriever是要用原本PAQ的retriever model:pytorch_model.bin
    python -m paq.retrievers.retrieve     --model_name_or_path ./data/models/retrievers/retriever_multi_base_256     --qas_to_answer data/annotated_datasets/NQ-open.train-train.jsonl      --qas_to_retrieve_from nsc385.jsonl   --top_k 50     --output_file XXX/my_retrieval_results.jsonl --faiss_index_path XXX/my_index.hnsw.faiss     --fp16     --memory_friendly_parsing     --verbose


## training process
    cd DPR-main
    python -m torch.distributed.launch --nproc_per_node=1 train_dense_encoder.py train=biencoder_nq train_datasets=[nq_train] dev_datasets=[nq_dev] train=biencoder_nq output_dir="train"

--nproc_per_node: 最高8 看你幾核心
把你的training&dev&test data overwrite到"downloads/data/retriever"當中的nq_train nq-dev


## transfer model to PAQ model
after training  run transfrom_to_albert.ipynb 
就可以得到對應query和ctx的 model，把它們移到PAQ資料夾中的PAQ-main/data/models/retrievers/retriever_multi_base_256/底下
並修改"PAQ-main/paq/retrievers/retriever_utils.py"中有bin檔的部分(pytorch_model.bin就是原始的PAQ retriever model)(PAQ中query 跟 ctx encoder是一樣的)
就可以拿來做embed,indexing,retrieval


# hybrid retrieval:
從https://github.com/castorini/pyserini  修改

## build BM25 index

run  PAQ-main/transfrom_to_BM25.ipynb to transform to bm25 format

first download javac        
        sudo apt-get update
        sudo apt-get install openjdk-11-jdk
build bm25 index
        –collection JsonCollection --input bm25/ --index bm25/2_5_bm25 --generator DefaultLuceneDocumentGenerator --threads 1 --storeDocvectors
        
## hybrid retrieval
run BM.ipynb
即可得到混合排名後的retrieved result，重新 eval retrieved result 一次即可得到最終結果。


---------
global search:
PAQ/simple_paq_search.ipynb

  

