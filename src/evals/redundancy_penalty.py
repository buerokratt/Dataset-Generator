from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer, util
import re
import numpy as np
import itertools
import os
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import sys
from utils import Config
import json
from constants import TURNS
# remove the default stderr handler
logger.remove()
# add stout handler
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


config = Config()

# Multilingual model for Estonian
model = SentenceTransformer(config.get("redundancy_penalty.embedding_model"))

def clean_utterances(text: str) -> List[str]:

    turns = re.split(TURNS, text.strip(), flags=re.IGNORECASE)
    return [t.strip() for t in turns if len(t.strip()) > 10]

def compute_intra_conversation_redundancy(conversation: str) -> float:

    turns = clean_utterances(conversation)
    if len(turns) < 2:
        return 0.0  
    
    embeddings = model.encode(turns, convert_to_tensor=True)
    
    similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings)
    n = len(turns)
    
    redundant_pairs = 0
    total_pairs = 0
    
    for i, j in itertools.combinations(range(n), 2):
        if similarity_matrix[i][j].item() >= config.get("redundancy_penalty.intra_conversation_similarity_threshold"):
            redundant_pairs += 1
        total_pairs += 1
    
    redundancy_ratio = redundant_pairs / total_pairs if total_pairs > 0 else 0
    return redundancy_ratio 

def compute_inter_conversation_redundancy(conversations) -> Dict:

    if len(conversations) < 2:
        return {
            "redundancy_score": 0.0,
            "redundant_pairs": [],
            "similarity_matrix": np.array([[0.0]])
        }
    

    conversation_embeddings = []
    
    for conv in conversations:
        turns = clean_utterances(conv)
        if not turns:
            conversation_embeddings.append(np.zeros(384)) 
            continue
            
        turn_embeddings = model.encode(turns)
        avg_embedding = np.mean(turn_embeddings, axis=0)
        conversation_embeddings.append(avg_embedding)
    
    conversation_embeddings = np.array(conversation_embeddings)
    
    n = len(conversations)
    similarity_matrix = np.zeros((n, n))
    
    redundant_pairs = []
    for i, j in itertools.combinations(range(n), 2):
        sim = util.cos_sim(
            conversation_embeddings[i], 
            conversation_embeddings[j]
        ).item()
        
        similarity_matrix[i, j] = sim
        similarity_matrix[j, i] = sim  # Symmetric
        
        if sim >= config.get("redundancy_penalty.inter_conversation_similarity_threshold"):
            redundant_pairs.append((i, j, sim))
    
    total_pairs = (n * (n - 1)) // 2
    redundancy_score = len(redundant_pairs) / total_pairs if total_pairs > 0 else 0.0
    
    return {
        "redundancy_score": redundancy_score,
        "redundant_pairs": redundant_pairs,
        "similarity_matrix": similarity_matrix
    }

def get_pairwise_comparison(conversation_files: List[str]):

    conversations = []
    filenames = []
    
    for file_path in conversation_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                conversations.append(f.read())
                filenames.append(os.path.basename(file_path))
        except Exception as e:
            logger.info(f"Error reading {file_path}: {e}")
    
    redundancy_info = compute_inter_conversation_redundancy(
        conversations
    )
    
    named_redundant_pairs = [
        {
            "file1": filenames[i],
            "file2": filenames[j],
            "similarity": sim,
            "index1": i,
            "index2": j
        }
        for i, j, sim in redundancy_info["redundant_pairs"]
    ]
    
    named_redundant_pairs.sort(key=lambda x: x["similarity"], reverse=True)
    
    redundancy_info["filenames"] = filenames
    redundancy_info["named_redundant_pairs"] = named_redundant_pairs
    
    return redundancy_info

def plot_similarity_heatmap(redundancy_info: Dict, 
                           output_file: str = "conversation_similarity.png") -> None:
    """
    plot a heatmap of conversation similarities.

    """
    similarity_matrix = redundancy_info["similarity_matrix"]
    filenames = redundancy_info["filenames"]
    
    mask = np.zeros_like(similarity_matrix, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    
    plt.figure(figsize=(12, 10))
    
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(
        similarity_matrix,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=0.0,
        center=0.5,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .5},
        xticklabels=[f.split('.')[0] for f in filenames],
        yticklabels=[f.split('.')[0] for f in filenames]
    )
    
    plt.title('Conversation Similarity Matrix')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def generate_redundancy_report(topic_dir: str, 
                              output_file: str = "redundancy_report.json") -> None:

    conversation_files = glob(os.path.join(topic_dir, "conversation_*.txt"))
    
    if not conversation_files:
        logger.info(f"No conversation files found in {topic_dir}")
        return
    
    redundancy_info = get_pairwise_comparison(conversation_files)
    
    plot_similarity_heatmap(redundancy_info, 
                           os.path.join(os.path.dirname(output_file), "similarity_heatmap.png"))
    results = {}
        
    results["topic_dir"] = topic_dir
    results["num_conversations"] = len(conversation_files)
    results["redundancy_score"] = redundancy_info['redundancy_score']
    results["similarity_threshold"] = config.get("redundancy_penalty.inter_conversation_similarity_threshold")
    results["num_redundant_pairs"] = len(redundancy_info['redundant_pairs'])
        
    if redundancy_info['redundant_pairs']:
        results["redundant_pairs"] = redundancy_info['redundant_pairs']
      
            
    results["intra_redundancy"] = []
    for file_path in conversation_files:
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f_conv:
                conversation = f_conv.read()
                
            intra_redundancy = compute_intra_conversation_redundancy(conversation)
                
            results["intra_redundancy"].append({"file_path":file_path, "intra_redundancy": intra_redundancy})
        except Exception as e:
            results["intra_redundancy"].append({"file_path":file_path, "intra_redundancy": None})
    #   Save results to json 
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    logger.info(f"Redundancy report generated at {output_file}")

