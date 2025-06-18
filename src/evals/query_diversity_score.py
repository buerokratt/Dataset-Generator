from typing import List, Dict, Tuple, Set, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import re
import ast
from utils import load_stopwords
from loguru import logger
import os
import sys
import glob
from pathlib import Path
import json
from utils import Config
from utils import NumpyEncoder
from utils import preprocess_text
from utils import read_file
from constants import USER_QUERY_PATTERNS


logger.remove()
# add stout handler
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")




class QueryDiversityAnalyzer:
    """
    A class for analyzing the diversity of user queries in a set of conversations.
    Evaluates both lexical diversity (word usage variety) and semantic diversity (intent variety).
    """
    
    def __init__(self):
        """
        Initialize the query diversity analyzer.
        
        Args:
            embedding_model: Name of sentence transformer model to use
            semantic_similarity_threshold: Threshold for considering queries semantically similar
            min_queries_for_analysis: Minimum number of queries needed for meaningful analysis
        """
        self.config = Config()
        self.embedding_model_name = self.config.get("query_diversity.embedding_model")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.semantic_similarity_threshold = self.config.get("query_diversity.semantic_similarity_threshold")
        self.min_queries_for_analysis = self.config.get("query_diversity.min_queries_for_analysis")
        self.stopwords = load_stopwords()
        
    
    def extract_user_queries(self, conversation: str) -> List[str]:
        """
        Extract user queries/questions from a conversation.
        Focuses only on user turns, not system responses.
        
        Args:
            conversation: The conversation text
            
        Returns:
            List of user queries
        """

        
        queries = []
        for pattern in USER_QUERY_PATTERNS:
            matches = re.findall(pattern, conversation, re.DOTALL | re.IGNORECASE)
            queries.extend([preprocess_text(m) for m in matches if m.strip()])
        
        if not queries:
            sentences = re.split(r'(?<=[.!?])\s+', conversation)
            queries = [preprocess_text(s.strip()) for s in sentences if s.strip().endswith('?')]
        
        return queries
    
    def compute_lexical_diversity(self, queries: List[str]) -> Dict:
        """
        Compute lexical diversity metrics for a set of queries.
        Measures variety in vocabulary and phrasing.
        

        """
        if not queries or len(queries) < self.min_queries_for_analysis:
            return {
                "lexical_diversity_score": 0.0,
                "unique_token_ratio": 0.0,
                "total_tokens": 0,
                "unique_tokens": 0,
                "reason": "Too few queries for meaningful analysis"
            }
        
        all_text = " ".join(queries)
        
        try:
            tokens = word_tokenize(all_text)
            
            tokens = [t for t in tokens if t not in self.stopwords and len(t) > 2]
            
            total_tokens = len(tokens)
            unique_tokens = len(set(tokens))
            
            unique_token_ratio = unique_tokens / max(1, total_tokens)
            
            vectorizer = TfidfVectorizer(
                ngram_range=ast.literal_eval(self.config.get("query_diversity.ngram_range")),  
                min_df=self.config.get("query_diversity.min_df"),
                max_df=self.config.get("query_diversity.max_df"),
            )
            
            try:
                # we do need at least two documents for meaningful TF-IDF
                if len(queries) < 2:
                    queries.append("dummy text")  
                
                tfidf_matrix = vectorizer.fit_transform(queries)
                
                # calculate average pairwise similarity 
                similarities = cosine_similarity(tfidf_matrix)
                
                upper_tri = similarities[np.triu_indices_from(similarities, k=1)]
                
                avg_similarity = np.mean(upper_tri) if len(upper_tri) > 0 else 0.0
                
                ngram_diversity = 1.0 - avg_similarity
                
            except Exception as e:
                logger.warning(f"Error calculating n-gram diversity: {e}")
                ngram_diversity = 0.0
            
            # Calculate overall lexical diversity score 
            lexical_diversity_score = (0.7 * unique_token_ratio) + (0.3 * ngram_diversity)
            
            return {
                "lexical_diversity_score": float(lexical_diversity_score),
                "unique_token_ratio": float(unique_token_ratio),
                "ngram_diversity": float(ngram_diversity),
                "total_tokens": total_tokens,
                "unique_tokens": unique_tokens,
                "avg_query_length": total_tokens / len(queries)
            }
            
        except Exception as e:
            logger.error(f"Error computing lexical diversity: {e}")
            return {
                "lexical_diversity_score": 0.0,
                "error": str(e)
            }
    
    def compute_semantic_diversity(self, queries: List[str]) -> Dict:
        """
        Compute semantic diversity metrics for a set of queries.
        Measures variety in intent and meaning.

        """
        if not queries or len(queries) < self.min_queries_for_analysis:
            return {
                "semantic_diversity_score": 0.0,
                "cluster_count": 0,
                "avg_similarity": 0.0,
                "reason": "Too few queries for meaningful analysis"
            }
        
        try:
            embeddings = self.embedding_model.encode(queries)            
            similarities = cosine_similarity(embeddings)
            
            upper_tri = similarities[np.triu_indices_from(similarities, k=1)]
            
            avg_similarity = np.mean(upper_tri) if len(upper_tri) > 0 else 0.0
            
            clusters = []
            query_cluster_map = {}
            
            for i in range(len(queries)):
                if i in query_cluster_map:
                    continue
                    
                cluster = [i]
                query_cluster_map[i] = len(clusters)
                
                for j in range(i + 1, len(queries)):
                    if j not in query_cluster_map and similarities[i, j] >= self.semantic_similarity_threshold:
                        cluster.append(j)
                        query_cluster_map[j] = len(clusters)
                
                clusters.append(cluster)
            
            cluster_count = len(clusters)
            avg_cluster_size = np.mean([len(c) for c in clusters]) if clusters else 0
            largest_cluster_size = max([len(c) for c in clusters]) if clusters else 0
            largest_cluster_ratio = largest_cluster_size / len(queries) if queries else 0

            
            cluster_ratio = cluster_count / len(queries)
            avg_dissimilarity = 1.0 - avg_similarity
            size_balance = 1.0 - largest_cluster_ratio
            
            semantic_diversity_score = (0.4 * cluster_ratio) + (0.4 * avg_dissimilarity) + (0.2 * size_balance)
            
            cluster_examples = []
            for cluster_idx, cluster in enumerate(clusters):
                representative_idx = cluster[0]  
                cluster_examples.append({
                    "cluster_id": cluster_idx,
                    "size": len(cluster),
                    "example_query": queries[representative_idx]
                })
            
            return {
                "semantic_diversity_score": float(semantic_diversity_score),
                "cluster_count": cluster_count,
                "avg_similarity": float(avg_similarity),
                "cluster_ratio": float(cluster_ratio),
                "largest_cluster_ratio": float(largest_cluster_ratio),
                "avg_cluster_size": float(avg_cluster_size),
                "cluster_examples": cluster_examples
            }
            
        except Exception as e:
            logger.error(f"Error computing semantic diversity: {e}")
            return {
                "semantic_diversity_score": 0.0,
                "error": str(e)
            }
    
    def compute_query_diversity(self, conversations: List[str], 
                              topic_label: str = None) -> Dict:
        """
        Compute comprehensive query diversity metrics for conversations.
        
        Args:
            conversations: List of conversation texts
            topic_label: Optional topic label for reporting
            
        Returns:
            Dictionary with diversity metrics
        """
        all_queries = []
        queries_by_conversation = []
        
        for conv in conversations:
            queries = self.extract_user_queries(conv)
            if queries:
                all_queries.extend(queries)
                queries_by_conversation.append(queries)
        
        if not all_queries:
            return {
                "query_diversity_score": 0.0,
                "query_count": 0,
                "topic": topic_label,
                "reason": "No user queries found"
            }
        
        lexical_diversity = self.compute_lexical_diversity(all_queries)        
        semantic_diversity = self.compute_semantic_diversity(all_queries)

        if "lexical_diversity_score" in lexical_diversity and "semantic_diversity_score" in semantic_diversity:
            overall_score = (lexical_diversity["lexical_diversity_score"] + 
                           semantic_diversity["semantic_diversity_score"]) / 2
        else:
            overall_score = 0.0
        
        first_queries = []
        for queries in queries_by_conversation:
            if queries:
                first_queries.append(queries[0])
        
        first_query_semantic = self.compute_semantic_diversity(first_queries) if first_queries else {"semantic_diversity_score": 0.0}
        first_query_lexical = self.compute_lexical_diversity(first_queries) if first_queries else {"lexical_diversity_score": 0.0}
        
        first_query_diversity = (
            first_query_semantic.get("semantic_diversity_score", 0.0) * 0.7 +
            first_query_lexical.get("lexical_diversity_score", 0.0) * 0.3
        )
        
        return {
            "query_diversity_score": float(overall_score),
            "first_query_diversity": float(first_query_diversity),
            "lexical_diversity": lexical_diversity,
            "semantic_diversity": semantic_diversity,
            "query_count": len(all_queries),
            "unique_query_count": semantic_diversity.get("cluster_count", 0),
            "conversations_analyzed": len(conversations),
            "topic": topic_label
        }
    
    def generate_diversity_report(self, results: Dict, output_file: str = None) -> None:
        """
        Generate a detailed report from diversity analysis results.

        """
        if output_file:
            try:
                # write results to json 
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)
                
                logger.info(f"Diversity report written to {output_file}")
            except Exception as e:
                logger.error(f"Error writing diversity report: {e}")
        else:
            logger.info("Diversity report not saved. No output file specified.")
        
        return results


def compute_query_diversity_for_topic(conversations: List[str], 
                                    topic_label: str = None) -> float:
    """
    Simple interface for computing query diversity score.
    
    Args:
        conversations: List of conversations for a specific topic
        topic_label: Optional topic label
        
    Returns:
        Float score between 0 and 1
    """
    analyzer = QueryDiversityAnalyzer()
    results = analyzer.compute_query_diversity(conversations, topic_label)
    
    return results["query_diversity_score"]


def read_conversations_from_directory(directory: str, pattern: str = "conversation_*.txt") -> Tuple[List[str], List[str]]:
    """
    Read all conversation files from a directory matching a pattern.
    
    Returns:
        Tuple of (conversation_contents, filenames)
    """
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    contents = []
    filenames = []
    
    for f in files:
        content = read_file(f)
        if content.strip():  
            contents.append(content)
            filenames.append(os.path.basename(f))
    
    return contents, filenames

def analyze_query_diversity_from_files(
    conversation_path: str,
    output_file: str = None,
    topic_label: str = None
) -> Dict:
    """
    Analyze query diversity using conversation files.
    
    Args:
        conversation_path: Path to conversation file or directory
        output_file: Optional path to save the report
        topic_label: Optional topic label
        
    Returns:
        Dictionary with diversity analysis results
    """
    if os.path.isdir(conversation_path):
        conversations, _ = read_conversations_from_directory(conversation_path)
    else:
        # Single conversation file
        content = read_file(conversation_path)
        if content:
            conversations = [content]
        else:
            conversations = []
    
    if not conversations:
        logger.error(f"No conversations found at {conversation_path}")
        return {"error": "No conversations found"}
    
    if not topic_label:
        path = Path(conversation_path)
        if path.is_dir():
            topic_label = path.name
        else:
            topic_label = path.parent.name
    
    analyzer = QueryDiversityAnalyzer()
    
    results = analyzer.compute_query_diversity(conversations, topic_label)
    
    if output_file:
        analyzer.generate_diversity_report(results, output_file)
    
    return results

def analyze_multiple_topics(
    base_directory: str,
    output_directory: str = None
) -> Dict:
    """
    analyze query diversity for multiple topics in subdirectories

    """
    topic_dirs = [d for d in glob.glob(os.path.join(base_directory, "*")) if os.path.isdir(d)]
    
    if not topic_dirs:
        logger.error(f"No topic directories found in {base_directory}")
        return {"error": "No topic directories found"}
    
    all_results = []
    
    for topic_dir in topic_dirs:
        topic_name = os.path.basename(topic_dir)
        
        output_file = None
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
            output_file = os.path.join(output_directory, f"{topic_name}_query_diversity.json")
        
        result = analyze_query_diversity_from_files(
            conversation_path=topic_dir,
            output_file=output_file,
            topic_label=topic_name
        )
        
        if "error" not in result:
            all_results.append(result)
    
    if all_results:
        summary = {
            "topics_analyzed": len(all_results),
            "average_diversity_score": np.mean([r["query_diversity_score"] for r in all_results]),
            "average_first_query_diversity": np.mean([r["first_query_diversity"] for r in all_results]),
            "total_queries_analyzed": sum([r["query_count"] for r in all_results]),
            "topic_results": all_results
        }
        
        if output_directory:
            summary_file = os.path.join(output_directory, "diversity_summary.json")
            # write to json
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=4)
            
            logger.info(f"Summary report written to {summary_file}")
        
        return summary
    else:
        return {"error": "No valid results from any topic"}
    