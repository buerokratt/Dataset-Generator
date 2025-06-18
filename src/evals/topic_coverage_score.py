from typing import List, Dict, Tuple, Set, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import  DBSCAN
import re
from collections import Counter, defaultdict
from loguru import logger
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import json
import seaborn as sns
import sys
from utils import Config
from utils import load_stopwords
from utils import preprocess_text
from utils import read_file
from utils import NumpyEncoder
logger.remove()
# add stout handler
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


class TopicCoverageAnalyzer:
    """
    A class for analyzing whether all important topics in a source document 
    are covered by the generated conversations.
    """
    
    def __init__(self):
        """
        Initialize the topic coverage analyzer.
    
        """
        self.config = Config()
        self.embedding_model_name = self.config.get("topic_coverage.embedding_model")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.min_segment_length = self.config.get("topic_coverage.min_segment_length")
        self.max_segment_length = self.config.get("topic_coverage.max_segment_length")
        self.min_topic_size = self.config.get("topic_coverage.min_topic_size")
        self.clustering_threshold = self.config.get("topic_coverage.clustering_threshold")
        self.stopwords = load_stopwords()
        
        # Will store results
        self.document_segments = []
        self.document_topics = []
        self.conversation_topics = []
        self.uncovered_topics = []
        self.coverage_scores = {}
        
    
    def _segment_document(self, document: str) -> List[str]:
        """
        Split document into meaningful segments for topic identification.
        Uses paragraph and sentence boundaries to create logical segments.
        """
        document = re.sub(r'\s+', ' ', document).strip()
        
        paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n', document)
        
        segments = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            if len(para) <= self.max_segment_length:
                segments.append(para)
                continue
                
            sentences = re.split(r'(?<=[.!?])\s+', para)
            current_segment = ""
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                if len(current_segment) + len(sentence) > self.max_segment_length:
                    if current_segment:
                        segments.append(current_segment)
                    current_segment = sentence
                else:
                    current_segment += " " + sentence if current_segment else sentence
            
            if current_segment:
                segments.append(current_segment)
        
        segments = [seg for seg in segments if len(seg) >= self.min_segment_length]
        
        segments = [preprocess_text(seg) for seg in segments]
        
        return segments
    
    def _extract_keywords(self, text: str, n: int = 10) -> List[str]:
        """Extract important keywords from text using TF-IDF."""
        try:
            vectorizer = TfidfVectorizer(
                stop_words=self.stopwords,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.9
            )
            
            tfidf_matrix = vectorizer.fit_transform([text, "dummy text"])
            feature_names = vectorizer.get_feature_names_out()
            
            scores = zip(feature_names, tfidf_matrix[0].toarray()[0])
            sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
            
            return [word for word, score in sorted_scores[:n]]
            
        except Exception as e:
            logger.warning(f"Error extracting keywords: {e}")
            words = text.lower().split()
            word_counts = Counter(words)
            #word_counts = {w: c for w, c in word_counts.items() if w not in self.stopwords and len(w) > 2}
            return [w for w, c in word_counts.most_common(n)]
    
    def _identify_document_topics(self, segments: List[str]) -> List[Dict]:
        """
        Identify distinct topics within the document by clustering segments.
        
        Returns:
            List of topics, each containing details about the segments and keywords
        """
        if not segments:
            return []
            
        try:
            embeddings = self.embedding_model.encode(segments)
            
            clustering = DBSCAN(
                eps=self.clustering_threshold,
                min_samples=self.min_topic_size,
                metric='cosine'
            ).fit(embeddings)
            
            labels = clustering.labels_
            
            topics = defaultdict(list)
            for i, label in enumerate(labels):
                if label != -1:  
                    topics[label].append({
                        "segment": segments[i],
                        "segment_id": i,
                        "embedding": embeddings[i]
                    })
            
            topic_details = []
            for topic_id, topic_segments in topics.items():
                combined_text = " ".join([s["segment"] for s in topic_segments])
                
                keywords = self._extract_keywords(combined_text)
                
                centroid = np.mean([s["embedding"] for s in topic_segments], axis=0)
                
                topic_details.append({
                    "topic_id": topic_id,
                    "segments": topic_segments,
                    "segment_count": len(topic_segments),
                    "keywords": keywords,
                    "centroid": centroid,
                })
            
            return topic_details
            
        except Exception as e:
            logger.error(f"Error identifying document topics: {e}")
            return []
    
    def _evaluate_conversation_topic_coverage(self, 
                                             conversations: List[str], 
                                             document_topics: List[Dict]) -> Dict:
        """
        Evaluate how well conversations cover the document topics.
        
        Returns:
            Dictionary with coverage analysis results
        """
        if not conversations or not document_topics:
            return {
                "total_topics": 0,
                "covered_topics": 0,
                "coverage_percentage": 0.0,
                "uncovered_topics": [],
                "topic_coverage_scores": {},
                "conversation_coverage": []
            }
        
        preprocessed_conversations = [preprocess_text(conv) for conv in conversations]
        
        conversation_embeddings = self.embedding_model.encode(preprocessed_conversations)
        
        topic_coverage = {}
        
        for topic in document_topics:
            topic_id = topic["topic_id"]
            topic_centroid = topic["centroid"]
            
            similarities = cosine_similarity([topic_centroid], conversation_embeddings)[0]
            
            best_match_idx = np.argmax(similarities)
            best_match_score = similarities[best_match_idx]
            
            is_covered = best_match_score >= 0.5
            
            topic_coverage[str(topic_id)] = {
                "topic_id": str(topic_id),
                "keywords": topic["keywords"],
                "segment_count": topic["segment_count"],
                "is_covered": 1 if is_covered else 0,
                "best_match_score": str(best_match_score),
                "best_match_conversation_idx": str(best_match_idx)
            }
        
        uncovered_topics = [
            {
                "topic_id": t["topic_id"],
                "keywords": t["keywords"],
                "segment_count": t["segment_count"]
            }
            for t in document_topics
            if not bool(topic_coverage[str(t["topic_id"])]["is_covered"])
        ]
        
        total_topics = len(document_topics)
        covered_topics = sum(1 for t in topic_coverage.values() if bool(t["is_covered"]))
        coverage_percentage = str((covered_topics / total_topics) * 100) if total_topics > 0 else "0"
        
        conversation_coverage = []
        for i, (conv, embedding) in enumerate(zip(conversations, conversation_embeddings)):
            covered_topics = []
            for topic in document_topics:
                topic_id = topic["topic_id"]
                similarity = cosine_similarity([topic["centroid"]], [embedding])[0][0]
                if similarity >= 0.5:
                    covered_topics.append({
                        "topic_id": str(topic_id),
                        "similarity": str(similarity),
                        "keywords": topic["keywords"]
                    })
            
            keywords = self._extract_keywords(conv)
            
            conversation_coverage.append({
                "conversation_idx": str(i),
                "covered_topics": covered_topics,
                "topic_count": str(len(covered_topics)),
                "keywords": keywords
            })
        
        return {
            "total_topics": total_topics,
            "covered_topics": covered_topics,
            "coverage_percentage": coverage_percentage,
            "uncovered_topics": uncovered_topics,
            "topic_coverage_scores": topic_coverage,
            "conversation_coverage": conversation_coverage
        }
    
    def analyze_topic_coverage(self, 
                              document: str, 
                              conversations: List[str]) -> Dict:
        """
        Analyze how well the conversations cover topics in the document.
        

        """
        self.document_segments = self._segment_document(document)
        
        if not self.document_segments:
            logger.error("Could not extract meaningful segments from document")
            return {"error": "Could not extract meaningful segments from document"}
        
        self.document_topics = self._identify_document_topics(self.document_segments)
        
        if not self.document_topics:
            logger.error("Could not identify distinct topics in document")
            return {"error": "Could not identify distinct topics in document"}
        
        coverage_results = self._evaluate_conversation_topic_coverage(
            conversations, 
            self.document_topics
        )
        
        self.uncovered_topics = coverage_results["uncovered_topics"]
        self.coverage_scores = coverage_results["topic_coverage_scores"]
        
        return coverage_results
    
    def generate_coverage_report(self, 
                               document: str, 
                               conversations: List[str],
                               conversation_files: List[str] = None,
                               output_dir: str = None) -> Dict:
        """
        Generate comprehensive topic coverage report.
        
        Args:
            document: Source document text
            conversations: List of conversation texts
            conversation_files: Optional list of conversation filenames
            output_dir: Optional directory to save report files
            
        Returns:
            Dictionary with coverage analysis and report file paths
        """
        coverage_results = self.analyze_topic_coverage(document, conversations)
        
        if "error" in coverage_results:
            return coverage_results
        
        if not conversation_files:
            conversation_files = [f"conversation_{i+1}.txt" for i in range(len(conversations))]
        
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for i, coverage in enumerate(coverage_results["conversation_coverage"]):
            if i < len(conversation_files):
                coverage["filename"] = conversation_files[i]
        
        output_files = {}
        if output_dir:
            md_report_path = os.path.join(output_dir, "topic_coverage_report.json")
            self._generate_markdown_report(coverage_results, md_report_path)
            output_files["markdown_report"] = md_report_path
            
            heatmap_path = os.path.join(output_dir, "topic_coverage_heatmap.png")
            self._generate_coverage_heatmap(coverage_results, heatmap_path)
            output_files["heatmap"] = heatmap_path
            
        
        coverage_results["output_files"] = output_files
        
        return coverage_results
    
    def _generate_markdown_report(self, 
                                coverage_results: Dict, 
                                output_file: str) -> None:
        """Generate detailed markdown report of topic coverage."""
        try:
            # write coverage results to json 
            results = convert_dict_keys(coverage_results)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, cls=NumpyEncoder, ensure_ascii=False, indent=4)
            logger.info(f"Markdown report written to {output_file}")
            
        except Exception as e:
            logger.error(f"Error generating markdown report: {e}")
    
    def _generate_coverage_heatmap(self, 
                                 coverage_results: Dict, 
                                 output_file: str) -> None:
        """Generate heatmap visualization of topic-conversation coverage."""
        try:
            topic_ids = sorted([t["topic_id"] for t in self.document_topics])
            conversation_ids = [i for i in range(len(coverage_results["conversation_coverage"]))]
            
            coverage_matrix = np.zeros((len(conversation_ids), len(topic_ids)))
            
            for conv_idx, conv_coverage in enumerate(coverage_results["conversation_coverage"]):
                for topic in conv_coverage["covered_topics"]:
                    topic_idx = topic_ids.index(topic["topic_id"])
                    coverage_matrix[conv_idx, topic_idx] = topic["similarity"]
            
            plt.figure(figsize=(max(8, len(topic_ids)*0.5), max(6, len(conversation_ids)*0.4)))
            
            conv_labels = []
            for i, conv in enumerate(coverage_results["conversation_coverage"]):
                if "filename" in conv:
                    # Shorten filename if needed
                    name = os.path.basename(conv["filename"])
                    if len(name) > 20:
                        name = name[:17] + "..."
                    conv_labels.append(name)
                else:
                    conv_labels.append(f"Conv {i+1}")
            
            topic_labels = []
            for topic_id in topic_ids:
                topic = next((t for t in self.document_topics if t["topic_id"] == topic_id), None)
                if topic:
                    keywords = " ".join(topic["keywords"][:2])  # Just top 2 keywords
                    topic_labels.append(f"Topic {topic_id}\n({keywords})")
                else:
                    topic_labels.append(f"Topic {topic_id}")
            
            ax = sns.heatmap(
                coverage_matrix,
                annot=True,
                cmap="YlGnBu",
                vmin=0.0,
                vmax=1.0,
                xticklabels=topic_labels,
                yticklabels=conv_labels,
                linewidths=0.5,
                fmt=".2f"
            )
            
            plt.title("Topic Coverage by Conversation")
            plt.ylabel("Conversations")
            plt.xlabel("Topics (with key terms)")
            plt.tight_layout()
            
            plt.xticks(rotation=45, ha="right")
            
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            plt.close()
            
            logger.info(f"Coverage heatmap written to {output_file}")
            
        except Exception as e:
            logger.error(f"Error generating coverage heatmap: {e}")
   
def convert_dict_keys(d):
    """Convert all NumPy int64 keys in a dictionary to Python int."""
    if not isinstance(d, dict):
        return d
    
    result = {}
    for k, v in d.items():
        # Convert the key if it's a NumPy type
        if isinstance(k, np.integer):
            k = str(k)
        elif isinstance(k, np.floating):
            k = str(k)
        elif isinstance(k, bool):
            k = "1" if k else "0"
            
        # Recursively convert nested dictionaries
        if isinstance(v, dict):
            v = convert_dict_keys(v)
        elif isinstance(v, list):
            v = [convert_dict_keys(item) if isinstance(item, dict) else item for item in v]
            
        result[k] = v
    return result   
 


def read_topic_documents(topic_docs_path: str) -> str:
    """Read topic document content."""
    if os.path.isfile(topic_docs_path):
        return read_file(topic_docs_path)
    elif os.path.isdir(topic_docs_path):
        files = glob.glob(os.path.join(topic_docs_path, "*.txt"))
        content = []
        for f in files:
            content.append(read_file(f))
        return "\n\n".join(content)
    else:
        logger.error(f"Invalid topic document path: {topic_docs_path}")
        return ""

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
        if content.strip():  # Skip empty files
            contents.append(content)
            filenames.append(os.path.basename(f))
    
    return contents, filenames

def analyze_topic_coverage_from_files(
    conversation_path: str,
    topic_docs_path: str,
    output_dir: str = None
) -> Dict:
    """
    Analyze topic coverage using conversation files and topic document.

    """
    document = read_topic_documents(topic_docs_path)
    
    if not document:
        logger.error(f"Could not read topic document from {topic_docs_path}")
        return {"error": "Could not read topic document"}
    
    if os.path.isdir(conversation_path):
        conversations, filenames = read_conversations_from_directory(conversation_path)
    else:
        content = read_file(conversation_path)
        if content:
            conversations = [content]
            filenames = [os.path.basename(conversation_path)]
        else:
            conversations = []
            filenames = []
    
    if not conversations:
        logger.error(f"No conversations found at {conversation_path}")
        return {"error": "No conversations found"}
    
    analyzer = TopicCoverageAnalyzer()
    
    return analyzer.generate_coverage_report(
        document,
        conversations,
        filenames,
        output_dir
    )

