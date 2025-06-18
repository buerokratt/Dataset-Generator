from typing import List, Dict, Tuple, Optional
import re
import numpy as np
import math
from loguru import logger
import sys
from collections import Counter
from utils import Config
from constants import SPEAKER_PATTERNS
logger.remove()
# add stout handler
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


def detect_turns(conversation: str) -> Dict:
    """
    Accurately detects conversation turns using multiple pattern recognition strategies.
    
    Returns a dictionary with turn counts and metadata.
    """
    cleaned_text = re.sub(r'\s+', ' ', conversation).strip()
    
    
    explicit_turns = []
    for pattern in SPEAKER_PATTERNS:
        matches = re.findall(pattern, cleaned_text, re.IGNORECASE)
        explicit_turns.extend(matches)
    
    lines = re.split(r'\n+', cleaned_text)
    complete_lines = [l.strip() for l in lines if len(l.strip()) > 10]
    
    questions = re.findall(r'[^.!?]+\?', cleaned_text)

    sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)
    sentence_lengths = [len(s) for s in sentences if len(s.strip()) > 0]
    
    alternating_pattern = 0
    if len(sentence_lengths) >= 4:
        for i in range(0, len(sentence_lengths) - 1, 2):
            if i+1 < len(sentence_lengths):
                if sentence_lengths[i] < sentence_lengths[i+1]:
                    alternating_pattern += 1
    
    if len(explicit_turns) > 0:
        turn_count = len(explicit_turns)
        method = "explicit_speakers"
    elif len(complete_lines) > 1:
        turn_count = len(complete_lines)
        method = "complete_lines"
    elif len(questions) > 0:
        turn_count = len(questions) * 2
        method = "question_answer_pairs"
    elif alternating_pattern > 0:
        turn_count = alternating_pattern * 2
        method = "dialogue_pattern"
    else:
        word_count = len(cleaned_text.split())
        turn_count = max(1, word_count // 20)  
        method = "text_length_estimate"
    
    return {
        "turn_count": turn_count,
        "detection_method": method,
        "explicit_markers": len(explicit_turns),
        "complete_lines": len(complete_lines),
        "questions": len(questions),
        "alternating_patterns": alternating_pattern,
        "word_count": len(cleaned_text.split()),
        "character_count": len(cleaned_text)
    }

def analyze_topic_complexity(topic_docs: List[str]) -> Dict:
    """
    Perform analysis of topic complexity
    
    """
    full_text = " ".join(topic_docs)
    cleaned_text = re.sub(r'\s+', ' ', full_text).strip()
    
    word_count = len(cleaned_text.split())
    character_count = len(cleaned_text)
    sentence_count = len(re.split(r'[.!?]+', cleaned_text))
    
    avg_sentence_length = word_count / max(1, sentence_count)
    
    words = cleaned_text.lower().split()
    unique_words = len(set(words))
    vocabulary_diversity = unique_words / max(1, word_count)
    
    complex_words = sum(1 for w in words if len(w) > 8) 
    complex_ratio = complex_words / max(1, word_count)
    
    sentences = re.split(r'[.!?]+', cleaned_text)
    non_initial_caps = 0
    for sentence in sentences:
        words_in_sentence = sentence.strip().split()
        if len(words_in_sentence) > 1:

            non_initial_caps += sum(1 for w in words_in_sentence[1:] 
                                    if w and w[0].isupper())

    information_density = (vocabulary_diversity + complex_ratio + 
                           (non_initial_caps / max(1, word_count))) / 3
    

    size_factor = min(1.0, word_count / 300)  
    complexity_factor = min(1.0, information_density * 2) 
    
    overall_complexity = (size_factor * 0.7) + (complexity_factor * 0.3)
    
    config = Config()
    # Determine complexity category, can change the thresholds later on
    if overall_complexity < config.get("length_appropriateness.overall_complexity_simple"):
        complexity_category = "simple"
    elif overall_complexity < config.get("length_appropriateness.overall_complexity_moderate"):
        complexity_category = "moderate"
    else:
        complexity_category = "complex"
    
    return {
        "word_count": word_count,
        "character_count": character_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_length,
        "vocabulary_diversity": vocabulary_diversity,
        "complex_words": complex_words,
        "complex_word_ratio": complex_ratio,
        "named_entities_approx": non_initial_caps,
        "information_density": information_density,
        "overall_complexity": overall_complexity,
        "complexity_category": complexity_category
    }

def calculate_optimal_turn_range(complexity_analysis: Dict) -> Dict:
    """
    Calculates the optimal turn range based on topic complexity analysis.
    
    Returns a dictionary with min, max, and ideal turn counts.
    """
    word_count = complexity_analysis["word_count"]
    complexity_score = complexity_analysis["overall_complexity"]
    category = complexity_analysis["complexity_category"]
    
    # Base ranges for different complexity categories, again can change it later on
    config = Config()
    ranges = {
        "simple": {"min": config.get("length_appropriateness.simple_turn_min"), "max": config.get("length_appropriateness.simple_turn_threshold"), "ideal": config.get("length_appropriateness.simple_turn_ideal")},
        "moderate": {"min": config.get("length_appropriateness.moderate_turn_min"), "max": config.get("length_appropriateness.moderate_turn_threshold"), "ideal": config.get("length_appropriateness.moderate_turn_ideal")},
        "complex": {"min": config.get("length_appropriateness.complex_turn_min"), "max": config.get("length_appropriateness.complex_turn_threshold"), "ideal": config.get("length_appropriateness.complex_turn_ideal")}
    }
    
    base_range = ranges[category]
    
    # fine-tune based on specific word count and complexity
    # adjust for very short or very long topics
    if word_count < 50:
        scaling = 0.7  # reduce turn count for very short topics
    elif word_count > 500:
        scaling = 1.3  # increase turn count for very long topics
    else:
        # linear interpolation between 0.7 and 1.3
        scaling = 0.7 + (word_count - 50) * (0.6 / 450)
        
    # apply scaling
    min_turns = max(2, round(base_range["min"] * scaling))
    max_turns = max(min_turns + 1, round(base_range["max"] * scaling))
    ideal_turns = (min_turns + max_turns) // 2
    # adding some variance
    variance = round(complexity_score * 2)
    
    return {
        "min_turns": min_turns,
        "max_turns": max_turns,
        "ideal_turns": ideal_turns,
        "variance": variance,
        "scaled_from": category
    }

def compute_length_appropriateness_score(conversation: str, topic_docs: List[str]) -> Dict:
    """
    Computes a comprehensive length appropriateness score with detailed analysis.
    
    Returns a dictionary with the score and detailed metrics.
    """
    # analyze the conversation turns
    turn_analysis = detect_turns(conversation)
    actual_turns = turn_analysis["turn_count"]
    
    # analyze topic complexity
    complexity_analysis = analyze_topic_complexity(topic_docs)
    
    # determine optimal turn range
    turn_range = calculate_optimal_turn_range(complexity_analysis)
    min_turns = turn_range["min_turns"]
    max_turns = turn_range["max_turns"]
    ideal_turns = turn_range["ideal_turns"]
    
    # calculate score based on distance from ideal range
    if min_turns <= actual_turns <= max_turns:
        distance_from_ideal = abs(actual_turns - ideal_turns)
        range_size = (max_turns - min_turns) / 2
        
        if range_size > 0:
            score = 1.0 - (distance_from_ideal / range_size) * 0.3  
        else:
            score = 1.0
    elif actual_turns < min_turns:
        shortfall = min_turns - actual_turns
        score = 1.0 - ((shortfall / min_turns) * 0.8)  
    else:
        excess = actual_turns - max_turns
        max_excess = max_turns 
        score = 1.0 - ((excess / max_excess) * 0.6) 
    
    score = max(0.0, min(score, 1.0))
    
    return {
        "appropriateness_score": float(score),
        "actual_turns": actual_turns,
        "recommended_min_turns": min_turns,
        "recommended_max_turns": max_turns,
        "ideal_turns": ideal_turns,
        "turn_analysis": turn_analysis,
        "topic_complexity": complexity_analysis,
        "is_within_range": min_turns <= actual_turns <= max_turns,
        "deviation_from_ideal": actual_turns - ideal_turns
    }

def compute_length_appropriateness(conversation: str, topic_docs: List[str]) -> float:
    """
    Simple interface that returns just the appropriateness score.
    Maintains backward compatibility with the original function.
    """
    results = compute_length_appropriateness_score(conversation, topic_docs)
    return results["appropriateness_score"]

def analyze_conversation_set_length(
    conversations: List[str], 
    topic_docs: List[str]
) -> Dict:
    """
    Analyzes length appropriateness across a set of conversations.
    
    Returns statistical summary and identifies conversations needing adjustment.
    """
    scores = []
    detailed_results = []
    
    complexity_analysis = analyze_topic_complexity(topic_docs)
    turn_range = calculate_optimal_turn_range(complexity_analysis)
    
    for i, conv in enumerate(conversations):
        turn_analysis = detect_turns(conv)
        actual_turns = turn_analysis["turn_count"]
        
        if turn_range["min_turns"] <= actual_turns <= turn_range["max_turns"]:
            distance_from_ideal = abs(actual_turns - turn_range["ideal_turns"])
            range_size = (turn_range["max_turns"] - turn_range["min_turns"]) / 2
            
            if range_size > 0:
                score = 1.0 - (distance_from_ideal / range_size) * 0.3
            else:
                score = 1.0
        elif actual_turns < turn_range["min_turns"]:
            shortfall = turn_range["min_turns"] - actual_turns
            score = 1.0 - ((shortfall / turn_range["min_turns"]) * 0.8)
        else:
            excess = actual_turns - turn_range["max_turns"]
            max_excess = turn_range["max_turns"]
            score = 1.0 - ((excess / max_excess) * 0.6)
        
        score = max(0.0, min(score, 1.0))
        scores.append(score)
        
        detailed_results.append({
            "conversation_index": i,
            "appropriateness_score": float(score),
            "actual_turns": actual_turns,
            "deviation_from_ideal": actual_turns - turn_range["ideal_turns"],
            "is_within_range": turn_range["min_turns"] <= actual_turns <= turn_range["max_turns"],
        })
    
    sorted_results = sorted(detailed_results, key=lambda x: x["appropriateness_score"])
    
    too_short = sum(1 for r in detailed_results if r["actual_turns"] < turn_range["min_turns"])
    too_long = sum(1 for r in detailed_results if r["actual_turns"] > turn_range["max_turns"])
    good_length = len(detailed_results) - too_short - too_long
    
    return {
        "mean_score": np.mean(scores) if scores else 0.0,
        "topic_complexity": complexity_analysis,
        "recommended_turn_range": turn_range,
        "good_length_count": good_length,
        "too_short_count": too_short,
        "too_long_count": too_long,
        "total_count": len(scores),
        "good_length_percentage": (good_length / len(scores) * 100) if scores else 0,
        "detailed_results": sorted_results
    }
