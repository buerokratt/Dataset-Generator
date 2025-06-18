USER_QUERY_PATTERNS = [
            r'\*\*Kasutaja\*\*:(.*?)(?:\*\*Robot\*\*:|$)', 
            r'Kasutaja:(.*?)(?:Robot:|$)',                   
            r'User:(.*?)(?:Assistant:|$)',                   
            r'Human:(.*?)(?:AI:|$)'                        
        ]

SPEAKER_PATTERNS = [
    r'\*\*(?:Kasutaja|Robot|User|Assistant|Human|AI)\*\*\s*:',
        r'(?:Kasutaja|Robot|User|Assistant|Human|AI)\s*:',
    ]

TURNS = r'[\n\r]+|(?:(?:\*\*Kasutaja\*\*:|\*\*Robot\*\*:)\b)'

PREPROCESS_TEXT_PATTERN = r'https?://\S+|www\.\S+|\S+@\S+'

AGENCY_CONFUSION_KEYS = {
    "OVERALL_CONFUSION_RATE": "overall_confusion_rate",
    "OVERALL_SIMILARITY": "overall_similarity",
    "TOTAL_CONFUSION_PAIRS": "total_confusion_pairs",
    "AGENCY_PAIR_RESULTS": "agency_pair_results",
    "AGENCY_TOPIC_COUNTS": "agency_topic_counts",
    "MOST_CONFUSABLE_PAIR": "most_confusable_pair"
}