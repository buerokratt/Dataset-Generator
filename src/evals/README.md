## Evaluation Metrics

This document introduces the evaluation metrics created to evaluate the LLM generated conversation data.

## Quantitative Metrics

### Information Coverage Score

**Purpose:**  
The Information Coverage Score is designed to evaluate how well an automatically generated conversation covers the key information present in reference topic documents.

**Technical Implementation**  
The metric employs semantic similarity between conversation chunks and topic document chunks to quantify information coverage.

**Key Components**
- **Model Selection:** paraphrase-multilingual-mpnet-base-v2
  - A multilingual sentence transformer model that supports Estonian
  - Provides strong performance on semantic similarity tasks for less-resourced languages
  - Alternative models are available as commented options for different performance-speed tradeoffs
- **Text Preprocessing:**
  - Cleaning: Removal of excess whitespace and normalization
  - Chunking: Splitting text into manageable segments at sentence boundaries
- **Similarity Computation:**
  - Embedding Generation: Each text chunk is converted to a vector representation
  - Cosine Similarity: Measures semantic similarity between embeddings
  - Maximum Similarity: For each topic chunk, finds the most similar conversation chunk
- **Coverage Scoring:**
  - Threshold Application: A similarity score ≥ 0.5 indicates a match
  - Coverage Calculation: Proportion of topic chunks that have a match in the conversation
  - Output: A float score between 0 and 1, plus the list of matched chunks

**Interpretation of Scores**

| Score Range | Interpretation |
|-------------|----------------|
| 0.8 - 1.0   | Excellent coverage of reference information |
| 0.6 - 0.8   | Good coverage of reference information |
| 0.4 - 0.6   | Moderate coverage of reference information |
| 0.2 - 0.4   | Limited coverage of reference information |
| 0.0 - 0.2   | Poor coverage of reference information |


### Redundancy Penalty

The Conversation Redundancy Metric is designed to evaluate the uniqueness and diversity of automatically generated conversations for training data. It operates on two levels:
- Intra-conversation redundancy: Measures repetition within a single conversation
- Inter-conversation redundancy: Measures similarity between different conversations

This dual approach ensures that generated conversations are both internally coherent (without excessive repetition) and collectively diverse (providing varied training examples for agency classification).

**Technical Implementation**  
The metric employs semantic similarity between conversation turns and between entire conversations to quantify redundancy.

**Key Components**
- **Model Selection:** paraphrase-multilingual-mpnet-base-v2
  - A multilingual sentence transformer model with strong performance on Estonian
  - Converts text into semantic vector representations for similarity comparisons
- **Text Preprocessing:**
  - Conversation Parsing: Splits text into turns based on user/robot patterns
  - Cleaning: Removes extraneous whitespace and filters out very short turns
- **Intra-Conversation Redundancy:**
  - Turn-by-Turn Analysis: Compares each turn with every other turn in the same conversation
  - Pair Counting: Identifies turns with similarity above threshold (default: 0.8)
  - Scoring: Calculates the ratio of redundant pairs to total pairs
- **Inter-Conversation Redundancy:**
  - Conversation Embedding: Creates a representative embedding for each conversation
  - Cross-Comparison: Measures similarity between all conversation pairs
  - Redundancy Detection: Identifies conversation pairs with similarity above threshold (default: 0.7)
  - Overall Score: Calculates the ratio of redundant conversation pairs to total pairs
- **Visualization:**
  - Similarity Heatmap: Displays the similarity matrix between all conversations
  - Highlights problematic pairs for closer inspection

**Intra-Conversation Scores**

| Score Range | Interpretation |
|-------------|----------------|
| 0.0 - 0.1   | Excellent variety with minimal repetition |
| 0.1 - 0.3   | Good variety, acceptable repetition |
| 0.3 - 0.5   | Moderate repetition, may need review |
| 0.5 - 1.0   | Excessive repetition, needs improvement |

**Inter-Conversation Scores**

| Score Range | Interpretation |
|-------------|----------------|
| 0.0 - 0.1   | Excellent diversity across conversations |
| 0.1 - 0.2   | Good diversity, some similarity acceptable |
| 0.2 - 0.4   | Concerning similarity levels, review needed |
| 0.4 - 1.0   | Insufficient diversity, regeneration recommended |


### Relevance Metric

The Conversation Relevance Metric evaluates how well a generated conversation aligns with its source topic documents. This metric is crucial for ensuring that generated training data stays on topic and contains appropriate information for agency classification.

This multi-faceted approach captures different aspects of relevance to provide a comprehensive assessment of conversation quality.

**Technical Implementation**  
The metric employs a weighted combination of three complementary relevance measures.

**Key Components**
- **Model Selection:** paraphrase-multilingual-mpnet-base-v2
  - Multilingual model with strong performance on Estonian
  - Provides accurate semantic representations for text comparison
- **Multi-dimensional Relevance Analysis:** 
  - **a. Segment-level Relevance (60% weight)**
    - Splits both conversation and topic documents into meaningful segments
    - Computes cross-similarities between all segments
    - Captures fine-grained topical alignment across different parts of the conversation
  - **b. Query-focused Relevance (30% weight)**
    - Extracts user queries/questions from the conversation
    - Compares these directly to topic documents
    - Emphasizes the initial query which usually contains the main user intent
    - Ensures the conversation addresses the specific information need
  - **c. Key Term Overlap (10% weight)**
    - Identifies important terminology using TF-IDF analysis
    - Measures the Jaccard similarity between key terms in conversation and topic
    - Ensures domain-specific terminology is properly represented
- **Text Processing:**
  - Conversation parsing for various formats
  - Intelligent segmentation using sentence boundaries
  - Special handling for user queries vs. assistant responses
- **Analysis Tools:**
  - Per-conversation detailed metrics
  - Conversation set analysis for quality distribution
  - Identification of top-performing conversations

**Interpretation of Scores**

| Score Range | Interpretation |
|-------------|----------------|
| 0.8 - 1.0   | Excellent relevance - conversation closely aligns with topic |
| 0.7 - 0.8   | Good relevance - conversation covers topic well |
| 0.5 - 0.7   | Acceptable relevance - conversation is on topic but may miss some aspects |
| 0.3 - 0.5   | Poor relevance - conversation partially addresses topic |
| 0.0 - 0.3   | Irrelevant - conversation strays significantly from topic |


### Conversation Length Appropriateness Metric

The Conversation Length Appropriateness Metric evaluates whether a generated conversation has an appropriate number of turns for the complexity and size of the topic it addresses. This metric ensures that conversations are neither too brief to cover the necessary information nor excessively long for simple topics.

The metric is particularly valuable for training data generation, as properly sized conversations improve both the efficiency and effectiveness of the resulting classifier.

**Technical Implementation**  
The metric employs a multi-dimensional analysis approach to determine optimal conversation length and evaluate actual conversations against this standard.

**Key Components**
- **Turn Detection:**
  - Multiple pattern recognition strategies identify conversation turns
  - Explicit speaker markers (e.g., "User:", "Robot:")
  - Complete thought identification
  - Question-answer pair detection
  - Dialogue pattern recognition
- **Topic Complexity Analysis:**
  - Word count and sentence analysis
  - Vocabulary diversity measurement
  - Complex word and specialized terminology detection
  - Information density calculation
  - Overall complexity scoring (simple, moderate, complex)
- **Optimal Turn Range Calculation:**
  - Base ranges determined by complexity category
  - Fine-tuning based on specific word count
  - Scaling factors for very short or very long topics
  - Variance allowance for highly complex topics
- **Scoring Algorithm:**
  - Perfect score for conversations with ideal turn count
  - Graduated penalties for deviations from the ideal range
  - Steeper penalties for being too short than too long
  - Normalized scores between 0 and 1

**Interpretation of Scores**

| Score Range | Interpretation |
|-------------|----------------|
| 0.9 - 1.0   | Excellent - conversation length is ideal for the topic |
| 0.7 - 0.9   | Good - conversation length is appropriate with minor deviation |
| 0.5 - 0.7   | Acceptable - conversation length has noticeable deviation but remains functional |
| 0.3 - 0.5   | Poor - conversation is significantly too short or too long |
| 0.0 - 0.3   | Inadequate - conversation length is entirely inappropriate for the topic |


### Topic Consistency Metric

The Topic Consistency Metric evaluates how well a generated conversation maintains focus on and covers the subject matter from its source topic document. This metric is essential for ensuring that generated training data properly aligns with specific agency domains for accurate downstream classification.

The metric combines two key aspects of topic quality:
- Topic coherence - Internal consistency of the conversation
- Topic alignment - How well the conversation relates to its source document

**Technical Implementation**  
The metric employs advanced semantic analysis techniques specifically adapted for Estonian language conversations.

**Key Components**
- **Estonian-optimized Embedding Model:**
  - Uses paraphrase-multilingual-mpnet-base-v2 for high-quality Estonian language understanding
  - Captures semantic relationships between conversation turns and topic content
- **Multi-level Content Analysis:** 
  - **a. Conversation Structure Analysis**
    - Automatically identifies user queries and system responses
    - Extracts conversation turns from various formatting styles
    - Processes natural language segments for meaningful comparison
  - **b. Turn-level Coherence Analysis**
    - Measures semantic similarity between adjacent turns
    - Quantifies consistency throughout the conversation flow
    - Identifies potential topic drift or abrupt shifts
  - **c. Topic Alignment Assessment**
    - Compares conversation to source document using multiple approaches:
      - Full text semantic similarity
      - User query to topic similarity (weighted for intent)
      - Segment-level matching for granular comparison
      - Keyword overlap analysis
- **Estonian Language Processing:**
  - Preserves Estonian special characters (ä, ö, ü, õ)
  - Applies custom stopword filtering
  - Handles Estonian grammatical structures
  - Normalizes text while maintaining semantic integrity
- **Comprehensive Scoring System:**
  - Weighted combination of coherence and alignment
  - Segment-level and document-level analysis
  - Normalized scores between 0 and 1
  - Detailed diagnostic breakdowns

**Overall Topic Quality Score**

| Score Range | Interpretation |
|-------------|----------------|
| 0.8 - 1.0   | Excellent - highly coherent conversation with strong topic alignment |
| 0.6 - 0.8   | Good - consistent conversation with appropriate topic coverage |
| 0.4 - 0.6   | Acceptable - conversation stays generally on topic with minor issues |
| 0.2 - 0.4   | Poor - conversation shows topic drift or weak alignment |
| 0.0 - 0.2   | Inadequate - conversation lacks coherence or topic relevance |


### Topic Coverage Gap Analysis 

The Topic Coverage Gap Analysis tool identifies whether all important topics in a source document are adequately covered by generated conversations. This metric is critical for ensuring comprehensive training data that represents the full breadth of information an agency handles.

Unlike other metrics that evaluate conversation quality, this tool focuses on completeness of topic coverage across the entire conversation set, identifying potential gaps that need to be addressed.

**Technical Implementation**  
The metric employs advanced natural language processing techniques to identify distinct topics in source documents and evaluate how well conversations cover those topics.

**Key Components**
- **Intelligent Document Segmentation:**
  - Splits source documents into meaningful segments using paragraph and sentence boundaries
  - Preserves contextual relationships between related content
  - Creates logical units for topic identification
- **Topic Discovery:**
  - Uses density-based clustering (DBSCAN) to identify distinct topic clusters
  - Extracts representative keywords for each topic
  - Calculates topic centroids for semantic comparison
- **Comprehensive Coverage Analysis:**
  - Measures semantic similarity between conversations and identified topics
  - Evaluates coverage at both topic and conversation levels
  - Identifies specifically which topics remain uncovered
- **Visualization and Reporting:**
  - Generates detailed Markdown reports identifying gaps
  - Creates coverage heatmaps showing topic-conversation relationships
  - Produces data files for further analysis

**Output Details**  
The tool provides comprehensive analysis outputs:

**Coverage Metrics**
- Total topics: Number of distinct topics identified in the source document
- Topics covered: Number of topics adequately addressed by at least one conversation
- Coverage percentage: Proportion of topics covered (higher is better)
- Uncovered topics: List of topics with inadequate conversation coverage

**Detailed Analysis**
- Per-topic coverage: Which conversation best covers each topic
- Per-conversation topics: Which topics each conversation addresses
- Keyword alignment: Key terms shared between topics and conversations
- Coverage strength: Quantified similarity between conversations and topics

**Visualization**
- Coverage heatmap: Visual representation of topic-conversation relationships
- Detailed report: Markdown document highlighting gaps and suggestions
- Data export: Excel/CSV files for integration with other tools

**Interpretation Guidelines**  
For a comprehensive training dataset:
- Coverage percentage should be at least 90%, with higher being better
- Uncovered topics list should ideally be empty
- Each identified topic should have at least one conversation with similarity > 0.5
- Topics with the following characteristics deserve special attention:
  - Topics with high segment counts (substantial content in source document)
  - Topics with distinctive, domain-specific keywords
  - Topics that represent key agency functionalities

### Query Diversity Metric

The Query Diversity Metric evaluates the variety and range of user queries across a collection of generated conversations for a given topic. This metric is specifically designed to focus on user inputs rather than system responses, making it ideal for assessing training data quality for classification systems.

Unlike other metrics that evaluate individual conversation quality, this metric analyzes the diversity of ways users might ask about the same topic, which is crucial for building robust classification models that can handle query variations.

**Technical Implementation**  
The metric employs both lexical and semantic analysis to quantify the diversity of user queries.

**Key Components**
- **User Query Extraction:**
  - Identifies user turns using various conversation format patterns
  - Filters out system/assistant responses
  - Focuses on initial queries which are most important for classification
  - Handles various conversation structures and formats
- **Lexical Diversity Analysis:**
  - Measures vocabulary variety using unique token ratio
  - Analyzes n-gram diversity to capture phrasing variations
  - Accounts for Estonian language characteristics
  - Quantifies word usage patterns across queries
- **Semantic Diversity Analysis:**
  - Uses multilingual embeddings optimized for Estonian
  - Identifies clusters of semantically similar queries
  - Measures intent variety independent of specific wording
  - Quantifies the distribution of query intents
- **First-Query Focus:**
  - Gives special attention to conversation-starting queries
  - Evaluates diversity of initial user intents
  - Prioritizes classification-relevant query variations
  - Provides targeted metrics for classifier training

**Overall Query Diversity Score**

| Score Range | Interpretation |
|-------------|----------------|
| 0.8 - 1.0   | Excellent - wide variety of query formulations |
| 0.6 - 0.8   | Good - diverse ways of asking about the topic |
| 0.4 - 0.6   | Moderate - some variation exists but limited range |
| 0.2 - 0.4   | Low - most queries follow similar patterns |
| 0.0 - 0.2   | Very low - highly repetitive query formulations |

**First Query Diversity Score**

| Score Range | Interpretation |
|-------------|----------------|
| 0.8 - 1.0   | Excellent - diverse conversation starters |
| 0.6 - 0.8   | Good - varied initial queries |
| 0.4 - 0.6   | Moderate - some variety in how conversations begin |
| 0.2 - 0.4   | Low - limited variety in opening queries |
| 0.0 - 0.2   | Very low - nearly identical conversation starters |

**Main Metrics**
- **query_diversity_score:** Combined measure of overall query diversity
- **first_query_diversity:** Specific score for initial user query diversity
- **lexical_diversity:** Vocabulary and phrasing variety measurements
- **semantic_diversity:** Intent and meaning variety measurements

**Detailed Analysis**
- **unique_query_count:** Number of distinct query intents identified
- **cluster_examples:** Representative queries from each intent cluster
- **cluster_ratio:** Proportion of unique intents to total queries
- **vocabulary statistics:** Token counts, ratios, and n-gram diversity

### Agency Confusion Analysis

The Agency Confusion Analysis Metric identifies potential classification confusion between conversations from different agencies or between topics within the same agency. This metric is critical for ensuring a robust agency classification system that can accurately route user queries to the correct government entity.

This analysis specifically focuses on the user queries within conversations, examining how similar they appear across agencies and how likely they might be to cause classification errors.

**Technical Implementation**  
The metric employs semantic similarity analysis to identify potentially confusable conversation pairs between agencies.

**Key Components**
- **User Query Extraction:**
  - Isolates first user query from each conversation (most important for classification)
  - Supports various conversation formats (marked with "Kasutaja:", "User:", etc.)
  - Focuses exclusively on user inputs, not system responses
  - Preprocesses text to maintain Estonian language characteristics
- **Cross-Agency Similarity Analysis:**
  - Generates semantic embeddings for all user queries
  - Computes pairwise similarity between queries from different agencies
  - Identifies highly similar query pairs above a configurable threshold
  - Calculates confusion rates and overall similarity metrics
- **Within-Agency Topic Analysis:**
  - Analyzes confusion between different topics within the same agency
  - Identifies overlapping user intent patterns between topics
  - Helps refine topic differentiation within agencies
  - Supports hierarchical analysis of confusion
- **Visualization and Reporting:**
  - Generates confusion matrix heatmaps for visual analysis
  - Produces detailed reports of potentially confusable queries
  - Provides actionable recommendations to reduce confusion
  - Enables comparison across multiple agencies

**Confusion Rate**

| Rate Range | Interpretation |
|------------|----------------|
| 0.00 - 0.01 | Excellent - very low potential for confusion |
| 0.01 - 0.05 | Good - minimal confusion likely |
| 0.05 - 0.10 | Moderate - some queries may be misclassified |
| 0.10 - 0.20 | High - significant potential for misclassification |
| > 0.20      | Very high - classifier will likely struggle with accuracy |


## Qualitative Metrics

The Qualitative Conversation Evaluation Metric provides comprehensive human-like assessment of conversation quality using a lightweight language model designed to run on GPUs with limited VRAM (9GB). This metric goes beyond quantitative measurements to evaluate subjective aspects that typically require human judgment.

**Technical Implementation**  
This implementation leverages a small but capable language model to perform detailed evaluations across multiple quality dimensions.

**Key Components**
- **Lightweight Model Selection:**
  - Uses Google's Gemma-2B-IT model by default (fits within 9GB VRAM)
  - Implements 4-bit quantization to further reduce memory requirements
  - Preserves evaluation quality while ensuring hardware compatibility
- **Multi-dimensional Evaluation Framework:**
  - Overall Quality: Holistic assessment of conversation effectiveness
  - Coherence: Logical flow and connection between turns
  - Relevance: How directly responses address user queries
  - Factual Accuracy: Correctness of information provided
  - Helpfulness: Practical value and problem-solving effectiveness
  - Natural Language: Human-like and appropriate language use
  - Completeness: Coverage of all aspects of user queries
- **Qualitative Analysis Features:**
  - Strengths & Weaknesses: Detailed assessment of positive and negative aspects
  - Improvement Suggestions: Actionable recommendations for enhancement
  - Reasoning: Explanation behind each numerical score
- **Hierarchical Evaluation Support:**
  - Handles individual conversations, topic directories, or entire agency structures
  - Aggregates results at multiple levels for comparative analysis
  - Generates reports appropriate to each organizational level

**Numerical Scores (1-5 scale)**

| Score Range | Interpretation |
|-------------|----------------|
| 4.5 - 5.0   | Excellent - Professional quality conversation |
| 4.0 - 4.4   | Good - Effective conversation with minor issues |
| 3.0 - 3.9   | Acceptable - Functional but needs improvement |
| 2.0 - 2.9   | Poor - Significant issues affecting usefulness |
| 1.0 - 1.9   | Inadequate - Major problems, not fit for purpose |

**Output Formats**  
The evaluation produces several output formats:

**JSON Results:** Structured data for each conversation with:
- Numerical scores for each criteria
- Qualitative feedback and reasoning
- Strengths, weaknesses, and improvement suggestions

**Markdown Reports:**
- Conversation-level detailed assessments
- Topic-level summaries and distributions
- Agency-level comparative analysis

**Summary Statistics:**
- Average scores by criteria
- Quality distribution analysis
- Cross-topic and cross-agency comparisons