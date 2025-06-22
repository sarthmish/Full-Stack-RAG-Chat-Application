from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
import logging

logger = logging.getLogger("root")
logger.info("Imported Data Schemas module")

### Retrieval Grader

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

### Hallucination Grader

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

### Answer Grader

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

### Query Pre-processing Schemas

class PreProcessedQuery(BaseModel):
    """Pre-processed query"""

    pre_processed_query: str = Field(
        description="Pre-Processed Query without grammatical or logical mistakes"
    )

class MultiQueryList(BaseModel):
    """List of multiple queries generated from a single one."""

    multi_query_list: List[str] = Field(
        description="List of queries generated from a single query"
    )


# EVALUATION SCHEMAS

# from enum import Enum

# Define Enum for scores
# class Score(str, Enum):
#     no_relevance = "0"
#     low_relevance = "1"
#     medium_relevance = "2"
#     high_relevance = "3"

# Define a constant for the score description
SCORE_DESCRIPTION = (
    "Score as a integer between 0 and 3. "
    "0: No relevance/Not grounded/Irrelevant - The context/answer is completely unrelated or not based on the context. "
    "1: Low relevance/Low groundedness/Somewhat relevant - The context/answer has minimal relevance or grounding. "
    "2: Medium relevance/Medium groundedness/Mostly relevant - The context/answer is somewhat relevant or grounded. "
    "3: High relevance/High groundedness/Fully relevant - The context/answer is highly relevant or grounded."
)

# Define separate classes for each criterion with detailed descriptions
class ContextRelevance(BaseModel):
    explanation: str = Field(
        description=(
            "Step-by-step reasoning explaining how the retrieved context aligns with the user's query. "
            "Consider the relevance of the information to the query's intent and the appropriateness of the context "
            "in providing a coherent and useful response. "
            "The reasoning should be concise and not more than 2 sentences."
        )
    )
    score: int = Field(description=SCORE_DESCRIPTION)

class AnswerRelevance(BaseModel):
    explanation: str = Field(
        description=(
            "Step-by-step reasoning explaining how well the generated answer addresses the user's original query. "
            "Consider the helpfulness and on-point nature of the answer, aligning with the user's intent and providing valuable insights. "
            "The reasoning should be concise and not more than 2 sentences."
        )
    )
    score: int = Field(description=SCORE_DESCRIPTION)

class Groundedness(BaseModel):
    explanation: str = Field(
        description=(
            "Step-by-step reasoning explaining how faithful the generated answer is to the retrieved context. "
            "Consider the factual accuracy and reliability of the answer, ensuring it is grounded in the retrieved information. "
            "The reasoning should be concise and not more than 2 sentences."
        )
    )
    score: int = Field(description=SCORE_DESCRIPTION)

# # Print the evaluation
# print("🏆 RAG Evaluation:")
# print("\nCriteria: Context Relevance")
# print(f"Reasoning: {evaluation.context_relevance.explanation}")
# print(f"Score: {evaluation.context_relevance.score.value}/3")

# print("\nCriteria: Answer Relevance")
# print(f"Reasoning: {evaluation.answer_relevance.explanation}")
# print(f"Score: {evaluation.answer_relevance.score.value}/3")

# print("\nCriteria: Groundedness")
# print(f"Reasoning: {evaluation.groundedness.explanation}")
# print(f"Score: {evaluation.groundedness.score.value}/3")