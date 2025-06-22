from data_schemas import (
    GradeAnswer,
    GradeDocuments,
    GradeHallucinations,
    PreProcessedQuery,
    MultiQueryList,
    ContextRelevance,
    AnswerRelevance,
    Groundedness,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging

logger = logging.getLogger("root")
logger.info("Imported Agent Chains module")


def get_retrieval_grader(llm):
    """
    Creates a grader to assess the relevance of retrieved documents to a user question.

    Args:
        llm: Language model instance

    Returns:
        Chain that grades document relevance with yes/no output
    """
    # Configure LLM for structured output
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    # Define grading criteria in system prompt
    system_prompt = """
    You are a grader assessing relevance of a retrieved document to a user question.
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
    """

    # Create prompt template
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )

    return grade_prompt | structured_llm_grader


def get_answer_grader(llm):
    """
    Creates a grader to assess whether an answer properly addresses a question.

    Args:
        llm: Language model instance

    Returns:
        Chain that grades answer completeness with yes/no output
    """
    structured_llm_grader = llm.with_structured_output(GradeAnswer)

    system_prompt = """
    You are a grader assessing whether an answer addresses / resolves a question.
    Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question.
    """

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ]
    )

    return answer_prompt | structured_llm_grader


def get_response_generator(llm):
    """
    Creates a chain for generating answers based on retrieved context.

    Args:
        llm: Language model instance

    Returns:
        Chain that generates answers based on context
    """
    system_prompt = """
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use all of the necessary details that are available in the context and do not create any information on your own.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Question: \n\n {question} \n\n Retrieved Context: {context}"),
        ]
    )

    return prompt | llm | StrOutputParser()


def get_hallucination_grader(llm):
    """
    Creates a grader to assess whether generated answers are grounded in retrieved facts.

    Args:
        llm: Language model instance

    Returns:
        Chain that grades hallucination with yes/no output
    """
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)

    system_prompt = """
    You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
    Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.
    """

    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )

    return hallucination_prompt | structured_llm_grader


def get_question_rewriter(llm):
    """
    Creates a chain for improving questions for better vector store retrieval.

    Args:
        llm: Language model instance

    Returns:
        Chain that rewrites questions for optimal retrieval
    """
    system_prompt = """
You are a question rewriter. Your job is to take a user's original question and rewrite it into a clearer, more focused version that is optimized for retrieval in a vector-based search system (e.g., a vectorstore).
You should analyze the underlying semantic intent of the original question and rephrase it to better reflect the core meaning, using more concrete or specific terms where appropriate.
Avoid overly broad, vague, or ambiguous language. The rewritten question should preserve the user's intent while improving clarity and relevance for retrieval.

Instructions:
    Focus on making the question more semantically rich and specific.
    Avoid changing the meaning of the question.
    Aim for clearer phrasing, disambiguation, and better alignment with how knowledge is stored in documents.
    Do not include explanations—just return the rewritten question.

Examples:
1. Original: How does AI help businesses?
Rewritten: What are the main ways artificial intelligence improves business operations and decision-making?
2. Original: What did Einstein say about time?
Rewritten: What were Albert Einstein’s key theories or statements about the nature of time?
3. Original: Tell me about Tesla Inc.
Rewritten: What are the key facts about Tesla Inc., including its history, mission, and major products?
5. Original: How do I get better at interviews?
Rewritten: What are effective strategies and techniques for improving performance in job interviews?
    """

    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Original: \n\n {question} \n"),
        ]
    )

    return re_write_prompt | llm | StrOutputParser()


def get_multi_query_translator(llm):
    """
    Creates a chain for improving questions for better vector store retrieval.

    Args:
        llm: Language model instance

    Returns:
        Chain that rewrites questions for optimal retrieval
    """

    structured_llm_translator = llm.with_structured_output(MultiQueryList)

    system_prompt = """
You are a helpful assistant tasked with simplifying complex natural language queries into multiple, standalone sub-questions. These sub-questions will be used as input for a search or retrieval system.
Your goal is to break down each complex query into simpler components that can be answered independently of each other.
Always return a list of clearly written question strings. You may generate as many sub-questions as necessary to fully decompose the original query.

Instructions:
Focus on isolating distinct facts, comparisons, or concepts within the original query.
Ensure that each sub-question is self-contained and does not rely on context from the original query or from other sub-questions.
The resulting questions should be factual and straightforward, optimized for retrieval purposes.

Examples
1. Query: Did Microsoft or Google make more money last year?
   Decomposed Questions: ['How much profit did Microsoft make last year?', 'How much profit did Google make last year?']
2. Query: What is the difference between A, B and C?
   Decomposed Questions: ['What is A', 'What is B', What is C]
    """

    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Query: \n\n {question} \n"),
        ]
    )

    return re_write_prompt | structured_llm_translator | StrOutputParser()


def get_question_pre_processor(llm):
    """
    Creates a chain for pre processing the query

    Args:
        llm: Language model instance

    Returns:
        Chain that rewrites questions for optimal retrieval
    """

    structured_llm_preprocessor = llm.with_structured_output(PreProcessedQuery)

    system_prompt = """
    You are a language expert tasked with refining user queries for a retrieval system.
Your job is to correct any grammatical, syntactic, or logical issues in the query without altering the intended meaning or introducing new information.
Do not paraphrase unnecessarily. Retain original keywords, entities, and phrasing as much as possible, unless clarity requires a small change.
Return only the improved query.
    """

    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Here is the initial question: \n\n {question} \n\n"),
        ]
    )

    return re_write_prompt | structured_llm_preprocessor | StrOutputParser()


# EVALUATOR TRIAD


def get_context_relevance_evaluator(llm):
    structured_llm_grader = llm.with_structured_output(ContextRelevance)

    system_prompt = """
You are an EXPERT SEARCH RESULT RATER. You are given a USER QUERY and a SEARCH RESULT.
Your task is to rate the search result based on its relevance to the user query. You should rate the search result on a scale of 0 to 3, where:

0: The search result has no relevance to the user query.

1: The search result has low relevance to the user query. It may contain some information that is very slightly related to the user query but not enough to answer it. The search result contains some references or very limited information about some entities present in the user query. In case the query is a statement on a topic, the search result should be tangentially related to it.

2: The search result has medium relevance to the user query. If the user query is a question, the search result may contain some information that is relevant to the user query but not enough to answer it. If the user query is a search phrase/sentence, either the search result is centered around most but not all entities present in the user query, or if all the entities are present in the result, the search result while not being centered around it has medium level of relevance. In case the query is a statement on a topic, the search result should be related to the topic.

3: The search result has high relevance to the user query. If the user query is a question, the search result contains information that can answer the user query. Otherwise, if the search query is a search phrase/sentence, it provides relevant information about all entities that are present in the user query and the search result is centered around the entities mentioned in the query. In case the query is a statement on a topic, the search result should be either directly addressing it or be on the same topic.

You should think step by step about the user query and the search result and rate the search result. Be critical and strict with your ratings to ensure accuracy.

Think step by step about the user query and the search result and rate the search result. Provide a reasoning for your rating.
For context relevance, we provide qualitative descriptions for each possible score. Additionally, we provide additional guidance for the LLM to be “critical and strict” with ratings to avoid inflated relevance scores.
    """

    context_relevance_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                """
Now given the USER QUERY and SEARCH RESULT below, rate the search result based on its relevance to the user query and provide a reasoning for your rating.
USER QUERY: {question}
SEARCH RESULT: {documents}
""",
            ),
        ]
    )

    return context_relevance_prompt | structured_llm_grader


def get_answer_relevance_evaluator(llm):
    structured_llm_grader = llm.with_structured_output(AnswerRelevance)

    system_prompt = """
You are a RELEVANCE grader; providing the relevance of the given RESPONSE to the given PROMPT.

- RESPONSE must be relevant to the entire PROMPT to get a maximum score of 3.
- RELEVANCE score should increase as the RESPONSE provides RELEVANT context to more parts of the PROMPT.
- RESPONSE that is RELEVANT to none of the PROMPT should get a minimum score of 0.
- RESPONSE that is RELEVANT and answers the entire PROMPT completely should get a score of 3.
- RESPONSE that is confidently FALSE should get a score of 0.
- RESPONSE that is only seemingly RELEVANT should get a score of 0.
- Answers that intentionally do not answer the question, such as 'I don't know' and model refusals, should also be counted as the least RELEVANT and get a score of 0.
    """

    answer_relevance_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                """
Now given the PROMPT and RESPONSE below, rate the search result based on its relevance to the user query and provide a reasoning for your rating.
PROMPT: {question}
RESPONSE: {generation}
""",
            ),
        ]
    )

    return answer_relevance_prompt | structured_llm_grader


def get_groundedness_evaluator(llm):
    structured_llm_grader = llm.with_structured_output(Groundedness)

    system_prompt = """
You are an INFORMATION OVERLAP classifier; providing the overlap of information (entailment or groundedness) between the source and statement.
    
- Statements that are directly supported by the source should be considered grounded and should get a high score.
- Statements that are not directly supported by the source should be considered not grounded and should get a low score.
- Statements of doubt, that admissions of uncertainty or not knowing the answer are considered abstention, and should be counted as the most overlap and therefore get a max score of 3.
- Consider indirect or implicit evidence, or the context of the statement, to avoid penalizing potentially factual claims due to lack of explicit support.
- Be cautious of false positives; ensure that high scores are only given when there is clear supporting evidence.
- Pay special attention to ensure that indirect evidence is not mistaken for direct support.

Please address all of the below points:
Explanation: <individual claims from LLM response, Identify and describe the location in the source where the information matches the statement. Provide a detailed, human-readable summary indicating the path or key details. if nothing matches, say NOTHING FOUND. For the case where the statement is an abstention, say ABSTENTION>
Score: <Output a number based on the scoring output>
    """

    groundedness_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                """
Source: {documents}
Statement: {generation}
""",
            ),
        ]
    )

    return groundedness_prompt | structured_llm_grader