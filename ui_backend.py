import os
import tomllib
import logging
from dataclasses import dataclass
from typing import List, TypedDict, Any
from pprint import pprint

# --- Local Imports & Initial Setup ---
import rag_logger
from doc_ingest import DocIngest
from docling_loader import DoclingLoader
from agent_chains import (
    get_answer_grader,
    get_hallucination_grader,
    get_question_rewriter,
    get_response_generator,
    get_retrieval_grader,
    get_question_pre_processor,
    get_multi_query_translator,
    get_answer_relevance_evaluator,
    get_context_relevance_evaluator,
    get_groundedness_evaluator,
)

from langgraph.graph import StateGraph, START, END
from langchain_openai import AzureChatOpenAI
from langchain_community.vectorstores import Chroma
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb.config import Settings


# --- Configuration and Logging ---

logger = logging.getLogger("root")

def parse_toml_file(file_path):
    with open(file_path, "rb") as f:
        config = tomllib.load(f)
    return config

config = parse_toml_file("config.toml")
logger.info("Loaded Config File and Imported UIBackend Module")

# --- Graph State Definition ---

@dataclass
class GraphState(TypedDict):
    """State container for the RAG workflow."""
    question: str
    current_questions: List[str]
    documents: List[Any]  # Can be complex nested lists
    generation: str
    failure_counter: int
    generation_counter: int
    original_chunks: List[Any]

# --- Model and Retriever Initialization ---

LLM = AzureChatOpenAI(
    azure_deployment=config["LLM"]["azure_deployment"],
    azure_endpoint=config["LLM"]["azure_endpoint"],
    openai_api_key="dummy",  # required but not used
    openai_api_type=config["LLM"]["openai_api_type"],
    openai_api_version=config["LLM"]["openai_api_version"],
    model=config["LLM"]["model"],
    default_headers={"genaiplatform-farm-subscription-key": config["LLM"]["key"]},
)

EMBEDDING_MODEL_NAME = config["EMBEDDING_MODEL"]["model_path"]
EMBEDDING_MODEL_KWARGS = {'device': 'cuda'}
EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs=EMBEDDING_MODEL_KWARGS,
    show_progress=True
)

RETRIEVER = None
DATABASE = None

# --- Helper Functions ---

def clip_text(text, threshold=100):
    return f"{text[:threshold]}..." if len(text) > threshold else text

def extract_metadata(input_dict):
    result = {}
    result['filename'] = input_dict.get("dl_meta.origin.filename")
    result['page_no'] = input_dict.get('dl_meta.doc_items.0.prov.0.page_no')

    self_ref_entries = []
    for key, value in input_dict.items():
        if key.startswith("dl_meta.doc_items.") and key.endswith(".self_ref"):
            parts = key.split(".")
            if len(parts) >= 4 and parts[2].isdigit():
                index = int(parts[2])
                self_ref_entries.append((index, value))

    self_ref_entries.sort(key=lambda x: x[0])
    result['self_refs'] = [value for _, value in self_ref_entries]
    return result

def get_values_by_filename_and_ref(data_dict, target_filename, target_ref):
    base_filename = target_filename.rsplit('.', 1)[0].replace(' ', '_')
    matched_items = []
    for key, value in data_dict.items():
        parts = key.split('$')
        if len(parts) != 3:
            continue
        try:
            index = int(parts[0])
        except ValueError:
            continue
        filename_part = parts[1]
        refs_part = parts[2].split('_')
        if filename_part == base_filename and target_ref in refs_part:
            matched_items.append((index, target_filename, value))
    matched_items.sort(key=lambda x: x[0])
    return matched_items

# --- Graph Nodes ---

def pre_process_query(state: GraphState) -> GraphState:
    logger.info("---PRE-PROCESS QUERY----")
    question = state["question"]
    question_pre_processor = get_question_pre_processor(LLM)
    better_question = question_pre_processor.invoke({"question": question})
    state["question"] = better_question.pre_processed_query
    state["current_questions"] = [better_question.pre_processed_query]
    return state

def retrieve(state: GraphState) -> GraphState:
    logger.info("---RETRIEVE---")
    current_questions = state["current_questions"]
    documents = []
    for question in current_questions:
        logger.info(f"Retrieving for question: {question}")
        documents.append(RETRIEVER.invoke(question))

    final_docs = []
    for doc_list in documents:
        expanded_docs_for_question = []
        for document in doc_list:
            temp_meta = extract_metadata(document.metadata)
            doc_chunk_map = {}
            for self_ref in temp_meta["self_refs"]:
                results = get_values_by_filename_and_ref(DATABASE, temp_meta["filename"], self_ref)
                for index, filename, value in results:
                    doc_chunk_map[(index, filename)] = (index, filename, temp_meta["page_no"], value)
            doc_chunks = [tup for _, tup in sorted(doc_chunk_map.items())]
            expanded_docs_for_question.append(doc_chunks)
        final_docs.append(expanded_docs_for_question)
    
    logger.info(f"Retrieved {len(final_docs)} sets of documents.")
    state["documents"] = final_docs
    return state

def grade_documents(state: GraphState) -> GraphState:
    logger.info("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    current_questions = state["current_questions"]
    documents = state["documents"]
    retrieval_grader = get_retrieval_grader(LLM)

    final_docs = []
    for i, question in enumerate(current_questions):
        for document_group in documents[i]:
            doc_content = "\n-------------------------------------\n".join([content for _, _, _, content in document_group])
            score = retrieval_grader.invoke({"question": question, "document": doc_content})
            if score.binary_score == "yes":
                logger.info("---GRADE: DOCUMENT RELEVANT---")
                final_docs.append(document_group)
            else:
                logger.warning("---GRADE: DOCUMENT NOT RELEVANT---")

    state["documents"] = final_docs
    state["generation_counter"] = 0
    return state

def generate(state: GraphState) -> GraphState:
    logger.info("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    doc_chunk_map = {}
    for item in documents:
        for index, filename, page_no, value in item:
            doc_chunk_map[(index, filename)] = (index, filename, page_no, value)

    doc_chunks = [tup for _, tup in sorted(doc_chunk_map.items())]
    pprint(doc_chunks)

    rag_chain = get_response_generator(LLM)
    generation = rag_chain.invoke({"context": doc_chunks, "question": question})

    separator = "\n\n---------------------------\n\n"
    source_info = "Sources:\n"
    for idx, (index, filename, page_no, value) in enumerate(doc_chunks, start=1):
        source_info += f"\nSource {idx}:\nFilename: {filename}\nPage No: {page_no}\nText: {clip_text(value)}"
    
    result = generation + separator + f"\n\n{source_info}"

    state["documents"] = doc_chunks
    state["generation"] = result
    state["generation_counter"] += 1
    return state

def evaluate(state: GraphState) -> GraphState:
    logger.info("---EVALUATE---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    context_relevance = get_context_relevance_evaluator(LLM).invoke({"question": question, "documents": documents})
    answer_relevance = get_answer_relevance_evaluator(LLM).invoke({"question": question, "generation": generation})
    groundedness = get_groundedness_evaluator(LLM).invoke({"documents": documents, "generation": generation})

    eval_results = [
        f"Context Relevance : {context_relevance.score}/3 : {context_relevance.explanation}",
        f"Answer Relevance : {answer_relevance.score}/3 : {answer_relevance.explanation}",
        f"Groundedness : {groundedness.score}/3 : {groundedness.explanation}",
    ]
    separator = "\n\n---------------------------\n\n"
    result = generation + separator + "\n\n".join(eval_results)

    state["generation"] = result
    state["generation_counter"] += 1
    return state

def increment_failure_counter(state: GraphState) -> GraphState:
    logger.error("---INCREMENT FAILURE COUNTER---")
    state["failure_counter"] += 1
    return state

def answer_not_found_in_context(state: GraphState) -> GraphState:
    logger.error("---ANSWER NOT FOUND IN CONTEXT---")
    state["generation"] = "Answer was not found in context. Try to rephrase the question or add more details."
    state["failure_counter"] += 1
    return state

def not_supported_limit(state: GraphState) -> GraphState:
    logger.error("---NOT SUPPORTED LIMIT REACHED---")
    state["generation"] = "Answer was not grounded in context. Try to rephrase the question or add more details."
    state["failure_counter"] += 1
    return state

def transform_query_rewrite(state: GraphState) -> GraphState:
    logger.info("---TRANSFORM QUERY REWRITE---")
    question = state["question"]
    question_rewriter = get_question_rewriter(LLM)
    better_question = question_rewriter.invoke({"question": question})
    state["current_questions"] = [better_question]
    return state

def transform_query_MQT(state: GraphState) -> GraphState:
    logger.info("---TRANSFORM QUERY MQT---")
    question = state["question"]
    question_rewriter = get_multi_query_translator(LLM)
    better_question = question_rewriter.invoke({"question": question})
    state["current_questions"] = better_question.multi_query_list
    return state

# --- Graph Edges ---

def decide_failure_path(state: GraphState) -> str:
    logger.warning("---DECIDE FAILURE PATH---")
    failure_counter = state["failure_counter"]
    if failure_counter == 1:
        logger.info("---DECISION: TRANSFORM QUERY REWRITE---")
        return "transform_query_rewrite"
    elif failure_counter == 2:
        logger.info("---DECISION: TRANSFORM QUERY MQT---")
        return "transform_query_MQT"
    else:
        return "answer_not_found_in_context"

def decide_to_generate(state: GraphState) -> str:
    logger.info("---ASSESS GRADED DOCUMENTS---")
    if not state["documents"]:
        logger.error("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT, TRANSFORM QUERY---")
        return "not_useful"
    else:
        logger.info("---DECISION: GENERATE---")
        return "generate"

def grade_generation_v_documents_and_question(state: GraphState) -> str:
    logger.info("---CHECK HALLUCINATIONS AND ANSWER RELEVANCE---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    hallucination_grader = get_hallucination_grader(LLM)
    answer_grader = get_answer_grader(LLM)

    hallucination_score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    if hallucination_score.binary_score == "yes":
        logger.info("---DECISION: GENERATION IS GROUNDED---")
        answer_score = answer_grader.invoke({"question": question, "generation": generation})
        if answer_score.binary_score == "yes":
            logger.info("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            logger.error("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not_useful"
    else:
        logger.error("---DECISION: GENERATION IS NOT GROUNDED, RE-TRYING---")
        if state["generation_counter"] >= 3:
            return "not_supported_limit"
        return "not_supported"

# --- Graph Builder ---

def get_graph():
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("pre_process_query", pre_process_query)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("evaluate", evaluate)
    workflow.add_node("transform_query_rewrite", transform_query_rewrite)
    workflow.add_node("transform_query_MQT", transform_query_MQT)
    workflow.add_node("increment_failure_counter", increment_failure_counter)
    workflow.add_node("answer_not_found_in_context", answer_not_found_in_context)
    workflow.add_node("not_supported_limit", not_supported_limit)

    # Build graph edges
    workflow.add_edge(START, "pre_process_query")
    workflow.add_edge("pre_process_query", "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges("grade_documents", decide_to_generate, {"not_useful": "increment_failure_counter", "generate": "generate"})
    workflow.add_conditional_edges("increment_failure_counter", decide_failure_path, {"transform_query_rewrite": "transform_query_rewrite", "transform_query_MQT": "transform_query_MQT", "answer_not_found_in_context": "answer_not_found_in_context"})
    workflow.add_edge("transform_query_rewrite", "retrieve")
    workflow.add_edge("transform_query_MQT", "retrieve")
    workflow.add_conditional_edges("generate", grade_generation_v_documents_and_question, {"not_supported": "generate", "not_supported_limit": "not_supported_limit", "useful": "evaluate", "not_useful": "increment_failure_counter"})
    
    workflow.add_edge("answer_not_found_in_context", END)
    workflow.add_edge("not_supported_limit", END)
    workflow.add_edge("evaluate", END)

    return workflow.compile()

# --- Main Application Class ---

class UIBackend:
    def __init__(self, directory_name='data_store'):
        self.directory_name = self._setup_directory(directory_name)
        self.master_list = {}
        self.load_json_files()
        self.init_app()

    @staticmethod
    def _setup_directory(directory_name):
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
            logger.info(f"Directory '{directory_name}' created.")
        else:
            logger.info(f"Directory '{directory_name}' already exists.")
        return directory_name

    def load_json_files(self):
        file_list = os.listdir(self.directory_name)
        if not file_list:
            logger.info(f"Directory '{self.directory_name}' is empty")
            return

        for index, filename in enumerate(file_list):
            if filename.endswith('.json'):
                logger.info(f"Loading {filename}")
                file_path = os.path.relpath(os.path.join(self.directory_name, filename))
                temp_chroma_path = os.path.relpath(f"./chroma_vectorstore/{filename[9:-5]}")
                try:
                    if not os.path.isdir(temp_chroma_path):
                        document = DocIngest(file_path)
                        document.load_file()
                        self.create_vector_store(document, temp_chroma_path)
                    else:
                        logger.info(f"Vector store for '{filename}' already exists at '{temp_chroma_path}'.")
                except Exception as e:
                    logging.exception(f"Error processing file {filename}: {e}")
                    raise

                if os.path.isfile(file_path) and os.path.isdir(temp_chroma_path):
                    self.master_list[index] = {"document": file_path, "chroma": temp_chroma_path}

    def add_pdf_file(self, file_path):
        if not os.path.isfile(file_path) or not file_path.endswith('.pdf'):
            logger.error("Provided path is not a valid PDF file.")
            return False

        pdf_file_name = os.path.basename(file_path)
        base_name = os.path.splitext(pdf_file_name)[0].replace(' ', '_')
        json_file_name = f"document_{base_name}.json"
        json_file_path = os.path.relpath(os.path.join(self.directory_name, json_file_name))
        temp_chroma_path = os.path.relpath(f"./chroma_vectorstore/{base_name}")

        if os.path.isdir(temp_chroma_path):
            logger.warning(f"Vector store for '{pdf_file_name}' already exists.")
            return False

        try:
            document = DocIngest(file_path)
            document.load_file(output_path=self.directory_name)
            self.create_vector_store(document, temp_chroma_path)
        except Exception as e:
            logging.exception(f"Error parsing PDF from file: {file_path}: {e}")
            return False

        if not os.path.isfile(json_file_path) or not os.path.isdir(temp_chroma_path):
            logger.error("Failed to create necessary JSON or ChromaDB files.")
            return False

        index = len(self.master_list)
        self.master_list[index] = {"document": json_file_path, "chroma": temp_chroma_path}
        return True

    @staticmethod
    def create_vector_store(document, dir_path):
        dir_path = os.path.abspath(dir_path)
        loader = DoclingLoader(conv_res_list=[document], embed_model_id=EMBEDDING_MODEL_NAME)
        docs = loader.load()

        def flatten_dict(d, parent_key='', sep='.'):
            items = []
            if isinstance(d, dict):
                for k, v in d.items():
                    new_key = f"{parent_key}{sep}{k}" if parent_key else k
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(d, list):
                for i, v in enumerate(d):
                    new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((parent_key, d))
            return dict(items)

        self_refs_list = ["_".join([item["self_ref"] for item in doc.metadata["dl_meta"]["doc_items"]]) for doc in docs]
        ids = [f"{i}${document.result.name.replace(' ', '_')}${self_refs_list[i]}" for i in range(len(docs))]

        for doc in docs:
            doc.metadata = flatten_dict(doc.metadata)

        Chroma.from_documents(
            documents=docs,
            persist_directory=dir_path,
            collection_name=document.result.name.replace(" ", "_"),
            embedding=EMBEDDING_MODEL,
            client_settings=Settings(anonymized_telemetry=False, is_persistent=True),
            ids=ids
        )
        logger.info(f"Loaded vector store for {document.result.name} at {dir_path}")
        return True

    @staticmethod
    def get_display_list(master_list):
        def get_display_name(file_path):
            file_name = os.path.basename(file_path)
            if file_name.startswith("document_") and file_name.endswith(".json"):
                base_name = file_name[9:-5]
                return f"{base_name}.pdf"
            elif file_name.endswith('.pdf'):
                return file_name
            else:
                return "Unsupported file type"
        return {k: get_display_name(v["document"]) for k, v in master_list.items()}

    @staticmethod
    def list_chroma_collections(chroma_db_path: str):
        try:
            client = chromadb.PersistentClient(path=chroma_db_path)
            collections = client.list_collections()
            if not collections:
                logger.error(f"No collections found at {chroma_db_path}")
                return None, None
            logger.info(f"Collections at {chroma_db_path}: {[c.name for c in collections]}")
            return client, collections[0].name
        except Exception as e:
            logger.error(f"Error accessing ChromaDB at '{chroma_db_path}': {e}")
            return None, None

    def load_docling_documents(self, required_indices):
        vector_paths = [self.master_list.get(i, {}).get("chroma") for i in required_indices if i in self.master_list]
        persistent_clients = [self.list_chroma_collections(path) for path in vector_paths if path]

        vectorstore_list = [
            Chroma(
                client=client,
                collection_name=name,
                embedding_function=EMBEDDING_MODEL
            ) for client, name in persistent_clients if client and name
        ]

        if not vectorstore_list:
            logger.error("No valid vector stores to load.")
            return False
            
        # Merge vector stores
        vectorstore = vectorstore_list[0]
        for vs in vectorstore_list[1:]:
            logger.info(f"Merging collection {vs._collection.name}...")
            vs_data = vs._collection.get(include=['documents', 'metadatas', 'embeddings', 'ids'])
            vectorstore._collection.add(**vs_data)

        global RETRIEVER, DATABASE
        RETRIEVER = vectorstore.as_retriever()
        temp_database = vectorstore.get(include=["metadatas", "documents"])
        DATABASE = dict(zip(temp_database["ids"], temp_database["documents"]))
        self.retriever = RETRIEVER
        self.database = DATABASE
        logger.info("Successfully loaded and merged documents.")
        return True

    def init_app(self):
        self.app = get_graph()

    def invoke_rag(self, question):
        if not self.retriever or not self.database:
            return "Error: No documents have been loaded. Please select documents to query."

        inputs = {
            "question": question,
            "current_questions": [],
            "documents": [],
            "generation": None,
            "failure_counter": 0,
            "generation_counter": 0,
        }
        final_output = None
        for output in self.app.stream(inputs):
            for key, value in output.items():
                logger.info(f"Node '{key}':")
            final_output = value
        
        return final_output.get("generation", "No generation produced.")


if __name__ == "__main__":
    # Example usage (commented out)
    # backend = UIBackend()
    # print("-------- Initialized Backend Class")
    # input_path = r"path\to\your\document.pdf"
    # print("-------- Adding PDF")
    # backend.add_pdf_file(input_path)
    # print(backend.master_list)
    #
    # print("-------- Loading PDF")
    # backend.load_docling_documents([0]) # Assuming the added file is at index 0
    # print("-------- Asking Question")
    # pprint(backend.invoke_rag("What is the main topic of the document?"))
    pass