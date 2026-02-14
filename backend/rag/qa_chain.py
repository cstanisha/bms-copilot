"""
Main RAG QA Chain with Validation
Integrates retrieval, generation, and validation
"""

from typing import Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from backend.rag.prompts import (
    SYSTEM_PROMPT,
    CONDENSE_QUESTION_PROMPT,
    LOW_CONFIDENCE_RESPONSE,
    NO_INFO_RESPONSE,
)
from backend.rag.memory_manager import ConversationMemoryManager
from backend.validators.hallucination_detector import HallucinationDetector
from backend.vectorstore.pinecone_vector import PineconeManager


class ValidatedQAChain:
    """
    RAG QA Chain with multi-layer validation
    Ensures answers are accurate and source-grounded
    """

    def __init__(
        self,
        pinecone_manager: PineconeManager,
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        top_k: int = 5,
        similarity_threshold: float = 0.3,
        confidence_threshold: float = 0.3,
    ):
        self.pinecone_manager = pinecone_manager
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.confidence_threshold = confidence_threshold

        # LLM
        self.llm = ChatOpenAI(model=llm_model, temperature=temperature)

        # Memory + validation
        self.memory_manager = ConversationMemoryManager(k=5, llm_model=llm_model)
        self.hallucination_detector = HallucinationDetector(min_source_coverage=0.8)

        # Prompt template (used for condensation only)
        self.qa_prompt = PromptTemplate(
            template=SYSTEM_PROMPT,
            input_variables=["context", "chat_history", "question"],
        )

    # ------------------------------------------------------------------
    # QUERY
    # ------------------------------------------------------------------
    def query(
        self,
        question: str,
        namespace: Optional[str] = None,
        metadata_filter: Optional[Dict] = None,
    ) -> Dict:
        """Process a question and return validated answer"""

        # Ensure namespace exists
        if not namespace:
            namespace = "default"

        #print(" Namespace used:", namespace)

        # Step 1 — reformulate question
        reformulated_question = self._reformulate_question(question)

        # Step 2 — retrieve docs
        retrieved_docs = self.pinecone_manager.similarity_search(
            query=reformulated_question,
            k=self.top_k,
            filter=metadata_filter,
            namespace=namespace,
            score_threshold=self.similarity_threshold,
        )

        #print(" Retrieved docs count:", len(retrieved_docs))

        if not retrieved_docs:
            return self._handle_no_information(question)

        # Step 3 — prepare context
        context = self._format_context(retrieved_docs)
        chat_history = self.memory_manager.get_chat_history_string()

        # Step 4 — CORRECT CHAT FORMAT (critical fix)
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(
                    context=context,
                    chat_history=chat_history,
                    question=question,
                ),
            },
            {"role": "user", "content": question},
        ]

        response = self.llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)

        # Step 5 — validate
        validation_result = self._validate_answer(
            answer=answer,
            source_docs=[doc for doc, _ in retrieved_docs],
            question=question,
        )

        # Step 6 — confidence
        retrieval_scores = [score for _, score in retrieved_docs]
        avg_retrieval_score = sum(retrieval_scores) / len(retrieval_scores)

        overall_confidence = self._calculate_confidence(
            validation_result=validation_result,
            retrieval_score=avg_retrieval_score,
        )

        # Step 7 — low confidence handling
        if overall_confidence < self.confidence_threshold:
            return self._handle_low_confidence(
                question=question,
                partial_answer=answer,
                retrieved_docs=retrieved_docs,
                confidence=overall_confidence,
            )

        # Step 8 — save memory
        self.memory_manager.add_exchange(
            question=question,
            answer=answer,
            metadata={"confidence": overall_confidence},
        )

        return {
            "answer": answer,
            "confidence": overall_confidence,
            "confidence_level": self._get_confidence_level(overall_confidence),
            "sources": self._format_sources(retrieved_docs),
            "validation": validation_result,
            "retrieved_docs_count": len(retrieved_docs),
            "reformulated_question": reformulated_question,
        }

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------
    def _reformulate_question(self, question: str) -> str:
        chat_history = self.memory_manager.get_chat_history_string()
        if not chat_history:
            return question

        condense_prompt = PromptTemplate(
            template=CONDENSE_QUESTION_PROMPT,
            input_variables=["chat_history", "question"],
        )

        try:
            prompt_text = condense_prompt.format(
                chat_history=chat_history, question=question
            )
            response = self.llm.invoke(prompt_text)
            return (
                response.content.strip()
                if hasattr(response, "content")
                else str(response).strip()
            )
        except Exception:
            return question

    def _format_context(self, retrieved_docs: List[tuple]) -> str:
        parts = []
        for idx, (doc, score) in enumerate(retrieved_docs, 1):
            meta = doc.metadata
            src = (
                f"[Doc {idx}] Page {meta.get('page','N/A')} | "
                f"Section {meta.get('section','N/A')} | Score {score:.2f}"
            )
            parts.append(f"{src}\n{doc.page_content}\n")
        return "\n---\n".join(parts)

    def _validate_answer(self, answer: str, source_docs: List[Document], question: str) -> Dict:
        return self.hallucination_detector.validate_answer(
            answer=answer, source_documents=source_docs, query=question
        )

    def _calculate_confidence(self, validation_result: Dict, retrieval_score: float) -> float:
        validation_conf = validation_result.get("confidence", 0)
        confidence = 0.6 * validation_conf + 0.4 * retrieval_score

        if validation_result.get("external_knowledge_detected", False):
            confidence *= 0.5
        if not validation_result.get("has_citations", True):
            confidence *= 0.8

        return min(confidence, 1.0)

    def _get_confidence_level(self, confidence: float) -> str:
        if confidence >= 0.85:
            return "HIGH"
        if confidence >= 0.70:
            return "MEDIUM"
        if confidence >= 0.50:
            return "LOW"
        return "INSUFFICIENT"

    def _format_sources(self, retrieved_docs: List[tuple]) -> List[Dict]:
        sources = []
        for doc, score in retrieved_docs:
            meta = doc.metadata
            sources.append(
                {
                    "page": meta.get("page", "N/A"),
                    "section": meta.get("section", "N/A"),
                    "content_type": meta.get("content_type", "general"),
                    "relevance_score": round(score, 3),
                    "snippet": doc.page_content[:200] + "...",
                }
            )
        return sources

    def _handle_no_information(self, question: str) -> Dict:
        response = NO_INFO_RESPONSE.format(query=question, searched_sections="All available sections")
        return {
            "answer": response,
            "confidence": 0.0,
            "confidence_level": "INSUFFICIENT",
            "sources": [],
            "validation": {"valid": False, "reason": "No relevant information found"},
            "retrieved_docs_count": 0,
        }

    def _handle_low_confidence(
        self, question: str, partial_answer: str, retrieved_docs: List[tuple], confidence: float
    ) -> Dict:
        refs = [
            f"Page {doc.metadata.get('page','N/A')}, Section {doc.metadata.get('section','N/A')}"
            for doc, _ in retrieved_docs
        ]

        response = LOW_CONFIDENCE_RESPONSE.format(
            partial_info=partial_answer[:300] + "...",
            source_reference="; ".join(refs[:2]),
        )

        return {
            "answer": response,
            "confidence": confidence,
            "confidence_level": "LOW",
            "sources": self._format_sources(retrieved_docs),
            "validation": {"valid": False, "reason": "Low confidence"},
            "retrieved_docs_count": len(retrieved_docs),
            "partial_answer": partial_answer,
        }

    # ------------------------------------------------------------------
    # MEMORY
    # ------------------------------------------------------------------
    def clear_history(self):
        self.memory_manager.clear_memory()

    def get_conversation_history(self) -> str:
        return self.memory_manager.get_chat_history_string()

