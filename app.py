from __future__ import annotations

import traceback

import streamlit as st

from src.chat import ChatService
from src.config import settings
from src.csv_query import CSVRegistry
from src.evaluation import (
    EvaluationCase,
    EvaluationStore,
    badge_for_confidence,
    evaluate_answer,
    grounded_label,
    summarize_runs,
)
from src.ingest import IngestionService
from src.llm import LocalLLMUnavailableError, QuotaExceededError
from src.storage import VectorStore

st.set_page_config(page_title="Reliable RAG Chatbot", page_icon="📚", layout="wide")


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("evaluation_runs", [])


init_state()

st.title("Reliable Document Chatbot")
st.caption(
    "Upload PDF, scanned PDF, images, TXT, MD, DOCX, HTML, JSON, CSV, TSV, or XLSX files. "
    "The assistant answers from uploaded evidence and can run grounded tabular analytics with pandas/SQLite."
)

vector_store = VectorStore(path=str(settings.qdrant_path), collection_name=settings.collection_name)
csv_registry = CSVRegistry(settings)
evaluation_store = EvaluationStore(settings)
indexed_docs = vector_store.list_documents() if settings.qdrant_path.exists() else []
indexed_csvs = csv_registry.list_datasets()

with st.sidebar:
    st.header("Setup")
    st.write("The app can run fully locally with no API key. OpenAI is only needed if you explicitly set `ANSWER_PROVIDER=openai`.")
    st.write("Indexed storage uses local Qdrant mode plus local SQLite for tabular analytics. OCR is used when available for scanned PDFs and images.")
    st.divider()
    st.subheader("Reliability settings")
    st.write(f"Strict grounded mode: `{settings.strict_grounded_mode}`")
    st.write(f"Answer provider: `{settings.answer_provider}`")
    st.write(f"Ollama base URL: `{settings.ollama_base_url}`")
    st.write(f"Ollama chat model: `{settings.ollama_chat_model}`")
    st.write(f"Embedding provider: `{settings.embedding_provider}`")
    st.write(f"Embedding model: `{settings.local_embedding_model if settings.embedding_provider == 'local' else settings.openai_embedding_model}`")
    st.write(f"Top context chunks: `{settings.max_context_chunks}`")
    st.write(f"Vector candidates: `{settings.rerank_candidates}`")
    st.write(f"BM25 candidates: `{settings.bm25_candidates}`")
    st.write(f"Adjacent chunk window: `{settings.adjacent_chunk_window}`")
    st.write(f"Cross-encoder reranker: `{settings.enable_cross_encoder_reranker}`")
    st.write(f"Reranker model: `{settings.reranker_model_name}`")
    st.write(f"Chunk size / overlap: `{settings.chunk_size}` / `{settings.chunk_overlap}`")
    st.write(f"Minimum chunk chars: `{settings.min_chunk_chars}`")
    st.write(f"Minimum similarity: `{settings.min_similarity_score}`")
    st.write(f"OCR enabled: `{settings.ocr_enabled}`")
    st.write(f"OCR language: `{settings.ocr_language}`")
    st.write(f"Max OCR PDF pages: `{settings.ocr_max_pdf_pages}`")
    st.write(f"Max SQL result rows: `{settings.max_sql_result_rows}`")
    st.divider()
    st.subheader("Indexed knowledge")
    st.write(f"Documents in vector store: `{len(indexed_docs)}`")
    st.write(f"Tabular datasets with structured analytics: `{len(indexed_csvs)}`")
    if indexed_csvs:
        with st.expander("Tabular datasets"):
            for dataset in indexed_csvs:
                st.markdown(f"- **{dataset['file_name']}** - {dataset['row_count']} rows")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1) Upload knowledge files")
    uploaded_files = st.file_uploader(
        "Upload one or more files",
        type=["pdf", "txt", "md", "csv", "tsv", "docx", "xlsx", "json", "jsonl", "html", "htm", "xml", "png", "jpg", "jpeg", "webp", "tif", "tiff"],
        accept_multiple_files=True,
    )

    if st.button("Index uploaded files", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one file.")
        else:
            try:
                ingestor = IngestionService(settings)
                results = []
                for file in uploaded_files:
                    result = ingestor.ingest_bytes(file.name, file.getvalue())
                    results.append(result)
                st.success("Indexing complete.")
                for result in results:
                    st.write(f"- {result.file_name}: {result.chunks_created} chunks")
            except Exception as exc:
                st.error(f"Indexing failed: {exc}")
                st.code(traceback.format_exc())

with col2:
    st.subheader("2) Ask questions")
    question = st.text_area(
        "Ask a grounded question",
        placeholder="Examples: What does the handbook say about minimum attendance? | Which students are below 75% attendance? | Count orders per region. | What text appears in this scanned notice?",
        height=120,
    )
    if st.button("Get answer", use_container_width=True):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            try:
                chat = ChatService(settings)
                result = chat.answer_question(question.strip())
                st.session_state.last_result = result
                st.session_state.messages.append({"role": "user", "content": question.strip()})
                st.session_state.messages.append({"role": "assistant", "content": result.answer})
            except QuotaExceededError as exc:
                st.error(str(exc))
            except LocalLLMUnavailableError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Answer generation failed: {exc}")
                st.code(traceback.format_exc())

st.divider()

tab_chat, tab_eval = st.tabs(["Chat", "Evaluation"])

with tab_chat:
    left, right = st.columns([1.3, 0.7])
    with left:
        st.subheader("Conversation")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    with right:
        st.subheader("Answer diagnostics")
        result = st.session_state.last_result
        if result is None:
            st.info("Your latest grounded answer will appear here.")
        else:
            st.write(f"Confidence: {badge_for_confidence(result.confidence)}")
            st.write(f"Status: **{grounded_label(result)}**")
            st.markdown("**Citations**")
            if result.citations:
                for cite in result.citations:
                    file_name = cite.get("file_name", "unknown")
                    locator = cite.get("locator", "n/a")
                    quote = cite.get("quote", "")
                    st.markdown(f"- **{file_name}** - {locator}")
                    if quote:
                        st.caption(quote)
            else:
                st.write("No citation objects returned.")

            with st.expander("Debug details"):
                st.json(result.debug)

with tab_eval:
    st.subheader("Evaluation workspace")
    st.caption("Save benchmark questions, run them against the current indexed knowledge, and track whether answers stay grounded with the expected evidence.")

    with st.form("evaluation_case_form", clear_on_submit=True):
        eval_question = st.text_area("Question", height=100, placeholder="What should the assistant answer reliably?")
        eval_expected_answer = st.text_area("Expected answer or gold summary", height=100, placeholder="Optional reference answer for overlap scoring.")
        eval_required_terms = st.text_input("Required terms", placeholder="attendance, minimum, 75%")
        eval_expected_files = st.text_input("Expected cited files", placeholder="handbook.pdf, attendance.csv")
        eval_notes = st.text_input("Notes", placeholder="Optional scenario notes")
        add_case = st.form_submit_button("Add evaluation case", use_container_width=True)
        if add_case:
            if not eval_question.strip():
                st.warning("Question is required.")
            else:
                evaluation_store.add_case(
                    EvaluationCase(
                        question=eval_question,
                        expected_answer=eval_expected_answer,
                        required_terms=[item.strip() for item in eval_required_terms.split(",") if item.strip()],
                        expected_files=[item.strip() for item in eval_expected_files.split(",") if item.strip()],
                        notes=eval_notes,
                    )
                )
                st.success("Evaluation case saved.")
                st.rerun()

    cases = evaluation_store.list_cases()
    st.write(f"Saved evaluation cases: `{len(cases)}`")

    if not cases:
        st.info("Add a few benchmark questions here to start measuring retrieval and grounded answer quality.")
    else:
        run_col, clear_col = st.columns([1, 1])
        with run_col:
            if st.button("Run all evaluation cases", type="primary", use_container_width=True):
                runs: list[dict] = []
                chat = ChatService(settings)
                progress = st.progress(0.0)
                for idx, case in enumerate(cases, start=1):
                    result = chat.answer_question(str(case.get("question", "")).strip())
                    evaluation = evaluate_answer(case, result)
                    runs.append(
                        {
                            "question": case.get("question", ""),
                            "expected_answer": case.get("expected_answer", ""),
                            "notes": case.get("notes", ""),
                            "answer": result.answer,
                            "status": grounded_label(result),
                            **evaluation,
                        }
                    )
                    progress.progress(idx / len(cases))
                st.session_state.evaluation_runs = runs
                st.success("Evaluation run complete.")

        with clear_col:
            if st.button("Clear last evaluation run", use_container_width=True):
                st.session_state.evaluation_runs = []
                st.rerun()

        with st.expander("Saved cases", expanded=True):
            for idx, case in enumerate(cases):
                st.markdown(f"**{idx + 1}. {case.get('question', '')}**")
                if case.get("expected_answer"):
                    st.caption(f"Expected answer: {case.get('expected_answer')}")
                if case.get("required_terms"):
                    st.caption(f"Required terms: {', '.join(case.get('required_terms', []))}")
                if case.get("expected_files"):
                    st.caption(f"Expected files: {', '.join(case.get('expected_files', []))}")
                if case.get("notes"):
                    st.caption(f"Notes: {case.get('notes')}")
                if st.button(f"Delete case {idx + 1}", key=f"delete_eval_case_{idx}", use_container_width=True):
                    evaluation_store.delete_case(idx)
                    st.rerun()

        runs = st.session_state.evaluation_runs
        if runs:
            st.divider()
            summary = summarize_runs(runs)
            metrics = st.columns(4)
            metrics[0].metric("Pass rate", f"{summary['pass_rate']}%")
            metrics[1].metric("Grounded rate", f"{summary['grounded_rate']}%")
            metrics[2].metric("Citation rate", f"{summary['citation_rate']}%")
            metrics[3].metric("Avg overlap", str(summary["average_overlap"]) if summary["average_overlap"] is not None else "n/a")

            st.markdown("**Latest evaluation results**")
            table_rows = [
                {
                    "question": run["question"],
                    "passed": run["passed"],
                    "grounded": run["grounded"],
                    "confidence": run["confidence"],
                    "citation_count": run["citation_count"],
                    "missing_terms": ", ".join(run["missing_terms"]),
                    "missing_files": ", ".join(run["missing_files"]),
                    "answer_overlap": run["answer_overlap"],
                }
                for run in runs
            ]
            st.dataframe(table_rows, use_container_width=True)

            with st.expander("Detailed evaluation outputs"):
                for run in runs:
                    st.markdown(f"**Question:** {run['question']}")
                    st.write(f"Passed: `{run['passed']}`")
                    st.write(f"Grounded: `{run['grounded']}`")
                    st.write(f"Confidence: `{run['confidence']}`")
                    st.write(f"Answer: {run['answer']}")
                    if run["missing_terms"]:
                        st.write(f"Missing terms: {', '.join(run['missing_terms'])}")
                    if run["missing_files"]:
                        st.write(f"Missing expected files: {', '.join(run['missing_files'])}")
                    st.caption(f"Cited files: {', '.join(run['cited_files']) if run['cited_files'] else 'none'}")
                    st.divider()
