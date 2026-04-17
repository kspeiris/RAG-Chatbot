from __future__ import annotations

import traceback
from uuid import uuid4

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
    st.session_state.setdefault("session_id", uuid4().hex[:12])
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("evaluation_runs", [])
    st.session_state.setdefault("use_source_scope", False)
    st.session_state.setdefault("selected_sources", [])


init_state()

st.title("Reliable Document Chatbot")
st.caption(
    "Upload PDF, scanned PDF, images, TXT, MD, DOCX, HTML, JSON, CSV, TSV, or XLSX files. "
    "The assistant answers from uploaded evidence and can run grounded tabular analytics with pandas/SQLite."
)

session_settings = settings.with_session(st.session_state.session_id)

vector_store = VectorStore(path=str(session_settings.qdrant_path), collection_name=session_settings.scoped_collection_name)
csv_registry = CSVRegistry(session_settings)
evaluation_store = EvaluationStore(session_settings)
indexed_docs = vector_store.list_documents() if session_settings.qdrant_path.exists() else []
indexed_csvs = csv_registry.list_datasets()
all_sources = sorted(set(indexed_docs + [item["file_name"] for item in indexed_csvs]))

with st.sidebar:
    st.header("Setup")
    st.caption(f"Session: `{st.session_state.session_id}`")
    st.write("The app can run fully locally with no API key. OpenAI is only needed if you explicitly set `ANSWER_PROVIDER=openai`.")
    st.write("Indexed storage uses local Qdrant mode plus local SQLite for tabular analytics. OCR is used when available for scanned PDFs and images.")
    if st.button("Start new session", use_container_width=True):
        st.session_state.session_id = uuid4().hex[:12]
        st.session_state.messages = []
        st.session_state.last_result = None
        st.session_state.evaluation_runs = []
        st.session_state.selected_sources = []
        st.rerun()
    st.divider()
    st.subheader("Indexed knowledge")
    st.write(f"Documents in vector store: `{len(indexed_docs)}`")
    st.write(f"Tabular datasets with structured analytics: `{len(indexed_csvs)}`")
    default_sources = [source for source in st.session_state.selected_sources if source in all_sources] or all_sources
    with st.expander("Advanced source scope"):
        use_source_scope = st.checkbox(
            "Limit queries to selected files",
            value=st.session_state.use_source_scope,
            help="By default the whole current session is queried. Enable this only when you want to debug or narrow retrieval.",
        )
        selected_sources = st.multiselect(
            "Selected files",
            options=all_sources,
            default=default_sources,
            help="These files are used only when advanced source scoping is enabled.",
        )
    st.session_state.use_source_scope = use_source_scope
    st.session_state.selected_sources = selected_sources
    if st.button("Clear indexed knowledge", use_container_width=True):
        vector_store.clear_collection()
        csv_registry.clear()
        st.session_state.messages = []
        st.session_state.last_result = None
        st.session_state.use_source_scope = False
        st.session_state.selected_sources = []
        st.success("Indexed documents and tabular datasets were cleared.")
        st.rerun()
    if indexed_csvs:
        with st.expander("Tabular datasets"):
            for dataset in indexed_csvs:
                st.markdown(f"- **{dataset['file_name']}** - {dataset['row_count']} rows")
    with st.expander("Advanced diagnostics"):
        st.write(f"Strict grounded mode: `{session_settings.strict_grounded_mode}`")
        st.write(f"Answer provider: `{session_settings.answer_provider}`")
        st.write(f"Ollama base URL: `{session_settings.ollama_base_url}`")
        st.write(f"Ollama chat model: `{session_settings.ollama_chat_model}`")
        st.write(f"Embedding provider: `{session_settings.embedding_provider}`")
        st.write(
            f"Embedding model: `{session_settings.local_embedding_model if session_settings.embedding_provider == 'local' else session_settings.openai_embedding_model}`"
        )
        st.write(f"Session collection: `{session_settings.scoped_collection_name}`")
        st.write(f"Session vector path: `{session_settings.qdrant_path}`")
        st.write(f"Session data path: `{session_settings.data_dir}`")
        st.write(f"Top context chunks: `{session_settings.max_context_chunks}`")
        st.write(f"Vector candidates: `{session_settings.rerank_candidates}`")
        st.write(f"BM25 candidates: `{session_settings.bm25_candidates}`")
        st.write(f"Adjacent chunk window: `{session_settings.adjacent_chunk_window}`")
        st.write(f"Cross-encoder reranker: `{session_settings.enable_cross_encoder_reranker}`")
        st.write(f"Reranker model: `{session_settings.reranker_model_name}`")
        st.write(f"Chunk size / overlap: `{session_settings.chunk_size}` / `{session_settings.chunk_overlap}`")
        st.write(f"Minimum chunk chars: `{session_settings.min_chunk_chars}`")
        st.write(f"Minimum similarity: `{session_settings.min_similarity_score}`")
        st.write(f"OCR enabled: `{session_settings.ocr_enabled}`")
        st.write(f"OCR language: `{session_settings.ocr_language}`")
        st.write(f"Max OCR PDF pages: `{session_settings.ocr_max_pdf_pages}`")
        st.write(f"Max SQL result rows: `{session_settings.max_sql_result_rows}`")

status_cols = st.columns(4)
status_cols[0].metric("Session", st.session_state.session_id)
status_cols[1].metric("Documents", len(indexed_docs))
status_cols[2].metric("Tables", len(indexed_csvs))
scope_label = "All session files"
if st.session_state.use_source_scope and st.session_state.selected_sources:
    scope_label = f"{len(st.session_state.selected_sources)} selected"
status_cols[3].metric("Scope", scope_label)

if all_sources:
    with st.expander("Session status", expanded=False):
        st.write("Files available in this session:")
        for source in all_sources:
            st.markdown(f"- {source}")
else:
    st.info("This session is empty. Upload one or more files to start.")

st.divider()

st.subheader("1) Upload knowledge files")
replace_existing = st.checkbox("Replace previously indexed knowledge", value=False)
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
            ingestor = IngestionService(session_settings)
            if replace_existing:
                vector_store.clear_collection()
                csv_registry.clear()
                st.session_state.messages = []
                st.session_state.last_result = None
                st.session_state.use_source_scope = False
            results = []
            for file in uploaded_files:
                result = ingestor.ingest_bytes(file.name, file.getvalue())
                results.append(result)
            st.session_state.selected_sources = [result.file_name for result in results]
            st.success("Indexing complete.")
            for result in results:
                st.write(f"- {result.file_name}: {result.chunks_created} chunks")
        except Exception as exc:
            st.error(f"Indexing failed: {exc}")
            st.code(traceback.format_exc())

st.divider()

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
            chat = ChatService(session_settings)
            source_scope = (
                set(st.session_state.selected_sources)
                if st.session_state.use_source_scope and st.session_state.selected_sources
                else None
            )
            result = chat.answer_question(question.strip(), allowed_sources=source_scope)
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
            route_type = result.debug.get("router", {}).get("route_type", result.debug.get("question_scope", "unknown"))
            diag_cards = st.columns(3)
            diag_cards[0].metric("Route", route_type)
            diag_cards[1].metric("Confidence", badge_for_confidence(result.confidence))
            diag_cards[2].metric("Status", grounded_label(result))
            st.markdown("**Evidence summary**")
            st.write(f"Citations: `{len(result.citations)}`")
            if result.debug.get("source_scope"):
                scope_value = result.debug.get("source_scope")
                if isinstance(scope_value, list):
                    st.write(f"Scope: `{len(scope_value)}` selected file(s)")
                else:
                    st.write(f"Scope: `{scope_value}`")
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

            with st.expander("Advanced debug"):
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
                chat = ChatService(session_settings)
                progress = st.progress(0.0)
                for idx, case in enumerate(cases, start=1):
                    source_scope = (
                        set(st.session_state.selected_sources)
                        if st.session_state.use_source_scope and st.session_state.selected_sources
                        else None
                    )
                    result = chat.answer_question(str(case.get("question", "")).strip(), allowed_sources=source_scope)
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
