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
from src.workspaces import WorkspaceManager

APP_NAME = "DocAnchor"

st.set_page_config(page_title=APP_NAME, page_icon="D", layout="wide")


CUSTOM_CSS = """
<style>
    :root {
        --bg: #f4f7fb;
        --surface: rgba(255, 255, 255, 0.92);
        --surface-strong: #ffffff;
        --surface-muted: #eef3f8;
        --text: #122033;
        --text-soft: #5b6b80;
        --border: rgba(18, 32, 51, 0.08);
        --primary: #0f6cbd;
        --primary-soft: rgba(15, 108, 189, 0.12);
        --shadow: 0 18px 48px rgba(15, 32, 51, 0.08);
        --radius-lg: 24px;
        --radius-md: 18px;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(15, 108, 189, 0.10), transparent 28%),
            radial-gradient(circle at top right, rgba(31, 132, 90, 0.08), transparent 24%),
            linear-gradient(180deg, #f8fbff 0%, var(--bg) 100%);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2.5rem;
        max-width: 1320px;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1723 0%, #162233 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    [data-testid="stSidebar"] * {
        color: #e8eef7;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stCaption {
        color: #c7d4e5 !important;
    }

    .hero {
        padding: 2rem 2.2rem;
        border-radius: var(--radius-lg);
        background: linear-gradient(135deg, rgba(15, 108, 189, 0.98), rgba(13, 148, 136, 0.92));
        color: white;
        box-shadow: var(--shadow);
        margin-bottom: 1.25rem;
    }

    .hero-kicker {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        opacity: 0.78;
        margin-bottom: 0.65rem;
        font-weight: 700;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.35rem;
        line-height: 1.08;
        letter-spacing: -0.03em;
        color: white;
    }

    .hero p {
        margin: 0.9rem 0 0;
        max-width: 840px;
        color: rgba(255, 255, 255, 0.88);
        font-size: 1rem;
    }

    .section-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        box-shadow: var(--shadow);
        padding: 1.15rem 1.2rem;
        backdrop-filter: blur(12px);
        margin-bottom: 1rem;
    }

    .section-title {
        margin-bottom: 0.7rem;
    }

    .section-title h3 {
        margin: 0;
        color: var(--text);
        font-size: 1.08rem;
        letter-spacing: -0.01em;
    }

    .section-title p {
        margin: 0.3rem 0 0;
        color: var(--text-soft);
        font-size: 0.93rem;
    }

    .stat-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(244,248,253,0.92));
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        box-shadow: 0 14px 28px rgba(15, 32, 51, 0.06);
        padding: 1rem 1.05rem;
        min-height: 120px;
    }

    .stat-label {
        color: var(--text-soft);
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.45rem;
        font-weight: 700;
    }

    .stat-value {
        color: var(--text);
        font-size: 1.65rem;
        font-weight: 700;
        line-height: 1.1;
        letter-spacing: -0.03em;
    }

    .stat-caption {
        color: var(--text-soft);
        margin-top: 0.45rem;
        font-size: 0.9rem;
    }

    .pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.8rem;
    }

    .pill {
        background: var(--primary-soft);
        color: var(--primary);
        border: 1px solid rgba(15, 108, 189, 0.14);
        padding: 0.38rem 0.7rem;
        border-radius: 999px;
        font-size: 0.86rem;
        font-weight: 600;
    }

    .soft-panel {
        background: var(--surface-muted);
        border-radius: var(--radius-md);
        padding: 0.95rem 1rem;
        border: 1px solid rgba(18, 32, 51, 0.06);
    }

    .upload-list {
        margin: 0.75rem 0 0;
        padding-left: 1rem;
        color: var(--text-soft);
    }

    .answer-shell {
        background: linear-gradient(180deg, rgba(255,255,255,1), rgba(247,250,253,0.96));
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 1rem 1.05rem;
    }

    .stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        border-radius: 12px;
        min-height: 2.9rem;
        font-weight: 600;
        border: 1px solid rgba(18, 32, 51, 0.08);
    }

    .stButton > button[kind="primary"],
    div[data-testid="stFormSubmitButton"] > button[kind="primary"] {
        background: linear-gradient(135deg, #0f6cbd 0%, #1270c8 55%, #0d9488 100%);
        color: white;
        border: none;
        box-shadow: 0 12px 24px rgba(15, 108, 189, 0.24);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        margin-bottom: 1rem;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.62);
        border-radius: 999px;
        border: 1px solid rgba(18, 32, 51, 0.08);
        padding: 0.55rem 0.95rem;
    }

    .stTabs [aria-selected="true"] {
        background: white !important;
        box-shadow: 0 8px 20px rgba(15, 32, 51, 0.08);
    }

    @media (max-width: 900px) {
        .hero h1 {
            font-size: 1.85rem;
        }
    }
</style>
"""


def init_state() -> None:
    st.session_state.setdefault("workspace_id", "")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("last_ingest_results", [])
    st.session_state.setdefault("evaluation_runs", [])
    st.session_state.setdefault("new_workspace_name", "")
    st.session_state.setdefault("use_source_scope", False)
    st.session_state.setdefault("selected_sources", [])


def inject_styles() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_hero(workspace_name: str) -> None:
    st.markdown(
        f"""
        <section class="hero">
            <div class="hero-kicker">{APP_NAME}</div>
            <h1>Professional document Q&amp;A with grounded evidence and tabular analysis.</h1>
            <p>
                Upload mixed-format knowledge, organize it by workspace, and get traceable answers backed by
                citations, confidence signals, and structured analytics. Active workspace: <strong>{workspace_name}</strong>.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_section_title(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="section-title">
            <h3>{title}</h3>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stat_card(label: str, value: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
            <div class="stat-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_heading(title: str, caption: str) -> None:
    st.markdown(f"### {title}")
    st.caption(caption)


def reset_workspace_views(clear_scope: bool = True) -> None:
    st.session_state.messages = []
    st.session_state.last_result = None
    st.session_state.last_ingest_results = []
    st.session_state.evaluation_runs = []
    st.session_state.new_workspace_name = ""
    if clear_scope:
        st.session_state.use_source_scope = False
        st.session_state.selected_sources = []


init_state()
inject_styles()

workspace_manager = WorkspaceManager(settings)
available_workspaces = workspace_manager.list_workspaces()
if not available_workspaces:
    available_workspaces = [workspace_manager.ensure_default_workspace()]

available_workspace_ids = {workspace["id"] for workspace in available_workspaces}
if not st.session_state.workspace_id or st.session_state.workspace_id not in available_workspace_ids:
    st.session_state.workspace_id = available_workspaces[0]["id"]

current_workspace = workspace_manager.get_workspace(st.session_state.workspace_id) or available_workspaces[0]
workspace_manager.touch_workspace(current_workspace["id"])
session_settings = settings.with_workspace(current_workspace["id"])

vector_store = VectorStore(path=str(session_settings.qdrant_path), collection_name=session_settings.scoped_collection_name)
csv_registry = CSVRegistry(session_settings)
evaluation_store = EvaluationStore(session_settings)
indexed_docs = vector_store.list_documents() if session_settings.qdrant_path.exists() else []
indexed_csvs = csv_registry.list_datasets()
all_sources = sorted(set(indexed_docs + [item["file_name"] for item in indexed_csvs]))

with st.sidebar:
    render_sidebar_heading("Workspace Control", "Switch between persistent knowledge spaces and keep each corpus cleanly separated.")
    workspace_labels = {
        f"{workspace['name']} ({workspace['id']})": workspace["id"] for workspace in available_workspaces
    }
    current_workspace_label = next(
        label for label, workspace_id in workspace_labels.items() if workspace_id == current_workspace["id"]
    )
    selected_workspace_label = st.selectbox(
        "Workspace",
        options=list(workspace_labels.keys()),
        index=list(workspace_labels.keys()).index(current_workspace_label),
    )
    selected_workspace_id = workspace_labels[selected_workspace_label]
    if selected_workspace_id != current_workspace["id"]:
        st.session_state.workspace_id = selected_workspace_id
        reset_workspace_views()
        st.rerun()

    with st.expander("Create new workspace"):
        new_workspace_name = st.text_input(
            "Workspace name",
            value=st.session_state.new_workspace_name,
            key="create_workspace_name",
            placeholder="Quarterly review, HR policy, legal intake...",
        )
        st.session_state.new_workspace_name = new_workspace_name
        if st.button("Create workspace", use_container_width=True):
            if not new_workspace_name.strip():
                st.warning("Enter a workspace name.")
            else:
                new_workspace = workspace_manager.create_workspace(new_workspace_name)
                st.session_state.workspace_id = new_workspace["id"]
                reset_workspace_views()
                st.rerun()

    st.divider()
    render_sidebar_heading("Knowledge Scope", "Choose whether the assistant should query the whole workspace or a hand-picked subset of files.")
    st.write("The app can run fully locally. OpenAI is only used if `ANSWER_PROVIDER=openai` is configured.")
    default_sources = [source for source in st.session_state.selected_sources if source in all_sources] or all_sources
    use_source_scope = st.checkbox(
        "Limit queries to selected files",
        value=st.session_state.use_source_scope,
        help="Leave this off for the broadest retrieval. Turn it on when you want targeted debugging or focused Q&A.",
    )
    selected_sources = st.multiselect(
        "Selected files",
        options=all_sources,
        default=default_sources,
        disabled=not all_sources,
        help="These files are only used when source scoping is enabled.",
    )
    st.session_state.use_source_scope = use_source_scope
    st.session_state.selected_sources = selected_sources

    if st.button("Clear workspace knowledge", use_container_width=True):
        vector_store.clear_collection()
        csv_registry.clear()
        reset_workspace_views()
        st.success("Workspace documents and tabular datasets were cleared.")
        st.rerun()

    if indexed_csvs:
        with st.expander("Tabular datasets"):
            for dataset in indexed_csvs:
                st.markdown(f"- **{dataset['file_name']}**  \n  {dataset['row_count']} rows")

    with st.expander("Advanced diagnostics"):
        st.write(f"Strict grounded mode: `{session_settings.strict_grounded_mode}`")
        st.write(f"Answer provider: `{session_settings.answer_provider}`")
        st.write(f"Ollama base URL: `{session_settings.ollama_base_url}`")
        st.write(f"Ollama chat model: `{session_settings.ollama_chat_model}`")
        st.write(f"Embedding provider: `{session_settings.embedding_provider}`")
        st.write(
            f"Embedding model: `{session_settings.local_embedding_model if session_settings.embedding_provider == 'local' else session_settings.openai_embedding_model}`"
        )
        st.write(f"Workspace id: `{current_workspace['id']}`")
        st.write(f"Workspace collection: `{session_settings.scoped_collection_name}`")
        st.write(f"Workspace vector path: `{session_settings.qdrant_path}`")
        st.write(f"Workspace data path: `{session_settings.data_dir}`")
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

render_hero(current_workspace["name"])

scope_label = "All workspace files"
scope_caption = "Broad retrieval across the active workspace."
if st.session_state.use_source_scope and st.session_state.selected_sources:
    scope_label = f"{len(st.session_state.selected_sources)} selected"
    scope_caption = "Scoped retrieval is active for targeted answers."

status_cols = st.columns(4)
with status_cols[0]:
    render_stat_card("Workspace", current_workspace["name"], "Current persistent knowledge space")
with status_cols[1]:
    render_stat_card("Documents", str(len(indexed_docs)), "Indexed narrative or document files")
with status_cols[2]:
    render_stat_card("Tables", str(len(indexed_csvs)), "Datasets available for structured analysis")
with status_cols[3]:
    render_stat_card("Query Scope", scope_label, scope_caption)

tab_knowledge, tab_assistant, tab_eval = st.tabs(["Knowledge Hub", "Ask & Analyze", "Evaluation Lab"])

with tab_knowledge:
    info_col, upload_col = st.columns([0.95, 1.25], gap="large")

    with info_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        render_section_title(
            "Workspace readiness",
            "A quick operational view before you upload or ask questions.",
        )
        st.markdown(
            """
            <div class="soft-panel">
                This workspace supports PDF, scanned PDF, images, TXT, MD, DOCX, HTML, XML, JSON, CSV, TSV, and XLSX uploads.
                Retrieval stays grounded to uploaded evidence, and tabular questions can route through pandas/SQLite analytics.
            </div>
            """,
            unsafe_allow_html=True,
        )
        if all_sources:
            st.markdown('<div class="pill-row">', unsafe_allow_html=True)
            for source in all_sources[:12]:
                st.markdown(f'<span class="pill">{source}</span>', unsafe_allow_html=True)
            if len(all_sources) > 12:
                st.markdown(f'<span class="pill">+{len(all_sources) - 12} more</span>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            with st.expander("View all workspace files"):
                for source in all_sources:
                    st.markdown(f"- {source}")
        else:
            st.info("This workspace is empty. Upload files to build a searchable knowledge base.")
        st.markdown("</div>", unsafe_allow_html=True)

    with upload_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        render_section_title(
            "Upload and index",
            "Bring in fresh documents, replace an outdated corpus, or expand the workspace incrementally.",
        )
        replace_existing = st.checkbox("Replace existing workspace knowledge before indexing", value=False)
        uploaded_files = st.file_uploader(
            "Upload one or more files",
            type=["pdf", "txt", "md", "csv", "tsv", "docx", "xlsx", "json", "jsonl", "html", "htm", "xml", "png", "jpg", "jpeg", "webp", "tif", "tiff"],
            accept_multiple_files=True,
            help="Large mixed-format uploads are supported. Structured data files become available for analytics as well as retrieval.",
        )
        if uploaded_files:
            st.markdown("**Queued for indexing**")
            for uploaded_file in uploaded_files:
                st.markdown(f"- {uploaded_file.name}")

        if st.button("Index uploaded files", type="primary", use_container_width=True):
            if not uploaded_files:
                st.warning("Please upload at least one file.")
            else:
                try:
                    ingestor = IngestionService(session_settings)
                    if replace_existing:
                        vector_store.clear_collection()
                        csv_registry.clear()
                        reset_workspace_views(clear_scope=False)
                        st.session_state.use_source_scope = False
                    results = []
                    for file in uploaded_files:
                        result = ingestor.ingest_bytes(file.name, file.getvalue())
                        results.append(result)
                    st.session_state.last_ingest_results = [
                        {
                            "file_name": result.file_name,
                            "chunks_created": result.chunks_created,
                        }
                        for result in results
                    ]
                    st.session_state.selected_sources = [result.file_name for result in results]
                    st.rerun()
                except Exception as exc:
                    st.error(f"Indexing failed: {exc}")
                    st.code(traceback.format_exc())
        if st.session_state.last_ingest_results:
            st.success("Indexing complete.")
            for result in st.session_state.last_ingest_results:
                st.write(f"- {result['file_name']}: {result['chunks_created']} chunks")
        st.markdown("</div>", unsafe_allow_html=True)

with tab_assistant:
    question_col, answer_col = st.columns([1.15, 0.95], gap="large")

    with question_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        render_section_title(
            "Grounded question workspace",
            "Ask natural-language questions about documents, extracted OCR text, or uploaded tables.",
        )
        with st.form("question_form"):
            question = st.text_area(
                "Ask a grounded question",
                placeholder="What does the handbook say about minimum attendance? Which students are below 75% attendance? Count orders per region. What text appears in this scanned notice?",
                height=140,
            )
            submit_question = st.form_submit_button("Get answer", use_container_width=True, type="primary")
        if submit_question:
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

        st.markdown("**Conversation**")
        if not st.session_state.messages:
            st.info("Conversation history will appear here after the first question.")
        else:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    if msg["role"] == "assistant":
                        st.write(msg["content"])
                    else:
                        st.markdown(msg["content"])
        st.markdown("</div>", unsafe_allow_html=True)

    with answer_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        render_section_title(
            "Answer diagnostics",
            "Review confidence, routing, citations, and raw debug details for the latest answer.",
        )
        result = st.session_state.last_result
        if result is None:
            st.info("Your latest grounded answer will appear here.")
        else:
            route_type = result.debug.get("router", {}).get("route_type", result.debug.get("question_scope", "unknown"))
            st.markdown('<div class="answer-shell">', unsafe_allow_html=True)
            diag_cols = st.columns(3)
            diag_cols[0].metric("Route", route_type)
            diag_cols[1].metric("Confidence", badge_for_confidence(result.confidence))
            diag_cols[2].metric("Status", grounded_label(result))
            st.markdown("**Answer**")
            st.write(result.answer)
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
                    st.markdown(f"**{file_name}**")
                    st.caption(locator)
                    if quote:
                        st.caption(quote)
                    st.divider()
            else:
                st.write("No citation objects returned.")

            with st.expander("Advanced debug"):
                st.json(result.debug)
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

with tab_eval:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    render_section_title(
        "Evaluation workspace",
        "Save benchmark questions, run them against the current corpus, and track grounded-answer quality over time.",
    )

    form_col, actions_col = st.columns([1.2, 0.8], gap="large")
    with form_col:
        with st.form("evaluation_case_form", clear_on_submit=True):
            eval_question = st.text_area("Question", height=100, placeholder="What should the assistant answer reliably?")
            eval_expected_answer = st.text_area(
                "Expected answer or gold summary",
                height=100,
                placeholder="Optional reference answer for overlap scoring.",
            )
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
    with actions_col:
        st.info(f"{len(cases)} benchmark question(s) ready for evaluation in this workspace.")

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
            with metrics[0]:
                render_stat_card("Pass rate", f"{summary['pass_rate']}%", "Benchmarks meeting current criteria")
            with metrics[1]:
                render_stat_card("Grounded rate", f"{summary['grounded_rate']}%", "Answers marked as grounded")
            with metrics[2]:
                render_stat_card("Citation rate", f"{summary['citation_rate']}%", "Runs that returned citations")
            with metrics[3]:
                render_stat_card(
                    "Average overlap",
                    str(summary["average_overlap"]) if summary["average_overlap"] is not None else "n/a",
                    "Reference-answer overlap score",
                )

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
    st.markdown("</div>", unsafe_allow_html=True)
