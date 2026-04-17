from __future__ import annotations

import io
import json
import re
import sqlite3
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import Settings
from src.utils import normalize_whitespace
from src.llm import LLMService, LocalLLMUnavailableError
from src.models import AnswerResult
from src.prompts import SQL_ANSWER_SYSTEM_PROMPT, SQL_PLANNER_SYSTEM_PROMPT
from src.utils import sha256_bytes

_SAFE_IDENT = re.compile(r"[^A-Za-z0-9_]+")
_FORBIDDEN_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|reindex|trigger)\b",
    re.IGNORECASE,
)


class CSVRegistry:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir = self.settings.data_dir / "uploads"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        if not self.settings.csv_registry_path.exists():
            self._write_registry({"datasets": []})

    def register_tabular_file(self, file_name: str, data: bytes) -> list[dict[str, Any]]:
        suffix = Path(file_name).suffix.lower()
        if suffix in {".csv", ".tsv"}:
            return [self._register_dataframe(file_name, self._read_delimited(data), data)]
        if suffix == ".xlsx":
            datasets: list[dict[str, Any]] = []
            excel = pd.ExcelFile(io.BytesIO(data), engine="openpyxl")
            for sheet_name in excel.sheet_names:
                df = pd.read_excel(io.BytesIO(data), sheet_name=sheet_name, engine="openpyxl")
                pseudo_name = f"{file_name} [{sheet_name}]"
                sheet_bytes = df.to_csv(index=False).encode("utf-8")
                datasets.append(self._register_dataframe(pseudo_name, df, sheet_bytes, sheet_name=sheet_name))
            return datasets
        raise ValueError(f"Unsupported tabular file type: {suffix}")

    def register_csv(self, file_name: str, data: bytes) -> dict[str, Any]:
        return self.register_tabular_file(file_name, data)[0]

    def list_datasets(self) -> list[dict[str, Any]]:
        return list(self._read_registry().get("datasets", []))

    def has_datasets(self) -> bool:
        return bool(self.list_datasets())

    def get_dataset_by_hash(self, file_hash: str) -> dict[str, Any] | None:
        for dataset in self.list_datasets():
            if dataset.get("file_hash") == file_hash:
                return dataset
        return None

    def schema_context(self) -> str:
        datasets = self.list_datasets()
        if not datasets:
            return "No tabular datasets are registered."

        parts: list[str] = []
        for dataset in datasets:
            parts.append(f"File: {dataset['file_name']}")
            parts.append(f"Table: {dataset['table_name']}")
            parts.append(f"Rows: {dataset['row_count']}")
            if dataset.get("sheet_name"):
                parts.append(f"Sheet: {dataset['sheet_name']}")
            parts.append("Columns:")
            for column in dataset.get("columns", []):
                parts.append(
                    f"- {column['original']} | sql_name={column['sql']} | dtype={column['dtype']}"
                )
            sample_rows = self._sample_rows(dataset["table_name"], limit=self.settings.csv_schema_sample_rows)
            if sample_rows:
                parts.append("Sample rows:")
                parts.append(json.dumps(sample_rows, ensure_ascii=False))
            parts.append("")
        return "\n".join(parts).strip()

    def execute_query(self, sql: str, limit: int) -> pd.DataFrame:
        cleaned = self._validate_sql(sql)
        wrapped_sql = self._apply_limit(cleaned, limit)
        with sqlite3.connect(self.settings.sqlite_db_path) as conn:
            return pd.read_sql_query(wrapped_sql, conn)

    def tables_mentioned(self, sql: str) -> list[dict[str, Any]]:
        sql_lower = sql.lower()
        matches: list[dict[str, Any]] = []
        for dataset in self.list_datasets():
            if dataset["table_name"].lower() in sql_lower:
                matches.append(dataset)
        return matches

    def _sample_rows(self, table_name: str, limit: int = 3) -> list[dict[str, Any]]:
        if not self.settings.sqlite_db_path.exists():
            return []
        with sqlite3.connect(self.settings.sqlite_db_path) as conn:
            df = pd.read_sql_query(f'SELECT * FROM "{table_name}" LIMIT {int(limit)}', conn)
        return json.loads(df.to_json(orient="records", force_ascii=False))

    def _read_registry(self) -> dict[str, Any]:
        return json.loads(self.settings.csv_registry_path.read_text(encoding="utf-8"))

    def _write_registry(self, payload: dict[str, Any]) -> None:
        self.settings.csv_registry_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _register_dataframe(
        self,
        file_name: str,
        df: pd.DataFrame,
        raw_bytes: bytes,
        sheet_name: str | None = None,
    ) -> dict[str, Any]:
        if df.empty and len(df.columns) == 0:
            raise ValueError(f"Tabular file {file_name} has no readable columns")
        file_hash = sha256_bytes(raw_bytes)
        existing = self.get_dataset_by_hash(file_hash)
        if existing:
            return existing

        original_columns = [str(col) for col in df.columns]
        sql_columns = self._unique_sql_columns(original_columns)
        rename_map = dict(zip(original_columns, sql_columns))
        df = df.fillna("").rename(columns=rename_map)
        df.insert(0, "__row_number__", range(1, len(df) + 1))

        table_name = self._make_table_name(Path(file_name).stem, file_hash)
        saved_path = self.upload_dir / f"{file_hash[:12]}_{Path(file_name).name}"
        saved_path.write_bytes(raw_bytes)

        self.settings.sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.settings.sqlite_db_path) as conn:
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_name}_rownum ON "{table_name}"(__row_number__)')
            conn.commit()

        dataset = {
            "file_name": file_name,
            "file_hash": file_hash,
            "saved_path": str(saved_path),
            "table_name": table_name,
            "row_count": int(len(df)),
            "sheet_name": sheet_name,
            "columns": [
                {"original": orig, "sql": sql, "dtype": str(df[sql].dtype)}
                for orig, sql in zip(original_columns, sql_columns)
            ],
        }
        registry = self._read_registry()
        registry["datasets"] = [d for d in registry.get("datasets", []) if d.get("file_hash") != file_hash]
        registry["datasets"].append(dataset)
        self._write_registry(registry)
        return dataset

    def _read_delimited(self, data: bytes) -> pd.DataFrame:
        text = self._decode_text_bytes(data)
        try:
            return pd.read_csv(io.StringIO(text), sep=None, engine="python")
        except Exception:
            return pd.read_csv(io.StringIO(text))

    def _make_table_name(self, base_name: str, file_hash: str) -> str:
        safe = self._sanitize_identifier(base_name)[:32]
        return f"t_{safe}_{file_hash[:8]}"

    def _unique_sql_columns(self, columns: list[str]) -> list[str]:
        seen: dict[str, int] = {}
        output: list[str] = []
        for column in columns:
            base = self._sanitize_identifier(column)[:40]
            count = seen.get(base, 0)
            seen[base] = count + 1
            output.append(base if count == 0 else f"{base}_{count + 1}")
        return output

    @staticmethod
    def _sanitize_identifier(value: str) -> str:
        cleaned = _SAFE_IDENT.sub("_", value.strip())
        cleaned = re.sub(r"_+", "_", cleaned).strip("_") or "col"
        if cleaned[0].isdigit():
            cleaned = f"c_{cleaned}"
        return cleaned.lower()

    def _validate_sql(self, sql: str) -> str:
        cleaned = normalize_whitespace(sql).strip()
        if not cleaned:
            raise ValueError("SQL planner returned an empty query")
        if cleaned.endswith(";"):
            cleaned = cleaned[:-1].strip()
        if ";" in cleaned or "--" in cleaned or "/*" in cleaned:
            raise ValueError("Only a single safe read-only SQL statement is allowed")
        if not re.match(r"^(select|with)\b", cleaned, re.IGNORECASE):
            raise ValueError("Only SELECT queries are allowed")
        if _FORBIDDEN_SQL.search(cleaned):
            raise ValueError("Unsafe SQL was rejected")
        return cleaned

    @staticmethod
    def _apply_limit(sql: str, limit: int) -> str:
        if re.search(r"\blimit\s+\d+\b", sql, re.IGNORECASE):
            return sql
        return f"SELECT * FROM ({sql}) LIMIT {int(limit)}"

    @staticmethod
    def _decode_text_bytes(data: bytes) -> str:
        for encoding in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                return data.decode(encoding)
            except Exception:
                continue
        return data.decode("utf-8", errors="ignore")


class CSVQueryService:
    def __init__(self, settings: Settings, llm: LLMService | None = None):
        self.settings = settings
        self.registry = CSVRegistry(settings)
        self.llm = llm or LLMService(settings)

    def has_datasets(self) -> bool:
        return self.registry.has_datasets()

    def try_answer(self, question: str) -> AnswerResult | None:
        if not self.registry.has_datasets():
            return None

        if not self.llm.supports_chat_json():
            return self._try_answer_local(question)

        try:
            planner = self.llm.chat_json(
                SQL_PLANNER_SYSTEM_PROMPT,
                self._planner_prompt(question),
                temperature=0,
            )
        except LocalLLMUnavailableError:
            return self._try_answer_local(question)
        if not bool(planner.get("use_structured_query", False)):
            return None

        sql = str(planner.get("sql", "")).strip()
        if not sql:
            return None

        result_df = self.registry.execute_query(sql, self.settings.max_sql_result_rows)
        tables_used = self.registry.tables_mentioned(sql)
        file_names = [item["file_name"] for item in tables_used] or [item["file_name"] for item in self.registry.list_datasets()]
        result_records = json.loads(result_df.to_json(orient="records", force_ascii=False))
        result_preview = json.dumps(result_records[: self.settings.max_sql_result_rows], ensure_ascii=False)
        try:
            answer_payload = self.llm.chat_json(
                SQL_ANSWER_SYSTEM_PROMPT,
                self._answer_prompt(question, planner, sql, file_names, result_df, result_preview),
                temperature=0,
            )
        except LocalLLMUnavailableError:
            answer_payload = {
                "answer": self._summarize_local_result(question, planner, result_df),
                "citations": self._fallback_citations(tables_used, result_df),
                "confidence": "high" if not result_df.empty else "medium",
                "grounded": True,
                "unsupported_claims": [],
            }

        answer = str(answer_payload.get("answer", "")).strip() or "I could not produce a grounded tabular answer."
        citations = answer_payload.get("citations", []) if isinstance(answer_payload.get("citations", []), list) else []
        confidence = self._normalize_confidence(answer_payload.get("confidence", "medium"))
        grounded = bool(answer_payload.get("grounded", True))
        unsupported_claims = answer_payload.get("unsupported_claims", [])
        if not isinstance(unsupported_claims, list):
            unsupported_claims = []

        if not citations:
            citations = self._fallback_citations(tables_used, result_df)

        citation_check = self._validate_citations(citations, tables_used, result_df)
        if self.settings.strict_grounded_mode and (citation_check["invalid_count"] > 0 or unsupported_claims):
            answer = "I executed a tabular query, but I could not verify a fully reliable natural-language answer. Please review the citations and SQL result preview."
            grounded = False
            confidence = "low"

        return AnswerResult(
            answer=answer,
            citations=citations,
            confidence=confidence,
            grounded=grounded,
            debug={
                "mode": "csv_structured",
                "sql_planner": planner,
                "sql": sql,
                "tables_used": file_names,
                "result_rows": int(len(result_df)),
                "result_preview": result_records[: min(len(result_records), 10)],
                "citation_validation": citation_check,
                "unsupported_claims": unsupported_claims,
            },
        )

    def _try_answer_local(self, question: str) -> AnswerResult | None:
        planner = self._local_planner(question)
        if not bool(planner.get("use_structured_query", False)):
            return None

        sql = str(planner.get("sql", "")).strip()
        if not sql:
            return None

        result_df = self.registry.execute_query(sql, self.settings.max_sql_result_rows)
        tables_used = self.registry.tables_mentioned(sql)
        file_names = [item["file_name"] for item in tables_used] or [item["file_name"] for item in self.registry.list_datasets()]
        citations = self._fallback_citations(tables_used, result_df)
        answer = self._summarize_local_result(question, planner, result_df)
        citation_check = self._validate_citations(citations, tables_used, result_df)
        confidence = "high" if not result_df.empty else "medium"
        grounded = citation_check["invalid_count"] == 0

        return AnswerResult(
            answer=answer,
            citations=citations,
            confidence=confidence,
            grounded=grounded,
            debug={
                "mode": "csv_structured_local",
                "sql_planner": planner,
                "sql": sql,
                "tables_used": file_names,
                "result_rows": int(len(result_df)),
                "result_preview": json.loads(result_df.head(10).to_json(orient="records", force_ascii=False)),
                "citation_validation": citation_check,
                "unsupported_claims": [],
            },
        )

    def _local_planner(self, question: str) -> dict[str, Any]:
        dataset = self._best_dataset(question)
        if not dataset:
            return {"use_structured_query": False, "reason": "no_dataset_match"}

        operation = self._detect_operation(question)
        column = self._best_column(question, dataset)
        where_clause = self._detect_numeric_filter(question, dataset)
        table_name = dataset["table_name"]

        if operation == "count":
            sql = f'SELECT COUNT(*) AS count FROM "{table_name}"{where_clause}'
        elif operation == "sum" and column:
            sql = f'SELECT SUM("{column}") AS sum_{column} FROM "{table_name}"{where_clause}'
        elif operation == "avg" and column:
            sql = f'SELECT AVG("{column}") AS avg_{column} FROM "{table_name}"{where_clause}'
        elif operation == "min" and column:
            sql = f'SELECT MIN("{column}") AS min_{column} FROM "{table_name}"{where_clause}'
        elif operation == "max" and column:
            sql = f'SELECT MAX("{column}") AS max_{column} FROM "{table_name}"{where_clause}'
        else:
            select_columns = ['"__row_number__"']
            select_columns.extend(f'"{col["sql"]}"' for col in dataset.get("columns", [])[: min(5, len(dataset.get("columns", [])))])
            sql = f'SELECT {", ".join(select_columns)} FROM "{table_name}"{where_clause} LIMIT {int(self.settings.max_sql_result_rows)}'

        return {
            "use_structured_query": True,
            "mode": "local_heuristic",
            "dataset": dataset["file_name"],
            "operation": operation,
            "column": column,
            "sql": sql,
        }

    def _summarize_local_result(self, question: str, planner: dict[str, Any], result_df: pd.DataFrame) -> str:
        if result_df.empty:
            return "No matching rows were found in the tabular data."

        operation = str(planner.get("operation", "list"))
        first_column = result_df.columns[0]
        first_value = result_df.iloc[0][first_column]
        if operation == "count":
            return f"The matching row count is {first_value}."
        if operation in {"sum", "avg", "min", "max"}:
            label = {
                "sum": "sum",
                "avg": "average",
                "min": "minimum",
                "max": "maximum",
            }[operation]
            return f"The {label} is {first_value}."

        preview = json.loads(result_df.head(min(len(result_df), 5)).to_json(orient="records", force_ascii=False))
        return f"I found {len(result_df)} matching rows. Preview: {json.dumps(preview, ensure_ascii=False)}"

    def _best_dataset(self, question: str) -> dict[str, Any] | None:
        datasets = self.registry.list_datasets()
        if not datasets:
            return None
        q = question.lower()
        best: tuple[float, dict[str, Any]] | None = None
        for dataset in datasets:
            haystack = " ".join(
                [dataset["file_name"], dataset["table_name"]]
                + [column["original"] for column in dataset.get("columns", [])]
                + [column["sql"] for column in dataset.get("columns", [])]
            ).lower()
            score = SequenceMatcher(None, q, haystack).ratio()
            score += sum(1 for token in re.findall(r"[A-Za-z0-9_]+", q) if token in haystack) * 0.02
            if best is None or score > best[0]:
                best = (score, dataset)
        return best[1] if best else None

    def _best_column(self, question: str, dataset: dict[str, Any]) -> str | None:
        q = question.lower()
        best_score = 0.0
        best_column: str | None = None
        for column in dataset.get("columns", []):
            names = [str(column["original"]).lower(), str(column["sql"]).lower()]
            score = max(SequenceMatcher(None, q, name).ratio() for name in names)
            score += sum(0.1 for name in names if name in q)
            if score > best_score:
                best_score = score
                best_column = str(column["sql"])
        return best_column

    def _detect_operation(self, question: str) -> str:
        lowered = question.lower()
        if any(token in lowered for token in ["how many", "count", "number of"]):
            return "count"
        if any(token in lowered for token in ["average", "avg", "mean"]):
            return "avg"
        if any(token in lowered for token in ["sum", "total"]):
            return "sum"
        if any(token in lowered for token in ["minimum", "lowest", "smallest"]):
            return "min"
        if any(token in lowered for token in ["maximum", "highest", "largest"]):
            return "max"
        return "list"

    def _detect_numeric_filter(self, question: str, dataset: dict[str, Any]) -> str:
        lowered = question.lower()
        column = self._best_column(question, dataset)
        if not column:
            return ""
        patterns = [
            (r"(?:below|under|less than)\s+(-?\d+(?:\.\d+)?)", "<"),
            (r"(?:above|over|greater than|more than)\s+(-?\d+(?:\.\d+)?)", ">"),
            (r"(?:equal to|equals|is)\s+(-?\d+(?:\.\d+)?)", "="),
        ]
        for pattern, operator in patterns:
            match = re.search(pattern, lowered)
            if match:
                return f' WHERE "{column}" {operator} {match.group(1)}'
        return ""

    def _planner_prompt(self, question: str) -> str:
        return (
            f"Question:\n{question}\n\n"
            f"Available tabular schemas:\n{self.registry.schema_context()}\n\n"
            "Return JSON only."
        )

    def _answer_prompt(
        self,
        question: str,
        planner: dict[str, Any],
        sql: str,
        file_names: list[str],
        result_df: pd.DataFrame,
        result_preview: str,
    ) -> str:
        schema = self.registry.schema_context()
        files_line = ", ".join(file_names) if file_names else "unknown"
        return (
            f"Question:\n{question}\n\n"
            f"Planner output:\n{json.dumps(planner, ensure_ascii=False)}\n\n"
            f"Source files:\n{files_line}\n\n"
            f"Schema context:\n{schema}\n\n"
            f"Executed SQL:\n{sql}\n\n"
            f"Result row count: {len(result_df)}\n"
            f"Result columns: {list(result_df.columns)}\n"
            f"Result preview JSON:\n{result_preview}\n\n"
            "Return JSON only."
        )

    def _fallback_citations(self, tables_used: list[dict[str, Any]], result_df: pd.DataFrame) -> list[dict[str, Any]]:
        if result_df.empty:
            file_name = tables_used[0]["file_name"] if tables_used else "tabular_result"
            return [{"file_name": file_name, "locator": "SQL result (0 rows)", "quote": "No matching rows were returned."}]

        citations: list[dict[str, Any]] = []
        file_name = tables_used[0]["file_name"] if tables_used else "tabular_result"
        if "__row_number__" in result_df.columns and len(tables_used) <= 1:
            for _, row in result_df.head(3).iterrows():
                row_num = int(row["__row_number__"])
                row_dict = {col: row[col] for col in result_df.columns if col != "__row_number__"}
                quote = json.dumps(row_dict, ensure_ascii=False, default=str)
                citations.append(
                    {
                        "file_name": file_name,
                        "locator": f"row {row_num}",
                        "quote": quote[:200],
                    }
                )
        else:
            quote = json.dumps(json.loads(result_df.head(3).to_json(orient="records", force_ascii=False)), ensure_ascii=False)
            for dataset in tables_used[:3] or [{"file_name": "tabular_result"}]:
                citations.append(
                    {
                        "file_name": dataset["file_name"],
                        "locator": f"SQL result ({len(result_df)} rows)",
                        "quote": quote[:200],
                    }
                )
        return citations

    def _validate_citations(
        self,
        citations: list[dict[str, Any]],
        tables_used: list[dict[str, Any]],
        result_df: pd.DataFrame,
    ) -> dict[str, Any]:
        valid_file_names = {table["file_name"] for table in tables_used} or {d["file_name"] for d in self.registry.list_datasets()}
        result_text = result_df.head(10).to_json(orient="records", force_ascii=False).lower()
        row_numbers = {int(v) for v in result_df["__row_number__"].tolist()} if "__row_number__" in result_df.columns else set()
        valid = 0
        invalid: list[dict[str, Any]] = []

        for citation in citations:
            if not isinstance(citation, dict):
                invalid.append({"citation": citation, "reason": "not_an_object"})
                continue
            file_name = str(citation.get("file_name", "")).strip()
            locator = str(citation.get("locator", "")).strip().lower()
            quote = str(citation.get("quote", "")).strip().lower()
            if file_name not in valid_file_names:
                invalid.append({"citation": citation, "reason": "unknown_file"})
                continue
            if locator.startswith("row ") and row_numbers:
                try:
                    row_num = int(locator.split()[1])
                except Exception:
                    invalid.append({"citation": citation, "reason": "bad_row_locator"})
                    continue
                if row_num not in row_numbers:
                    invalid.append({"citation": citation, "reason": "row_not_in_result"})
                    continue
                valid += 1
                continue
            if not locator.startswith("sql result") and not locator.startswith("row "):
                invalid.append({"citation": citation, "reason": "unsupported_locator"})
                continue
            if quote and quote not in result_text:
                invalid.append({"citation": citation, "reason": "quote_not_in_result_preview"})
                continue
            valid += 1

        return {"valid_count": valid, "invalid_count": len(invalid), "invalid": invalid}

    @staticmethod
    def _normalize_confidence(value: Any) -> str:
        normalized = str(value).strip().lower()
        if normalized in {"high", "medium", "low"}:
            return normalized
        return "medium"
