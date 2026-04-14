"""
Stream NVD CVE JSON feeds (nvdcve-2.0-YYYY.json) into RAG corpus chunks.

NVD schema: top-level object with a "vulnerabilities" array; each element has {"cve": {...}}.
We keep only lightweight fields compatible with scripts/lib/rag_corpus.py:

  id    — CVE id (e.g. CVE-2026-0544)
  title — short header: CVE id + vuln status
  text  — English description + optional published line (same role as former "description" body)

Large feeds must be streamed (ijson); do not json.load the whole file.

Examples (repo root):

  # JSONL only (full merge, good for huge corpora + build_rag_index on .jsonl)
  python scripts/nvdcve_to_rag_corpus.py \\
    --inputs data/nvdcve-2.0-2024.json data/nvdcve-2.0-2025.json data/nvdcve-2.0-2026.json \\
    --out data/rag_corpus_nvd.jsonl --format jsonl

  # Replace data/rag_corpus.json with education snippets + first 12k CVEs (bounded for git)
  python scripts/nvdcve_to_rag_corpus.py \\
    --inputs data/nvdcve-2.0-2024.json data/nvdcve-2.0-2025.json data/nvdcve-2.0-2026.json \\
    --out data/rag_corpus.json --format json --max-nvd 12000 \\
    --merge-education data/rag_corpus_education.json

Data source: NIST NVD (public domain US government work; see NVD terms of use on nvd.nist.gov).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List

_MAX_TEXT_CHARS = 3500


def _pick_en_description(descriptions: Any) -> str:
    if not isinstance(descriptions, list):
        return ""
    for d in descriptions:
        if isinstance(d, dict) and d.get("lang") == "en":
            v = (d.get("value") or "").strip()
            if v:
                return v
    for d in descriptions:
        if isinstance(d, dict):
            v = (d.get("value") or "").strip()
            if v:
                return v
    return ""


def _cve_to_chunk(cve: Dict[str, Any]) -> Optional[Dict[str, str]]:
    cid = (cve.get("id") or "").strip()
    if not cid:
        return None
    desc = _pick_en_description(cve.get("descriptions"))
    if not desc:
        return None
    if len(desc) > _MAX_TEXT_CHARS:
        desc = desc[: _MAX_TEXT_CHARS - 1] + "…"
    status = (cve.get("vulnStatus") or "").strip()
    title = f"{cid} ({status})" if status else cid
    published = (cve.get("published") or "").strip()
    body = desc
    if published:
        body = f"{desc}\n\nPublished (NVD): {published}"
    return {"id": cid, "title": title, "text": body}


def _iter_chunks_from_file(path: Path) -> Iterator[Dict[str, str]]:
    import ijson  # noqa: PLC0415

    with open(path, "rb") as f:
        for item in ijson.items(f, "vulnerabilities.item"):
            if not isinstance(item, dict):
                continue
            cve = item.get("cve")
            if not isinstance(cve, dict):
                continue
            chunk = _cve_to_chunk(cve)
            if chunk:
                yield chunk


def _load_education(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON array")
    return data


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more nvdcve-2.0-*.json feed files (NVD_CVE format).",
    )
    p.add_argument("--out", required=True, help="Output path (.json array or .jsonl).")
    p.add_argument(
        "--format",
        choices=("jsonl", "json"),
        default="jsonl",
        help="jsonl streams one chunk per line; json writes a single array (needs RAM for --max-nvd rows).",
    )
    p.add_argument(
        "--max-nvd",
        type=int,
        default=None,
        help="Stop after this many NVD-derived chunks (applied across all --inputs in order). Default: no limit.",
    )
    p.add_argument(
        "--merge-education",
        default=None,
        help="Optional JSON array file (e.g. data/rag_corpus_education.json) prepended before NVD rows.",
    )
    args = p.parse_args()

    inputs = [Path(x) for x in args.inputs]
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(path)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written_nvd = 0
    education: List[Dict[str, Any]] = []
    if args.merge_education:
        education = _load_education(Path(args.merge_education))

    if args.format == "jsonl":
        with open(out_path, "w", encoding="utf-8") as outf:
            for row in education:
                outf.write(json.dumps(row, ensure_ascii=False) + "\n")
            for path in inputs:
                for chunk in _iter_chunks_from_file(path):
                    if args.max_nvd is not None and written_nvd >= args.max_nvd:
                        break
                    outf.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    written_nvd += 1
                if args.max_nvd is not None and written_nvd >= args.max_nvd:
                    break
        print(f"Wrote {len(education)} education + {written_nvd} NVD rows to {out_path} (jsonl)")
        return

    # json array — accumulate NVD up to max-nvd (required for bounded memory)
    if args.max_nvd is None:
        raise SystemExit("--format json requires --max-nvd to bound memory use (or use jsonl for full export).")

    nvd_rows: List[Dict[str, str]] = []
    for path in inputs:
        for chunk in _iter_chunks_from_file(path):
            if written_nvd >= args.max_nvd:
                break
            nvd_rows.append(chunk)
            written_nvd += 1
        if written_nvd >= args.max_nvd:
            break

    combined: List[Any] = list(education) + nvd_rows
    out_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(education)} education + {len(nvd_rows)} NVD = {len(combined)} chunks to {out_path} (json)")


if __name__ == "__main__":
    main()
