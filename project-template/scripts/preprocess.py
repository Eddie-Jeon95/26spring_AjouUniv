"""
Raw 데이터를 학습용 processed CSV로 정리하는 스크립트.

사용 예시:
    python scripts/preprocess.py \
      --input data/raw/banknote.txt \
      --output data/processed/banknote_v1.csv \
      --target label \
      --sep "," \
      --header none \
      --columns variance,skewness,curtosis,entropy,label \
      --data-version banknote-v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def file_sha256(path: Path) -> str:
    """파일 checksum을 계산한다."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_csv_list(value: str | None) -> list[str] | None:
    if value is None or value.strip() == "":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_mapping(value: str | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if value is None or value.strip() == "":
        return mapping
    for item in value.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"mapping 항목은 old=new 형식이어야 합니다: {item}")
        key, mapped_value = item.split("=", 1)
        mapping[key.strip()] = mapped_value.strip()
    return mapping


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def update_data_manifest(path: Path, record: dict[str, Any]) -> None:
    manifest = read_json(path, {"datasets": []})
    datasets = manifest.setdefault("datasets", [])
    datasets.append(record)
    write_json(path, manifest)


def load_raw_csv(
    input_path: Path,
    sep: str,
    header: str,
    columns: list[str] | None,
) -> pd.DataFrame:
    if header == "none":
        if not columns:
            raise ValueError("--header none을 사용할 때는 --columns col1,col2 형식으로 컬럼명을 지정하세요.")
        return pd.read_csv(input_path, sep=sep, header=None, names=columns)
    if header == "infer":
        return pd.read_csv(input_path, sep=sep, header=0, names=columns)
    raise ValueError("--header는 infer 또는 none이어야 합니다.")


def preprocess(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"raw 파일을 찾을 수 없습니다: {input_path}")

    columns = parse_csv_list(args.columns)
    drop_columns = parse_csv_list(args.drop_columns) or []
    rename_map = parse_mapping(args.rename)
    target_map = parse_mapping(args.target_map)

    df = load_raw_csv(input_path=input_path, sep=args.sep, header=args.header, columns=columns)
    original_rows, original_columns = df.shape

    df.columns = [str(column).strip() for column in df.columns]
    if rename_map:
        missing_rename = sorted(set(rename_map) - set(df.columns))
        if missing_rename:
            raise ValueError(f"rename 대상 컬럼이 없습니다: {missing_rename}")
        df = df.rename(columns=rename_map)

    target_column = args.target.strip()
    if target_column not in df.columns:
        raise ValueError(f"target column '{target_column}'이 데이터에 없습니다. 현재 컬럼: {list(df.columns)}")

    if drop_columns:
        missing_drop = sorted(set(drop_columns) - set(df.columns))
        if missing_drop:
            raise ValueError(f"drop 대상 컬럼이 없습니다: {missing_drop}")
        if target_column in drop_columns:
            raise ValueError("target column은 drop할 수 없습니다.")
        df = df.drop(columns=drop_columns)

    if target_map:
        df[target_column] = df[target_column].astype(str).replace(target_map)

    duplicate_rows_removed = 0
    if not args.keep_duplicate_rows:
        before = len(df)
        df = df.drop_duplicates()
        duplicate_rows_removed = before - len(df)

    target_missing_rows_removed = 0
    if not args.keep_rows_missing_target:
        before = len(df)
        df = df.dropna(subset=[target_column])
        target_missing_rows_removed = before - len(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    data_version = args.data_version or output_path.stem
    record = {
        "data_version": data_version,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": args.source or str(input_path),
        "raw_file_path": str(input_path),
        "processed_file_path": str(output_path),
        "target_column": target_column,
        "raw_rows": int(original_rows),
        "raw_columns": int(original_columns),
        "processed_rows": int(df.shape[0]),
        "processed_columns": int(df.shape[1]),
        "raw_checksum_sha256": file_sha256(input_path),
        "processed_checksum_sha256": file_sha256(output_path),
        "preprocessing": {
            "sep": args.sep,
            "header": args.header,
            "columns": columns,
            "rename": rename_map,
            "drop_columns": drop_columns,
            "target_map": target_map,
            "duplicate_rows_removed": int(duplicate_rows_removed),
            "target_missing_rows_removed": int(target_missing_rows_removed),
            "fit_based_transforms": "Not applied here. Keep scaler/encoder/imputer inside train pipeline.",
        },
        "notes": args.notes,
    }
    update_data_manifest(Path(args.manifest), record)
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Raw 데이터를 학습용 processed CSV로 정리합니다.")
    parser.add_argument("--input", required=True, help="raw CSV/TXT/TSV 파일 경로")
    parser.add_argument("--output", required=True, help="저장할 processed CSV 경로")
    parser.add_argument("--target", required=True, help="target column 이름")
    parser.add_argument("--data-version", default=None, help="data_manifest.json에 기록할 데이터 버전")
    parser.add_argument("--sep", default=",", help="pandas.read_csv separator")
    parser.add_argument("--header", choices=["infer", "none"], default="infer", help="header가 없으면 none")
    parser.add_argument("--columns", default=None, help="header가 없거나 컬럼명을 덮어쓸 때 col1,col2 형식")
    parser.add_argument("--drop-columns", default=None, help="제거할 컬럼 목록: col1,col2")
    parser.add_argument("--rename", default=None, help="컬럼명 변경: old=new,old2=new2")
    parser.add_argument("--target-map", default=None, help="target 값 mapping: old=new,old2=new2")
    parser.add_argument("--keep-duplicate-rows", action="store_true", help="중복 row를 제거하지 않음")
    parser.add_argument("--keep-rows-missing-target", action="store_true", help="target 결측 row를 제거하지 않음")
    parser.add_argument("--manifest", default="data_manifest.json", help="갱신할 data manifest 경로")
    parser.add_argument("--source", default=None, help="데이터 출처 메모")
    parser.add_argument("--notes", default="", help="전처리 메모")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        record = preprocess(args)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
