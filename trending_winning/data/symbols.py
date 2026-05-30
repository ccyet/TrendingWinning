from __future__ import annotations

from pathlib import Path

import pandas as pd

from trending_winning.data.schema import TIMEFRAME_DIR_NAMES, normalize_symbol, unique_symbols

SYMBOL_METADATA_COLUMNS = ["stock_code", "stock_name", "source", "path"]
DEFAULT_STOCK_NAME_BY_CODE = {
    "000001.SZ": "平安银行",
    "000002.SZ": "万科A",
    "000003.SZ": "PT金田A",
    "000333.SZ": "美的集团",
    "002415.SZ": "海康威视",
    "300059.SZ": "东方财富",
    "300750.SZ": "宁德时代",
    "600000.SH": "浦发银行",
    "600036.SH": "招商银行",
    "600519.SH": "贵州茅台",
    "601318.SH": "中国平安",
    "688001.SH": "华兴源创",
}


def load_symbol_metadata(data_root: str | Path, *, tdx_path: str | Path = "") -> pd.DataFrame:
    """加载股票代码和名称；优先使用行情目录 sidecar，其次使用 TDX 本地缓存。"""
    frames = [_load_sidecar_symbol_metadata(data_root)]
    if tdx_path:
        frames.append(load_tdx_symbol_metadata(tdx_path))
    return _merge_symbol_metadata(frames)


def resolve_symbol_names(
    symbols: list[str] | tuple[str, ...],
    *,
    data_root: str | Path | None = None,
    tdx_path: str | Path = "",
) -> dict[str, str]:
    """按标准代码返回股票名称；sidecar/TDX 覆盖默认常用代码表。"""
    requested = unique_symbols(tuple(symbols))
    names = {symbol: DEFAULT_STOCK_NAME_BY_CODE[symbol] for symbol in requested if symbol in DEFAULT_STOCK_NAME_BY_CODE}
    if data_root is not None:
        metadata = load_symbol_metadata(data_root, tdx_path=tdx_path)
        for row in metadata.itertuples(index=False):
            symbol = str(row.stock_code)
            if symbol in requested:
                names[symbol] = str(row.stock_name)
    return names


def load_tdx_symbol_metadata(tdx_path: str | Path) -> pd.DataFrame:
    """从通达信 hq_cache 的 shm/szm/bjm.tnf 读取股票名称。"""
    rows: list[dict[str, object]] = []
    for path in _tdx_tnf_candidates(tdx_path):
        rows.extend(_read_tdx_tnf_file(path))
    return _metadata_frame(rows)


def _load_sidecar_symbol_metadata(data_root: str | Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in _sidecar_symbol_files(data_root):
        rows.extend(_read_sidecar_symbol_file(path))
    return _metadata_frame(rows)


def _sidecar_symbol_files(data_root: str | Path) -> list[Path]:
    roots = _metadata_roots(data_root)
    names = ("symbols.csv", "stock_names.csv", "symbol_names.csv")
    files: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        for name in names:
            for path in (root / name, root / "metadata" / name):
                key = str(path)
                if key in seen or not path.exists():
                    continue
                seen.add(key)
                files.append(path)
    return files


def _metadata_roots(data_root: str | Path) -> list[Path]:
    root = Path(data_root).expanduser()
    candidates = [root]
    if root.name.lower() in set(TIMEFRAME_DIR_NAMES.values()):
        candidates.append(root.parent)
    candidates.extend(root.parents[:2])
    seen: set[str] = set()
    result: list[Path] = []
    for item in candidates:
        key = str(item)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _read_sidecar_symbol_file(path: Path) -> list[dict[str, object]]:
    frame = pd.read_csv(path)
    code_column = _first_existing_column(frame, ("stock_code", "symbol", "code", "代码", "证券代码"))
    name_column = _first_existing_column(frame, ("stock_name", "name", "名称", "证券名称", "股票名称"))
    if not code_column or not name_column:
        raise ValueError(f"股票名称文件缺少代码或名称列：{path}")
    rows: list[dict[str, object]] = []
    for raw_code, raw_name in zip(frame[code_column], frame[name_column], strict=False):
        symbol = normalize_symbol(raw_code)
        name = _clean_text(raw_name)
        if not symbol or not name:
            continue
        rows.append({"stock_code": symbol, "stock_name": name, "source": "sidecar_csv", "path": str(path)})
    return rows


def _first_existing_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    normalized = {str(column).strip().lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        found = normalized.get(candidate.lower())
        if found is not None:
            return found
    return ""


def _tdx_tnf_candidates(tdx_path: str | Path) -> list[Path]:
    base = Path(tdx_path).expanduser()
    if base.is_file() and base.suffix.lower() == ".tnf":
        return [base]
    candidates: list[Path] = []
    roots = [base, *base.parents]
    for root in roots:
        for folder in (root / "T0002" / "hq_cache", root / "hq_cache"):
            for name in ("shm.tnf", "szm.tnf", "bjm.tnf"):
                path = folder / name
                if path.exists():
                    candidates.append(path)
    return _unique_paths(candidates)


def _read_tdx_tnf_file(path: Path) -> list[dict[str, object]]:
    payload = path.read_bytes()
    if len(payload) <= 50:
        return []
    exchange = _tdx_exchange_from_filename(path.name)
    rows: list[dict[str, object]] = []
    for offset in range(50, len(payload), 314):
        record = payload[offset : offset + 314]
        if len(record) < 314:
            continue
        code = _decode_record_field(record[0:6], encoding="ascii")
        if not code.isdigit():
            continue
        symbol = normalize_symbol(f"{code}.{exchange}") if exchange else normalize_symbol(code)
        name = _tdx_record_name(record)
        if not symbol or not name:
            continue
        rows.append({"stock_code": symbol, "stock_name": name, "source": "tdx_tnf", "path": str(path)})
    return rows


def _tdx_exchange_from_filename(name: str) -> str:
    lower = name.lower()
    if lower.startswith("sh"):
        return "SH"
    if lower.startswith("sz"):
        return "SZ"
    if lower.startswith("bj"):
        return "BJ"
    return ""


def _tdx_record_name(record: bytes) -> str:
    candidates = [
        _decode_record_field(record[23:31]),
        _decode_record_field(record[23:39]),
        _decode_record_field(record[6:14]),
        _decode_record_field(record[6:24]),
    ]
    names = [name for name in candidates if _looks_like_stock_name(name)]
    return max(names, key=len) if names else ""


def _decode_record_field(raw: bytes, *, encoding: str = "gbk") -> str:
    value = raw.split(b"\x00", 1)[0].strip()
    if not value:
        return ""
    encodings = (encoding,) if encoding == "ascii" else ("gbk", "utf-8")
    for item in encodings:
        try:
            return _clean_text(value.decode(item, errors="ignore"))
        except UnicodeDecodeError:
            continue
    return ""


def _looks_like_stock_name(value: str) -> bool:
    if not value or value.isdigit():
        return False
    return any("\u4e00" <= char <= "\u9fff" for char in value) or any(char.isalpha() for char in value)


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).replace("\x00", "").strip()
    return "".join(char for char in text if char.isprintable()).strip()


def _merge_symbol_metadata(frames: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return _metadata_frame([])
    merged = pd.concat(non_empty, ignore_index=True)
    return (
        merged.drop_duplicates(subset=["stock_code"], keep="first")
        .sort_values("stock_code", kind="mergesort")
        .reset_index(drop=True)
    )


def _metadata_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=pd.Index(SYMBOL_METADATA_COLUMNS))
    frame = pd.DataFrame(rows, columns=SYMBOL_METADATA_COLUMNS)
    frame["stock_code"] = frame["stock_code"].map(normalize_symbol)
    frame["stock_name"] = frame["stock_name"].map(_clean_text)
    frame = frame.loc[frame["stock_code"].ne("") & frame["stock_name"].ne("")]
    return frame.loc[:, SYMBOL_METADATA_COLUMNS].reset_index(drop=True)


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result
