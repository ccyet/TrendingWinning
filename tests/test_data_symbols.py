from __future__ import annotations

from pathlib import Path

import pandas as pd

from trending_winning.data.symbols import load_symbol_metadata, load_tdx_symbol_metadata, resolve_symbol_names


def test_load_symbol_metadata_reads_sidecar_csv_with_chinese_columns(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    data_root.mkdir(parents=True)
    pd.DataFrame({"代码": ["000001", "600519.SH"], "名称": ["平安银行", "贵州茅台"]}).to_csv(
        tmp_path / "market" / "stock_names.csv",
        index=False,
    )

    metadata = load_symbol_metadata(data_root)

    by_code = metadata.set_index("stock_code")
    assert by_code.loc["000001.SZ", "stock_name"] == "平安银行"
    assert by_code.loc["600519.SH", "stock_name"] == "贵州茅台"
    assert by_code.loc["000001.SZ", "source"] == "sidecar_csv"


def test_load_tdx_symbol_metadata_reads_tnf_cache(tmp_path: Path) -> None:
    tdx_root = tmp_path / "new_tdx64"
    hq_cache = tdx_root / "T0002" / "hq_cache"
    hq_cache.mkdir(parents=True)
    _write_tnf(hq_cache / "shm.tnf", [("600519", "贵州茅台")])
    _write_tnf(hq_cache / "szm.tnf", [("000001", "平安银行")])

    metadata = load_tdx_symbol_metadata(tdx_root / "PYPlugins" / "user")

    by_code = metadata.set_index("stock_code")
    assert by_code.loc["600519.SH", "stock_name"] == "贵州茅台"
    assert by_code.loc["000001.SZ", "stock_name"] == "平安银行"
    assert by_code.loc["600519.SH", "source"] == "tdx_tnf"


def test_resolve_symbol_names_prefers_sidecar_over_defaults(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    data_root.mkdir(parents=True)
    pd.DataFrame({"stock_code": ["000001.SZ"], "stock_name": ["自定义银行"]}).to_csv(
        tmp_path / "market" / "symbols.csv",
        index=False,
    )

    names = resolve_symbol_names(["000001.SZ", "300750.SZ", "999999.SZ"], data_root=data_root)

    assert names["000001.SZ"] == "自定义银行"
    assert names["300750.SZ"] == "宁德时代"
    assert "999999.SZ" not in names


def _write_tnf(path: Path, rows: list[tuple[str, str]]) -> None:
    header = bytes(50)
    records: list[bytes] = []
    for code, name in rows:
        record = bytearray(314)
        record[0:6] = code.encode("ascii")
        name_bytes = name.encode("gbk")
        record[23 : 23 + len(name_bytes)] = name_bytes
        records.append(bytes(record))
    path.write_bytes(header + b"".join(records))
