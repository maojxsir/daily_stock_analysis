# -*- coding: utf-8 -*-
"""
市场情绪与风向分析模块。

数据来源（不编造数据）：
- 涨停池：akshare `stock_zt_pool_em`（东方财富）
- 跌停池：akshare `stock_zt_pool_dtgc_em`（东方财富）
- 市场活跃度（乐咕）：akshare `stock_market_activity_legu`
- 个股基本信息（行业/主营）：akshare `stock_individual_info_em`（东方财富）
- 概念/板块（兜底）：efinance `stock.get_belong_board`（东方财富）
- 可选新闻参考：SearchService（仅对涨停 TOP 概念做主题搜索，不逐股搜索）

本模块会被 `main.py` 和 `feishu_bot.py` 直接导入：
    from market_sentiment_analyzer import MarketSentimentAnalyzer
"""

from __future__ import annotations

import logging
import os
import random
import sys
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import datetime, date as date_cls
import math
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd  # type: ignore[import-not-found]

def _ensure_project_paths() -> None:
    """
    确保在“直接运行脚本”时也能导入项目模块。

    说明：
    - 项目核心模块位于 `./src/`，其中 `src/config.py` 需要以 `import config` 方式被解析。
    - 当通过 `python market_sentiment_analyzer.py` 运行时，默认 sys.path 未包含 `./src`，
      会导致 `ModuleNotFoundError: No module named 'config'`。
    """
    root = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(root, "src")
    # `src` 放前面，保证 `import config` 指向 `src/config.py`
    if src not in sys.path:
        sys.path.insert(0, src)
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_project_paths()

from config import get_config

logger = logging.getLogger(__name__)


class StockInfoProvider(ABC):
    """
    股票信息获取基础类。

    设计目标：
    - 统一“行业/主营”等基础信息的获取路径
    - 支持多数据源（AkShare / Tushare），按可用性与返回质量自动兜底
    """

    name: str = "base"

    @property
    def is_available(self) -> bool:
        return True

    @abstractmethod
    def get_individual_info(self, code: str) -> Dict[str, str]:
        """
        获取个股基础信息。

        Returns:
            dict: {"industry": "...", "main_business": "..."}，空字符串表示未知/获取失败。
        """


class AkshareStockInfoProvider(StockInfoProvider):
    name = "akshare"

    def __init__(self, analyzer: "MarketSentimentAnalyzer"):
        self._analyzer = analyzer

    @property
    def is_available(self) -> bool:
        try:
            import akshare as _  # type: ignore[import-not-found]

            return True
        except Exception:
            return False

    def get_individual_info(self, code: str) -> Dict[str, str]:
        try:
            import akshare as ak  # type: ignore[import-not-found]
        except Exception as e:
            raise MarketSentimentError(f"akshare import failed: {e}") from e

        df = self._analyzer._call_akshare_with_retry(
            lambda: ak.stock_individual_info_em(symbol=code),
            name=f"ak.stock_individual_info_em({code})",
            attempts=int(getattr(self._analyzer.config, "market_sentiment_individual_info_attempts", 2)),
            cache_key=None,
            break_on_network_error=True,
        )

        industry = ""
        main_business = ""
        if isinstance(df, pd.DataFrame) and not df.empty:
            for _, r in df.iterrows():
                item = str(r.get("item", "")).strip()
                value = str(r.get("value", "")).strip()
                if not item or not value or value.lower() == "nan":
                    continue
                if ("所属行业" in item) or (item == "行业") or ("行业" in item and not industry):
                    industry = value
                if ("主营业务" in item) or ("经营范围" in item):
                    main_business = value

        return {"industry": industry, "main_business": main_business}


class TushareStockInfoProvider(StockInfoProvider):
    name = "tushare"

    def __init__(self, analyzer: "MarketSentimentAnalyzer"):
        self._analyzer = analyzer

    @property
    def is_available(self) -> bool:
        token = getattr(self._analyzer.config, "tushare_token", None)
        if not token:
            return False
        try:
            import tushare as _  # type: ignore[import-not-found]

            return True
        except Exception:
            return False

    @staticmethod
    def _to_ts_code(code: str) -> str:
        # 这里用最简单的规则：6 位代码 + 交易所后缀
        # 说明：tushare 通常使用 ts_code，如 600000.SH / 000001.SZ
        if not code or len(code) != 6 or (not code.isdigit()):
            return code
        if code.startswith("6"):
            return f"{code}.SH"
        return f"{code}.SZ"

    @staticmethod
    def _from_ts_code(ts_code: str) -> str:
        # 600000.SH -> 600000
        s = str(ts_code or "").strip()
        if not s:
            return ""
        return s.split(".")[0]

    @staticmethod
    def _chunk_list(items: List[str], chunk_size: int) -> List[List[str]]:
        if chunk_size <= 0:
            return [items]
        return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]

    def get_individual_info(self, code: str) -> Dict[str, str]:
        try:
            import tushare as ts  # type: ignore[import-not-found]
        except Exception as e:
            raise MarketSentimentError(f"tushare import failed: {e}") from e

        token = getattr(self._analyzer.config, "tushare_token", None)
        if not token:
            return {"industry": "", "main_business": ""}

        pro = ts.pro_api(token)
        ts_code = self._to_ts_code(code)

        # 优先 stock_company 拿经营范围（近似主营），行业用 stock_basic 的 industry 字段
        industry = ""
        main_business = ""
        try:
            df_basic = pro.stock_basic(ts_code=ts_code, fields="ts_code,industry")
            if isinstance(df_basic, pd.DataFrame) and (not df_basic.empty):
                industry = str(df_basic.iloc[0].get("industry", "") or "").strip()
        except Exception as e:
            logger.debug(f"[sentiment] tushare.stock_basic failed {ts_code}: {e}")

        try:
            df_company = pro.stock_company(ts_code=ts_code, fields="ts_code,business_scope,main_business")
            if isinstance(df_company, pd.DataFrame) and (not df_company.empty):
                # tushare 字段可能不稳定，做兜底取值
                mb = df_company.iloc[0].get("main_business", "") or ""
                scope = df_company.iloc[0].get("business_scope", "") or ""
                main_business = str(mb or scope or "").strip()
        except Exception as e:
            logger.debug(f"[sentiment] tushare.stock_company failed {ts_code}: {e}")

        return {"industry": industry, "main_business": main_business}

    def get_individual_info_batch(self, codes: List[str], chunk_size: int = 200) -> Dict[str, Dict[str, str]]:
        """
        批量获取个股基础信息（行业/主营）。

        说明：
        - 优先使用 Tushare 批量接口，减少逐股请求导致的断连/限流。
        - 结果可能部分缺失：缺失项会返回空字符串，后续由其它 provider 兜底。

        Args:
            codes: 6 位股票代码列表（如 600000、000001）
            chunk_size: 每次请求的 ts_code 数量，默认 200（保守，避免 URL/服务端限制）
        """
        result: Dict[str, Dict[str, str]] = {}
        if not codes:
            return result

        token = getattr(self._analyzer.config, "tushare_token", None)
        if not token:
            return result

        try:
            import tushare as ts  # type: ignore[import-not-found]
        except Exception as e:
            raise MarketSentimentError(f"tushare import failed: {e}") from e

        pro = ts.pro_api(token)

        # 规范化：去重 + 过滤非法
        codes_norm = []
        for c in codes:
            c = _normalize_code(c)
            if c and c.isdigit() and len(c) == 6:
                codes_norm.append(c)
        codes_norm = list(dict.fromkeys(codes_norm))
        if not codes_norm:
            return result

        ts_codes = [self._to_ts_code(c) for c in codes_norm]
        ts_codes = [x for x in ts_codes if x]
        if not ts_codes:
            return result

        industry_by_ts: Dict[str, str] = {}
        area_by_ts: Dict[str, str] = {}
        mainbiz_by_ts: Dict[str, str] = {}

        for part in self._chunk_list(ts_codes, chunk_size):
            ts_code_str = ",".join(part)
            try:
                df_basic = pro.stock_basic(ts_code=ts_code_str, fields="ts_code,industry,area")
                if isinstance(df_basic, pd.DataFrame) and (not df_basic.empty):
                    for _, r in df_basic.iterrows():
                        tsc = str(r.get("ts_code", "") or "").strip()
                        ind = str(r.get("industry", "") or "").strip()
                        area = str(r.get("area", "") or "").strip()
                        if tsc:
                            industry_by_ts[tsc] = ind
                            area_by_ts[tsc] = area
            except Exception as e:
                logger.debug(f"[sentiment] tushare.stock_basic batch failed ({len(part)}): {e}")

            try:
                df_company = pro.stock_company(ts_code=ts_code_str, fields="ts_code,business_scope,main_business")
                if isinstance(df_company, pd.DataFrame) and (not df_company.empty):
                    for _, r in df_company.iterrows():
                        tsc = str(r.get("ts_code", "") or "").strip()
                        mb = str(r.get("main_business", "") or "").strip()
                        scope = str(r.get("business_scope", "") or "").strip()
                        if tsc:
                            mainbiz_by_ts[tsc] = mb or scope
            except Exception as e:
                logger.debug(f"[sentiment] tushare.stock_company batch failed ({len(part)}): {e}")

        # 合并回 6 位代码
        for tsc in ts_codes:
            code = self._from_ts_code(tsc)
            if not code:
                continue
            result[code] = {
                "industry": industry_by_ts.get(tsc, "") or "",
                "area": area_by_ts.get(tsc, "") or "",
                "main_business": mainbiz_by_ts.get(tsc, "") or "",
            }

        return result


class MarketSentimentError(Exception):
    """市场情绪分析基础异常。"""


@dataclass
class LimitStock:
    """单只涨停/跌停个股数据（含补全字段）。"""

    code: str
    name: str
    price: float = 0.0
    change_pct: float = 0.0
    turnover_rate: float = 0.0
    amount: str = ""  # 成交额（展示用，尽量保留原始单位/口径）
    amount_value: float = 0.0  # 成交额数值（用于排序，尽量统一为“元”口径）
    last_limit_time: str = ""  # 最后涨/跌停时间（展示用，取数据源原始格式）

    # 补全字段
    concepts: List[str] = field(default_factory=list)
    industry: str = ""
    area: str = ""  # 地域板块（省份/城市）
    main_business: str = ""

    # 原因字段：优先使用涨/跌停池自带字段，不做启发式推断。
    reason: str = ""


@dataclass
class MarketSentiment:
    """当日市场情绪快照（汇总数据）。"""

    date: str

    # 乐咕市场活跃度快照（ak.stock_market_activity_legu）
    market_activity_legu: List[Dict[str, Any]] = field(default_factory=list)

    # 市场广度（来自乐咕：上涨/下跌/平盘）
    up_count: int = 0
    down_count: int = 0
    flat_count: int = 0

    # 真实涨停/跌停（来自乐咕：真实涨停/真实跌停）
    real_limit_up: int = 0
    real_limit_down: int = 0
    # 真实涨停率/真实跌停率（以乐咕涨停/跌停为分母），范围 [0, 1]
    real_limit_up_ratio: float = 0.0
    real_limit_down_ratio: float = 0.0

    limit_up_count: int = 0
    limit_down_count: int = 0
    limit_up_stocks: List[LimitStock] = field(default_factory=list)
    limit_down_stocks: List[LimitStock] = field(default_factory=list)

    top_concepts: List[Dict[str, Any]] = field(default_factory=list)  # 涨停热点概念
    top_down_concepts: List[Dict[str, Any]] = field(default_factory=list)  # 跌停热点概念
    top_industries: List[Dict[str, Any]] = field(default_factory=list)

    # 热点概念新闻参考（按概念搜索，不逐股搜索）
    top_concept_news: List[Dict[str, Any]] = field(default_factory=list)

    sentiment_score: float = 0.0
    market_trend: str = ""
    trend_plot_path: str = ""

    # 子分数（便于解释“为什么是这个趋势”）
    strength_score: float = 0.0   # 涨停/跌停强弱
    quality_score: float = 0.0    # 真实涨停/跌停质量
    structure_score: float = 0.0  # 热点集中度/轮动结构


def _safe_float(v: Any) -> float:
    try:
        if v is None:
            return 0.0
        if isinstance(v, str):
            v = v.replace("%", "").strip()
        return float(v)
    except Exception:
        return 0.0


def _normalize_code(code: Any) -> str:
    s = str(code).strip()
    if not s:
        return ""
    # 部分数据源可能返回数字/短码，这里统一补齐为 6 位。
    if s.isdigit() and len(s) < 6:
        s = s.zfill(6)
    return s


def _pick_first_existing(row: pd.Series, candidates: List[str]) -> str:
    for c in candidates:
        if c in row.index:
            v = str(row.get(c, "")).strip()
            if v and v.lower() != "nan":
                return v
    return ""


def _pick_first_existing_with_col(row: pd.Series, candidates: List[str]) -> Tuple[str, str]:
    """
    在一组候选列里找到第一个可用值，并返回 (列名, 值)。

    用途：某些字段需要根据列名判断单位（例如 成交额(万元)）。
    """
    for c in candidates:
        if c in row.index:
            v = str(row.get(c, "")).strip()
            if v and v.lower() != "nan":
                return c, v
    return "", ""

def _split_concepts(raw: str, max_items: int = 8) -> List[str]:
    """
    将概念/题材/板块字段解析成列表。

    说明：不同数据源/不同接口字段格式不一致，这里做宽松解析并去重。
    """
    s = str(raw or "").strip()
    if not s or s.lower() == "nan":
        return []

    # 常见格式清洗：去括号、换行、引号
    for ch in ["[", "]", "【", "】", "(", ")", "（", "）", "\"", "'"]:
        s = s.replace(ch, " ")
    s = s.replace("\n", " ").replace("\r", " ").strip()

    # 多分隔符切分
    seps = ["、", ";", "；", ",", "，", "|", "/", "\\", " "]
    parts = [s]
    for sep in seps:
        next_parts: List[str] = []
        for p in parts:
            next_parts.extend([x for x in p.split(sep)])
        parts = next_parts

    cleaned: List[str] = []
    seen = set()
    for p in parts:
        name = str(p).strip()
        if not name or name.lower() == "nan":
            continue
        if name in seen:
            continue
        seen.add(name)
        cleaned.append(name)
        if max_items > 0 and len(cleaned) >= max_items:
            break
    return cleaned


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def _amount_to_value(v: Any) -> float:
    """
    将成交额（可能带单位）解析为数值，尽量统一为“元”。

    说明：
    - 若输入为数字：视为原始数值直接返回
    - 若输入形如 “1.23亿/4567万”：按中文单位换算
    - 无法解析则返回 0
    """
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        try:
            return float(v)
        except Exception:
            return 0.0

    s = str(v).strip()
    if not s or s.lower() == "nan":
        return 0.0

    import re

    m = re.match(r"^\s*([-+]?\d+(?:\.\d+)?)\s*([亿万]?)\s*$", s)
    if not m:
        # 有些源会返回带逗号的数字
        s2 = s.replace(",", "")
        m2 = re.match(r"^\s*([-+]?\d+(?:\.\d+)?)\s*([亿万]?)\s*$", s2)
        if not m2:
            return 0.0
        m = m2

    num = _safe_float(m.group(1))
    unit = m.group(2)
    if unit == "亿":
        return num * 1e8
    if unit == "万":
        return num * 1e4
    return num


def _time_to_sort_key(text: Any) -> int:
    """
    将“最后涨停/跌停时间”转为可排序的秒数（越大表示越晚）。

    支持格式：
    - HH:MM
    - HH:MM:SS
    - 其他格式解析失败则返回 0
    """
    s = _normalize_24h_time(text)
    if not s or s.lower() == "nan":
        return 0
    try:
        parts = s.split(":")
        if len(parts) == 2:
            h, m = int(parts[0]), int(parts[1])
            sec = 0
        elif len(parts) == 3:
            h, m, sec = int(parts[0]), int(parts[1]), int(parts[2])
        else:
            return 0
        if h < 0 or h > 23 or m < 0 or m > 59 or sec < 0 or sec > 59:
            return 0
        return h * 3600 + m * 60 + sec
    except Exception:
        return 0


def _normalize_24h_time(text: Any) -> str:
    """
    将时间文本规范为 24 小时制 HH:MM 或 HH:MM:SS（尽量保留秒）。

    支持常见输入：
    - HH:MM / H:MM
    - HH:MM:SS / H:MM:SS
    - 2026-01-01 14:30:00（自动取最后的时间部分）
    - 上午/下午 + 时间（例如 “下午2:05” -> “14:05”）
    - AM/PM（例如 “2:05 PM” -> “14:05”）
    """
    s = str(text or "").strip()
    if not s or s.lower() == "nan":
        return ""

    import re

    # 若包含日期，取最后一个空格后的时间部分
    if " " in s and re.search(r"\d{1,2}:\d{2}", s):
        s = s.split()[-1].strip()

    # 处理中文“时分秒”
    s = s.replace("时", ":").replace("分", "").replace("秒", "")

    # 识别上午/下午 或 AM/PM
    is_pm = False
    is_am = False
    if "下午" in s:
        is_pm = True
        s = s.replace("下午", "").strip()
    if "上午" in s:
        is_am = True
        s = s.replace("上午", "").strip()

    # AM/PM（大小写都支持）
    if re.search(r"\bPM\b", s, flags=re.IGNORECASE):
        is_pm = True
        s = re.sub(r"\bPM\b", "", s, flags=re.IGNORECASE).strip()
    if re.search(r"\bAM\b", s, flags=re.IGNORECASE):
        is_am = True
        s = re.sub(r"\bAM\b", "", s, flags=re.IGNORECASE).strip()

    # 抽取 H:MM(:SS) 或 HHMMSS（无冒号格式，如 092500）
    m = re.search(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", s)
    if not m:
        # 尝试匹配无冒号格式 HHMMSS（如 092500）或 HHMM（如 0925）
        m_no_colon = re.match(r"^(\d{2})(\d{2})(\d{2})?$", s)
        if m_no_colon:
            h = int(m_no_colon.group(1))
            mm = int(m_no_colon.group(2))
            ss_str = m_no_colon.group(3)
            sec = int(ss_str) if ss_str else None
            if sec is not None:
                return f"{h:02d}:{mm:02d}:{sec:02d}"
            else:
                return f"{h:02d}:{mm:02d}"
        return ""

    h = int(m.group(1))
    mm = int(m.group(2))
    ss = m.group(3)
    sec = int(ss) if ss is not None else None

    # 归一化小时
    if is_pm:
        # 下午/PM：1-11 -> +12，12 保持
        if 1 <= h <= 11:
            h += 12
    elif is_am:
        # 上午/AM：12 -> 0
        if h == 12:
            h = 0

    if h < 0 or h > 23 or mm < 0 or mm > 59 or (sec is not None and (sec < 0 or sec > 59)):
        return ""

    if sec is None:
        return f"{h:02d}:{mm:02d}"
    return f"{h:02d}:{mm:02d}:{sec:02d}"

def _compact_main_business(text: Any, max_len: int = 60) -> str:
    """
    归一化“主营业务/经营范围”文本，尽量只保留核心信息。

    目标：
    - 不引入 AI，总是可重复、可解释
    - 先去掉常见前缀，再取第一句/第一段
    - 仍过长则取前若干个关键分句，最后按长度截断
    """
    s = str(text or "").strip()
    if not s or s.lower() == "nan":
        return ""

    import re

    # 去掉常见前缀（不同数据源口径不一致）
    s = re.sub(r"^(主营业务|经营范围|主营业务范围|主要业务|业务范围)[:：\s]*", "", s)
    s = re.sub(r"^公司(主要)?从事[:：\s]*", "", s)
    s = re.sub(r"^主要从事[:：\s]*", "", s)

    # 优先取第一句/第一段
    parts = [p.strip() for p in re.split(r"[；;。\n\r]+", s) if p.strip()]
    if parts:
        s = parts[0]

    # 若仍较长，按分隔符取前 2~3 个“分句”
    if max_len > 0 and len(s) > max_len:
        segs = [x.strip() for x in re.split(r"[，,、]+", s) if x.strip()]
        if segs:
            s = "，".join(segs[:3])

    # 最终兜底：按长度截断
    if max_len > 0 and len(s) > max_len:
        s = s[:max_len] + "..."

    return s


class MarketSentimentAnalyzer:
    """使用真实数据源分析当日市场情绪。"""

    def __init__(self, search_service: Optional[Any] = None):
        self.config = get_config()
        self.search_service = search_service

        self._cache: Dict[str, Any] = {}
        self._cache_expiry: Dict[str, float] = {}

        self._cache_ttl = int(getattr(self.config, "market_sentiment_cache_ttl", 3600))

        # 股票基础信息获取提供者（按优先级兜底）
        self._stock_info_providers: List[StockInfoProvider] = [
            AkshareStockInfoProvider(self),
            TushareStockInfoProvider(self),
        ]

        # 0 表示“不限制”
        max_up = int(getattr(self.config, "market_sentiment_max_limit_up_stocks", 0))
        max_down = int(getattr(self.config, "market_sentiment_max_limit_down_stocks", 0))
        max_analyze = int(getattr(self.config, "market_sentiment_max_analyze_stocks", 0))
        self._max_limit_up: Optional[int] = None if max_up <= 0 else max_up
        self._max_limit_down: Optional[int] = None if max_down <= 0 else max_down
        self._max_analyze_stocks: Optional[int] = None if max_analyze <= 0 else max_analyze

        self._w_up = float(getattr(self.config, "market_sentiment_limit_up_weight", 1.0))
        self._w_down = float(getattr(self.config, "market_sentiment_limit_down_weight", 2.0))
        self._w_sector = float(getattr(self.config, "market_sentiment_sector_rotation_weight", 3.0))

    def _cache_get(self, key: str) -> Optional[Any]:
        exp = self._cache_expiry.get(key, 0.0)
        if key in self._cache and time.time() < exp:
            return self._cache[key]
        if key in self._cache:
            self._cache.pop(key, None)
            self._cache_expiry.pop(key, None)
        return None

    def _cache_set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        if value is None:
            return
        self._cache[key] = value
        self._cache_expiry[key] = time.time() + float(ttl or self._cache_ttl)

    def _sleep_for_akshare(self, attempt: int) -> None:
        # 防止触发反爬：首次随机延迟，失败后指数退避。
        if attempt <= 1:
            sleep_s = random.uniform(
                float(getattr(self.config, "akshare_sleep_min", 2.0)),
                float(getattr(self.config, "akshare_sleep_max", 5.0)),
            )
        else:
            sleep_s = min(2**attempt, float(getattr(self.config, "retry_max_delay", 30.0)))
        time.sleep(sleep_s)

    @staticmethod
    def _looks_like_proxy_ssl_issue(err: Exception) -> bool:
        msg = str(err)
        # 常见于 HTTPS 请求被错误代理到 HTTP 端口，或代理的 TLS 握手不兼容。
        keywords = [
            "WRONG_VERSION_NUMBER",
            "wrong version number",
            "SSLError",
            "TLSV1_ALERT_PROTOCOL_VERSION",
        ]
        return any(k in msg for k in keywords)

    @staticmethod
    def _looks_like_network_issue(err: Exception) -> bool:
        """
        判断是否属于“连接失败/超时”等网络问题。

        说明：AkShare 底层通常基于 requests/httpx，异常类型可能被包装，
        这里同时用关键字做兜底匹配。
        """
        msg = str(err)
        keywords = [
            "Connection aborted",
            "RemoteDisconnected",
            "ConnectionError",
            "ConnectTimeout",
            "ReadTimeout",
            "Read timed out",
            "timed out",
            "Timeout",
            "Max retries exceeded",
            "Connection reset by peer",
            "EOF occurred in violation of protocol",
            "Temporary failure in name resolution",
            "Name or service not known",
        ]
        return any(k in msg for k in keywords)

    @staticmethod
    @contextmanager
    def _without_proxy_env():
        """
        临时禁用代理环境变量，降低 AkShare 请求出现 SSL 兼容问题的概率。

        典型场景：代理环境变量配置不正确导致
        '[SSL: WRONG_VERSION_NUMBER] wrong version number'。
        """
        keys = [
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "http_proxy",
            "https_proxy",
            "ALL_PROXY",
            "all_proxy",
        ]
        backup: Dict[str, Optional[str]] = {k: os.environ.get(k) for k in keys}
        try:
            for k in keys:
                os.environ.pop(k, None)
            yield
        finally:
            for k, v in backup.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def _call_akshare_with_retry(
        self,
        fn,
        name: str,
        attempts: int = 3,
        cache_key: Optional[str] = None,
        break_on_network_error: bool = False,
    ):
        if cache_key:
            cached = self._cache_get(cache_key)
            if cached is not None:
                return cached

        last_err: Optional[Exception] = None
        try_without_proxy_on_error = True
        use_no_proxy = False

        for attempt in range(1, attempts + 1):
            try:
                self._sleep_for_akshare(attempt)
                ctx = self._without_proxy_env() if use_no_proxy else nullcontext()
                with ctx:
                    result = fn()
                if cache_key:
                    self._cache_set(cache_key, result)
                return result
            except Exception as e:
                last_err = e
                logger.warning(
                    f"[sentiment] {name} failed (attempt {attempt}/{attempts}) "
                    f"[{type(e).__name__}]: {e}"
                )
                if break_on_network_error and self._looks_like_network_issue(e):
                    # 网络类错误直接切换到其它 provider，不做长时间重试等待
                    break
                if try_without_proxy_on_error and (not use_no_proxy) and self._looks_like_proxy_ssl_issue(e):
                    # 下一次重试：临时禁用代理环境变量。
                    use_no_proxy = True

        logger.error(f"[sentiment] {name} failed after {attempts} attempts: {last_err}")
        return None

    def _extract_reason_from_pool_row(self, row: pd.Series, is_limit_up: bool) -> str:
        if is_limit_up:
            candidates = [
                "涨停原因类别",
                "涨停原因",
                "涨停原因明细",
                "原因",
                "原因类别",
            ]
        else:
            candidates = [
                "跌停原因类别",
                "跌停原因",
                "跌停原因明细",
                "原因",
                "原因类别",
            ]
        return _pick_first_existing(row, candidates)

    def _fetch_market_activity_legu(self) -> List[Dict[str, Any]]:
        """
        通过 AkShare 获取乐咕的市场活跃度快照。

        API：ak.stock_market_activity_legu()
        预期字段：item, value
        """
        logger.info("[sentiment] 开始获取市场活跃度数据（乐咕）...")
        try:
            import akshare as ak  # type: ignore[import-not-found]
        except Exception as e:
            logger.warning(f"[sentiment] akshare import failed for market activity: {e}")
            return []

        today = datetime.now().strftime("%Y%m%d")
        cache_key = f"market_activity_legu_{today}"
        df = self._call_akshare_with_retry(
            lambda: ak.stock_market_activity_legu(),
            name="ak.stock_market_activity_legu",
            attempts=2,
            cache_key=cache_key,
        )

        if not isinstance(df, pd.DataFrame) or df.empty:
            logger.info("[sentiment] 市场活跃度数据（乐咕）为空。")
            return []

        item_col = "item" if "item" in df.columns else None
        value_col = "value" if "value" in df.columns else None
        if not item_col or not value_col:
            try:
                return df.head(20).to_dict("records")
            except Exception:
                return []

        rows: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            item = str(row.get(item_col, "")).strip()
            value = row.get(value_col, "")
            if not item:
                continue
            rows.append({"item": item, "value": value})
        logger.info(f"[sentiment] 市场活跃度数据（乐咕）获取完成：{len(rows)} 行。")
        return rows

    @staticmethod
    def _legu_to_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        将乐咕 market_activity_legu 的 rows 转成 {指标: 数值}。

        rows 预期形如：{"item": "上涨", "value": 1538}
        """
        metrics: Dict[str, float] = {}
        for r in rows or []:
            item = str(r.get("item", "") or "").strip()
            if not item:
                continue
            metrics[item] = _safe_float(r.get("value", 0))
        return metrics

    def _fill_quality_from_legu(self, sentiment: MarketSentiment) -> None:
        metrics = self._legu_to_metrics(sentiment.market_activity_legu)

        up = int(metrics.get("上涨", 0) or 0)
        down = int(metrics.get("下跌", 0) or 0)
        flat = int(metrics.get("平盘", 0) or 0)

        sentiment.up_count = max(0, up)
        sentiment.down_count = max(0, down)
        sentiment.flat_count = max(0, flat)

        # 注意：这里用乐咕的“涨停/跌停”作为分母，因为你实际的池子长度可能因口径不同不一致
        zt_total = float(max(1, int(metrics.get("涨停", 0) or 0)))
        dt_total = float(max(1, int(metrics.get("跌停", 0) or 0)))

        sentiment.real_limit_up = int(metrics.get("真实涨停", 0) or 0)
        sentiment.real_limit_down = int(metrics.get("真实跌停", 0) or 0)
        sentiment.real_limit_up_ratio = float(max(0.0, sentiment.real_limit_up)) / zt_total
        sentiment.real_limit_down_ratio = float(max(0.0, sentiment.real_limit_down)) / dt_total

    def _fetch_limit_pool(self, is_limit_up: bool) -> pd.DataFrame:
        try:
            import akshare as ak  # type: ignore[import-not-found]
        except Exception as e:
            raise MarketSentimentError(f"akshare import failed: {e}") from e

        today = datetime.now().strftime("%Y%m%d")
        logger.info(f"[sentiment] 开始获取{'涨停' if is_limit_up else '跌停'}股票池：{today} ...")
        if is_limit_up:
            cache_key = f"zt_pool_{today}"
            df = self._call_akshare_with_retry(
                lambda: ak.stock_zt_pool_em(date=today),
                name="ak.stock_zt_pool_em",
                attempts=3,
                cache_key=cache_key,
            )
        else:
            cache_key = f"dt_pool_{today}"
            df = self._call_akshare_with_retry(
                lambda: ak.stock_zt_pool_dtgc_em(date=today),
                name="ak.stock_zt_pool_dtgc_em",
                attempts=3,
                cache_key=cache_key,
            )

        if df is None or getattr(df, "empty", True):
            logger.info(f"[sentiment] {'涨停' if is_limit_up else '跌停'}股票池为空。")
            return pd.DataFrame()
        if not isinstance(df, pd.DataFrame):
            try:
                return pd.DataFrame(df)
            except Exception:
                return pd.DataFrame()
        logger.info(f"[sentiment] {'涨停' if is_limit_up else '跌停'}股票池获取完成：{len(df)} 行。")
        return df

    def _parse_limit_stocks(self, df: pd.DataFrame, is_limit_up: bool) -> List[LimitStock]:
        if df is None or df.empty:
            return []

        max_n = self._max_limit_up if is_limit_up else self._max_limit_down

        code_candidates = ["代码", "股票代码", "code"]
        name_candidates = ["名称", "股票简称", "name"]
        price_candidates = ["最新价", "现价", "price"]
        change_candidates = ["涨跌幅", "涨幅", "change_pct"]
        turnover_candidates = ["换手率", "turnover_rate"]
        amount_candidates = ["成交额", "成交金额", "成交额(元)", "成交金额(元)", "成交额(万元)", "amount"]
        if is_limit_up:
            last_limit_time_candidates = ["最后封板时间", "最后涨停时间", "最终涨停时间", "封板时间", "最后涨停"]
        else:
            last_limit_time_candidates = ["最后封板时间", "最后跌停时间", "最终跌停时间", "封板时间", "最后跌停"]
        industry_candidates = ["所属行业", "行业", "行业名称", "所属行业名称"]
        concept_candidates = [
            "所属概念",
            "概念",
            "概念板块",
            "所属题材",
            "题材",
            "板块",
            "所属板块",
            "热点",
            "热点概念",
        ]

        stocks: List[LimitStock] = []
        for _, row in df.iterrows():
            code = _normalize_code(_pick_first_existing(row, code_candidates))
            name = _pick_first_existing(row, name_candidates)
            if not code or not name:
                continue

            price = _safe_float(_pick_first_existing(row, price_candidates))
            change_pct = _safe_float(_pick_first_existing(row, change_candidates))
            turnover_rate = _safe_float(_pick_first_existing(row, turnover_candidates))
            amount_col, amount_raw = _pick_first_existing_with_col(row, amount_candidates)
            amount_value = _amount_to_value(amount_raw)
            # 若列名显式为“万元”口径，则转换为元用于排序/展示
            if amount_col and ("万元" in amount_col) and amount_value:
                amount_value *= 1e4
            # 展示统一为“亿元”
            amount = f"{amount_value/1e8:.2f}" if amount_value else ""
            last_limit_time = _normalize_24h_time(_pick_first_existing(row, last_limit_time_candidates))
            reason = self._extract_reason_from_pool_row(row, is_limit_up=is_limit_up)
            industry = _pick_first_existing(row, industry_candidates)
            concepts_raw = _pick_first_existing(row, concept_candidates)
            concepts = _split_concepts(concepts_raw, max_items=8)

            stocks.append(
                LimitStock(
                    code=code,
                    name=name,
                    price=price,
                    change_pct=change_pct,
                    turnover_rate=turnover_rate,
                    amount=amount,
                    amount_value=amount_value,
                    last_limit_time=last_limit_time,
                    reason=reason,
                    industry=industry,
                    concepts=concepts,
                )
            )

            if max_n is not None and len(stocks) >= max_n:
                break
        logger.info(
            f"[sentiment] 解析{'涨停' if is_limit_up else '跌停'}个股完成：{len(stocks)} 只"
            + (f"（限制前 {max_n} 只）" if max_n is not None else "（不限制）")
        )
        return stocks

    def _enrich_basic_info(self, stock: LimitStock) -> None:
        """通过 provider 体系补全行业、地域与主营业务。"""
        info_key = f"ind_info_{stock.code}"
        cached = self._cache_get(info_key)
        if isinstance(cached, dict):
            stock.industry = cached.get("industry", "") or ""
            stock.area = cached.get("area", "") or ""
            stock.main_business = _compact_main_business(cached.get("main_business", "") or "")
            return

        # 如果在涨/跌停池中已经带出了行业/主营，则直接写入缓存，避免额外请求
        if (stock.industry or "").strip() or (stock.main_business or "").strip():
            stock.main_business = _compact_main_business(stock.main_business)
            self._cache_set(
                info_key,
                {"industry": (stock.industry or "").strip(), "area": (stock.area or "").strip(), "main_business": (stock.main_business or "").strip()},
            )
            return

        industry = ""
        area = ""
        main_business = ""
        used_provider = ""

        for p in self._stock_info_providers:
            if not getattr(p, "is_available", True):
                continue
            try:
                info = p.get_individual_info(stock.code)
                if isinstance(info, dict):
                    industry = str(info.get("industry", "") or "").strip()
                    area = str(info.get("area", "") or "").strip()
                    main_business = _compact_main_business(info.get("main_business", "") or "")
                if industry or main_business:
                    used_provider = getattr(p, "name", p.__class__.__name__)
                    break
            except Exception as e:
                logger.debug(f"[sentiment] {getattr(p, 'name', p.__class__.__name__)} 获取基础信息失败 {stock.code}: {e}")
                continue

        stock.industry = industry
        stock.area = area
        stock.main_business = _compact_main_business(main_business)
        self._cache_set(info_key, {"industry": industry, "area": area, "main_business": stock.main_business})
        if used_provider:
            logger.debug(f"[sentiment] 基础信息补全成功 {stock.code} via {used_provider}")

    def _prefetch_basic_info_with_tushare(self, stocks: List[LimitStock]) -> None:
        """
        用 Tushare 批量预取基础信息，减少逐股请求带来的断连/限流。

        说明：
        - 仅对“缓存未命中”的股票生效
        - 批量结果写入 cache，后续 `_enrich_basic_info` 会直接命中
        """
        provider = next((p for p in self._stock_info_providers if isinstance(p, TushareStockInfoProvider)), None)
        if not provider or (not provider.is_available):
            return

        need_codes: List[str] = []
        for s in stocks:
            key = f"ind_info_{s.code}"
            if self._cache_get(key) is None:
                need_codes.append(s.code)

        if not need_codes:
            return

        try:
            batch = provider.get_individual_info_batch(need_codes, chunk_size=200)
        except Exception as e:
            logger.debug(f"[sentiment] tushare 批量预取失败（已忽略）: {e}")
            return

        if not isinstance(batch, dict) or not batch:
            return

        for code, info in batch.items():
            if not isinstance(info, dict):
                continue
            industry = str(info.get("industry", "") or "").strip()
            area = str(info.get("area", "") or "").strip()
            main_business = _compact_main_business(info.get("main_business", "") or "")
            self._cache_set(f"ind_info_{code}", {"industry": industry, "area": area, "main_business": main_business})

    def _enrich_concepts(self, stock: LimitStock, top_n: int = 5) -> None:
        """通过 efinance（兜底数据源）补全概念/板块信息。"""
        key = f"concepts_{stock.code}"
        cached = self._cache_get(key)
        if isinstance(cached, list):
            stock.concepts = [str(x).strip() for x in cached if str(x).strip()]
            return

        # 如果在涨/跌停池中已经带出了概念/板块，则直接写入缓存，避免额外请求
        if stock.concepts:
            concepts_clean = [str(x).strip() for x in (stock.concepts or []) if str(x).strip()]
            stock.concepts = concepts_clean
            self._cache_set(key, concepts_clean)
            return

        concepts: List[str] = []
        try:
            import efinance as ef  # type: ignore[import-not-found]

            df = ef.stock.get_belong_board(stock.code)
            if isinstance(df, pd.DataFrame) and not df.empty:
                col_candidates = ["板块名称", "name"]
                col = next((c for c in col_candidates if c in df.columns), None)
                if col:
                    concepts = [str(x).strip() for x in df[col].head(top_n).tolist() if str(x).strip()]
        except Exception as e:
            logger.debug(f"[sentiment] efinance 获取概念/板块失败 {stock.code}: {e}")

        stock.concepts = concepts
        self._cache_set(key, concepts)

    @staticmethod
    def _top_k(items: List[str], k: int = 10) -> List[Dict[str, Any]]:
        counts: Dict[str, int] = {}
        for it in items:
            it = str(it).strip()
            if not it:
                continue
            counts[it] = counts.get(it, 0) + 1
        return [{"name": name, "count": cnt} for name, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:k]]

    @staticmethod
    def _count_concepts(stocks: List[LimitStock]) -> Dict[str, int]:
        """
        统计概念出现次数。

        说明：一只股票可能属于多个概念，因此会对多个概念同时计数。
        """
        counts: Dict[str, int] = {}
        for s in stocks:
            for c in (s.concepts or []):
                name = str(c).strip()
                if not name:
                    continue
                counts[name] = counts.get(name, 0) + 1
        return counts

    @staticmethod
    def _sort_stocks_by_first_concept(stocks: List[LimitStock]) -> List[LimitStock]:
        """
        Sort stocks by first concept keyword, grouping similar concepts together.

        Sorting rules:
        1. Group by first concept (if exists)
        2. Within each group, sort by limit time (earlier first)
        3. Stocks without concepts are placed at the end

        Args:
            stocks: List of limit stocks

        Returns:
            Sorted list of limit stocks
        """
        if not stocks:
            return stocks

        def get_first_concept(stock: LimitStock) -> str:
            """Get the first concept or empty string."""
            if stock.concepts and len(stock.concepts) > 0:
                return str(stock.concepts[0]).strip()
            return ""

        def get_sort_key(stock: LimitStock) -> Tuple[int, str, str]:
            """
            Sort key: (has_concept, first_concept, limit_time)
            - has_concept: 0 if has concept, 1 if not (so stocks with concepts come first)
            - first_concept: the first concept name for grouping
            - limit_time: limit time for secondary sorting
            """
            first_concept = get_first_concept(stock)
            has_concept = 0 if first_concept else 1
            limit_time = getattr(stock, 'last_limit_time', '') or ''
            return (has_concept, first_concept, limit_time)

        return sorted(stocks, key=get_sort_key)

    def _analyze_distributions(self, limit_up_stocks: List[LimitStock]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        all_concepts: List[str] = []
        all_industries: List[str] = []
        for s in limit_up_stocks:
            all_concepts.extend(s.concepts or [])
            if s.industry:
                all_industries.append(s.industry)
        return self._top_k(all_concepts, k=10), self._top_k(all_industries, k=10)

    def _analyze_down_concepts(self, limit_down_stocks: List[LimitStock]) -> List[Dict[str, Any]]:
        """Analyze concept distribution for limit-down stocks."""
        all_concepts: List[str] = []
        for s in limit_down_stocks:
            all_concepts.extend(s.concepts or [])
        return self._top_k(all_concepts, k=10)

    def _persist_daily_concept_stats(
        self,
        sentiment: MarketSentiment,
        concept_counts_up: Dict[str, int],
        concept_counts_down: Dict[str, int],
    ) -> None:
        """
        将当日统计数据写入 SQLite（通过项目内 DatabaseManager）。

        该步骤为 best-effort：写库失败不会影响报告生成。
        """
        try:
            from storage import get_db

            db = get_db()
            target_date = datetime.strptime(sentiment.date, "%Y-%m-%d").date()

            db.upsert_market_sentiment_daily(
                target_date=target_date,
                limit_up_count=sentiment.limit_up_count,
                limit_down_count=sentiment.limit_down_count,
            )

            db.replace_market_concept_daily_stats(
                target_date=target_date,
                limit_type="up",
                concept_counts=concept_counts_up,
            )
            db.replace_market_concept_daily_stats(
                target_date=target_date,
                limit_type="down",
                concept_counts=concept_counts_down,
            )

            logger.info("[sentiment] 已写入数据库：市场情绪汇总 + 概念统计。")
        except Exception as e:
            logger.warning(f"[sentiment] 写入数据库失败（已忽略，不影响报告生成）: {e}")

    def _search_topic_news(self, query: str, max_results: int = 3) -> Optional[Any]:
        """
        通过 SearchService 做“主题搜索”（优先直接调用 provider），不逐股搜索。

        Returns:
            SearchResponse 类似对象或 None
        """
        if not self.search_service:
            return None

        providers = getattr(self.search_service, "_providers", None)
        if isinstance(providers, list) and providers:
            # 优先使用 Tavily：其 SDK 支持 days 参数，能更好控制“最近一周”
            ordered = sorted(
                providers,
                key=lambda p: 0 if str(getattr(p, "name", "")).lower() == "tavily" else 1,
            )
            for p in ordered:
                try:
                    if not getattr(p, "is_available", False):
                        continue
                    resp = p.search(query, max_results=max_results)
                    if getattr(resp, "success", False) and getattr(resp, "results", None):
                        return resp
                except Exception:
                    continue

        # 兜底：复用“个股新闻搜索”接口作为通用搜索包装（不推荐，但保证可用性）
        try:
            if hasattr(self.search_service, "search_stock_news"):
                return self.search_service.search_stock_news(
                    stock_code="",
                    stock_name=query,
                    max_results=max_results,
                )
        except Exception:
            return None
        return None

    def _search_top_concepts_news(self, top_concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        对涨停概念 TOP5 做新闻搜索（替代逐股新闻搜索）。

        输出仅用于“参考”，只记录标题/来源等，不做原因推断。
        """
        if not self.search_service:
            return []
        if not top_concepts:
            return []

        # 强调时效性：只希望拿到最近一周的新闻
        week_tag = "最近一周"
        results: List[Dict[str, Any]] = []
        logger.info("[sentiment] 开始按涨停概念 TOP5 搜索新闻参考（非逐股）...")
        for c in top_concepts[:5]:
            concept = str(c.get("name", "")).strip()
            if not concept:
                continue

            query = f"{concept} 概念 板块 涨停 领涨 {week_tag}"
            logger.info(f"[sentiment] 概念新闻查询：{concept}")
            resp = self._search_topic_news(query, max_results=3)
            if not resp or not getattr(resp, "success", False):
                continue

            items = []
            for r in (getattr(resp, "results", None) or [])[:3]:
                title = str(getattr(r, "title", "") or "").strip()
                source = str(getattr(r, "source", "") or "").strip()
                published_date = str(getattr(r, "published_date", "") or "").strip()
                snippet = str(getattr(r, "snippet", "") or "").strip()
                items.append(
                    {
                        "title": title,
                        "source": source,
                        "published_date": published_date,
                        "snippet": snippet,
                    }
                )

            results.append(
                {
                    "concept": concept,
                    "query": getattr(resp, "query", query),
                    "provider": getattr(resp, "provider", ""),
                    "items": items,
                }
            )

            time.sleep(0.3)

        logger.info(f"[sentiment] 概念新闻搜索完成：{len(results)} 个概念有结果。")
        return results

    def _calculate_sentiment_score(self, sentiment: MarketSentiment) -> float:
        # 1) 强度：涨停多、跌停少 => 更强
        base = sentiment.limit_up_count * self._w_up - sentiment.limit_down_count * self._w_down
        # 使用 tanh 做平滑压缩，避免极端日把分数拉爆
        strength = math.tanh(float(base) / 60.0) * 100.0

        # 2) 质量：真实涨停率高、真实跌停率低 => 更强（来自乐咕）
        quality = _clamp(sentiment.real_limit_up_ratio - sentiment.real_limit_down_ratio, -1.0, 1.0) * 100.0

        # 3) 结构：热点概念集中度（简单版本：出现次数>=3 的概念数）
        hot = 0
        if sentiment.top_concepts:
            hot = len([c for c in sentiment.top_concepts if c.get("count", 0) >= 3])
        structure = _clamp(float(hot) / 5.0, 0.0, 1.0) * 100.0

        sentiment.strength_score = float(strength)
        sentiment.quality_score = float(quality)
        sentiment.structure_score = float(structure)

        # 总分：强度为主，其次质量，结构用于微调
        total = 0.70 * strength + 0.25 * quality + 0.05 * (structure - 50.0)
        return _clamp(total, -100.0, 100.0)

    @staticmethod
    def _trend_label(score: float) -> str:
        if score >= 60:
            return "强势"
        if score >= 30:
            return "偏强"
        if score >= -30:
            return "震荡"
        if score >= -60:
            return "偏弱"
        return "弱势"

    def get_market_sentiment(self) -> MarketSentiment:
        today = datetime.now().strftime("%Y-%m-%d")
        sentiment = MarketSentiment(date=today)

        logger.info(f"[sentiment] 开始构建市场情绪快照：{today} ...")

        # 0) 市场活跃度快照
        sentiment.market_activity_legu = self._fetch_market_activity_legu()
        self._fill_quality_from_legu(sentiment)

        # 1) 获取涨停/跌停池并解析
        zt_df = self._fetch_limit_pool(is_limit_up=True)
        dt_df = self._fetch_limit_pool(is_limit_up=False)
        sentiment.limit_up_stocks = self._parse_limit_stocks(zt_df, is_limit_up=True)
        sentiment.limit_down_stocks = self._parse_limit_stocks(dt_df, is_limit_up=False)

        # 1.1) 排序：按最后涨/跌停时间升序（从早到晚）-> 成交额降序 -> 换手率降序
        sentiment.limit_up_stocks.sort(
            key=lambda s: (
                _time_to_sort_key(getattr(s, "last_limit_time", "")),  # 时间升序
                -float(getattr(s, "amount_value", 0.0) or 0.0),        # 成交额降序
                -float(getattr(s, "turnover_rate", 0.0) or 0.0),       # 换手率降序
            ),
            reverse=False,
        )
        sentiment.limit_down_stocks.sort(
            key=lambda s: (
                _time_to_sort_key(getattr(s, "last_limit_time", "")),  # 时间升序
                -float(getattr(s, "amount_value", 0.0) or 0.0),        # 成交额降序
                -float(getattr(s, "turnover_rate", 0.0) or 0.0),       # 换手率降序
            ),
            reverse=False,
        )
        sentiment.limit_up_count = len(sentiment.limit_up_stocks)
        sentiment.limit_down_count = len(sentiment.limit_down_stocks)
        logger.info(f"[sentiment] 数量统计：涨停={sentiment.limit_up_count}，跌停={sentiment.limit_down_count}")

        # 2) 补全行业/主营 + 概念（是否限制取决于配置）
        up_stocks = (
            sentiment.limit_up_stocks
            if self._max_analyze_stocks is None
            else sentiment.limit_up_stocks[: self._max_analyze_stocks]
        )
        # 2.0) 先批量预取基础信息，降低逐股调用失败概率
        self._prefetch_basic_info_with_tushare(up_stocks)
        logger.info(
            f"[sentiment] 开始补全涨停个股信息（行业/主营 + 概念）：{len(up_stocks)} 只"
            + ("（不限制）" if self._max_analyze_stocks is None else f"（限制前 {self._max_analyze_stocks} 只）")
        )
        start_enrich = time.time()
        for i, s in enumerate(up_stocks, 1):
            if i == 1 or i % 20 == 0:
                logger.info(f"[sentiment] 涨停补全进度：{i}/{len(up_stocks)}")
            self._enrich_basic_info(s)
            self._enrich_concepts(s, top_n=5)
        logger.info(f"[sentiment] 涨停补全完成，耗时 {time.time() - start_enrich:.1f} 秒。")

        down_stocks = (
            sentiment.limit_down_stocks
            if self._max_analyze_stocks is None
            else sentiment.limit_down_stocks[: self._max_analyze_stocks]
        )
        # 2.0) 先批量预取基础信息，降低逐股调用失败概率
        self._prefetch_basic_info_with_tushare(down_stocks)
        logger.info(
            f"[sentiment] 开始补全跌停个股信息（行业/主营 + 概念）：{len(down_stocks)} 只"
            + ("（不限制）" if self._max_analyze_stocks is None else f"（限制前 {self._max_analyze_stocks} 只）")
        )
        start_enrich = time.time()
        for i, s in enumerate(down_stocks, 1):
            if i == 1 or i % 20 == 0:
                logger.info(f"[sentiment] 跌停补全进度：{i}/{len(down_stocks)}")
            self._enrich_basic_info(s)
            self._enrich_concepts(s, top_n=5)
        logger.info(f"[sentiment] 跌停补全完成，耗时 {time.time() - start_enrich:.1f} 秒。")

        # 3) 分布统计（基于“已补全概念/行业”的涨停个股集合）
        # 说明：当开启 market_sentiment_max_analyze_stocks 限制时，只有前 N 只会补全概念；
        # 为避免统计被大量“未补全”稀释，这里以实际补全的集合为准。
        sentiment.top_concepts, sentiment.top_industries = self._analyze_distributions(up_stocks)
        sentiment.top_down_concepts = self._analyze_down_concepts(down_stocks)
        logger.info(
            f"[sentiment] 分布统计完成：涨停概念TOP={len(sentiment.top_concepts)}，跌停概念TOP={len(sentiment.top_down_concepts)}，行业TOP={len(sentiment.top_industries)}"
        )

        # 3.2) 入库：当日统计（涨停/跌停概念统计）
        concept_counts_up = self._count_concepts(up_stocks)
        concept_counts_down = self._count_concepts(down_stocks)
        self._persist_daily_concept_stats(sentiment, concept_counts_up, concept_counts_down)

        # 3.3) 生成趋势图（从数据库读取最近 N 天数据）
        sentiment.trend_plot_path = self._generate_concept_trend_plot(days=30) or ""

        # 3.5) 按热点概念搜索新闻（不逐股）
        if self.search_service:
            sentiment.top_concept_news = self._search_top_concepts_news(sentiment.top_concepts)
        else:
            sentiment.top_concept_news = []
            logger.info("[sentiment] 未配置搜索服务，跳过概念新闻搜索。")

        # 4) 评分与趋势判断
        sentiment.sentiment_score = self._calculate_sentiment_score(sentiment)
        sentiment.market_trend = self._trend_label(sentiment.sentiment_score)
        logger.info(f"[sentiment] 计算完成：情绪评分={sentiment.sentiment_score:.1f}，趋势={sentiment.market_trend}")

        return sentiment

    def _generate_concept_trend_plot(self, days: int = 30) -> Optional[str]:
        """
        从本地数据库绘制“涨停/跌停家数 + 概念分布”趋势图。

        该步骤为 best-effort：绘图失败不会影响报告生成。
        """
        try:
            from datetime import timedelta
            from pathlib import Path

            from storage import get_db

            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=int(days))
            out_path = Path("./reports/market_sentiment_trends.png")
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # 延迟导入：避免在无 matplotlib 环境下影响主流程
            from plot_market_sentiment_concepts import plot_market_sentiment_trends

            plot_market_sentiment_trends(
                db=get_db(),
                start_date=start_date,
                end_date=end_date,
                out_path=str(out_path),
                top_n=8,
                style="stacked",
            )
            logger.info(f"[sentiment] 趋势图已生成：{out_path.resolve()}")
            return str(out_path)
        except Exception as e:
            logger.warning(f"[sentiment] 生成趋势图失败（已忽略，不影响报告生成）: {e}")
            return None

    def generate_report(self, sentiment: MarketSentiment) -> str:
        lines: List[str] = []
        lines.append(f"# 📊 {sentiment.date} 市场情绪与风向分析")
        lines.append("")

        if sentiment.market_activity_legu:
            lines.append("## 市场活跃度")
            lines.append("")
            lines.append("| 指标 | 数值 |")
            lines.append("|---|---|")
            for r in sentiment.market_activity_legu[:20]:
                item = str(r.get("item", "")).strip()
                value = r.get("value", "")
                if not item:
                    item = str(r.get("指标", "")).strip() or str(r.get("name", "")).strip()
                    value = r.get("数值", "") or r.get("value", "")
                if not item:
                    continue
                lines.append(f"| {item} | {value} |")
            lines.append("")

        lines.append("## 一、情绪概览")
        lines.append("")
        # 更柔和、少冗余的概览展示
        if (sentiment.real_limit_up or sentiment.real_limit_down) or (
            sentiment.real_limit_up_ratio > 0 or sentiment.real_limit_down_ratio > 0
        ):
            lines.append(
                f"- **涨停**：{sentiment.limit_up_count}（真实 {sentiment.real_limit_up}），"
                f"**跌停**：{sentiment.limit_down_count}（真实 {sentiment.real_limit_down}）"
            )
        else:
            lines.append(f"- **涨停**：{sentiment.limit_up_count}，**跌停**：{sentiment.limit_down_count}")

        if (sentiment.up_count + sentiment.down_count + sentiment.flat_count) > 0:
            up_text = f'<span style="color:red">上涨 {sentiment.up_count}</span>'
            down_text = f'<span style="color:green">下跌 {sentiment.down_count}</span>'
            lines.append(
                f"- **涨跌家数**: {up_text} / {down_text} / 平盘 {sentiment.flat_count}"
            )

        lines.append(f"- **情绪**: {sentiment.sentiment_score:.1f}/100（{sentiment.market_trend}）")
        if sentiment.trend_plot_path:
            lines.append(f"- **趋势图**: {sentiment.trend_plot_path}")
        lines.append("")

        if sentiment.top_concepts:
            lines.append("## 二、涨停热点概念（按出现次数）")
            lines.append("")
            lines.append("| 概念 | 涨停数量 |")
            lines.append("|---|---:|")
            for c in sentiment.top_concepts[:10]:
                lines.append(f"| {c['name']} | {c['count']} |")
            lines.append("")

        if sentiment.top_industries:
            lines.append("## 三、涨停行业分布（按出现次数）")
            lines.append("")
            lines.append("| 行业 | 涨停数量 |")
            lines.append("|---|---:|")
            for c in sentiment.top_industries[:10]:
                lines.append(f"| {c['name']} | {c['count']} |")
            lines.append("")

        if sentiment.top_down_concepts:
            lines.append("## 四、跌停热点概念（按出现次数）")
            lines.append("")
            lines.append("| 概念 | 跌停数量 |")
            lines.append("|---|---:|")
            for c in sentiment.top_down_concepts[:10]:
                lines.append(f"| {c['name']} | {c['count']} |")
            lines.append("")

        # if sentiment.limit_up_stocks:
        #     lines.append("## 五、涨停股票（前40）")
        #     lines.append("")
        #     lines.append(f"共 {len(sentiment.limit_up_stocks)} 只涨停股票，详见图片。")
        #     lines.append("")
        #     lines.append("<!-- LIMIT_UP_IMAGE_PLACEHOLDER -->")
        #     lines.append("")

        # if sentiment.limit_down_stocks:
        #     lines.append("## 六、跌停股票（前30）")
        #     lines.append("")
        #     lines.append(f"共 {len(sentiment.limit_down_stocks)} 只跌停股票，详见图片。")
        #     lines.append("")
        #     lines.append("<!-- LIMIT_DOWN_IMAGE_PLACEHOLDER -->")
        #     lines.append("")

        lines.append("---")
        lines.append(f"*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        return "\n".join(lines)

    def _generate_limit_table_csv(
        self,
        stocks: List[LimitStock],
        output_path: str,
        is_limit_up: bool = True,
    ) -> Optional[str]:
        """
        Generate a CSV file for limit-up or limit-down stocks.
        Stock code is always 6 digits (zero-padded).
        """
        if not stocks:
            return None

        try:
            import csv

            if is_limit_up:
                headers = ['代码', '名称', '涨跌幅', '换手率', '成交额(亿)', '涨停时间', '概念板块', '主营业务']
            else:
                headers = ['代码', '名称', '涨跌幅', '换手率', '成交额(亿)', '概念板块', '主营业务']

            with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                for s in stocks:
                    concepts = "、".join((s.concepts or [])[:3]) if s.concepts else ""
                    main_biz = (s.main_business or "").strip()
                    # Ensure stock code is 6 digits (zero-padded)
                    code = str(s.code or "").strip()
                    if code.isdigit() and len(code) < 6:
                        code = code.zfill(6)

                    if is_limit_up:
                        row = [
                            code,
                            s.name,
                            f"{s.change_pct:.2f}%",
                            f"{s.turnover_rate:.2f}%",
                            (s.amount or "").strip(),
                            (s.last_limit_time or "").strip(),
                            concepts,
                            main_biz,
                        ]
                    else:
                        row = [
                            code,
                            s.name,
                            f"{s.change_pct:.2f}%",
                            f"{s.turnover_rate:.2f}%",
                            (s.amount or "").strip(),
                            concepts,
                            main_biz,
                        ]
                    writer.writerow(row)

            logger.info(f"[sentiment] CSV文件已生成: {output_path}")
            return output_path

        except Exception as e:
            logger.warning(f"[sentiment] 生成CSV文件失败: {e}")
            return None

    def _generate_limit_table_image(
        self,
        stocks: List[LimitStock],
        title: str,
        output_path: str,
        is_limit_up: bool = True,
    ) -> Optional[str]:
        """
        Generate a table image for limit-up or limit-down stocks.
        Columns are compact except for main_business which shows more content.
        """
        if not stocks:
            return None

        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend to avoid blocking
            import matplotlib.pyplot as plt
            import matplotlib.font_manager as fm

            # Try to use Chinese font (cross-platform: Linux/Docker, macOS, Windows)
            # Rebuild font cache to ensure newly installed fonts are detected
            fm._load_fontmanager(try_read_cache=False)

            chinese_fonts = [
                # Linux/Docker fonts (priority for server deployment)
                'Noto Sans CJK SC', 'Noto Sans CJK', 'Source Han Sans SC',
                'Noto Serif CJK SC', 'Noto Serif CJK',  # Serif variant (also common)
                'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Droid Sans Fallback',
                # macOS fonts
                'PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'Heiti TC', 'Songti SC',
                # Windows fonts
                'SimHei', 'Microsoft YaHei', 'SimSun', 'FangSong',
                # Universal fallback
                'Arial Unicode MS',
            ]
            font_name = None

            # Method 1: Try known font names
            for cf in chinese_fonts:
                try:
                    path = fm.findfont(cf, fallback_to_default=False)
                    if path and 'DejaVu' not in path and 'LastResort' not in path:
                        font_name = cf
                        logger.debug(f"[sentiment] Found Chinese font: {cf} at {path}")
                        break
                except Exception:
                    continue

            # Method 2: Scan all available fonts for CJK support
            if not font_name:
                for font in fm.fontManager.ttflist:
                    fname = font.name.lower()
                    # Look for fonts with CJK-related keywords
                    if any(kw in fname for kw in ['cjk', 'noto', 'pingfang', 'heiti', 'simhei',
                                                   'yahei', 'simsun', 'songti', 'wenquanyi',
                                                   'source han', 'droid sans fallback']):
                        font_name = font.name
                        logger.debug(f"[sentiment] Found CJK font via scan: {font_name} at {font.fname}")
                        break

            if font_name:
                plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
                plt.rcParams['font.family'] = 'sans-serif'
            else:
                # Fallback: warn user and use default font (Chinese chars may not render)
                logger.warning(
                    "[sentiment] No Chinese font found. Chinese characters may not display correctly. "
                    "Consider installing: fonts-noto-cjk (apt), google-noto-sans-cjk-fonts (yum/dnf), "
                    "or noto-fonts-cjk (pacman). "
                    "After installation, delete ~/.cache/matplotlib to refresh font cache."
                )
                plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False

            # Prepare data - compact columns, main_business shows more
            if is_limit_up:
                headers = ['代码', '名称', '涨幅', '换手', '成交额', '时间', '概念', '主营业务']
                # Column widths: compact for most, wide for main_business
                cell_widths = [0.6, 0.6, 0.5, 0.5, 0.6, 0.5, 1.2, 3.5]
            else:
                headers = ['代码', '名称', '跌幅', '换手', '成交额', '概念', '主营业务']
                cell_widths = [0.6, 0.6, 0.5, 0.5, 0.6, 1.2, 3.5]

            data = []
            for s in stocks[:40]:  # Limit to 40 rows
                concepts = "、".join((s.concepts or [])[:2]) if s.concepts else ""
                if len(concepts) > 10:
                    concepts = concepts[:8] + ".."
                main_biz = (s.main_business or "").strip()
                # Main business truncated to fit column width (max 28 chars)
                if len(main_biz) > 28:
                    main_biz = main_biz[:26] + ".."

                if is_limit_up:
                    row = [
                        s.code,
                        s.name[:4] if len(s.name) > 4 else s.name,
                        f"{s.change_pct:.1f}%",
                        f"{s.turnover_rate:.1f}%",
                        (s.amount or "").strip(),
                        (s.last_limit_time or "").strip(),
                        concepts,
                        main_biz,
                    ]
                else:
                    row = [
                        s.code,
                        s.name[:4] if len(s.name) > 4 else s.name,
                        f"{s.change_pct:.1f}%",
                        f"{s.turnover_rate:.1f}%",
                        (s.amount or "").strip(),
                        concepts,
                        main_biz,
                    ]
                data.append(row)

            # Calculate figure size
            n_rows = len(data) + 1  # +1 for header
            n_cols = len(headers)
            cell_height = 0.30
            fig_width = sum(cell_widths) + 0.3
            fig_height = n_rows * cell_height + 0.4

            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            ax.axis('off')
            # 固定标题位置（Axes 坐标系）
            TITLE_Y = 0.965   # 标题位置
            TABLE_TOP = 0.92  # 表格顶部（下面还会用）

            ax.text(
                0.5,
                TITLE_Y,
                title,
                ha='center',
                va='center',
                fontsize=11,
                fontweight='bold',
                transform=ax.transAxes,
            )

            # Create table
            table = ax.table(
                cellText=data,
                colLabels=headers,
                cellLoc='center',
                colWidths=[w / sum(cell_widths) for w in cell_widths],
                bbox=[0, 0, 1, TABLE_TOP],
            )

            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.1, 1.4)

            # Header style
            for j in range(n_cols):
                cell = table[(0, j)]
                cell.set_facecolor('#4472C4')
                cell.set_text_props(color='white', fontweight='bold', fontsize=8)

            # Row styles (alternating colors)
            for i in range(1, n_rows):
                for j in range(n_cols):
                    cell = table[(i, j)]
                    if i % 2 == 0:
                        cell.set_facecolor('#E7E6E6')
                    else:
                        cell.set_facecolor('#FFFFFF')
                    # Highlight change percentage column
                    if j == 2:  # Change pct column
                        try:
                            pct_val = float(data[i-1][2].replace('%', ''))
                            if pct_val > 0:
                                cell.set_text_props(color='#CC0000')
                            elif pct_val < 0:
                                cell.set_text_props(color='#00AA00')
                        except:
                            pass
                    # Left-align main_business column (last column)
                    if j == n_cols - 1:
                        cell._loc = 'left'
                        cell.PAD = 0.02

            # Also left-align header for main_business column
            header_cell = table[(0, n_cols - 1)]
            header_cell._loc = 'left'

            fig.subplots_adjust(
                left=0.01,
                right=0.99,
                top=0.92,
                bottom=0.02
            )

            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', pad_inches=0.1)
            plt.close(fig)

            logger.info(f"[sentiment] 表格图片已生成: {output_path}")
            return output_path

        except Exception as e:
            logger.warning(f"[sentiment] 生成表格图片失败: {e}")
            return None

    def _send_image_to_feishu(self, image_path: str) -> bool:
        """
        Send image to Feishu using rich text format (to pass keyword check).

        Args:
            image_path: Path to the image file

        Returns:
            True if sent successfully, False otherwise
        """
        import json
        import requests

        if not image_path or not os.path.exists(image_path):
            logger.warning(f"[sentiment] 图片文件不存在: {image_path}")
            return False

        config = get_config()
        feishu_url = getattr(config, 'feishu_webhook_url', None)
        app_id = getattr(config, 'feishu_app_id', None)
        app_secret = getattr(config, 'feishu_app_secret', None)

        if not feishu_url:
            logger.debug("[sentiment] 飞书 Webhook 未配置，跳过图片推送")
            return False

        if not app_id or not app_secret:
            logger.warning("[sentiment] 飞书 App ID/Secret 未配置，无法发送图片")
            return False

        try:
            # Step 1: Get tenant_access_token
            token_url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
            token_resp = requests.post(token_url, json={
                "app_id": app_id,
                "app_secret": app_secret
            }, timeout=10)
            token_data = token_resp.json()
            if token_data.get("code") != 0:
                logger.error(f"[sentiment] 获取飞书 token 失败: {token_data}")
                return False
            access_token = token_data.get("tenant_access_token")

            # Step 2: Upload image
            upload_url = "https://open.feishu.cn/open-apis/im/v1/images"
            with open(image_path, 'rb') as f:
                files = {'image': f}
                data = {'image_type': 'message'}
                headers = {'Authorization': f'Bearer {access_token}'}
                upload_resp = requests.post(upload_url, headers=headers, files=files, data=data, timeout=30)
            upload_data = upload_resp.json()
            if upload_data.get("code") != 0:
                logger.error(f"[sentiment] 上传飞书图片失败: {upload_data}")
                return False
            image_key = upload_data.get("data", {}).get("image_key")
            if not image_key:
                logger.error("[sentiment] 获取 image_key 失败")
                return False

            # Step 3: Send image via webhook using post (rich text) format
            filename = os.path.basename(image_path)
            if 'limit_up' in filename:
                title = "市场情绪与风向分析 - 涨停股票"
            elif 'limit_down' in filename:
                title = "市场情绪与风向分析 - 跌停股票"
            else:
                title = "市场情绪与风向分析"

            payload = {
                "msg_type": "post",
                "content": {
                    "post": {
                        "zh_cn": {
                            "title": title,
                            "content": [
                                [
                                    {
                                        "tag": "img",
                                        "image_key": image_key
                                    }
                                ]
                            ]
                        }
                    }
                }
            }
            resp = requests.post(feishu_url, json=payload, timeout=10)
            result = resp.json()
            if result.get("code") == 0 or result.get("StatusCode") == 0:
                logger.info(f"[sentiment] 飞书图片发送成功: {filename}")
                return True
            else:
                logger.error(f"[sentiment] 飞书图片发送失败: {result}")
                return False

        except Exception as e:
            logger.error(f"[sentiment] 发送飞书图片异常: {e}")
            return False

    def _send_image_to_wechat(self, image_path: str) -> bool:
        """
        Send image to WeChat Work using base64 encoding.

        Args:
            image_path: Path to the image file

        Returns:
            True if sent successfully, False otherwise
        """
        import base64
        import hashlib
        import requests

        if not image_path or not os.path.exists(image_path):
            logger.warning(f"[sentiment] 图片文件不存在: {image_path}")
            return False

        config = get_config()
        wechat_url = getattr(config, 'wechat_webhook_url', None)

        if not wechat_url:
            logger.debug("[sentiment] 企业微信 Webhook 未配置，跳过图片推送")
            return False

        try:
            # Read image and encode to base64
            with open(image_path, 'rb') as f:
                image_data = f.read()

            image_base64 = base64.b64encode(image_data).decode('utf-8')
            image_md5 = hashlib.md5(image_data).hexdigest()

            payload = {
                "msgtype": "image",
                "image": {
                    "base64": image_base64,
                    "md5": image_md5
                }
            }

            response = requests.post(wechat_url, json=payload, timeout=30)

            if response.status_code == 200:
                result = response.json()
                if result.get('errcode') == 0:
                    logger.info(f"[sentiment] 企业微信图片发送成功: {os.path.basename(image_path)}")
                    return True
                else:
                    logger.error(f"[sentiment] 企业微信图片发送失败: {result}")
                    return False
            else:
                logger.error(f"[sentiment] 企业微信图片请求失败: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"[sentiment] 发送企业微信图片异常: {e}")
            return False

    def run_sentiment_analysis(self) -> str:
        logger.info("[sentiment] 市场情绪与风向分析开始执行...")
        start = time.time()
        sentiment = self.get_market_sentiment()
        report = self.generate_report(sentiment)

        # Generate table images and CSV files
        today_str = datetime.now().strftime('%Y%m%d')
        today_display = datetime.now().strftime('%Y-%m-%d')  # Readable date for titles
        reports_dir = os.path.join(os.path.dirname(__file__), 'src', 'reports')
        if not os.path.exists(reports_dir):
            reports_dir = os.path.join(os.path.dirname(__file__), 'reports')
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir, exist_ok=True)

        # Store generated image paths for external access
        self.limit_up_image_path = None
        self.limit_down_image_path = None

        if sentiment.limit_up_stocks:
            # Sort stocks by first concept before generating image/CSV
            sorted_up_stocks = self._sort_stocks_by_first_concept(sentiment.limit_up_stocks)[:40]
            # Generate image
            up_img_path = os.path.join(reports_dir, f'limit_up_table_{today_str}.png')
            if self._generate_limit_table_image(
                sorted_up_stocks,
                f"{today_display}-涨停股票（前40）",
                up_img_path,
                is_limit_up=True
            ):
                self.limit_up_image_path = up_img_path
            # Generate CSV
            up_csv_path = os.path.join(reports_dir, f'limit_up_table_{today_str}.csv')
            self._generate_limit_table_csv(
                sorted_up_stocks,
                up_csv_path,
                is_limit_up=True
            )

        if sentiment.limit_down_stocks:
            # Sort stocks by first concept before generating image/CSV
            sorted_down_stocks = self._sort_stocks_by_first_concept(sentiment.limit_down_stocks)[:30]
            # Generate image
            down_img_path = os.path.join(reports_dir, f'limit_down_table_{today_str}.png')
            if self._generate_limit_table_image(
                sorted_down_stocks,
                f"{today_display}-跌停股票（前30）",
                down_img_path,
                is_limit_up=False
            ):
                self.limit_down_image_path = down_img_path
            # Generate CSV
            down_csv_path = os.path.join(reports_dir, f'limit_down_table_{today_str}.csv')
            self._generate_limit_table_csv(
                sorted_down_stocks,
                down_csv_path,
                is_limit_up=False
            )

        # Send images to Feishu and WeChat Work
        if self.limit_up_image_path:
            self._send_image_to_feishu(self.limit_up_image_path)
            self._send_image_to_wechat(self.limit_up_image_path)
        if self.limit_down_image_path:
            self._send_image_to_feishu(self.limit_down_image_path)
            self._send_image_to_wechat(self.limit_down_image_path)

        logger.info(f"[sentiment] 市场情绪与风向分析执行完成，耗时 {time.time() - start:.1f} 秒。")
        return report


if __name__ == "__main__":
    # 允许直接以脚本方式运行
    sys.path.insert(0, ".")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
    )

    # 初始化搜索服务（可选）
    from search_service import SearchService

    config = get_config()
    search_service = None
    if config.bocha_api_keys or config.tavily_api_keys or config.serpapi_keys:
        search_service = SearchService(
            bocha_keys=config.bocha_api_keys,
            tavily_keys=config.tavily_api_keys,
            serpapi_keys=config.serpapi_keys,
        )

    analyzer = MarketSentimentAnalyzer(search_service=search_service)
    report = analyzer.run_sentiment_analysis()
    print("\n" + "=" * 60)
    print(report)

