# -*- coding: utf-8 -*-
"""
从本地 SQLite 数据库绘制市场情绪趋势图。

读取数据：
- market_sentiment_daily：每日涨停/跌停家数
- market_concept_daily_stat：每日概念计数（按涨停/跌停区分）

输出：
- 生成一张 PNG 图，横轴为日期。
"""

from __future__ import annotations

import argparse
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from storage import get_db


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _build_top_concept_by_date(rows) -> Dict[date, Tuple[str, int]]:
    """
    构建映射：date -> (当日 TOP1 概念名, 个数)

    假设 rows 已按 (date 升序, stock_count 降序) 排序。
    """
    result: Dict[date, Tuple[str, int]] = {}
    for r in rows:
        if r.date not in result:
            result[r.date] = (r.concept_name, int(r.stock_count or 0))
    return result


def _rows_to_date_map(rows) -> Dict[date, Dict[str, int]]:
    """转换为映射：date -> {concept_name: stock_count}"""
    result: Dict[date, Dict[str, int]] = {}
    for r in rows:
        d = r.date
        if d not in result:
            result[d] = {}
        name = str(getattr(r, "concept_name", "") or "").strip()
        if not name:
            continue
        result[d][name] = int(getattr(r, "stock_count", 0) or 0)
    return result


def _pick_top_concepts_overall(
    date_map: Dict[date, Dict[str, int]],
    top_n: int,
) -> List[str]:
    """按全时间段累积个数挑选 TopN 概念。"""
    total: Dict[str, int] = {}
    for _, m in date_map.items():
        for name, cnt in m.items():
            total[name] = total.get(name, 0) + int(cnt or 0)
    return [k for k, _ in sorted(total.items(), key=lambda x: x[1], reverse=True)[: max(0, int(top_n))]]


def _plot_line_with_top1(
    ax,
    dates: List[date],
    counts: List[int],
    top1_by_date: Dict[date, Tuple[str, int]],
    title: str,
    color: Optional[str] = None,
) -> None:
    ax.plot(dates, counts, marker="o", linewidth=2, label="家数", color=color)
    ax.set_title(title)
    ax.set_ylabel("家数")
    ax.grid(True, alpha=0.3)
    for x, y in zip(dates, counts):
        if x in top1_by_date:
            concept, cnt = top1_by_date[x]
            ax.annotate(
                f"{concept}({cnt})",
                (x, y),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
            )


def _plot_stacked_concepts(
    ax,
    dates: List[date],
    total_counts: List[int],
    concept_date_map: Dict[date, Dict[str, int]],
    title: str,
    top_n: int = 8,
) -> None:
    """
    绘制概念 TopN 的堆叠柱状图，并叠加总家数曲线。

    注意：概念计数是“归属次数”（单只股票多概念会重复计数），与总家数不一定相等。
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit(f"绘图需要 matplotlib，请先安装（例如 `pip install matplotlib`）。错误: {e}")

    top_concepts = _pick_top_concepts_overall(concept_date_map, top_n=top_n)
    bottoms = [0] * len(dates)

    # 颜色池（可重复使用）
    colors = list(plt.cm.tab20.colors)  # type: ignore[attr-defined]

    # 逐概念堆叠
    for i, concept in enumerate(top_concepts):
        series = [int(concept_date_map.get(d, {}).get(concept, 0) or 0) for d in dates]
        ax.bar(
            dates,
            series,
            bottom=bottoms,
            label=concept,
            color=colors[i % len(colors)],
            alpha=0.9,
        )
        bottoms = [b + v for b, v in zip(bottoms, series)]

    # 其他
    others = []
    for d in dates:
        total = sum(int(v or 0) for v in concept_date_map.get(d, {}).values())
        selected = sum(int(concept_date_map.get(d, {}).get(c, 0) or 0) for c in top_concepts)
        others.append(max(0, total - selected))
    if any(v > 0 for v in others):
        ax.bar(dates, others, bottom=bottoms, label="其他", color="lightgray", alpha=0.8)

    # 叠加总家数曲线（更符合“涨停/跌停个数”的直觉）
    ax.plot(dates, total_counts, color="black", linewidth=2, marker="o", label="总家数")

    ax.set_title(title)
    ax.set_ylabel("概念计数 / 总家数")
    ax.grid(True, alpha=0.25)


def plot_market_sentiment_trends(
    db,
    start_date: date,
    end_date: date,
    out_path: str,
    top_n: int = 8,
    style: str = "stacked",
) -> str:
    """
    绘制涨停/跌停趋势图并保存。

    Args:
        db: DatabaseManager 实例
        start_date: 开始日期
        end_date: 结束日期
        out_path: 输出 PNG 路径
        top_n: 堆叠图中展示的 TopN 概念数量
        style: 'stacked'（默认）或 'line'

    Returns:
        输出文件路径
    """
    daily = db.get_market_sentiment_daily_range(start_date, end_date)
    up_rows = db.get_market_concept_stats_range(start_date, end_date, limit_type="up")
    down_rows = db.get_market_concept_stats_range(start_date, end_date, limit_type="down")

    dates: List[date] = [d.date for d in daily]
    up_counts: List[int] = [int(d.limit_up_count or 0) for d in daily]
    down_counts: List[int] = [int(d.limit_down_count or 0) for d in daily]

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit(f"绘图需要 matplotlib，请先安装（例如 `pip install matplotlib`）。错误: {e}")

    plt.rcParams["figure.figsize"] = (16, 10)
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    if style == "line":
        top_up = _build_top_concept_by_date(up_rows)
        top_down = _build_top_concept_by_date(down_rows)
        _plot_line_with_top1(ax1, dates, up_counts, top_up, title="涨停趋势（标注当日 TOP1 概念）")
        _plot_line_with_top1(ax2, dates, down_counts, top_down, title="跌停趋势（标注当日 TOP1 概念）", color="tab:red")
    else:
        up_map = _rows_to_date_map(up_rows)
        down_map = _rows_to_date_map(down_rows)
        _plot_stacked_concepts(ax1, dates, up_counts, up_map, title=f"涨停：总家数 + 概念分布 Top{int(top_n)}（堆叠）", top_n=top_n)
        _plot_stacked_concepts(
            ax2,
            dates,
            down_counts,
            down_map,
            title=f"跌停：总家数 + 概念分布 Top{int(top_n)}（堆叠）",
            top_n=top_n,
        )

        # 统一 legend（放到图外，避免遮挡）
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=6, fontsize=9, frameon=False)
        fig.suptitle("市场情绪趋势图（概念计数为归属次数，单股多概念会重复计数）", fontsize=12)

    fig.autofmt_xdate(rotation=45)
    ax2.set_xlabel("日期")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(out, dpi=160)
    return str(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="绘制涨停/跌停数量与热点概念趋势图")
    parser.add_argument("--start", type=str, default=None, help="开始日期（YYYY-MM-DD），默认：今天-30天")
    parser.add_argument("--end", type=str, default=None, help="结束日期（YYYY-MM-DD），默认：今天")
    parser.add_argument("--out", type=str, default="./reports/market_sentiment_trends.png", help="输出 PNG 路径")
    parser.add_argument(
        "--style",
        type=str,
        default="stacked",
        choices=["stacked", "line"],
        help="绘图风格：stacked（Top 概念堆叠）或 line（标注 TOP1 概念）",
    )
    parser.add_argument("--top-n", type=int, default=8, help="堆叠图展示的 Top 概念数量（默认 8）")
    args = parser.parse_args()

    end_date = _parse_date(args.end) if args.end else date.today()
    start_date = _parse_date(args.start) if args.start else (end_date - timedelta(days=30))

    plot_market_sentiment_trends(
        db=get_db(),
        start_date=start_date,
        end_date=end_date,
        out_path=args.out,
        top_n=int(args.top_n),
        style=str(args.style),
    )
    out_path = Path(args.out)
    print(f"已保存: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

