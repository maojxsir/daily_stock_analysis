# -*- coding: utf-8 -*-
"""
===================================
市场情绪与风向分析模块
===================================

职责：
1. 分析当天涨停股票（数量、原因、板块、主营业务）
2. 分析当天跌停股票（数量、个股、板块、主营业务）
3. 统计概念板块表现
4. 生成市场情绪报告

数据来源：
1. **A股实时行情**：akshare.stock_zh_a_spot_em() -> 东方财富网 (eastmoney.com)
   - 获取所有A股实时行情数据
   - 筛选涨停/跌停股票（涨跌幅 >= 9.9% 或 <= -9.9%）

2. **股票基本信息**：akshare.stock_individual_info_em() -> 东方财富网
   - 获取股票行业、主营业务等信息

3. **股票所属板块**：efinance.stock.get_belong_board() -> 东方财富网（备选）
   - 获取股票所属的概念板块、行业板块

4. **涨停/跌停原因**：SearchService（可选）
   - 通过搜索服务（Tavily/SerpAPI/Bocha）搜索相关新闻
   - 从新闻中提取涨停/跌停原因

注意事项：
- akshare 通过爬虫获取数据，可能被反爬机制限制
- 已实现防封禁策略：随机休眠、指数退避重试
- 如果连接失败，会自动重试最多3次
- 建议在交易时间内使用，非交易时间可能无法获取实时数据
"""

import logging
import time
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List

import akshare as ak
import pandas as pd

from config import get_config
from search_service import SearchService


# 自定义异常类
class MarketSentimentError(Exception):
    """市场情绪分析基础异常"""
    pass


class DataSourceError(MarketSentimentError):
    """数据源错误"""
    pass


class APIError(MarketSentimentError):
    """API 调用错误"""
    pass


class CacheError(MarketSentimentError):
    """缓存错误"""
    pass


class AnalysisError(MarketSentimentError):
    """分析错误"""
    pass

logger = logging.getLogger(__name__)


@dataclass
class LimitUpStock:
    """涨停股票信息"""
    code: str                    # 股票代码
    name: str                    # 股票名称
    price: float = 0.0          # 涨停价
    change_pct: float = 0.0     # 涨跌幅
    volume: float = 0.0         # 成交量
    turnover_rate: float = 0.0  # 换手率
    concepts: List[str] = field(default_factory=list)  # 概念板块
    industry: str = ""           # 所属行业
    main_business: str = ""      # 主营业务
    reason: str = ""            # 涨停原因（从新闻/公告分析）

    def to_dict(self) -> Dict[str, Any]:
        return {
            'code': self.code,
            'name': self.name,
            'price': self.price,
            'change_pct': self.change_pct,
            'volume': self.volume,
            'turnover_rate': self.turnover_rate,
            'concepts': self.concepts,
            'industry': self.industry,
            'main_business': self.main_business,
            'reason': self.reason,
        }


@dataclass
class LimitDownStock:
    """跌停股票信息"""
    code: str                    # 股票代码
    name: str                    # 股票名称
    price: float = 0.0          # 跌停价
    change_pct: float = 0.0     # 涨跌幅
    volume: float = 0.0         # 成交量
    turnover_rate: float = 0.0  # 换手率
    concepts: List[str] = field(default_factory=list)  # 概念板块
    industry: str = ""           # 所属行业
    main_business: str = ""      # 主营业务
    reason: str = ""            # 跌停原因（从新闻/公告分析）

    def to_dict(self) -> Dict[str, Any]:
        return {
            'code': self.code,
            'name': self.name,
            'price': self.price,
            'change_pct': self.change_pct,
            'volume': self.volume,
            'turnover_rate': self.turnover_rate,
            'concepts': self.concepts,
            'industry': self.industry,
            'main_business': self.main_business,
            'reason': self.reason,
        }


@dataclass
class MarketSentiment:
    """市场情绪数据"""
    date: str
    limit_up_count: int = 0                    # 涨停家数
    limit_down_count: int = 0                  # 跌停家数
    limit_up_stocks: List[LimitUpStock] = field(default_factory=list)    # 涨停股票列表
    limit_down_stocks: List[LimitDownStock] = field(default_factory=list)  # 跌停股票列表

    # 板块统计
    top_concepts: List[Dict] = field(default_factory=list)  # 涨停股票最多的概念板块
    top_industries: List[Dict] = field(default_factory=list)  # 涨停股票最多的行业

    # 情绪指标
    sentiment_score: float = 0.0  # 情绪评分（-100到100，正数表示乐观）
    market_trend: str = ""        # 市场趋势：强势/偏强/震荡/偏弱/弱势


class MarketSentimentAnalyzer:
    """
    市场情绪分析器

    功能：
    1. 获取涨停股票列表及详细信息
    2. 获取跌停股票列表及详细信息
    3. 分析涨停/跌停原因
    4. 统计概念板块和行业分布
    5. 生成市场情绪报告
    """

    def __init__(self, search_service: Optional[SearchService] = None):
        """
        初始化市场情绪分析器

        Args:
            search_service: 搜索服务实例（用于获取涨停/跌停原因）
        """
        self.config = get_config()
        self.search_service = search_service
        self.cache = {}
        self.cache_expiry = {}
        # 使用默认缓存有效期（3600秒）
        self.cache_ttl = getattr(self.config, 'cache_ttl', 3600)  # 从配置中获取缓存有效期，如果不存在则使用默认值

    def _get_cached_data(self, key: str) -> Optional[Any]:
        """获取缓存数据
        
        Args:
            key: 缓存键
            
        Returns:
            缓存数据，如果缓存不存在或已过期则返回 None
        """
        if key in self.cache:
            expiry = self.cache_expiry.get(key, 0)
            if time.time() < expiry:
                logger.debug(f"[情绪分析] 从缓存获取数据: {key}")
                return self.cache[key]
            else:
                # 缓存已过期，删除
                del self.cache[key]
                del self.cache_expiry[key]
                logger.debug(f"[情绪分析] 缓存已过期: {key}")
        return None

    def _set_cached_data(self, key: str, data: Any, ttl: Optional[int] = None):
        """设置缓存数据
        
        Args:
            key: 缓存键
            data: 缓存数据
            ttl: 缓存有效期（秒），默认使用全局设置
        """
        if data is not None:
            expiry = time.time() + (ttl or self.cache_ttl)
            self.cache[key] = data
            self.cache_expiry[key] = expiry
            logger.debug(f"[情绪分析] 设置缓存数据: {key}, 有效期: {ttl or self.cache_ttl} 秒")

    def _call_akshare_with_retry(self, fn, name: str, attempts: int = 3, cache_key: Optional[str] = None):
        """
        调用 akshare API 并重试

        数据来源说明：
        - akshare 库通过爬取东方财富网等网站获取数据
        - 数据源：东方财富网 (eastmoney.com)
        - 特点：免费、无需Token，但可能被反爬机制限制

        防封禁策略：
        1. 每次请求前随机休眠 2-5 秒
        2. 指数退避重试（2秒、4秒、8秒...）
        3. 捕获连接错误并重试
        """
        import random

        # 检查缓存
        if cache_key:
            try:
                cached_data = self._get_cached_data(cache_key)
                if cached_data is not None:
                    return cached_data
            except Exception as e:
                logger.warning(f"[情绪分析] 缓存读取失败: {e}")

        last_error: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                # 防封禁策略：随机休眠 2-5 秒（模拟真实用户行为）
                if attempt == 1:
                    sleep_time = random.uniform(self.config.akshare_sleep_min, self.config.akshare_sleep_max)
                else:
                    # 重试时使用指数退避
                    sleep_time = min(2 ** attempt, 10)

                logger.debug(f"[情绪分析] {name} 请求前休眠 {sleep_time:.2f} 秒 (attempt {attempt}/{attempts})")
                time.sleep(sleep_time)

                result = fn()
                
                # 设置缓存
                if cache_key and result is not None:
                    try:
                        self._set_cached_data(cache_key, result)
                    except Exception as e:
                        logger.warning(f"[情绪分析] 缓存设置失败: {e}")
                
                return result

            except ConnectionError as e:
                last_error = DataSourceError(f"连接错误: {str(e)}")
                error_type = "连接错误"
            except TimeoutError as e:
                last_error = DataSourceError(f"超时错误: {str(e)}")
                error_type = "超时错误"
            except Exception as e:
                last_error = APIError(f"API 调用错误: {str(e)}")
                error_type = "API 错误"

            logger.warning(f"[情绪分析] {name} 获取失败 ({error_type}, attempt {attempt}/{attempts}): {last_error}")

            if attempt < attempts:
                # 指数退避：2秒、4秒、8秒...
                retry_delay = min(2 ** attempt, self.config.retry_max_delay)
                logger.info(f"[情绪分析] {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
            else:
                logger.error(f"[情绪分析] {name} 最终失败，已重试 {attempts} 次")

        return None

    def _batch_enrich_stock_info(self, stocks: List[Any]):
        """
        批量丰富股票信息（板块、主营业务）
        
        Args:
            stocks: 股票列表（LimitUpStock 或 LimitDownStock）
        """
        if not stocks:
            return
        
        logger.info(f"[情绪分析] 开始批量获取 {len(stocks)} 只股票的详细信息...")
        
        # 1. 按股票代码分组
        stock_map = {stock.code: stock for stock in stocks}
        stock_codes = list(stock_map.keys())
        
        # 2. 批量获取股票基本信息
        try:
            # 这里可以实现批量获取逻辑，目前先使用单个获取
            # 后续可以通过其他数据源或API实现真正的批量获取
            for code, stock in stock_map.items():
                # 使用缓存机制减少重复请求
                info_cache_key = f"stock_info_{code}"
                cached_info = self._get_cached_data(info_cache_key)
                
                if cached_info:
                    stock.industry = cached_info.get('industry', '')
                    stock.main_business = cached_info.get('main_business', '')
                else:
                    # 使用 akshare 获取股票基本信息
                    info_df = self._call_akshare_with_retry(
                        lambda: ak.stock_individual_info_em(symbol=code),
                        f"股票基本信息({code})（数据源：东方财富网）",
                        attempts=2
                    )

                    if info_df is not None and not info_df.empty:
                        # 解析基本信息
                        industry = ''
                        main_business = ''
                        for _, row in info_df.iterrows():
                            item = str(row.get('item', '')).strip()
                            value = str(row.get('value', '')).strip()

                            if '所属行业' in item or '行业' in item:
                                industry = value
                                stock.industry = value
                            elif '主营业务' in item or '经营范围' in item:
                                main_business = value[:200]  # 限制长度
                                stock.main_business = main_business
                        
                        # 设置缓存
                        self._set_cached_data(info_cache_key, {
                            'industry': industry,
                            'main_business': main_business
                        })
        except Exception as e:
            logger.error(f"[情绪分析] 批量获取股票基本信息失败: {e}")
        
        # 3. 批量获取股票概念板块
        try:
            # 首先获取所有概念板块列表
            concepts_df = self._call_akshare_with_retry(
                lambda: ak.stock_board_concept_name_em(),
                "概念板块列表",
                attempts=1,
                cache_key="concept_board_list"
            )
            
            if concepts_df is not None and not concepts_df.empty:
                # 遍历每个概念板块，查找包含这些股票的板块
                for _, row in concepts_df.iterrows():
                    board_name = str(row.get('板块名称', '')).strip()
                    if board_name:
                        # 获取该概念板块的成分股
                        stock_list_df = self._call_akshare_with_retry(
                            lambda: ak.stock_board_concept_cons_em(symbol=board_name),
                            f"概念板块({board_name})成分股",
                            attempts=1
                        )
                        
                        if stock_list_df is not None and not stock_list_df.empty:
                            # 检查这些股票是否在成分股中
                            code_col = '代码' if '代码' in stock_list_df.columns else 'code'
                            if code_col in stock_list_df.columns:
                                stock_codes_in_board = stock_list_df[code_col].astype(str).str.strip().tolist()
                                for code in stock_codes:
                                    if code in stock_codes_in_board:
                                        stock = stock_map.get(code)
                                        if stock and board_name not in stock.concepts:
                                            stock.concepts.append(board_name)
        except Exception as e:
            logger.error(f"[情绪分析] 批量获取股票概念板块失败: {e}")
        
        # 4. 对于未获取到概念板块的股票，尝试使用 efinance
        try:
            import efinance as ef
            for code, stock in stock_map.items():
                if not stock.concepts:
                    # 使用 efinance 获取股票所属板块
                    board_df = ef.stock.get_belong_board(code)
                    
                    if board_df is not None and not board_df.empty:
                        # 提取概念板块名称
                        if '板块名称' in board_df.columns:
                            concepts = board_df['板块名称'].head(5).tolist()
                        elif 'name' in board_df.columns:
                            concepts = board_df['name'].head(5).tolist()
                        else:
                            concepts = []
                        
                        stock.concepts = [str(c).strip() for c in concepts if c]
                        
                        # 设置缓存
                        cache_key = f"stock_concepts_{code}"
                        self._set_cached_data(cache_key, stock.concepts)
        except Exception as e:
            logger.error(f"[情绪分析] efinance 批量获取概念板块失败: {e}")
        
        logger.info(f"[情绪分析] 批量获取股票信息完成")

    def get_market_sentiment(self) -> MarketSentiment:
        """
        获取市场情绪数据

        Returns:
            MarketSentiment: 市场情绪数据对象
        """
        today = datetime.now().strftime('%Y-%m-%d')
        sentiment = MarketSentiment(date=today)

        logger.info("[情绪分析] 开始获取市场情绪数据...")
        logger.info("[情绪分析] 数据来源：优先使用 efinance (优先级0)，备选 akshare (优先级1) -> 东方财富网")

        # 1. 获取涨停股票列表
        limit_up_stocks = self._get_limit_stocks(is_limit_up=True)
        sentiment.limit_up_stocks = limit_up_stocks
        sentiment.limit_up_count = len(limit_up_stocks)

        # 2. 获取跌停股票列表
        limit_down_stocks = self._get_limit_stocks(is_limit_up=False)
        sentiment.limit_down_stocks = limit_down_stocks
        sentiment.limit_down_count = len(limit_down_stocks)

        # 3. 丰富股票详细信息（板块、主营业务）
        if limit_up_stocks:
            logger.info("[情绪分析] 开始获取涨停股票详细信息...")
            self._batch_enrich_stock_info(limit_up_stocks)
            # 分析涨停股票板块分布
            sentiment.top_concepts = self._analyze_concept_distribution(limit_up_stocks)
            sentiment.top_industries = self._analyze_industry_distribution(limit_up_stocks)

        if limit_down_stocks:
            logger.info("[情绪分析] 开始获取跌停股票详细信息...")
            self._batch_enrich_stock_info(limit_down_stocks)

        # 4. 计算情绪评分
        sentiment.sentiment_score = self._calculate_sentiment_score(sentiment)
        sentiment.market_trend = self._determine_market_trend(sentiment)

        logger.info(f"[情绪分析] 完成: 涨停{sentiment.limit_up_count}只, 跌停{sentiment.limit_down_count}只, "
                   f"情绪评分{sentiment.sentiment_score:.1f}, 趋势{sentiment.market_trend}")

        return sentiment

    def _analyze_industry_distribution(self, stocks: List[Any]) -> List[Dict]:
        """分析涨停/跌停股票的行业分布"""
        industry_count = {}

        for stock in stocks:
            if stock.industry:
                industry_count[stock.industry] = industry_count.get(stock.industry, 0) + 1

        # 按数量排序
        sorted_industries = sorted(industry_count.items(), key=lambda x: x[1], reverse=True)

        return [
            {'name': name, 'count': count}
            for name, count in sorted_industries[:10]  # 返回前10个
        ]

    def get_market_overview(self) -> Dict[str, Any]:
        """
        获取市场总貌数据

        数据来源：
        - 上海证券交易所: ak.stock_sse_summary()
        - 深圳证券交易所: ak.stock_szse_summary()

        Returns:
            Dict[str, Any]: 市场总貌数据
        """
        logger.info("[情绪分析] 开始获取市场总貌数据...")
        
        overview = {
            'sse': None,  # 上交所数据
            'szse': None,  # 深交所数据
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        try:
            # 获取上交所数据
            sse_df = self._call_akshare_with_retry(
                lambda: ak.stock_sse_summary(),
                "上海证券交易所总貌数据",
                attempts=2
            )
            if sse_df is not None and not sse_df.empty:
                overview['sse'] = sse_df.to_dict('records')
                logger.info("[情绪分析] 成功获取上交所总貌数据")
        except Exception as e:
            logger.error(f"[情绪分析] 获取上交所总貌数据失败: {e}")

        try:
            # 获取深交所数据
            today = datetime.now().strftime('%Y%m%d')
            szse_df = self._call_akshare_with_retry(
                lambda: ak.stock_szse_summary(date=today),
                "深圳证券交易所总貌数据",
                attempts=2
            )
            if szse_df is not None and not szse_df.empty:
                overview['szse'] = szse_df.to_dict('records')
                logger.info("[情绪分析] 成功获取深交所总貌数据")
        except Exception as e:
            logger.error(f"[情绪分析] 获取深交所总貌数据失败: {e}")

        return overview

    def _get_limit_stocks(self, is_limit_up: bool) -> List[Any]:
        """获取涨停或跌停股票列表
        
        Args:
            is_limit_up: 是否为涨停股票
            
        Returns:
            股票列表（LimitUpStock 或 LimitDownStock）
        """
        stocks = []
        stock_class = LimitUpStock if is_limit_up else LimitDownStock
        # 使用默认值：涨停股票最多获取50只，跌停股票最多获取30只
        limit_count = getattr(self.config, 'max_limit_up_stocks', 50) if is_limit_up else getattr(self.config, 'max_limit_down_stocks', 30)
        action_name = "涨停" if is_limit_up else "跌停"

        try:
            logger.info(f"[情绪分析] 获取{action_name}股票列表...")

            if is_limit_up:
                # 使用 ak.stock_zt_pool_em() 获取涨停股票池
                try:
                    today = datetime.now().strftime('%Y%m%d')
                    cache_key = f"stock_zt_pool_{today}"
                    zt_pool_df = self._call_akshare_with_retry(
                        lambda: ak.stock_zt_pool_em(date=today),
                        "涨停股票池（数据源：akshare -> 东方财富网）",
                        attempts=3,
                        cache_key=cache_key
                    )

                    if zt_pool_df is not None and not zt_pool_df.empty:
                        logger.info(f"[情绪分析] 从涨停股票池获取到 {len(zt_pool_df)} 只涨停股票")

                        # 转换为股票对象
                        for _, row in zt_pool_df.iterrows():
                            code = str(row.get('代码', row.get('code', ''))).strip()
                            name = str(row.get('名称', row.get('name', ''))).strip()

                            if not code or not name:
                                continue

                            # 兼容不同的列名
                            price_col = '最新价' if '最新价' in row.index else '现价' if '现价' in row.index else 'price'
                            change_col = '涨跌幅' if '涨跌幅' in row.index else 'change_pct'
                            volume_col = '成交量' if '成交量' in row.index else 'volume'
                            turnover_col = '换手率' if '换手率' in row.index else 'turnover_rate'

                            stock = stock_class(
                                code=code,
                                name=name,
                                price=float(row.get(price_col, row.get('最新价', 0)) or 0),
                                change_pct=float(row.get(change_col, 0) or 0),
                                volume=float(row.get(volume_col, row.get('成交量', 0)) or 0),
                                turnover_rate=float(row.get(turnover_col, row.get('换手率', 0)) or 0),
                            )

                            stocks.append(stock)

                            # 限制数量，避免处理时间过长
                            if len(stocks) >= limit_count:
                                logger.info(f"[情绪分析] 已获取前{limit_count}只涨停股票，停止获取")
                                break

                        logger.info(f"[情绪分析] 成功从涨停股票池获取 {len(stocks)} 只涨停股票信息")
                except Exception as e:
                    logger.error(f"[情绪分析] 使用 stock_zt_pool_em 获取失败: {e}")

            else:
                # 使用 ak.stock_zt_pool_dtgc_em() 获取跌停股票池
                try:
                    today = datetime.now().strftime('%Y%m%d')
                    cache_key = f"stock_dtgc_pool_{today}"
                    dtgc_pool_df = self._call_akshare_with_retry(
                        lambda: ak.stock_zt_pool_dtgc_em(date=today),
                        "跌停股票池（数据源：akshare -> 东方财富网）",
                        attempts=3,
                        cache_key=cache_key
                    )

                    if dtgc_pool_df is not None and not dtgc_pool_df.empty:
                        logger.info(f"[情绪分析] 从跌停股票池获取到 {len(dtgc_pool_df)} 只跌停股票")

                        # 转换为股票对象
                        for _, row in dtgc_pool_df.iterrows():
                            code = str(row.get('代码', row.get('code', ''))).strip()
                            name = str(row.get('名称', row.get('name', ''))).strip()

                            if not code or not name:
                                continue

                            # 兼容不同的列名
                            price_col = '最新价' if '最新价' in row.index else '现价' if '现价' in row.index else 'price'
                            change_col = '涨跌幅' if '涨跌幅' in row.index else 'change_pct'
                            volume_col = '成交量' if '成交量' in row.index else 'volume'
                            turnover_col = '换手率' if '换手率' in row.index else 'turnover_rate'

                            stock = stock_class(
                                code=code,
                                name=name,
                                price=float(row.get(price_col, row.get('最新价', 0)) or 0),
                                change_pct=float(row.get(change_col, 0) or 0),
                                volume=float(row.get(volume_col, row.get('成交量', 0)) or 0),
                                turnover_rate=float(row.get(turnover_col, row.get('换手率', 0)) or 0),
                            )

                            stocks.append(stock)

                            # 限制数量，避免处理时间过长
                            if len(stocks) >= limit_count:
                                logger.info(f"[情绪分析] 已获取前{limit_count}只跌停股票，停止获取")
                                break

                        logger.info(f"[情绪分析] 成功从跌停股票池获取 {len(stocks)} 只跌停股票信息")
                except Exception as e:
                    logger.error(f"[情绪分析] 使用 stock_zt_pool_dtgc_em 获取失败: {e}")

        except Exception as e:
            logger.error(f"[情绪分析] 获取{action_name}股票失败: {e}")

        return stocks


    def _get_limit_up_stocks(self) -> List[LimitUpStock]:
        """获取涨停股票列表"""
        return self._get_limit_stocks(is_limit_up=True)


    def _get_limit_down_stocks(self) -> List[LimitDownStock]:
        """获取跌停股票列表"""
        return self._get_limit_stocks(is_limit_up=False)

    def _get_stock_concepts(self, code: str) -> List[str]:
        """获取股票所属概念板块
        
        Args:
            code: 股票代码
            
        Returns:
            概念板块列表
        """
        # 检查缓存
        cache_key = f"stock_concepts_{code}"
        cached_concepts = self._get_cached_data(cache_key)
        if cached_concepts:
            return cached_concepts
        
        concepts = []
        
        # 1. 尝试使用 akshare 获取股票概念板块
        try:
            # 首先获取所有概念板块列表
            concepts_df = self._call_akshare_with_retry(
                lambda: ak.stock_board_concept_name_em(),
                "概念板块列表",
                attempts=1,
                cache_key="concept_board_list"
            )
            
            if concepts_df is not None and not concepts_df.empty:
                # 遍历每个概念板块，查找包含该股票的板块
                for _, row in concepts_df.iterrows():
                    board_name = str(row.get('板块名称', '')).strip()
                    if board_name:
                        # 获取该概念板块的成分股
                        stock_list_df = self._call_akshare_with_retry(
                            lambda: ak.stock_board_concept_cons_em(symbol=board_name),
                            f"概念板块({board_name})成分股",
                            attempts=1
                        )
                        
                        if stock_list_df is not None and not stock_list_df.empty:
                            # 检查该股票是否在成分股中
                            code_col = '代码' if '代码' in stock_list_df.columns else 'code'
                            if code_col in stock_list_df.columns:
                                stock_codes = stock_list_df[code_col].astype(str).str.strip().tolist()
                                if code in stock_codes:
                                    concepts.append(board_name)
                        
                        # 限制获取的概念板块数量
                        if len(concepts) >= 5:
                            break
        except Exception as e:
            logger.debug(f"[情绪分析] akshare 获取 {code} 概念板块失败: {e}")
        
        # 2. 如果 akshare 获取失败，尝试使用 efinance
        if not concepts:
            try:
                # 备选数据源：efinance -> 东方财富网
                import efinance as ef
                # 使用 efinance 获取股票所属板块
                board_df = ef.stock.get_belong_board(code)
                
                if board_df is not None and not board_df.empty:
                    # 提取概念板块名称
                    if '板块名称' in board_df.columns:
                        concepts = board_df['板块名称'].head(5).tolist()
                    elif 'name' in board_df.columns:
                        concepts = board_df['name'].head(5).tolist()
            except Exception as e:
                logger.debug(f"[情绪分析] efinance 获取 {code} 概念板块失败: {e}")
        
        result = [str(c).strip() for c in concepts if c]
        # 设置缓存
        cache_key = f"stock_concepts_{code}"
        self._set_cached_data(cache_key, result)
        return result


    def _enrich_stock_info(self, stock: Any):
        """
        丰富股票信息（板块、主营业务）

        Args:
            stock: LimitUpStock 或 LimitDownStock 对象
        """
        try:
            code = stock.code

            # 1. 获取股票基本信息（行业、主营业务）
            try:
                # 检查缓存
                info_cache_key = f"stock_info_{code}"
                cached_info = self._get_cached_data(info_cache_key)
                
                if cached_info:
                    stock.industry = cached_info.get('industry', '')
                    stock.main_business = cached_info.get('main_business', '')
                else:
                    # 使用 akshare 获取股票基本信息
                    # 数据来源：akshare -> 东方财富网
                    # API: ak.stock_individual_info_em() - 获取股票基本信息
                    info_df = self._call_akshare_with_retry(
                        lambda: ak.stock_individual_info_em(symbol=code),
                        f"股票基本信息({code})（数据源：东方财富网）",
                        attempts=2
                    )

                    if info_df is not None and not info_df.empty:
                        # 解析基本信息
                        industry = ''
                        main_business = ''
                        for _, row in info_df.iterrows():
                            item = str(row.get('item', '')).strip()
                            value = str(row.get('value', '')).strip()

                            if '所属行业' in item or '行业' in item:
                                industry = value
                                stock.industry = value
                            elif '主营业务' in item or '经营范围' in item:
                                main_business = value[:200]  # 限制长度
                                stock.main_business = main_business
                        
                        # 设置缓存
                        self._set_cached_data(info_cache_key, {
                            'industry': industry,
                            'main_business': main_business
                        })

            except Exception as e:
                logger.debug(f"[情绪分析] 获取 {code} 基本信息失败: {e}")

            # 2. 获取股票所属概念板块
            stock.concepts = self._get_stock_concepts(code)

            # 3. 如果仍未获取到行业信息，尝试从概念板块中提取
            if not stock.industry and stock.concepts:
                # 简单逻辑：如果概念板块中包含行业相关词汇，作为行业信息
                industry_keywords = ['行业', '板块', '产业']
                for concept in stock.concepts:
                    if any(keyword in concept for keyword in industry_keywords):
                        stock.industry = concept
                        break

        except Exception as e:
            logger.debug(f"[情绪分析] 丰富股票信息失败: {e}")

    def _analyze_concept_distribution(self, stocks: List[LimitUpStock]) -> List[Dict]:
        """分析涨停股票的概念板块分布"""
        concept_count = {}

        for stock in stocks:
            for concept in stock.concepts:
                if concept:
                    concept_count[concept] = concept_count.get(concept, 0) + 1

        # 按数量排序
        sorted_concepts = sorted(concept_count.items(), key=lambda x: x[1], reverse=True)

        return [
            {'name': name, 'count': count}
            for name, count in sorted_concepts[:10]  # 返回前10个
        ]

    def _analyze_industry_distribution(self, stocks: List[LimitUpStock]) -> List[Dict]:
        """分析涨停股票的行业分布"""
        industry_count = {}

        for stock in stocks:
            if stock.industry:
                industry_count[stock.industry] = industry_count.get(stock.industry, 0) + 1

        # 按数量排序
        sorted_industries = sorted(industry_count.items(), key=lambda x: x[1], reverse=True)

        return [
            {'name': name, 'count': count}
            for name, count in sorted_industries[:10]  # 返回前10个
        ]

    def _calculate_sentiment_score(self, sentiment: MarketSentiment) -> float:
        """
        计算市场情绪评分

        评分规则：
        - 涨停数量越多，评分越高
        - 跌停数量越多，评分越低
        - 板块轮动：热点板块的数量和强度
        - 范围：-100 到 100
        """
        # 使用默认权重值
        limit_up_weight = getattr(self.config, 'limit_up_weight', 1.0)
        limit_down_weight = getattr(self.config, 'limit_down_weight', 1.5)
        sector_rotation_weight = getattr(self.config, 'sector_rotation_weight', 5.0)
        
        # 1. 基础评分（涨停/跌停数量）
        base_score = sentiment.limit_up_count * limit_up_weight - sentiment.limit_down_count * limit_down_weight

        # 2. 板块轮动因子
        sector_rotation = 0.0
        if sentiment.top_concepts:
            # 热点板块数量和强度
            top_concept_count = len([c for c in sentiment.top_concepts if c['count'] >= 3])
            sector_rotation = top_concept_count * sector_rotation_weight
        
        # 3. 综合评分
        total_score = base_score + sector_rotation
        
        # 4. 归一化到 -100 到 100 范围
        # 假设涨停数最多200只，跌停数最多50只，板块轮动最高50
        max_score = 200 * limit_up_weight + sector_rotation_weight * 5
        min_score = -50 * limit_down_weight
        
        if max_score - min_score > 0:
            normalized_score = ((total_score - min_score) / (max_score - min_score)) * 200 - 100
        else:
            normalized_score = 0
        
        # 限制范围
        score = max(-100, min(100, normalized_score))
        
        return score

    def _determine_market_trend(self, sentiment: MarketSentiment) -> str:
        """判断市场趋势"""
        score = sentiment.sentiment_score

        if score >= 60:
            return "强势"
        elif score >= 30:
            return "偏强"
        elif score >= -30:
            return "震荡"
        elif score >= -60:
            return "偏弱"
        else:
            return "弱势"

    def analyze_limit_reasons(self, stocks: List[Any], is_limit_up: bool = True) -> None:
        """
        分析涨停/跌停原因（通过搜索新闻）

        Args:
            stocks: 股票列表（LimitUpStock 或 LimitDownStock）
            is_limit_up: 是否为涨停股票
        """
        if not self.search_service:
            logger.warning("[情绪分析] 搜索服务未配置，跳过原因分析")
            return

        logger.info(f"[情绪分析] 开始分析{'涨停' if is_limit_up else '跌停'}原因...")

        for i, stock in enumerate(stocks[:self.config.max_analyze_stocks]):  # 只分析前N只
            try:
                # 搜索该股票的最新新闻
                query = f"{stock.name} {stock.code} {'涨停' if is_limit_up else '跌停'}"
                response = self.search_service.search_stock_news(
                    stock_code=stock.code,
                    stock_name=stock.name,
                    max_results=3,
                    focus_keywords=['涨停' if is_limit_up else '跌停', '公告', '消息']
                )

                if response and response.results:
                    # 提取可能的原因关键词
                    reasons = []
                    for result in response.results[:2]:  # 只取前2条
                        title = result.title if hasattr(result, 'title') else result.get('title', '')
                        snippet = result.snippet if hasattr(result, 'snippet') else result.get('snippet', '')

                        # 提取关键词
                        text = f"{title} {snippet}".lower()
                        if any(kw in text for kw in ['业绩', '合同', '订单', '中标']):
                            reasons.append('业绩/订单利好')
                        elif any(kw in text for kw in ['政策', '扶持', '补贴']):
                            reasons.append('政策利好')
                        elif any(kw in text for kw in ['重组', '并购', '收购']):
                            reasons.append('重组/并购')
                        elif any(kw in text for kw in ['减持', '处罚', '立案']):
                            reasons.append('利空消息')
                        elif any(kw in text for kw in ['概念', '板块', '热点']):
                            reasons.append('概念炒作')

                    if reasons:
                        stock.reason = '、'.join(set(reasons))  # 去重
                    else:
                        stock.reason = '概念/资金推动' if is_limit_up else '利空/资金出逃'

                # 避免请求过快
                if i < len(stocks) - 1:
                    time.sleep(1)

            except Exception as e:
                logger.debug(f"[情绪分析] 分析 {stock.code} 原因失败: {e}")
                stock.reason = '原因待查'

    def generate_sentiment_report(self, sentiment: MarketSentiment, market_reason_counts: dict = None) -> str:
        """
        生成市场情绪报告

        Args:
            sentiment: 市场情绪数据
            market_reason_counts: 市场涨停原因分布

        Returns:
            情绪报告文本（Markdown格式）
        """
        report_lines = []

        # 标题
        report_lines.append(f"# 📊 {sentiment.date} 市场情绪与风向分析")
        report_lines.append("")

        # 情绪概览
        report_lines.append("## 一、情绪概览")
        report_lines.append("")
        report_lines.append(f"- **涨停家数**: {sentiment.limit_up_count} 只")
        report_lines.append(f"- **跌停家数**: {sentiment.limit_down_count} 只")
        report_lines.append(f"- **情绪评分**: {sentiment.sentiment_score:.1f}/100")
        report_lines.append(f"- **市场趋势**: {sentiment.market_trend}")
        report_lines.append("")

        # 涨停分析
        if sentiment.limit_up_stocks:
            report_lines.append("## 二、涨停股票分析")
            report_lines.append("")

            # 涨停股票列表（前20只）
            report_lines.append("### 涨停股票列表（前20只）")
            report_lines.append("")
            report_lines.append("| 代码 | 名称 | 价格 | 涨跌幅 | 换手率 | 涨停原因 | 概念板块 | 主营业务 |")
            report_lines.append("|------|------|------|--------|--------|----------|----------|----------|")

            for stock in sentiment.limit_up_stocks[:20]:
                reason = stock.reason if stock.reason else '待分析'
                concepts = '、'.join(stock.concepts[:3]) if stock.concepts else '无'
                main_business = stock.main_business[:50] + '...' if len(stock.main_business) > 50 else stock.main_business
                main_business = main_business if main_business else '待获取'
                report_lines.append(
                    f"| {stock.code} | {stock.name} | {stock.price:.2f} | {stock.change_pct:.2f}% | "
                    f"{stock.turnover_rate:.2f}% | {reason} | {concepts} | {main_business} |"
                )
            report_lines.append("")

            # 市场涨停原因分布
            if market_reason_counts:
                report_lines.append("### 市场涨停原因分布")
                report_lines.append("")
                report_lines.append("| 涨停原因 | 提及频次 |")
                report_lines.append("|----------|----------|")
                
                # 按频次排序
                sorted_reasons = sorted(market_reason_counts.items(), key=lambda x: x[1], reverse=True)
                for reason, count in sorted_reasons:
                    report_lines.append(f"| {reason} | {count} |")
                report_lines.append("")

            # 涨停股票概念板块分布
            if sentiment.top_concepts:
                report_lines.append("### 涨停股票概念板块分布")
                report_lines.append("")
                report_lines.append("| 概念板块 | 涨停数量 |")
                report_lines.append("|----------|----------|")
                for concept in sentiment.top_concepts[:10]:
                    report_lines.append(f"| {concept['name']} | {concept['count']} |")
                report_lines.append("")

        # 跌停分析
        if sentiment.limit_down_stocks:
            report_lines.append("## 三、跌停股票分析")
            report_lines.append("")

            # 跌停股票列表
            report_lines.append("### 跌停股票列表")
            report_lines.append("")
            report_lines.append("| 代码 | 名称 | 价格 | 涨跌幅 | 换手率 | 概念板块 | 主营业务 |")
            report_lines.append("|------|------|------|--------|--------|----------|----------|")

            for stock in sentiment.limit_down_stocks[:20]:
                concepts = '、'.join(stock.concepts[:3]) if stock.concepts else '无'
                main_business = stock.main_business[:50] + '...' if len(stock.main_business) > 50 else stock.main_business
                main_business = main_business if main_business else '待获取'
                report_lines.append(
                    f"| {stock.code} | {stock.name} | {stock.price:.2f} | {stock.change_pct:.2f}% | "
                    f"{stock.turnover_rate:.2f}% | {concepts} | {main_business} |"
                )
            report_lines.append("")

        # 总结
        report_lines.append("## 四、市场风向总结")
        report_lines.append("")

        if sentiment.limit_up_count > 50:
            report_lines.append(f"- ✅ 市场情绪**高涨**，涨停股票数量较多（{sentiment.limit_up_count}只），显示资金活跃")
        elif sentiment.limit_up_count > 20:
            report_lines.append(f"- ⚡ 市场情绪**偏强**，涨停股票数量适中（{sentiment.limit_up_count}只）")
        else:
            report_lines.append(f"- ⚠️ 市场情绪**偏弱**，涨停股票数量较少（{sentiment.limit_up_count}只）")

        if sentiment.limit_down_count > 10:
            report_lines.append(f"- ❌ 市场存在**风险**，跌停股票数量较多（{sentiment.limit_down_count}只），需注意风险")
        elif sentiment.limit_down_count > 0:
            report_lines.append(f"- ⚠️ 市场存在**局部风险**，跌停股票数量为{sentiment.limit_down_count}只")
        else:
            report_lines.append(f"- ✅ 市场**无跌停股票**，整体风险可控")

        # 基于涨停原因的总结
        if market_reason_counts:
            top_reasons = [reason for reason, _ in sorted(market_reason_counts.items(), key=lambda x: x[1], reverse=True)[:3]]
            if top_reasons:
                report_lines.append(f"- 🎯 **主要涨停原因**: {', '.join(top_reasons)}")

        # 基于概念板块的总结
        if sentiment.top_concepts:
            top_concepts = [concept['name'] for concept in sentiment.top_concepts[:3]]
            if top_concepts:
                report_lines.append(f"- 🔥 **热门概念板块**: {', '.join(top_concepts)}")

        report_lines.append("")
        report_lines.append("---")
        report_lines.append(f"*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

        return "\n".join(report_lines)

    def analyze_market_limit_reasons(self) -> dict:
        """
        分析整体市场的涨停原因趋势

        Returns:
            dict: 涨停原因分布
        """
        if not self.search_service:
            logger.warning("[情绪分析] 搜索服务未配置，跳过市场涨停原因分析")
            return {}

        logger.info("[情绪分析] 开始分析市场涨停原因趋势...")

        # 常见的涨停原因类别
        reason_categories = {
            '业绩/订单利好': ['业绩', '利润', '增长', '订单', '合同', '中标'],
            '政策利好': ['政策', '扶持', '补贴', '规划', '纲要'],
            '概念炒作': ['概念', '板块', '热点', '题材', '赛道'],
            '重组/并购': ['重组', '并购', '收购', '借壳', '整合'],
            '技术突破': ['技术', '创新', '突破', '研发', '专利'],
            '资金推动': ['资金', '主力', '游资', '机构', '买入']
        }

        # 统计各类原因的出现频次
        reason_counts = {}

        # 搜索市场整体涨停情况
        query = "今日涨停 原因 板块"
        response = self.search_service.search_stock_news(
            stock_code='',
            stock_name='市场整体',
            max_results=10,
            focus_keywords=['涨停', '板块', '原因']
        )

        if response and response.results:
            for result in response.results:
                title = result.title if hasattr(result, 'title') else result.get('title', '')
                snippet = result.snippet if hasattr(result, 'snippet') else result.get('snippet', '')
                text = f"{title} {snippet}".lower()

                # 统计各类原因的出现频次
                for reason, keywords in reason_categories.items():
                    if any(kw in text for kw in keywords):
                        reason_counts[reason] = reason_counts.get(reason, 0) + 1

        logger.info(f"[情绪分析] 市场涨停原因分析完成: {reason_counts}")
        return reason_counts

    def run_sentiment_analysis(self) -> str:
        """
        执行完整的市场情绪分析流程

        Returns:
            情绪分析报告文本
        """
        logger.info("========== 开始市场情绪分析 ==========")

        # 1. 获取市场情绪数据
        sentiment = self.get_market_sentiment()

        # 2. 分析市场整体涨停原因趋势（如果有搜索服务）
        market_reason_counts = {}
        if self.search_service:
            market_reason_counts = self.analyze_market_limit_reasons()

        # 3. 生成报告
        report = self.generate_sentiment_report(sentiment, market_reason_counts)

        logger.info("========== 市场情绪分析完成 ==========")

        return report


# 测试入口
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    )

    # 初始化搜索服务（可选）
    from search_service import SearchService
    from config import get_config

    config = get_config()
    search_service = None
    if config.bocha_api_keys or config.tavily_api_keys or config.serpapi_keys:
        search_service = SearchService(
            bocha_keys=config.bocha_api_keys,
            tavily_keys=config.tavily_api_keys,
            serpapi_keys=config.serpapi_keys,
        )

    analyzer = MarketSentimentAnalyzer(search_service=search_service)

    # 执行分析
    report = analyzer.run_sentiment_analysis()
    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)
