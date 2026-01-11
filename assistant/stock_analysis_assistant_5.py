import os
import dashscope
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
import pandas as pd
from qwen_agent.tools.base import BaseTool, register_tool
import matplotlib.pyplot as plt

import sqlite3

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']  # 优先使用的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 配置 DashScope
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', '')  # 从环境变量获取 API Key
dashscope.timeout = 30  # 设置超时时间为 30 秒

# ====== 股票助手 system prompt 和函数描述 ======
system_prompt = """我是股票分析助手，以下是关于股票数据表相关的字段，我可能会编写对应的SQL，对数据进行查询
-- 股票数据表
CREATE TABLE stock_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_code TEXT NOT NULL,      -- 股票代码
    trade_date DATE NOT NULL,   -- 交易日期
    open REAL,                  -- 开盘价
    high REAL,                  -- 最高价
    low REAL,                   -- 最低价
    close REAL,                 -- 收盘价
    vol INTEGER,                -- 成交量
    amount REAL,                -- 成交额
    name TEXT NOT NULL          -- 股票名称
);

数据库中的股票数据最新更新至2026年1月9日，包含以下股票：
- 贵州茅台 (600519.SH)
- 五粮液 (000858.SZ)
- 国泰君安 (601211.SH)
- 中芯国际 (688981.SH)

注意：
- trade_date 格式为 YYYYMMDD（整数类型）
- 当用户询问"最近"数据时，请根据实际情况查询，目前最新数据是2026年1月9日
- 查询股票数据时，通常使用 name 列过滤股票名称，使用 trade_date 过滤时间范围
- 查询收盘价走势时，通常需要选择 name, trade_date, close 等字段

当我回答用户关于具体数据的问题时（如股价、成交量等），我应该：
1. 优先使用表格形式展示数据，而不是文本列表
2. 表格应包含清晰的列标题
3. 对于时间序列数据，日期列应放在表格左侧
4. 日期格式应转换为 YYYY-MM-DD 的可读形式

例如，对于收盘价数据，应使用如下格式：
| 交易日期 | 股票名称 | 收盘价 |
|----------|----------|--------|
| 2026-01-09 | 贵州茅台 | 1419.10 |
| 2026-01-08 | 贵州茅台 | 1412.30 |

当我需要预测股票价格时，我会使用arima_stock工具，传入股票代码和预测天数。
当用户需要使用MACD指标分析股票交易信号时，我会使用macd_stock工具，传入股票代码。
当用户需要使用布林带分析股票超买超卖状态时，我会使用boll_detection工具，传入股票代码和其他可选参数。
当用户需要分析股票价格的周期性模式（趋势、周季节性、年季节性）时，我会使用prophet_analysis工具，传入股票代码和时间范围。

每当 exc_sql_stock、arima_stock、macd_stock、boll_detection 或 prophet_analysis 工具返回 markdown 表格。
"""

functions_desc = [
    {
        "name": "exc_sql_stock",
        "description": "对于生成的SQL，进行股票数据查询",
        "parameters": {
            "type": "object",
            "properties": {
                "sql_input": {
                    "type": "string",
                    "description": "生成的SQL语句",
                }
            },
            "required": ["sql_input"],
        },
    },
    {
        "name": "arima_stock",
        "description": "使用ARIMA模型预测股票价格，传入股票代码和预测天数",
        "parameters": {
            "type": "object",
            "properties": {
                "ts_code": {
                    "type": "string",
                    "description": "股票代码，必填项",
                },
                "n": {
                    "type": "integer",
                    "description": "预测天数，默认为5天",
                }
            },
            "required": ["ts_code"],
        },
    },
    {
        "name": "macd_stock",
        "description": "使用MACD指标分析股票交易信号，传入股票代码，返回过去一年的买卖点及收益率",
        "parameters": {
            "type": "object",
            "properties": {
                "ts_code": {
                    "type": "string",
                    "description": "股票代码，必填项",
                }
            },
            "required": ["ts_code"],
        },
    },
    {
        "name": "boll_detection",
        "description": "使用布林带检测股票的超买和超卖点，传入股票代码和时间范围，返回超买超卖日期及收益率",
        "parameters": {
            "type": "object",
            "properties": {
                "ts_code": {
                    "type": "string",
                    "description": "股票代码，必填项",
                },
                "period": {
                    "type": "integer",
                    "description": "布林带计算周期，默认为20",
                },
                "std_dev": {
                    "type": "number",
                    "description": "标准差倍数，默认为2",
                },
                "start_date": {
                    "type": "string",
                    "description": "开始日期，格式为YYYYMMDD，如不提供则默认为一年前",
                },
                "end_date": {
                    "type": "string",
                    "description": "结束日期，格式为YYYYMMDD，如不提供则默认为今天",
                }
            },
            "required": ["ts_code"],
        },
    },
    {
        "name": "prophet_analysis",
        "description": "使用Prophet模型分析股票价格的周期性，包括趋势、每周和每年的规律，以及可视化展示",
        "parameters": {
            "type": "object",
            "properties": {
                "ts_code": {
                    "type": "string",
                    "description": "股票代码，必填项",
                },
                "start_date": {
                    "type": "string",
                    "description": "开始日期，格式为YYYYMMDD，如不提供则默认为一年前",
                },
                "end_date": {
                    "type": "string",
                    "description": "结束日期，格式为YYYYMMDD，如不提供则默认为今天",
                }
            },
            "required": ["ts_code"],
        },
    },
]

# ====== 会话隔离 DataFrame 存储 ======
# 用于存储每个会话的 DataFrame，避免多用户数据串扰
_last_df_dict = {}

def get_session_id(kwargs):
    """根据 kwargs 获取当前会话的唯一 session_id，这里用 messages 的 id"""
    messages = kwargs.get('messages')
    if messages is not None:
        return id(messages)
    return None

# ====== exc_sql_stock 工具类实现 ======
@register_tool('exc_sql_stock')
class ExcStockSQLTool(BaseTool):
    """
    股票SQL查询工具，执行传入的SQL语句并返回结果，并自动进行可视化。
    """
    description = '对于生成的SQL，进行股票数据查询，并自动可视化'
    parameters = [{
        'name': 'sql_input',
        'type': 'string',
        'description': '生成的SQL语句',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        import json
        import matplotlib.pyplot as plt
        import io, os, time
        import numpy as np
        from datetime import datetime
        import base64
        import uuid
        
        args = json.loads(params)
        sql_input = args['sql_input']
        
        # 使用SQLite数据库连接
        db_path = os.path.join(os.path.dirname(__file__), 'db_data', 'stock_data.db')
        if not os.path.exists(db_path):
            return f"数据库文件不存在: {db_path}"
        
        try:
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query(sql_input, conn)
            conn.close()
            
            # 检查是否有查询结果
            if df.empty:
                return f"查询结果为空。请检查SQL语句：{sql_input}\n\n可能的原因：\n1. 指定的日期范围内没有数据\n2. 股票名称拼写错误\n3. 数据库中不存在符合条件的记录"
            
            # 处理日期格式，如果存在trade_date列，将其从YYYYMMDD格式转换为YYYY-MM-DD格式
            original_df = df.copy()  # 保留原始数据用于绘图
            if 'trade_date' in df.columns:
                df_copy = df.copy()
                # 将整数类型的日期转换为datetime类型
                df_copy['trade_date'] = pd.to_datetime(df_copy['trade_date'], format='%Y%m%d')
                # 再转换为字符串格式显示
                df_copy['trade_date'] = df_copy['trade_date'].dt.strftime('%Y-%m-%d')
                display_df = df_copy.head(10)
            else:
                display_df = df.head(10)
            
            md = display_df.to_markdown(index=False)
            
            # 如果数据超过10条，添加说明
            if len(df) > 10:
                summary_info = f"\n\n注：表格显示前10条数据，共查询到{len(df)}条记录。完整数据已在图表中可视化展示。"
            else:
                summary_info = f"\n\n已显示全部{len(df)}条查询结果。"
            
            # 生成图表
            chart_info = ""
            if 'trade_date' in df.columns and any(col in df.columns for col in ['close', 'open', 'high', 'low']):
                # 准备绘图
                plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
                plt.rcParams['axes.unicode_minus'] = False
                
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # 如果有多个股票，分别绘制
                if 'name' in df.columns:
                    stocks = df['name'].unique()
                    for stock in stocks:
                        stock_data = df[df['name'] == stock].copy()
                        # 使用原始数据进行日期转换
                        stock_dates = pd.to_datetime(stock_data['trade_date'], format='%Y%m%d', errors='coerce')
                        
                        # 按日期排序
                        sorted_indices = stock_dates.argsort()
                        stock_data_sorted = stock_data.iloc[sorted_indices]
                        stock_dates_sorted = stock_dates.iloc[sorted_indices]
                        
                        # 如果数据点太多，进行采样以避免日期标签过于密集
                        if len(stock_data_sorted) > 20:
                            # 计算采样步长，确保最多显示20个数据点
                            step = len(stock_data_sorted) // 20
                            sampled_indices = np.arange(0, len(stock_data_sorted), step)
                            
                            # 确保至少包含最后一个数据点
                            if len(sampled_indices) == 0 or sampled_indices[-1] != len(stock_data_sorted) - 1:
                                sampled_indices = np.append(sampled_indices, len(stock_data_sorted) - 1)
                            
                            stock_data_sorted = stock_data_sorted.iloc[sampled_indices]
                            stock_dates_sorted = stock_dates_sorted.iloc[sampled_indices]
                        
                        # 绘制价格线，尝试按重要性顺序选择列
                        if 'close' in stock_data_sorted.columns:
                            ax.plot(stock_dates_sorted, stock_data_sorted['close'], label=f'{stock} 收盘价', marker='o', linewidth=2)
                        elif 'open' in stock_data_sorted.columns:
                            ax.plot(stock_dates_sorted, stock_data_sorted['open'], label=f'{stock} 开盘价', marker='o', linewidth=2)
                        elif 'high' in stock_data_sorted.columns:
                            ax.plot(stock_dates_sorted, stock_data_sorted['high'], label=f'{stock} 最高价', marker='o', linewidth=2)
                        elif 'low' in stock_data_sorted.columns:
                            ax.plot(stock_dates_sorted, stock_data_sorted['low'], label=f'{stock} 最低价', marker='o', linewidth=2)
                else:
                    # 单一股票或无名称区分
                    df_plot = df.copy()
                    plot_dates = pd.to_datetime(df_plot['trade_date'], format='%Y%m%d', errors='coerce')
                    
                    # 按日期排序
                    sorted_indices = plot_dates.argsort()
                    df_plot_sorted = df_plot.iloc[sorted_indices]
                    plot_dates_sorted = plot_dates.iloc[sorted_indices]
                    
                    # 如果数据点太多，进行采样以避免日期标签过于密集
                    if len(df_plot_sorted) > 20:
                        # 计算采样步长，确保最多显示20个数据点
                        step = len(df_plot_sorted) // 20
                        sampled_indices = np.arange(0, len(df_plot_sorted), step)
                        
                        # 确保至少包含最后一个数据点
                        if len(sampled_indices) == 0 or sampled_indices[-1] != len(df_plot_sorted) - 1:
                            sampled_indices = np.append(sampled_indices, len(df_plot_sorted) - 1)
                        
                        df_plot_sorted = df_plot_sorted.iloc[sampled_indices]
                        plot_dates_sorted = plot_dates_sorted.iloc[sampled_indices]
                    
                    if 'close' in df_plot_sorted.columns:
                        ax.plot(plot_dates_sorted, df_plot_sorted['close'], label='收盘价', marker='o', linewidth=2)
                    elif 'open' in df_plot_sorted.columns:
                        ax.plot(plot_dates_sorted, df_plot_sorted['open'], label='开盘价', marker='o', linewidth=2)
                    elif 'high' in df_plot_sorted.columns:
                        ax.plot(plot_dates_sorted, df_plot_sorted['high'], label='最高价', marker='o', linewidth=2)
                    elif 'low' in df_plot_sorted.columns:
                        ax.plot(plot_dates_sorted, df_plot_sorted['low'], label='最低价', marker='o', linewidth=2)
                
                ax.set_xlabel('日期')
                ax.set_ylabel('价格')
                ax.set_title('股价走势图')
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.6)
                
                # 旋转x轴标签以适应较长的日期格式
                plt.xticks(rotation=45)
                
                # 调整布局
                plt.tight_layout()
                
                # 定义资源目录路径
                resource_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resource')
                
                # 生成唯一的文件名
                unique_filename = f"chart_{uuid.uuid4().hex}.png"
                chart_path = os.path.join(resource_dir, unique_filename)
                
                # 确保资源目录存在
                os.makedirs(resource_dir, exist_ok=True)
                
                # 保存图表到文件
                plt.savefig(chart_path, format='png', dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                # 返回文件路径信息
                chart_info = f'\n\n图表已保存至本地文件：{chart_path}'
            
            return f"## 查询结果表格\n\n{md}{summary_info}{chart_info}"
        except Exception as e:
            return f"SQL执行或可视化出错: {str(e)}\n\nSQL语句：{sql_input}"

# ====== ARIMA股票预测工具实现 ======
@register_tool('arima_stock')
class ArimaStockTool(BaseTool):
    """
    ARIMA股票预测工具，根据历史数据预测未来N天的股票价格
    """
    description = '使用ARIMA模型预测股票价格，输入股票代码和预测天数'
    parameters = [
        {
            'name': 'ts_code',
            'type': 'string',
            'description': '股票代码，必填项',
            'required': True
        },
        {
            'name': 'n',
            'type': 'integer',
            'description': '预测天数，默认为5天',
            'required': False
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        import json
        import matplotlib.pyplot as plt
        import io
        import os
        import numpy as np
        from datetime import datetime, timedelta
        import base64
        import uuid
        from statsmodels.tsa.arima.model import ARIMA
        from pandas.tseries.offsets import BDay  # Business Day for trading days
        
        args = json.loads(params)
        ts_code = args['ts_code']
        n = args.get('n', 5)  # 默认预测5天
        
        # 使用SQLite数据库连接
        db_path = os.path.join(os.path.dirname(__file__), 'db_data', 'stock_data.db')
        if not os.path.exists(db_path):
            return f"数据库文件不存在: {db_path}"
        
        try:
            # 从数据库获取股票数据
            conn = sqlite3.connect(db_path)
            
            # 获取当前日期和一年前的日期
            today = datetime.now()
            one_year_ago = today - timedelta(days=365)
            date_str = int(one_year_ago.strftime('%Y%m%d'))
            
            # 查询过去一年的数据
            query = f"""
            SELECT trade_date, close 
            FROM stock_data 
            WHERE ts_code = '{ts_code}' 
            AND trade_date >= {date_str}
            ORDER BY trade_date ASC
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                return f"数据库中没有找到股票代码为 {ts_code} 的数据或指定时间范围内的数据"
            
            # 将日期从整数格式转换为日期格式
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
            df.sort_values(by='trade_date', inplace=True)
            
            # 提取收盘价序列
            ts = df['close'].dropna()
            
            if len(ts) < 10:
                return f"数据不足，无法进行ARIMA模型拟合。股票 {ts_code} 只有 {len(ts)} 条有效数据。"
            
            # 使用ARIMA模型进行拟合和预测
            # 使用ARIMA(5,1,5)模型参数
            try:
                model = ARIMA(ts, order=(5, 1, 5))
                fitted_model = model.fit()
                
                # 预测未来n天
                forecast_result = fitted_model.forecast(steps=n)
                forecast_dates = []
                last_date = df['trade_date'].iloc[-1]
                
                # 计算未来交易日（跳过周末）
                for i in range(1, n + 1):
                    next_date = last_date + BDay(i)  # 使用交易日计算，跳过周末
                    forecast_dates.append(next_date)
                
                # 创建预测结果DataFrame
                forecast_df = pd.DataFrame({
                    'trade_date': forecast_dates,
                    'predicted_close': forecast_result
                })
                
                # 为了可视化，我们也需要历史数据的一部分
                history_for_plot = df.tail(30)  # 取最近30天的数据用于可视化
                
                # 准备绘图
                plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
                plt.rcParams['axes.unicode_minus'] = False
                
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # 绘制历史数据
                ax.plot(history_for_plot['trade_date'], history_for_plot['close'], 
                       label='历史收盘价', marker='o', linewidth=2)
                
                # 绘制预测数据
                ax.plot(forecast_df['trade_date'], forecast_df['predicted_close'], 
                       label=f'预测收盘价({n}天)', marker='s', linestyle='--', linewidth=2, color='red')
                
                # 添加分割线标识预测开始点
                if not history_for_plot.empty and not forecast_df.empty:
                    split_x = [history_for_plot['trade_date'].iloc[-1]] * 2
                    split_y = [min(min(history_for_plot['close']), min(forecast_df['predicted_close'])) * 0.95, 
                              max(max(history_for_plot['close']), max(forecast_df['predicted_close'])) * 1.05]
                    ax.plot(split_x, split_y, linestyle=':', color='gray', alpha=0.7)
                
                ax.set_xlabel('日期')
                ax.set_ylabel('价格')
                ax.set_title(f'股票 {ts_code} ARIMA模型价格预测')
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.6)
                
                # 旋转x轴标签以适应较长的日期格式
                plt.xticks(rotation=45)
                
                # 调整布局
                plt.tight_layout()
                
                # 创建资源目录
                resource_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resource')
                os.makedirs(resource_dir, exist_ok=True)
                
                # 生成唯一文件名
                unique_filename = f"arima_chart_{uuid.uuid4().hex}.png"
                chart_path = os.path.join(resource_dir, unique_filename)
                
                # 保存图表
                plt.savefig(chart_path, format='png', dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                # 准备预测结果表格
                forecast_df_display = forecast_df.copy()
                forecast_df_display['trade_date'] = forecast_df_display['trade_date'].dt.strftime('%Y-%m-%d')
                forecast_df_display.rename(columns={'trade_date': '交易日期', 'predicted_close': '预测收盘价'}, inplace=True)
                
                md = forecast_df_display.to_markdown(index=False)
                
                return f"""## ARIMA股票价格预测结果
股票代码：{ts_code}
预测天数：{n}天

### 预测详情：
{md}

### 预测图表：
图表已保存至本地文件：{chart_path}"""
                
            except Exception as model_error:
                return f"ARIMA模型训练或预测过程中出现错误: {str(model_error)}"
                
        except Exception as e:
            return f"数据库查询或预测过程出错: {str(e)}"

# ====== MACD股票交易分析工具实现 ======
@register_tool('macd_stock')
class MacdStockTool(BaseTool):
    """
    MACD股票交易分析工具，识别过去一年的买卖点并计算收益率
    """
    description = '使用MACD指标分析股票交易信号，输入股票代码，返回过去一年的买卖点及收益率'
    parameters = [
        {
            'name': 'ts_code',
            'type': 'string',
            'description': '股票代码，必填项',
            'required': True
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        import json
        import matplotlib.pyplot as plt
        import io
        import os
        import numpy as np
        from datetime import datetime, timedelta
        import base64
        import uuid
        import pandas as pd
        import sqlite3
        
        args = json.loads(params)
        ts_code = args['ts_code']
        
        # 使用SQLite数据库连接
        db_path = os.path.join(os.path.dirname(__file__), 'db_data', 'stock_data.db')
        if not os.path.exists(db_path):
            return f"数据库文件不存在: {db_path}"
        
        try:
            # 从数据库获取股票数据
            conn = sqlite3.connect(db_path)
            
            # 获取当前日期和一年前的日期
            today = datetime.now()
            one_year_ago = today - timedelta(days=365)
            date_str = int(one_year_ago.strftime('%Y%m%d'))
            
            # 查询过去一年的数据
            query = f"""
            SELECT trade_date, close, name
            FROM stock_data 
            WHERE ts_code = '{ts_code}' 
            AND trade_date >= {date_str}
            ORDER BY trade_date ASC
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                return f"数据库中没有找到股票代码为 {ts_code} 的数据或指定时间范围内的数据"
            
            # 将日期从整数格式转换为日期格式
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
            df.sort_values(by='trade_date', inplace=True)
            
            if len(df) < 30:
                return f"数据不足，无法进行MACD分析。股票 {ts_code} 只有 {len(df)} 条有效数据。"
            
            # 计算MACD指标
            # 短期EMA (12天)
            ema_short = df['close'].ewm(span=12).mean()
            # 长期EMA (26天)
            ema_long = df['close'].ewm(span=26).mean()
            # DIF (差离值)
            dif = ema_short - ema_long
            # DEA (DIF的9天EMA)
            dea = dif.ewm(span=9).mean()
            # MACD柱状图 (DIF-DEA)*2
            bar = (dif - dea) * 2
            
            # 将计算结果添加到df中
            df['ema_short'] = ema_short
            df['ema_long'] = ema_long
            df['dif'] = dif
            df['dea'] = dea
            df['macd'] = bar
            
            # 识别买入和卖出信号
            # 金叉：DIF上穿DEA (DIF从下方穿越到DEA上方)
            buy_signals = []
            sell_signals = []
            
            for i in range(1, len(df)):
                # 金叉 - 买入信号
                if df['dif'].iloc[i-1] <= df['dea'].iloc[i-1] and df['dif'].iloc[i] > df['dea'].iloc[i]:
                    buy_signals.append({
                        'date': df['trade_date'].iloc[i],
                        'price': df['close'].iloc[i],
                        'type': 'BUY',
                        'signal': '金叉'
                    })
                # 死叉 - 卖出信号
                elif df['dif'].iloc[i-1] >= df['dea'].iloc[i-1] and df['dif'].iloc[i] < df['dea'].iloc[i]:
                    sell_signals.append({
                        'date': df['trade_date'].iloc[i],
                        'price': df['close'].iloc[i],
                        'type': 'SELL',
                        'signal': '死叉'
                    })
            
            # 交易模拟
            initial_amount = 10000  # 初始资金
            cash = initial_amount
            shares = 0
            transactions = []  # 记录所有交易
            balance_history = [{'date': df['trade_date'].iloc[0], 'balance': cash}]  # 记录资金历史
            
            # 按时间顺序处理交易信号
            buy_idx = 0
            sell_idx = 0
            
            for idx, row in df.iterrows():
                # 检查是否有买入信号在当前日期
                while (buy_idx < len(buy_signals) and 
                       buy_signals[buy_idx]['date'] == row['trade_date']):
                    if cash > 0:  # 有现金可以买入
                        shares = cash / row['close']  # 满仓买入
                        cash = 0
                        transactions.append({
                            'date': buy_signals[buy_idx]['date'],
                            'type': 'BUY',
                            'price': buy_signals[buy_idx]['price'],
                            'shares': shares,
                            'cash': cash,
                            'action': '买入'
                        })
                    buy_idx += 1
                
                # 检查是否有卖出信号在当前日期
                while (sell_idx < len(sell_signals) and 
                       sell_signals[sell_idx]['date'] == row['trade_date']):
                    if shares > 0:  # 有股票可以卖出
                        cash = shares * row['close']  # 清仓卖出
                        shares = 0
                        transactions.append({
                            'date': sell_signals[sell_idx]['date'],
                            'type': 'SELL',
                            'price': sell_signals[sell_idx]['price'],
                            'shares': 0,
                            'cash': cash,
                            'action': '卖出'
                        })
                    sell_idx += 1
                
                # 记录当前资金情况（如果有持仓则加上股票价值）
                current_balance = cash + shares * row['close']
                balance_history.append({
                    'date': row['trade_date'],
                    'balance': current_balance
                })
            
            # 如果最后还有持仓，则卖出
            if shares > 0:
                final_cash = shares * df['close'].iloc[-1]
                transactions.append({
                    'date': df['trade_date'].iloc[-1],
                    'type': 'SELL',
                    'price': df['close'].iloc[-1],
                    'shares': 0,
                    'cash': final_cash,
                    'action': '最终卖出'
                })
                cash = final_cash
                shares = 0
            
            # 计算收益率
            final_balance = cash
            total_return = (final_balance - initial_amount) / initial_amount * 100
            
            # 准备交易信号表格
            signals_df = pd.DataFrame(buy_signals + sell_signals)
            if not signals_df.empty:
                signals_df['trade_date'] = signals_df['date'].dt.strftime('%Y-%m-%d')
                signals_df = signals_df.rename(columns={
                    'trade_date': '交易日期',
                    'price': '价格',
                    'signal': '信号类型'
                })
                signals_md = signals_df[['交易日期', '价格', '信号类型']].to_markdown(index=False)
            else:
                signals_md = "在过去一年中未检测到任何MACD交易信号"
            
            # 准备交易记录表格
            if transactions:
                transactions_df = pd.DataFrame(transactions)
                transactions_df['date'] = transactions_df['date'].dt.strftime('%Y-%m-%d')
                transactions_df = transactions_df.rename(columns={
                    'date': '交易日期',
                    'price': '价格',
                    'action': '操作'
                })
                transactions_md = transactions_df[['交易日期', '操作', '价格']].to_markdown(index=False)
            else:
                transactions_md = "在过去一年中未执行任何交易"
            
            # 准备绘图
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False
            
            fig, axes = plt.subplots(2, 1, figsize=(14, 12))
            
            # 第一个子图：价格和MACD线
            axes[0].plot(df['trade_date'], df['close'], label='收盘价', color='black')
            
            # 标记买入点
            buy_dates = [s['date'] for s in buy_signals]
            buy_prices = [s['price'] for s in buy_signals]
            if buy_dates:
                axes[0].scatter(buy_dates, buy_prices, c='red', label='买入信号', marker='^', s=100)
            
            # 标记卖出点
            sell_dates = [s['date'] for s in sell_signals]
            sell_prices = [s['price'] for s in sell_signals]
            if sell_dates:
                axes[0].scatter(sell_dates, sell_prices, c='green', label='卖出信号', marker='v', s=100)
            
            axes[0].set_title(f'股票 {ts_code} MACD交易信号分析')
            axes[0].set_ylabel('价格')
            axes[0].legend()
            axes[0].grid(True, linestyle='--', alpha=0.6)
            axes[0].tick_params(axis='x', rotation=45)
            
            # 第二个子图：MACD指标
            axes[1].plot(df['trade_date'], df['dif'], label='DIF', color='blue')
            axes[1].plot(df['trade_date'], df['dea'], label='DEA', color='orange')
            # 绘制MACD柱状图
            colors = ['red' if val >= 0 else 'green' for val in df['macd']]
            axes[1].bar(df['trade_date'], df['macd'], label='MACD柱', color=colors, alpha=0.3, width=1)
            axes[1].set_title('MACD指标 (DIF, DEA, MACD柱)')
            axes[1].set_xlabel('日期')
            axes[1].set_ylabel('MACD值')
            axes[1].legend()
            axes[1].grid(True, linestyle='--', alpha=0.6)
            axes[1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # 创建资源目录
            resource_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resource')
            os.makedirs(resource_dir, exist_ok=True)
            
            # 生成唯一文件名
            unique_filename = f"macd_chart_{uuid.uuid4().hex}.png"
            chart_path = os.path.join(resource_dir, unique_filename)
            
            # 保存图表
            plt.savefig(chart_path, format='png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # 获取股票名称
            stock_name = df['name'].iloc[0] if not df.empty else '未知股票'
            
            return f"""## MACD股票交易分析结果
股票代码：{ts_code} ({stock_name})
分析周期：过去一年

### MACD交易信号：
{signals_md}

### 交易记录：
{transactions_md}

### 收益情况：
- 初始资金：10,000元
- 最终资金：{final_balance:.2f}元
- 总收益：{final_balance - 10000:.2f}元
- 收益率：{total_return:.2f}%

### MACD分析图表：
图表已保存至本地文件：{chart_path}"""


        except Exception as e:
            return f"MACD分析过程出错: {str(e)}"

# ====== BOLL布林带检测工具实现 ======
@register_tool('boll_detection')
class BollDetectionTool(BaseTool):
    """
    BOLL布林带检测工具，用于检测股票的超买和超卖点
    """
    description = '使用布林带检测股票的超买和超卖点，输入股票代码和时间范围，返回超买超卖日期及收益率'
    parameters = [
        {
            'name': 'ts_code',
            'type': 'string',
            'description': '股票代码，必填项',
            'required': True
        },
        {
            'name': 'period',
            'type': 'integer',
            'description': '布林带计算周期，默认为20',
            'required': False
        },
        {
            'name': 'std_dev',
            'type': 'number',
            'description': '标准差倍数，默认为2',
            'required': False
        },
        {
            'name': 'start_date',
            'type': 'string',
            'description': '开始日期，格式为YYYYMMDD，如不提供则默认为一年前',
            'required': False
        },
        {
            'name': 'end_date',
            'type': 'string',
            'description': '结束日期，格式为YYYYMMDD，如不提供则默认为今天',
            'required': False
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        import json
        import matplotlib.pyplot as plt
        import io
        import os
        import numpy as np
        from datetime import datetime, timedelta
        import base64
        import uuid
        import pandas as pd
        import sqlite3
        
        args = json.loads(params)
        ts_code = args['ts_code']
        period = args.get('period', 20)  # 默认20日周期
        std_dev = args.get('std_dev', 2)  # 默认2倍标准差
        start_date = args.get('start_date')
        end_date = args.get('end_date')
        
        # 使用SQLite数据库连接
        db_path = os.path.join(os.path.dirname(__file__), 'db_data', 'stock_data.db')
        if not os.path.exists(db_path):
            return f"数据库文件不存在: {db_path}"
        
        try:
            # 从数据库获取股票数据
            conn = sqlite3.connect(db_path)
            
            # 构建查询条件
            query_conditions = f"WHERE ts_code = '{ts_code}'"
            
            if start_date:
                query_conditions += f" AND trade_date >= {start_date}"
            else:
                # 默认获取一年前的数据
                today = datetime.now()
                one_year_ago = today - timedelta(days=365)
                date_str = int(one_year_ago.strftime('%Y%m%d'))
                query_conditions += f" AND trade_date >= {date_str}"
                
            if end_date:
                query_conditions += f" AND trade_date <= {end_date}"
            else:
                # 默认结束日期为今天
                today = datetime.now()
                query_conditions += f" AND trade_date <= {int(today.strftime('%Y%m%d'))}"
            
            # 查询数据
            query = f"""
            SELECT trade_date, close, name
            FROM stock_data 
            {query_conditions}
            ORDER BY trade_date ASC
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                return f"数据库中没有找到股票代码为 {ts_code} 的数据或指定时间范围内的数据"
            
            # 将日期从整数格式转换为日期格式
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
            df.sort_values(by='trade_date', inplace=True)
            
            if len(df) < period:
                return f"数据不足，无法进行布林带计算。股票 {ts_code} 在指定时间段只有 {len(df)} 条有效数据，但计算布林带需要至少 {period} 条数据。"
            
            # 计算布林带
            df['MA'] = df['close'].rolling(window=period).mean()  # 移动平均线
            df['STD'] = df['close'].rolling(window=period).std()  # 标准差
            df['upper_band'] = df['MA'] + (df['STD'] * std_dev)  # 上轨
            df['lower_band'] = df['MA'] - (df['STD'] * std_dev)  # 下轨
            
            # 识别超买和超卖点
            oversold_points = []  # 超卖点 - 价格触及下轨后反弹
            overbought_points = []  # 超买点 - 价格触及上轨后回落
            
            # 寻找突破上下轨的点位
            for i in range(1, len(df)):
                prev_close = df['close'].iloc[i-1]
                curr_close = df['close'].iloc[i]
                prev_lower = df['lower_band'].iloc[i-1]
                curr_lower = df['lower_band'].iloc[i]
                prev_upper = df['upper_band'].iloc[i-1]
                curr_upper = df['upper_band'].iloc[i]
                
                # 检查是否从下轨下方突破回到轨道内（超卖反弹）
                if prev_close <= prev_lower and curr_close > curr_lower:
                    oversold_points.append({
                        'date': df['trade_date'].iloc[i],
                        'price': df['close'].iloc[i],
                        'type': 'OVERSOLD',
                        'band_type': 'LOWER'
                    })
                
                # 检查是否从上轨上方跌破回到轨道内（超买回落）
                if prev_close >= prev_upper and curr_close < curr_upper:
                    overbought_points.append({
                        'date': df['trade_date'].iloc[i],
                        'price': df['close'].iloc[i],
                        'type': 'OVERBOUGHT',
                        'band_type': 'UPPER'
                    })
            
            # 交易模拟
            initial_amount = 10000  # 初始资金
            cash = initial_amount
            shares = 0
            transactions = []  # 记录所有交易
            balance_history = [{'date': df['trade_date'].iloc[0], 'balance': cash}]  # 记录资金历史
            
            # 按时间顺序处理交易信号
            oversold_idx = 0
            overbought_idx = 0
            
            for idx, row in df.iterrows():
                # 检查是否有超卖信号在当前日期
                while (oversold_idx < len(oversold_points) and 
                       oversold_points[oversold_idx]['date'] == row['trade_date']):
                    # 超卖买入 - 满仓买入
                    if cash > 0:
                        shares = cash / row['close']  # 满仓买入
                        cash = 0
                        transactions.append({
                            'date': oversold_points[oversold_idx]['date'],
                            'type': 'BUY',
                            'price': oversold_points[oversold_idx]['price'],
                            'shares': shares,
                            'cash': cash,
                            'action': '超卖买入'
                        })
                    oversold_idx += 1
                
                # 检查是否有超买信号在当前日期
                while (overbought_idx < len(overbought_points) and 
                       overbought_points[overbought_idx]['date'] == row['trade_date']):
                    # 超买卖出 - 清仓卖出
                    if shares > 0:
                        cash = shares * row['close']  # 清仓卖出
                        shares = 0
                        transactions.append({
                            'date': overbought_points[overbought_idx]['date'],
                            'type': 'SELL',
                            'price': overbought_points[overbought_idx]['price'],
                            'shares': 0,
                            'cash': cash,
                            'action': '超卖卖出'
                        })
                    overbought_idx += 1
                
                # 记录当前资金情况（如果有持仓则加上股票价值）
                current_balance = cash + shares * row['close']
                balance_history.append({
                    'date': row['trade_date'],
                    'balance': current_balance
                })
            
            # 如果最后还有持仓，则卖出
            if shares > 0:
                final_cash = shares * df['close'].iloc[-1]
                transactions.append({
                    'date': df['trade_date'].iloc[-1],
                    'type': 'SELL',
                    'price': df['close'].iloc[-1],
                    'shares': 0,
                    'cash': final_cash,
                    'action': '最终卖出'
                })
                cash = final_cash
                shares = 0
            
            # 计算收益率
            final_balance = cash
            total_return = (final_balance - initial_amount) / initial_amount * 100
            
            # 准备超买超卖信号表格
            oversold_df = pd.DataFrame(oversold_points)
            overbought_df = pd.DataFrame(overbought_points)
            
            all_signals = []
            if not oversold_df.empty:
                for _, signal in oversold_df.iterrows():
                    all_signals.append({
                        '交易日期': signal['date'].strftime('%Y-%m-%d'),
                        '价格': signal['price'],
                        '信号类型': '超卖买入'
                    })
            if not overbought_df.empty:
                for _, signal in overbought_df.iterrows():
                    all_signals.append({
                        '交易日期': signal['date'].strftime('%Y-%m-%d'),
                        '价格': signal['price'],
                        '信号类型': '超买卖出'
                    })
            
            if all_signals:
                signals_df = pd.DataFrame(all_signals)
                signals_df = signals_df.sort_values(by='交易日期')
                signals_md = signals_df.to_markdown(index=False)
            else:
                signals_md = "在指定时间段内未检测到任何布林带交易信号"
            
            # 准备交易记录表格
            if transactions:
                transactions_df = pd.DataFrame(transactions)
                transactions_df['date'] = transactions_df['date'].dt.strftime('%Y-%m-%d')
                transactions_df = transactions_df.rename(columns={
                    'date': '交易日期',
                    'price': '价格',
                    'action': '操作'
                })
                transactions_md = transactions_df[['交易日期', '操作', '价格']].to_markdown(index=False)
            else:
                transactions_md = "在指定时间段内未执行任何交易"
            
            # 准备绘图
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False
            
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # 绘制价格线
            ax.plot(df['trade_date'], df['close'], label='收盘价', color='black', linewidth=1)
            
            # 绘制移动平均线
            ax.plot(df['trade_date'], df['MA'], label=f'{period}日移动平均线', color='blue', linestyle='--')
            
            # 绘制布林带
            ax.plot(df['trade_date'], df['upper_band'], label='上轨', color='red', linestyle='--')
            ax.plot(df['trade_date'], df['lower_band'], label='下轨', color='green', linestyle='--')
            ax.fill_between(df['trade_date'], df['upper_band'], df['lower_band'], alpha=0.1, color='gray')
            
            # 标记超卖点
            if oversold_points:
                oversold_dates = [s['date'] for s in oversold_points]
                oversold_prices = [s['price'] for s in oversold_points]
                ax.scatter(oversold_dates, oversold_prices, c='green', label='超卖买入信号', marker='^', s=100, zorder=5)
            
            # 标记超买点
            if overbought_points:
                overbought_dates = [s['date'] for s in overbought_points]
                overbought_prices = [s['price'] for s in overbought_points]
                ax.scatter(overbought_dates, overbought_prices, c='red', label='超买卖出信号', marker='v', s=100, zorder=5)
            
            ax.set_title(f'股票 {ts_code} 布林带分析 (周期={period}, 标准差={std_dev})')
            ax.set_xlabel('日期')
            ax.set_ylabel('价格')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # 创建资源目录
            resource_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resource')
            os.makedirs(resource_dir, exist_ok=True)
            
            # 生成唯一文件名
            unique_filename = f"boll_chart_{uuid.uuid4().hex}.png"
            chart_path = os.path.join(resource_dir, unique_filename)
            
            # 保存图表
            plt.savefig(chart_path, format='png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # 获取股票名称
            stock_name = df['name'].iloc[0] if not df.empty else '未知股票'
            
            # 构建结果字符串
            result = f"""## 布林带分析结果
股票代码：{ts_code} ({stock_name})
分析周期：{period}日，标准差：{std_dev}倍
时间范围：{df['trade_date'].iloc[0].strftime('%Y-%m-%d')} 至 {df['trade_date'].iloc[-1].strftime('%Y-%m-%d')}

### 布林带交易信号：
{signals_md}

### 交易记录：
{transactions_md}

### 收益情况：
- 初始资金：10,000元
- 最终资金：{final_balance:.2f}元
- 总收益：{final_balance - 10000:.2f}元
- 收益率：{total_return:.2f}%

### 布林带分析图表：
图表已保存至本地文件：{chart_path}"""

            return result

        except Exception as e:
            return f"布林带分析过程出错: {str(e)}"

# ====== Prophet股票周期性分析工具实现 ======
@register_tool('prophet_analysis')
class ProphetAnalysisTool(BaseTool):
    """
    Prophet股票周期性分析工具，用于分析股票价格的趋势、周季节性和年季节性
    """
    description = '使用Prophet模型分析股票价格的周期性，包括趋势、每周和每年的规律，以及可视化展示'
    parameters = [
        {
            'name': 'ts_code',
            'type': 'string',
            'description': '股票代码，必填项',
            'required': True
        },
        {
            'name': 'start_date',
            'type': 'string',
            'description': '开始日期，格式为YYYYMMDD，如不提供则默认为一年前',
            'required': False
        },
        {
            'name': 'end_date',
            'type': 'string',
            'description': '结束日期，格式为YYYYMMDD，如不提供则默认为今天',
            'required': False
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        import json
        import matplotlib.pyplot as plt
        import io
        import os
        import numpy as np
        from datetime import datetime, timedelta
        import base64
        import uuid
        import pandas as pd
        import sqlite3
        from prophet import Prophet
        import warnings
        warnings.filterwarnings("ignore")

        args = json.loads(params)
        ts_code = args['ts_code']
        start_date = args.get('start_date')
        end_date = args.get('end_date')

        # 使用SQLite数据库连接
        db_path = os.path.join(os.path.dirname(__file__), 'db_data', 'stock_data.db')
        if not os.path.exists(db_path):
            return f"数据库文件不存在: {db_path}"

        try:
            # 从数据库获取股票数据
            conn = sqlite3.connect(db_path)

            # 构建查询条件
            query_conditions = f"WHERE ts_code = '{ts_code}'"

            if start_date:
                query_conditions += f" AND trade_date >= {start_date}"
            else:
                # 默认获取一年前的数据
                today = datetime.now()
                one_year_ago = today - timedelta(days=365)
                date_str = int(one_year_ago.strftime('%Y%m%d'))
                query_conditions += f" AND trade_date >= {date_str}"

            if end_date:
                query_conditions += f" AND trade_date <= {end_date}"
            else:
                # 默认结束日期为今天
                today = datetime.now()
                query_conditions += f" AND trade_date <= {int(today.strftime('%Y%m%d'))}"

            # 查询数据
            query = f"""
            SELECT trade_date, close
            FROM stock_data
            {query_conditions}
            ORDER BY trade_date ASC
            """
            df = pd.read_sql_query(query, conn)
            conn.close()

            if df.empty:
                return f"数据库中没有找到股票代码为 {ts_code} 的数据或指定时间范围内的数据"

            # 将日期从整数格式转换为日期格式，并重命名列以符合Prophet要求
            df['ds'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
            df['y'] = df['close']
            df = df[['ds', 'y']].dropna()

            if len(df) < 30:
                return f"数据不足，无法进行Prophet分析。股票 {ts_code} 只有 {len(df)} 条有效数据。"

            # 创建并拟合Prophet模型
            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                interval_width=0.95
            )

            # 添加月度季节性
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

            model.fit(df)

            # 创建未来数据框用于预测（实际上我们只是用它来做分解分析）
            future = model.make_future_dataframe(periods=0)

            # 进行预测和成分分解
            forecast = model.predict(future)

            # 生成组件图
            fig_components = model.plot_components(forecast)
            
            # 创建资源目录
            resource_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resource')
            os.makedirs(resource_dir, exist_ok=True)

            # 生成唯一文件名
            unique_filename = f"prophet_components_{uuid.uuid4().hex}.png"
            chart_path = os.path.join(resource_dir, unique_filename)

            # 保存图表
            fig_components.savefig(chart_path, format='png', dpi=150, bbox_inches='tight')
            plt.close(fig_components)

            # 获取股票名称
            conn = sqlite3.connect(db_path)
            name_query = f"SELECT DISTINCT name FROM stock_data WHERE ts_code = '{ts_code}' LIMIT 1"
            name_result = pd.read_sql_query(name_query, conn)
            conn.close()
            
            stock_name = name_result['name'].iloc[0] if not name_result.empty else '未知股票'

            # 构建结果字符串
            result = f"""## Prophet股票周期性分析结果
股票代码：{ts_code} ({stock_name})
分析时间范围：{df['ds'].min().strftime('%Y-%m-%d')} 至 {df['ds'].max().strftime('%Y-%m-%d')}

### 分析说明：
Prophet模型分析了该股票价格的周期性特征，包括：
1. **趋势(Trend)**：长期价格变动趋势
2. **周季节性(Weekly)**：一周内的价格变化模式
3. **年季节性(Yearly)**：一年内的价格变化模式
4. **月季节性(Monthly)**：一个月内的价格变化模式

### 周期性分析图表：
图表已保存至本地文件：{chart_path}

图表包含了四个部分：
- **Trend**: 显示股票价格的长期趋势变化
- **Weekly**: 显示一周内哪天更倾向于上涨或下跌
- **Yearly**: 显示一年中哪些月份更倾向于上涨或下跌
- **Monthly**: 显示一个月内哪几天更倾向于上涨或下跌
"""

            return result

        except ImportError:
            return "缺少必要的库: prophet。请安装: pip install prophet"
        except Exception as e:
            return f"Prophet分析过程出错: {str(e)}"

# ====== 初始化股票助手服务 ======
def init_agent_service():
    """初始化股票助手服务"""
    llm_cfg = {
        'model': 'qwen-turbo',
        'timeout': 30,
        'retry_count': 3,
    }
    try:
        bot = Assistant(
            llm=llm_cfg,
            name='股票助手',
            description='股票数据查询与分析',
            system_message=system_prompt,
            function_list=['exc_sql_stock', 'arima_stock', 'macd_stock', 'boll_detection', 'prophet_analysis'],  # 添加了prophet_analysis工具
        )
        print("股票助手初始化成功！")
        return bot
    except Exception as e:
        print(f"助手初始化失败: {str(e)}")
        raise

def app_gui():
    """图形界面模式，提供 Web 图形界面"""
    try:
        print("正在启动 Web 界面...")
        # 初始化助手
        bot = init_agent_service()
        # 配置聊天界面，列举典型股票查询问题
        chatbot_config = {
            'prompt.suggestions': [
                '查询贵州茅台最近的收盘价走势',
                '统计各股票的平均成交量',
                '比较五粮液和贵州茅台的收盘价变化趋势',
                '使用ARIMA模型预测贵州茅台未来5天的价格',
                '预测600519.SH股票未来一周的走势',
                '对000858.SZ股票进行未来10天的价格预测',
                '使用MACD分析600519.SH股票过去一年的交易机会',
                '告诉我贵州茅台过去一年的MACD买卖点和收益率',
                '使用布林带分析000858.SZ股票的超买超卖点',
                '检测600519.SH股票在过去一年的布林带买卖信号',
                '分析000858.SZ股票，周期20日，标准差2.5的情况下的超买超卖点',
                '使用Prophet分析贵州茅台的周期性趋势',
                '分析000858.SZ股票的价格周期性，包含趋势、周季节性和年季节性'
            ]
        }
        print("Web 界面准备就绪，正在启动服务...")
        # 启动 Web 界面
        WebUI(
            bot,
            chatbot_config=chatbot_config
        ).run(host='0.0.0.0', port=8080)
    except Exception as e:
        print(f"启动 Web 界面失败: {str(e)}")
        print("请检查网络连接和 API Key 配置")


if __name__ == '__main__':
    # 直接运行GUI模式
    print("启动股票分析助手 - GUI模式")
    app_gui()