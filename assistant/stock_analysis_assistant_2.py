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

每当 exc_sql_stock 或 arima_stock 工具返回 markdown 表格。
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
            function_list=['exc_sql_stock', 'arima_stock'],
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
        # 配置聊天界面，列举3个典型股票查询问题
        chatbot_config = {
            'prompt.suggestions': [
                '查询贵州茅台最近的收盘价走势',
                '统计各股票的平均成交量',
                '比较五粮液和贵州茅台的收盘价变化趋势',
                '使用ARIMA模型预测贵州茅台未来5天的价格',
                '预测600519.SH股票未来一周的走势',
                '对000858.SZ股票进行未来10天的价格预测'
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