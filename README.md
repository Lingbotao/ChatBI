# ChatBI
# 股票分析助手 (Stock Analysis Assistant)

股票分析助手是一个基于自然语言处理的智能股票数据分析工具。用户可以通过简单的自然语言提问，获得复杂的股票数据分析结果，包括历史趋势、价格预测、技术指标分析等。

## 功能

股票分析助手结合了人工智能和金融分析技术，能够理解和执行用户提出的股票分析需求，包括：

- 股票数据查询和可视化
- 股价预测（ARIMA模型）
- MACD技术指标分析
- 布林带超买超卖分析
- Prophet周期性趋势分析

## 技术架构

- **AI模型**: 阿里云Qwen-Turbo模型
- **数据库**: SQLite数据库存储股票数据
- **分析库**: Statsmodels, Prophet, Pandas
- **可视化**: Matplotlib, Plotly

## 数据源
- 系统使用SQLite数据库存储股票数据，包含开盘价、最高价、最低价、收盘价、成交量、成交额等详细信息。


## 使用方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

或者单独执行install_requirements.bat：

### 2. 获取数据
- 执行stock_data.bat获取股票数据

### 3. Excel数据转换成db文件
- 执行excel_to_sqlite.bat

### 启动助手
- 运行start_assistant.bat 或直接运行stock_analysis_assistant_final.py
### 环境配置

需要设置以下环境变量：
```bash
DASHSCOPE_API_KEY=your_api_key_here
```
## 注意事项

- 需要有效的DASHSCOPE_API_KEY才能使用AI功能
- 某些分析功能需要足够的历史数据才能运行
- 所有分析结果仅供参考，不构成投资建议