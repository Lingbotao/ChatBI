import sqlite3
import pandas as pd

def check_db_data():
    """检查数据库中的数据情况"""
    db_path = 'stock_data.db'
    
    try:
        conn = sqlite3.connect(db_path)
        
        # 检查数据的日期范围
        date_range_df = pd.read_sql_query('SELECT MIN(trade_date), MAX(trade_date) FROM stock_data', conn)
        print('日期范围:', date_range_df.values)
        
        # 检查有哪些股票
        stocks_df = pd.read_sql_query('SELECT DISTINCT name, ts_code FROM stock_data ORDER BY name', conn)
        print('\n股票列表:')
        print(stocks_df)
        
        # 检查贵州茅台的最近数据
        guizhou_df = pd.read_sql_query("SELECT trade_date, open, high, low, close FROM stock_data WHERE name='贵州茅台' ORDER BY trade_date DESC LIMIT 5", conn)
        print('\n贵州茅台最近5个交易日数据:')
        print(guizhou_df)
        
        # 检查最近一个月的数据是否存在（以当前日期2026年1月10日为参考）
        guizhou_recent_df = pd.read_sql_query("SELECT COUNT(*) as count FROM stock_data WHERE name='贵州茅台' AND trade_date >= '2025-12-10'", conn)
        print(f'\n贵州茅台最近一个月(2025-12-10之后)的数据条数: {guizhou_recent_df.iloc[0, 0]}')
        
        # 检查最近3个月的数据
        guizhou_3m_df = pd.read_sql_query("SELECT COUNT(*) as count FROM stock_data WHERE name='贵州茅' AND trade_date >= '2025-10-10'", conn)
        print(f'贵州茅台最近3个月(2025-10-10之后)的数据条数: {guizhou_3m_df.iloc[0, 0]}')
        
        conn.close()
        
    except Exception as e:
        print(f"检查数据库时出错: {str(e)}")

if __name__ == "__main__":
    check_db_data()