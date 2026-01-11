import pandas as pd
import sqlite3
from datetime import datetime
import os

def excel_to_sqlite(excel_file, db_file):
    """
    将Excel文件中的股票数据导入SQLite数据库
    """
    # 检查Excel文件是否存在
    if not os.path.exists(excel_file):
        print(f"Excel文件 {excel_file} 不存在")
        return False
    
    # 连接到SQLite数据库（如果不存在则会创建）
    conn = sqlite3.connect(db_file)
    
    try:
        # 读取Excel文件
        print(f"正在读取Excel文件 {excel_file} ...")
        df = pd.read_excel(excel_file)
        
        # 打印数据的基本信息
        print(f"数据形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")
        
        # 根据之前的了解，股票数据应该有以下列：
        # ts_code, trade_date, open, high, low, close, vol, amount, name
        print("正在创建数据库表结构...")
        
        # 创建表
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS stock_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_code TEXT NOT NULL,
            trade_date DATE NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            vol INTEGER,
            amount REAL,
            name TEXT NOT NULL
        );
        """
        conn.execute(create_table_sql)
        
        # 清空现有数据（可选）
        # conn.execute("DELETE FROM stock_data;")
        
        print("正在导入数据到SQLite数据库...")
        
        # 将DataFrame数据插入到数据库中
        # 使用pandas的to_sql方法直接将DataFrame导入SQLite
        df.to_sql('stock_data', conn, if_exists='replace', index=False)
        
        # 提交事务
        conn.commit()
        
        print(f"成功导入 {len(df)} 条记录到 {db_file} 数据库中")
        
        # 验证数据导入情况
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM stock_data;")
        count = cursor.fetchone()[0]
        print(f"数据库中实际记录数: {count}")
        
        # 显示前几条记录作为验证
        cursor.execute("SELECT * FROM stock_data LIMIT 5;")
        records = cursor.fetchall()
        print("\n前5条记录预览:")
        column_names = [description[0] for description in cursor.description]
        print(column_names)
        for record in records:
            print(record)
            
        return True
        
    except Exception as e:
        print(f"导入数据时出错: {str(e)}")
        return False
    finally:
        # 关闭数据库连接
        conn.close()

def main():
    # 确保目录存在
    os.makedirs('stock_xlsx', exist_ok=True)
    os.makedirs('db_data', exist_ok=True)
    
    # Excel文件路径
    excel_file = os.path.join('stock_xlsx', # 这里填写生成的对应xlsx文件)
    
    # SQLite数据库文件路径
    db_file = os.path.join('db_data', 'stock_data.db')
    
    print(f"开始将 {excel_file} 数据导入到 {db_file}")
    
    # 执行导入操作
    success = excel_to_sqlite(excel_file, db_file)
    
    if success:
        print(f"数据导入完成! 数据库文件: {os.path.abspath(db_file)}")
    else:
        print("数据导入失败!")

if __name__ == "__main__":
    main()