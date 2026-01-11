import tushare as ts
import pandas as pd
from datetime import datetime
import os


def get_stock_data():
    """
    获取指定股票的历史价格数据，并保存到Excel文件
    """
    # 设置tushare的token，你需要替换为自己的token
    # 可以在tushare官网免费注册获取token
    token = input("请输入您的tushare token（可在tushare官网获取）：")
    ts.set_token(token)
    pro = ts.pro_api()

    # 定义要查询的股票列表
    # 股票代码：贵州茅台(600519.SH)，五粮液(000858.SZ)，国泰君安(601211.SH)，中芯国际(688981.SH)
    stocks = [
        {'code': '600519.SH', 'name': '贵州茅台'},
        {'code': '000858.SZ', 'name': '五粮液'},
        {'code': '601211.SH', 'name': '国泰君安'},
        {'code': '688981.SH', 'name': '中芯国际'}
    ]

    # 设定开始日期和结束日期
    start_date = '20200101'
    end_date = datetime.now().strftime('%Y%m%d')

    # 创建stock_xlsx文件夹（如果不存在）
    output_dir = 'stock_xlsx'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建Excel文件名
    excel_filename = os.path.join(output_dir, f'stock_history_{start_date}_to_{end_date}.xlsx')
    
    # 创建一个空DataFrame来存储所有股票的数据
    all_stock_data = pd.DataFrame()
    
    for stock in stocks:
        print(f"正在获取 {stock['name']} ({stock['code']}) 的数据...")
        
        try:
            # 获取股票历史交易数据
            df = pro.daily(ts_code=stock['code'], 
                          start_date=start_date, 
                          end_date=end_date,
                          fields='ts_code,trade_date,open,high,low,close,vol,amount')
            
            # 添加股票名称列
            df['name'] = stock['name']
            
            # 按交易日期排序（从旧到新）
            df = df.sort_values('trade_date', ascending=True)
            
            # 合并到总的数据框
            all_stock_data = pd.concat([all_stock_data, df], ignore_index=True)
            
            print(f"已获取 {stock['name']} 数据，共 {len(df)} 条记录")
            
        except Exception as e:
            print(f"获取 {stock['name']} 数据时出错: {str(e)}")

    # 将所有数据写入Excel文件的一个sheet
    if not all_stock_data.empty:
        # 按照trade_date从小到大排序
        all_stock_data = all_stock_data.sort_values('trade_date', ascending=True)
        
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            all_stock_data.to_excel(writer, sheet_name='All_Stocks_Data', index=False)
        
        print(f"\n所有数据已保存到 {excel_filename} 的 'All_Stocks_Data' 工作表中")
        print(f"总共 {len(all_stock_data)} 条记录")
    else:
        print("\n没有获取到任何数据，请检查网络连接和tushare token是否有效")


if __name__ == "__main__":
    get_stock_data()