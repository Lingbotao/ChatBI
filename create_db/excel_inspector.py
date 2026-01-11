import pandas as pd
import os

def inspect_excel_file():
    """检查Excel文件结构"""
    excel_file = # 这里填写生成的对应xlsx文件
    
    if not os.path.exists(excel_file):
        print(f"文件 {excel_file} 不存在")
        return
    
    try:
        # 先尝试读取前几行数据
        df = pd.read_excel(excel_file, nrows=5)
        
        print("Excel文件前5行数据:")
        print(f"列名: {df.columns.tolist()}")
        print(df.head())
        print("\n数据类型:")
        print(df.dtypes)
        
        # 获取完整的行数（通过读取表头来获取列数，然后使用openpyxl获取实际行数）
        try:
            # 使用pandas获取所有行的计数（但不加载数据）
            xl = pd.ExcelFile(excel_file)
            total_rows = pd.read_excel(xl, nrows=0).shape[1]  # 获取列数
            # 重新打开文件来获取行数
            with pd.ExcelFile(excel_file) as xls:
                total_rows = xls.book.active.max_row - 1  # 减去表头行
            print(f"\n总行数: 预览5行，总计约{total_rows}行")
        except:
            # 如果无法准确获取总行数，则使用替代方法
            print(f"\n总行数: 预览5行")
        
    except Exception as e:
        print(f"读取Excel文件时出错: {str(e)}")

if __name__ == "__main__":
    inspect_excel_file()