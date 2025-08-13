import pandas as pd
import quality as qt

def load_data_KNN():
    data = qt.grade()
    split_res = data['resolution'].str.split('×', expand=True)  # expand=True返回DataFrame

    # 转换为整数（先转浮点数再转整数，处理小数点）
    data['width'] = split_res[0].astype(float).astype(int)   # 宽度列
    data['height'] = split_res[1].astype(float).astype(int)  # 高度列
    data = data.drop(columns=['resolution'])  # 删除原分辨率列
    data = data.apply(pd.to_numeric, errors='coerce')  # 转换为数值类型
    return data
