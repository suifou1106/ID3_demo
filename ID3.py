import pandas as pd
import numpy as np
from pprint import pprint

# 1. Tải dữ liệu (Thêm fallback tạo dữ liệu mẫu nếu không có file csv để test)
try:
    df = pd.read_csv('job_approval.csv')
    print("Đã tải dữ liệu từ 'play_tennis.csv' thành công!\n")
except FileNotFoundError:
    print("Không tìm thấy file 'play_tennis.csv'. Tự động sử dụng dữ liệu mẫu Play Tennis...\n")
    data = {
        'Day': ['D1','D2','D3','D4','D5','D6','D7','D8','D9','D10','D11','D12','D13','D14'],
        'Outlook': ['Sunny','Sunny','Overcast','Rain','Rain','Rain','Overcast','Sunny','Sunny','Rain','Sunny','Overcast','Overcast','Rain'],
        'Temperature': ['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild'],
        'Humidity': ['High','High','High','High','Normal','Normal','Normal','High','Normal','Normal','Normal','High','Normal','High'],
        'Wind': ['Weak','Strong','Weak','Weak','Weak','Strong','Strong','Weak','Weak','Weak','Strong','Strong','Weak','Strong'],
        'PlayTennis': ['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']
    }
    df = pd.DataFrame(data)
    # Loại bỏ cột Day (ID) để ID3 không bị overfit ngay lập tức trên tập mẫu này
    df = df.drop('Day', axis=1) 

# ================= CÁC HÀM CƠ BẢN VÀ ID3 =================

def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    probabilities = counts / counts.sum()
    ent = np.sum([-p * np.log2(p) for p in probabilities])
    return ent

def information_gain(data, split_attribute_name, target_name):
    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    
    weighted_entropy = np.sum([
        (counts[i] / np.sum(counts)) * entropy(data.where(data[split_attribute_name] == vals[i]).dropna()[target_name])
        for i in range(len(vals))
    ])
    
    info_gain = total_entropy - weighted_entropy
    return info_gain

def id3(data, original_data, features, target_attribute_name, parent_node_class=None):
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    elif len(data) == 0:
        return np.unique(original_data[target_attribute_name])[np.argmax(np.unique(original_data[target_attribute_name], return_counts=True)[1])]
    elif len(features) == 0:
        return parent_node_class
    else:
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
        
        item_values = [information_gain(data, feature, target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        
        tree = {best_feature: {}}
        features = [i for i in features if i != best_feature]
        
        for value in np.unique(data[best_feature]):
            sub_data = data.where(data[best_feature] == value).dropna()
            subtree = id3(sub_data, original_data, features, target_attribute_name, parent_node_class)
            tree[best_feature][value] = subtree
            
        return tree

# ================= CÁC HÀM CHO C4.5 =================

# Hàm tính Split Information (Thông tin phân chia)
def split_info(data, split_attribute_name):
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    probabilities = counts / counts.sum()
    s_info = np.sum([-p * np.log2(p) for p in probabilities])
    return s_info

# Hàm tính Gain Ratio (Tỷ lệ độ lợi)
def gain_ratio(data, split_attribute_name, target_name):
    info_gain = information_gain(data, split_attribute_name, target_name)
    s_info = split_info(data, split_attribute_name)
    
    # Tránh lỗi chia cho 0 nếu split_info = 0
    if s_info == 0:
        return 0
    return info_gain / s_info

# Giải thuật C4.5 đệ quy
def c45(data, original_data, features, target_attribute_name, parent_node_class=None):
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    elif len(data) == 0:
        return np.unique(original_data[target_attribute_name])[np.argmax(np.unique(original_data[target_attribute_name], return_counts=True)[1])]
    elif len(features) == 0:
        return parent_node_class
    else:
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
        
        # SỬ DỤNG GAIN RATIO THAY VÌ INFORMATION GAIN
        item_values = [gain_ratio(data, feature, target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        
        tree = {best_feature: {}}
        features = [i for i in features if i != best_feature]
        
        for value in np.unique(data[best_feature]):
            sub_data = data.where(data[best_feature] == value).dropna()
            subtree = c45(sub_data, original_data, features, target_attribute_name, parent_node_class)
            tree[best_feature][value] = subtree
            
        return tree

# ================= HÀM DỰ ĐOÁN (PREDICT) =================

def predict(query, tree, default="No"):
    for key in list(query.keys()):
        if key in list(tree.keys()):
            # Lấy giá trị của thuộc tính trong câu truy vấn
            try:
                result = tree[key][query[key]]
            except KeyError:
                # Nếu giá trị chưa từng xuất hiện trong tập huấn luyện, trả về default
                return default
            
            # Nếu kết quả là 1 dictionary (tức là còn nhánh), tiếp tục đệ quy
            if isinstance(result, dict):
                return predict(query, result, default)
            else:
                return result
    return default

def print_tree(tree, indent=""):
    if not isinstance(tree, dict):
        print(f"Result: {tree}")
    feature_name = list(tree.keys())[0]
    print(f"[{feature_name}]")

    branches = tree[feature_name]
    branch_keys = list(branches.keys())
    for i, branch_val in enumerate(branch_keys):
        is_last = (i == len(branch_keys) - 1)
        prefix = "└── " if is_last else "├── "
        next_indent = "    " if is_last else "│   "
        print(f"{indent}{prefix}{branch_val}", end="")
        subtree = branches[branch_val]
        if isinstance(subtree, dict):
            print(" -> ", end="")
            print_tree(subtree, indent + next_indent)
        else:
            print(f" -> Result: {subtree}")

# ================= CHẠY VÀ SO SÁNH =================

features = df.columns[:-1].tolist() 
target_col = df.columns[-1]

# 1. Tạo cây
tree_id3 = id3(df, df, features, target_attribute_name=target_col)
tree_c45 = c45(df, df, features, target_attribute_name=target_col)

print("🌲 Decision Tree's structures (ID3):")
print_tree(tree_id3)
print("\n" + "="*50 + "\n")

print("🌲 Decision Tree's structures (C4.5):")
print_tree(tree_c45)
print("\n" + "="*50 + "\n")

# 2. Dự đoán một ngày mới
# Giả sử hôm nay là một ngày: Trời mưa (Rain), Lạnh (Cool), Độ ẩm bình thường (Normal), Gió mạnh (Strong)
# new_day = {"Outlook": "Rain", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Strong"}
#
# print(f"🔍 PREDICT NEW DAY:\n{new_day}\n")
#
# pred_id3 = predict(new_day, tree_id3)
# pred_c45 = predict(new_day, tree_c45)
#
# print(f"🎯 Result from ID3  : {pred_id3}")
# print(f"🎯 Result from C4.5 : {pred_c45}")
