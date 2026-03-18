import pandas as pd
import numpy as np
from pprint import pprint

try:
    df = pd.read_csv('play_tennis.csv')
    print("Đã tải dữ liệu thành công:\n", df.head(), "\n")
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file 'play_tennis.csv'. Vui lòng kiểm tra lại đường dẫn.")
    exit()

# 2. Hàm tính Entropy
def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    probabilities = counts / counts.sum()
    ent = np.sum([-p * np.log2(p) for p in probabilities])
    return ent

# 3. Hàm tính Information Gain
def information_gain(data, split_attribute_name, target_name="PlayTennis"):
    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    
    weighted_entropy = np.sum([
        (counts[i] / np.sum(counts)) * entropy(data.where(data[split_attribute_name] == vals[i]).dropna()[target_name])
        for i in range(len(vals))
    ])
    
    info_gain = total_entropy - weighted_entropy
    return info_gain

# 4. Giải thuật ID3 đệ quy
def id3(data, original_data, features, target_attribute_name="PlayTennis", parent_node_class=None):
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

# 5. Chạy thuật toán và in kết quả
# Tự động lấy tất cả các cột trừ cột cuối cùng làm đặc trưng (features)
features = df.columns[:-1].tolist() 
# Cột cuối cùng sẽ được dùng làm biến mục tiêu (target)
target_col = df.columns[-1]

tree = id3(df, df, features, target_attribute_name=target_col)

print("Cấu trúc Cây Quyết Định (ID3) được tạo ra:")
pprint(tree)