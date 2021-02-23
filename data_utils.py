import os
import numpy as np
from matplotlib import pyplot as plt

# ln: last name
# fn: first name

# data_dir = "./data/Traditional_Chinese_data/"
data_dir = "./data/mini_train/"
dict_char2index = {}
dict_index2char = {}
chars = os.listdir(data_dir)
char_count = len(chars)

for i, char in enumerate(chars):
    dict_char2index[char] = i
    dict_index2char[i] = char

def onehot_index(index):
    onehot = np.zeros((char_count,), dtype=np.int)
    onehot[index] = 1
    return onehot

def onehot_char(char):
    dict_index = char2index[char]
    return onehot_index(index)

top100_lns = [
    '陳', '林', '黃', '張', '李', '王', '吳', '劉', '蔡', '楊',  
    '許', '鄭', '謝', '郭', '洪', '曾', '邱', '廖', '賴', '周',  
    '徐', '蘇', '葉', '莊', '呂', '江', '何', '蕭', '羅', '高',  
    '簡', '朱', '鍾', '施', '游', '詹', '沈', '彭', '胡', '余',  
    '盧', '潘', '顏', '梁', '趙', '柯', '翁', '魏', '方', '孫',  
    '張簡', '戴', '范', '歐陽', '宋', '鄧', '杜', '侯', '曹', '薛',  
    '傅', '丁', '溫', '紀', '范姜', '蔣', '歐', '藍', '連', '唐',  
    '馬', '董', '石', '卓', '程', '姚', '康', '馮', '古', '姜',  
    '湯', '汪', '白', '田', '涂', '鄒', '巫', '尤', '鐘', '龔',  
    '嚴', '韓', '黎', '阮', '袁', '童', '陸', '金', '錢', '邵']

lns = [ln for ln in top100_lns if dict_char2index.get(ln)]