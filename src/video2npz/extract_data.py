import json
import os

data_dict = {}
file_list = os.listdir("video")

for filename in file_list:
        data_dict[filename] = {}
        data_dict[filename]["beats"] = {}
        data_dict[filename]["beats"]["start"] = []
        data_dict[filename]["beats"]["type"] = []
        data_dict[filename]["beats"]["weight"] = []
        data_dict[filename]["beats"]["index"] = []
        data_dict[filename]["beats"]["unrolled_start"] = []
        data_dict[filename]["beats"]["is_active"] = []
        with open("beats/" + filename + ".txt", "r") as fin:
                lines = fin.readlines()
        for line in lines:
                if line.strip() != "":
                        kv = line.strip().split(":")
                        if kv[0] in ["start", "weight"]:
                                data_dict[filename]["beats"][kv[0]].append(float(kv[1]))
                        elif kv[0] == "is_active":
                                data_dict[filename]["beats"][kv[0]].append(int(kv[1]))
                        else:
                                data_dict[filename]["beats"][kv[0]].append(kv[1])
        with open("tempo/" + filename + ".txt", "r") as fin:
                data_dict[filename]["tempo"] = float(fin.readline().strip())
print(data_dict)
json.dump(data_dict, open("data.json", "w"))
