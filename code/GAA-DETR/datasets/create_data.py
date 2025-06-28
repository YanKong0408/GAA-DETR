import json


with open('C:\\Users\\kongyan\\Desktop\\dataset\\Fungus\\coco\\data_1\\ky\\instances_test2017.json', 'r') as file1:
    data1 = json.load(file1)

# with open('C:\\Users\\kongyan\\Desktop\\dataset\\Fungus\\coco\\data_1\\ky\\instances_val2017.json', 'r') as file2:
#     data2 = json.load(file2)

# # 更新第二个数据集的images和annotations的序号
# img_id_offset = 412
# ann_id_offset = 622

# print(len(data1['images']),data1['images'][0],data1['images'][-1])
# print(len(data1['annotations']),data1['annotations'][0],data1['annotations'][-1])

for img in data1['images']:
    img['file_name'] = img['file_name'].replace("D:\\Fungus\\self_bbox_gaze\\coco\\test2017","c:\\Users\\kongyan\\Desktop\\dataset\\Fungus\\all_raw\\data_1_cropped")

# for img in data2['images']:
#     img['id'] += img_id_offset + 1
#     img['file_name'] = img['file_name'].replace("D:\\Fungus\\self_bbox_gaze\\coco\\val2017","c:\\Users\\kongyan\\Desktop\\dataset\\Fungus\\all_raw\\data_1_cropped")
#     data1['images'].append(img)

# for ann in data2['annotations']:
#     ann['id'] += ann_id_offset
#     ann['image_id'] += img_id_offset + 1
#     data1['annotations'].append(ann)

# print(len(data1['images']),data1['images'][0],data1['images'][-1])
# print(len(data1['annotations']),data1['annotations'][0],data1['annotations'][-1])

with open('C:\\Users\\kongyan\\Desktop\\dataset\\Fungus\\coco\\data_1\\ky_refined\\instances_test2017.json', 'w') as f:
    json.dump(data1, f)