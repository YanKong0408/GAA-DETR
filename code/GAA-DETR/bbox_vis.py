import json
import cv2
import os

# 设置 JSON 文件的路径和输出文件夹
json_file_path = 'C:\\Users\\kongyan\\Desktop\\dataset\\Fungus\\coco\\data_1\\ky_refined\\annotations\\instances_test2017.json'
output_folder = 'output_images'

# 创建输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 读取 JSON 文件
with open(json_file_path, 'r') as f:
    coco_data = json.load(f)

# 创建一个类别字典以便于识别类别名称
categories = {category['id']: category['name'] for category in coco_data['categories']}

# 遍历每个图像
for image in coco_data['images']:
    image_id = image['id']
    image_file_name = image['file_name']
    print(image_id)
    # 读取图像
    image_path = os.path.join('path/to/images', image_file_name)  # 修改为实际图像文件夹路径
    img = cv2.imread(image_path)
    
    # 查找该图像的所有标注
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]

    # 遍历每个标注并绘制边框
    for ann in annotations:
        bbox = ann['bbox']  # [x, y, width, height]
        x, y, width, height = map(int, bbox)
        
        # 绘制矩形框
        cv2.rectangle(img, (x, y), (x + width, y + height), (255, 0, 0), 2)
        
        # 添加类别标签
        category_name = categories[ann['category_id']]
        cv2.putText(img, category_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 保存带有标注的图像
    # output_path = os.path.join(output_folder, image_file_name)
    # cv2.imwrite(output_path, img)

    # print(f'Saved annotated image: {output_path}')