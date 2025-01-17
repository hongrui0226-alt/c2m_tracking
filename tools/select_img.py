import os
import shutil

def select_every_other_image(input_dir, output_dir):
    """
    从输入目录选择隔一个图像到输出目录
    
    参数:
    - input_dir: 输入图像目录
    - output_dir: 输出图像目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图像文件
    images = sorted([f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    # 选择隔一个图像
    selected_images = images[::2]
    
    # 复制选中的图像
    for image in selected_images:
        src_path = os.path.join(input_dir, image)
        dst_path = os.path.join(output_dir, image)
        shutil.copy2(src_path, dst_path)
        print(f"复制: {image}")
    
    print(f"总共选择了 {len(selected_images)} 张图像")

# 使用示例
select_every_other_image('../data/test2', '../data/test2_selected')