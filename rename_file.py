import os

folder = "D:/Github Clone/odp/dataset/pic/New folder/5"   # 你的图片folder
start_number = 775

files = sorted(os.listdir(folder))

for i, filename in enumerate(files):
    
    old_path = os.path.join(folder, filename)

    new_name = f"{start_number + i}.jpg"
    new_path = os.path.join(folder, new_name)

    os.rename(old_path, new_path)

print("Renaming completed.")