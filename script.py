import os
import shutil

def move_files(src_dir, dst_dir):
    if not os.path.exists(src_dir):
        print(f"El directorio fuente '{src_dir}' no existe.")
        return

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for file in os.listdir(src_dir):
        if file.endswith(".pdb") and "_" in file:
            src_path = os.path.join(src_dir, file)
            dst_path = os.path.join(dst_dir, file)
            shutil.move(src_path, dst_path)
            print(f"Movido: {file}")

move_files("input/pdb_data", "input/pdb_data_temp")