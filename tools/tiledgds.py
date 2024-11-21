import gdstk
import os
import random

# 定义输入和输出目录
input_directory = '/research/d5/gds/zxwang22/storage/resist/cells/gds'  # 输入文件夹路径
output_file = '/research/d5/gds/zxwang22/storage/resist/cells/tiled_output.gds'  # 输出文件路径

# 创建一个新的库
lib = gdstk.Library()

# 创建一个单独的 cell 用于合并所有图形
merged_cell = gdstk.Cell(name='merged_tiles')

# 定义每个 tile 的大小
tile_size = 4.0  # 每个 tile 占地 2.0 微米
# spacing = 1.0    # tile 之间的间隔

# 获取所有 GDS 文件
gds_files = [f for f in os.listdir(input_directory) if f.endswith('.gds')]
selected_files = sorted(gds_files)

# 随机选择 16384 个 GDS 文件
# selected_files = random.sample(gds_files, 16384)

# 确保选择的文件数量为 128*128
# assert len(selected_files) == 128 * 128, "选定的文件数量必须为 16384（128*128）"

# 初始化坐标
x_offset = 0
y_offset = 0

# 当前处理的 tile 计数
tile_count = 0

# 遍历选中的 GDS 文件
for filename in selected_files:
    file_path = os.path.join(input_directory, filename)
    # 读取 GDS 文件
    tmp_lib = gdstk.read_gds(file_path)

    # 检查库是否有效
    if tmp_lib is None:
        print(f"警告: 无法读取文件 {file_path}")
        continue

    # 遍历读取到的单元并添加到合并的 cell 中
    for cell in tmp_lib.cells:
        # 打印 cell 名称和多边形数量
        print(f"Processing cell: {cell.name}, Number of polygons: {len(cell.polygons)}")

        # 遍历单元中的所有图形并进行平移
        for polygon in cell.polygons:
            new_polygon = polygon.copy()
            new_polygon.translate(x_offset, y_offset)
            merged_cell.add(new_polygon)

        # 如果有路径等其他图形，类似地处理
        for path in cell.paths:
            new_path = path.copy()
            new_path.translate(x_offset, y_offset)
            merged_cell.add(new_path)

        # 更新 tile 计数
        tile_count += 1

        # 更新坐标，准备放置下一个 tile
        x_offset += tile_size

        # 如果 x_offset 超过边界，则换行
        if tile_count % 128 == 0:  # 每 128 个 tile 换行
            x_offset = 0
            y_offset += tile_size

# 将合并的 cell 添加到库中
lib.add(merged_cell)

# 将合并后的库写入新的 GDS 文件
lib.write_gds(output_file)

print(f"合并完成，输出文件为: {output_file}")