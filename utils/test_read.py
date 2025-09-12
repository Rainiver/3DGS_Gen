from plyfile import PlyData
import struct

ply_path = '/data/vde/zhongyuhe/workshop/mycode/3DGS_Gen/out/human/cano.ply'

vertex_count = 50000  # 从文件头信息获取
# 根据PLY文件定义的顶点格式创建一个struct的format字符串
vertex_format = '<fffBBB'  # 小端序，3个float代表x, y, z坐标，3个unsigned char代表颜色
bytes_per_vertex = struct.calcsize(vertex_format)

import re

header_terminator = b'end_header\n'

with open(ply_path, 'rb') as f:
    # 读取头部信息
    header = b''
    while True:
        chunk = f.read(1024)
        if not chunk:
            # 文件结束，但没有找到头部结束标记
            raise RuntimeError("PLY header not found")

        header += chunk
        pos = header.find(header_terminator)
        if pos >= 0:
            # 找到头部结束
            break

    # 要移动到实际数据的开始位置，需要添加头部终止符的长度
    data_start_pos = pos + len(header_terminator)
    f.seek(data_start_pos)
    # Now you can read the vertex/face data from the file.
    vertex_data = f.read(bytes_per_vertex)
    vertex = struct.unpack(vertex_format, vertex_data)
    print(vertex)


