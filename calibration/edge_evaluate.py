import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial.distance import cdist

# 提取边缘点的函数
def extract_edge_points(binary_array):
    # 进行腐蚀操作，然后通过相减获得边缘
    eroded = binary_erosion(binary_array)
    edge = binary_array - eroded
    # 提取边缘点的坐标
    edge_points = np.column_stack(np.nonzero(edge))
    return edge_points

# 计算欧式距离矩阵的函数
def compute_euclidean_distance_matrix(points1, points2):
    # 计算两个点集之间的欧式距离矩阵
    distance_matrix = cdist(points1, points2, metric='euclidean')
    return distance_matrix

if __name__=="__main__":

    # 示例二值化数组
    array1 = np.array([[0, 0, 1, 1],
                    [0, 1, 1, 0],
                    [1, 1, 0, 0],
                    [1, 0, 0, 0]])

    array2 = np.array([[0, 1, 1, 0],
                    [1, 1, 0, 0],
                    [1, 0, 0, 1],
                    [0, 0, 1, 1]])

    # 提取边缘点
    points1 = extract_edge_points(array1)
    points2 = extract_edge_points(array2)

    print("Array 1 边缘点坐标:\n", points1)
    print("Array 2 边缘点坐标:\n", points2)

    # 计算欧式距离矩阵
    distance_matrix = compute_euclidean_distance_matrix(points1, points2)
    
    print("欧式距离矩阵:\n", distance_matrix)
    print(distance_matrix.shape)