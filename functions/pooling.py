

def maximum(depth_slice):
    value = depth_slice[0][0]
    index = (0, 0)
    for i in range(1, depth_slice.shape[0]):
        for j in range(depth_slice.shape[1]):
            if depth_slice[i][j] > value:
                value = depth_slice[i][j]
                index = (i, j)
    return value, index

