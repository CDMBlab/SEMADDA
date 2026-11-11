import scipy.io
import numpy as np


def display_mat_preview(mat_file_path, rows=10, cols=10):
    """
    显示MAT文件中每个数组的前N行M列
    """
    print(f"=== MAT文件预览: {mat_file_path} ===")

    data = scipy.io.loadmat(mat_file_path)

    for key in data.keys():
        if not key.startswith('__'):
            value = data[key]

            if isinstance(value, np.ndarray):
                print(f"\n变量: {key}")
                print(f"  类型: numpy数组")
                print(f"  形状: {value.shape}")
                print(f"  维度: {value.ndim}")

                if value.ndim == 1:
                    print(f"  内容 (前{min(rows, len(value))}个):")
                    preview = value[:min(rows, len(value))]
                    print(f"  {preview}")

                elif value.ndim == 2:
                    rows_to_show = min(rows, value.shape[0])
                    cols_to_show = min(cols, value.shape[1])
                    print(f"  内容 (前{rows_to_show}行, 前{cols_to_show}列):")

                    for i in range(rows_to_show):
                        row_preview = value[i, :cols_to_show]
                        print(f"  行{i + 1}: {row_preview}")

                else:
                    print(f"  内容 (展平后前{min(rows, value.size)}个):")
                    flat_preview = value.flatten()[:min(rows, value.size)]
                    print(f"  {flat_preview}")

            else:
                print(f"\n变量: {key}")
                print(f"  类型: {type(value).__name__}")
                print(f"  值: {value}")


# 使用示例
display_mat_preview("f-A_new.mat", rows=10, cols=10)