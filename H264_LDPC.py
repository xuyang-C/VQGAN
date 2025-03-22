import subprocess
import os
import sys
import time
from tqdm import tqdm
from PIL import Image
import numpy as np
import pyldpc
import pyldpc.ldpc_images as ldpc_images
import pyldpc.utils_img as utils_img
from pyldpc import make_ldpc, encode, decode

# 读取图像并将其转换为二进制数据
def load_image(image_path):
    img = Image.open(image_path).convert('RGB')  # 转为RGB
    img_data = np.array(img)  # 转换为NumPy数组
    binary_data = utils_img.rgb2bin(img_data)  # 将像素值转换为二进制序列
    return binary_data, binary_data.shape


# 将解码后的二进制数据转换回图像
def binary_to_image(binary_data, original_shape, output_path):
    img = Image.fromarray(binary_data.reshape(original_shape))  # 重构为原始形状
    img.save(output_path)
    print(f"图像保存为 {output_path}")


def int_to_bit_array(arr, width):
    return np.array([list(np.binary_repr(x, width=width)) for x in arr], dtype=np.uint8).flatten()


def bits_to_int_array(bits, bit_width=13):
    num_elements = len(bits) // bit_width
    return np.array([int("".join(bits[i * bit_width:(i + 1) * bit_width].astype(str)), 2) for i in range(num_elements)],
                    dtype=np.int16)


'''
在 LDPC 码中，比特节点和校验节点的度数之间的关系满足以下公式：
n*d_v = m*d_c
其中 n 是码字长度，m 是校验节点的数量，d_v 是比特节点的度数，d_c 是校验节点的度数。
d_v通常取2到3，d_c通常取6到9
(2048,6144)表示信息位k=2048，码字长度n=6144，校验节点m=4096，码率R=k/n=1/3
又d_v/d_c=m/n=(n-k)/n=1-k/n, 即d_v/d_c=1-R, R=1-d_v/d_c
'''


def LDPC_img(image_path, output_path, snr):
    # 加载图像数据
    print(f"Processing file: {image_path}-> {output_path}")
    img_data_bin, img_data_shape = load_image(image_path)
    n = 18  # 码字长度,较大n导致更高编码增益和更低的误码率，但编解码复杂度增加
    d_v = 3  # 变量节点度数 增大意味与更多校验节点连接，提高纠错能力，解码复杂度增加
    d_c = 6  # 校验节点度数，通常大于d_v，
    H, G = pyldpc.make_ldpc(n, d_v, d_c, systematic=True, sparse=True)  # 生成 LDPC 矩阵
    print("信息位：{}, 码字长度：{}".format(G.shape[1], n))
    # 使用 LDPC 编码图像
    codeword, encoded_img = ldpc_images.encode_img(G, img_data_bin, snr=snr)
    # 使用 LDPC 纠错解码图像
    decoded_img = ldpc_images.decode_img(G, H, codeword, img_shape=img_data_shape, snr=snr,
                                         maxiter=50)  # 解码耗时比较长,迭代次数不宜过大
    print(f"解码后的图像数据形状: {decoded_img.shape}")
    Image.fromarray(decoded_img).save(output_path)  # 保存解码后的图像


def LDPC_UAV(size, codebook_size, snr):
    # Step 1: 生成随机数组，大小为30x30，值在 [0, 8191] 范围内
    # array_shape = (resolution, resolution)
    # 13 bits, 最大值为 8191
    original_array = np.random.randint(0, codebook_size, size=int(size))

    # 将数组展开成一维数组
    original_bits = original_array.flatten()
    # Step 2: 使用 pyldpc 进行编码
    # 生成 LDPC 码的矩阵
    n = 100  # 编码后的比特数 (码字长度)
    d_v = 2  # 列权重
    d_c = 4  # 行权重
    H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
    k = G.shape[1]  # 生成矩阵维度（信息位）
    print("信息位：{}, 码字长度：{}".format(G.shape[1], n))
    # 将原始数组转换为二进制
    # original_bits = int_to_bit_array(original_flat, width=7)
    encoded_blocks = []
    for i in range(0, len(original_bits), k):
        block = original_bits[i:i + k]
        if len(block) < k:
            block = np.pad(block, (0, k - len(block)), 'constant')
        encoded = encode(G, block, snr)
        encoded_blocks.append(encoded)
    decoded_bits = []
    for noisy in encoded_blocks:
        decoded = decode(H, noisy, snr=snr, maxiter=100)
        decoded_bits.extend(decoded[:k])

    # Step 4: 将解码后的比特流转换回 int，再转为原始数组
    # decoded_bits_uint7 = bits_to_int_array(decoded_bits, bit_width=7)

    # Step 5: 比较原始数组和解码后的数组
    # 统计原始数组和解码后的数组中不同元素的个数
    num_differences = np.sum(original_bits != np.array(decoded_bits[:len(original_bits)], dtype=np.uint8))
    differences_ratio = num_differences / len(original_bits)
    print(f"原始数组和解码后数组中不同的元素个数为: {num_differences}")
    print(f"不同元素的比例为: {differences_ratio:.2%}")


def LDPC_img_dir(directory):
    jpg_file = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".jpg"):
                jpg_file.append(os.path.join(root, file))

    for file_path in tqdm(jpg_file, desc="Processing images"):
        output_path = file_path.replace("gtres", "recon")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        LDPC_img(file_path, output_path, snr=3)


def ldpc_video_process(input_path: str, output_path: str, snr: float = 10):
    """
    LDPC视频编解码完整流程
    参数:
        input_path: 输入视频路径
        output_path: 输出视频路径
        snr: 模拟信道信噪比(控制错误率)
    """
    # 1. 二进制读取视频文件
    with open(input_path, 'rb') as f:
        video_data = f.read()
    print(f"原始视频大小: {len(video_data)} bytes")

    # 2. 转换为numpy数组进行LDPC处理
    data_bits = np.unpackbits(np.frombuffer(video_data, dtype=np.uint8))

    # 3. 创建LDPC编码器（参数需根据数据长度调整）
    n = 1000  # 码长
    d_v = 2  # 变量节点度
    d_c = 4  # 校验节点度
    H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)

    # 4. 分块编码（处理大数据量）
    block_size = k = G.shape[1]
    encoded_blocks = []
    for i in range(0, len(data_bits), k):
        block = data_bits[i:i + k]
        if len(block) < k:
            block = np.pad(block, (0, k - len(block)), 'constant')
        encoded = encode(G, block,snr=snr)
        encoded_blocks.append(encoded)

    # 5. 模拟信道噪声
    noisy_blocks = []
    for block in encoded_blocks:
        noise = np.random.randn(len(block)) * 10 ** (-snr / 20)
        noisy_blocks.append(block + noise)

    # 6. LDPC解码
    decoded_bits = []
    for noisy in encoded_blocks:
        decoded = decode(H, noisy, snr=snr, maxiter=100)
        decoded_bits.extend(decoded[:k])  # 提取系统位
    num_differences = np.sum(data_bits != np.array(decoded_bits[:len(data_bits)], dtype=np.uint8))
    differences_ratio = num_differences / len(data_bits)
    print(f"原始数组和解码后数组中不同的元素个数为: {num_differences}")
    print(f"不同元素的比例为: {differences_ratio:.4%}")
    # 7. 转换回字节数据
    decoded_bytes = np.packbits(decoded_bits[:len(data_bits)]).tobytes()

    # 8. 写入重建视频
    with open(output_path, 'wb') as f:
        f.write(decoded_bytes)
    print(f"重建视频大小: {os.path.getsize(output_path)} bytes")


def process_ldpc_videos(input_dir: str, output_dir: str, snr: float = 10):
    """
    批量处理目录下的视频文件，进行LDPC编解码并统计差异
    参数:
        input_dir: 输入视频目录
        output_dir: 输出目录
        snr: 信噪比(dB)
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 支持的视频格式
    video_exts = ['.mp4', '.avi', '.mov', '.mkv','.266']

    # 初始化统计变量
    total_diff = 0
    total_ratio = 0.0
    processed_files = 0

    # 遍历输入目录
    for filename in os.listdir(input_dir):
        # 检查视频格式
        if not any(filename.lower().endswith(ext) for ext in video_exts):
            continue

        input_path = os.path.join(input_dir, filename)
        base_name = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1]
        output_name = f"{base_name}{ext}"
        output_path = os.path.join(output_dir, output_name)

        # 处理单个文件
        file_diff, file_ratio = process_single_video(input_path, output_path, snr)

        # 累加统计量
        total_diff += file_diff
        total_ratio += file_ratio
        processed_files += 1

    # 计算平均值
    if processed_files > 0:
        avg_diff = total_diff / processed_files
        avg_ratio = total_ratio / processed_files
        print(f"\n处理完成！共处理 {processed_files} 个文件")
        print(f"平均差异数: {avg_diff:.2f}")
        print(f"平均差异率: {avg_ratio:.4%}")
    else:
        print("未找到可处理的视频文件")


def process_single_video(input_path: str, output_path: str, snr: float):
    """处理单个视频文件的核心逻辑"""
    # 1. 读取二进制数据
    with open(input_path, 'rb') as f:
        video_data = f.read()

    # 2. 转换为比特流
    data_bits = np.unpackbits(np.frombuffer(video_data, dtype=np.uint8))
    original_length = len(data_bits)

    # 3. 初始化LDPC编码器
    n = 1000  # 码长
    d_v, d_c = 2, 4  # 变量节点和校验节点度
    H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
    k = G.shape[1]  # 信息位长度

    # 4. 分块编码
    encoded_blocks = []
    for i in range(0, original_length, k):
        block = data_bits[i:i + k] if (i + k) <= original_length else np.pad(data_bits[i:],
                                                                             (0, k - (original_length - i)))
        encoded = encode(G, block,snr=snr)
        encoded_blocks.append(encoded)


    # 6. 解码处理
    decoded_bits = []
    for noisy in encoded_blocks:
        decoded = decode(H, noisy, snr=snr, maxiter=100)
        decoded_bits.extend(decoded[:k])

    # 7. 差异统计
    decoded_bits = np.array(decoded_bits[:original_length], dtype=np.uint8)
    num_diff = np.sum(data_bits != decoded_bits)
    diff_ratio = num_diff / original_length

    # 8. 重建视频
    decoded_bytes = np.packbits(decoded_bits).tobytes()
    with open(output_path, 'wb') as f:
        f.write(decoded_bytes)

    print(f"处理完成: {os.path.basename(input_path)} -> {os.path.basename(output_path)}")
    print(f"当前文件差异数: {num_diff}，差异率: {diff_ratio:.4%}\n")

    return num_diff, diff_ratio
begin_time = time.time()
# LDPC_UAV(74e3*8, 128, snr=4)

# LDPC_img("/home/xuyang/VQGAN-pytorch/data/UCF50/HorseRiding/gt/frame_0001.jpg",
#          "/home/xuyang/VQGAN-pytorch/data/UCF50/HorseRiding/recon/frame_0001.jpg", snr=3)

# LDPC_img_dir("/home/xuyang/VQGAN-pytorch/data/UCF50/HorseRiding/gtres/")
# ldpc_video_process("/home/xuyang/taming-transformers-master/data/HorseRiding/Horse_gt/74kb/v_HorseRiding_g25_c23.avi", "/home/xuyang/taming-transformers-master/data/HorseRiding/output_snr6.mp4", snr=6)
snr=0
print(f"Test with snr: {snr}")
process_ldpc_videos(
        input_dir="/home/xuyang/taming-transformers-master/data/HorseRiding/Horse_gt/h266_44kb_test/",
        output_dir=f"/home/xuyang/taming-transformers-master/data/HorseRiding/h266_44kb_test_snr{snr}",
        snr=snr)
end_time = time.time()
print(f"Time elapsed: {end_time - begin_time:.2f}s")
