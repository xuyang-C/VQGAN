import os
import math
import random
from reedsolo import RSCodec, ReedSolomonError


class GELossChannel:
    def __init__(self, p_gb=0.068, p_bg=0.852, p_loss_good=0.04, p_loss_bad=0.5):
        """
        初始化GE丢包信道模型

        参数:
        p_gb: 从好状态转移到坏状态的概率
        p_bg: 从坏状态转移到好状态的概率
        p_loss_good: 好状态下的丢包率
        p_loss_bad: 坏状态下的丢包率
        """
        self.p_gb = p_gb
        self.p_bg = p_bg
        self.p_loss_good = p_loss_good
        self.p_loss_bad = p_loss_bad
        self.current_state = 'good'  # 初始状态设为好状态

    def simulate_packet(self):
        """
        模拟单个数据包的传输

        返回:
        lost: 布尔值，表示数据包是否丢失
        """
        # 根据当前状态确定丢包概率
        if self.current_state == 'good':
            loss_prob = self.p_loss_good
        else:
            loss_prob = self.p_loss_bad

        # 决定数据包是否丢失
        lost = random.random() < loss_prob

        # 状态转移
        if self.current_state == 'good':
            if random.random() < self.p_gb:
                self.current_state = 'bad'
        else:
            if random.random() < self.p_bg:
                self.current_state = 'good'

        return lost
def split_mp4_to_packets(input_file, packet_size=1024):
    packets = []
    with open(input_file, 'rb') as f:
        while True:
            chunk = f.read(packet_size)
            if not chunk:
                break
            # 尾包不足 packet_size 时补零，保证等长
            if len(chunk) < packet_size:
                chunk += bytes(packet_size - len(chunk))
            packets.append(bytearray(chunk))
    return packets, os.path.getsize(input_file)

def join_packets_to_file(packets, output_file, original_filesize):
    # 拼接并截断到原文件大小，避免因尾包补零而多写
    written = 0
    with open(output_file, 'wb') as f:
        for p in packets:
            need = min(len(p), original_filesize - written)
            if need <= 0:
                break
            f.write(p[:need])
            written += need

def make_blocks(packets, k):
    """把数据包按 k 个为一组分块；最后一组不满则用全零包补齐"""
    blocks = []
    packet_size = len(packets[0]) if packets else 0
    zero_packet = bytearray([0]*packet_size)
    for i in range(0, len(packets), k):
        group = packets[i:i+k]
        if len(group) < k:
            group = group + [bytearray(zero_packet) for _ in range(k - len(group))]
        blocks.append(group)
    return blocks

def rs_encode_block_across_packets(data_block, r):
    """
    输入：data_block = [k 个 bytearray 包]，每个长度相同
    输出：parity_block = [r 个冗余 bytearray 包]
    """
    k = len(data_block)
    packet_size = len(data_block[0])
    rs = RSCodec(r)  # 码率 k/(k+r)
    # 初始化 r 个冗余包（与数据包等长）
    parity_block = [bytearray(packet_size) for _ in range(r)]

    # 按“列”编码：对每个字节偏移做一次 RS(k->k+r)
    for offset in range(packet_size):
        msg = bytes(p[offset] for p in data_block)  # k 字节
        codeword = rs.encode(msg)                   # 长度 k+r
        # codeword 前 k 字节是原始，后 r 字节是冗余
        parity_bytes = codeword[k:]
        for j in range(r):
            parity_block[j][offset] = parity_bytes[j]
    return parity_block

def rs_decode_block_across_packets(received_block, k, r):
    """
    输入：长度 n=k+r 的列表，其中缺失的包为 None；返回还原后的前 k 个数据包列表
    """
    n = k + r
    packet_size = len(next(p for p in received_block if p is not None))
    rs = RSCodec(r)
    # 准备输出：k 个数据包
    recovered_data = [bytearray(packet_size) for _ in range(k)]

    # 找到擦除位置
    erase_pos = [i for i, p in enumerate(received_block) if p is None]
    if len(erase_pos) > r:
        raise ReedSolomonError(f"该组丢失 {len(erase_pos)} 个包，超过冗余 {r}，无法完全恢复")

    # 按列解码
    for offset in range(packet_size):
        # 构造长度 n 的码字（缺失处填 0），并记录擦除位置
        codeword = bytearray(n)
        for i in range(n):
            if received_block[i] is None:
                codeword[i] = 0
            else:
                codeword[i] = received_block[i][offset]
        # 解码（告知擦除位置）
        decoded = rs.decode(bytes(codeword), erase_pos=erase_pos, only_erasures=True)[0]  # 返回原始 k 字节
        # 写回前 k 个数据包
        for i in range(k):
            recovered_data[i][offset] = decoded[i]
    return recovered_data

def simulate_packet_loss_ge(packets, ge_channel):
    """使用GE模型模拟丢包"""
    out = []
    for p in packets:
        if not ge_channel.simulate_packet():
            out.append(p)
        else:
            out.append(None)  # 用None表示整包丢失
    return out

def encode_file_to_blocks(input_mp4, packet_size=1024, k=10, r=4):
    data_packets, filesize = split_mp4_to_packets(input_mp4, packet_size)
    original_data_packet_count = len(data_packets)  # 统计：原始数据包数量（未包含k对齐补到整组的零包）
    blocks = make_blocks(data_packets, k)
    encoded_stream = []
    for block in blocks:
        parity = rs_encode_block_across_packets(block, r)
        group = block + parity
        encoded_stream.append(group)
    return encoded_stream, filesize, original_data_packet_count, len(blocks)

# def lose_and_decode_blocks(encoded_stream, k=10, r=4, loss_rate=0.2):
#     """
#     返回：recovered_packets, stats
#     stats = {
#         'total_groups': ..., 'k': ..., 'r': ...,
#         'total_encoded_packets': ...,     # 发送的总包数（含冗余）
#         'lost_packets': ...,              # 丢失的总包数（含冗余）
#         'loss_rate': ...,                 # 丢包率 = lost / total_encoded
#         'recoverable_groups': ...,        # 可完全恢复的数据组数（每组丢失 <= r）
#         'unrecoverable_groups': ...,      # 不可完全恢复的数据组数（每组丢失 > r）
#         'recovered_data_packets': ...,    # 成功恢复的数据包数量（仅统计前k）
#         'original_data_packets': ...,     # 原始数据包数量（用于百分比）
#         'recovery_rate': ...              # 恢复率 = recovered / original_data_packets
#     }
#     """
#     total_groups = len(encoded_stream)
#     n = k + r
#
#     total_encoded_packets = total_groups * n
#     lost_packets = 0
#     recoverable_groups = 0
#     unrecoverable_groups = 0
#     recovered_data_packets = 0  # 仅统计前k个数据包的成功恢复数量
#
#     recovered_packets = []
#     for group in encoded_stream:
#         # 模拟丢包
#         lost_group = simulate_packet_loss(group, loss_rate=loss_rate)
#         # 统计该组丢包数（含冗余）
#         group_lost = sum(1 for p in lost_group if p is None)
#         lost_packets += group_lost
#
#         # 判断该组是否可完全恢复
#         if group_lost <= r:
#             recoverable_groups += 1
#             rec = rs_decode_block_across_packets(lost_group, k, r)  # 能完整还原前k个
#             recovered_data_packets += k
#         else:
#             unrecoverable_groups += 1
#             # 恢复失败：只能保留未丢失的前k数据包，其余用零填充（也可选择丢弃）
#             packet_size = len(next(p for p in group if p is not None))
#             zero = bytearray([0]*packet_size)
#             rec = []
#             for i in range(k):
#                 rec.append(lost_group[i] if (i < len(lost_group) and lost_group[i] is not None) else zero)
#             # 统计该组前k中实际没丢的数量
#             recovered_data_packets += sum(1 for i in range(k) if i < len(lost_group) and lost_group[i] is not None)
#
#         recovered_packets.extend(rec)
#
#     stats = {
#         'total_groups': total_groups,
#         'k': k,
#         'r': r,
#         'total_encoded_packets': total_encoded_packets,
#         'lost_packets': lost_packets,
#         'loss_rate': (lost_packets / total_encoded_packets) if total_encoded_packets else 0.0,
#         # 下面两个字段先占位（original_data_packets 在 main 里填充后再更新 recovery_rate）
#         'recoverable_groups': recoverable_groups,
#         'unrecoverable_groups': unrecoverable_groups,
#         'recovered_data_packets': recovered_data_packets,
#         'original_data_packets': None,
#         'recovery_rate': None,
#     }
#     return recovered_packets, stats
def lose_and_decode_blocks_ge(encoded_stream, k=10, r=4, ge_channel=None):
    """
    返回：recovered_packets, stats
    stats = {
        'total_groups': ..., 'k': ..., 'r': ...,
        'total_encoded_packets': ...,     # 发送的总包数（含冗余）
        'lost_packets': ...,              # 丢失的总包数（含冗余）
        'loss_rate': ...,                 # 丢包率 = lost / total_encoded
        'recoverable_groups': ...,        # 可完全恢复的数据组数（每组丢失 <= r）
        'unrecoverable_groups': ...,      # 不可完全恢复的数据组数（每组丢失 > r）
        'recovered_data_packets': ...,    # 成功恢复的数据包数量（仅统计前k）
        'original_data_packets': ...,     # 原始数据包数量（用于百分比）
        'recovery_rate': ...,             # 恢复率 = recovered / original_data_packets
        'unrecoverable_groups_info': ...  # 记录每个不可恢复组丢失包数
    }
    """
    total_groups = len(encoded_stream)
    n = k + r

    total_encoded_packets = total_groups * n
    lost_packets = 0
    recoverable_groups = 0
    unrecoverable_groups = 0
    recovered_data_packets = 0  # 仅统计前k个数据包的成功恢复数量
    unrecoverable_groups_info = []  # 记录每个不可恢复组的丢失包数

    recovered_packets = []
    for group in encoded_stream:
        # 模拟丢包
        lost_group = simulate_packet_loss_ge(group, ge_channel)
        # 统计该组丢包数（含冗余）
        group_lost = sum(1 for p in lost_group if p is None)
        lost_packets += group_lost

        # 判断该组是否可完全恢复
        if group_lost <= r:
            recoverable_groups += 1
            rec = rs_decode_block_across_packets(lost_group, k, r)  # 能完整还原前k个
            recovered_data_packets += k
        else:
            unrecoverable_groups += 1
            # 记录该组丢失的包数
            unrecoverable_groups_info.append(group_lost)
            # 恢复失败：只能保留未丢失的前k数据包，其余用零填充（也可选择丢弃该组，按需求改）
            packet_size = len(next(p for p in group if p is not None))
            zero = bytearray([0]*packet_size)
            rec = []
            for i in range(k):
                rec.append(lost_group[i] if (i < len(lost_group) and lost_group[i] is not None) else zero)
            # 统计该组前k中实际没丢的数量
            recovered_data_packets += sum(1 for i in range(k) if i < len(lost_group) and lost_group[i] is not None)

        recovered_packets.extend(rec)

    stats = {
        'total_groups': total_groups,
        'k': k,
        'r': r,
        'total_encoded_packets': total_encoded_packets,
        'lost_packets': lost_packets,
        'loss_rate': (lost_packets / total_encoded_packets) if total_encoded_packets else 0.0,
        # 下面两个字段先占位（original_data_packets 在 main 里填充后再更新 recovery_rate）
        'recoverable_groups': recoverable_groups,
        'unrecoverable_groups': unrecoverable_groups,
        'recovered_data_packets': recovered_data_packets,
        'original_data_packets': None,
        'recovery_rate': None,
        'unrecoverable_groups_info': unrecoverable_groups_info  # 新增：不可恢复组丢失包数
    }
    return recovered_packets, stats


def main():
    input_mp4 = "/home/xuyang/taming-transformers-master/data/UVG/UVG_h265_1920x1080/1692kb/Beauty_265.mp4"
    output_dir = "/home/xuyang/taming-transformers-master/data/UVG/UVG_h265_1920x1080/FEC100%_GE75%_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # 冗余包数r，信息包数k
    # 最大纠错能力r//2, 最大恢复能力r（恢复丢失的包）

    packet_size = 1460 # MTU 1500 Byte MTU - IP头 - TCP头 = 1500 - 20 - 20 = 1460字节
    k, r = 10, 10
    num_runs = 10  # 模拟次数
    # 编码
    encoded_stream, orig_size, original_data_packet_count, total_groups = encode_file_to_blocks(
        input_mp4, packet_size, k, r
    )

    print(f"原始包数量: {original_data_packet_count}")
    print(f"总分组: {total_groups}，每组 n=k+r={k}+{r}={k + r}")

    for run in range(1, num_runs + 1):
        # 创建GE信道模型（使用默认参数或自定义）
        ge_channel = GELossChannel(
            p_gb=0.92,
            p_bg=0.5,
            p_loss_good=0.04,
            p_loss_bad=0.75  # 丢包级别
        )



        # 丢包 + 解码（返回统计）
        recovered_packets, stats = lose_and_decode_blocks_ge(encoded_stream, k, r, ge_channel)

        # 用原始数据包数量完善统计并打印
        stats['original_data_packets'] = original_data_packet_count
        stats['recovery_rate'] = stats['recovered_data_packets'] / original_data_packet_count if original_data_packet_count else 0.0
        print(f"\nGE信道参数:")
        print(f"好→坏转移概率: {ge_channel.p_gb}, 坏→好转移概率: {ge_channel.p_bg}")
        print(f"好状态丢包率: {ge_channel.p_loss_good}, 坏状态丢包率: {ge_channel.p_loss_bad}")
        print(f"\n统计结果:")
        print(f"发送总包数(含冗余): {stats['total_encoded_packets']}")
        print(f"丢失包数量(含冗余): {stats['lost_packets']}，丢包率: {stats['loss_rate']:.2%}")
        print(f"成功恢复的数据包数量: {stats['recovered_data_packets']}，恢复率: {stats['recovery_rate']:.2%}")
        print(f"可完全恢复的组: {stats['recoverable_groups']}，不可完全恢复的组: {stats['unrecoverable_groups']}")
        # 打印不可恢复组的丢失包数
        print("不可恢复的组丢失包数:")
        for idx, lost in enumerate(stats['unrecoverable_groups_info']):
            print(f"组 {idx + 1}: 丢失包数 = {lost}")
        # 输出文件路径
        output_mp4 = os.path.join(output_dir, f"Beauty_265_FEC10_GE_run{run}.mp4")
        join_packets_to_file(recovered_packets, output_mp4, original_filesize=orig_size)
        print(f"输出已写入: {output_mp4}")


if __name__ == "__main__":
    main()
