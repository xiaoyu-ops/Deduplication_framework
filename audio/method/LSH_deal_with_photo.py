import os
import numpy as np
import hashlib
from tqdm import tqdm

#生成签名矩阵
def generate_minhash_signatures(matrix, num_hashes):
    num_rows, num_cols = matrix.shape
    signature_matrix = np.full((num_hashes, num_cols), np.inf)
    
    for i in range(num_hashes):
        # 生成行的随机排列（这是哈希函数的体现）
        permutation = np.random.permutation(num_rows)
        
        # 对每列找到排列后第一个1的位置
        for col in range(num_cols):
            for row_idx in permutation:
                if matrix[row_idx, col] == 1:
                    signature_matrix[i, col] = row_idx
                    break
    
    return signature_matrix

def caculate_S(b,r):
    """
    计算相似度所需的band数量和每个band的行数
    :param b: 分区(band)的数量
    :param r: 每个分区的行数
    """
    test = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for s in test:
        if s <= 0 or s >= 1:
            raise ValueError("相似度s必须在(0, 1)之间")
        P = 1- (1 - s ** r) ** b
        if P <= 0 or P >= 1:
            raise ValueError("计算出的概率P必须在(0, 1)之间")
        if P > 0.8 and s <= 0.5:
            print(f"此时P为{P}")
            print(f"当相似度s={s}时,分区数量b={b}和每个分区的行数r={r}满足P > 0.9,此时是不合理的，因为在低阈值时被分到同一个桶的概率过高")
            return print("请调整b和r的值以满足合理的概率范围")
        if P < 0.8 and s <= 0.5:
            print(f"此时P为{P}")
            print(f"当相似度s={s}时,分区数量b={b}和每个分区的行数r={r}满足P > 0.9,此时是合理的")     
        elif P > 0.5 and s <= 0.7:
            print(f"此时P为{P}")
            print(f"当相似度s={s}时,分区数量b={b}和每个分区的行数r={r}满足P > 0.5,此时是合理的")
        elif P > 0.85 and s <= 0.9 and s > 0.7:
            print(f"此时P为{P}")
            print(f"当相似度s={s}时,分区数量b={b}和每个分区的行数r={r}满足P > 0.85,此时是合理的")
    return print("这个b,r组合是合理的")


# 经过上面的计算，我们可以知道签名矩阵的行数应该是200，分区数量b=20，每个分区的行数r=10
# 但是我们现在测试的矩阵是比较小的，所以我们可以将b和r设置为5和2来进行测试
def minHash(input_matrix, b, r):
    """
    将相似向量映射到同一个哈希桶中
    :param input_matrix: 输入矩阵
    :param b: 分区(band)的数量
    :param r: 每个分区的行数
    :return 哈希桶: 一个字典，键是哈希值，值是列号
    """

    hashBuckets = {}

    # 对矩阵进行n次排列
    n = b * r

    # 生成签名矩阵
    sigMatrix = generate_minhash_signatures(input_matrix, n)

    # 分区行的起始和结束位置
    begin, end = 0, r

    # 计算分区级别数量
    count = 0

    while end <= sigMatrix.shape[0]:#此处sigMatrix.shape[0]代表的是总行数

        count += 1

        # 遍历签名矩阵的列
        for colNum in tqdm(range(sigMatrix.shape[1]),desc="处理列"): # colNum代表列号

            # 生成哈希对象，我们使用md5
            hashObj = hashlib.md5()

            # 计算哈希值
            band = str(sigMatrix[begin: begin + r, colNum]) + str(count)
            hashObj.update(band.encode())

            # 使用哈希值作为桶标签
            tag = hashObj.hexdigest()

            # 更新字典
            if tag not in hashBuckets:
                hashBuckets[tag] = [colNum]
            elif colNum not in hashBuckets[tag]:
                hashBuckets[tag].append(colNum)
        begin += r
        end += r

    # 返回一个字典
    return hashBuckets


# 现在我们可以使用哈希桶来查找相似的向量
# 多次在同一个哈希桶的我们可以认为是相似的

def count_bucket_collisions(hash_buckets):
    """统计每对向量共享同一个哈希桶的次数"""
    collision_counts = {}
    
    for bucket, items in tqdm(hash_buckets.items(),desc="统计在一个桶"):#bucket接受的是键，items接受的是值
        if len(items) > 1:
            for i in range(len(items)):
                for j in range(i+1, len(items)):
                    pair = tuple(sorted([items[i], items[j]]))
                    collision_counts[pair] = collision_counts.get(pair, 0) + 1
    
    return collision_counts

def verify_similarity(pair, original_matrix, min_collisions=2, similarity_threshold=0.6):
    """计算并验证两个向量的实际相似度"""
    vec1 = original_matrix[:, pair[0]]
    vec2 = original_matrix[:, pair[1]]
    
    # 计算Jaccard相似度
    intersection = np.sum(np.logical_and(vec1, vec2))
    union = np.sum(np.logical_or(vec1, vec2))
    
    similarity = intersection / union if union > 0 else 0
    return similarity >= similarity_threshold, similarity

def find_similar_items(hash_buckets, matrix, collision_threshold=2, similarity_threshold=0.75):
    """找出真正相似的项目"""
    # 第一阶段：找出候选对
    collisions = count_bucket_collisions(hash_buckets)
    candidate_pairs = [pair for pair, count in collisions.items() if count >= collision_threshold]
    
    # 第二阶段：验证相似度
    similar_pairs = []
    for pair in tqdm(candidate_pairs, desc="验证相似度"):
        is_similar, score = verify_similarity(pair, matrix, similarity_threshold=similarity_threshold)
        if is_similar:
            similar_pairs.append((pair, score))
    
    # 按相似度排序
    return sorted(similar_pairs, key=lambda x: x[1], reverse=True)

    # 保存相似结果到文件
def save_similar_pairs_to_file(similar_pairs, filename="similar_pairs.txt"):
        """将相似对保存到文件"""
        with open(filename, 'w') as f:
            for pair, score in similar_pairs:
                f.write(f"Pair: {pair}, Similarity: {score}\n")
        print(f"相似对已保存到 {filename}")



if __name__ == "__main__":

    array = open("binary_array_dict.npy", "rb")
    binary_array_dict = np.load(array, allow_pickle=True).item()
    matrix_true = np.array(list(binary_array_dict.values())).T
    print(f"matrix_true: {matrix_true}")

    #学习LSH算法
    dataset = [ [1,1,0,0,0,1,1],[0,0,1,1,1,0,0],[1,0,0,0,0,1,1]]
    query = [0,1,1,1,1,0,0]
    dataset.append(query)
    matrix = np.array(dataset).T

    signature_matrix = generate_minhash_signatures(matrix, 5)
    print("签名矩阵:\n", signature_matrix)

    caculate_S(20,10)

    hashBuckets = minHash(matrix, 5, 2)
    print("哈希桶:\n", hashBuckets)

    print(f"最后得到的相似对结果{find_similar_items(hashBuckets, matrix)}")
    # 以上是基础测试，接下来进行真的数据测试
    hashBuckets_true = minHash(matrix_true, 20, 10)
    print("真实数据hash桶:\n", hashBuckets_true)
    print(f"真实数据最后得到的相似对结果{find_similar_items(hashBuckets_true, matrix_true)}")

    save_similar_pairs_to_file(find_similar_items(hashBuckets_true, matrix_true), "similar_pairs.txt")
    #这个代码目前还有错误，我们只需要知道要去重的就行。
