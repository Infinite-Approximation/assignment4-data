from collections import Counter, defaultdict
import json
import os
import re
import string
import mmh3
from typing import List
import unicodedata
from itertools import combinations
import shutil
from pathlib import Path
from tqdm import tqdm

def exact_line_deduplication(
    input_files: list[os.PathLike] = None, 
    output_directory: os.PathLike = None,
    input_json_file: os.PathLike = 'cs336_data/data/train.jsonl', # 这个参数存在的话 input_files 和 output_directory 就不需要了
):
    """
    对输入的文本进行精确去重。
    分为输入是一个json文件和输入是一个目录的两种情况。json文件的每一行都是一个网页内容，而目录下面每一个文件是一个网页内容。
    """
    # 第一遍扫描，获取每一行出现的次数。这里使用行对应hash来作为key，从而减少内存消耗
    line_count = Counter()
    if input_files is None: # 对json文件的text字段进行去重
        with open(input_json_file, mode="r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Counting line occurrences"):
                content = json.loads(line)["text"]
                content = content.rstrip("\n")  # 防止最后一行内容一样，但是没有 \n 导致hash和其他行不同
                line_count[hash(content)] += 1
    else: # 对目录下的文件进行去重
        for input_file in input_files:
            with open(file=input_file, mode="r", encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip(
                        "\n"
                    )  # 防止最后一行内容一样，但是没有 \n 导致hash和其他行不同
                    line_count[hash(line)] += 1

    # 第二遍扫码，只保留出现一次的行
    if input_files is None:
        output_json_file = Path(input_json_file).with_suffix(".exact_dedup.jsonl")
        with open(output_json_file, mode="w", encoding="utf-8") as wf:
            with open(input_json_file, mode="r", encoding="utf-8") as rf:
                for line in tqdm(rf, desc="Writing unique lines to output"):
                    content = json.loads(line)["text"]
                    content = content.rstrip("\n")
                    if line_count[hash(content)] == 1:
                        wf.write(json.dumps({"text": content}, ensure_ascii=False) + "\n")
    else:
        for input_file in input_files:
            basename = os.path.basename(input_file)
            output_file = os.path.join(output_directory, basename)
            with open(file=output_file, mode="w", encoding="utf-8") as wf:
                with open(file=input_file, mode="r", encoding="utf-8") as rf:
                    for line in rf:
                        if (
                            line_count[hash(line.rstrip("\n"))] == 1
                        ):  # 这里和前面对应，使用 rstripe('\n')
                            wf.write(line)


def preprocess_file_content(content: str) -> str:
    # applying NFD unicode normalization
    content = unicodedata.normalize("NFD", content)
    # removing accents
    content = "".join(c for c in content if unicodedata.category(c) != "Mn")
    # lowercasing
    content = content.lower()
    # removing punctuation
    translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
    content = content.translate(translator)
    # normalizing whitespaces
    content = " ".join(content.split())

    return content


def get_ngrams_words(text: list[str], ngrams: int) -> set[str]:
    text_len = len(text)
    i = 0
    ngrams_words = set()
    while i + ngrams - 1 < text_len:
        ngrams_words.add(" ".join(text[i : i + ngrams]))
        i += 1
    return ngrams_words


def minhash(S: set[str], seed: int):
    # 使用mm3.hash来计算每个n-grams word的hash值，这个hash函数是有seed决定的。
    return min(mmh3.hash(x, seed) for x in S)


def jaccard_similarity(a: set[str], b: set[str]):
    """
    jaccard相似度等于 交集的大小/并集的大小
    """
    intersection_set = a.intersection(b)
    union_set = a.union(b)
    return len(intersection_set) / len(union_set)


def get_duplicate_content(
    pairs: set, content_ids: list[os.PathLike] | List[int]
) -> dict[os.PathLike, set]:
    """
    利用并查集的思想来合并这些pair
    """
    parent = {id: id for id in content_ids}

    def find(id):
        if parent[id] == id:
            return id
        parent[id] = find(parent[id])
        return parent[id]

    def union(id_a, id_b):
        root_a = find(id_a)
        root_b = find(id_b)
        if root_a != root_b:
            parent[root_a] = root_b

    for id_a, id_b in pairs:
        union(id_a, id_b)

    duplicate_dict = defaultdict(set)
    for c, p in parent.items():
        duplicate_dict[find(c)].add(c)

    return duplicate_dict


def minhash_deduplication(
    input_files: list[os.PathLike] = None,
    num_hashes: int = 10,
    num_bands: int = 2,
    ngrams: int = 2,
    jaccard_threshold: float = 0.8,
    output_directory: os.PathLike = None,
    input_json_file: os.PathLike = 'cs336_data/data/train.jsonl', # 这个参数存在的话 input_files 和 output_directory 就不需要了
):
    if output_directory is not None:
        os.makedirs(output_directory, exist_ok=True)
    # 在指定band_idx和hash值上，相同的content有哪些，如 {(1, (12, 32, 32)) : ["D1", "D2"]}
    band2content = defaultdict(set)
    # 根据content id(行号或者文件名)找到对应的ngrams words，如 {"D1": set("a b c", "b c d"), "D2": set("a b c", "b c e")}
    # content2ngrams = defaultdict(set)

    contents = []
    # 根据参数判断是对目录下的文件去重，还是对json里面的text字段进行去重
    if input_files is None:
        with open(input_json_file, mode="r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                content_id = i
                content = json.loads(line)["text"]
                contents.append((content_id, content))
    else:
        for input_file in input_files:
            with open(input_file, mode="r", encoding="utf-8") as f:
                content_id = input_file
                content = f.read()
                contents.append((content_id, content))

    content_ids = [content_id for content_id, _ in contents]

    for content_id, content in tqdm(contents, desc="Calculating MinHash signatures"):
        signature = []  # 长度为 num_hashes
        # 预处理文档
        processed_content = preprocess_file_content(content)
        # 得到 ngrams_words
        ngrams_words = get_ngrams_words(processed_content.split(" "), ngrams)
        # content2ngrams[content_id] = ngrams_words
        # 计算该文章的signature
        signature = [minhash(ngrams_words, seed) for seed in range(num_hashes)]
        # 将signature分成 num_bands 段
        num_band_ele = num_hashes // num_bands
        for i in range(0, num_hashes, num_band_ele):
            # 注意，key里面得包含band_idx，因为两个文件在相同band_idx上的哈希值一样，才能被认定为duplicate
            band2content[
                (i // num_band_ele, tuple(signature[i : i + num_band_ele]))
            ].add(content_id)

    # 将在同一个band_idx下hash值相同的文档两两一对放到set中
    print("Finding candidate pairs...")
    candidate_pair = set()
    for ids in band2content.values():
        if len(ids) > 1:
            for content_id1, content_id2 in combinations(ids, 2):  # 取出所有的两元对
                candidate_pair.add((content_id1, content_id2))

    verified_pair = candidate_pair
    # 下面这个可以没有，因为我看课程里面是直接将在同个band_idx下hash值相同的文档认定为duplicate了。
    # 利用jaccard相似度判断是否真的是重复的。
    # verified_pair = set()
    # for content_id1, content_id2 in tqdm(candidate_pair, desc="Verifying candidate pairs"):
    #     ngrams_a = get_ngrams_words(processed_content.split(" "), ngrams) 
    #     ngrams_b = content2ngrams[content_id1], content2ngrams[content_id2]
    #     if jaccard_similarity(ngrams_a, ngrams_b) > jaccard_threshold:
    #         verified_pair.add((content_id1, content_id2))

    # 利用并查集将重复的文档放到一起
    duplicate_dict = get_duplicate_content(pairs=verified_pair, content_ids=content_ids)
    
    if input_files is None:
        output_json_file = Path(input_json_file).with_suffix(".dedup.jsonl")
        with open(output_json_file, mode="w", encoding="utf-8") as wf:
            for content_id in tqdm(duplicate_dict.keys(), desc="write to json file"):  # key才是需要保留下来的文件
                content = contents[content_id][1]
                item = {"text": content}
                wf.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        for input_file in duplicate_dict.keys():  # key才是需要保留下来的文件
            basename = os.path.basename(input_file)
            output_file = os.path.join(output_directory, basename)
            shutil.copy2(input_file, output_file)

    print(f"去重完成，{len(contents)} -> {len(duplicate_dict)}")
