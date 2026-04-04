from collections import Counter, defaultdict
import os
import re
import string
import mmh3
from typing import List
import unicodedata
from itertools import combinations
import shutil

def exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    # 第一遍扫描，获取每一行出现的次数。这里使用行对应hash来作为key，从而减少内存消耗
    line_count = Counter()
    for input_file in input_files:
        with open(file=input_file, mode='r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n') # 防止最后一行内容一样，但是没有 \n 导致hash和其他行不同
                line_count[hash(line)] += 1

    # 第二遍扫码，只保留出现一次的行
    for input_file in input_files:
        basename = os.path.basename(input_file)
        output_file = os.path.join(output_directory, basename)
        with open(file=output_file, mode='w', encoding='utf-8') as wf:
            with open(file=input_file, mode='r', encoding='utf-8') as rf:
                for line in rf:
                    if line_count[hash(line.rstrip('\n'))] == 1: # 这里和前面对应，使用 rstripe('\n')
                        wf.write(line)

def preprocess_file_content(content: str) -> str:
    # applying NFD unicode normalization
    content = unicodedata.normalize('NFD', content)
    # removing accents
    content = ''.join(c for c in content if unicodedata.category(c) != 'Mn')
    # lowercasing
    content = content.lower()
    # removing punctuation
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    content = content.translate(translator)
    # normalizing whitespaces
    content = ' '.join(content.split())
    
    return content

def get_ngrams_words(text: list[str], ngrams: int) -> set[str]:
    text_len = len(text)
    i = 0
    ngrams_words = set()
    while i + ngrams - 1 < text_len:
        ngrams_words.add(' '.join(text[i: i + ngrams]))
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

def get_duplicate_docs(pairs: set, input_files: list[os.PathLike]) -> dict[os.PathLike, set]:
    """
    利用并查集的思想来合并这些pair
    """
    parent = {doc: doc for doc in input_files}
    def find(doc):
        if parent[doc] == doc:
            return doc
        parent[doc] = find(parent[doc])
        return parent[doc]
    
    def union(doc_a, doc_b):
        root_a = find(doc_a)
        root_b = find(doc_b)
        if root_a != root_b:
            parent[root_a] = root_b

    for doc_a, doc_b in pairs:
        union(doc_a, doc_b)
    
    duplicate_dict = defaultdict(set)
    for c, p in parent.items():
        duplicate_dict[find(c)].add(c)
    
    return duplicate_dict

def minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    os.makedirs(output_directory, exist_ok=True)

    band2doc = defaultdict(set) # band对应的文档有哪些，如 {(1, (12, 32, 32)) : ["D1", "D2"]}
    doc2ngrams = defaultdict(set)
    for input_file in input_files:
        with open(input_file, mode='r', encoding='utf-8') as f:
            signature = [] # 长度为 num_hashes
            content = f.read()
            # 预处理文档
            processed_content = preprocess_file_content(content)
            # 得到 ngrams_words
            ngrams_words = get_ngrams_words(processed_content.split(' '), ngrams)
            doc2ngrams[input_file] = ngrams_words
            # 计算该文章的signature
            signature = [minhash(ngrams_words, seed) for seed in range(num_hashes)]
            # 将signature分成 num_bands 段
            num_band_ele = num_hashes // num_bands
            for i in range(0, num_hashes, num_band_ele):
                # 注意，key里面得包含band_idx，因为需要相同band_idx的哈希值一样，才能被认定为duplicate
                band2doc[(i//num_band_ele, tuple(signature[i: i+num_band_ele]))].add(input_file)
    
    # 将在同一个band中hash值相同的文档两两一对放到set中
    candidate_pair = set()
    for docs in band2doc.values():
        if len(docs) > 1:
            for doc_a, doc_b in combinations(docs, 2): # 取出所有的两元对
                candidate_pair.add((doc_a, doc_b))

    # 利用jaccard相似度判断是否真的是重复的
    verified_pair = set()
    for doc_a, doc_b in candidate_pair:
        ngrams_a, ngrams_b = doc2ngrams[doc_a], doc2ngrams[doc_b]
        if jaccard_similarity(ngrams_a, ngrams_b) > jaccard_threshold:
            verified_pair.add((doc_a, doc_b))

    # 利用并查集将重复的文档放到一起
    duplicate_dict = get_duplicate_docs(pairs=verified_pair, input_files=input_files)
    for input_file in duplicate_dict.keys(): # key才是需要保留下来的文件
        basename = os.path.basename(input_file)
        output_file = os.path.join(output_directory, basename)
        shutil.copy2(input_file, output_file)