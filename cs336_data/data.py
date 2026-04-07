from collections import Counter
from functools import partial
import multiprocessing
import os
import time
import fasttext
import numpy as np
import concurrent.futures
import json
from datasets import load_dataset
from transformers import AutoTokenizer
from pathlib import Path
from tqdm import tqdm
from fastwarc.warc import ArchiveIterator, WarcRecordType
from tldextract import TLDExtract
from transformers import logging
from cs336_data.deduplicate import exact_line_deduplication, minhash_deduplication
from cs336_data.harmful_detect import is_harmful_content
from cs336_data.language_identification import is_en
from cs336_data.mask import mask_pii
from cs336_data.quality_classifier import classify_quality
from cs336_data.quality_filter import gopher_classify_quality

# 只显示 Error，忽略 Warning
logging.set_verbosity_error()

def build_valid_bin(
    output_path: str = "cs336_data/data/tokenized_paloma_c4_100_domains_validation.bin",
):
    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print("正在加载数据集...")
    c4_val_dataset = load_dataset(
        path="allenai/paloma", name="c4_100_domains", split="val", streaming=True
    )  # 每一个元素都是一个样本，包含text，id，subdomain之类的
    print("正在tokenize")
    with open(output_path, mode="wb") as f:
        for ele in c4_val_dataset:
            ids = gpt2_tokenizer.encode(ele["text"], add_special_tokens=False)
            ids.append(gpt2_tokenizer.eos_token_id)
            np.asarray(ids, dtype=np.uint16).tofile(f)
    print(f"Save to {output_path}")


def check_valid_bin(
    input_path: str = "cs336_data/data/tokenized_paloma_c4_100_domains_validation.bin",
):
    data = np.fromfile(input_path, dtype=np.uint16)
    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    text = gpt2_tokenizer.decode(data[:2000])
    print(text)


def process_single_wet_file(
    input_path: str,
    output_path: str,
    language_identification_model_path: str = "cs336_data/checkpoint/lid.176.bin",
    nsfw_detection_model_path: str = "cs336_data/checkpoint/jigsaw_fasttext_bigrams_nsfw_final.bin",
    toxic_detection_model_path: str = "cs336_data/checkpoint/jigsaw_fasttext_bigrams_hatespeech_final.bin",
    quality_classify_model_path: str = "cs336_data/checkpoint/quality.bin",
):
    """
    输入是wet，输出是jsonl
    """
    stats = {
        "non_en": 0, # 第一个阶段过滤了多少
        "harmful_content": 0, # 第二个阶段过滤了多少
        "gopher_filter": 0,
        "low_quality": 0
    }
    language_identification_model = fasttext.load_model(
        path=language_identification_model_path
    )
    nsfw_detection_model = fasttext.load_model(path=nsfw_detection_model_path)
    toxic_detection_model = fasttext.load_model(path=toxic_detection_model_path)
    quality_classify_model = fasttext.load_model(quality_classify_model_path)

    with open(output_path, "w", encoding="utf-8") as wf:  # 写入jsonl
        with open(input_path, "rb") as stream:  # 从wet中读取
            for record in ArchiveIterator(
                stream, record_types=WarcRecordType.conversion
            ):
                text = record.reader.read().decode("utf-8", errors="replace")
                # 开始对text进行过滤
                # 1. 过滤非英语
                if not is_en(text, model=language_identification_model):
                    stats['non_en'] += 1
                    continue
                # 将个人信息进行掩码，不算过滤
                text = mask_pii(text)
                # 2. 过滤harmful content
                if is_harmful_content(
                    text,
                    nsfw_detection_model=nsfw_detection_model,
                    toxic_detection_model=toxic_detection_model,
                ):
                    stats['harmful_content'] += 1
                    continue
                # 3. 利用gopher规则过滤
                if gopher_classify_quality(text) == False:
                    stats['gopher_filter'] += 1
                    continue
                # 4. 利用2.7节训练的quality classifier来过滤
                quality_label, quality_score = classify_quality(
                    text=text, model=quality_classify_model
                )
                if quality_label != "high_quality" or quality_score < 0.5:
                    stats['low_quality'] += 1
                    continue

                url = record.headers.get("WARC-Target-URI")
                item = {"text": text, "url": url}
                wf.write(json.dumps(item, ensure_ascii=False) + "\n")

    return output_path, stats 


def process_wet_files(
    wet_file_dir: os.PathLike = "cs336_data/data/wet_files",
    output_dir: os.PathLike = "cs336_data/data/processed_wet_files",
    final_json_path: os.PathLike = "cs336_data/data/train.jsonl",
):
    """
    并行的处理多个wet files
    """
    start_time = time.perf_counter()
    os.makedirs(output_dir, exist_ok=True)

    num_cpus = len(os.sched_getaffinity(0))
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus)

    wet_files_path = Path(wet_file_dir)
    output_path = Path(output_dir)
    futures = []
    for wet_file_path in wet_files_path.iterdir():
        if wet_file_path.is_file():  # 只对file处理
            basename = wet_file_path.with_suffix(".jsonl").name
            future = executor.submit(
                process_single_wet_file, wet_file_path, output_path / basename
            )
            futures.append(future)


    accumulative_counter = Counter()
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        output_file, stats = future.result()
        # 将output_file写入到final_json_path中
        with open(output_file, "r", encoding="utf-8") as rf, \
             open(final_json_path, "a", encoding="utf-8") as wf:
            for line in rf:
                wf.write(line)
        # 将每个wet file的过滤统计结果进行累加
        accumulative_counter = accumulative_counter + Counter(stats)
        print(f"Output file written: {output_file}")
    print(f"第一个阶段(non_en)过滤了 {accumulative_counter['non_en']} 个样本\n，\
            第二个阶段(harmful_content)过滤了 {accumulative_counter['harmful_content']} 个样本\n，\
            第三个阶段(gopher规则)过滤了 {accumulative_counter['gopher_filter']} 个样本\n，\
            第四个阶段(quality classifier)过滤了 {accumulative_counter['low_quality']} 个样本。\n")

    # 对json文件进行去重
    minhash_deduplication(input_json_file=final_json_path)

    end_time = time.perf_counter()
    print(f"Processing completed in {end_time - start_time:.2f} seconds.")

def tokenize_line_and_add_eos(
        tokenizer: AutoTokenizer,
        line: str
) -> list[int]:
        return tokenizer.encode(line) + [tokenizer.eos_token_id]

def get_train_bin(
    input_json_file: os.PathLike = "cs336_data/data/train.dedup.jsonl",
    chunksize: int = 100,
    output_path: os.PathLike = "cs336_data/data/train.bin"
):
    lines = []
    with open(input_json_file, "r", encoding="utf-8") as f:
        for line in f:
            content = json.loads(line)["text"]
            lines.append(content)

    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenize_line_and_add_eos_partial = partial(tokenize_line_and_add_eos, gpt2_tokenizer)
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        results = []
        for result in tqdm(
            pool.imap(tokenize_line_and_add_eos_partial, lines, chunksize=chunksize), 
            total=len(lines),
            desc="Tokenizing lines"
        ):
            results.append(result)

    # 展平token id
    all_ids = [id for result in results for id in result]
    print(f"Tokenized and encoded {input_json_file} into {len(all_ids)} tokens")
    ids_array = np.array(all_ids, dtype=np.uint16)
    ids_array.tofile(output_path)

if __name__ == "__main__":
    # build_valid_bin()
    # check_valid_bin()
    # process_wet_files()
    # exact_line_deduplication(input_json_file="cs336_data/data/train.jsonl")
    # minhash_deduplication(input_json_file="cs336_data/data/train.jsonl")
    get_train_bin()