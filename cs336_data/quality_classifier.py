import os
import random
import subprocess
from typing import Any

import fasttext

from cs336_data.extrace_text import extract_warc_file
from cs336_data.harmful_detect import classify_nsfw, classify_toxic_speech
from cs336_data.language_identification import identify_language
from cs336_data.quality_filter import gopher_classify_quality
from tqdm import tqdm

def sample_url(input_file: os.PathLike,
               output_file: os.PathLike,
               sample_num: int = 200,
               seed: int = 42):
    with open(input_file, mode='r') as rf:
        input_urls = rf.read().split('\n')
    random.seed(seed)
    output_urls = random.sample(input_urls, k=sample_num)
    with open(output_file, mode='w') as wf:
        wf.write('\n'.join(output_urls))

def sample_positive_urls_and_download(
    file_path: str = "cs336_data/data/enwiki-20240420-extracted_urls.txt",
    max_urls: int = 10000,
    output_file_path: str = "cs336_data/data/subsampled_positive_urls.txt",
):
    """

    wget --timeout=5 \
        --tries=1 \
        --max-redirect=1 \
        --connect-timeout=2 \
        --read-timeout=5 \
        --user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" \
        -i cs336_data/data/subsampled_positive_urls.txt \
        --warc-file=cs336_data/data/subsampled_positive_urls_10000 \
        -O /dev/null

    抓取url的内容并转化为warc文件
    """
    sample_url(input_file=file_path, output_file=output_file_path, sample_num=max_urls)
    download_commmand = f'wget  –-timeout=5 \
                                --tries=1 \
                                --max-redirect=1 \
                                --connect-timeout=2 \
                                --read-timeout=5 \
                                --user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36" \
                                -i {output_file_path} \
                                --warc-file=cs336_data/data/subsampled_positive_urls \
                                -O /dev/null'
    subprocess.run(download_commmand, shell=True, check=True)


def is_low_quality_data(
    text: str,
    language_identification_model: Any,
    nsfw_detection_model: Any,
    toxic_detection_model: Any,
) -> bool:
    """
    判断一个html内容是否是低质量文本，这个相比于 classify_quality 函数来说，是一个粗略的过滤函数
    """
    # 过滤质量太低的
    if gopher_classify_quality(text) == False:
        return True
    # 过滤非英语的
    language, score = identify_language(text=text, model=language_identification_model)
    if language != "en" or score < 0.8:
        return True
    # 取出有害内容
    nsfw_label, nsfw_score = classify_nsfw(text=text, model=nsfw_detection_model)
    if nsfw_label != "non-nsfw" or nsfw_score < 0.8:
        return True
    toxic_label, toxic_score = classify_toxic_speech(text=text, model=toxic_detection_model)
    if toxic_label != "non-toxic" or toxic_score < 0.8:
        return True
    return False


def prepare_data(
    positive_sample_warc_file: os.PathLike = "cs336_data/data/subsampled_positive_urls.warc.gz",
    negative_sample_warc_file: os.PathLike = "cs336_data/data/CC-MAIN-20250417135010-20250417165010-00065.warc.gz",
    output_file: os.PathLike = "cs336_data/data/train.txt",
    max_records: int = 10000,
    language_identification_model_path: str = "cs336_data/checkpoint/lid.176.bin",
    nsfw_detection_model_path: str = "cs336_data/checkpoint/jigsaw_fasttext_bigrams_nsfw_final.bin",
    toxic_detection_model_path: str = "cs336_data/checkpoint/jigsaw_fasttext_bigrams_hatespeech_final.bin",
):
    print("正在取出positive samples...")
    positive_samples = extract_warc_file(
        positive_sample_warc_file, max_records=max_records
    )
    # 可能取不到max_records
    if len(positive_samples) != max_records:
        max_records = len(positive_samples)
    print(f"len(positive_samples)={len(positive_samples)}")
    print("正在取出negative samples...")
    negative_samples = extract_warc_file(
        negative_sample_warc_file, max_records=max_records * 5
    )
    # 随机采样
    random.seed(42)
    negative_samples = random.sample(negative_samples, max_records)
    print("正在加载模型")
    language_identification_model = fasttext.load_model(
        path=language_identification_model_path
    )
    nsfw_detection_model = fasttext.load_model(path=nsfw_detection_model_path)
    toxic_detection_model = fasttext.load_model(path=toxic_detection_model_path)
    positive_texts = []
    negative_texts = []
    with open(output_file, mode="w") as f:
        for i in tqdm(range(max_records), desc="process record"):
            # 从positive sample中剔除较差的样本
            # 这里的positive是指wiki页面指向的链接的内容
            if not is_low_quality_data(
                positive_samples[i],
                language_identification_model,
                nsfw_detection_model,
                toxic_detection_model,
            ):
                positive_texts.append(positive_samples[i].replace('\n', ' '))
            
            # 从negative sample中剔除较差的样本
            # 这里的negative是指随机从cc中拿的数据
            if not is_low_quality_data(
                negative_samples[i],
                language_identification_model,
                nsfw_detection_model,
                toxic_detection_model,
            ):
                negative_texts.append(negative_samples[i].replace('\n', ' '))

    print(f"过滤出来 {len(positive_texts)} 个正样本和 {len(negative_texts)} 个负样本")
            
    # 构造数据集，写入txt
    with open(output_file, mode='w') as f:
        for i in range(min(len(positive_texts), len(negative_texts))):                
            f.write(f"__label__high_quality {positive_texts[i]}\n")
            f.write(f"__label__low_quality {negative_texts[i]}\n")



def train_quality_classification_model(
    train_txt: os.PathLike = "cs336_data/data/train.txt",
    model_save_path: str = "cs336_data/checkpoint/quality.bin",
):
    model = fasttext.train_supervised(input=train_txt)
    model.save_model(model_save_path)


def classify_quality(
    text: str, 
    model_path: str = "cs336_data/checkpoint/quality.bin",
    model: Any = None
) -> tuple[Any, float]:
    if model is None:
        model = fasttext.load_model(model_path)
    labels, scores = model.predict(
        text.replace("\n", " ")
    )  # ('__label__low_quality',) [0.50003731]
    return labels[0].replace("__label__", ""), scores[0]


if __name__ == "__main__":
    prepare_data()
    train_quality_classification_model()
    with open('tests/fixtures/high_quality_wiki_reference.txt') as f:
        text = f.read()
    res = classify_quality(text)
    print(res)

    # sample_url(
    #     input_file='cs336_data/data/wet.paths',
    #     output_file='cs336_data/data/sampled_wet.paths',
    #     sample_num=1000
    # )
