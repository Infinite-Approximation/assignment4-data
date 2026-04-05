from typing import Any
import fasttext

from cs336_data.extrace_text import extract_warc_file


def classify_nsfw(
    text: str,
    model_path: str = "cs336_data/checkpoint/jigsaw_fasttext_bigrams_nsfw_final.bin",
    model: Any = None
) -> tuple[Any, float]:
    if model is None:
        model = fasttext.load_model(path=model_path)
    labels, scores = model.predict(
        text.replace("\n", " ")
    )  # (('__label__non-nsfw',), array([1.00001001]))
    return (labels[0].replace("__label__", ""), scores[0])


def classify_toxic_speech(
    text: str,
    model_path: str = "cs336_data/checkpoint/jigsaw_fasttext_bigrams_hatespeech_final.bin",
    model: Any = None
) -> tuple[Any, float]:
    if model is None:
        model = fasttext.load_model(path=model_path)
    labels, scores = model.predict(
        text.replace("\n", " ")
    )  # (('__label__non-toxic',), array([1.00001001]))
    return (labels[0].replace("__label__", ""), scores[0])


def classify_harmful_content(file_path: str, max_records: int = 50):
    records = extract_warc_file(file_path=file_path, max_records=max_records)
    for i, record in enumerate(records):
        print(f"正在处理第 {i} 个 record")
        record = record.replace("\n", "")
        nsfw_label, nsfw_score = classify_nsfw(record)
        toxic_label, toxic_score = classify_toxic_speech(record)
        print(record)
        print(nsfw_label, nsfw_score)
        print(toxic_label, toxic_score)


if __name__ == "__main__":
    file_path = "CC-MAIN-20250417135010-20250417165010-00065.warc.gz"
    classify_harmful_content(file_path)
