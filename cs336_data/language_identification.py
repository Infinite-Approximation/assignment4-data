import fasttext
from typing import Any
from cs336_data.extrace_text import extract_warc_file

def identify_language(text: str, model_path: str = 'cs336_data/checkpoint/lid.176.bin') -> tuple[Any, float]:
    model = fasttext.load_model(path=model_path)
    lables, scores = model.predict(text=text.replace('\n', ''), k=1) # ('__label__en',) [0.15209572]
    return (lables[0].replace('__label__', ''), scores[0])


def identify_language_in_warc_file(file_path: str, max_records: int = 20):
    records = extract_warc_file(file_path=file_path, max_records=max_records)
    for record in records:
        label, score = identify_language(record)
        print(label, score)
    

if __name__ == '__main__':
    file_path = 'CC-MAIN-20250417135010-20250417165010-00065.warc.gz'
    identify_language_in_warc_file(file_path)    