from typing import Any, List
from nltk import word_tokenize

from cs336_data.extrace_text import extract_warc_file

def mean_word_length(words: List[str]) -> float:
    total_length = sum([len(word) for word in words])
    return total_length / len(words)

def end_with_ellipsis_ratio(lines: List[str]) -> float:
    end_with_ellipsis_count = sum([line.endswith('...') for line in lines])
    return end_with_ellipsis_count / len(lines)

def alphabetic_word_ratio(words: List[str]):
    """
    包含至少一个alphabetic character的word占所有words的ratio
    """
    count = 0
    for word in words:
        alphabetic_char_count = sum(c.isalpha() for c in word)
        count += alphabetic_char_count > 0
    return count / len(words)

def gopher_classify_quality(text: str) -> tuple[Any, float]:
    words = word_tokenize(text)
    # Contain less than 50 or more than 100,000 words
    if len(words) < 50 or len(words) > 100000:
        return False
    
    # Have a mean word length outside the range of 3 to 10 characters.
    mean_length = mean_word_length(words)
    if mean_length < 3 or mean_length > 10:
        return False

    # Have more than 30% of lines ending with an ellipsis (“...”).
    if end_with_ellipsis_ratio(text.split('\n')) > 0.3:
        return False
    
    # Contain less than 80% of words with at least one alphabetic character.
    if alphabetic_word_ratio(words) < 0.8:
        return False
    
    return True

def gopher_classify_quality_in_warc_file(file_path: str, max_records: int = 20):
    records = extract_warc_file(file_path=file_path, max_records=max_records)
    for record in records:
        res = gopher_classify_quality(record)
        print('=' * 60)
        print(record.replace('\n', ''))
        print(res)

if __name__ == '__main__':
    file_path = 'CC-MAIN-20250417135010-20250417165010-00065.warc.gz'
    gopher_classify_quality_in_warc_file(file_path)    