# 1 Assignment Overview
For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment4_data.pdf](./cs336_spring2025_assignment4_data.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

This directory is organized as follows:

- [`./cs336-basics`](./cs336-basics): directory containing a module
  `cs336_basics` and its associated `pyproject.toml`. This module contains the staff 
  implementation of the language model from assignment 1. You will use this training code
  to train an LM on your filtered data. You should not modify the training logic, since
  your leaderboard submission must use it exactly.
- [`./cs336_data`](./cs336_data): This folder is basically empty! This is the
  module where you will implement code to filter and process the data.

Visually, it should look something like:

``` sh
.
├── cs336_basics  # A python module named cs336_basics
│   └── ... an optimized training implementation ...
├── cs336_data  # TODO(you): code that you'll write for assignment 4
│   ├── __init__.py
│   └── ... TODO(you): any other files or folders you need for assignment 4 ...
├── README.md
├── pyproject.toml
└── ... TODO(you): other files or folders you need for assignment 4 ...
```

As in previous assignments, we use `uv` to manage dependencies.
# 2 Filtering Common Crawl

## 2.1 Looking at the data

使用下面的命令下载WARC文件（2018 年 4 月的一次抓取）

```sh
wget https://data.commoncrawl.org/crawl-data/CC-MAIN-2025-18/segments/1744889135610.12/warc/CC-MAIN-20250417135010-20250417165010-00065.warc.gz
```

下载对应的WET文件

```sh
wget https://data.commoncrawl.org/crawl-data/CC-MAIN-2025-18/segments/1744889135610.12/wet/CC-MAIN-20250417135010-20250417165010-00065.warc.wet.gz
```

**Problem (look_at_cc)**

(a) URL是 http://0371rykj.com/ipfhsb/34.html，现在不能访问了。

这个网页是卖产品的，比如恒温恒湿实验箱。

(b) 有一些成人的内容应该被删除。

会输出有有害的信息。

能够清除的描述一个产品。

(c) 这个数据适用于训练一个客服机器人。但是不适用于训练一个数学领域的机器人。

(d) 

1 中文，0371rykj.com，首页。中文网页老是有很多成人内容。

2 中文，10www.chinatikfans.com，论坛帖子页

3 英文，13.usnccm.org，Academic conference homepage，属于高质量网页。所以看到第三个网页就看到了高质量网页。

4 中文，176.utchat888.com，网站登录页

5 中文，176766.cn，网站首页

剩下的20网页也是这样重复操作。。。

## 2.2 HTML to text conversion

**Problem (extract_text)**

(a) Write a function that extracts text from a byte string containing raw HTML

```python
from resiliparse.parse.encoding import detect_encoding, bytes_to_str 
from resiliparse.extract.html2text import extract_plain_text

def extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    plain_text = bytes_to_str(html_bytes, detect_encoding(html_bytes))
    text = extract_plain_text(plain_text)
    return text
```

(b) Run your text extraction function on a single WARC file

```python
def extract_warc_file(file_path: str, max_records: int = 2):
    from fastwarc.warc import ArchiveIterator, WarcRecordType
    records = []
    cur_iter = 0
    with open(file_path, 'rb') as stream:
        for record in ArchiveIterator(stream, record_types=WarcRecordType.response): # 取出record type为response的record
            html_content_in_bytes = record.reader.read()
            html_content = extract_text_from_html_bytes(html_content_in_bytes)
            records.append(html_content)
            print(html_content)
            cur_iter += 1
            if cur_iter == max_records:
                return records

if __name__ == '__main__':
    file_path = 'CC-MAIN-20250417135010-20250417165010-00065.warc.gz'
    extract_warc_file(file_path)
```

1. 使用我写的函数提取出来的纯文本内容有很多的空行，而WET里面没有空行，看起来紧凑
2. 我写的函数会将大部分的成人内容剔除掉。

## 2.3 Language identification

**Problem (language_identification)**

(a) 

```python
import fasttext
from typing import Any

def identify_language(text: str, model_path: str = 'cs336_data/checkpoint/lid.176.bin') -> tuple[Any, float]:
    model = fasttext.load_model(path=model_path)
    lables, scores = model.predict(text=text.replace('\n', ''), k=1) # ('__label__en',) [0.15209572]
    return (lables[0].replace('__label__', ''), scores[0])
```

(b)

比如想要在英文文本上进行训练，结果识别出来的是一些德语，导致模型输出德语。

使用集成方案，使用多模型集成判断。

(c)

```python
def identify_language_in_warc_file(file_path: str, max_records: int = 20):
    records = extract_warc_file(file_path=file_path, max_records=max_records)
    for record in records:
        label, score = identify_language(record)
        print(label, score)
    

if __name__ == '__main__':
    file_path = 'CC-MAIN-20250417135010-20250417165010-00065.warc.gz'
    identify_language_in_warc_file(file_path) 
```

标注结果：    zh zh en zh zh zh zh zh nl el el zh tr en en zh zh zh zh zh

分类器结果： zh zh en zh zh zh zh zh nl el el zh tr en en zh zh zh zh zh

对于前二十个样本，分类器没有任何的错误。有15%的文档是英文。置信度给85%。

## 2.4 Personal identifiable information

**Problem (mask_pii)**

1.

学习一下Python的正则表达式： 

`[]` 表示 character class，可以从里面选character。 `()` 表示group，配合符号可以指定 `()` 里面的内容出现几次。 

`?` 表示出现0次或1次，`*` 表示出现任意多次(包括0次)， `+` 表示至少出现一次

`[]` 内的字符就是原本的字符，不需要转义。比如 `.`  就表示字符本身，但是 `[]` 外的字符如果想使用它本身，就需要加上 `\` 来转义。但是注意在 `[]` 中 `-` 表示范围，如果需要字符本身要转义。

```python
import re
def mask_emails(text: str) -> tuple[str, int]:
    pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9-]+(\.[a-zA-Z0-9-]+)*\.[a-zA-Z]+'
    result_str, count = re.subn(pattern, "|||EMAIL_ADDRESS|||", text)
    return (result_str, count)
```

2.

```python
def mask_phone_numbers(text: str) -> tuple[str, int]:
    pattern = r'(\(\d{3}\)|\d{3})[.\- ]?\d{3}[.\- ]?\d{4}'
    result_str, count = re.subn(pattern, "|||PHONE_NUMBER|||", text)
    return (result_str, count)
```

3.

```python
def mask_ips(text: str) -> tuple[str, int]:
    pattern = r'(25[0-9]|2[0-4]\d|[01]?\d\d?\.){3}(25[0-9]|2[0-4]\d|[01]?\d\d?)'
    result_str, count = re.subn(pattern, "|||IP_ADDRESS|||", text)
    return (result_str, count)
```

4.

问题：会让模型学会输出 `|||EMAIL_ADDRESS|||` 这种格式。

解决方案：修改成根本不存在的数字就行。

5.

false positive是指识别出来是PII，但不是PII。false negative是指没识别出来是PII，但它是PII。

```python
def mask_pii(file_path: str, max_records: int = 200):
    from cs336_data.extrace_text import extract_warc_file
    texts = extract_warc_file(file_path=file_path, max_records=max_records)
    replace_count = 0
    for text in texts:
        mask_email_str, mask_email_count = mask_emails(text)
        mask_phone_numbers_str, mask_phone_numbers_count = mask_phone_numbers(mask_email_str)
        mask_ips_str, mask_ips_count = mask_ips(mask_phone_numbers_str)
        if mask_email_count or mask_phone_numbers_count or mask_ips_count:
            print('=' * 60)
            print(mask_ips_str)
            print('=' * 60)
            replace_count += 1
            if replace_count == 20:
                return

if __name__ == '__main__':
    file_path = 'CC-MAIN-20250417135010-20250417165010-00065.warc.gz'
    mask_pii(file_path=file_path)
```

false positive的例子没找到。

false negative：一些其他类型的号码。

## 2.5 Harmful content

1

```python
def classify_nsfw(text: str, model_path: str = 'cs336_data/checkpoint/jigsaw_fasttext_bigrams_nsfw_final.bin') -> tuple[Any, float]:
    model = fasttext.load_model(path=model_path)
    labels, scores = model.predict(text) # (('__label__non-nsfw',), array([1.00001001]))
    return (labels[0].replace('__label__', ''), scores[0])
```

2

```python
def classify_toxic_speech(text: str, model_path: str = 'cs336_data/checkpoint/jigsaw_fasttext_bigrams_hatespeech_final.bin') -> tuple[Any, float]:
    model = fasttext.load_model(path=model_path)
    labels, scores = model.predict(text) # (('__label__non-toxic',), array([1.00001001]))
    return (labels[0].replace('__label__', ''), scores[0])
```

3

潜在的下游问题：一些性知识可能会被过滤，导致模型根本不知道性知识。模型不知道什么是有害内容，就不会知道什么是好内容。

缓解方案：1. 提高分类器的分类阈值，只有高置信度的才会被剔除。2. 使用人工审核。

4

模型识别出来都是 non-nsfw 和 non-toxic 的。

但是对于一些网页内有成人广告的应该识别为 nsfw 的。

## 2.6 Quality Rules

**Problem (gopher_quality_filters)**

a

```python
from typing import Any, List
from nltk import word_tokenize

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
```

b

没有问题

## 2.7 Quality Classifier

使用下面的命令下载包含**维基百科指向链接**的文件，这些链接的网页可以认为是高质量网页。

```sh
wget https://nlp.stanford.edu/data/nfliu/cs336-spring-2024/assignment4/enwiki-20240420-extracted_urls.txt.gz
```

然后进行解压：

```sh
gunzip https://nlp.stanford.edu/data/nfliu/cs336-spring-2024/assignment4/enwiki-20240420-extracted_urls.txt.gz
```

我们就得到了 `enwiki-20240420-extracted_urls.txt` 文件。里面有43.5M个链接。使用python函数进行采样：

```python
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
```

这里的 `sample_num` 我设置为了 10000。

得到采样后的文件 `cs336_data/data/subsampled_positive_urls.txt`。然后使用wget进行下载：

```sh
wget --timeout=5 \
    --tries=1 \
    --max-redirect=1 \
    --connect-timeout=2 \
    --read-timeout=5 \
    --user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" \
    -i cs336_data/data/subsampled_positive_urls.txt \
    --warc-file=cs336_data/data/subsampled_positive_urls_10000 \
    -O /dev/null
```

其实最好还是使用 aria2 进行多线程下载，在下面的WET数据收集的时候我就是使用aria2的。

最后我们得到下载好的文件：`subsampled_positive_urls_10000.war.gz`

然后我们利用这个数据，构建质量分类模型所需要的数据。具体来说，正样本是从 `subsampled_positive_urls_10000.war.gz` 中获取的，负样本是从 `CC-MAIN-20250417135010-20250417165010-00065.warc.gz` (这是一次common crawl得到的文件)获取的。

我们认为wiki网页指向的链接所包含的内容是高质量的，而随机爬虫得到的网页内容是低质量的。

但是我们都得这两个内容进行一次粗略的过滤，使用下面的函数：

```python
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
```

这里面的filter都是之前写过的函数。

那么可以得到我们的数据集构建函数：

```python
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
    # 这里先取出5倍max_records个负面样本，然后在下面进行随机采样max_records个
    negative_samples = extract_warc_file(
        negative_sample_warc_file, max_records=max_records * 5
    )
    # 随机采样
    random.seed(42)
    # 得到 max_records 个负面样本
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
```

然后我们可以开始训练我们的模型了：

```python
def train_quality_classification_model(
    train_txt: os.PathLike = "cs336_data/data/train.txt",
    model_save_path: str = "cs336_data/checkpoint/quality.bin",
):
    model = fasttext.train_supervised(input=train_txt)
    model.save_model(model_save_path)
```

最后写出 `classify_quality` 函数，它利用上面训练好的模型对文本质量进行评估：

```python
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
```

# 3 Deduplication

## 3.1 Exact line deduplication

**Problem (exact_deduplication)**

```python
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
```

## 3.2 MinHash + LSH document deduplication

我们先写几个必要的函数。

实现MinHash：

```python
import mmh3
def minhash(S: set[str], seed: int):
    """
    S是ngram_words，seed对应一个hash函数
    """
    # 使用mm3.hash来计算每个n-grams word的hash值，这个hash函数是由seed决定的。
    return min(mmh3.hash(x, seed) for x in S)
```

对文本进行正则化处理的函数：

```python
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
```

实现jaccard_similarity函数：

```python
def jaccard_similarity(a: set[str], b: set[str]):
    """
    jaccard相似度等于 交集的大小/并集的大小
    """
    intersection_set = a.intersection(b)
    union_set = a.union(b)
    return len(intersection_set) / len(union_set)
```

获取一个文本对饮的ngram_words：

```python
def get_ngrams_words(text: list[str], ngrams: int) -> set[str]:
    text_len = len(text)
    i = 0
    ngrams_words = set()
    while i + ngrams - 1 < text_len:
        ngrams_words.add(' '.join(text[i: i + ngrams]))
        i += 1
    return ngrams_words
```

利用并查集思想来合并相似的文档(pair形式)

```python
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
```

最后让我们开始完成我们的去重函数：

```python
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
    # 1. 将在相同band_idx上hash值相同的文档放到在一个list里面。需要使用到 band2doc 变量
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
    
    # 2. 将在同一个band中hash值相同的文档两两一对放到set中，为了使用jaccard similarty进行过滤
    candidate_pair = set()
    for docs in band2doc.values():
        if len(docs) > 1:
            for doc_a, doc_b in combinations(docs, 2): # 取出所有的两元对
                candidate_pair.add((doc_a, doc_b))

    # 3。 利用jaccard相似度判断是否真的是重复的
    verified_pair = set()
    for doc_a, doc_b in candidate_pair:
        ngrams_a, ngrams_b = doc2ngrams[doc_a], doc2ngrams[doc_b]
        if jaccard_similarity(ngrams_a, ngrams_b) > jaccard_threshold:
            verified_pair.add((doc_a, doc_b))

    # 4. 利用并查集将重复的文档放到一起
    duplicate_dict = get_duplicate_docs(pairs=verified_pair, input_files=input_files)
    for input_file in duplicate_dict.keys(): # key才是需要保留下来的文件
        basename = os.path.basename(input_file)
        output_file = os.path.join(output_directory, basename)
        shutil.copy2(input_file, output_file)
```

# 4 Leaderboard: filter data for language modeling

## 验证集构建

由于我们无法访问 `/data/paloma/tokenized_paloma_c4_100_domains_validation.bin` ，所以也没有验证集，需要我们自己去构建。

构建脚本如下：

```python
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

def build_valid_bin(
    output_path: str = 'cs336_data/data/tokenized_paloma_c4_100_domains_validation.bin'
):
    gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2')
    print("正在加载数据集...")
    c4_val_dataset = load_dataset(
        path='allenai/paloma',
        name='c4_100_domains',
        split='val',
        streaming=True
    ) # 每一个元素都是一个样本，包含text，id，subdomain之类的
    print("正在tokenize")
    with open(output_path, mode='wb') as f:
        for ele in c4_val_dataset:
            ids = gpt2_tokenizer.encode(ele['text'], add_special_tokens=False)
            ids.append(gpt2_tokenizer.eos_token_id)
            np.asarray(ids, dtype=np.uint16).tofile(f)
    print(f"Save to {output_path}")

if __name__ == '__main__':
    build_valid_bin()
```

注意，需要自己进入 `https://huggingface.co/datasets/allenai/paloma` 网页来允许访问 paloma 数据集，然后在本机配置好huggingface的token：

```bash
uv add datasets huggingface_hub
uv run huggingface-cli login # 填入hugginface的access token
```

运行脚本之后就可以得到 `tokenized_paloma_c4_100_domains_validation.bin` 文件了。

使用下面的脚本来查看前2000个token对应的文本（pdf里面有提到）：

```python
def check_valid_bin(
    input_path: str = 'cs336_data/data/tokenized_paloma_c4_100_domains_validation.bin'
):  
    data = np.fromfile(
        input_path,
        dtype=np.uint16
    )
    gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2')
    text = gpt2_tokenizer.decode(data[:2000])
    print(text)
```

得到结果如下：

![image-20260406103142790](https://typora-jkd.oss-cn-hangzhou.aliyuncs.com/%20img/image-20260406103142790.png)

![image-20260406103159731](https://typora-jkd.oss-cn-hangzhou.aliyuncs.com/%20img/image-20260406103159731.png)

由于我们构建评估集的时候添加了 eos_token_id，所以在deocde的时候能看到 `<|endoftext|>`

## WET数据收集

由于无法访问Together集群，所以需要自己去下载5000个WET file。

进入 https://commoncrawl.org/overview 这个网站，我们可以每一次Common Crawl爬取的数据。

我们选取CC-MAIN-2026-12这个最新的数据，然后下载对应的 `wet.paths.gz` 文件，里面每一行都是一个WET文件的地址。访问这个地址就能获得一个WET文件，表示一段时间内网络上的数据。

```sh
curl -s https://data.commoncrawl.org/crawl-data/CC-MAIN-2026-12/wet.paths.gz | gzip -dc > wet.paths
```

`wet.paths` 数据如下所示：

![image-20260405204951602](https://typora-jkd.oss-cn-hangzhou.aliyuncs.com/%20img/image-20260405204951602.png)

然后写一个函数来从里面随机获取 5000 个地址，

其实使用shell的 `shuf` 命令更快，但是随机种子的设置较为麻烦。所以这里我们采用python来实现：

```python
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
        
if __name__ == "__main__":
    sample_url(
        input_file='cs336_data/data/wet.paths',
        output_file='cs336_data/data/sampled_wet.paths',
        sample_num=1000
    )
```

然后给该文件里面的每个url添加上 `https://data.commoncrawl.org/` 前缀。

在每一行前内容的命令如下：

```sh
sed -i 's|模式|替换内容|' 文件名    # 其中 -i 表示原地修改该文件
```

所以我们对我们的文件进行如下操作：

```sh
sed -i 's|^|https://data.commoncrawl.org/|' cs336_data/data/sampled_wet.paths
```

最终得到采样后的wet path文件：

![image-20260405205627492](https://typora-jkd.oss-cn-hangzhou.aliyuncs.com/%20img/image-20260405205627492.png)

由于5000个wet文件大约要占用400G的磁盘空间，而我的设备没有那么大的空间，所以我只下载1000个wet文件。

然后为了提高下载速率，推荐使用 `aria2` 来下载。

```sh
aria2c -i cs336_data/data/sampled_wet.paths \
	   -d cs336_data/data/wet_files \
	   -j 10 \ 
	   -x 5 \
	   -c
```

命令参数解释如下：

<img src="https://typora-jkd.oss-cn-hangzhou.aliyuncs.com/%20img/image-20260405211202791.png" alt="image-20260405211202791" style="zoom:50%;" />

下载好之后可以看到有很多wet文件：

<img src="https://typora-jkd.oss-cn-hangzhou.aliyuncs.com/%20img/image-20260405224951963.png" alt="image-20260405224951963" style="zoom: 67%;" />

大小为 58GB

## 处理 WET文件

首先我们需要写一个函数来处理单个 WET 文件：

```python
def process_single_wet_file(
    input_path: str, # wet文件路径
    output_path: str, # jsonl文件路径
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
```

简单介绍上面代码，最重要的是遍历WET文件中的record(只需要conversion类型就可以)，代码如下：

```python
with open(input_path, 'rb') as stream: # 从wet中读取
     for record in ArchiveIterator(stream, record_types=WarcRecordType.conversion):
```

然后对应的headers(元数据，比如url之类的)通过 `record.headers` 得到，文本内容通过 `record.reader.read().decode('utf-8', replace='True')`得到。举个具体的例子：

下面这个是 `record.headers` ，描述了这个record的元数据。

<img src="https://typora-jkd.oss-cn-hangzhou.aliyuncs.com/%20img/image-20260406112818853.png" alt="image-20260406112818853" style="zoom:67%;" />

下面这个是 `record.reader.read().decode('utf-8', replace='True')` ，

<img src="https://typora-jkd.oss-cn-hangzhou.aliyuncs.com/%20img/image-20260406112945571.png" alt="image-20260406112945571" style="zoom:67%;" />

然后使用 `concourrent.futures` 来并行处理多个wet文件:

```python
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
        with open(output_file, "r", encoding="utf-8") as rf, open(final_json_path, "a", encoding="utf-8") as wf:
            for line in rf:
                wf.write(line)
        # 将每个wet file的过滤统计结果进行累加
        accumulative_counter = accumulative_counter + Counter(stats)
        print(f"Output file written: {output_file}")
    print(f"第一个阶段(non_en)过滤了 {accumulative_counter['non_en']} 个样本\n，\
            第二个阶段(harmful_content)过滤了 {accumulative_counter['harmful_content']} 个样本\n，\
            第三个阶段(gopher规则)过滤了 {accumulative_counter['gopher_filter']} 个样本\n，\
            第四个阶段(quality classifier)过滤了 {accumulative_counter['low_quality']} 个样本。\n")

    end_time = time.perf_counter()
    print(f"Processing completed in {end_time - start_time:.2f} seconds.")
```

运行该函数，得到输出：

```txt
第一个阶段(non_en)过滤了 19042226 个样本
第二个阶段(harmful_content)过滤了 514 个样本
第三个阶段(gopher规则)过滤了 92771 个样本
第四个阶段(quality classifier)过滤了 11583 个样本。
Processing completed in 1693.15 seconds.
```

可以看到处理1000个wet文件花费了 0.47h，那么100,000个文件需要花费 47h！

注意，上面所写的只是单纯过滤的时间，还没有加上去重逻辑，但是数据处理肯定是需要去重的，所以接下来会对 得到的 `cs336_data/data/train.jsonl` 使用 MinHash + LSH 去重。而且由于之前的模糊去重函数不支持接受一个json文件，所以需要修改我们的函数。

```python
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
    # 不使用下面这个是因为会爆内存，而且后面发现工业界认为两个文本在同一个band_idx上的哈希值相同就是重复文档，不需要再进一步使用 ngram_words 来计算 jaccard相似度
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
```

最终将样本数从 116638 -> 113876。这说明我们原来的样本的重复率不高。花费了约13min。

然后完成一下 **Problem (inspect_filtered_data)**

(a)

![image-20260406153306754](https://typora-jkd.oss-cn-hangzhou.aliyuncs.com/%20img/image-20260406153306754.png)

 我看大部分都是高质量数据，比如访谈类的，新闻类的，但有些质量不高，比如文本里面有导航栏之类的内容。

(b) 

按照我的过滤规则，非英语的都被剔除了，有害内容也被剔除了，不满足gopher规则的也被剔除了以及不满足quality classifier的也被剔除了。然后个人邮件，美国电话以及ip地址都被掩码了。

(c) 无

## tokenize data

使用 `pool.imap` 方法来惰性的tokenize data。惰性是指如果传入的对象是一个迭代器对象，那么会按需所取，不会一起加载全部。

```python
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
```

最终输出：Tokenized and encoded cs336_data/data/train.dedup.jsonl into 388440257 tokens

## 开始训练

首先需要使用wandb来记录数据，但是由于项目默认的wandb版本太低，所以我们需要使用

```sh 
uv add wandb==0.25.1
```

来更新wandb。

修改好 `cs336-basics/configs/experiment/your_data.yaml` 文件以及将batch_size调至8(我是用单卡3090)，然后进入 `cs336-basics` 目录，执行：

```sh
python scripts/train.py --config-name=experiment/your_data
```

按照提示输入好自己的wandb token。

然后训练开始，可以在wandb看到我们的loss曲线：

![image-20260407174337544](https://typora-jkd.oss-cn-hangzhou.aliyuncs.com/%20img/image-20260407174337544.png)

由于我只使用了 1000 个wet文件，加上降低了batch_size以及数据预处理的时候只使用了该作业要求的那些过滤方法，没有尝试一些方法，比如使用n-gram perplexity来过滤样本。所以导致eval_loss最低是4.37。

然后我们可以使用下面的命令来进行推理：

```sh
python scripts/generate_with_gpt2_tok.py /root/assignment4-data/cs336_data/data/train
```

最后那个参数是你在 `cs336-basics/configs/experiment/your_data.yaml` 中指定的 `model_output` 的值。

看一下一个样本的推理效果：

```txt
Prefix:  Linda was on a walk in the park
----------------------------------------------------------------------------------------------------
Generated:   with her dad. Later, she said that she was not expecting to be able to move. So, she was told she would not move for a few minutes.
In the afternoon, she went on a walk. She didn’t have time to look at me. But her aunt and cousin, and I'm sure she would have been able to get out and ask for a few minutes.
And she was really happy to be able to stay with me.
So, I was in a lot of pain. But, my whole life, I would be happy to be able to do this again, because I really would like to make sure I was able to do this again.
In the afternoon, I also have a little bit of anxiety as well.
So, there was a lot of when I was in the quiet of the park. But, I didn't know that I had to give up.
And, I didn't know that I had to give up.
And, I can't tell you why.
So, I had a lot of anxiety as well.
And, I think, I had a lot of anxiety, because I really didn't know that I had to give up.
So, I had to give
```

可以看到语法方面基本没有问题，但是逻辑方面有些问题。
