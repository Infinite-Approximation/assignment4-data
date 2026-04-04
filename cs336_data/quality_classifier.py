import subprocess
from typing import Any

def sample_positive_urls(
        file_path: str = 'cs336_data/data/enwiki-20240420-extracted_urls.txt', 
        max_urls: int = 1000000,
        output_file_path: str = 'cs336_data/data/subsampled_positive_urls.txt'
    ):
    """
    使用 shuf -n {max_urls} {file_path} > {output_file_path} 采样url
    然后使用 
    wget –-timeout=5 \
        --tries=1 \
        --max-redirect=1 \
        --connect-timeout=2 \
        --read-timeout=2 \
        --user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36" \
        -i cs336_data/data/subsampled_positive_urls.txt \
        --warc-file=cs336_data/data/subsampled_positive_urls.warc \
        -O /dev/null
    来将url的内容转化为warc文件
    """
    sample_command = f"shuf -n {max_urls} {file_path} > {output_file_path}"
    subprocess.run(sample_command, shell=True, check=True)
    download_commmand = f'wget –-timeout=5 \
                            --user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36" \
                            -i {output_file_path} \
                            --warc-file=cs336_data/data/subsampled_positive_urls.warc \
                            -O /dev/null'
    subprocess.run(download_commmand, shell=True, check=True)

def train_quality_classification_model():
    ...

def classify_quality(text: str) -> tuple[Any, float]:
    ...