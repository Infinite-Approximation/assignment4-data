from typing import List

from resiliparse.parse.encoding import detect_encoding, bytes_to_str 
from resiliparse.extract.html2text import extract_plain_text

def extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    """
    将bytes序列转化为字符串
    """
    plain_text = bytes_to_str(html_bytes, detect_encoding(html_bytes))
    text = extract_plain_text(plain_text)
    return text

def extract_warc_file(file_path: str, max_records: int = 2) -> List[str]:
    """
    返回没有html标签的response
    """
    from fastwarc.warc import ArchiveIterator, WarcRecordType
    records = []
    cur_iter = 0
    with open(file_path, 'rb') as stream:
        for record in ArchiveIterator(stream, record_types=WarcRecordType.response): # 取出record type为response的record
            status_code = record.http_headers.status_code 
            if status_code != 200:
                continue
            html_content_in_bytes = record.reader.read()
            html_content = extract_text_from_html_bytes(html_content_in_bytes)
            records.append(html_content)
            # print(html_content)
            cur_iter += 1
            if cur_iter >= max_records:
                break
    return records
                    
if __name__ == '__main__':
    file_path = 'cs336_data/data/CC-MAIN-20250417135010-20250417165010-00065.warc.gz'
    extract_warc_file(file_path, max_records=10000)