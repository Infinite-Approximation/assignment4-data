import re


def mask_emails(text: str) -> tuple[str, int]:
    pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9-]+(\.[a-zA-Z0-9-]+)*\.[a-zA-Z]+'
    result_str, count = re.subn(pattern, "|||EMAIL_ADDRESS|||", text)
    return (result_str, count)

def mask_phone_numbers(text: str) -> tuple[str, int]:
    pattern = r'(\(\d{3}\)|\d{3})[.\- ]?\d{3}[.\- ]?\d{4}'
    result_str, count = re.subn(pattern, "|||PHONE_NUMBER|||", text)
    return (result_str, count)

def mask_ips(text: str) -> tuple[str, int]:
    pattern = r'(25[0-9]|2[0-4]\d|[01]?\d\d?\.){3}(25[0-9]|2[0-4]\d|[01]?\d\d?)'
    result_str, count = re.subn(pattern, "|||IP_ADDRESS|||", text)
    return (result_str, count)

def mask_pii(text: str) -> str:
    text = mask_emails(text)[0]
    text = mask_phone_numbers(text)[0]
    text = mask_ips(text)[0]
    return text

def mask_pii_in_warc_file(file_path: str, max_records: int = 200):
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
    # file_path = 'CC-MAIN-20250417135010-20250417165010-00065.warc.gz'
    # mask_pii_in_warc_file(file_path=file_path)

    print(mask_pii(text="My phone number is 122-332-2345, My email is 2310572998jin@gmail.com"))