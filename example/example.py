import json
import logging
import os
import CWN_WSD

def main():
    with open("input.json") as f:
        data = json.load(f)
    all_ans = CWN_WSD.wsd(data)
    print(all_ans)


if __name__ == "__main__":
    loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
    main()

