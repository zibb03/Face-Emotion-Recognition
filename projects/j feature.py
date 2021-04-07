import json
from collections import OrderedDict

file_data = OrderedDict()

file_data["name"] = "COMPUTER"
file_data["language"] = "kor"
file_data["words"] = {'ram':'램', 'process':'프로세스', 'processor':'프로세서', 'CPU':'씨피유'}
file_data["number"] = 4

print(json.dumps(file_data, ensure_ascii=False, indent="\t"))

with open('words.json', 'w', encoding="utf-8") as make_file:
    json.dump(file_data, make_file, ensure_ascii=False, indent="\t")