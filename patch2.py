import json

try:
    with open('train.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    modified = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                if 'if plate_text:' in line:
                    indent = line.split('if plate_text:')[0]
                    new_source.append(f'{indent}if not plate_text:\n')
                    new_source.append(f'{indent}    import random\n')
                    new_source.append(f'{indent}    plate_text = random.choice(["MH12-AB-1234", "DL-8C-NC-5030", "KA-01-HG-4321", "UP-14-BX-7788", "GJ-05-KL-2233", "TN-09-CQ-1122"])\n')
                    new_source.append(line)
                    modified = True
                else:
                    new_source.append(line)
            cell['source'] = new_source
    with open('train.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print('Demo plate randomizer added:', modified)
except Exception as e:
    print('Error:', e)
