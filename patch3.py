import json

try:
    with open('train.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                if 'if plate_text:' in line:
                    indent = line.split('if plate_text:')[0]
                    new_source.append(f'{indent}if not plate_text: plate_text = "UNKNOWN"\n')
                    new_source.append(f'{indent}if True:\n')
                    continue
                new_source.append(line)
            cell['source'] = new_source
    with open('train.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print('Updated train.ipynb to send UNKNOWN plates to DB')
except Exception as e:
    print('Error:', e)
