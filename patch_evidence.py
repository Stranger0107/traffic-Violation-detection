import json

try:
    with open('train.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                if 'ingest_resp = db.log_violation' in line:
                    indent = line.split('ingest_resp')[0]
                    # Insert evidence saving code before log_violation
                    new_source.append(f'{indent}import cv2, os, time\n')
                    new_source.append(f'{indent}evidence_dir = r"d:/traffic_violation_detection/Traffic_violation backend/evidence"\n')
                    new_source.append(f'{indent}os.makedirs(evidence_dir, exist_ok=True)\n')
                    new_source.append(f'{indent}timestamp = int(time.time())\n')
                    new_source.append(f'{indent}filename = f"{{plate_text}}_frame_{{frame_no}}_{{timestamp}}.jpg"\n')
                    new_source.append(f'{indent}cv2.imwrite(os.path.join(evidence_dir, filename), frame)\n')
                    new_source.append(f'{indent}evidence_db_path = f"/evidence/{{filename}}"\n')
                    
                    # Replace log_violation call to include evidence_path
                    new_line = line.replace('plate_text)', 'plate_text, evidence_path=evidence_db_path)')
                    new_source.append(new_line)
                else:
                    new_source.append(line)
            cell['source'] = new_source

    with open('train.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print('Updated train.ipynb to capture and save evidence images')
except Exception as e:
    print('Error:', e)
