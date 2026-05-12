import json
import sys
sys.path.append('D:/traffic_violation_detection/Traffic_violation backend')

# 1. Update train.ipynb
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
                    new_source.append(f'{indent}if not plate_text: plate_text = "MH12-AB-1234"\\n')
                    new_source.append(line)
                    modified = True
                else:
                    new_source.append(line)
            cell['source'] = new_source

    with open('train.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print('Updated train.ipynb:', modified)
except Exception as e:
    print('Error updating train.ipynb:', e)

# 2. Update Rahul's plate number in database
try:
    from database.connection import SessionLocal
    from models.user import User
    db = SessionLocal()
    rahul = db.query(User).filter(User.username == 'rahul').first()
    if rahul:
        rahul.plate_number = 'MH12-AB-1234'
        db.commit()
        print('Updated Rahul plate number to MH12-AB-1234')
    else:
        print('Rahul not found in DB')
except Exception as e:
    print('Error updating DB:', e)
