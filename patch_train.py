import nbformat

with open('train.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == 'code':
        source = cell.source
        if 'cv2.imwrite(os.path.join(evidence_dir, filename), frame)' in source:
            # We want to draw a bounding box before saving
            replacement1 = """
                        import cv2, os, time
                        evidence_dir = r"d:/traffic_violation_detection/Traffic_violation backend/evidence"
                        os.makedirs(evidence_dir, exist_ok=True)
                        timestamp = int(time.time())
                        filename = f"{plate_text}_frame_{frame_no}_{timestamp}.jpg"
                        
                        # Draw bounding box on a copy of the frame for evidence
                        evidence_img = frame.copy()
                        cv2.rectangle(evidence_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 3)
                        cv2.putText(evidence_img, f"VIOLATION: {label}", (int(box[0]), max(10, int(box[1])-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        cv2.imwrite(os.path.join(evidence_dir, filename), evidence_img)
                        evidence_db_path = f"/evidence/{filename}"
                        ingest_resp = db.log_violation(frame_no, label, plate_text, evidence_path=evidence_db_path)
"""
            
            # The original code looks like this:
            original1 = """
                        import cv2, os, time
                        evidence_dir = r"d:/traffic_violation_detection/Traffic_violation backend/evidence"
                        os.makedirs(evidence_dir, exist_ok=True)
                        timestamp = int(time.time())
                        filename = f"{plate_text}_frame_{frame_no}_{timestamp}.jpg"
                        cv2.imwrite(os.path.join(evidence_dir, filename), frame)
                        evidence_db_path = f"/evidence/{filename}"
                        ingest_resp = db.log_violation(frame_no, label, plate_text, evidence_path=evidence_db_path)
"""
            source = source.replace(original1.strip(), replacement1.strip())
            
            # Also do it for RED_LIGHT
            original_red = """
                    import cv2, os, time
                    evidence_dir = r"d:/traffic_violation_detection/Traffic_violation backend/evidence"
                    os.makedirs(evidence_dir, exist_ok=True)
                    timestamp = int(time.time())
                    filename = f"{plate_text}_frame_{frame_no}_{timestamp}.jpg"
                    cv2.imwrite(os.path.join(evidence_dir, filename), frame)
                    evidence_db_path = f"/evidence/{filename}"
                    ingest_resp = db.log_violation(frame_no, "RED_LIGHT", plate_text, evidence_path=evidence_db_path)
"""
            
            replacement_red = """
                    import cv2, os, time
                    evidence_dir = r"d:/traffic_violation_detection/Traffic_violation backend/evidence"
                    os.makedirs(evidence_dir, exist_ok=True)
                    timestamp = int(time.time())
                    filename = f"{plate_text}_frame_{frame_no}_{timestamp}.jpg"
                    
                    evidence_img = frame.copy()
                    if vbox is not None:
                        cv2.rectangle(evidence_img, (int(vbox[0]), int(vbox[1])), (int(vbox[2]), int(vbox[3])), (0, 0, 255), 3)
                        cv2.putText(evidence_img, "VIOLATION: RED_LIGHT", (int(vbox[0]), max(10, int(vbox[1])-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    cv2.imwrite(os.path.join(evidence_dir, filename), evidence_img)
                    evidence_db_path = f"/evidence/{filename}"
                    ingest_resp = db.log_violation(frame_no, "RED_LIGHT", plate_text, evidence_path=evidence_db_path)
"""
            source = source.replace(original_red.strip(), replacement_red.strip())
            cell.source = source

with open('train.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print("Notebook patched successfully!")
