from pathlib import Path
for p in ['data/enroll','data/db','models','src','book'] : Path(p).mkdir(parents=True,exist_ok=True)
print('face-recognition-5pt project structure created successfully.')
