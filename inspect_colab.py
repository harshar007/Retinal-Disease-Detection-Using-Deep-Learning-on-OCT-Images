import json

with open('downloaded_colab.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    print(f"=== Cell {i} ({cell.get('cell_type')}) ===")
    print(''.join(cell.get('source', [])))
    print()
