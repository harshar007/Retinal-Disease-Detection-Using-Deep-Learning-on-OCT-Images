import pyarrow.parquet as pq
import pandas as pd
from huggingface_hub import hf_hub_download
import io
from PIL import Image

path = hf_hub_download(repo_id='zacharielegault/Kermany2017-OCT', filename='data/test-00000-of-00001.parquet', repo_type='dataset')
table = pq.read_table(path)
print("Schema:")
print(table.schema)
df = table.select(['label']).to_pandas()
print("\nClass distribution in test set:")
print(df['label'].value_counts())

# Read first image
row0 = table.slice(0, 1).to_pydict()
img_bytes = row0['image'][0]['bytes']
img = Image.open(io.BytesIO(img_bytes))
print(f"\nImage sample 0: format={img.format}, size={img.size}, mode={img.mode}")
