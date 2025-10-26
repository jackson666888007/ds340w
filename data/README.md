Source: Stanford AI — Large Movie Review Dataset v1.0 (aclImdb). 
We have kept the **full CSV file locally** due to size constraints. To create this dataset:
1) Download `aclImdb_v1.tar.gz` from the Stanford page.
2) Extract and convert it to a single CSV file with columns "text" and "label" (script in repo).
3) Use our fixed split indices found in `data/split_indices.json` (**seed=340**).


### Files tracked in this folder
- `IMDB_50k_sample.csv` — 500 - rows for reference of the structure.  
  SHA256: ``
- `IMDB_50k.csv.sha256` — checksum of the full CSV you just generated from before.  
  SHA256: ``
- `aclImdb_v1.tar.gz.sha256` — checksum of original tarball.  
  SHA256: ``
- `split_indices.json` — exact indices for the 80/10/10 split (**seed=340**).
