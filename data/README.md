## Dataset: IMDB Large Movie Review (50k)

Source: Stanford AI — Large Movie Review Dataset v1.0 (aclImdb).  
We keep the **full CSV locally** due to size limits. To reproduce:
1) Download `aclImdb_v1.tar.gz` from the Stanford page.
2) Extract and convert to a single CSV with columns `text,label` (script in repo).
3) Use our fixed split indices in `data/split_indices.json` (**seed=340**).

### Files tracked in this folder
- `IMDB_50k_sample.csv` — 500 rows for structure reference.  
  SHA256: `<paste from data/IMDB_50k_sample.csv.sha256>`
- `IMDB_50k.csv.sha256` — checksum of the full CSV generated locally.  
  SHA256: `<paste from data/IMDB_50k.csv.sha256>`
- `aclImdb_v1.tar.gz.sha256` — checksum of the original tarball.  
  SHA256: `<paste from data/aclImdb_v1.tar.gz.sha256>`
- `split_indices.json` — exact 80/10/10 split indices (**seed=340**).

