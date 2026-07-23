# Text re-ingest (20260723 · full)

- skip_faiss: true (ColQwen2 FAISS untouched)
- replace_text: true (TRUNCATE chunks first)
- table_context: 1
- table_summary: on
- max_pages: all

## Counts
```
chunks_before 21
type table 2
type text 19
section_path_filled 0
table_summary_filled 2
---
chunks_after 8835
type table 2305
type text 6530
section_path_filled 0
prev_filled 3796
table_summary_filled 2305
samples:
  ('table', '', 'The table outlines a contents page for a document. It includes two columns: "Cha')
  ('table', '', 'The table outlines personnel and their locations within an aircraft refueling op')
  ('table', '', 'The table outlines changes to a document with 12 pages related to an Aircraft Fi')
```

## Next

1. Full_zerank2 100q vs Boot-CP Arm-A (same 100q protocol)
2. Optional: enable expand/boost arms
3. E2E / table subset
