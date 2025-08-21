# GPTBatchAnnotator

Generate OpenAI **Batch API** payloads from a folder of images, upload them, pull results, and split per-image JSON that matches your **Pydantic v2** schema.

## Install

```bash
pip install "openai>=1.0.0" "pydantic>=2.0.0"
export OPENAI_API_KEY="sk-…"
```

## Quick start

```python
# schemas.py
from pydantic import BaseModel
class PersonAttributes(BaseModel):
    has_face: bool
    estimated_age: int | None
    clothing_color: str | None
```

```python
# run.py
from gpt_batch_annotator import GPTBatchAnnotator
from schemas import PersonAttributes

ann = GPTBatchAnnotator("images/", PersonAttributes, output_folder="labels", model="gpt-4o-mini", batch_size=500)
ann.create_batch_files()   # writes labels/batches/*.jsonl
ann.upload_batches()       # submits to OpenAI
# …later…
ann.get_status()
ann.retrieve_results()     # writes results/, errors/
ann.extract_labels()       # writes labels/labels/<image>.json
```

## What to verify (don’t skip)

* **Endpoint & payload**: Using `/v1/chat/completions` with image content and `response_format={"type":"json_schema"}`. Confirm this matches your SDK’s current contract.
* **Size limits**: Base64 inlines are large; check per-request and per-file caps. If tight, serve images via URLs instead.
* **Pydantic**: Code uses `model_json_schema()` (v2). v1 will break.

## Notes

* Images: `.jpg .jpeg .png .bmp .gif .webp`
* Output tree under `output_folder`: `batches/`, `results/`, `errors/`, `labels/`.

## License

MIT. Validate assumptions before production.
