from __future__ import annotations

import base64
import json
import os
from openai import OpenAI
from pathlib import Path
from pydantic import BaseModel
from typing import Any, Dict, Iterable, List, Sequence, Type, Union


class GPTBatchAnnotator:
    """
    Batch-prepares, uploads, and post-processes image labeling jobs for the OpenAI Batch API.

    This utility:
      1) Scans an input directory for image files and chunks them into .jsonl batch payloads.
      2) Uploads those batch payloads to the OpenAI Batch endpoint (`/v1/chat/completions`).
      3) Polls/report statuses for the created batches.
      4) Downloads results/error files once batches complete.
      5) Extracts model JSON outputs per image into individual `.json` files.

    The model is prompted as a vision Q/A assistant and constrained via `response_format={"type":"json_schema"}`,
    where the schema is derived from a provided Pydantic model class.

    Parameters
    ----------
    input_folder : str | os.PathLike[str]
        Directory containing input images. Non-image files are ignored.
    output_schema : Type[pydantic.BaseModel]
        A Pydantic model **class** (not an instance). Its JSON schema is used to enforce structured outputs.
    output_folder : str | os.PathLike[str], default "labels"
        Root output directory. Subfolders are created:
        - `batches/` : generated .jsonl payloads
        - `labels/`  : parsed, per-image JSON outputs
        - `results/` : raw batch result files from OpenAI
        - `errors/`  : raw batch error files from OpenAI
    model : str, default "gpt-4o-mini"
        Vision-capable model used by `/v1/chat/completions`.
    batch_size : int, default 1000
        Maximum images per batch file (.jsonl).
    batch_base_name : str, default "batch"
        Base filename used for batch payloads (e.g., `batch-1.jsonl`).

    Notes
    -----
    - Supported image extensions: .jpg, .jpeg, .png, .bmp, .gif, .webp
    - Each JSONL line posts a single `chat.completions` request with the image inlined as base64.
    - The OpenAI Batch API processes asynchronously; use `get_status()` and `retrieve_results()` to manage lifecycle.

    Example
    -------
    >>> annotator = GPTBatchAnnotator(
    ...     input_folder="images/",
    ...     output_schema=YourPydanticSchema,
    ...     output_folder="labels",
    ...     model="gpt-4o-mini",
    ...     batch_size=500,
    ... )
    >>> annotator.create_batch_files()
    >>> annotator.upload_batches()
    >>> annotator.get_status()
    >>> annotator.retrieve_results()
    >>> annotator.extract_labels()
    """

    # Attribute type annotations for static analyzers
    client: OpenAI
    model: str
    batch_size: int
    batch_base_name: str
    schema: Type[BaseModel]
    input_folder: Path
    output_folder: Path
    batches_folder: Path
    labels_folder: Path
    results_folder: Path
    errors_folder: Path

    def __init__(
            self,
            input_folder: str,
            output_schema: Type[BaseModel],
            output_folder: str = "labels",
            model: str = "gpt-4o-mini",
            batch_size: int = 1000,
            batch_base_name: str = "batch",
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.batch_base_name = batch_base_name
        self.schema = output_schema
        self.client = OpenAI()

        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.batches_folder = self.output_folder / "batches"
        self.batches_folder.mkdir(parents=True, exist_ok=True)

        self.labels_folder = self.output_folder / "labels"
        self.labels_folder.mkdir(parents=True, exist_ok=True)

        self.results_folder = self.output_folder / "results"
        self.results_folder.mkdir(parents=True, exist_ok=True)

        self.errors_folder = self.output_folder / "errors"
        self.errors_folder.mkdir(parents=True, exist_ok=True)

    def create_batch_files(self) -> None:
        """
        Create `.jsonl` batch payloads from images in `input_folder`.

        Raises
        ------
        ValueError
            If no valid image files are found in `input_folder`.
        """
        image_files: List[Path] = [
            f for f in self.input_folder.iterdir() if f.is_file() and self._is_image_file(f)
        ]
        if not image_files:
            raise ValueError("No valid image files found in the specified directory.")

        batches: List[List[Path]] = self._chunk_files(image_files, self.batch_size)

        for idx, batch in enumerate(batches, start=1):
            batch_name = f"{self.batch_base_name}-{idx}.jsonl"
            batch_path = self.batches_folder / batch_name

            with open(batch_path, "w", encoding="utf-8") as jsonl_file:
                for image_file in batch:
                    encoded_image = self._encode_image_base64(image_file)
                    entry: Dict[str, Any] = {
                        "custom_id": image_file.name,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": self.model,
                            "response_format": {
                                "type": "json_schema",
                                "json_schema": self._pydantic_to_openai_schema(self.schema),
                            },
                            "messages": [
                                {
                                    "role": "system",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": (
                                                "You are a visual Q/A assistant, helping users answers questions "
                                                "about the details of a person in an image from a security camera"
                                            ),
                                        }
                                    ],
                                },
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/{image_file.suffix[1:]};base64,{encoded_image}",
                                                "detail": "low",
                                            },
                                        }
                                    ],
                                },
                            ],
                        },
                    }
                    jsonl_file.write(json.dumps(entry) + "\n")

            # Correct: report images per *this* batch, not the whole set
            print(f"Batch file created: {batch_path} ({len(batch)} images)")

    @staticmethod
    def _encode_image_base64(image_path: PathLike) -> str:
        """Return the base64-encoded contents of an image file."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    @staticmethod
    def _is_image_file(file_path: Path) -> bool:
        """True if `file_path` has a supported image extension."""
        return file_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

    @staticmethod
    def _is_batch_file(file_path: Path) -> bool:
        """True if `file_path` looks like a batch payload (.json or .jsonl)."""
        return file_path.suffix.lower() in {".jsonl", ".json"}

    @staticmethod
    def _pydantic_to_openai_schema(pydantic_model: Type[BaseModel]) -> Dict[str, Any]:
        """
        Convert a Pydantic model class to an OpenAI-compatible `json_schema` block.

        Parameters
        ----------
        pydantic_model : Type[pydantic.BaseModel]
            Pydantic model **class** to be converted.

        Returns
        -------
        Dict[str, Any]
            Dictionary suitable for use under `response_format.json_schema`.
        """
        pydantic_schema: Dict[str, Any] = pydantic_model.model_json_schema()
        pydantic_schema["additionalProperties"] = False

        if "$defs" in pydantic_schema:
            for definition in pydantic_schema["$defs"].values():
                if isinstance(definition, dict):
                    definition["additionalProperties"] = False

        return {
            "name": pydantic_model.__name__,
            "strict": True,
            "schema": pydantic_schema,
        }

    @staticmethod
    def _chunk_files(file_list: Sequence[Path], chunk_size: int) -> List[List[Path]]:
        """
        Split a sequence of files into chunks of size `chunk_size`.

        Parameters
        ----------
        file_list : Sequence[pathlib.Path]
            Sequence of file paths.
        chunk_size : int
            Max number of files per chunk. Must be > 0.

        Returns
        -------
        List[List[pathlib.Path]]
            List of chunks (sublists).

        Raises
        ------
        ValueError
            If `chunk_size` <= 0.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        return [list(file_list[i: i + chunk_size]) for i in range(0, len(file_list), chunk_size)]

    def upload_batch(self, batch_path: PathLike) -> None:
        """
        Upload a single `.jsonl` batch payload to the OpenAI Batch API.
        """
        batch_input_file = self.client.files.create(file=open(batch_path, "rb"), purpose="batch")
        print(f"Uploading batch {batch_path}...")
        batch_input_file_id = batch_input_file.id

        self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"batch_name": f"{self._get_filename(batch_path)}"},
        )
        print(f"Batch uploaded: {batch_path}")

    def upload_batches(self) -> None:
        """
        Upload all locally prepared batch payloads found in `batches_folder`.
        """
        batches: List[Path] = [
            f for f in self.batches_folder.iterdir() if f.is_file() and self._is_batch_file(f)
        ]
        for batch in batches:
            self.upload_batch(batch)

    def get_status(self) -> None:
        """
        Print status lines for all local batch payloads if they exist in OpenAI's batch list.
        """
        local_batches: List[Path] = [
            f for f in self.batches_folder.iterdir() if f.is_file() and self._is_batch_file(f)
        ]

        open_ai_batches = {
            batch.metadata["batch_name"]: batch
            for batch in self.client.batches.list(limit=100).data
            if "batch_name" in batch.metadata
        }

        for batch_path in local_batches:
            if batch_path.name in open_ai_batches:
                batch = open_ai_batches[batch_path.name]
                print(
                    f"Id {batch.id} -- {batch.status} -- {batch.metadata} -- Outfile: {batch.output_file_id}"
                )
            else:
                print(f"Couldn't find batch {batch_path} in open ai processed batches")

    def retrieve_results(self) -> None:
        """
        Download result and error files for all local batch payloads that exist in OpenAI's batch list.
        """
        local_batches: List[Path] = [
            f for f in self.batches_folder.iterdir() if f.is_file() and self._is_batch_file(f)
        ]

        open_ai_batches = {
            batch.metadata["batch_name"]: batch
            for batch in self.client.batches.list(limit=100).data
            if "batch_name" in batch.metadata
        }

        for batch_path in local_batches:
            if batch_path.name not in open_ai_batches:
                print(f"Couldn't find batch {batch_path} in open ai processed batches")
                continue

            batch = open_ai_batches[batch_path.name]
            output_file = batch.output_file_id
            error_file = batch.error_file_id
            file_name = self._remove_extension(batch_path.name)

            if output_file:
                file_response = self.client.files.content(output_file)
                file_response.write_to_file(f"{self.results_folder / file_name}.txt")
                print(f"Saved Results for batch {batch_path}")

            if error_file:
                file_response = self.client.files.content(error_file)
                file_response.write_to_file(f"{self.errors_folder / file_name}.txt")
                print(f"Saved Error for batch {batch_path}")

    def extract_labels(self) -> None:
        """
        Parse downloaded batch result files and persist per-image JSON outputs into `labels_folder`.
        """
        results: List[Path] = [f for f in self.results_folder.iterdir() if f.is_file()]
        for result_file in results:
            self._save_parsed_content(result_file, self.labels_folder)

    @staticmethod
    def _get_filename(file_path: PathLike) -> str:
        """
        Return the filename (with extension) from a path.
        """
        return os.path.basename(file_path)

    @staticmethod
    def _remove_extension(file_name: str) -> str:
        """Drop the file extension from a filename."""
        return os.path.splitext(file_name)[0]

    @staticmethod
    def _save_parsed_content(input_file: PathLike, output_dir: PathLike) -> None:
        """
        Read a batch results file (newline-delimited JSON) and write each parsed,
        schema-conformant content blob into `output_dir/<image_basename>.json`.

        Expects each line to have:
          - `custom_id` : name of the original image file
          - `response.body.choices[0].message.content` : JSON string that matches the enforced schema
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with open(input_file, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    custom_id = data.get("custom_id")
                    if not custom_id:
                        print(f"Line {line_number}: Missing 'custom_id'. Skipping.")
                        continue

                    try:
                        content_str = data["response"]["body"]["choices"][0]["message"]["content"]
                        parsed_content = json.loads(content_str)
                    except (KeyError, json.JSONDecodeError) as e:
                        print(f"Line {line_number}: Failed to extract/parse 'content' - {e}")
                        continue

                    base_name = os.path.splitext(custom_id)[0]
                    output_filename = f"{base_name}.json"
                    (output_path / output_filename).write_text(
                        json.dumps(parsed_content, indent=2), encoding="utf-8"
                    )

                except json.JSONDecodeError as e:
                    print(f"Line {line_number}: JSON decoding failed - {e}")
                except Exception as e:
                    print(f"Line {line_number}: Unexpected error - {e}")
