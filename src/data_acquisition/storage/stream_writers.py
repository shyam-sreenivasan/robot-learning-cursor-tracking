import csv

from dataclasses import asdict, fields
from PIL import Image


class CSVWriter:

    def __init__(self, filename, contract):
        self.filename = filename
        self.contract = contract
        self.file = None
        self.writer = None

    def open(self, output_dir):

        path = output_dir / self.filename

        self.file = path.open("x", newline="")

        fieldnames = [
            field.name for field in fields(self.contract)
        ]

        self.writer = csv.DictWriter(
            self.file,
            fieldnames=fieldnames
        )

        self.writer.writeheader()

    def write(self, record):

        self.writer.writerow(asdict(record))

    def close(self):

        if self.file is not None:
            self.file.close()


class ImageWriter:

    def __init__(self):
        self.output_dir = None
        self.index_file = None
        self.index_writer = None

    def open(self, output_dir):

        self.output_dir = output_dir / "observations"
        self.output_dir.mkdir()

        self.index_file = (
            self.output_dir / "index.csv"
        ).open("x", newline="")

        self.index_writer = csv.DictWriter(
            self.index_file,
            fieldnames=[
                "timestamp",
                "frame_id",
                "filename"
            ]
        )

        self.index_writer.writeheader()

    def write(self, record):

        filename = f"frame_{record.frame_id:06d}.png"

        path = self.output_dir / filename

        with path.open("xb") as file:
            Image.fromarray(record.image).save(
                file,
                format="PNG"
            )

        self.index_writer.writerow({
            "timestamp": record.timestamp,
            "frame_id": record.frame_id,
            "filename": filename
        })

    def close(self):

        if self.index_file is not None:
            self.index_file.close()