"""Project management for wenbi CLI harness."""
import os
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any


@dataclass
class Project:
    """Represents a wenbi processing project with input files and settings."""
    name: str = ""
    input_path: str = ""
    output_dir: str = ""
    files: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_file(self, filepath: str) -> None:
        abs_path = os.path.abspath(filepath)
        if abs_path not in self.files:
            self.files.append(abs_path)

    def remove_file(self, filepath: str) -> None:
        abs_path = os.path.abspath(filepath)
        self.files = [f for f in self.files if f != abs_path]

    def list_files(self) -> List[str]:
        return list(self.files)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Project":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def validate(self) -> List[str]:
        errors = []
        for f in self.files:
            if not os.path.exists(f):
                errors.append(f"File not found: {f}")
        if self.output_dir and not os.path.isdir(self.output_dir):
            errors.append(f"Output directory does not exist: {self.output_dir}")
        return errors

    def info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "input_path": self.input_path,
            "output_dir": self.output_dir,
            "file_count": len(self.files),
            "files": self.files,
            "metadata": self.metadata,
        }