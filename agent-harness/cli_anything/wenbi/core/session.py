"""Session management for wenbi CLI harness."""
import os
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any


@dataclass
class Session:
    """Manages current session state including project and processing history."""
    project_name: str = "default"
    output_dir: str = ""
    llm: str = "ollama/qwen3"
    lang: str = "Chinese"
    chunk_length: int = 20
    max_tokens: int = 130000
    timeout: int = 3600
    temperature: float = 0.1
    transcribe_model: str = "paraformer-zh"
    transcribe_lang: str = ""
    multi_language: bool = False
    cite_timestamps: bool = False
    keep_original_lang: bool = False
    deepl_key: str = ""
    use_deepl: bool = True
    verbose: bool = False
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def get_process_params(self) -> dict:
        """Return parameters dict suitable for process_input()."""
        return {
            "output_dir": self.output_dir,
            "llm": self.llm,
            "lang": self.lang,
            "chunk_length": self.chunk_length,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "temperature": self.temperature,
            "transcribe_model": self.transcribe_model,
            "transcribe_lang": self.transcribe_lang,
            "multi_language": self.multi_language,
            "cite_timestamps": self.cite_timestamps,
            "keep_original_lang": self.keep_original_lang,
            "use_deepl": self.use_deepl,
            "deepl_key": self.deepl_key,
            "verbose": self.verbose,
        }

    def record(self, command: str, input_path: str, output_path: str = "",
               status: str = "ok", error: str = "") -> None:
        self.history.append({
            "command": command,
            "input": input_path,
            "output": output_path,
            "status": status,
            "error": error,
        })

    def show(self) -> Dict[str, Any]:
        return {
            "project_name": self.project_name,
            "output_dir": self.output_dir or "(not set)",
            "llm": self.llm,
            "lang": self.lang,
            "chunk_length": self.chunk_length,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "temperature": self.temperature,
            "transcribe_model": self.transcribe_model,
            "transcribe_lang": self.transcribe_lang or "(auto)",
            "multi_language": self.multi_language,
            "cite_timestamps": self.cite_timestamps,
            "keep_original_lang": self.keep_original_lang,
            "use_deepl": self.use_deepl,
            "history_count": len(self.history),
        }

    def set_value(self, key: str, value: str) -> str:
        """Set a session parameter by name. Returns confirmation message."""
        known_fields = {f for f in self.__dataclass_fields__ if f != "history"}
        if key not in known_fields:
            raise ValueError(f"Unknown session parameter: {key}. Valid: {sorted(known_fields)}")

        field_type = type(getattr(self, key))
        if field_type == bool:
            setattr(self, key, value.lower() in ("true", "1", "yes"))
        elif field_type == int:
            setattr(self, key, int(value))
        elif field_type == float:
            setattr(self, key, float(value))
        else:
            setattr(self, key, value)
        return f"Set {key} = {getattr(self, key)}"

    def reset(self) -> None:
        """Reset session to defaults."""
        default = Session()
        for field_name in self.__dataclass_fields__:
            if field_name != "history":
                setattr(self, field_name, getattr(default, field_name))
        self.history = []

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Session":
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)