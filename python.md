# Python IDE Rules

## Core Principles

- **Functions over classes**: Use functions for logic, classes only for types (StrEnum, Pydantic BaseModel)
- **Type everything**: All inputs and outputs must have type hints
- **Modern typing**: Use `str | None` instead of `Optional[str]`, `dict`/`list` instead of `Dict`/`List`
- **Test-driven**: Use pytest for all testing
- **Performance-first**: Use Polars (not pandas), async patterns, lazy evaluation
- **Logging**: Use extensive logging and rich logging
- **Documentation**: Use docstrings for documentation
- **Error Handling**: Use try/except/finally for error handling
- **Match Statements**: Use match statements for pattern matching
- **Configuration**: Use Pydantic BaseModel for configuration, avoid YAML, JSON, INI
- **Config file**: Use TOML if config file required

## Type System

### Fixed Choices - Use StrEnum or Literal
```python
from enum import StrEnum
from typing import Literal

class Status(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"

ProcessingMode = Literal["batch", "stream"]
```

### Complex Types - Use Pydantic BaseModel
```python
from pydantic import BaseModel, Field
from datetime import datetime

class ProcessingConfig(BaseModel):
    batch_size: int = Field(100, gt=0, le=10000)
    timeout_seconds: float = Field(30.0, gt=0)
    
    @classmethod
    def from_env(cls) -> "ProcessingConfig":
        return cls(
            batch_size=int(os.getenv("BATCH_SIZE", 100)),
            timeout_seconds=float(os.getenv("TIMEOUT", 30.0))
        )
```

### Function Typing
```python
from collections.abc import Callable, Sequence

def process_data(
    data: Sequence[str],
    config: ProcessingConfig,
    callback: Callable[[str], None] | None = None,
) -> dict[str, int]:
    results: dict[str, int] = {}
    return results
```

## Testing with pytest

```python
import pytest

def test_empty_input_returns_empty_dict():
    config = ProcessingConfig()
    result = process_data([], config)
    assert result == {}

@pytest.mark.parametrize("batch_size,expected", [
    (1, 1), (100, 100), (1000, 1000)
])
def test_batch_size_respected(batch_size, expected):
    config = ProcessingConfig(batch_size=batch_size)
    assert config.batch_size == expected

@pytest.fixture
def sample_config():
    return ProcessingConfig(batch_size=10, timeout_seconds=5.0)
```

## Data Processing with Polars

**Always use Polars, never pandas. Use lazy evaluation.**

```python
import polars as pl

def transform_data(input_path: str, output_path: str) -> pl.LazyFrame:
    return (
        pl.scan_parquet(input_path)
        .with_columns([
            pl.col("amount").cast(pl.Float64),
            pl.col("date").str.to_date()
        ])
        .filter(pl.col("amount") > 0)
        .group_by("category")
        .agg([
            pl.col("amount").sum().alias("total"),
            pl.col("amount").count().alias("count")
        ])
        .sink_parquet(output_path)
    )
```

## Visualization with Altair

**Use Altair directly with Polars - never convert to pandas.**

```python
import altair as alt
import polars as pl

def create_chart(data: pl.DataFrame) -> alt.Chart:
    return (
        alt.Chart(data)
        .mark_point()
        .encode(x="x:Q", y="y:Q", color="category:N")
        .properties(title="Data Visualization")
    )
```

## Async Programming

```python
import asyncio
import aiohttp
from typing import AsyncGenerator

async def fetch_data_batch(urls: list[str]) -> list[dict]:
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_single(session, url) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=True)

async def process_stream(
    data_stream: AsyncGenerator[str, None]
) -> AsyncGenerator[dict, None]:
    async for item in data_stream:
        yield await process_item_async(item)
```

## Configuration Management

**Use Pydantic BaseModel with classmethods, not BaseSettings.**

```python
import os
from pydantic import BaseModel, Field

class AppConfig(BaseModel):
    debug: bool = Field(False)
    worker_count: int = Field(4, gt=0)
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            debug=os.getenv("DEBUG", "false").lower() == "true",
            worker_count=int(os.getenv("WORKERS", "4"))
        )
    
    @classmethod
    def from_file(cls, path: str) -> "AppConfig":
        import json
        with open(path) as f:
            return cls(**json.load(f))
```

## FastAPI Best Practices

```python
from fastapi import FastAPI, Depends, HTTPException
from typing import Annotated

app = FastAPI(title="Enterprise API")

class DataService:
    def __init__(self, config: AppConfig):
        self.config = config
    
    async def get_data(self, id: str) -> dict:
        return {"id": id, "data": "example"}

def get_data_service() -> DataService:
    return DataService(AppConfig.from_env())

@app.post("/process")
async def process_endpoint(
    request: ProcessRequest,
    service: Annotated[DataService, Depends(get_data_service)]
) -> ProcessResponse:
    try:
        result = process_data(request.items, request.config)
        return ProcessResponse(result=result, status="success")
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
```

## CLI with Typer

**Use Annotated types and prefer Options over Arguments.**

```python
import typer
from pathlib import Path
from typing import Annotated
from rich.console import Console

app = typer.Typer()
console = Console()

@app.command()
def process(
    input_file: Annotated[Path, typer.Option(
        ..., "--input", "-i", exists=True, dir_okay=False
    )],
    output_file: Annotated[Path, typer.Option(..., "--output", "-o")],
    batch_size: Annotated[int, typer.Option(100, "--batch-size", min=1, max=10000)],
    verbose: Annotated[bool, typer.Option(False, "--verbose", "-v")]
) -> None:
    try:
        config = ProcessingConfig(batch_size=batch_size)
        result = process_file(input_file, output_file, config)
        console.print(f"✅ Processed {len(result)} items", style="green")
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")
        raise typer.Exit(1)
```

## Documentation

```python
def complex_function(
    data: list[dict[str, Any]], 
    config: ProcessingConfig
) -> tuple[list[dict], ProcessingStats]:
    """Process complex data with configuration.
    
    Args:
        data: List of data dictionaries to process
        config: Processing configuration
        
    Returns:
        Tuple of (processed_data, processing_stats)
        
    Raises:
        ValidationError: If data format is invalid
        ProcessingError: If processing fails
    """
    pass