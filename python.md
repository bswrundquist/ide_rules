## Core Principles

- **Functions over classes**: Write code in functions, not classes (unless defining types or required by frameworks)
- **Type safety first**: Use comprehensive typing as the first line of defense
- **Test-driven development**: Test all code components thoroughly with high coverage
- **Automation**: Use Make targets and CI/CD for all common operations
- **Performance awareness**: Write efficient, scalable code

## Code Organization

### Functions First

Functions should be the primary way to organize code logic. This makes code:

- Easier to test and mock
- More maintainable and debuggable
- More composable and reusable
- Simpler to reason about and review

Only use classes when:

- Defining types (Enums, Pydantic models, dataclasses)
- Required by a specific library/framework
- Implementing design patterns that require state management

**What this accomplishes:**

- Promotes functional programming principles
- Improves code testability and maintainability
- Reduces complexity and cognitive load
- Enables better code reuse and composition

## Type System

### Enums for Fixed Choices

```python
from enum import StrEnum


class DataSource(StrEnum):
    """Data sources we can process."""
    CLOUD_STORAGE = "cloud_storage"
    LOCAL_FILE = "local_file"
    DATABASE = "database"
```

### Pydantic Models for Complex Types

```python
from pydantic import BaseModel, Field, validator
from datetime import datetime


class ProcessingConfig(BaseModel):
    """Configuration for data processing."""
    batch_size: int = Field(100, gt=0, le=10000, description="Number of items to process at once")
    max_retries: int = Field(3, ge=0, le=10, description="Maximum number of retry attempts")
    timeout_seconds: float = Field(30.0, gt=0, le=300, description="Operation timeout in seconds")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @validator('batch_size')
    def validate_batch_size(cls, v):
        if v > 1000:
            raise ValueError('Batch size should not exceed 1000 for optimal performance')
        return v
```

### Type Guidelines

All code must use comprehensive type hints:

```python
from collections.abc import Callable, Sequence


def process_data(
    data: Sequence[str],
    config: ProcessingConfig,
    source: DataSource,
    callback: Callable[[str], None] | None = None,
) -> dict[str, int | float]:
    """Process data according to configuration.

    Args:
        data: Sequence of strings to process
        config: Processing configuration
        source: Source of the data
        callback: Optional callback function for progress updates

    Returns:
        Dictionary mapping data items to their processed values

    Raises:
        ProcessingError: If data processing fails
        ValidationError: If input data is invalid
    """
    results: dict[str, int | float] = {}

    for item in data:
        if callback:
            callback(f"Processing {item}")
        # ... implementation

    return results
```

**What this accomplishes:**

- Provides compile-time type checking and IDE support
- Documents function signatures and expected types
- Catches type-related bugs early in development
- Improves code readability and maintainability

### Test Organization

```python
import pytest
from unittest.mock import Mock, patch
from your_package.core import process_data
from your_package.types import ProcessingConfig, DataSource


class TestProcessData:
    """Test suite for process_data function."""

    def test_process_data_handles_empty_input(self):
        """Should return empty dict for empty input."""
        # Arrange
        config = ProcessingConfig()
        source = DataSource.LOCAL_FILE

        # Act
        result = process_data([], config, source)

        # Assert
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_process_data_counts_occurrences(self):
        """Should count occurrences of each item."""
        # Arrange
        config = ProcessingConfig(batch_size=2)
        source = DataSource.LOCAL_FILE
        data = ["a", "b", "a"]

        # Act
        result = process_data(data, config, source)

        # Assert
        assert result == {"a": 2, "b": 1}

    @patch('your_package.core.external_service')
    def test_process_data_handles_external_failure(self, mock_service):
        """Should handle external service failures gracefully."""
        # Arrange
        mock_service.side_effect = ConnectionError("Service unavailable")
        config = ProcessingConfig(max_retries=1)

        # Act & Assert
        with pytest.raises(ProcessingError):
            process_data(["test"], config, DataSource.API_ENDPOINT)
```

### Test Coverage Requirements

- **100% coverage for critical business logic**
- **Integration tests for external dependencies**
- **End-to-end tests for user workflows**

**What this accomplishes:**

- Ensures code reliability through comprehensive testing
- Catches regressions early in development cycle
- Provides confidence for refactoring and changes
- Documents expected behavior through test cases

## Error Handling & Logging

### Custom Exceptions

```python
class ProcessingError(Exception):
    """Raised when data processing fails."""

    def __init__(self, message: str, error_code: str, context: dict | None = None):
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}


class ValidationError(ProcessingError):
    """Raised when input validation fails."""
    pass
```

### Logging Standards

```python
import logging
import json
from datetime import datetime
from typing import Any


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


def setup_logging(level: str = "INFO") -> None:
    """Setup JSON logging configuration."""
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Add console handler with JSON formatter
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JSONFormatter())
    logger.addHandler(console_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)


def process_with_logging(data: list[str]) -> dict[str, Any]:
    """Example of proper logging in functions."""
    logger = get_logger(__name__)
    logger.info("Starting data processing", extra={"extra_fields": {"count": len(data)}})

    try:
        # Process data
        result = process_data(data)
        logger.info("Processing completed successfully", extra={"extra_fields": {"result_count": len(result)}})
        return result
    except ProcessingError as e:
        logger.error(
            "Processing failed",
            extra={
                "extra_fields": {
                    "error_code": e.error_code,
                    "context": e.context
                }
            },
            exc_info=True
        )
        raise
```

**What this accomplishes:**

- Provides structured, machine-readable logs for monitoring and debugging
- Ensures consistent log format across the application
- Enables easy log aggregation and analysis
- Maintains traceability for debugging production issues

## Configuration Management

### Environment-Based Configuration

```python
import os
from pydantic import BaseSettings, Field


class DatabaseConfig(BaseModel):
    """Database connection configuration."""
    host: str = Field(..., description="Database host")
    port: int = Field(5432, description="Database port")
    database: str = Field(..., description="Database name")
    username: str = Field(..., description="Database username")
    password: str = Field(..., description="Database password")


class AppConfig(BaseSettings):
    """Application configuration loaded from environment."""

    # Application settings
    app_name: str = Field("MyApp", description="Application name")
    debug: bool = Field(False, description="Debug mode")
    log_level: str = Field("INFO", description="Logging level")

    # External services
    api_key: str = Field(..., description="API key for external service")
    api_endpoint: str = Field(..., description="External API endpoint")

    # Database
    database: DatabaseConfig

    # Performance
    worker_count: int = Field(4, gt=0, description="Number of worker processes")
    max_connections: int = Field(100, gt=0, description="Maximum database connections")

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
        case_sensitive = False

    @classmethod
    def load(cls) -> "AppConfig":
        """Load configuration with validation."""
        try:
            return cls()
        except ValidationError as e:
            logger.error("Configuration validation failed", errors=e.errors())
            raise
```

**What this accomplishes:**

- Centralizes configuration management with type safety
- Validates configuration at startup to catch errors early
- Supports environment-based deployment configurations
- Provides clear documentation of required settings

## Performance & Scalability

### Async Programming

```python
import asyncio
from typing import AsyncGenerator
import aiohttp


async def fetch_data_batch(
    urls: list[str],
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore
) -> list[dict]:
    """Fetch data from multiple URLs concurrently with rate limiting."""

    async def fetch_single(url: str) -> dict:
        async with semaphore:  # Rate limiting
            async with session.get(url) as response:
                return await response.json()

    tasks = [fetch_single(url) for url in urls]
    return await asyncio.gather(*tasks, return_exceptions=True)


async def process_stream(data_stream: AsyncGenerator[str, None]) -> AsyncGenerator[dict, None]:
    """Process streaming data efficiently."""
    async for batch in batch_generator(data_stream, batch_size=100):
        processed = await process_batch_async(batch)
        for item in processed:
            yield item
```

### DataFrame Manipulation with Polars

- Use Polars for data processing
- Use lazy evaluation for performance
- Keep everything in a single pipeline unless it makes sense to split it

```python
import polars as pl
from datetime import date


def transform_data(
    input_path: str,
    output_path: str,
    filter_date: date | None = None
) -> pl.LazyFrame:
    """Transform data using Polars lazy evaluation."""

    query = (
        pl.scan_parquet(input_path)
        .with_columns([
            pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"),
            pl.col("amount").cast(pl.Float64),
        ])
    )

    if filter_date:
        query = query.filter(pl.col("timestamp").dt.date() >= filter_date)

    result = (
        query
        .group_by("category")
        .agg([
            pl.col("amount").sum().alias("total_amount"),
            pl.col("amount").mean().alias("avg_amount"),
            pl.col("amount").count().alias("transaction_count"),
        ])
        .sort("total_amount", descending=True)
    )

    # Execute and save
    result.sink_parquet(output_path)
    return result
```

### Use Altair for plotting

- Default to Altair for all static and interactive plotting
- Use the Polars package directly (**do not use .to_pandas()**)
- Make good titles, give descriptive names to axis
- Put plots together when it makes sense

```python
# Example Altair plot
import altair as alt
import polars as pl

data = pl.DataFrame({"x": [1, 2, 3, 4, 5], "y": [1, 4, 9, 16, 25]})

chart = alt.Chart(data).mark_point().encode(x="x", y="y")

chart.save("example_plot.png")
```

**What this accomplishes:**

- Enables high-performance concurrent operations
- Optimizes memory usage through streaming and lazy evaluation
- Provides scalable data processing capabilities
- Maintains code readability while achieving performance goals

## API Development

### FastAPI Best Practices with Dependency Injection

```python
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from pydantic import BaseModel
import logging
from typing import Annotated
from functools import lru_cache

app = FastAPI(
    title="Enterprise API",
    description="Production-ready API with proper error handling",
    version="1.0.0",
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

security = HTTPBearer()
logger = logging.getLogger(__name__)


# Dependency Injection
class DatabaseService:
    """Database service for dependency injection."""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string

    async def get_data(self, id: str) -> dict:
        """Get data from database."""
        # Implementation here
        return {"id": id, "data": "example"}


class ExternalAPIService:
    """External API service for dependency injection."""

    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url

    async def call_external_api(self, endpoint: str) -> dict:
        """Call external API."""
        # Implementation here
        return {"result": "success"}


@lru_cache()
def get_database_service() -> DatabaseService:
    """Get database service singleton."""
    return DatabaseService("postgresql://localhost/db")


@lru_cache()
def get_external_api_service() -> ExternalAPIService:
    """Get external API service singleton."""
    return ExternalAPIService("api_key", "https://api.example.com")


class ProcessRequest(BaseModel):
    """Request model for data processing."""
    data: list[str]
    config: ProcessingConfig


class ProcessResponse(BaseModel):
    """Response model for data processing."""
    results: dict[str, int | float]
    processing_time: float
    status: str


@app.post(
    "/process",
    response_model=ProcessResponse,
    status_code=status.HTTP_200_OK,
    summary="Process data",
    description="Process input data according to configuration"
)
async def process_data_endpoint(
    request: ProcessRequest,
    token: Annotated[str, Depends(security)],
    db_service: Annotated[DatabaseService, Depends(get_database_service)],
    api_service: Annotated[ExternalAPIService, Depends(get_external_api_service)]
) -> ProcessResponse:
    """Process data with proper error handling and logging."""

    start_time = time.time()
    request_id = str(uuid.uuid4())

    logger.info(
        "Processing request started",
        extra={"extra_fields": {
            "request_id": request_id,
            "data_count": len(request.data)
        }}
    )

    try:
        # Validate authentication
        validate_token(token.credentials)

        # Use injected services
        db_data = await db_service.get_data("example_id")
        api_result = await api_service.call_external_api("/process")

        # Process data
        results = await process_data_async(request.data, request.config)

        processing_time = time.time() - start_time

        logger.info(
            "Processing completed successfully",
            extra={"extra_fields": {
                "request_id": request_id,
                "processing_time": processing_time
            }}
        )

        return ProcessResponse(
            results=results,
            processing_time=processing_time,
            status="success"
        )

    except ValidationError as e:
        logger.warning("Validation error", extra={"extra_fields": {"request_id": request_id, "error": str(e)}})
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation error: {e}"
        )
    except ProcessingError as e:
        logger.error("Processing error", extra={"extra_fields": {"request_id": request_id, "error": str(e)}})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal processing error"
        )
```

**What this accomplishes:**

- Provides clean separation of concerns through dependency injection
- Enables easy testing and mocking of dependencies
- Improves code maintainability and testability
- Ensures consistent error handling and logging across endpoints

## CLI Development

### Typer Best Practices

- Use Annotated types for better type hints and IDE support
- Prefer Options over Arguments for most parameters
- Add validation to CLI parameters (min/max values, file existence)
- Use Rich for beautiful terminal output and progress indicators

```python
import typer
from pathlib import Path
from typing import Annotated
from rich.console import Console
from rich.progress import Progress

app = typer.Typer(help="Enterprise CLI application")
console = Console()


@app.command()
def process(
    input_file: Annotated[Path, typer.Option(
        ...,
        "--input", "-i",
        help="Input file path",
        exists=True,
        dir_okay=False,
        readable=True
    )],
    output_file: Annotated[Path, typer.Option(
        ...,
        "--output", "-o",
        help="Output file path",
        dir_okay=False,
        writable=True
    )],
    batch_size: Annotated[int, typer.Option(
        100,
        "--batch-size", "-b",
        help="Processing batch size",
        min=1,
        max=10000
    )],
    max_retries: Annotated[int, typer.Option(
        3,
        "--max-retries", "-r",
        help="Maximum retry attempts",
        min=0,
        max=10
    )],
    verbose: Annotated[bool, typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose logging"
    )],
    config_file: Annotated[Path | None, typer.Option(
        None,
        "--config", "-c",
        help="Configuration file",
        exists=True,
        dir_okay=False,
        readable=True
    )],
) -> None:
    """Process data from input file to output file."""

    if verbose:
        console.print(f"Processing {input_file} -> {output_file}")

    try:
        config = ProcessingConfig(
            batch_size=batch_size,
            max_retries=max_retries
        )

        if config_file:
            config = ProcessingConfig.parse_file(config_file)

        with Progress() as progress:
            task = progress.add_task("Processing...", total=100)

            # Process with progress updates
            result = process_file_with_progress(
                input_file, output_file, config, progress, task
            )

        console.print(f"✅ Processing completed: {len(result)} items processed")

    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
```

**What this accomplishes:**

- Provides user-friendly command-line interfaces with rich output
- Ensures proper parameter validation and error handling
- Delivers excellent developer experience with type hints
- Enables easy integration with automation and CI/CD pipelines

## Security Standards

### Input Validation

```python
from pydantic import validator, Field
import re
from typing import Pattern

EMAIL_REGEX: Pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
SAFE_STRING_REGEX: Pattern = re.compile(r'^[a-zA-Z0-9\s\-_.]+$')


class UserInput(BaseModel):
    """Validated user input model."""

    email: str = Field(..., description="User email address")
    name: str = Field(..., min_length=1, max_length=100, description="User name")
    description: str | None = Field(None, max_length=1000, description="Description")

    @validator('email')
    def validate_email(cls, v):
        if not EMAIL_REGEX.match(v):
            raise ValueError('Invalid email format')
        return v.lower()

    @validator('name', 'description')
    def validate_safe_string(cls, v):
        if v and not SAFE_STRING_REGEX.match(v):
            raise ValueError('Contains invalid characters')
        return v
```

**What this accomplishes:**

- Prevents injection attacks and malicious input
- Ensures data integrity and validation
- Provides clear error messages for invalid input
- Maintains security best practices throughout the application

## Documentation Standards

### Docstring Format

```python
def complex_function(
    data: list[dict[str, Any]],
    config: ProcessingConfig,
    callback: Callable[[int], None] | None = None,
) -> tuple[list[dict[str, Any]], ProcessingStats]:
    """Process complex data with detailed documentation.

    This function processes a list of data dictionaries according to the provided
    configuration. It supports progress callbacks and returns both processed data
    and processing statistics.

    Args:
        data: List of dictionaries containing the data to process. Each dictionary
            must contain at least 'id' and 'value' keys.
        config: Processing configuration object that defines batch size, timeouts,
            and other processing parameters.
        callback: Optional callback function that receives the current progress
            as an integer percentage (0-100).

    Returns:
        A tuple containing:
        - List of processed data dictionaries with additional 'processed_at' field
        - ProcessingStats object with timing and error information

    Raises:
        ValidationError: If input data format is invalid or missing required fields.
        ProcessingError: If processing fails due to external service errors.
        TimeoutError: If processing exceeds the configured timeout.

    Example:
        >>> config = ProcessingConfig(batch_size=10, timeout_seconds=30)
        >>> data = [{'id': 1, 'value': 'test'}]
        >>> processed, stats = complex_function(data, config)
        >>> print(f"Processed {len(processed)} items in {stats.duration}s")

    Note:
        This function is CPU-intensive for large datasets. Consider using
        the async version `complex_function_async` for better performance.
    """
    # Implementation here
    pass
```

**What this accomplishes:**

- Provides comprehensive documentation for complex functions
- Includes usage examples and edge case handling
- Documents all parameters, return values, and exceptions
- Enables better code understanding and maintenance
