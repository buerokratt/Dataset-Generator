import os
import re
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Iterator, Union

from src.utils.logger import logger, setup_logger

setup_logger("synthetic-data-service", "INFO")


class DataSource:
    """
    Represents a single data source (file or content) with associated metadata.

    This class provides a unified interface for accessing file content and metadata,
    implementing lazy loading to improve performance when handling multiple files.
    It serves as the fundamental unit of data representation in the dataset generation
    pipeline.

    Attributes:
        path (str): Path to the data source file or identifier
        metadata (Dict[str, Any]): Dictionary containing metadata about the source
        _content (str): Private attribute holding the loaded content
        _content_loaded (bool): Flag indicating whether content has been loaded

    Properties:
        content (str): The lazily-loaded content of the data source
        name (str): The filename without extension (derived from path)
        extension (str): The file extension without the dot
    """

    def __init__(self, path: str, metadata: Dict[str, Any] = None, content: str = None):
        """
        Initialize a data source

        Args:
            path: Path to the data source file or identifier
            metadata: Additional metadata about the source
            content: Optional pre-loaded content
        """
        self.path = path
        self.metadata = metadata or {}
        self._content = content
        self._content_loaded = content is not None

    @property
    def content(self) -> str:
        """Lazy-load and return the content of the data source"""
        if not self._content_loaded:
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._content = f.read()
                self._content_loaded = True
                self.metadata["content_size_bytes"] = len(self._content)
            except Exception as e:
                logger.error(f"Error loading content from {self.path}: {e}")
                self._content = ""

        return self._content

    @property
    def name(self) -> str:
        """Return the name of the data source (derived from path)"""
        return Path(self.path).stem

    @property
    def extension(self) -> str:
        """Return the extension of the data source file (without dot)"""
        return Path(self.path).suffix.lstrip(".")

    def __repr__(self) -> str:
        return f"DataSource(path='{self.path}', name='{self.name}', size={self.metadata.get('content_size_bytes', 'unknown')})"


class DataSourceFilter:
    """
    Filter for data sources based on various criteria.

    This class provides a flexible filtering mechanism for DataSource objects based on
    file extensions, path patterns, file sizes, and custom filter functions. It is used
    by traversal strategies to filter out irrelevant data sources during directory traversal.

    Attributes:
        extensions (List[str]): List of allowed file extensions (without dot, lowercase)
        pattern_objects (List[re.Pattern]): Compiled regular expression patterns
        min_size (int): Minimum file size in bytes
        max_size (int): Maximum file size in bytes
        custom_filter (Callable): Custom filter function that takes a DataSource and returns bool

    Example:
        ```python
        # Create a filter for text files with size constraints
        filter = DataSourceFilter(
            extensions=["txt", "md"],
            patterns=[".*government.*"],
            min_size=100,
            max_size=1_000_000
        )

        # Check if a data source matches the filter
        if filter.matches(data_source):
            process_source(data_source)

        # Use with a data source manager
        sources = manager.load_sources(
            base_path="data/documents",
            strategy_name="recursive",
            filter_config={"extensions": ["txt"], "min_size": 100}
        )
        ```
    """

    def __init__(
        self,
        extensions: Optional[List[str]] = None,
        patterns: Optional[List[str]] = None,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        custom_filter: Optional[Callable[[DataSource], bool]] = None,
    ):
        """
        Initialize a data source filter

        Args:
            extensions: List of file extensions to include (without dot)
            patterns: List of regex patterns to match against file paths
            min_size: Minimum file size in bytes
            max_size: Maximum file size in bytes
            custom_filter: Custom filter function that takes a DataSource and returns a boolean
        """
        self.extensions = (
            [ext.lstrip(".").lower() for ext in extensions] if extensions else None
        )
        self.pattern_objects = (
            [re.compile(p, re.IGNORECASE) for p in patterns] if patterns else None
        )
        self.min_size = min_size
        self.max_size = max_size
        self.custom_filter = custom_filter

    def matches(self, data_source: Union[DataSource, str]) -> bool:
        """
        Check if a data source matches this filter

        Args:
            data_source: DataSource object or path string to check

        Returns:
            True if the data source matches the filter criteria
        """
        # Convert string path to DataSource if needed
        if isinstance(data_source, str):
            data_source = DataSource(data_source)

        # Check extension
        if self.extensions is not None:
            if data_source.extension.lower() not in self.extensions:
                return False

        # Check patterns
        if self.pattern_objects is not None:
            path_matched = False
            for pattern in self.pattern_objects:
                if pattern.search(data_source.path):
                    path_matched = True
                    break
            if not path_matched:
                return False

        # Check size (only if the source has been loaded or has metadata)
        if "content_size_bytes" in data_source.metadata:
            size = data_source.metadata["content_size_bytes"]

            if self.min_size is not None and size < self.min_size:
                return False

            if self.max_size is not None and size > self.max_size:
                return False

        # Apply custom filter if provided
        if self.custom_filter is not None:
            return self.custom_filter(data_source)

        # All criteria passed
        return True


class DataSourceTraversalStrategy:
    """
    Abstract base strategy for traversing data sources in different organizational patterns.

    This class defines the interface for all traversal strategies in the system. Each
    concrete implementation provides a different method for discovering and traversing
    data sources, whether through flat directories, recursive scanning, institutional
    organization, or pattern-based matching.

    Traversal strategies are responsible for:
    1. Finding relevant files according to their specific organizational pattern
    2. Creating DataSource objects for discovered files
    3. Applying any filters to exclude irrelevant sources
    4. Adding appropriate metadata based on the organizational structure

    Concrete implementations include:
    - FlatDirectoryTraversalStrategy: Non-recursive traversal of a single directory
    - RecursiveDirectoryTraversalStrategy: Traverses directories recursively with depth control
    - InstitutionalTraversalStrategy: Specialized for 'institution/topic.txt' organization
    - PatternBasedTraversalStrategy: Finds files using glob patterns

    Example:
        ```python
        # Create a strategy
        strategy = RecursiveDirectoryTraversalStrategy(max_depth=2)

        # Use the strategy directly
        for data_source in strategy.traverse("data/documents", filter_obj):
            print(f"Found source: {data_source.path}")

        # Or through the DataSourceManager
        manager = DataSourceManager()
        sources = manager.load_sources(
            base_path="data/documents",
            strategy_name="recursive"
        )
        ```
    """

    def traverse(
        self, base_path: str, filter_obj: Optional[DataSourceFilter] = None
    ) -> Iterator[DataSource]:
        """
        Traverse the data source and yield DataSource objects

        Args:
            base_path: Base path to start traversal
            filter_obj: Optional filter to apply to data sources

        Yields:
            DataSource objects that match the filter
        """
        raise NotImplementedError("Subclasses must implement traverse()")


class FlatDirectoryTraversalStrategy(DataSourceTraversalStrategy):
    """
    Strategy for traversing files in a single directory without recursion.

    This strategy implements a flat directory traversal approach, examining only
    the files directly within the specified directory without descending into
    subdirectories. It's useful for processing files in a single organizational level.

    The strategy creates DataSource objects for each file found, and applies any
    specified filters to exclude files that don't match the criteria.

    This is the simplest traversal strategy and is appropriate when:
    - All content is stored in a flat structure without subdirectories
    - Only top-level files in a directory are needed
    - Performance is a concern and deep traversal is unnecessary

    Example:
        ```python
        # Create the strategy
        strategy = FlatDirectoryTraversalStrategy()

        # Use the strategy directly
        for source in strategy.traverse("data/documents", filter_obj):
            print(f"Found file: {source.name}")

        # Or through the DataSourceManager
        manager = DataSourceManager()
        sources = manager.load_sources(
            base_path="data/documents",
            strategy_name="flat"
        )
        ```

    Sample folder structure this strategy works with:
    ```
    data/
    ├── document1.txt
    ├── document2.txt
    ├── report.md
    ├── statistics.json
    └── subdirectory/     # contents in this directory would be ignored
        ├── file1.txt
        └── file2.txt
    ```

    Only the files directly in the 'data' directory would be processed,
    while the files in 'subdirectory' would be ignored.
    """

    def traverse(
        self, base_path: str, filter_obj: Optional[DataSourceFilter] = None
    ) -> Iterator[DataSource]:
        """
        Traverse files in a flat directory (non-recursive)

        Args:
            base_path: Base path to directory
            filter_obj: Optional filter to apply

        Yields:
            DataSource objects for matching files
        """
        base_dir = Path(base_path)

        if not base_dir.exists() or not base_dir.is_dir():
            logger.warning(
                f"Base path '{base_path}' does not exist or is not a directory"
            )
            return

        for file_path in base_dir.iterdir():
            if file_path.is_file():
                data_source = DataSource(str(file_path))

                if filter_obj is None or filter_obj.matches(data_source):
                    yield data_source


class RecursiveDirectoryTraversalStrategy(DataSourceTraversalStrategy):
    """
    Strategy for recursively traversing directories to find data sources.

    This strategy implements a depth-first search through a directory structure,
    examining files at all levels of the hierarchy with optional depth limiting.
    It's useful for processing complex nested folder structures where content
    may be organized across multiple subdirectories.

    The strategy adds depth-related metadata to each DataSource it creates:
    - 'depth': The nesting level of the file relative to the base path
    - 'relative_path': The file path relative to the base directory

    Attributes:
        max_depth (Optional[int]): Maximum directory depth to traverse (None for unlimited)

    Example:
        ```python
        # Create a strategy with depth limiting
        strategy = RecursiveDirectoryTraversalStrategy(max_depth=2)

        # Use the strategy directly
        for source in strategy.traverse("data/documents", filter_obj):
            print(f"Found file: {source.name} (depth: {source.metadata['depth']})")

        # Or through the DataSourceManager
        manager = DataSourceManager()
        sources = manager.load_sources(
            base_path="data/documents",
            strategy_name="recursive"
        )
        ```

    Sample folder structure this strategy works with:
    ```
    data/
    ├── document1.txt          # depth 0
    ├── reports/
    │   ├── report1.md         # depth 1
    │   ├── report2.md         # depth 1
    │   └── quarterly/
    │       ├── q1_report.txt  # depth 2
    │       └── q2_report.txt  # depth 2
    └── statistics/
        ├── yearly.json        # depth 1
        └── monthly/
            └── january.json   # depth 2
    ```

    All files at all levels would be processed, unless a max_depth
    limit is set or a filter is applied.
    """

    def __init__(self, max_depth: Optional[int] = None):
        """
        Initialize the recursive traversal strategy

        Args:
            max_depth: Maximum traversal depth (None for unlimited)
        """
        self.max_depth = max_depth

    def traverse(
        self, base_path: str, filter_obj: Optional[DataSourceFilter] = None
    ) -> Iterator[DataSource]:
        """
        Recursively traverse directories and yield DataSource objects

        Args:
            base_path: Base path to start traversal
            filter_obj: Optional filter to apply

        Yields:
            DataSource objects for matching files
        """
        base_dir = Path(base_path)

        if not base_dir.exists() or not base_dir.is_dir():
            logger.warning(
                f"Base path '{base_path}' does not exist or is not a directory"
            )
            return

        for file_path in self._traverse_recursive(base_dir, 0):
            if file_path.is_file():
                data_source = DataSource(str(file_path))

                if filter_obj is None or filter_obj.matches(data_source):
                    # Add depth metadata
                    rel_path = file_path.relative_to(base_dir)
                    data_source.metadata["depth"] = len(rel_path.parts) - 1
                    data_source.metadata["relative_path"] = str(rel_path)

                    yield data_source

    def _traverse_recursive(self, dir_path: Path, current_depth: int) -> Iterator[Path]:
        """
        Internal recursive traversal helper

        Args:
            dir_path: Current directory path
            current_depth: Current traversal depth

        Yields:
            Path objects for all files in the directory tree
        """
        if self.max_depth is not None and current_depth > self.max_depth:
            return

        for path in dir_path.iterdir():
            yield path

            if path.is_dir():
                yield from self._traverse_recursive(path, current_depth + 1)


class InstitutionalTraversalStrategy(DataSourceTraversalStrategy):
    """
    Specialized strategy for traversing institutional data structures.

    This strategy is designed to work with a specific hierarchical organization where:
    - Top-level directories represent institutions (e.g., ministries, agencies)
    - Files within these directories represent topics related to that institution

    The strategy adds institution-specific metadata to each DataSource it creates:
    - 'institution': The name of the institution (derived from directory name)
    - 'topic': The topic name (derived from the file stem)

    This strategy is particularly useful for government or organizational content where
    information is naturally divided by institutional boundaries and topical areas.

    Example:
        ```python
        # Create the strategy
        strategy = InstitutionalTraversalStrategy()

        # Use the strategy directly
        for source in strategy.traverse("data/government", filter_obj):
            print(f"Institution: {source.metadata['institution']}")
            print(f"Topic: {source.metadata['topic']}")

        # Or through the DataSourceManager
        manager = DataSourceManager()
        sources = manager.load_sources(
            base_path="data/government",
            strategy_name="institutional",
            filter_config={"extensions": ["txt"]}
        )
        ```

    Sample folder structure this strategy works with:
    ```
    data/
    ├── ministry_of_finance/
    │   ├── taxes.txt
    │   ├── budget.txt
    │   └── investments.txt
    ├── ministry_of_education/
    │   ├── schools.txt
    │   ├── universities.txt
    │   └── scholarships.txt
    └── ministry_of_health/
        ├── hospitals.txt
        ├── insurance.txt
        └── medications.txt
    ```

    Each file is processed as a separate data source with its institutional context
    preserved in the metadata, making it available for template parameters during
    dataset generation.
    """

    def traverse(
        self, base_path: str, filter_obj: Optional[DataSourceFilter] = None
    ) -> Iterator[DataSource]:
        """
        Traverse institutional data structure

        Args:
            base_path: Base path to directory with institution folders
            filter_obj: Optional filter to apply

        Yields:
            DataSource objects with institution metadata
        """
        base_dir = Path(base_path)

        if not base_dir.exists() or not base_dir.is_dir():
            logger.warning(
                f"Base path '{base_path}' does not exist or is not a directory"
            )
            return

        # For each institution directory
        for institution_dir in base_dir.iterdir():
            if not institution_dir.is_dir():
                continue

            institution_name = institution_dir.name

            # For each topic file in the institution directory
            for file_path in institution_dir.iterdir():
                if file_path.is_file():
                    data_source = DataSource(
                        str(file_path),
                        metadata={
                            "institution": institution_name,
                            "topic": file_path.stem,
                        },
                    )

                    if filter_obj is None or filter_obj.matches(data_source):
                        yield data_source


class PatternBasedTraversalStrategy(DataSourceTraversalStrategy):
    """
    Strategy for finding files based on glob patterns.

    This strategy uses glob patterns to find matching files, supporting both flat and
    recursive searching. It's particularly useful for finding files by name patterns,
    extensions, or path structures without requiring a specific directory organization.

    The strategy adds pattern-specific metadata to each DataSource it creates:
    - 'relative_path': The file path relative to the base directory
    - 'pattern': The specific pattern that matched this file
    - 'level_X': Path components at each directory level (e.g., level_0, level_1)

    Attributes:
        patterns (List[str]): List of glob patterns to match files
        recursive (bool): Whether to search recursively into subdirectories

    Example:
        ```python
        # Create a strategy with specific patterns
        strategy = PatternBasedTraversalStrategy(
            patterns=["**/*.txt", "**/*.md"],
            recursive=True
        )

        # Use the strategy directly
        for source in strategy.traverse("data/documents", filter_obj):
            print(f"Found file: {source.name} matched by {source.metadata['pattern']}")

        # Or through the DataSourceManager
        manager = DataSourceManager()
        sources = manager.load_sources(
            base_path="data/documents",
            strategy_name="pattern",
            filter_config={"extensions": ["txt", "md"]}
        )
        ```

    Sample folder structure this strategy works with:
    ```
    data/
    ├── document1.txt
    ├── reports/
    │   ├── report1.md
    │   ├── report2.pdf       # would be ignored with "**/*.txt" pattern
    │   └── quarterly/
    │       ├── q1_report.txt  # would be included with "**/*.txt" pattern
    │       └── q2_data.xlsx  # would be ignored with "**/*.txt" pattern
    └── statistics/
        └── data.json         # would be ignored with "**/*.txt" pattern
    ```

    Only files matching the specified patterns would be processed, regardless
    of their location in the directory structure (if recursive=True).
    """

    def __init__(self, patterns: List[str], recursive: bool = True):
        """
        Initialize with glob patterns

        Args:
            patterns: List of glob patterns to match files
            recursive: Whether to search recursively
        """
        self.patterns = patterns
        self.recursive = recursive

    def traverse(
        self, base_path: str, filter_obj: Optional[DataSourceFilter] = None
    ) -> Iterator[DataSource]:
        """
        Find files matching the specified glob patterns

        Args:
            base_path: Base path for the patterns
            filter_obj: Optional filter to apply

        Yields:
            DataSource objects for matching files
        """

        for pattern in self.patterns:
            pattern_path = os.path.join(base_path, pattern)

            for file_path in glob.glob(pattern_path, recursive=self.recursive):
                if os.path.isfile(file_path):
                    # Extract relative path components for metadata
                    rel_path = os.path.relpath(file_path, base_path)
                    path_parts = Path(rel_path).parts

                    # Create metadata based on path structure
                    metadata = {
                        "relative_path": rel_path,
                        "pattern": pattern,
                    }

                    # Add path parts as metadata (for hierarchical organization)
                    for i, part in enumerate(path_parts[:-1]):  # Skip the filename
                        metadata[f"level_{i}"] = part

                    data_source = DataSource(file_path, metadata)

                    if filter_obj is None or filter_obj.matches(data_source):
                        yield data_source


class DataSourceManager:
    """
    Manager for handling data sources with flexible configuration.

    This class provides a unified interface for loading and filtering data sources using
    different traversal strategies. It handles the complexities of directory traversal,
    file filtering, and metadata extraction, allowing consumers to easily access content
    from various structured and unstructured sources.

    The manager supports multiple traversal strategies:
    - 'flat': Non-recursive traversal of a single directory
    - 'recursive': Traverses directories recursively with optional depth limit
    - 'institutional': Specialized for the structure 'institution/topic.txt'
    - 'pattern': Finds files using glob patterns

    Attributes:
        config (Dict[str, Any]): Configuration dictionary for the manager
        _strategy_map (Dict[str, Any]): Mapping of strategy names to implementations

    Example:
        ```python
        # Create with default configuration
        manager = DataSourceManager()

        # Load sources using institutional strategy
        sources = manager.load_sources(
            base_path="data/documents",
            strategy_name="institutional",
            filter_config={"extensions": ["txt", "md"]}
        )

        # Access content and metadata
        for source in sources:
            print(f"Institution: {source.metadata.get('institution')}")
            print(f"Topic: {source.metadata.get('topic')}")
            print(f"Content: {source.content[:100]}...")
        ```
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the data source manager

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self._strategy_map = {
            "flat": FlatDirectoryTraversalStrategy,
            "recursive": lambda: RecursiveDirectoryTraversalStrategy(),
            "institutional": InstitutionalTraversalStrategy,
            "pattern": lambda: PatternBasedTraversalStrategy(
                patterns=self.config.get("patterns", ["**/*.txt"]),
                recursive=self.config.get("recursive", True),
            ),
        }

    def get_strategy(self, strategy_name: str) -> DataSourceTraversalStrategy:
        """
        Get a traversal strategy by name

        Args:
            strategy_name: Name of the strategy

        Returns:
            DataSourceTraversalStrategy instance

        Raises:
            ValueError: If the strategy_name is unknown
        """
        if strategy_name not in self._strategy_map:
            raise ValueError(f"Unknown traversal strategy: {strategy_name}")

        strategy_factory = self._strategy_map[strategy_name]

        if callable(strategy_factory):
            return strategy_factory()
        else:
            return strategy_factory()

    def create_filter_from_config(
        self, filter_config: Dict[str, Any]
    ) -> DataSourceFilter:
        """
        Create a filter from configuration

        Args:
            filter_config: Filter configuration dictionary

        Returns:
            DataSourceFilter instance
        """
        return DataSourceFilter(
            extensions=filter_config.get("extensions"),
            patterns=filter_config.get("patterns"),
            min_size=filter_config.get("min_size"),
            max_size=filter_config.get("max_size"),
        )

    def load_sources(
        self,
        base_path: str = None,
        strategy_name: str = None,
        filter_config: Dict[str, Any] = None,
    ) -> List[DataSource]:
        """
        Load data sources using the specified strategy and filter.

        This method is the main entry point for discovering and filtering data sources.
        It creates an appropriate traversal strategy based on the strategy name, applies
        any specified filters, and returns a list of matching DataSource objects.

        Args:
            base_path (str, optional): Base path for data sources. If None, uses the value
                from the instance configuration.
            strategy_name (str, optional): Name of traversal strategy to use ('flat', 'recursive',
                'institutional', or 'pattern'). If None, defaults to 'recursive'.
            filter_config (Dict[str, Any], optional): Configuration dictionary for filtering
                sources. Can include 'extensions', 'patterns', 'min_size', and 'max_size'.

        Returns:
            List[DataSource]: A list of DataSource objects that match the criteria.

        Raises:
            ValueError: If no base path is provided or available in configuration.
            ValueError: If an unknown traversal strategy is specified.

        Example:
            ```python
            # Load all .txt files from a directory using institutional organization
            sources = manager.load_sources(
                base_path="data/government",
                strategy_name="institutional",
                filter_config={"extensions": ["txt"]}
            )
            ```
        """
        # Use values from instance config if not explicitly provided
        base_path = base_path or self.config.get("base_path")
        strategy_name = strategy_name or self.config.get("strategy", "recursive")
        filter_config = filter_config or self.config.get("filter", {})

        if not base_path:
            raise ValueError("Base path must be provided")

        # Get strategy
        strategy = self.get_strategy(strategy_name)

        # Create filter if configuration is provided
        filter_obj = None
        if filter_config:
            filter_obj = self.create_filter_from_config(filter_config)

        # Traverse and collect data sources
        sources = list(strategy.traverse(base_path, filter_obj))

        logger.info(
            f"Loaded {len(sources)} data sources from {base_path} using '{strategy_name}' strategy"
        )

        return sources

    # @classmethod
    # def from_config_file(cls, config_path: str) -> 'DataSourceManager':
    #     """
    #     Create a DataSourceManager instance from a configuration file

    #     Args:
    #         config_path: Path to configuration file (YAML or JSON)

    #     Returns:
    #         DataSourceManager instance

    #     Raises:
    #         ValueError: If the file format is not supported
    #     """
    #     file_ext = Path(config_path).suffix.lower()

    #     with open(config_path, 'r', encoding='utf-8') as f:
    #         if file_ext == '.yaml' or file_ext == '.yml':
    #             config = yaml.safe_load(f)
    #         elif file_ext == '.json':
    #             config = json.load(f)
    #         else:
    #             raise ValueError(f"Unsupported configuration file format: {file_ext}")

    #     return cls(config)
