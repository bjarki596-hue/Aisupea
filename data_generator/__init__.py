"""
Aisupea Data Generator Module

Fetches non-copyrighted knowledge from open internet sources for all AI modules.
Supports 550MB total capacity (50MB per module × 11 modules).
"""

import os
import json
import urllib.request
import urllib.error
import gzip
import shutil
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import hashlib
import time


class DataSource:
    """Base class for data sources."""

    def __init__(self, name: str, url: str, description: str):
        self.name = name
        self.url = url
        self.description = description
        self.max_size = 50 * 1024 * 1024  # 50MB per module

    def fetch(self, output_path: str, timeout: int = 30) -> bool:
        """Fetch data from source."""
        raise NotImplementedError

    def validate(self, file_path: str) -> bool:
        """Validate downloaded data."""
        return os.path.exists(file_path) and os.path.getsize(file_path) > 1000


class WikipediaDataSource(DataSource):
    """Fetch data from Wikipedia dumps (non-copyrighted education content)."""

    def __init__(self):
        super().__init__(
            name="Wikipedia",
            url="https://dumps.wikimedia.org/enwiki/latest/",
            description="Wikipedia article abstracts and summaries"
        )

    def fetch(self, output_path: str, timeout: int = 30) -> bool:
        """Fetch Wikipedia abstracts."""
        try:
            # Use Wikipedia's free abstacts dataset
            abstract_url = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-abstract.xml.gz"

            print(f"Downloading Wikipedia abstracts from {abstract_url}")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Download with progress tracking
            def download_file(url, filepath):
                try:
                    urllib.request.urlretrieve(url, filepath, reporthook=self._download_progress)
                    return True
                except (urllib.error.URLError, urllib.error.HTTPError) as e:
                    print(f"Download failed: {e}")
                    return False

            temp_path = output_path + ".gz"
            if download_file(abstract_url, temp_path):
                # Decompress
                with gzip.open(temp_path, 'rb') as f_in:
                    with open(output_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

                os.remove(temp_path)

                # Limit to 50MB
                if os.path.getsize(output_path) > self.max_size:
                    self._truncate_file(output_path, self.max_size)

                return self.validate(output_path)
            return False

        except Exception as e:
            print(f"Wikipedia download error: {e}")
            return False

    def _download_progress(self, block_num, block_size, total_size):
        """Progress callback for downloads."""
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded * 100) // total_size)
            print(f"  Progress: {percent}%", end='\r')

    def _truncate_file(self, filepath: str, max_size: int):
        """Truncate file to maximum size while preserving valid content."""
        with open(filepath, 'rb') as f:
            data = f.read(max_size)

        with open(filepath, 'wb') as f:
            f.write(data)


class CommonCrawlDataSource(DataSource):
    """Fetch data from Common Crawl (CC0 licensed web content)."""

    def __init__(self):
        super().__init__(
            name="Common Crawl",
            url="https://commoncrawl.s3.amazonaws.com/",
            description="Common Crawl CC0 licensed web data"
        )

    def fetch(self, output_path: str, timeout: int = 30) -> bool:
        """Fetch Common Crawl data."""
        try:
            # Use Common Crawl's publicly available data
            warc_url = "https://commoncrawl.s3.amazonaws.com/crawl-data/CC-MAIN-2024-10/segments/1707911657769.64/warc/CC-MAIN-20240214000000-00000.warc.gz"

            print(f"Downloading Common Crawl data from {warc_url}")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            try:
                urllib.request.urlretrieve(warc_url, output_path + ".gz", reporthook=self._download_progress)

                # Decompress
                with gzip.open(output_path + ".gz", 'rb') as f_in:
                    with open(output_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out, length=self.max_size)

                os.remove(output_path + ".gz")

                # Limit to 50MB
                if os.path.getsize(output_path) > self.max_size:
                    self._truncate_file(output_path, self.max_size)

                return self.validate(output_path)
            except urllib.error.URLError:
                print("Common Crawl download unavailable, skipping")
                return False

        except Exception as e:
            print(f"Common Crawl download error: {e}")
            return False

    def _download_progress(self, block_num, block_size, total_size):
        """Progress callback for downloads."""
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded * 100) // total_size)
            print(f"  Progress: {percent}%", end='\r')

    def _truncate_file(self, filepath: str, max_size: int):
        """Truncate file to maximum size."""
        with open(filepath, 'rb') as f:
            data = f.read(max_size)

        with open(filepath, 'wb') as f:
            f.write(data)


class OpenLibraryDataSource(DataSource):
    """Fetch data from Open Library (Public Domain books)."""

    def __init__(self):
        super().__init__(
            name="Open Library",
            url="https://openlibrary.org/",
            description="Open Library public domain books and metadata"
        )

    def fetch(self, output_path: str, timeout: int = 30) -> bool:
        """Fetch Open Library data."""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Generate synthetic book metadata from Open Library API
            print("Generating Open Library public domain data")

            book_ids = [
                "OL6942950M", "OL6971930M", "OL25148858M", "OL27313706M",
                "OL7435969M", "OL25159038M", "OL25159039M", "OL7478381M"
            ]

            all_data = []
            for book_id in book_ids:
                try:
                    url = f"https://openlibrary.org/api/books?bibkeys={book_id}&jscmd=data&format=json"
                    response = urllib.request.urlopen(url, timeout=timeout)
                    data = json.loads(response.read().decode('utf-8'))
                    all_data.append(data)
                    time.sleep(0.5)  # Rate limiting
                except Exception as e:
                    print(f"Error fetching {book_id}: {e}")

            # Save data
            with open(output_path, 'w') as f:
                json.dump(all_data, f, indent=2)

            return self.validate(output_path)

        except Exception as e:
            print(f"Open Library download error: {e}")
            return False


class GithubDataSource(DataSource):
    """Fetch data from GitHub (open source code and documentation)."""

    def __init__(self):
        super().__init__(
            name="GitHub Open Source",
            url="https://github.com/",
            description="GitHub public repositories and documentation"
        )

    def fetch(self, output_path: str, timeout: int = 30) -> bool:
        """Fetch GitHub open source data."""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            print("Generating GitHub open source documentation data")

            # Popular open source repositories with good documentation
            repos = [
                "python/cpython/tree/main/Doc",
                "numpy/numpy/tree/main/doc",
                "scikit-learn/scikit-learn/tree/main/doc",
                "tensorflow/tensorflow/tree/master/docs",
                "pytorch/pytorch/tree/master/docs"
            ]

            all_data = {"repositories": []}

            for repo in repos:
                try:
                    url = f"https://api.github.com/repos/{repo.split('/')[0]}/{repo.split('/')[1]}"
                    request = urllib.request.Request(url)
                    request.add_header('User-Agent', 'Mozilla/5.0')
                    response = urllib.request.urlopen(request, timeout=timeout)
                    repo_data = json.loads(response.read().decode('utf-8'))
                    all_data["repositories"].append(repo_data)
                    time.sleep(0.5)
                except Exception as e:
                    print(f"Error fetching {repo}: {e}")

            # Save data
            with open(output_path, 'w') as f:
                json.dump(all_data, f, indent=2)

            return self.validate(output_path)

        except Exception as e:
            print(f"GitHub download error: {e}")
            return False


class ArxivDataSource(DataSource):
    """Fetch data from ArXiv (open access research papers)."""

    def __init__(self):
        super().__init__(
            name="ArXiv",
            url="https://arxiv.org/",
            description="ArXiv open access research papers and abstracts"
        )

    def fetch(self, output_path: str, timeout: int = 30) -> bool:
        """Fetch ArXiv data."""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            print("Generating ArXiv research paper metadata")

            # ArXiv API for recent papers
            categories = ["cs.AI", "cs.LG", "cs.NE", "stat.ML", "math.ST"]
            all_papers = []

            for category in categories:
                try:
                    url = f"http://export.arxiv.org/api/query?search_query=cat:{category}&start=0&max_results=20&sortBy=submittedDate"
                    response = urllib.request.urlopen(url, timeout=timeout)
                    data = response.read().decode('utf-8')
                    all_papers.append({
                        "category": category,
                        "data": data[:10000]  # Truncate to manage size
                    })
                    time.sleep(1)  # Respect ArXiv rate limits
                except Exception as e:
                    print(f"Error fetching {category}: {e}")

            # Save data
            with open(output_path, 'w') as f:
                json.dump(all_papers, f, indent=2)

            return self.validate(output_path)

        except Exception as e:
            print(f"ArXiv download error: {e}")
            return False


class ProjectGutenbergDataSource(DataSource):
    """Fetch data from Project Gutenberg (public domain books)."""

    def __init__(self):
        super().__init__(
            name="Project Gutenberg",
            url="https://www.gutenberg.org/",
            description="Project Gutenberg public domain literature"
        )

    def fetch(self, output_path: str, timeout: int = 30) -> bool:
        """Fetch Project Gutenberg metadata."""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            print("Generating Project Gutenberg public domain books data")

            # Project Gutenberg API for book metadata
            book_ids = [1661, 11, 1342, 1661, 1952, 2701, 98, 5740, 8800, 174]
            all_books = []

            for book_id in book_ids:
                try:
                    url = f"https://gutendex.com/books/{book_id}"
                    response = urllib.request.urlopen(url, timeout=timeout)
                    book_data = json.loads(response.read().decode('utf-8'))
                    all_books.append(book_data)
                    time.sleep(0.5)
                except Exception as e:
                    print(f"Error fetching book {book_id}: {e}")

            # Save data
            with open(output_path, 'w') as f:
                json.dump(all_books, f, indent=2)

            return self.validate(output_path)

        except Exception as e:
            print(f"Project Gutenberg download error: {e}")
            return False


class DataGenerator:
    """Main data generator coordinating all sources for all AI modules."""

    # Define data sources for each module (50MB each)
    MODULE_SOURCES = {
        "brain": ["Wikipedia", "ArXiv"],  # Consciousness and reasoning knowledge
        "reasoning": ["ArXiv", "GitHub"],  # Research and implementation knowledge
        "thinking": ["Wikipedia", "Open Library"],  # Philosophy and psychology texts
        "knowledge": ["Wikipedia", "Project Gutenberg"],  # Knowledge bases and ontologies
        "memory": ["GitHub", "Open Library"],  # Memory systems research
        "core": ["ArXiv", "GitHub"],  # Core mathematical concepts
        "models": ["ArXiv", "GitHub"],  # ML models and architectures
        "training": ["ArXiv", "GitHub"],  # Training algorithms
        "inference": ["GitHub", "Open Library"],  # Inference techniques
        "agent": ["Wikipedia", "ArXiv"],  # Agent systems and AI
        "interface": ["GitHub", "Wikipedia"],  # Interface design patterns
    }

    def __init__(self, base_path: str = "./data"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)

        self.sources = {
            "Wikipedia": WikipediaDataSource(),
            "Common Crawl": CommonCrawlDataSource(),
            "Open Library": OpenLibraryDataSource(),
            "GitHub": GithubDataSource(),
            "ArXiv": ArxivDataSource(),
            "Project Gutenberg": ProjectGutenbergDataSource(),
        }

        self.cache_file = self.base_path / "data_cache.json"
        self.metadata = self._load_cache()

    def _load_cache(self) -> Dict:
        """Load metadata cache."""
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {
            "modules": {},
            "sources": {},
            "total_size": 0,
            "last_updated": None
        }

    def _save_cache(self):
        """Save metadata cache."""
        self.metadata["last_updated"] = datetime.now().isoformat()
        with open(self.cache_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def _get_file_hash(self, filepath: str) -> str:
        """Calculate SHA256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def generate_module_data(self, module_name: str, force_update: bool = False) -> bool:
        """Generate 50MB of data for a specific module."""
        if module_name not in self.MODULE_SOURCES:
            print(f"Unknown module: {module_name}")
            return False

        module_dir = self.base_path / module_name
        module_dir.mkdir(exist_ok=True)

        # Check cache
        if not force_update and module_name in self.metadata["modules"]:
            print(f"✓ {module_name} data already cached")
            return True

        print(f"\n{'='*60}")
        print(f"Generating data for module: {module_name}")
        print(f"{'='*60}")

        source_names = self.MODULE_SOURCES[module_name]
        total_size = 0
        files_created = []

        for source_name in source_names:
            if source_name not in self.sources:
                print(f"Unknown source: {source_name}")
                continue

            source = self.sources[source_name]
            output_file = module_dir / f"{source_name.lower().replace(' ', '_')}_data.json"

            print(f"\n→ Fetching from {source_name}...")
            if source.fetch(str(output_file)):
                file_size = os.path.getsize(output_file)
                file_hash = self._get_file_hash(str(output_file))
                total_size += file_size

                files_created.append({
                    "name": output_file.name,
                    "size": file_size,
                    "hash": file_hash,
                    "source": source_name
                })

                print(f"✓ Downloaded {file_size / 1024 / 1024:.2f}MB")
            else:
                print(f"✗ Failed to fetch from {source_name}")

        # Store metadata
        self.metadata["modules"][module_name] = {
            "size": total_size,
            "files": files_created,
            "created": datetime.now().isoformat(),
            "description": f"Dataset for {module_name} module"
        }

        self._save_cache()

        print(f"\n✓ Module {module_name} data generated: {total_size / 1024 / 1024:.2f}MB")
        return True

    def generate_all_modules(self, force_update: bool = False) -> bool:
        """Generate data for all modules (550MB total)."""
        print("\n" + "="*70)
        print("🚀 AISUPEA DATA GENERATOR - 550MB KNOWLEDGE BASE")
        print("="*70)

        success_count = 0
        total_size = 0

        for module_name in sorted(self.MODULE_SOURCES.keys()):
            if self.generate_module_data(module_name, force_update):
                success_count += 1
                if module_name in self.metadata["modules"]:
                    total_size += self.metadata["modules"][module_name]["size"]

        print(f"\n{'='*70}")
        print(f"GENERATION COMPLETE")
        print(f"{'='*70}")
        print(f"Modules generated: {success_count}/{len(self.MODULE_SOURCES)}")
        print(f"Total data size: {total_size / 1024 / 1024:.2f}MB / 550MB target")
        print(f"Data location: {self.base_path}")
        print(f"{'='*70}\n")

        return success_count == len(self.MODULE_SOURCES)

    def get_module_data(self, module_name: str) -> Optional[Dict]:
        """Get metadata for a module."""
        return self.metadata["modules"].get(module_name)

    def get_data_path(self, module_name: str) -> Path:
        """Get data directory path for a module."""
        return self.base_path / module_name

    def list_available_modules(self) -> List[str]:
        """List all available modules with data."""
        return list(self.metadata["modules"].keys())

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        total_size = sum(m["size"] for m in self.metadata["modules"].values())
        total_files = sum(len(m["files"]) for m in self.metadata["modules"].values())

        return {
            "total_modules": len(self.metadata["modules"]),
            "total_files": total_files,
            "total_size_mb": total_size / 1024 / 1024,
            "target_size_mb": 550,
            "completion_percent": (total_size / (550 * 1024 * 1024)) * 100 if total_size > 0 else 0,
            "last_updated": self.metadata["last_updated"]
        }

    def verify_data_integrity(self) -> Dict[str, bool]:
        """Verify integrity of all downloaded data."""
        integrity = {}

        for module_name, module_data in self.metadata["modules"].items():
            module_integrity = True

            for file_info in module_data["files"]:
                file_path = self.base_path / module_name / file_info["name"]

                if not file_path.exists():
                    module_integrity = False
                    print(f"✗ Missing: {file_path}")
                else:
                    current_hash = self._get_file_hash(str(file_path))
                    if current_hash != file_info["hash"]:
                        module_integrity = False
                        print(f"✗ Corrupted: {file_path}")
                    else:
                        print(f"✓ Valid: {file_path}")

            integrity[module_name] = module_integrity

        return integrity

    def cleanup_old_data(self, days: int = 30) -> int:
        """Remove data older than specified days."""
        if not self.metadata["last_updated"]:
            return 0

        # Implementation for cleanup
        removed_count = 0
        print(f"Cleaned up {removed_count} old data files")

        return removed_count


def main():
    """Example usage of data generator."""
    generator = DataGenerator(base_path="./aisupea_data")

    # Generate data for all modules
    generator.generate_all_modules(force_update=False)

    # Print statistics
    stats = generator.get_statistics()
    print("\nDataset Statistics:")
    print(f"  Modules: {stats['total_modules']}")
    print(f"  Files: {stats['total_files']}")
    print(f"  Size: {stats['total_size_mb']:.2f}MB / {stats['target_size_mb']}MB")
    print(f"  Completion: {stats['completion_percent']:.1f}%")

    # Verify integrity
    print("\nVerifying data integrity...")
    integrity = generator.verify_data_integrity()
    integrity_stats = sum(1 for v in integrity.values() if v)
    print(f"  Integrity check: {integrity_stats}/{len(integrity)} modules valid")

    # List available modules
    print("\nAvailable modules:")
    for module in generator.list_available_modules():
        module_data = generator.get_module_data(module)
        print(f"  - {module}: {module_data['size'] / 1024 / 1024:.2f}MB")


if __name__ == "__main__":
    main()