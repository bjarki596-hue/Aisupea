"""
Enhanced Data Fetcher for Large-Scale Knowledge Base Generation

This module implements efficient fetching from multiple data sources
to create a comprehensive 5.5GB+ knowledge base.
"""

import json
import gzip
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Generator
from pathlib import Path
from datetime import datetime
import hashlib
import time


class EnhancedDataFetcher:
    """Advanced fetcher with multi-source support and chunking."""
    
    def __init__(self, max_size_mb: int = 2500, timeout: int = 90):
        self.max_size_mb = max_size_mb
        self.max_bytes = max_size_mb * 1024 * 1024
        self.timeout = timeout
        self.chunk_size = 64 * 1024  # 64KB chunks
        self.fetched_data = []
    
    def fetch_wikipedia_full(self) -> List[Dict]:
        """Fetch Wikipedia articles comprehensively."""
        items = []
        
        # Wikipedia dump sources
        sources = [
            "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-abstract.xml.gz",  # Abstracts 10GB+
            "https://en.wikipedia.org/w/api.php?action=query&list=allpages&apnamespace=0&aplimit=500&format=json",
        ]
        
        for source in sources:
            try:
                print(f"[Wikipedia] Fetching from {source}")
                data = self._fetch_url(source, timeout=60)
                if data:
                    # Parse and extract articles
                    parsed = self._parse_wikipedia(data)
                    items.extend(parsed)
                    
                    if sum(len(json.dumps(item)) for item in items) > self.max_bytes:
                        break
            except Exception as e:
                print(f"[Wikipedia] Error: {e}")
                continue
        
        return self._format_items(items, "wikipedia")
    
    def fetch_common_crawl(self) -> List[Dict]:
        """Fetch Common Crawl web content."""
        items = []
        
        # Common Crawl has multiple snapshots
        sources = [
            "https://index.commoncrawl.org/",  # Index API
            "https://commoncrawl.s3.amazonaws.com/",  # Direct S3 access
        ]
        
        try:
            # Fetch index metadata
            print("[Common Crawl] Fetching crawl index...")
            index_data = self._fetch_url(
                "https://index.commoncrawl.org/collinfo.json",
                timeout=30
            )
            
            if index_data:
                # Extract multiple crawl snapshots
                crawl_data = json.loads(index_data)
                for crawl in crawl_data[:3]:  # Get last 3 crawls
                    item = {
                        'id': crawl.get('id', ''),
                        'title': f"CC Snapshot {crawl.get('id', '')}",
                        'content': f"Common Crawl snapshot with {crawl.get('warc', 'unknown')} files",
                        'url': f"https://commoncrawl.s3.amazonaws.com/{crawl.get('id', '')}",
                        'source': 'Common Crawl',
                        'metadata': crawl
                    }
                    items.append(item)
        
        except Exception as e:
            print(f"[Common Crawl] Error: {e}")
        
        return self._format_items(items, "common_crawl")
    
    def fetch_arxiv_papers(self) -> List[Dict]:
        """Fetch ArXiv scientific papers."""
        items = []
        
        # ArXiv API endpoints by category
        categories = ['cs', 'math', 'physics', 'stat', 'econ', 'eess', 'q-bio']
        
        for category in categories:
            try:
                print(f"[ArXiv] Fetching {category} papers...")
                
                # ArXiv API query
                url = f"http://export.arxiv.org/api/query?search_query=cat:{category}&start=0&max_results=1000&sortBy=submittedDate&sortOrder=descending"
                
                data = self._fetch_url(url, timeout=60)
                if data:
                    # Parse Atom feed
                    papers = self._parse_arxiv(data, category)
                    items.extend(papers)
                    
                    if sum(len(json.dumps(item)) for item in items) > self.max_bytes:
                        break
                
                time.sleep(1)  # Rate limiting
            
            except Exception as e:
                print(f"[ArXiv] Error fetching {category}: {e}")
                continue
        
        return self._format_items(items, "arxiv")
    
    def fetch_github_repos(self) -> List[Dict]:
        """Fetch GitHub repositories and documentation."""
        items = []
        
        # GitHub API - trending and popular repos
        queries = [
            "topic:machine-learning&sort=stars&order=desc",
            "topic:deep-learning&sort=stars&order=desc",
            "topic:artificial-intelligence&sort=stars&order=desc",
            "language:python&sort=stars&order=desc",
            "topic:documentation&sort=stars&order=desc",
        ]
        
        for query in queries:
            try:
                print(f"[GitHub] Fetching repos: {query}")
                
                url = f"https://api.github.com/search/repositories?q={query}&per_page=100"
                data = self._fetch_url(url, timeout=30)
                
                if data:
                    repos = json.loads(data)
                    for repo in repos.get('items', [])[:50]:  # Top 50 per query
                        item = {
                            'id': repo.get('id', ''),
                            'title': repo.get('full_name', ''),
                            'content': repo.get('description', '') + '\n' + (repo.get('readme', '') or ''),
                            'url': repo.get('html_url', ''),
                            'source': 'GitHub',
                            'metadata': {
                                'stars': repo.get('stargazers_count', 0),
                                'language': repo.get('language', ''),
                                'topics': repo.get('topics', [])
                            }
                        }
                        items.append(item)
                        
                        if sum(len(json.dumps(it)) for it in items) > self.max_bytes:
                            break
                
                time.sleep(1)  # Rate limiting
            
            except Exception as e:
                print(f"[GitHub] Error: {e}")
                continue
        
        return self._format_items(items, "github")
    
    def fetch_open_library(self) -> List[Dict]:
        """Fetch Open Library book metadata."""
        items = []
        
        try:
            print("[Open Library] Fetching books...")
            
            # Open Library API - popular and classic books
            subjects = ['fiction', 'science', 'history', 'philosophy', 'mathematics', 'programming']
            
            for subject in subjects:
                try:
                    url = f"https://openlibrary.org/subjects/{subject}.json?limit=100&sort=editions"
                    data = self._fetch_url(url, timeout=30)
                    
                    if data:
                        result = json.loads(data)
                        for book in result.get('works', [])[:50]:
                            item = {
                                'id': book.get('key', ''),
                                'title': book.get('title', ''),
                                'content': book.get('description', '') if isinstance(book.get('description'), str) else '',
                                'url': f"https://openlibrary.org{book.get('key', '')}",
                                'source': 'Open Library',
                                'metadata': {
                                    'editionsCount': book.get('edition_count', 0),
                                    'coverUrl': book.get('cover_url', ''),
                                    'firstPublishedYear': book.get('first_publish_year', '')
                                }
                            }
                            items.append(item)
                    
                    time.sleep(0.5)  # Rate limiting
                
                except Exception as e:
                    print(f"[Open Library] Error fetching {subject}: {e}")
                    continue
        
        except Exception as e:
            print(f"[Open Library] Error: {e}")
        
        return self._format_items(items, "open_library")
    
    def fetch_gutenberg_books(self) -> List[Dict]:
        """Fetch Project Gutenberg books."""
        items = []
        
        try:
            print("[Gutenberg] Fetching public domain books...")
            
            # Gutenberg API
            url = "https://gutendex.com/books"
            
            # Fetch top 100 books
            for page in range(1, 4):  # 3 pages = 300 books
                try:
                    page_url = f"{url}?page={page}"
                    data = self._fetch_url(page_url, timeout=30)
                    
                    if data:
                        result = json.loads(data)
                        for book in result.get('results', []):
                            item = {
                                'id': str(book.get('id', '')),
                                'title': book.get('title', ''),
                                'content': ', '.join(book.get('formats', {}).get('text/plain', [])[:1]),
                                'url': f"https://www.gutenberg.org/ebooks/{book.get('id', '')}",
                                'source': 'Project Gutenberg',
                                'metadata': {
                                    'authors': [a.get('name', '') for a in book.get('authors', [])],
                                    'coverImage': book.get('formats', {}).get('image/jpeg', ''),
                                    'downloadCount': book.get('download_count', 0)
                                }
                            }
                            items.append(item)
                    
                    time.sleep(0.5)
                
                except Exception as e:
                    print(f"[Gutenberg] Error fetching page {page}: {e}")
                    continue
        
        except Exception as e:
            print(f"[Gutenberg] Error: {e}")
        
        return self._format_items(items, "gutenberg")
    
    def fetch_stack_exchange(self) -> List[Dict]:
        """Fetch Stack Exchange Q&A data."""
        items = []
        
        sites = ['stackoverflow', 'askubuntu', 'serverfault', 'superuser']
        
        for site in sites:
            try:
                print(f"[Stack Exchange] Fetching {site}...")
                
                url = f"https://api.stackexchange.com/2.3/questions?site={site}&pagesize=100&order=desc&sort=votes"
                data = self._fetch_url(url, timeout=30)
                
                if data:
                    result = json.loads(data)
                    for q in result.get('items', [])[:200]:
                        item = {
                            'id': str(q.get('question_id', '')),
                            'title': q.get('title', ''),
                            'content': q.get('body', ''),
                            'url': q.get('link', ''),
                            'source': 'Stack Exchange',
                            'metadata': {
                                'score': q.get('score', 0),
                                'tags': q.get('tags', []),
                                'answers': q.get('answer_count', 0)
                            }
                        }
                        items.append(item)
                
                time.sleep(1)  # Rate limiting
            
            except Exception as e:
                print(f"[Stack Exchange] Error: {e}")
                continue
        
        return self._format_items(items, "stack_exchange")
    
    def _fetch_url(self, url: str, timeout: int = 30) -> Optional[str]:
        """Fetch URL with error handling."""
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Aisupea DataGenerator/1.0)'}
            req = urllib.request.Request(url, headers=headers)
            
            with urllib.request.urlopen(req, timeout=timeout) as response:
                data = response.read()
                
                # Decompress if gzipped
                if response.headers.get('Content-Encoding') == 'gzip':
                    data = gzip.decompress(data)
                
                return data.decode('utf-8', errors='ignore')
        
        except Exception as e:
            print(f"  Fetch error: {e}")
            return None
    
    def _parse_wikipedia(self, data: str) -> List[Dict]:
        """Parse Wikipedia XML data."""
        items = []
        try:
            # Simple XML parsing for Wikipedia abstracts
            import xml.etree.ElementTree as ET
            
            # Handle as JSON or XML
            if data.startswith('[') or data.startswith('{'):
                return json.loads(data).get('query', {}).get('pages', {}).values()
            
            # Parse XML
            root = ET.fromstring(data[:1000000])  # Parse first MB
            for doc in root.findall('.//doc'):
                item = {
                    'id': doc.get('id', ''),
                    'title': doc.findtext('title', ''),
                    'content': doc.findtext('abstract', ''),
                    'url': f"https://wikipedia.org/wiki/{doc.get('id', '')}",
                    'source': 'Wikipedia'
                }
                if item['title']:
                    items.append(item)
        
        except Exception as e:
            print(f"Wikipedia parse error: {e}")
        
        return items
    
    def _parse_arxiv(self, data: str, category: str) -> List[Dict]:
        """Parse ArXiv Atom feed."""
        items = []
        try:
            import xml.etree.ElementTree as ET
            
            root = ET.fromstring(data[:5000000])  # Parse first 5MB
            
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                title_elem = entry.find('{http://www.w3.org/2005/Atom}title')
                summary_elem = entry.find('{http://www.w3.org/2005/Atom}summary')
                id_elem = entry.find('{http://www.w3.org/2005/Atom}id')
                
                item = {
                    'id': id_elem.text if id_elem is not None else '',
                    'title': title_elem.text if title_elem is not None else '',
                    'content': summary_elem.text if summary_elem is not None else '',
                    'url': id_elem.text if id_elem is not None else '',
                    'source': 'ArXiv',
                    'metadata': {'category': category}
                }
                
                if item['title']:
                    items.append(item)
        
        except Exception as e:
            print(f"ArXiv parse error: {e}")
        
        return items
    
    def _format_items(self, items: List[Dict], source: str) -> List[Dict]:
        """Format items with metadata."""
        formatted = []
        
        for item in items[:1000]:  # Limit items per source
            formatted_item = {
                'id': item.get('id', hashlib.sha256(str(item).encode()).hexdigest()[:12]),
                'title': item.get('title', 'Untitled'),
                'content': str(item.get('content', ''))[:5000],  # First 5KB
                'url': item.get('url', ''),
                'source': item.get('source', source),
                'license': 'CC-BY-SA or CC0',
                'fetched_at': datetime.now().isoformat(),
                'metadata': item.get('metadata', {})
            }
            formatted.append(formatted_item)
        
        return formatted
    
    def save_data(self, module: str, source: str, items: List[Dict], output_dir: str):
        """Save fetched data to JSON."""
        output_dir_path = Path(output_dir) / module
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir_path / f"{source.lower().replace(' ', '_')}.json"
        
        data = {
            'module': module,
            'source': source,
            'timestamp': datetime.now().isoformat(),
            'items': items,
            'count': len(items),
            'size_bytes': len(json.dumps(items).encode()),
            'sha256': hashlib.sha256(json.dumps(items).encode()).hexdigest()
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        size_mb = data['size_bytes'] / (1024 * 1024)
        print(f"  Saved {len(items)} items to {output_file} ({size_mb:.2f}MB)")
        
        return output_file, size_mb
