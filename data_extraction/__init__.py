"""
data_extraction/__init__.py
"""
from .fetcher import DataFetcher
from .unifier import DataUnifier
from .deduplicator import Deduplicator
from .ebsco_scraper import EBSCOScraper

__all__ = ["DataFetcher", "DataUnifier", "Deduplicator", "EBSCOScraper"]
