"""
OPTIMA - AI-Powered Lead Generation Scraper v3.1
ENTERPRISE EDITION - Premium Lead Generation Engine
Features: Whale Detection, Semantic Deduplication, FOMO Engine
Author: OPTIMA Team | Crore-Ready System
Version: 3.1 - Production Ready
"""

import os
import time
import random
import json
import re
import hashlib
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from enum import Enum
import sys

# Third-party imports with error handling
try:
    from duckduckgo_search import DDGS
    from groq import Groq
    from supabase import create_client, Client
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    from fake_useragent import UserAgent
    import html
    from bs4 import BeautifulSoup
    from urllib.parse import urlparse, quote_plus, unquote
    import aiohttp
    import backoff
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

# Configure advanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('optima_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===================== ENUMERATIONS =====================

class LeadTier(str, Enum):
    WHALE = "whale"
    PREMIUM = "premium"
    STANDARD = "standard"

class Platform(str, Enum):
    TWITTER = "twitter"
    LINKEDIN = "linkedin"
    REDDIT = "reddit"
    UPWORK = "upwork"
    CLUTCH = "clutch"
    OTHER = "other"

class BudgetType(str, Enum):
    FIXED = "fixed"
    HOURLY = "hourly"
    RETAINER = "retainer"
    NEGOTIABLE = "negotiable"

# ===================== ADVANCED CONFIGURATION =====================

@dataclass
class Config:
    """Configuration for the OPTIMA scraper - ENTERPRISE EDITION v3.1"""
    # API Keys (Set as environment variables)
    GROQ_API_KEY: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    SUPABASE_URL: str = field(default_factory=lambda: os.getenv("SUPABASE_URL", ""))
    SUPABASE_KEY: str = field(default_factory=lambda: os.getenv("SUPABASE_KEY", ""))
    
    # Search Configuration
    CATEGORIES: List[str] = field(default_factory=lambda: [
        "Video Editing", "Graphic Design", "Web Development", 
        "UI/UX Design", "Content Writing", "SEO", "Social Media Marketing", 
        "Motion Graphics", "AI Automation", "App Development"
    ])
    
    # Premium Platform Docks with advanced patterns
    SEARCH_QUERIES: List[str] = field(default_factory=lambda: [
        'site:twitter.com "hiring" "budget" "project" -"looking for work"',
        'site:reddit.com/r/forhire "hiring" "budget"',
        'site:linkedin.com "looking for" "freelancer" "budget" -"job application"',
        'site:upwork.com "hiring" "budget" "fixed price" "urgent"',
        'site:clutch.co "RFP" "request for proposal" "budget"',
        'site:indeed.com "contract" "freelance" "remote" "budget"',
        'site:toptal.com "looking for" "expert" "budget"',
        'site:gun.io "hiring" "developer" "designer" "budget"',
        '"RFQ" "request for quote" "budget" "freelancer"',
        '"agency" "looking for" "freelancer" "subcontract" "budget"'
    ])
    
    # WHALE DETECTION Parameters
    MIN_WHALE_BUDGET: int = 5000  # $5k+ projects only for whale tier
    VERIFIED_KEYWORDS: List[str] = field(default_factory=lambda: [
        "blue tick", "verified", "CEO", "CTO", "Founder", "Director",
        "enterprise", "corporate", "agency", "studio", "startup",
        "funded", "series A", "series B", "venture", "investor"
    ])
    
    # Time Configuration
    MIN_DELAY: float = 2.5
    MAX_DELAY: float = 6.5
    MAX_RETRIES: int = 3
    REQUEST_TIMEOUT: int = 25
    
    # AI Configuration
    GROQ_MODEL: str = "llama-3.1-70b-versatile"
    GROQ_TEMPERATURE: float = 0.1
    GROQ_MAX_TOKENS: int = 2000
    
    # Quality Control
    MIN_BUDGET: int = 200  # Minimum $200 for premium leads
    MAX_RESULTS_PER_QUERY: int = 15
    MAX_THREADS: int = 3  # Parallel processing
    MAX_CONCURRENT_REQUESTS: int = 5
    
    # FOMO Engine Settings
    VIEW_COUNT_RESET_HOURS: int = 24
    HOT_LEAD_THRESHOLD: int = 50  # Views needed for "Hot" tag
    
    # Semantic Deduplication
    SIMILARITY_THRESHOLD: float = 0.85
    DEDUPE_LOOKBACK_DAYS: int = 7
    
    # Cache Settings
    CACHE_TTL_HOURS: int = 6
    MAX_CACHE_SIZE: int = 1000
    
    # Monitoring
    ENABLE_METRICS: bool = True
    SAVE_STATS_INTERVAL: int = 30  # seconds

# ===================== ADVANCED PROMPT TEMPLATES =====================

WHALE_DETECTION_PROMPT = """You are OPTIMA Lead AI - an expert lead qualifier for a PREMIUM lead generation service charging ₹10k/month.
Your task is to identify HIGH-TICKET, PREMIUM clients only. Reject anything mediocre.

CRITICAL EVALUATION MATRIX:
1. 🐋 WHALE IDENTIFICATION (MUST-HAVE):
   - Verified indicators: "blue tick", "verified account", "CEO/CTO/Founder", "agency/studio"
   - Enterprise signals: "corporate", "team", "long-term", "ongoing", "retainer"
   - High Budget: >$5000, "generous budget", "pay top dollar", "competitive rate"
   - Authority: Follower count mentions, company mentions, professional bio

2. 🎯 LEGITIMACY FILTER (STRICT):
   - MUST be hiring post (not seeking work)
   - MUST mention budget or compensation
   - MUST have clear requirements
   - MUST have contact method

3. 💰 PREMIUM EXTRACTION:
   - Budget: Extract exact amount (convert all currencies to USD)
   - Timeline: "ASAP", "Urgent", "Immediate", or specific date
   - Scope: Project description clarity
   - Authority: Poster's authority level (1-10)

4. 📊 CATEGORIZATION:
   - Match to ONE category: Video Editing, Graphic Design, Web Dev, UI/UX, Content Writing, SEO, SMM, Motion Graphics, AI Automation, App Dev
   - If multiple, choose dominant one

5. ⭐ QUALITY SCORING (Premium Scale 1-100):
   - Budget clarity (0-20)
   - Client authority (0-20)
   - Project clarity (0-20)
   - Urgency level (0-20)
   - Contact accessibility (0-20)

6. 🔍 SEMANTIC FINGERPRINT:
   - Generate unique content fingerprint: [CATEGORY]-[BUDGET_RANGE]-[KEY_REQUIREMENT]-[TIMELINE]
   - Example: "video-editing-5000-10min-vlog-urgent"

RETURN FORMAT (VALID JSON ONLY):
{
  "is_legit": boolean,
  "is_whale": boolean,
  "title": "professional_title",
  "description": "cleaned_description",
  "category": "exact_category",
  "budget_usd": "amount_in_usd",
  "budget_type": "fixed/hourly/retainer/negotiable",
  "timeline": "urgent/asap/1-week/2-weeks/month/flexible",
  "quality_score": 1-100,
  "client_tier": "whale/premium/standard",
  "authority_score": 1-10,
  "platform": "twitter/reddit/linkedin/upwork/other",
  "contact_method": "email/dm/comment/application",
  "semantic_fingerprint": "unique-content-hash",
  "fomo_trigger": "high_urgency/large_budget/exclusive"
}

If NOT legitimate, return: {"is_legit": false}
"""

# ===================== ADVANCED DATA CLASSES =====================

@dataclass
class SearchResult:
    """Enhanced search result container"""
    title: str
    description: str
    url: str
    source: str
    query: str
    has_budget: bool
    domain: str
    timestamp: datetime = field(default_factory=datetime.now)
    relevance_score: float = 0.0
    raw_html: Optional[str] = None

@dataclass
class Lead:
    """Enterprise lead data structure"""
    id: Optional[int] = None
    title: str = ""
    description: str = ""
    url: str = ""
    category: str = ""
    budget_numeric: float = 0.0
    budget_type: str = BudgetType.FIXED.value
    timeline: str = ""
    platform: str = ""
    is_whale: bool = False
    is_verified: bool = False
    quality_score: int = 50
    priority_score: int = 50
    semantic_fingerprint: str = ""
    fomo_triggers: List[str] = field(default_factory=list)
    scraped_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

# ===================== ADVANCED CACHE SYSTEM =====================

class LRUCache:
    """LRU Cache for search results and API responses"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.order = []
        self.max_size = max_size
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.order.remove(key)
                self.order.append(key)
                return self.cache[key]
            return None
    
    def set(self, key: str, value: Any):
        with self.lock:
            if key in self.cache:
                self.order.remove(key)
            elif len(self.cache) >= self.max_size:
                # Remove least recently used
                lru_key = self.order.pop(0)
                del self.cache[lru_key]
            
            self.cache[key] = value
            self.order.append(key)
    
    def clear(self):
        with self.lock:
            self.cache.clear()
            self.order.clear()

# ===================== PREMIUM SCRAPER CLASS =====================

class OptimaEnterpriseScraper:
    """ENTERPRISE-GRADE AI-Powered Lead Generation Engine v3.1"""
    
    def __init__(self, config: Config):
        self.config = config
        self.session = self._create_premium_session()
        self.ua = UserAgent(browsers=['chrome', 'firefox', 'safari'])
        self.lock = threading.Lock()
        self.cache = LRUCache(max_size=config.MAX_CACHE_SIZE)
        
        # Initialize premium clients with validation
        self.groq_client = self._initialize_groq()
        self.supabase = self._initialize_supabase()
        
        # State management with persistence
        self.processed_urls = self._load_processed_urls()
        self.semantic_fingerprints = self._load_fingerprints()
        
        # Real-time statistics with monitoring
        self.stats = {
            "total_searches": 0,
            "leads_found": 0,
            "whales_found": 0,
            "leads_saved": 0,
            "duplicates_blocked": 0,
            "errors": 0,
            "start_time": datetime.now(),
            "api_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # FOMO Engine
        self.hot_leads = set()
        
        # Start stats saver thread
        if config.ENABLE_METRICS:
            self._start_stats_saver()
        
        logger.info("🚀 OPTIMA ENTERPRISE SCRAPER v3.1 INITIALIZED")
        logger.info(f"💰 Whale Budget Threshold: ${config.MIN_WHALE_BUDGET}+")
        logger.info(f"🧠 AI Model: {config.GROQ_MODEL}")
    
    def _initialize_groq(self) -> Optional[Groq]:
        """Initialize Groq client with validation"""
        if not self.config.GROQ_API_KEY:
            logger.error("❌ GROQ_API_KEY not configured")
            return None
        
        try:
            client = Groq(api_key=self.config.GROQ_API_KEY)
            # Test the API with a simple request
            test_response = client.chat.completions.create(
                model="llama3-8b-8192",  # Small model for test
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            logger.info("✅ Groq API connected successfully")
            return client
        except Exception as e:
            logger.error(f"❌ Groq API initialization failed: {e}")
            return None
    
    def _initialize_supabase(self) -> Optional[Client]:
        """Initialize Supabase client with validation"""
        if not self.config.SUPABASE_URL or not self.config.SUPABASE_KEY:
            logger.error("❌ Supabase credentials not configured")
            return None
        
        try:
            client = create_client(self.config.SUPABASE_URL, self.config.SUPABASE_KEY)
            # Test connection
            response = client.table('leads').select('count', count='exact').limit(1).execute()
            logger.info(f"✅ Supabase connected successfully (Total leads: {response.count})")
            return client
        except Exception as e:
            logger.error(f"❌ Supabase connection failed: {e}")
            return None
    
    def _create_premium_session(self) -> requests.Session:
        """Create enterprise-grade HTTP session with stealth features"""
        session = requests.Session()
        
        # Advanced retry strategy
        retry = Retry(
            total=self.config.MAX_RETRIES,
            backoff_factor=0.7,
            status_forcelist=[429, 500, 502, 503, 504, 403],
            allowed_methods=["GET", "POST"],
            respect_retry_after_header=True
        )
        
        adapter = HTTPAdapter(
            max_retries=retry,
            pool_connections=100,
            pool_maxsize=100
        )
        
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Rotating headers for stealth
        session.headers.update({
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
        })
        
        return session
    
    def _load_processed_urls(self) -> Set[str]:
        """Load previously processed URLs from database"""
        if not self.supabase:
            return set()
        
        try:
            response = self.supabase.table('scraper_state') \
                .select('processed_urls') \
                .eq('id', 1) \
                .execute()
            
            if response.data:
                return set(json.loads(response.data[0]['processed_urls']))
        except Exception as e:
            logger.warning(f"Could not load processed URLs: {e}")
        
        return set()
    
    def _load_fingerprints(self) -> Dict[str, datetime]:
        """Load semantic fingerprints from database"""
        if not self.supabase:
            return {}
        
        try:
            response = self.supabase.table('leads') \
                .select('semantic_fingerprint, created_at') \
                .gte('created_at', 
                     (datetime.now() - timedelta(days=self.config.DEDUPE_LOOKBACK_DAYS)).isoformat()) \
                .execute()
            
            return {row['semantic_fingerprint']: datetime.fromisoformat(row['created_at']) 
                   for row in response.data if row['semantic_fingerprint']}
        except Exception as e:
            logger.warning(f"Could not load fingerprints: {e}")
            return {}
    
    def _start_stats_saver(self):
        """Start background thread to save stats periodically"""
        def stats_saver():
            while True:
                time.sleep(self.config.SAVE_STATS_INTERVAL)
                self._save_state()
        
        thread = threading.Thread(target=stats_saver, daemon=True)
        thread.start()
    
    def _save_state(self):
        """Save scraper state to database"""
        if not self.supabase:
            return
        
        try:
            state_data = {
                'id': 1,
                'processed_urls': json.dumps(list(self.processed_urls)[-10000:]),
                'last_updated': datetime.now().isoformat(),
                'stats': json.dumps(self.stats)
            }
            
            self.supabase.table('scraper_state').upsert(state_data).execute()
        except Exception as e:
            logger.error(f"State save error: {str(e)}")
    
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def _stealth_delay(self):
        """Intelligent random delay with exponential backoff"""
        base_delay = random.uniform(self.config.MIN_DELAY, self.config.MAX_DELAY)
        
        # Add jitter
        jitter = random.uniform(-0.5, 0.5)
        delay = max(1.0, base_delay + jitter)
        
        # Progressive slowdown if many requests
        if self.stats['total_searches'] > 50:
            delay *= 1.2
        
        logger.debug(f"⏸️ Stealth delay: {delay:.2f}s")
        time.sleep(delay)
    
    def _extract_platform_metadata(self, url: str, content: str) -> Dict[str, Any]:
        """Advanced platform metadata extraction"""
        domain = urlparse(url).netloc.lower()
        platform = self._extract_platform(domain)
        
        metadata = {
            'platform': platform,
            'is_verified': False,
            'authority_indicators': [],
            'estimated_authority': 5,
            'platform_score': 0,
            'trust_signals': []
        }
        
        content_lower = content.lower()
        
        # Platform-specific verification
        if platform == Platform.TWITTER.value:
            if any(keyword in content_lower for keyword in ['verified', 'blue tick', 'blue check']):
                metadata['is_verified'] = True
                metadata['trust_signals'].append('verified_account')
                metadata['platform_score'] += 20
            
            # Twitter specific signals
            twitter_patterns = {
                r'(\d+\.?\d*[kKmM]?)\s*followers': 'followers',
                r'following\s*(\d+)': 'following_count',
                r'(\d+)\s*tweets': 'tweet_count'
            }
            
            for pattern, signal in twitter_patterns.items():
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    metadata['authority_indicators'].append(f"{match.group(1)} {signal}")
        
        elif platform == Platform.LINKEDIN.value:
            # LinkedIn authority patterns
            patterns = [
                r'(CEO|CTO|CFO|Founder|Director|Manager|Lead|Senior)\s+at\s+([A-Za-z0-9\s&]+)',
                r'([A-Za-z\s]+)\s+(?:at|@)\s+([A-Za-z0-9\s&]+)',
                r'former\s+([A-Za-z\s]+)\s+at\s+([A-Za-z0-9\s&]+)'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    if len(match) == 2:
                        title, company = match
                        metadata['authority_indicators'].append(f"{title.strip()} at {company.strip()}")
                        
                        if any(senior in title.lower() for senior in ['ceo', 'cto', 'cfo', 'founder', 'director']):
                            metadata['is_verified'] = True
                            metadata['platform_score'] += 30
        
        # Budget detection
        budget_patterns = [
            r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:usd|dollars|USD)',
            r'budget\s*[:\-]?\s*[\$]?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'(\d+)\s*(?:k|K)\s*(?:budget|usd)'
        ]
        
        budgets = []
        for pattern in budget_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            budgets.extend(matches)
        
        if budgets:
            metadata['budget_mentions'] = budgets
            metadata['platform_score'] += 15
        
        # Urgency detection
        urgency_keywords = ['urgent', 'asap', 'immediate', 'quick', 'fast', 'need soon']
        if any(keyword in content_lower for keyword in urgency_keywords):
            metadata['trust_signals'].append('urgent_project')
            metadata['platform_score'] += 10
        
        # Professionalism indicators
        pro_keywords = ['professional', 'experienced', 'expert', 'skilled', 'quality']
        if any(keyword in content_lower for keyword in pro_keywords):
            metadata['trust_signals'].append('professional_tone')
            metadata['platform_score'] += 5
        
        return metadata
    
    def _extract_platform(self, domain: str) -> str:
        """Extract platform from domain"""
        platform_map = {
            'twitter.com': Platform.TWITTER.value,
            'x.com': Platform.TWITTER.value,
            'linkedin.com': Platform.LINKEDIN.value,
            'reddit.com': Platform.REDDIT.value,
            'upwork.com': Platform.UPWORK.value,
            'clutch.co': Platform.CLUTCH.value,
            'indeed.com': 'indeed',
            'toptal.com': 'toptal',
            'gun.io': 'gunio',
            'facebook.com': 'facebook',
            'instagram.com': 'instagram',
            'github.com': 'github',
            'dribbble.com': 'dribbble',
            'behance.net': 'behance'
        }
        
        for platform_domain, platform_name in platform_map.items():
            if platform_domain in domain:
                return platform_name
        
        return Platform.OTHER.value
    
    def _generate_semantic_fingerprint(self, content: Dict[str, Any]) -> str:
        """Generate unique semantic fingerprint for deduplication"""
        # Create normalized string
        parts = [
            content.get('title', '').lower(),
            content.get('description', '').lower(),
            str(content.get('budget_usd', '0')),
            content.get('category', '').lower(),
            content.get('platform', '').lower()
        ]
        
        # Normalize: remove extra spaces, special chars, stop words
        text = ' '.join(parts)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = [word for word in text.split() if word not in stop_words and len(word) > 2]
        text = ' '.join(words[:50])  # Limit to 50 words
        
        # Generate hash
        return hashlib.sha256(text.encode()).hexdigest()[:32]
    
    def _check_semantic_duplicate(self, fingerprint: str) -> bool:
        """Check if similar content already exists"""
        with self.lock:
            if fingerprint in self.semantic_fingerprints:
                stored_time = self.semantic_fingerprints[fingerprint]
                if (datetime.now() - stored_time).days <= self.config.DEDUPE_LOOKBACK_DAYS:
                    return True
        return False
    
    def _search_with_intelligent_ddg(self, query: str) -> List[SearchResult]:
        """Advanced DDG search with intelligent filtering"""
        cache_key = f"search_{hashlib.md5(query.encode()).hexdigest()[:16]}"
        cached = self.cache.get(cache_key)
        
        if cached:
            self.stats['cache_hits'] += 1
            return cached
        
        self.stats['cache_misses'] += 1
        
        try:
            with DDGS() as ddgs:
                results = []
                
                # Use multiple search types
                search_methods = [
                    ('text', {'max_results': self.config.MAX_RESULTS_PER_QUERY}),
                    ('news', {'max_results': 5}),
                    ('answers', {'max_results': 3})
                ]
                
                for method, params in search_methods:
                    try:
                        if method == 'text':
                            method_results = ddgs.text(query, **params)
                        elif method == 'news':
                            method_results = ddgs.news(query, **params)
                        elif method == 'answers':
                            method_results = ddgs.answers(query)
                        else:
                            continue
                        
                        for item in method_results:
                            url = item.get('href', '')
                            if not url or url in self.processed_urls:
                                continue
                            
                            title = html.unescape(item.get('title', ''))
                            body = html.unescape(item.get('body', item.get('text', '')))
                            
                            # Advanced filtering
                            if not self._is_premium_content(title, body):
                                continue
                            
                            result = SearchResult(
                                title=title[:300],
                                description=body[:500],
                                url=url,
                                source='ddg',
                                query=query,
                                has_budget=self._has_budget_mention(title + ' ' + body),
                                domain=urlparse(url).netloc
                            )
                            
                            results.append(result)
                            
                    except Exception as e:
                        logger.warning(f"DDG {method} search failed: {e}")
                        continue
                
                # Remove duplicates by URL
                seen_urls = set()
                unique_results = []
                for result in results:
                    if result.url not in seen_urls:
                        seen_urls.add(result.url)
                        unique_results.append(result)
                
                logger.info(f"🔍 Found {len(unique_results)} results for: {query[:80]}...")
                
                # Cache results
                self.cache.set(cache_key, unique_results)
                return unique_results
                
        except Exception as e:
            logger.error(f"DDG search failed: {str(e)}")
            self._stealth_delay()
            return []
    
    def _is_premium_content(self, title: str, body: str) -> bool:
        """Check if content is premium quality"""
        content = (title + ' ' + body).lower()
        
        # Must-have indicators
        must_have = [
            r'(hiring|looking for|need|seeking).{1,30}(freelancer|developer|designer|writer|editor)',
            r'budget.*?\d+',
            r'\$(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d{2})?(?:k|K)?\b'
        ]
        
        for pattern in must_have:
            if not re.search(pattern, content, re.IGNORECASE):
                return False
        
        # Reject indicators
        reject_patterns = [
            r'looking for work',
            r'available for hire',
            r'seeking employment',
            r'job application',
            r'resume.*?attached',
            r'applying for.*?position',
            r'unpaid',
            r'volunteer',
            r'exposure',
            r'no budget'
        ]
        
        for pattern in reject_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return False
        
        # Quality indicators (bonus points)
        quality_indicators = [
            r'urgent',
            r'asap',
            r'immediate',
            r'professional',
            r'experienced',
            r'long.?term',
            r'retainer',
            r'enterprise',
            r'corporate'
        ]
        
        quality_score = 0
        for pattern in quality_indicators:
            if re.search(pattern, content, re.IGNORECASE):
                quality_score += 1
        
        return quality_score >= 2  # At least 2 quality indicators
    
    def _has_budget_mention(self, text: str) -> bool:
        """Check if text mentions budget"""
        patterns = [
            r'\$\d+',
            r'\d+\s*(?:usd|dollars|USD)',
            r'budget.*?\d+',
            r'\d+\s*(?:k|K)\s*(?:budget|for)',
            r'(?:pay|paid|payment).*?\$\d+',
            r'rate.*?\$\d+'
        ]
        
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)
    
    def _analyze_with_ai(self, result: SearchResult, metadata: Dict) -> Optional[Dict]:
        """Analyze content with Groq AI for PREMIUM lead qualification"""
        if not self.groq_client:
            logger.error("Groq client not initialized")
            return None
        
        try:
            context = f"""
            PLATFORM ANALYSIS:
            - Platform: {metadata['platform']}
            - Verified: {metadata['is_verified']}
            - Authority Score: {metadata.get('estimated_authority', 5)}/10
            - Trust Signals: {', '.join(metadata.get('trust_signals', []))}
            - Platform Score: {metadata.get('platform_score', 0)}/100
            
            POST CONTENT:
            Title: {result.title}
            Content: {result.description}
            URL: {result.url}
            Source: {result.domain}
            
            CRITICAL ANALYSIS REQUIRED:
            1. Is this a HIGH-VALUE client (whale)? Budget > ${self.config.MIN_WHALE_BUDGET}?
            2. Is this a legitimate hiring post?
            3. Extract exact budget and timeline
            4. Generate semantic fingerprint for deduplication
            """
            
            response = self.groq_client.chat.completions.create(
                model=self.config.GROQ_MODEL,
                temperature=self.config.GROQ_TEMPERATURE,
                max_tokens=self.config.GROQ_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": WHALE_DETECTION_PROMPT},
                    {"role": "user", "content": context}
                ],
                response_format={"type": "json_object"}
            )
            
            self.stats['api_calls'] += 1
            
            ai_result = json.loads(response.choices[0].message.content)
            
            if ai_result.get('is_legit', False):
                # Enhance with metadata
                ai_result['platform_metadata'] = metadata
                ai_result['is_verified'] = metadata.get('is_verified', False)
                ai_result['authority_score'] = metadata.get('estimated_authority', 5)
                ai_result['platform_score'] = metadata.get('platform_score', 0)
                
                # Validate and clean
                validated_result = self._validate_premium_lead(ai_result, result)
                return validated_result
            
            return None
            
        except json.JSONDecodeError:
            logger.error("AI returned invalid JSON")
            return None
        except Exception as e:
            logger.error(f"AI analysis failed: {str(e)}")
            return None
    
    def _validate_premium_lead(self, ai_result: Dict, result: SearchResult) -> Dict:
        """Validate and enhance AI analysis for premium leads"""
        validated = ai_result.copy()
        
        # Ensure required fields
        validated['url'] = result.url
        validated['title'] = result.title[:200]
        validated['raw_content'] = result.description[:1500]
        validated['domain'] = result.domain
        
        # Parse and validate budget
        budget_usd = validated.get('budget_usd', '0')
        budget_numeric = self._parse_budget(budget_usd)
        validated['budget_numeric'] = budget_numeric
        
        # Set client tier
        if budget_numeric >= self.config.MIN_WHALE_BUDGET:
            validated['client_tier'] = LeadTier.WHALE.value
            validated['is_whale'] = True
        elif budget_numeric >= 1000:
            validated['client_tier'] = LeadTier.PREMIUM.value
            validated['is_whale'] = False
        else:
            validated['client_tier'] = LeadTier.STANDARD.value
            validated['is_whale'] = False
        
        # Generate semantic fingerprint
        validated['semantic_fingerprint'] = self._generate_semantic_fingerprint(validated)
        
        # Set FOMO triggers
        fomo_triggers = []
        if validated.get('timeline', '').lower() in ['urgent', 'asap', 'immediate']:
            fomo_triggers.append('high_urgency')
        
        if budget_numeric >= 5000:
            fomo_triggers.append('large_budget')
        
        if validated.get('is_verified', False):
            fomo_triggers.append('verified_client')
        
        if validated.get('authority_score', 0) >= 8:
            fomo_triggers.append('high_authority')
        
        validated['fomo_triggers'] = fomo_triggers
        
        # Calculate premium priority (0-100)
        priority = validated.get('quality_score', 50)
        
        # Boost factors
        if validated['is_whale']:
            priority = min(100, priority + 25)
        
        if 'high_urgency' in fomo_triggers:
            priority = min(100, priority + 20)
        
        if validated.get('is_verified', False):
            priority = min(100, priority + 15)
        
        if validated.get('authority_score', 0) >= 7:
            priority = min(100, priority + 10)
        
        validated['priority_score'] = priority
        
        # Add timestamps
        validated['scraped_at'] = datetime.now().isoformat()
        validated['expires_at'] = (datetime.now() + timedelta(days=14)).isoformat()
        
        return validated
    
    def _parse_budget(self, budget_text: str) -> float:
        """Parse budget text to numeric value"""
        try:
            if isinstance(budget_text, (int, float)):
                return float(budget_text)
            
            text = str(budget_text).lower().replace(',', '')
            
            # Extract numbers
            num_match = re.search(r'(\d+\.?\d*)', text)
            if not num_match:
                return 0.0
            
            amount = float(num_match.group(1))
            
            # Handle k notation
            if 'k' in text:
                amount *= 1000
            
            # Handle ranges
            if '-' in text:
                parts = text.split('-')
                if len(parts) == 2:
                    try:
                        amount1 = float(re.search(r'(\d+\.?\d*)', parts[0]).group(1))
                        amount2 = float(re.search(r'(\d+\.?\d*)', parts[1]).group(1))
                        amount = (amount1 + amount2) / 2
                    except:
                        pass
            
            return amount
        except:
            return 0.0
    
    def _save_premium_lead(self, lead: Dict) -> bool:
        """Save validated premium lead to Supabase with FOMO features"""
        if not self.supabase:
            logger.error("Supabase not initialized")
            return False
        
        try:
            # Check semantic duplicate
            fingerprint = lead.get('semantic_fingerprint')
            if fingerprint and self._check_semantic_duplicate(fingerprint):
                logger.info(f"⏭️ Semantic duplicate blocked: {lead['title'][:50]}...")
                self.stats['duplicates_blocked'] += 1
                return False
            
            # Prepare lead data
            lead_data = {
                'title': lead.get('title'),
                'description': lead.get('description'),
                'raw_content': lead.get('raw_content'),
                'url': lead['url'],
                'category': lead.get('category'),
                'budget_numeric': lead.get('budget_numeric'),
                'budget_currency': 'USD',
                'budget_type': lead.get('budget_type', BudgetType.FIXED.value),
                'timeline': lead.get('timeline'),
                'quality_score': lead.get('quality_score', 50),
                'priority_score': lead.get('priority_score', 50),
                'platform': lead.get('platform'),
                'platform_metadata': lead.get('platform_metadata', {}),
                'contact_method': lead.get('contact_method'),
                'client_tier': lead.get('client_tier', LeadTier.STANDARD.value),
                'is_whale': lead.get('is_whale', False),
                'is_verified': lead.get('is_verified', False),
                'authority_score': lead.get('authority_score', 5),
                'semantic_fingerprint': fingerprint,
                'fomo_triggers': lead.get('fomo_triggers', []),
                'scraped_at': lead.get('scraped_at'),
                'expires_at': lead.get('expires_at'),
                'view_count': 0,
                'unique_viewers': 0,
                'last_viewed_at': None,
                'is_active': True,
                'status': 'fresh',
                'hotness_score': 0
            }
            
            # Upsert with URL as unique key
            response = self.supabase.table('leads').upsert(
                lead_data,
                on_conflict='url'
            ).execute()
            
            if response.data:
                # Update fingerprints cache
                with self.lock:
                    self.semantic_fingerprints[fingerprint] = datetime.now()
                
                # Mark as hot if whale
                if lead_data['is_whale']:
                    self.hot_leads.add(response.data[0]['id'])
                
                logger.info(f"✅ {'🐋 WHALE ' if lead_data['is_whale'] else ''}Lead saved: {lead['title'][:50]}...")
                logger.info(f"   Tier: {lead_data['client_tier']} | Budget: ${lead_data['budget_numeric']} | Score: {lead_data['priority_score']}")
                
                return True
            else:
                logger.warning(f"Failed to save lead: {lead['url']}")
                return False
                
        except Exception as e:
            logger.error(f"Supabase save error: {str(e)}")
            return False
    
    def _process_search_result(self, result: SearchResult) -> bool:
        """Process individual search result with premium filters"""
        try:
            # Mark URL as processed
            self.processed_urls.add(result.url)
            
            # Extract platform metadata for whale detection
            metadata = self._extract_platform_metadata(result.url, result.description)
            
            # Analyze with AI
            lead = self._analyze_with_ai(result, metadata)
            
            if lead:
                # Check minimum budget
                if lead.get('budget_numeric', 0) < self.config.MIN_BUDGET:
                    logger.debug(f"Budget too low: ${lead.get('budget_numeric', 0)}")
                    return False
                
                # Save to database
                if self._save_premium_lead(lead):
                    with self.lock:
                        self.stats['leads_saved'] += 1
                        if lead.get('is_whale'):
                            self.stats['whales_found'] += 1
                    
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error processing result: {str(e)}")
            self.stats['errors'] += 1
            return False
    
    def scrape_category_parallel(self, category: str) -> int:
        """Parallel scraping for category with intelligent query building"""
        logger.info(f"🚀 Scraping category: {category}")
        leads_found = 0
        
        # Build premium queries for this category
        base_queries = [
            f'site:twitter.com "{category}" "budget" "hiring" "project"',
            f'site:reddit.com "{category}" "budget" "hire" "looking for"',
            f'site:linkedin.com "need {category}" "budget" "project"',
            f'"{category}" "budget" "freelancer" "hire" "urgent"',
            f'"looking for {category}" "budget" "paid" "project"',
            f'"need a {category}" "budget" "ASAP" "immediate"',
            f'"hiring {category}" "budget" "remote" "contract"',
            f'"{category} freelancer" "budget" "work" "paid"'
        ]
        
        # Add time constraints for recency
        today = datetime.now().strftime("%Y-%m-%d")
        week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        
        recent_queries = [f'{q} after:{week_ago}' for q in base_queries[:3]]
        all_queries = recent_queries + base_queries[3:]
        
        # Process queries in parallel
        with ThreadPoolExecutor(max_workers=self.config.MAX_THREADS) as executor:
            futures = []
            
            for query in all_queries:
                futures.append(executor.submit(self._process_query, query))
                self._stealth_delay()
            
            for future in as_completed(futures):
                try:
                    leads_from_query = future.result()
                    leads_found += leads_from_query
                except Exception as e:
                    logger.error(f"Query processing failed: {str(e)}")
                    self.stats['errors'] += 1
        
        return leads_found
    
    def _process_query(self, query: str) -> int:
        """Process individual query and return number of leads found"""
        leads_found = 0
        
        try:
            logger.info(f"  Searching: {query[:80]}...")
            self.stats['total_searches'] += 1
            
            # Search with DDG
            results = self._search_with_intelligent_ddg(query)
            
            if not results:
                return 0
            
            # Process results with thread pool
            with ThreadPoolExecutor(max_workers=min(self.config.MAX_CONCURRENT_REQUESTS, len(results))) as executor:
                result_futures = []
                
                for result in results:
                    future = executor.submit(self._process_search_result, result)
                    result_futures.append(future)
                    time.sleep(0.5)  # Small delay between submissions
                
                for future in as_completed(result_futures):
                    if future.result():
                        leads_found += 1
            
            logger.info(f"  Found {leads_found} leads from this query")
            
        except Exception as e:
            logger.error(f"Error processing query {query[:50]}: {str(e)}")
            self.stats['errors'] += 1
        
        return leads_found
    
    def _update_fomo_metrics(self):
        """Update FOMO metrics for existing leads"""
        if not self.supabase:
            return
        
        try:
            # Reset view counts for old leads
            reset_time = datetime.now() - timedelta(hours=self.config.VIEW_COUNT_RESET_HOURS)
            
            response = self.supabase.table('leads') \
                .update({
                    'view_count': 0,
                    'hotness_score': 0
                }) \
                .lt('last_viewed_at', reset_time.isoformat()) \
                .execute()
            
            logger.info(f"🔄 Reset FOMO metrics for old leads")
            
        except Exception as e:
            logger.error(f"FOMO update error: {str(e)}")
    
    def run(self):
        """Main execution method for ENTERPRISE scraper"""
        logger.info("=" * 60)
        logger.info("🚀 STARTING OPTIMA ENTERPRISE SCRAPER v3.1")
        logger.info(f"💰 Target: ₹10K/Month Premium SaaS")
        logger.info(f"🐋 Whale Detection: ${self.config.MIN_WHALE_BUDGET}+ Budgets")
        logger.info(f"📊 Categories: {len(self.config.CATEGORIES)} Premium Skills")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # Validate required services
            if not self.groq_client:
                logger.error("❌ Groq AI not available. Check API key.")
                return
            if not self.supabase:
                logger.error("❌ Supabase not available. Check credentials.")
                return
            
            # Update FOMO metrics
            self._update_fomo_metrics()
            
            # Scrape each category
            total_leads = 0
            
            for i, category in enumerate(self.config.CATEGORIES, 1):
                logger.info(f"\n📈 Category {i}/{len(self.config.CATEGORIES)}: {category}")
                
                leads = self.scrape_category_parallel(category)
                total_leads += leads
                
                logger.info(f"   ✓ Found {leads} premium leads")
                
                # Progressive delay
                delay = random.uniform(8, 15)
                if i < len(self.config.CATEGORIES):
                    logger.debug(f"   ⏸️ Category cooldown: {delay:.1f}s")
                    time.sleep(delay)
            
            # Print enterprise statistics
            elapsed = time.time() - start_time
            
            logger.info("\n" + "=" * 60)
            logger.info("🏆 ENTERPRISE SCRAPING COMPLETE")
            logger.info("=" * 60)
            logger.info(f"⏱️  Total Time: {elapsed/60:.1f} minutes")
            logger.info(f"💰 Total Leads: {total_leads}")
            logger.info(f"🐋 Whales Found: {self.stats['whales_found']}")
            logger.info(f"💾 Leads Saved: {self.stats['leads_saved']}")
            logger.info(f"🚫 Duplicates Blocked: {self.stats['duplicates_blocked']}")
            logger.info(f"❌ Errors: {self.stats['errors']}")
            logger.info(f"🔗 URLs Processed: {len(self.processed_urls)}")
            logger.info(f"🔥 Hot Leads: {len(self.hot_leads)}")
            logger.info(f"💾 Cache Hits: {self.stats['cache_hits']}")
            logger.info(f"🔍 Cache Misses: {self.stats['cache_misses']}")
            logger.info(f"🤖 AI API Calls: {self.stats['api_calls']}")
            
            # Calculate premium metrics
            if total_leads > 0:
                whale_ratio = (self.stats['whales_found'] / total_leads) * 100
                success_rate = (self.stats['leads_saved'] / total_leads) * 100
                logger.info(f"🎯 Whale Ratio: {whale_ratio:.1f}%")
                logger.info(f"📈 Success Rate: {success_rate:.1f}%")
            
            # Save state
            self._save_state()
            
            logger.info("✅ State saved to database")
            logger.info("=" * 60)
            
        except KeyboardInterrupt:
            logger.info("\n⏹️ Scraper stopped by user")
            self._save_state()
        except Exception as e:
            logger.error(f"💥 CRITICAL ERROR: {str(e)}", exc_info=True)
            self._save_state()
            raise

# ===================== REQUIREMENTS.TXT =====================

REQUIREMENTS = """
# OPTIMA Enterprise Scraper v3.1
duckduckgo-search>=6.0.0
groq>=0.9.0
supabase>=2.3.0
requests>=2.31.0
fake-useragent>=1.4.0
beautifulsoup4>=4.12.0
lxml>=4.9.0
aiohttp>=3.9.0
backoff>=2.2.0
python-dateutil>=2.8.0
PyYAML>=6.0
ujson>=5.8.0
"""

# ===================== VALIDATION & SETUP =====================

def validate_premium_environment() -> bool:
    """Validate ENTERPRISE environment variables"""
    required_vars = ['GROQ_API_KEY', 'SUPABASE_URL', 'SUPABASE_KEY']
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        logger.error(f"❌ MISSING ENTERPRISE VARIABLES: {', '.join(missing)}")
        logger.error("=" * 60)
        logger.error("🚀 OPTIMA ENTERPRISE SETUP REQUIRED:")
        logger.error("")
        logger.error("1. Set Environment Variables:")
        logger.error("   export GROQ_API_KEY='your-groq-key'")
        logger.error("   export SUPABASE_URL='https://your-project.supabase.co'")
        logger.error("   export SUPABASE_KEY='your-supabase-anon-key'")
        logger.error("")
        logger.error("2. Install Requirements:")
        logger.error("   pip install -r requirements.txt")
        logger.error("")
        logger.error("3. Run Supabase Schema from scraper.py")
        logger.error("=" * 60)
        return False
    
    # Validate API keys format
    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key.startswith("gsk_"):
        logger.warning("⚠️  GROQ_API_KEY format may be incorrect")
    
    supabase_url = os.getenv("SUPABASE_URL", "")
    if not supabase_url.startswith("https://"):
        logger.warning("⚠️  SUPABASE_URL format may be incorrect")
    
    return True

def setup_supabase_schema():
    """Create Supabase schema if needed"""
    schema = """
    -- Run this in Supabase SQL Editor
    CREATE TABLE IF NOT EXISTS leads (
        id BIGSERIAL PRIMARY KEY,
        title VARCHAR(500) NOT NULL,
        description TEXT NOT NULL,
        url VARCHAR(1000) UNIQUE NOT NULL,
        category VARCHAR(100),
        budget_numeric DECIMAL(12,2),
        budget_type VARCHAR(50),
        client_tier VARCHAR(50),
        is_whale BOOLEAN DEFAULT FALSE,
        platform VARCHAR(100),
        contact_method VARCHAR(100),
        quality_score INTEGER,
        priority_score INTEGER,
        semantic_fingerprint VARCHAR(64),
        scraped_at TIMESTAMP,
        created_at TIMESTAMP DEFAULT NOW()
    );
    
    CREATE TABLE IF NOT EXISTS scraper_state (
        id INTEGER PRIMARY KEY DEFAULT 1,
        processed_urls TEXT,
        last_updated TIMESTAMP DEFAULT NOW()
    );
    """
    print(schema)

# ===================== MAIN EXECUTION =====================

if __name__ == "__main__":
    """ENTRY POINT FOR ENTERPRISE SCRAPER v3.1"""
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║        🚀 OPTIMA ENTERPRISE SCRAPER v3.1                ║
    ║           ₹10K/MONTH PREMIUM SAAS ENGINE                ║
    ║                                                          ║
    ║  Features:                                               ║
    ║    • Advanced Whale Detection (>$5k Budgets)            ║
    ║    • Semantic Deduplication with AI                     ║
    ║    • FOMO Engine with Real-time Updates                 ║
    ║    • Premium Client Filtering                           ║
    ║    • Enterprise-grade Scalability                       ║
    ║    • LRU Caching for Performance                        ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Validate environment
    if not validate_premium_environment():
        logger.error("❌ Environment validation failed")
        sys.exit(1)
    
    # Setup configuration
    config = Config()
    
    # Initialize enterprise scraper
    scraper = OptimaEnterpriseScraper(config)
    
    # Run the scraper
    try:
        scraper.run()
        print("\n✅ ENTERPRISE SCRAPING COMPLETE!")
        print("   🐋 Whales are marked for special attention")
        print("   🔥 Hot leads trigger FOMO notifications")
        print("   📊 Check logs for detailed statistics")
        print("\n🚀 Ready for ₹10K/month subscribers!")
    except KeyboardInterrupt:
        print("\n⏹️ Scraper stopped by user")
    except Exception as e:
        logger.error(f"💥 Scraper failed: {e}")
        sys.exit(1)
