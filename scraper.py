"""
OPTIMA - AI-Powered Lead Generation Scraper v3.0
ENTERPRISE EDITION - Premium Lead Generation Engine
Features: Whale Detection, Semantic Deduplication, FOMO Engine
Author: OPTIMA Team | Crore-Ready System
Version: 3.0
"""

import os
import time
import random
import json
import re
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Set
from dataclasses import dataclass, field
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Third-party imports
from duckduckgo_search import DDGS
from groq import Groq
from supabase import create_client, Client
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from fake_useragent import UserAgent
import html
from bs4 import BeautifulSoup
from urllib.parse import urlparse, quote_plus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optima_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===================== CONFIGURATION =====================

@dataclass
class Config:
    """Configuration for the OPTIMA scraper - ENTERPRISE EDITION"""
    # API Keys (Set as environment variables)
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")
    
    # Search Configuration
    CATEGORIES: List[str] = (
        "Video Editing", "Graphic Design", "Web Dev", 
        "UI/UX", "Content Writing", "SEO", "SMM", 
        "Motion Graphics", "AI Automation", "App Dev"
    )
    
    # Premium Platform Dorks
    SEARCH_QUERIES: List[str] = [
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
    ]
    
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
    
    # Quality Control
    MIN_BUDGET: int = 200  # Minimum $200 for premium leads
    MAX_RESULTS_PER_QUERY: int = 15
    MAX_THREADS: int = 3  # Parallel processing
    
    # FOMO Engine Settings
    VIEW_COUNT_RESET_HOURS: int = 24
    HOT_LEAD_THRESHOLD: int = 50  # Views needed for "Hot" tag
    
    # Semantic Deduplication
    SIMILARITY_THRESHOLD: float = 0.85
    DEDUPE_LOOKBACK_DAYS: int = 7

# ===================== AI-PROMPT TEMPLATES =====================

WHALE_DETECTION_PROMPT = """You are an expert lead qualifier for OPTIMA - a PREMIUM lead generation service charging ₹10k/month.
Your task is to identify HIGH-TICKET, PREMIUM clients only. Reject anything mediocre.

ANALYSIS STEPS:
1. WHALE IDENTIFICATION (CRITICAL):
   - Check for VERIFIED indicators: "blue tick", "verified account", "CEO/CTO/Founder", "agency/studio"
   - Look for ENTERPRISE signals: "corporate", "team", "long-term", "ongoing", "retainer"
   - Detect HIGH BUDGET: >$5000, "generous budget", "pay top dollar", "competitive rate"
   - Check AUTHORITY: Follower count mentions, company mentions, professional bio

2. LEGITIMACY FILTER (STRICT):
   - Must be HIRING post (not seeking work)
   - Must mention BUDGET or compensation
   - Must have CLEAR requirements
   - Must have CONTACT method

3. PREMIUM EXTRACTION:
   - Budget: Extract exact amount (convert all currencies to USD)
   - Timeline: "ASAP", "Urgent", "Immediate", or specific date
   - Scope: Project description clarity
   - Authority: Poster's authority level (1-10)

4. CATEGORIZATION:
   - Match to ONE category: Video Editing, Graphic Design, Web Dev, UI/UX, Content Writing, SEO, SMM, Motion Graphics, AI Automation, App Dev
   - If multiple, choose dominant one

5. QUALITY SCORING (Premium Scale 1-100):
   - Budget clarity (0-20)
   - Client authority (0-20)
   - Project clarity (0-20)
   - Urgency level (0-20)
   - Contact accessibility (0-20)

6. SEMANTIC FINGERPRINT:
   - Generate unique content fingerprint: [CATEGORY]-[BUDGET_RANGE]-[KEY_REQUIREMENT]-[TIMELINE]
   - Example: "video-editing-5000-10min-vlog-urgent"

RETURN FORMAT:
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

# ===================== PREMIUM SCRAPER CLASS =====================

class OptimaEnterpriseScraper:
    """ENTERPRISE-GRADE AI-Powered Lead Generation Engine"""
    
    def __init__(self, config: Config):
        self.config = config
        self.session = self._create_premium_session()
        self.ua = UserAgent()
        self.lock = threading.Lock()
        
        # Initialize premium clients
        self.groq_client = Groq(api_key=config.GROQ_API_KEY)
        self.supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
        
        # State management with persistence
        self.processed_urls = self._load_processed_urls()
        self.semantic_fingerprints = self._load_fingerprints()
        
        # Real-time statistics
        self.stats = {
            "total_searches": 0,
            "leads_found": 0,
            "whales_found": 0,
            "leads_saved": 0,
            "duplicates_blocked": 0,
            "errors": 0,
            "start_time": datetime.now()
        }
        
        # FOMO Engine
        self.hot_leads = set()
        
        logger.info("🚀 OPTIMA ENTERPRISE SCRAPER INITIALIZED")
        logger.info(f"💰 Whale Budget Threshold: ${config.MIN_WHALE_BUDGET}+")
    
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
        
        # Set stealth headers
        session.headers.update({
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
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
        try:
            response = self.supabase.table('scraper_state') \
                .select('processed_urls') \
                .eq('id', 1) \
                .execute()
            
            if response.data:
                return set(json.loads(response.data[0]['processed_urls']))
        except:
            pass
        return set()
    
    def _load_fingerprints(self) -> Dict[str, datetime]:
        """Load semantic fingerprints from database"""
        try:
            response = self.supabase.table('leads') \
                .select('semantic_fingerprint, created_at') \
                .gte('created_at', 
                     (datetime.now() - timedelta(days=self.config.DEDUPE_LOOKBACK_DAYS)).isoformat()) \
                .execute()
            
            return {row['semantic_fingerprint']: datetime.fromisoformat(row['created_at']) 
                   for row in response.data if row['semantic_fingerprint']}
        except:
            return {}
    
    def _save_state(self):
        """Save scraper state to database"""
        try:
            state_data = {
                'id': 1,
                'processed_urls': json.dumps(list(self.processed_urls)[-10000:]),  # Keep last 10k
                'last_updated': datetime.now().isoformat(),
                'stats': json.dumps(self.stats)
            }
            
            self.supabase.table('scraper_state').upsert(state_data).execute()
        except Exception as e:
            logger.error(f"State save error: {str(e)}")
    
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
    
    def _extract_platform_metadata(self, url: str, content: str) -> Dict:
        """Extract platform-specific metadata for whale detection"""
        platform = self._extract_platform(url)
        metadata = {
            'platform': platform,
            'is_verified': False,
            'authority_indicators': [],
            'estimated_authority': 5
        }
        
        content_lower = content.lower()
        
        # Twitter/X verification detection
        if platform == 'twitter':
            if any(keyword in content_lower for keyword in ['verified', 'blue tick', 'blue check']):
                metadata['is_verified'] = True
                metadata['estimated_authority'] = 8
            
            # Follower count detection
            follower_patterns = [
                r'(\d+\.?\d*[kKmM]?)\s*followers',
                r'followers:\s*(\d+\.?\d*[kKmM]?)',
                r'(\d+)\s*팔로워'  # Korean
            ]
            
            for pattern in follower_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    followers = match.group(1)
                    metadata['authority_indicators'].append(f"{followers} followers")
                    
                    # Convert to numeric
                    if 'k' in followers.lower():
                        num = float(followers.lower().replace('k', '')) * 1000
                    elif 'm' in followers.lower():
                        num = float(followers.lower().replace('m', '')) * 1000000
                    else:
                        num = float(followers)
                    
                    if num > 10000:
                        metadata['estimated_authority'] = min(10, metadata['estimated_authority'] + 3)
                    elif num > 1000:
                        metadata['estimated_authority'] = min(10, metadata['estimated_authority'] + 1)
        
        # LinkedIn authority detection
        elif platform == 'linkedin':
            title_patterns = [
                r'(CEO|CTO|CFO|Founder|Director|Manager|Lead|Senior)\s+at\s+(\w+)',
                r'(\w+)\s+(?:at|@)\s+(\w+)'
            ]
            
            for pattern in title_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    title = match.group(1)
                    company = match.group(2) if len(match.groups()) > 1 else "Unknown"
                    
                    if any(senior in title.lower() for senior in ['ceo', 'cto', 'cfo', 'founder', 'director']):
                        metadata['is_verified'] = True
                        metadata['estimated_authority'] = 9
                        metadata['authority_indicators'].append(f"{title} at {company}")
        
        # Enterprise keyword detection
        enterprise_keywords = ['enterprise', 'corporate', 'fortune 500', 'startup', 'series a', 'funded']
        for keyword in enterprise_keywords:
            if keyword in content_lower:
                metadata['authority_indicators'].append(keyword)
                metadata['estimated_authority'] = min(10, metadata['estimated_authority'] + 1)
        
        return metadata
    
    def _extract_platform(self, url: str) -> str:
        """Extract platform from URL with precision"""
        domain = urlparse(url).netloc.lower()
        
        platform_map = {
            'twitter.com': 'twitter',
            'x.com': 'twitter',
            'reddit.com': 'reddit',
            'linkedin.com': 'linkedin',
            'upwork.com': 'upwork',
            'fiverr.com': 'fiverr',
            'clutch.co': 'clutch',
            'indeed.com': 'indeed',
            'toptal.com': 'toptal',
            'gun.io': 'gunio',
            'facebook.com': 'facebook',
            'instagram.com': 'instagram'
        }
        
        for domain_part, platform in platform_map.items():
            if domain_part in domain:
                return platform
        
        return 'other'
    
    def _generate_semantic_fingerprint(self, content: Dict) -> str:
        """Generate unique semantic fingerprint for deduplication"""
        # Create a string of key content for hashing
        fingerprint_parts = [
            content.get('title', '')[:100],
            content.get('description', '')[:200],
            content.get('budget_usd', '0'),
            content.get('category', ''),
            content.get('timeline', '')
        ]
        
        fingerprint_text = '|'.join(str(part).lower().strip() for part in fingerprint_parts)
        
        # Remove noise (urls, special chars, extra spaces)
        fingerprint_text = re.sub(r'http\S+', '', fingerprint_text)
        fingerprint_text = re.sub(r'[^\w\s|]', '', fingerprint_text)
        fingerprint_text = re.sub(r'\s+', ' ', fingerprint_text)
        
        # Generate hash
        return hashlib.sha256(fingerprint_text.encode()).hexdigest()[:32]
    
    def _check_semantic_duplicate(self, fingerprint: str) -> bool:
        """Check if similar content already exists"""
        with self.lock:
            if fingerprint in self.semantic_fingerprints:
                # Check if it's recent enough to be considered duplicate
                stored_time = self.semantic_fingerprints[fingerprint]
                if (datetime.now() - stored_time).days <= self.config.DEDUPE_LOOKBACK_DAYS:
                    return True
        return False
    
    def _search_with_intelligent_ddg(self, query: str) -> List[Dict]:
        """Advanced DDG search with intelligent filtering"""
        try:
            with DDGS() as ddgs:
                # Use multiple search methods for better results
                results = []
                
                # Text search
                text_results = list(ddgs.text(
                    query,
                    region='wt-wt',
                    safesearch='off',
                    timelimit='d',  # Past day only
                    max_results=self.config.MAX_RESULTS_PER_QUERY
                ))
                
                # News search for recent posts
                news_results = list(ddgs.news(
                    query,
                    region='wt-wt',
                    timelimit='d',
                    max_results=5
                ))
                
                results.extend(text_results)
                results.extend(news_results)
                
                # Intelligent filtering
                filtered_results = []
                seen_titles = set()
                
                for result in results:
                    url = result.get('href', '')
                    
                    # Skip if already processed
                    if url in self.processed_urls:
                        continue
                    
                    title = result.get('title', '').strip()
                    body = result.get('body', '').strip()
                    
                    # Skip if too short
                    if len(title) < 15 or len(body) < 30:
                        continue
                    
                    # Skip common spam patterns
                    spam_patterns = [
                        r'looking for work',
                        r'hire me',
                        r'available for hire',
                        r'seeking employment',
                        r'job application',
                        r'resume',
                        r'cv attached',
                        r'applying for'
                    ]
                    
                    if any(re.search(pattern, f"{title} {body}", re.IGNORECASE) 
                           for pattern in spam_patterns):
                        continue
                    
                    # Title similarity check
                    title_hash = hashlib.md5(title.lower().encode()).hexdigest()[:16]
                    if title_hash in seen_titles:
                        continue
                    seen_titles.add(title_hash)
                    
                    # Budget mention check (premium filter)
                    budget_patterns = [
                        r'\$\d+',
                        r'\d+\s*(usd|dollars)',
                        r'budget.*?\d+',
                        r'\d+\s*(k|k?)\s*(budget|for)'
                    ]
                    
                    has_budget = any(re.search(pattern, f"{title} {body}", re.IGNORECASE) 
                                    for pattern in budget_patterns)
                    
                    if not has_budget:
                        continue  # Skip posts without budget mention
                    
                    filtered_results.append({
                        'title': html.unescape(title),
                        'description': html.unescape(body),
                        'url': url,
                        'source': 'ddg',
                        'query': query,
                        'has_budget': has_budget
                    })
                
                logger.info(f"🔍 Found {len(filtered_results)} premium results for: {query[:60]}...")
                return filtered_results
                
        except Exception as e:
            logger.error(f"DDG search failed: {str(e)}")
            self._stealth_delay()  # Extra delay on error
            return []
    
    def _analyze_with_ai(self, content: Dict, metadata: Dict) -> Optional[Dict]:
        """Analyze content with Groq AI for PREMIUM lead qualification"""
        try:
            # Enhanced context with metadata
            context = f"""
            PLATFORM METADATA:
            - Platform: {metadata['platform']}
            - Verified: {metadata['is_verified']}
            - Authority Score: {metadata['estimated_authority']}/10
            - Authority Indicators: {', '.join(metadata['authority_indicators'])}
            
            POST CONTENT:
            Title: {content['title']}
            Content: {content['description']}
            URL: {content['url']}
            
            CRITICAL ANALYSIS REQUIRED:
            1. Is this a HIGH-VALUE client (whale)?
            2. Is budget > ${self.config.MIN_WHALE_BUDGET}?
            3. Is this a legitimate hiring post?
            4. Generate semantic fingerprint for deduplication.
            """
            
            response = self.groq_client.chat.completions.create(
                model=self.config.GROQ_MODEL,
                temperature=self.config.GROQ_TEMPERATURE,
                messages=[
                    {"role": "system", "content": WHALE_DETECTION_PROMPT},
                    {"role": "user", "content": context}
                ],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            if result.get('is_legit', False):
                # Enhance with metadata
                result['platform_metadata'] = metadata
                result['is_verified'] = metadata['is_verified']
                result['authority_score'] = metadata['estimated_authority']
                
                # Validate and clean
                validated_result = self._validate_premium_lead(result, content)
                return validated_result
            
            return None
            
        except Exception as e:
            logger.error(f"AI analysis failed: {str(e)}")
            return None
    
    def _validate_premium_lead(self, ai_result: Dict, original_content: Dict) -> Dict:
        """Validate and enhance AI analysis for premium leads"""
        validated = ai_result.copy()
        
        # Ensure required fields
        validated['url'] = original_content['url']
        validated['title'] = original_content['title'][:200]
        validated['raw_content'] = original_content['description'][:1500]
        
        # Parse and validate budget
        budget_usd = validated.get('budget_usd', '0')
        
        # Convert budget to numeric
        budget_numeric = 0
        try:
            if isinstance(budget_usd, (int, float)):
                budget_numeric = float(budget_usd)
            elif isinstance(budget_usd, str):
                # Extract numbers
                num_match = re.search(r'(\d+\.?\d*)', budget_usd.replace(',', ''))
                if num_match:
                    budget_numeric = float(num_match.group(1))
                    
                    # Handle k notation
                    if 'k' in budget_usd.lower():
                        budget_numeric *= 1000
        except:
            budget_numeric = 0
        
        validated['budget_numeric'] = budget_numeric
        
        # Set client tier
        if budget_numeric >= self.config.MIN_WHALE_BUDGET:
            validated['client_tier'] = 'whale'
            validated['is_whale'] = True
        elif budget_numeric >= 1000:
            validated['client_tier'] = 'premium'
            validated['is_whale'] = False
        else:
            validated['client_tier'] = 'standard'
            validated['is_whale'] = False
        
        # Generate semantic fingerprint
        validated['semantic_fingerprint'] = self._generate_semantic_fingerprint(validated)
        
        # Set FOMO trigger
        fomo_triggers = []
        if validated.get('timeline', '').lower() in ['urgent', 'asap', 'immediate']:
            fomo_triggers.append('high_urgency')
        
        if budget_numeric >= 5000:
            fomo_triggers.append('large_budget')
        
        if validated.get('is_verified', False):
            fomo_triggers.append('verified_client')
        
        validated['fomo_triggers'] = fomo_triggers
        
        # Calculate premium priority (0-100)
        priority = validated.get('quality_score', 50)
        
        # Boost for whales
        if validated['is_whale']:
            priority = min(100, priority + 20)
        
        # Boost for urgency
        if 'high_urgency' in fomo_triggers:
            priority = min(100, priority + 15)
        
        # Boost for verified
        if validated.get('is_verified', False):
            priority = min(100, priority + 10)
        
        validated['priority_score'] = priority
        
        # Add timestamps
        validated['scraped_at'] = datetime.now().isoformat()
        validated['expires_at'] = (datetime.now() + timedelta(days=14)).isoformat()
        
        return validated
    
    def _save_premium_lead(self, lead: Dict) -> bool:
        """Save validated premium lead to Supabase with FOMO features"""
        try:
            # Check semantic duplicate
            fingerprint = lead.get('semantic_fingerprint')
            if fingerprint and self._check_semantic_duplicate(fingerprint):
                logger.info(f"⏭️ Semantic duplicate blocked: {lead['title'][:50]}...")
                self.stats['duplicates_blocked'] += 1
                return False
            
            # Prepare lead data with all premium features
            lead_data = {
                'title': lead.get('title'),
                'description': lead.get('description'),
                'raw_content': lead.get('raw_content'),
                'url': lead['url'],
                'category': lead.get('category'),
                'budget_numeric': lead.get('budget_numeric'),
                'budget_currency': 'USD',
                'budget_type': lead.get('budget_type', 'fixed'),
                'timeline': lead.get('timeline'),
                'quality_score': lead.get('quality_score', 50),
                'priority_score': lead.get('priority_score', 50),
                'platform': lead.get('platform'),
                'platform_metadata': lead.get('platform_metadata', {}),
                'contact_method': lead.get('contact_method'),
                'client_tier': lead.get('client_tier', 'standard'),
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
    
    def _process_search_result(self, result: Dict) -> bool:
        """Process individual search result with premium filters"""
        try:
            # Mark URL as processed
            self.processed_urls.add(result['url'])
            
            # Extract platform metadata for whale detection
            metadata = self._extract_platform_metadata(result['url'], result['description'])
            
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
            return False
    
    def scrape_category_parallel(self, category: str) -> int:
        """Parallel scraping for category with intelligent query building"""
        logger.info(f"🚀 Scraping category: {category}")
        leads_found = 0
        
        # Build premium queries for this category
        queries = [
            f'site:twitter.com "{category}" "budget" "hiring" "project" after:{datetime.now().strftime("%Y-%m-%d")}',
            f'site:reddit.com "{category}" "budget" "hire" "looking for"',
            f'site:linkedin.com "need {category}" "budget" "project"',
            f'"{category}" "budget" "freelancer" "hire" "urgent"',
            f'"looking for {category}" "budget" "paid" "project"',
            f'"need a {category}" "budget" "ASAP" "immediate"',
            f'"hiring {category}" "budget" "remote" "contract"',
            f'"{category} freelancer" "budget" "work" "paid"'
        ]
        
        # Process queries in parallel
        with ThreadPoolExecutor(max_workers=self.config.MAX_THREADS) as executor:
            futures = []
            
            for query in queries:
                futures.append(executor.submit(self._process_query, query))
                self._stealth_delay()  # Delay between query submissions
            
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
            
            # Search with DDG
            results = self._search_with_intelligent_ddg(query)
            
            if not results:
                return 0
            
            # Process results with thread pool for efficiency
            with ThreadPoolExecutor(max_workers=min(5, len(results))) as executor:
                result_futures = {executor.submit(self._process_search_result, result): result 
                                 for result in results}
                
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
        logger.info("🚀 STARTING OPTIMA ENTERPRISE SCRAPER")
        logger.info(f"💰 Target: ₹10K/Month Premium SaaS")
        logger.info(f"🐋 Whale Detection: ${self.config.MIN_WHALE_BUDGET}+ Budgets")
        logger.info(f"📊 Categories: {len(self.config.CATEGORIES)} Premium Skills")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # Update FOMO metrics
            self._update_fomo_metrics()
            
            # Scrape each category
            total_leads = 0
            
            for i, category in enumerate(self.config.CATEGORIES, 1):
                logger.info(f"\n📈 Category {i}/{len(self.config.CATEGORIES)}: {category}")
                
                leads = self.scrape_category_parallel(category)
                total_leads += leads
                
                logger.info(f"   ✓ Found {leads} premium leads")
                
                # Category completion delay (progressive)
                delay = random.uniform(8, 15)
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
            
            # Calculate premium metrics
            if total_leads > 0:
                whale_ratio = (self.stats['whales_found'] / total_leads) * 100
                logger.info(f"🎯 Whale Ratio: {whale_ratio:.1f}%")
            
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

# ===================== SUPABASE TABLE SETUP =====================

SUPABASE_SCHEMA = """
-- ENTERPRISE LEAD GENERATION SCHEMA FOR ₹10K/MONTH SAAS

-- Premium Leads Table
CREATE TABLE IF NOT EXISTS leads (
    id BIGSERIAL PRIMARY KEY,
    
    -- Lead Content
    title VARCHAR(500) NOT NULL,
    description TEXT NOT NULL,
    raw_content TEXT,
    url VARCHAR(1000) UNIQUE NOT NULL,
    
    -- Categorization
    category VARCHAR(100) NOT NULL,
    sub_category VARCHAR(100),
    
    -- Financials
    budget_numeric DECIMAL(12,2),
    budget_currency VARCHAR(10) DEFAULT 'USD',
    budget_type VARCHAR(50),
    client_tier VARCHAR(50) DEFAULT 'standard',
    is_whale BOOLEAN DEFAULT FALSE,
    
    -- Platform Details
    platform VARCHAR(100),
    platform_metadata JSONB DEFAULT '{}',
    contact_method VARCHAR(100),
    
    -- Quality Metrics
    quality_score INTEGER DEFAULT 50,
    priority_score INTEGER DEFAULT 50,
    authority_score INTEGER DEFAULT 5,
    is_verified BOOLEAN DEFAULT FALSE,
    
    -- Time Management
    timeline VARCHAR(100),
    scraped_at TIMESTAMP NOT NULL,
    expires_at TIMESTAMP,
    
    -- FOMO Engine
    view_count INTEGER DEFAULT 0,
    unique_viewers INTEGER DEFAULT 0,
    last_viewed_at TIMESTAMP,
    hotness_score INTEGER DEFAULT 0,
    fomo_triggers TEXT[] DEFAULT '{}',
    
    -- Semantic Deduplication
    semantic_fingerprint VARCHAR(64),
    
    -- Status Management
    is_active BOOLEAN DEFAULT TRUE,
    status VARCHAR(50) DEFAULT 'fresh',
    
    -- System
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    -- Premium Indexes
    INDEX idx_leads_category_status (category, status),
    INDEX idx_leads_priority (priority_score DESC),
    INDEX idx_leads_whale (is_whale, created_at DESC),
    INDEX idx_leads_fomo (hotness_score DESC, view_count DESC),
    INDEX idx_leads_fingerprint (semantic_fingerprint),
    INDEX idx_leads_fresh (created_at DESC) WHERE status = 'fresh'
);

-- Scraper State Table
CREATE TABLE IF NOT EXISTS scraper_state (
    id INTEGER PRIMARY KEY DEFAULT 1,
    processed_urls TEXT DEFAULT '[]',
    semantic_fingerprints JSONB DEFAULT '{}',
    last_updated TIMESTAMP DEFAULT NOW(),
    stats JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);

-- User Activity Log (for FOMO)
CREATE TABLE IF NOT EXISTS lead_views (
    id BIGSERIAL PRIMARY KEY,
    lead_id BIGINT REFERENCES leads(id) ON DELETE CASCADE,
    user_id UUID,
    viewed_at TIMESTAMP DEFAULT NOW(),
    session_id VARCHAR(100),
    
    INDEX idx_lead_views_lead (lead_id),
    INDEX idx_lead_views_time (viewed_at DESC)
);

-- Whale Alerts Table
CREATE TABLE IF NOT EXISTS whale_alerts (
    id BIGSERIAL PRIMARY KEY,
    lead_id BIGINT REFERENCES leads(id) ON DELETE CASCADE,
    alert_type VARCHAR(50),
    alert_data JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    notified_at TIMESTAMP,
    
    INDEX idx_whale_alerts_pending (notified_at) WHERE notified_at IS NULL
);

-- Real-time Function for Dashboard
CREATE OR REPLACE FUNCTION notify_new_lead()
RETURNS TRIGGER AS $$
BEGIN
    PERFORM pg_notify('new_lead', json_build_object(
        'id', NEW.id,
        'category', NEW.category,
        'title', NEW.title,
        'budget', NEW.budget_numeric,
        'is_whale', NEW.is_whale,
        'created_at', NEW.created_at
    )::text);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for real-time notifications
DROP TRIGGER IF EXISTS new_lead_trigger ON leads;
CREATE TRIGGER new_lead_trigger
    AFTER INSERT ON leads
    FOR EACH ROW
    EXECUTE FUNCTION notify_new_lead();
"""

# ===================== MAIN EXECUTION =====================

def validate_premium_environment():
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
        logger.error("2. Run Supabase Schema:")
        logger.error("   Copy the above SUPABASE_SCHEMA to Supabase SQL Editor")
        logger.error("")
        logger.error("3. Configure GitHub Actions Secrets")
        logger.error("=" * 60)
        return False
    
    return True

if __name__ == "__main__":
    """ENTRY POINT FOR ENTERPRISE SCRAPER"""
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║        🚀 OPTIMA ENTERPRISE SCRAPER v3.0                ║
    ║           ₹10K/MONTH PREMIUM SAAS ENGINE                ║
    ║                                                          ║
    ║  Features:                                               ║
    ║    • Whale Detection (>$5k Budgets)                     ║
    ║    • Semantic Deduplication AI                          ║
    ║    • FOMO Engine with Real-time Updates                 ║
    ║    • Premium Client Filtering                           ║
    ║    • Enterprise-grade Scalability                       ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Validate environment
    if not validate_premium_environment():
        exit(1)
    
    # Setup configuration
    config = Config()
    
    # Initialize enterprise scraper
    scraper = OptimaEnterpriseScraper(config)
    
    # Run the scraper
    try:
        scraper.run()
        print("\n✅ ENTERPRISE SCRAPING COMPLETE!")
        print("   Check Supabase for premium leads!")
        print("   🐋 Whales are marked for special attention")
        print("   🔥 Hot leads trigger FOMO notifications")
        print("\n🚀 Ready for ₹10K/month subscribers!")
    except Exception as e:
        logger.error(f"Scraper failed: {str(e)}")
        exit(1)
