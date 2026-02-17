"""Enrich startup ideas with structured fields using Mistral API.

This script reads ideas.csv and enriches each startup idea with:
- domain, business_model, estimated_cost_bucket, difficulty
- scalability, required_skills, target_customer, short_summary

Uses Mistral chat completion API with structured JSON output.
Processes in batches with progress logging and error handling.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from src.config import Config
from src.mistral_client import MistralClient


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IdeaEnricher:
    """Enriches startup ideas with structured fields using Mistral API."""
    
    ENRICHMENT_PROMPT_TEMPLATE = """You are a startup analysis expert. Analyze the following startup idea and provide structured classification.

Startup Idea: {idea}

Provide your analysis as a JSON object with these exact fields:
- domain: Choose ONE from [FinTech, HealthTech, EdTech, AI/ML, E-commerce, Sustainability, SaaS, Marketplace, HRTech, PropTech, FoodTech, Other]
- business_model: Choose from [B2B, B2C, Both]
- estimated_cost_bucket: Choose from [<1000, 1000-10000, 10000-50000, 50000+]
- difficulty: Choose from [Low, Medium, High]
- scalability: Choose from [Low, Medium, High]
- required_skills: Comma-separated list of 3-5 key skills needed
- target_customer: Short phrase describing primary customer (max 10 words)
- short_summary: Refined 1-2 sentence summary of the idea (max 50 words)

Return ONLY the JSON object, no explanations or markdown formatting."""

    VALID_DOMAINS = [
        "FinTech", "HealthTech", "EdTech", "AI/ML", "E-commerce",
        "Sustainability", "SaaS", "Marketplace", "HRTech", "PropTech",
        "FoodTech", "Other"
    ]
    VALID_BUSINESS_MODELS = ["B2B", "B2C", "Both"]
    VALID_COST_BUCKETS = ["<1000", "1000-10000", "10000-50000", "50000+"]
    VALID_DIFFICULTY = ["Low", "Medium", "High"]
    VALID_SCALABILITY = ["Low", "Medium", "High"]
    
    def __init__(self, mistral_client: MistralClient):
        """Initialize enricher with Mistral client.
        
        Args:
            mistral_client: Configured MistralClient instance
        """
        self.mistral_client = mistral_client
        logger.info("Initialized IdeaEnricher")
    
    def enrich_single_idea(self, idea: str, retries: int = 3) -> Optional[Dict]:
        """Enrich a single startup idea with structured fields.
        
        Args:
            idea: Startup idea description
            retries: Number of retry attempts for API failures
            
        Returns:
            Dictionary with enriched fields, or None if enrichment fails
        """
        if not idea or not isinstance(idea, str) or len(idea.strip()) == 0:
            logger.warning("Empty or invalid idea, skipping")
            return None
        
        prompt = self.ENRICHMENT_PROMPT_TEMPLATE.format(idea=idea.strip())
        
        for attempt in range(retries):
            try:
                # Call Mistral API
                messages = [
                    {"role": "system", "content": "You are a startup analysis expert. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ]
                
                response = self.mistral_client.chat_complete(
                    messages=messages,
                    temperature=0.3,  # Lower temperature for more deterministic output
                    max_tokens=500
                )
                
                # Parse JSON response
                enriched_data = self._parse_response(response)
                
                if enriched_data:
                    # Validate fields
                    if self._validate_enriched_data(enriched_data):
                        return enriched_data
                    else:
                        logger.warning(f"Invalid enriched data (attempt {attempt + 1}/{retries})")
                        if attempt < retries - 1:
                            time.sleep(1)
                            continue
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(1)
                    continue
            except Exception as e:
                logger.error(f"Enrichment error (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(2)
                    continue
        
        logger.error(f"Failed to enrich idea after {retries} attempts")
        return None
    
    def _parse_response(self, response: str) -> Optional[Dict]:
        """Parse Mistral API response to extract JSON.
        
        Args:
            response: Raw API response text
            
        Returns:
            Parsed dictionary or None if parsing fails
        """
        # Try to extract JSON from response
        response = response.strip()
        
        # Remove markdown code blocks if present
        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]
        
        if response.endswith("```"):
            response = response[:-3]
        
        response = response.strip()
        
        # Try to find JSON object
        start_idx = response.find("{")
        end_idx = response.rfind("}") + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                logger.warning("Failed to parse extracted JSON")
                return None
        
        return None
    
    def _validate_enriched_data(self, data: Dict) -> bool:
        """Validate enriched data has all required fields with valid values.
        
        Args:
            data: Enriched data dictionary
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = [
            "domain", "business_model", "estimated_cost_bucket",
            "difficulty", "scalability", "required_skills",
            "target_customer", "short_summary"
        ]
        
        # Check all required fields present
        if not all(field in data for field in required_fields):
            logger.warning(f"Missing fields. Got: {list(data.keys())}")
            return False
        
        # Validate enum fields
        if data["domain"] not in self.VALID_DOMAINS:
            logger.warning(f"Invalid domain: {data['domain']}")
            return False
        
        if data["business_model"] not in self.VALID_BUSINESS_MODELS:
            logger.warning(f"Invalid business_model: {data['business_model']}")
            return False
        
        if data["estimated_cost_bucket"] not in self.VALID_COST_BUCKETS:
            logger.warning(f"Invalid cost bucket: {data['estimated_cost_bucket']}")
            return False
        
        if data["difficulty"] not in self.VALID_DIFFICULTY:
            logger.warning(f"Invalid difficulty: {data['difficulty']}")
            return False
        
        if data["scalability"] not in self.VALID_SCALABILITY:
            logger.warning(f"Invalid scalability: {data['scalability']}")
            return False
        
        # Validate string fields are non-empty
        if not data["required_skills"] or len(data["required_skills"].strip()) == 0:
            logger.warning("Empty required_skills")
            return False
        
        if not data["target_customer"] or len(data["target_customer"].strip()) == 0:
            logger.warning("Empty target_customer")
            return False
        
        if not data["short_summary"] or len(data["short_summary"].strip()) == 0:
            logger.warning("Empty short_summary")
            return False
        
        return True
    
    def enrich_batch(self, ideas: List[str], batch_delay: float = 1.0) -> List[Optional[Dict]]:
        """Enrich a batch of ideas.
        
        Args:
            ideas: List of startup idea descriptions
            batch_delay: Delay in seconds between ideas to avoid rate limits
            
        Returns:
            List of enriched data dictionaries (None for failed enrichments)
        """
        enriched_results = []
        
        for i, idea in enumerate(ideas):
            logger.info(f"Enriching idea {i + 1}/{len(ideas)}")
            
            result = self.enrich_single_idea(idea)
            enriched_results.append(result)
            
            # Add delay between API calls to avoid rate limits
            if i < len(ideas) - 1:
                time.sleep(batch_delay)
        
        return enriched_results


def enrich_ideas_dataset(
    input_path: str,
    output_path: str,
    batch_size: int = 10,
    batch_delay: float = 1.0
) -> None:
    """Enrich entire ideas dataset.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save enriched CSV file
        batch_size: Number of ideas to process before saving checkpoint
        batch_delay: Delay between API calls in seconds
    """
    logger.info(f"Starting enrichment process")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    
    # Initialize Mistral client
    config = Config()
    mistral_client = MistralClient(config.MISTRAL_API_KEY, config.MISTRAL_MODEL)
    enricher = IdeaEnricher(mistral_client)
    
    # Load ideas
    logger.info("Loading ideas dataset...")
    df = pd.read_csv(input_path)
    total_rows = len(df)
    logger.info(f"Loaded {total_rows} ideas")
    
    # Get the column name (assumes single column or 'idea' column)
    if 'idea' in df.columns:
        idea_column = 'idea'
    elif 'ideas' in df.columns:
        idea_column = 'ideas'
    else:
        idea_column = df.columns[0]
    
    logger.info(f"Using column: '{idea_column}'")
    
    # Initialize result columns
    enriched_columns = [
        "domain", "business_model", "estimated_cost_bucket",
        "difficulty", "scalability", "required_skills",
        "target_customer", "short_summary"
    ]
    
    for col in enriched_columns:
        df[col] = None
    
    # Process in batches with progress bar
    successful = 0
    failed = 0
    
    with tqdm(total=total_rows, desc="Enriching ideas") as pbar:
        for start_idx in range(0, total_rows, batch_size):
            end_idx = min(start_idx + batch_size, total_rows)
            batch_ideas = df[idea_column].iloc[start_idx:end_idx].tolist()
            
            logger.info(f"\nProcessing batch {start_idx // batch_size + 1} (rows {start_idx + 1}-{end_idx})")
            
            # Enrich batch
            enriched_batch = enricher.enrich_batch(batch_ideas, batch_delay)
            
            # Update dataframe
            for i, enriched_data in enumerate(enriched_batch):
                row_idx = start_idx + i
                
                if enriched_data:
                    for col in enriched_columns:
                        df.at[row_idx, col] = enriched_data.get(col, "")
                    successful += 1
                else:
                    failed += 1
                
                pbar.update(1)
            
            # Save checkpoint after each batch
            df.to_csv(output_path, index=False)
            logger.info(f"Checkpoint saved. Progress: {successful} successful, {failed} failed")
    
    # Final save
    df.to_csv(output_path, index=False)
    
    logger.info("\n" + "=" * 60)
    logger.info("ENRICHMENT COMPLETE")
    logger.info(f"Total ideas processed: {total_rows}")
    logger.info(f"Successfully enriched: {successful} ({successful/total_rows*100:.1f}%)")
    logger.info(f"Failed: {failed} ({failed/total_rows*100:.1f}%)")
    logger.info(f"Output saved to: {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    # Configure paths
    input_file = "data/documents/ideas_ai - ideas.csv"
    output_file = "data/documents/ideas_enriched.csv"
    
    # Run enrichment
    enrich_ideas_dataset(
        input_path=input_file,
        output_path=output_file,
        batch_size=10,  # Process 10 ideas, then save checkpoint
        batch_delay=1.0  # 1 second delay between API calls
    )
