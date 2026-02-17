"""Process a sample of startup ideas for testing.

This processes the first N ideas to verify the enrichment works correctly.
"""

import logging
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from enrich_ideas import IdeaEnricher
from src.config import Config
from src.mistral_client import MistralClient


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Process a sample of ideas."""
    # Configuration
    input_file = Path("data/documents/ideas.csv")
    output_file = Path("data/ideas_enriched_SAMPLE.csv")
    sample_size = 100  # Process first 100 ideas
    
    logger.info(f"Processing SAMPLE of {sample_size} ideas")
    
    try:
        # Read CSV
        logger.info(f"Reading {input_file}...")
        df = pd.read_csv(input_file, nrows=sample_size)
        
        # Identify idea column
        idea_column = df.columns[0]  # First column
        logger.info(f"Using column '{idea_column}' containing {len(df)} ideas")
        
        # Show sample
        logger.info("\n=== Sample Ideas ===")
        print(df.head(3))
        
        # Initialize Mistral client
        logger.info("\nInitializing Mistral client...")
        config = Config()
        mistral_client = MistralClient(config.MISTRAL_API_KEY, model=config.MISTRAL_MODEL)
        
        # Create enricher
        enricher = IdeaEnricher(mistral_client, batch_size=10)
        
        # Enrich ideas
        logger.info(f"\nEnriching {len(df)} ideas...")
        enriched_df = enricher.enrich_dataframe(df, idea_column=idea_column)
        
        # Save enriched dataset
        enriched_df.to_csv(output_file, index=False)
        logger.info(f"\n✅ Enriched sample saved to: {output_file}")
        
        # Display results
        logger.info("\n=== Sample Enriched Ideas ===")
        display_cols = [idea_column, 'domain', 'business_model', 'difficulty', 'short_summary']
        available_cols = [col for col in display_cols if col in enriched_df.columns]
        print(enriched_df[available_cols].head(5).to_string())
        
        # Statistics
        success_count = (enriched_df['domain'] != '').sum()
        logger.info(f"\n=== Statistics ===")
        logger.info(f"Total processed: {len(enriched_df)}")
        logger.info(f"Successfully enriched: {success_count} ({success_count/len(enriched_df)*100:.1f}%)")
        logger.info(f"Failed: {len(enriched_df) - success_count}")
        
        if success_count > 0:
            logger.info("\nDomain distribution:")
            print(enriched_df[enriched_df['domain'] != '']['domain'].value_counts())
            
            logger.info("\nDifficulty distribution:")
            print(enriched_df[enriched_df['difficulty'] != '']['difficulty'].value_counts())
        
        # Estimate for full dataset
        logger.info(f"\n=== Full Dataset Estimates ===")
        logger.info(f"Full dataset size: 585,201 ideas")
        logger.info(f"Estimated time: ~49 hours (at 0.5s per idea)")
        logger.info(f"Estimated API calls: ~585,201 requests")
        logger.info("💡 Consider processing in chunks or filtering to reduce scope")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
