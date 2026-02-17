"""Recommendation engine for matching user profiles with startup ideas.

This module implements deterministic scoring and filtering logic to match
user profiles with enriched startup ideas. It is NOT an LLM-based chatbot,
but a structured data-driven recommendation system.
"""

import logging
from typing import List, Dict, Optional, Set
import pandas as pd
import numpy as np

from .schemas import UserProfile


logger = logging.getLogger(__name__)


class RecommendationEngine:
    """Scores and ranks startup ideas based on user profile matching."""
    
    # Domain mapping for compatibility
    DOMAIN_MAPPING = {
        "FinTech": ["FinTech"],
        "HealthTech": ["HealthTech"],
        "EdTech": ["EdTech"],
        "AI/ML": ["AI/ML"],
        "E-commerce": ["E-commerce"],
        "Sustainability": ["Sustainability"],
        "SaaS": ["SaaS"],
        "Marketplace": ["Marketplace"],
        "HRTech": ["HRTech"],
        "PropTech": ["PropTech"],
        "FoodTech": ["FoodTech"],
        "Other": ["Other"]
    }
    
    # Cost bucket to numeric range mapping
    COST_RANGES = {
        "<1000": (0, 1000),
        "1000-10000": (1000, 10000),
        "10000-50000": (10000, 50000),
        "50000+": (50000, float('inf'))
    }
    
    # Difficulty to numeric mapping
    DIFFICULTY_SCORES = {
        "Low": 1,
        "Medium": 2,
        "High": 3
    }
    
    # Scalability to numeric mapping
    SCALABILITY_SCORES = {
        "Low": 1,
        "Medium": 2,
        "High": 3
    }
    
    # Scoring weights (must sum to 1.0)
    WEIGHTS = {
        "domain_match": 0.20,      # Industry alignment
        "skill_overlap": 0.20,     # Skills matching
        "difficulty_fit": 0.15,    # Experience vs difficulty
        "scalability_fit": 0.15,   # Income goals vs scalability
        "cost_fit": 0.10,          # Budget vs estimated cost
        "business_model_match": 0.10,  # B2B/B2C preference
        "network_bonus": 0.05,     # Network strength for B2B
        "time_feasibility": 0.05   # Time commitment alignment
    }
    
    def __init__(self, ideas_csv_path: str):
        """Initialize recommendation engine with enriched ideas.
        
        Args:
            ideas_csv_path: Path to ideas_enriched.csv file
        """
        logger.info(f"Loading enriched ideas from: {ideas_csv_path}")
        self.ideas_df = pd.read_csv(ideas_csv_path)
        logger.info(f"Loaded {len(self.ideas_df)} enriched ideas")
        
        # Validate required columns
        required_cols = [
            'domain', 'business_model', 'estimated_cost_bucket',
            'difficulty', 'scalability', 'required_skills',
            'target_customer', 'short_summary'
        ]
        
        missing_cols = [col for col in required_cols if col not in self.ideas_df.columns]
        if missing_cols:
            logger.warning(f"Missing columns in ideas CSV: {missing_cols}")
    
    def get_all_ideas(self) -> pd.DataFrame:
        """Return all ideas for scoring.
        
        No hard filters are applied — everything is handled through
        soft scoring components so that results are guaranteed for
        any combination of user selections.
        
        Returns:
            Full DataFrame of all enriched ideas
        """
        return self.ideas_df.copy()
    
    def calculate_skill_overlap(
        self,
        user_skills: List[str],
        required_skills: str
    ) -> float:
        """Calculate skill overlap between user and required skills.
        
        Args:
            user_skills: List of user's skills
            required_skills: Comma-separated string of required skills
            
        Returns:
            Overlap score between 0 and 1
        """
        if not user_skills or pd.isna(required_skills):
            return 0.0
        
        # Normalize skills to lowercase for comparison
        user_skills_set = set(skill.lower().strip() for skill in user_skills)
        required_skills_list = [s.strip().lower() for s in str(required_skills).split(',')]
        required_skills_set = set(required_skills_list)
        
        if not required_skills_set:
            return 0.5  # Neutral if no skills specified
        
        # Calculate overlap
        overlap = len(user_skills_set.intersection(required_skills_set))
        max_possible = len(required_skills_set)
        
        score = overlap / max_possible if max_possible > 0 else 0.0
        return min(score, 1.0)
    
    def calculate_difficulty_fit(
        self,
        experience_level: str,
        difficulty: str
    ) -> float:
        """Calculate how well difficulty matches experience level.
        
        Args:
            experience_level: User's experience (Beginner, Intermediate, Expert)
            difficulty: Idea difficulty (Low, Medium, High)
            
        Returns:
            Fit score between 0 and 1
        """
        if pd.isna(difficulty):
            return 0.5
        
        experience_map = {
            "Beginner": 1,
            "Intermediate": 2,
            "Expert": 3
        }
        
        user_level = experience_map.get(experience_level, 2)
        idea_difficulty = self.DIFFICULTY_SCORES.get(difficulty, 2)
        
        # Perfect match: same level
        # Good match: within 1 level
        # Poor match: 2+ levels apart
        difference = abs(user_level - idea_difficulty)
        
        if difference == 0:
            return 1.0
        elif difference == 1:
            return 0.7
        else:
            return 0.3
    
    def calculate_scalability_fit(
        self,
        desired_income: Optional[float],
        scalability: str
    ) -> float:
        """Calculate fit between income goals and scalability.
        
        Args:
            desired_income: User's desired monthly income
            scalability: Idea's scalability (Low, Medium, High)
            
        Returns:
            Fit score between 0 and 1
        """
        if not desired_income or pd.isna(scalability):
            return 0.5
        
        scalability_score = self.SCALABILITY_SCORES.get(scalability, 2)
        
        # Map income goals to scalability needs
        if desired_income < 5000:
            # Low income: any scalability works
            return 0.8
        elif desired_income < 20000:
            # Medium income: prefer medium+ scalability
            if scalability_score >= 2:
                return 1.0
            else:
                return 0.5
        else:
            # High income: need high scalability
            if scalability_score == 3:
                return 1.0
            elif scalability_score == 2:
                return 0.6
            else:
                return 0.2
    
    def calculate_network_bonus(
        self,
        network_strength: str,
        business_model: str
    ) -> float:
        """Calculate bonus for B2B ideas if user has strong network.
        
        Args:
            network_strength: User's network strength (Weak, Moderate, Strong)
            business_model: Idea's business model
            
        Returns:
            Bonus score between 0 and 1
        """
        if pd.isna(business_model):
            return 0.5
        
        network_map = {
            "Weak": 0.3,
            "Moderate": 0.6,
            "Strong": 1.0
        }
        
        network_score = network_map.get(network_strength, 0.5)
        
        # Apply bonus for B2B if strong network
        if business_model in ["B2B", "Both"]:
            return network_score
        else:
            return 0.5  # Neutral for B2C
    
    def calculate_cost_fit(
        self,
        starting_capital: Optional[float],
        cost_bucket: str
    ) -> float:
        """Calculate how well the idea's cost fits the user's budget.
        
        Soft scoring: ideas within budget score highest, ideas slightly
        above budget get partial scores, ideas way above budget score low
        but are never excluded.
        
        Args:
            starting_capital: User's available budget
            cost_bucket: Idea's estimated cost bucket string
            
        Returns:
            Fit score between 0 and 1
        """
        if starting_capital is None or starting_capital <= 0:
            return 0.5  # No budget info, neutral
        if pd.isna(cost_bucket):
            return 0.6  # Unknown cost, slightly positive
        
        min_cost, max_cost = self.COST_RANGES.get(cost_bucket, (0, float('inf')))
        
        if starting_capital >= max_cost:
            return 1.0  # Budget comfortably covers it
        elif starting_capital >= min_cost:
            return 0.8  # Budget is within the range
        elif starting_capital >= min_cost * 0.5:
            return 0.5  # Budget is a stretch but feasible
        else:
            return 0.3  # Significantly over budget
    
    def calculate_time_feasibility(
        self,
        time_commitment: str,
        difficulty: str
    ) -> float:
        """Calculate if time commitment aligns with difficulty.
        
        Args:
            time_commitment: Hours per week user can commit
            difficulty: Idea difficulty
            
        Returns:
            Feasibility score between 0 and 1
        """
        if pd.isna(difficulty):
            return 0.5
        
        time_map = {
            "part_time": 1,
            "flexible": 2,
            "full_time": 3
        }
        
        time_level = time_map.get(time_commitment, 2)
        difficulty_level = self.DIFFICULTY_SCORES.get(difficulty, 2)
        
        # High difficulty needs full-time commitment
        if difficulty_level == 3 and time_level >= 3:
            return 1.0
        elif difficulty_level == 2 and time_level >= 2:
            return 1.0
        elif difficulty_level == 1:
            return 1.0
        else:
            # Not enough time for difficulty
            return 0.5
    
    def score_single_idea(
        self,
        idea_row: pd.Series,
        profile: UserProfile
    ) -> Dict:
        """Calculate match score for a single idea.
        
        Args:
            idea_row: Row from ideas DataFrame
            profile: User profile
            
        Returns:
            Dictionary with score breakdown
        """
        scores = {}
        
        # Domain match (binary: in selected domains or not)
        if profile.industry_interest:
            scores['domain_match'] = 1.0 if idea_row.get('domain') in profile.industry_interest else 0.0
        else:
            scores['domain_match'] = 0.5
        
        # Skill overlap
        scores['skill_overlap'] = self.calculate_skill_overlap(
            profile.skills,
            idea_row.get('required_skills', '')
        )
        
        # Difficulty fit
        scores['difficulty_fit'] = self.calculate_difficulty_fit(
            profile.experience_level,
            idea_row.get('difficulty', '')
        )
        
        # Scalability fit
        scores['scalability_fit'] = self.calculate_scalability_fit(
            profile.desired_income,
            idea_row.get('scalability', '')
        )
        
        # Cost fit (soft scoring — never eliminates ideas)
        scores['cost_fit'] = self.calculate_cost_fit(
            profile.starting_capital,
            idea_row.get('estimated_cost_bucket', '')
        )
        
        # Business model match
        if profile.business_model_preference and profile.business_model_preference != "Both":
            bm = idea_row.get('business_model', '')
            if bm == profile.business_model_preference or bm == "Both":
                scores['business_model_match'] = 1.0
            else:
                scores['business_model_match'] = 0.3
        else:
            scores['business_model_match'] = 0.5
        
        # Network bonus
        scores['network_bonus'] = self.calculate_network_bonus(
            profile.network_strength,
            idea_row.get('business_model', '')
        )
        
        # Time feasibility
        scores['time_feasibility'] = self.calculate_time_feasibility(
            profile.time_commitment,
            idea_row.get('difficulty', '')
        )
        
        # Calculate weighted total
        total_score = sum(
            scores[component] * self.WEIGHTS[component]
            for component in scores.keys()
        )
        
        return {
            'total_score': total_score,
            'component_scores': scores
        }
    
    def get_recommendations(
        self,
        profile: UserProfile,
        top_n: int = 10
    ) -> List[Dict]:
        """Get top N recommendations for a user profile.
        
        Args:
            profile: User profile with preferences
            top_n: Number of top recommendations to return
            
        Returns:
            List of recommendation dictionaries with scores and details
        """
        logger.info(f"Generating recommendations for profile")
        
        # Score ALL ideas — no hard filters. Preferences are handled
        # entirely through weighted scoring so results are guaranteed.
        all_ideas = self.get_all_ideas()
        
        # Score every idea
        recommendations = []
        
        for idx, row in all_ideas.iterrows():
            # Calculate match score
            score_data = self.score_single_idea(row, profile)
            
            # Build recommendation object
            rec = {
                'idea_index': int(idx),
                'idea_text': row.get(all_ideas.columns[0], ''),  # Original idea text
                'domain': row.get('domain', ''),
                'business_model': row.get('business_model', ''),
                'estimated_cost_bucket': row.get('estimated_cost_bucket', ''),
                'difficulty': row.get('difficulty', ''),
                'scalability': row.get('scalability', ''),
                'required_skills': row.get('required_skills', ''),
                'target_customer': row.get('target_customer', ''),
                'short_summary': row.get('short_summary', ''),
                'match_score': float(score_data['total_score']),
                'match_percentage': int(score_data['total_score'] * 100),
                'score_breakdown': score_data['component_scores']
            }
            
            recommendations.append(rec)
        
        # Sort by match score descending
        recommendations.sort(key=lambda x: x['match_score'], reverse=True)
        
        # Return top N
        top_recommendations = recommendations[:top_n]
        
        logger.info(f"Returning {len(top_recommendations)} recommendations")
        return top_recommendations
    
    def generate_match_explanation(
        self,
        recommendation: Dict,
        profile: UserProfile
    ) -> str:
        """Generate human-readable explanation for why idea matches.
        
        This is deterministic and rule-based, NOT LLM-generated.
        
        Args:
            recommendation: Recommendation dictionary with scores
            profile: User profile
            
        Returns:
            Natural language explanation
        """
        explanation_parts = []
        
        score_breakdown = recommendation.get('score_breakdown', {})
        
        # Domain match
        if score_breakdown.get('domain_match', 0) > 0.8:
            explanation_parts.append(
                f"✓ Matches your interest in {recommendation['domain']}"
            )
        
        # Skill overlap
        skill_score = score_breakdown.get('skill_overlap', 0)
        if skill_score > 0.7:
            explanation_parts.append("✓ Strong match with your skills")
        elif skill_score > 0.4:
            explanation_parts.append("◐ Partial skill match (opportunity to learn)")
        
        # Difficulty fit
        diff_score = score_breakdown.get('difficulty_fit', 0)
        if diff_score >= 0.7:
            explanation_parts.append(
                f"✓ {recommendation['difficulty']} difficulty suits your {profile.experience_level} level"
            )
        
        # Scalability fit
        scal_score = score_breakdown.get('scalability_fit', 0)
        if scal_score > 0.8:
            explanation_parts.append(
                f"✓ {recommendation['scalability']} scalability aligns with income goals"
            )
        
        # Business model
        if score_breakdown.get('business_model_match', 0) >= 0.8:
            explanation_parts.append(
                f"✓ {recommendation['business_model']} model matches your preference"
            )
        
        # Cost
        explanation_parts.append(
            f"💰 Estimated cost: {recommendation['estimated_cost_bucket']}"
        )
        
        return " | ".join(explanation_parts)
