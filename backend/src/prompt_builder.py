"""Prompt construction for Mistral AI LLM interactions.

Builds prompts for business recommendation generation and idea analysis
by combining user profiles, retrieved context, and instructions.
"""

import logging
from typing import List, Dict, Any, Optional

from .document_processor import Document


logger = logging.getLogger(__name__)


class PromptBuilder:
    """Construct prompts for Mistral AI.
    
    Formats user profiles and retrieved documents into structured prompts
    for generating business recommendations and analyzing business ideas.
    """
    
    def __init__(self, max_context_tokens: int = 3000):
        """Initialize prompt builder.
        
        Args:
            max_context_tokens: Maximum tokens to use for context documents
        """
        if max_context_tokens <= 0:
            raise ValueError(f"max_context_tokens must be positive, got {max_context_tokens}")
        
        self.max_context_tokens = max_context_tokens
        logger.info(f"Initialized PromptBuilder: max_context_tokens={max_context_tokens}")
    
    def build_recommendation_prompt(
        self,
        user_profile: Dict[str, Any],
        context_documents: List[Document],
        num_recommendations: int = 3
    ) -> List[Dict[str, str]]:
        """Build prompt for generating business recommendations.
        
        Combines user profile, retrieved context, and generation instructions
        into a structured prompt for the LLM.
        
        Args:
            user_profile: User questionnaire responses
            context_documents: Retrieved relevant documents
            num_recommendations: Number of ideas to generate
            
        Returns:
            List of message dicts for chat completion API
            
        Raises:
            ValueError: If user_profile is empty or num_recommendations is invalid
        """
        if not user_profile:
            raise ValueError("user_profile cannot be empty")
        
        if num_recommendations <= 0:
            raise ValueError(f"num_recommendations must be positive, got {num_recommendations}")
        
        # Format user profile
        profile_text = self._format_user_profile(user_profile)
        
        # Format context documents
        context_text = self._format_context(context_documents)
        
        # Build system message
        system_message = {
            "role": "system",
            "content": (
                "You are an expert business consultant specializing in helping individuals "
                "identify viable business opportunities based on their unique circumstances, "
                "skills, and resources. Your recommendations are practical, data-driven, "
                "and tailored to each person's situation."
            )
        }
        
        # Build user message with all components
        user_message_parts = [
            "# Task",
            f"Generate {num_recommendations} personalized business idea recommendations based on the user profile below.",
            "",
            "# User Profile",
            profile_text,
        ]
        
        # Add context if available
        if context_text:
            user_message_parts.extend([
                "",
                "# Relevant Business Knowledge",
                "Use the following information to inform your recommendations:",
                context_text,
            ])
        
        # Add instructions
        user_message_parts.extend([
            "",
            "# Instructions",
            f"Generate exactly {num_recommendations} distinct business ideas that:",
            "1. Match the user's professional status, time availability, and budget",
            "2. Leverage their existing skills and industry interests",
            "3. Align with their business model preferences and target market",
            "4. Consider their risk tolerance and revenue timeline expectations",
            "",
            "For each recommendation, provide:",
            "- name: A clear, descriptive business name/concept",
            "- description: Detailed explanation of the business (2-3 sentences)",
            "- fitReasons: List of 3-4 specific reasons why this fits the user",
            "- firstSteps: List of 3-5 concrete action items to get started",
            "- startupCost: Estimated cost range (e.g., '$1,000-$5,000')",
            "- timeToRevenue: Expected timeline to first revenue (e.g., '3-6 months')",
            "- scalability: Potential for growth (Low/Medium/High)",
            "- competition: Competition level (Low/Medium/High)",
            "- matchScore: Percentage match with user profile (0-100)",
            "",
            "Return ONLY valid JSON in this exact format:",
            "{",
            '  "recommendations": [',
            "    {",
            '      "name": "Business Name",',
            '      "description": "Detailed description...",',
            '      "fitReasons": ["reason 1", "reason 2", "reason 3"],',
            '      "firstSteps": ["step 1", "step 2", "step 3"],',
            '      "startupCost": "$X-$Y",',
            '      "timeToRevenue": "X months",',
            '      "scalability": "Medium",',
            '      "competition": "Low",',
            '      "matchScore": 85',
            "    }",
            "  ]",
            "}",
        ])
        
        user_message = {
            "role": "user",
            "content": "\n".join(user_message_parts)
        }
        
        messages = [system_message, user_message]
        
        logger.debug(
            f"Built recommendation prompt: profile_fields={len(user_profile)}, "
            f"context_docs={len(context_documents)}, num_recommendations={num_recommendations}"
        )
        
        return messages
    
    def build_analysis_prompt(
        self,
        business_idea: str,
        context_documents: List[Document],
        user_profile: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, str]]:
        """Build prompt for analyzing a business idea.
        
        Combines the business idea, relevant context, and optional user profile
        into a structured prompt for analysis.
        
        Args:
            business_idea: The business concept to analyze
            context_documents: Retrieved relevant market data and case studies
            user_profile: Optional user context for personalized analysis
            
        Returns:
            List of message dicts for chat completion API
            
        Raises:
            ValueError: If business_idea is empty
        """
        if not business_idea or not business_idea.strip():
            raise ValueError("business_idea cannot be empty")
        
        # Format context documents
        context_text = self._format_context(context_documents)
        
        # Build system message
        system_message = {
            "role": "system",
            "content": (
                "You are an expert business analyst specializing in evaluating business ideas "
                "for viability, market fit, and growth potential. Your analyses are thorough, "
                "objective, and provide actionable insights for improvement."
            )
        }
        
        # Build user message
        user_message_parts = [
            "# Task",
            "Analyze the following business idea and provide a comprehensive evaluation.",
            "",
            "# Business Idea",
            business_idea,
        ]
        
        # Add user profile if provided
        if user_profile:
            profile_text = self._format_user_profile(user_profile)
            user_message_parts.extend([
                "",
                "# User Context",
                profile_text,
            ])
        
        # Add context if available
        if context_text:
            user_message_parts.extend([
                "",
                "# Relevant Market Data",
                "Use the following information to inform your analysis:",
                context_text,
            ])
        
        # Add instructions
        user_message_parts.extend([
            "",
            "# Instructions",
            "Provide a detailed analysis covering:",
            "",
            "1. **Viability Score** (0-100): Overall assessment of the idea's potential",
            "2. **Market Fit**: Evaluate target market, demand, and positioning",
            "3. **Risks**: Identify 3-5 key risks and challenges",
            "4. **Suggestions**: Provide 3-5 specific improvement recommendations",
            "5. **Cost Structure**: Break down startup and operational costs",
            "6. **Scalability**: Assess growth potential and scaling challenges",
            "",
            "Return ONLY valid JSON in this exact format:",
            "{",
            '  "viabilityScore": 75,',
            '  "marketFit": {',
            '    "targetMarket": "Description of target customers",',
            '    "demand": "Assessment of market demand",',
            '    "positioning": "Competitive positioning strategy"',
            "  },",
            '  "risks": [',
            '    {"risk": "Risk description", "severity": "High/Medium/Low", "mitigation": "How to address"},',
            "  ],",
            '  "suggestions": [',
            '    {"suggestion": "Improvement idea", "impact": "Expected benefit", "priority": "High/Medium/Low"},',
            "  ],",
            '  "costStructure": {',
            '    "startup": "Initial investment needed",',
            '    "monthly": "Ongoing monthly costs",',
            '    "breakeven": "Time to break even"',
            "  },",
            '  "scalability": {',
            '    "potential": "High/Medium/Low",',
            '    "challenges": "Key scaling challenges",',
            '    "strategy": "Recommended scaling approach"',
            "  }",
            "}",
        ])
        
        user_message = {
            "role": "user",
            "content": "\n".join(user_message_parts)
        }
        
        messages = [system_message, user_message]
        
        logger.debug(
            f"Built analysis prompt: idea_length={len(business_idea)}, "
            f"context_docs={len(context_documents)}, has_profile={user_profile is not None}"
        )
        
        return messages
    
    def _format_user_profile(self, profile: Dict[str, Any]) -> str:
        """Format user profile for inclusion in prompts.
        
        Args:
            profile: User profile dictionary
            
        Returns:
            Formatted profile text
        """
        lines = []
        
        # Map of field names to display labels
        field_labels = {
            "professional_status": "Professional Status",
            "timeCommitment": "Time Availability",
            "budget": "Budget",
            "skills": "Skills",
            "industries": "Industry Interests",
            "business_model": "Preferred Business Model",
            "target_market": "Target Market",
            "location": "Location",
            "riskTolerance": "Risk Tolerance",
            "revenue_timeline": "Revenue Timeline",
            "primary_goal": "Primary Goal",
            "additionalContext": "Additional Context",
        }
        
        for field, label in field_labels.items():
            if field in profile and profile[field]:
                value = profile[field]
                
                # Format lists nicely
                if isinstance(value, list):
                    if value:  # Only include non-empty lists
                        value_str = ", ".join(str(v) for v in value)
                        lines.append(f"- {label}: {value_str}")
                else:
                    lines.append(f"- {label}: {value}")
        
        return "\n".join(lines) if lines else "No profile information provided"
    
    def _format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents as context for prompts.
        
        Enforces token limits by truncating context if needed.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context text (truncated if exceeds max_context_tokens)
        """
        if not documents:
            return ""
        
        context_parts = []
        total_tokens = 0
        
        for i, doc in enumerate(documents, 1):
            # Include document metadata for citation
            source = doc.metadata.source
            category = doc.metadata.category
            
            # Format document
            doc_text = [
                f"## Document {i}: {source}",
                f"Category: {category}",
                "",
                doc.content,
                "",
            ]
            
            doc_formatted = "\n".join(doc_text)
            
            # Calculate tokens for this document
            doc_tokens = self._count_tokens(doc_formatted)
            
            # Check if adding this document would exceed limit
            if total_tokens + doc_tokens > self.max_context_tokens:
                # Try to include partial document
                remaining_tokens = self.max_context_tokens - total_tokens
                
                # If this is the first document and it's too large, include partial
                if i == 1 and remaining_tokens < 100:
                    # First document is too large, truncate it
                    truncated_content = self._truncate_to_tokens(doc.content, self.max_context_tokens - 50)
                    doc_text = [
                        f"## Document {i}: {source}",
                        f"Category: {category}",
                        "",
                        truncated_content,
                        "[... truncated due to token limit ...]",
                        "",
                    ]
                    context_parts.append("\n".join(doc_text))
                elif remaining_tokens > 100:  # Only include if we have reasonable space
                    truncated_content = self._truncate_to_tokens(doc.content, remaining_tokens - 50)
                    doc_text = [
                        f"## Document {i}: {source}",
                        f"Category: {category}",
                        "",
                        truncated_content,
                        "[... truncated due to token limit ...]",
                        "",
                    ]
                    context_parts.append("\n".join(doc_text))
                
                # Stop adding more documents
                logger.debug(
                    f"Context truncated at document {i}/{len(documents)} "
                    f"to stay within {self.max_context_tokens} token limit"
                )
                break
            
            context_parts.append(doc_formatted)
            total_tokens += doc_tokens
        
        result = "\n".join(context_parts)
        
        logger.debug(
            f"Formatted context: {len(documents)} docs, "
            f"~{total_tokens} tokens (limit: {self.max_context_tokens})"
        )
        
        return result
    
    def _count_tokens(self, text: str) -> int:
        """Estimate token count for text.
        
        Uses simple word-based approximation. In production, use a proper
        tokenizer like tiktoken that matches the LLM's tokenization.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Estimated token count
        """
        # Simple approximation: ~1.3 tokens per word on average
        # This is a rough estimate; actual tokenization varies
        words = text.split()
        return int(len(words) * 1.3)
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to approximately max_tokens.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum token count
            
        Returns:
            Truncated text
        """
        if max_tokens <= 0:
            return ""
        
        # Estimate words needed
        max_words = int(max_tokens / 1.3)
        
        words = text.split()
        
        if len(words) <= max_words:
            return text
        
        # Truncate to max_words
        truncated_words = words[:max_words]
        return " ".join(truncated_words)
