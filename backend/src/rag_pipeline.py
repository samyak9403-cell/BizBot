"""RAG (Retrieval Augmented Generation) pipeline orchestration.

Coordinates retrieval from FAISS, prompt building, and LLM generation
to produce business recommendations and idea analyses.
"""

import logging
import json
from typing import List, Dict, Any, Optional

from .mistral_client import MistralClient, MistralAPIError
from .faiss_retriever import FAISSRetriever, FAISSRetrieverError
from .prompt_builder import PromptBuilder
from .document_processor import Document


logger = logging.getLogger(__name__)


class RAGPipelineError(Exception):
    """Raised when RAG pipeline operations fail."""
    pass


class RAGPipeline:
    """Orchestrate RAG pipeline for business recommendations and analysis.
    
    Combines semantic search (FAISS), prompt construction, and LLM generation
    to provide context-aware business recommendations and idea analyses.
    """
    
    def __init__(
        self,
        mistral_client: MistralClient,
        faiss_retriever: FAISSRetriever,
        prompt_builder: PromptBuilder,
        top_k_documents: int = 5
    ):
        """Initialize RAG pipeline with dependencies.
        
        Args:
            mistral_client: Mistral AI client for LLM operations
            faiss_retriever: FAISS retriever for semantic search
            prompt_builder: Prompt builder for formatting prompts
            top_k_documents: Number of documents to retrieve for context
            
        Raises:
            ValueError: If top_k_documents is invalid
        """
        if top_k_documents <= 0:
            raise ValueError(f"top_k_documents must be positive, got {top_k_documents}")
        
        self.mistral_client = mistral_client
        self.faiss_retriever = faiss_retriever
        self.prompt_builder = prompt_builder
        self.top_k_documents = top_k_documents
        
        logger.info(
            f"Initialized RAGPipeline: top_k_documents={top_k_documents}"
        )
    
    def _retrieve_context(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """Retrieve relevant documents from FAISS.
        
        Args:
            query: Search query text
            top_k: Number of documents to retrieve (uses default if None)
            
        Returns:
            List of relevant documents ordered by relevance
            
        Raises:
            RAGPipelineError: If retrieval fails
        """
        k = top_k if top_k is not None else self.top_k_documents
        
        try:
            logger.debug(f"Retrieving top {k} documents for query")
            
            # Check if index is empty
            if self.faiss_retriever.is_empty:
                logger.warning("FAISS index is empty, no context will be retrieved")
                return []
            
            # Search FAISS index
            results = self.faiss_retriever.search(query=query, top_k=k)
            
            # Extract documents (discard similarity scores for now)
            documents = [doc for doc, score in results]
            
            logger.info(
                f"Retrieved {len(documents)} documents, "
                f"top score: {results[0][1]:.4f}" if results else "no results"
            )
            
            return documents
            
        except FAISSRetrieverError as e:
            logger.error(f"FAISS retrieval failed: {str(e)}")
            raise RAGPipelineError(f"Context retrieval failed: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error during retrieval: {str(e)}", exc_info=True)
            raise RAGPipelineError(f"Context retrieval failed: {str(e)}") from e
    
    def _format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents for LLM prompt.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        # Use prompt builder's context formatting
        return self.prompt_builder._format_context(documents)

    def generate_recommendations(
        self,
        user_profile: Dict[str, Any],
        num_recommendations: int = 3
    ) -> List[Dict[str, Any]]:
        """Generate business recommendations using RAG.
        
        Retrieves relevant business knowledge from FAISS, builds a prompt
        with user profile and context, and generates personalized recommendations
        using Mistral AI.
        
        Args:
            user_profile: User questionnaire responses
            num_recommendations: Number of ideas to generate
            
        Returns:
            List of business idea recommendations
            
        Raises:
            ValueError: If user_profile is empty or num_recommendations is invalid
            RAGPipelineError: If generation fails
        """
        if not user_profile:
            raise ValueError("user_profile cannot be empty")
        
        if num_recommendations <= 0:
            raise ValueError(f"num_recommendations must be positive, got {num_recommendations}")
        
        logger.info(
            f"Generating {num_recommendations} recommendations for user profile"
        )
        
        try:
            # Step 1: Build query from user profile
            query = self._build_query_from_profile(user_profile)
            logger.debug(f"Built query from profile: {query[:100]}...")
            
            # Step 2: Retrieve relevant context
            context_documents = self._retrieve_context(query)
            
            if not context_documents:
                logger.warning(
                    "No context documents retrieved, using LLM knowledge only"
                )
            
            # Step 3: Build prompt
            messages = self.prompt_builder.build_recommendation_prompt(
                user_profile=user_profile,
                context_documents=context_documents,
                num_recommendations=num_recommendations
            )
            
            # Step 4: Call Mistral API with retry logic
            logger.debug("Calling Mistral API for recommendation generation")
            response_text = self.mistral_client.complete_with_retry(
                messages=messages,
                temperature=0.7,
                max_tokens=4000
            )
            
            # Step 5: Parse JSON response
            recommendations = self._parse_recommendations_response(response_text)
            
            # Validate we got the requested number
            if len(recommendations) < num_recommendations:
                logger.warning(
                    f"Generated {len(recommendations)} recommendations, "
                    f"expected {num_recommendations}"
                )
            
            logger.info(
                f"Successfully generated {len(recommendations)} recommendations"
            )
            
            return recommendations
            
        except (ValueError, MistralAPIError, FAISSRetrieverError) as e:
            # Re-raise known errors
            logger.error(f"Recommendation generation failed: {str(e)}")
            raise RAGPipelineError(f"Failed to generate recommendations: {str(e)}") from e
        except Exception as e:
            logger.error(
                f"Unexpected error during recommendation generation: {str(e)}",
                exc_info=True
            )
            raise RAGPipelineError(
                f"Failed to generate recommendations: {str(e)}"
            ) from e
    
    def _build_query_from_profile(self, user_profile: Dict[str, Any]) -> str:
        """Build search query from user profile.
        
        Converts user profile into a semantic query for retrieving
        relevant business knowledge.
        
        Args:
            user_profile: User questionnaire responses
            
        Returns:
            Search query string
        """
        # Extract key fields for query
        query_parts = []
        
        # Professional status and time commitment
        if 'professional_status' in user_profile:
            query_parts.append(f"professional status: {user_profile['professional_status']}")
        
        if 'timeCommitment' in user_profile:
            query_parts.append(f"time availability: {user_profile['timeCommitment']} hours/week")
        
        # Budget
        if 'budget' in user_profile:
            query_parts.append(f"budget: {user_profile['budget']}")
        
        # Skills
        if 'skills' in user_profile and user_profile['skills']:
            skills_str = ', '.join(user_profile['skills'])
            query_parts.append(f"skills: {skills_str}")
        
        # Industries
        if 'industries' in user_profile and user_profile['industries']:
            industries_str = ', '.join(user_profile['industries'])
            query_parts.append(f"industries: {industries_str}")
        
        # Business model preference
        if 'business_model' in user_profile:
            query_parts.append(f"business model: {user_profile['business_model']}")
        
        # Primary goal
        if 'primary_goal' in user_profile:
            query_parts.append(f"goal: {user_profile['primary_goal']}")
        
        # Combine into query
        query = "Business ideas for someone with: " + "; ".join(query_parts)
        
        return query
    
    def _parse_recommendations_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse JSON response from LLM into recommendations list.
        
        Args:
            response_text: Raw response text from LLM
            
        Returns:
            List of recommendation dictionaries
            
        Raises:
            RAGPipelineError: If response cannot be parsed or is invalid
        """
        try:
            # Try to extract JSON from response
            # LLM might include extra text before/after JSON
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON object found in response")
            
            json_text = response_text[json_start:json_end]
            
            # Parse JSON
            data = json.loads(json_text)
            
            # Extract recommendations array
            if 'recommendations' not in data:
                raise ValueError("Response missing 'recommendations' field")
            
            recommendations = data['recommendations']
            
            if not isinstance(recommendations, list):
                raise ValueError("'recommendations' field must be a list")
            
            # Validate each recommendation has required fields
            required_fields = [
                'name', 'description', 'fitReasons', 'firstSteps',
                'startupCost', 'timeToRevenue', 'scalability', 
                'competition', 'matchScore'
            ]
            
            for i, rec in enumerate(recommendations):
                for field in required_fields:
                    if field not in rec:
                        logger.warning(
                            f"Recommendation {i} missing field '{field}', "
                            f"setting to default"
                        )
                        # Set defaults for missing fields
                        if field in ['fitReasons', 'firstSteps']:
                            rec[field] = []
                        elif field == 'matchScore':
                            rec[field] = 0
                        else:
                            rec[field] = "Not specified"
            
            logger.debug(f"Successfully parsed {len(recommendations)} recommendations")
            
            return recommendations
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            logger.debug(f"Response text: {response_text[:500]}...")
            raise RAGPipelineError(
                f"Invalid JSON response from LLM: {str(e)}"
            ) from e
        except (ValueError, KeyError) as e:
            logger.error(f"Invalid response format: {str(e)}")
            logger.debug(f"Response text: {response_text[:500]}...")
            raise RAGPipelineError(
                f"Invalid response format: {str(e)}"
            ) from e

    def analyze_business_idea(
        self,
        business_idea: str,
        user_profile: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze a user-provided business idea.
        
        Retrieves relevant market data and case studies from FAISS,
        builds an analysis prompt, and generates a comprehensive
        evaluation using Mistral AI.
        
        Args:
            business_idea: The business concept to analyze
            user_profile: Optional user context for personalized analysis
            
        Returns:
            Analysis results with scores and suggestions
            
        Raises:
            ValueError: If business_idea is empty
            RAGPipelineError: If analysis fails
        """
        if not business_idea or not business_idea.strip():
            raise ValueError("business_idea cannot be empty")
        
        logger.info(f"Analyzing business idea: {business_idea[:100]}...")
        
        try:
            # Step 1: Retrieve relevant market data
            # Use the business idea itself as the query
            context_documents = self._retrieve_context(business_idea)
            
            if not context_documents:
                logger.warning(
                    "No context documents retrieved, using LLM knowledge only"
                )
            
            # Step 2: Build analysis prompt
            messages = self.prompt_builder.build_analysis_prompt(
                business_idea=business_idea,
                context_documents=context_documents,
                user_profile=user_profile
            )
            
            # Step 3: Call Mistral API with retry logic
            logger.debug("Calling Mistral API for business idea analysis")
            response_text = self.mistral_client.complete_with_retry(
                messages=messages,
                temperature=0.7,
                max_tokens=4000
            )
            
            # Step 4: Parse and validate response
            analysis = self._parse_analysis_response(response_text)
            
            logger.info(
                f"Successfully analyzed business idea, "
                f"viability score: {analysis.get('viabilityScore', 'N/A')}"
            )
            
            return analysis
            
        except (ValueError, MistralAPIError, FAISSRetrieverError) as e:
            # Re-raise known errors
            logger.error(f"Business idea analysis failed: {str(e)}")
            raise RAGPipelineError(f"Failed to analyze business idea: {str(e)}") from e
        except Exception as e:
            logger.error(
                f"Unexpected error during business idea analysis: {str(e)}",
                exc_info=True
            )
            raise RAGPipelineError(
                f"Failed to analyze business idea: {str(e)}"
            ) from e
    
    def _parse_analysis_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON response from LLM into analysis dictionary.
        
        Args:
            response_text: Raw response text from LLM
            
        Returns:
            Analysis dictionary with all required fields
            
        Raises:
            RAGPipelineError: If response cannot be parsed or is invalid
        """
        try:
            # Try to extract JSON from response
            # LLM might include extra text before/after JSON
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON object found in response")
            
            json_text = response_text[json_start:json_end]
            
            # Parse JSON
            analysis = json.loads(json_text)
            
            # Validate required fields
            required_fields = [
                'viabilityScore', 'marketFit', 'risks', 
                'suggestions', 'costStructure', 'scalability'
            ]
            
            for field in required_fields:
                if field not in analysis:
                    logger.warning(
                        f"Analysis missing field '{field}', setting to default"
                    )
                    # Set defaults for missing fields
                    if field == 'viabilityScore':
                        analysis[field] = 0
                    elif field in ['risks', 'suggestions']:
                        analysis[field] = []
                    else:
                        analysis[field] = {}
            
            # Validate viability score is in range
            if 'viabilityScore' in analysis:
                score = analysis['viabilityScore']
                if not isinstance(score, (int, float)) or score < 0 or score > 100:
                    logger.warning(
                        f"Invalid viabilityScore: {score}, clamping to [0, 100]"
                    )
                    analysis['viabilityScore'] = max(0, min(100, int(score)))
            
            logger.debug("Successfully parsed analysis response")
            
            return analysis
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            logger.debug(f"Response text: {response_text[:500]}...")
            raise RAGPipelineError(
                f"Invalid JSON response from LLM: {str(e)}"
            ) from e
        except (ValueError, KeyError) as e:
            logger.error(f"Invalid response format: {str(e)}")
            logger.debug(f"Response text: {response_text[:500]}...")
            raise RAGPipelineError(
                f"Invalid response format: {str(e)}"
            ) from e
