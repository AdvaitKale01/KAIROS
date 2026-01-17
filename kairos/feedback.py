"""
Generator Feedback Loop - KAIROS Memory Component
CLaRa Principle: Generator teaches retriever what's important.

Tracks which retrieved memories are actually used by the LLM,
and optimizes retrieval to prioritize useful memories.
"""
import time
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path
from .utils import logger


class GeneratorFeedback:
    """
    Feedback loop between generator (LLM) and retriever.
    
    Core CLaRa innovation: The generator provides signals about which
    retrieved memories were actually useful, allowing the retriever
    to improve over time.
    """
    
    def __init__(self, feedback_path: str = "./data/kairos_feedback"):
        """
        Initialize feedback tracker.
        
        Args:
            feedback_path: Directory for persisting feedback data
        """
        self.feedback_path = Path(feedback_path)
        self.feedback_path.mkdir(parents=True, exist_ok=True)
        
        # Track retrieval sessions
        self.sessions: Dict[str, Dict] = {}  # session_id -> session_data
        
        # Aggregate statistics
        self.token_usage_scores: Dict[str, float] = {}  # token_id -> avg_usage_score
        self.query_patterns: Dict[str, List[str]] = {}  # query_type -> [successful_token_ids]
        
        # Load existing feedback
        self._load_feedback()
    
    def start_retrieval_session(self, query: str, retrieved_tokens: List[Tuple[str, float, Dict]]) -> str:
        """
        Start tracking a retrieval session.
        
        Args:
            query: User query
            retrieved_tokens: List of (token_id, similarity, metadata) tuples
            
        Returns:
            session_id for tracking
        """
        session_id = f"session_{int(time.time() * 1000)}"
        
        self.sessions[session_id] = {
            'query': query,
            'retrieved_tokens': [
                {
                    'token_id': tid,
                    'similarity': sim,
                    'metadata': meta
                }
                for tid, sim, meta in retrieved_tokens
            ],
            'start_time': time.time(),
            'usage_scores': {},  # token_id -> usage_score
            'completed': False
        }
        
        return session_id
    
    def record_token_usage(self, session_id: str, token_id: str, usage_score: float):
        """
        Record how much a retrieved token was actually used by the generator.
        
        Args:
            session_id: Session identifier
            token_id: Token that was retrieved
            usage_score: 0-1 score indicating how much it was used
                        0 = not mentioned at all
                        0.5 = partially referenced
                        1.0 = heavily used in generation
        """
        if session_id not in self.sessions:
            return
        
        self.sessions[session_id]['usage_scores'][token_id] = usage_score
    
    def complete_session(self, session_id: str, generation_quality: float = 1.0):
        """
        Mark a session as complete and update aggregate statistics.
        
        Args:
            session_id: Session identifier
            generation_quality: Overall quality of generated response (0-1)
        """
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        session['completed'] = True
        session['generation_quality'] = generation_quality
        session['completion_time'] = time.time()
        
        # Update aggregate token usage scores
        for token_id, usage_score in session['usage_scores'].items():
            if token_id not in self.token_usage_scores:
                self.token_usage_scores[token_id] = []
            
            # Weighted by generation quality
            weighted_score = usage_score * generation_quality
            self.token_usage_scores[token_id].append(weighted_score)
        
        # Persist feedback
        self._save_session(session_id)
    
    def get_token_relevance_boost(self, token_id: str) -> float:
        """
        Get relevance boost for a token based on historical usage.
        
        This can be used to re-rank retrieval results.
        
        Returns:
            Boost factor (0-1), where higher means token was historically useful
        """
        if token_id not in self.token_usage_scores:
            return 0.5  # Neutral for unknown tokens
        
        scores = self.token_usage_scores[token_id]
        avg_score = sum(scores) / len(scores)
        
        # Boost recent feedback more
        if len(scores) > 10:
            recent_avg = sum(scores[-10:]) / 10
            return (avg_score * 0.3) + (recent_avg * 0.7)
        
        return avg_score
    
    def re_rank_results(self, results: List[Tuple[str, float, Dict]]) -> List[Tuple[str, float, Dict]]:
        """
        Re-rank retrieval results using feedback data.
        
        CLaRa principle: Use generator feedback to improve retrieval!
        
        Args:
            results: List of (token_id, similarity, metadata)
            
        Returns:
            Re-ranked results with boosted scores
        """
        re_ranked = []
        
        for token_id, similarity, metadata in results:
            # Get feedback-based boost
            boost = self.get_token_relevance_boost(token_id)
            
            # Combine similarity with historical usage
            # 70% similarity, 30% historical performance
            combined_score = (similarity * 0.7) + (boost * 0.3)
            
            re_ranked.append((token_id, combined_score, metadata))
        
        # Sort by combined score
        re_ranked.sort(key=lambda x: x[1], reverse=True)
        
        return re_ranked
    
    def get_stats(self) -> Dict:
        """Get feedback statistics."""
        total_sessions = len(self.sessions)
        completed_sessions = sum(1 for s in self.sessions.values() if s.get('completed', False))
        
        # Calculate average usage per token
        avg_usage = {}
        for token_id, scores in self.token_usage_scores.items():
            avg_usage[token_id] = sum(scores) / len(scores) if scores else 0
        
        return {
            'total_sessions': total_sessions,
            'completed_sessions': completed_sessions,
            'tokens_tracked': len(self.token_usage_scores),
            'avg_token_usage': sum(avg_usage.values()) / len(avg_usage) if avg_usage else 0
        }
    
    def _save_session(self, session_id: str):
        """Persist a completed session."""
        session = self.sessions[session_id]
        session_file = self.feedback_path / f"{session_id}.json"
        
        with open(session_file, 'w') as f:
            json.dump(session, f, indent=2)
    
    def _load_feedback(self):
        """Load existing feedback data."""
        if not self.feedback_path.exists():
            return
        
        for session_file in self.feedback_path.glob("session_*.json"):
            try:
                with open(session_file, 'r') as f:
                    session = json.load(f)
                    session_id = session_file.stem
                    self.sessions[session_id] = session
                    
                    # Rebuild aggregate stats
                    for token_id, usage_score in session.get('usage_scores', {}).items():
                        if token_id not in self.token_usage_scores:
                            self.token_usage_scores[token_id] = []
                        self.token_usage_scores[token_id].append(usage_score)
            except Exception as e:
                logger.warning(f"Failed to load feedback session {session_file}: {e}")

    def prune_useless_memories(self, store, threshold: float = 0.1, min_sessions: int = 5) -> int:
        """
        Active Forgetting: Prune memories that are consistently ignored.
        Mimics gradient descent pushing weights to zero.
        
        Args:
            store: The LatentTokenStore instance (to call delete)
            threshold: Usage score below which to prune (0-1)
            min_sessions: Minimum retrieval sessions before judging
            
        Returns:
            Count of pruned tokens
        """
        pruned_count = 0
        tokens_to_prune = []
        
        for token_id, scores in self.token_usage_scores.items():
            if len(scores) >= min_sessions:
                avg_score = sum(scores) / len(scores)
                if avg_score < threshold:
                    tokens_to_prune.append(token_id)
        
        for token_id in tokens_to_prune:
            # Delete from store
            if store.delete(token_id):
                pruned_count += 1
                
            # Remove from local stats
            if token_id in self.token_usage_scores:
                del self.token_usage_scores[token_id]
                
        if pruned_count > 0:
            logger.info(f"Active Forgetting: Pruned {pruned_count} useless memories (Avg Usage < {threshold})")
            
        return pruned_count
