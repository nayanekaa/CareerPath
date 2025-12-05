"""Database CRUD operations for psychometric scores and ratings."""

from datetime import datetime
from typing import List, Dict, Optional, Tuple
try:
    from sqlalchemy.orm import Session
    from sqlalchemy import func
    SQLALCHEMY_AVAILABLE = True
except Exception:
    # SQLAlchemy not installed in the environment; DB operations will fail at runtime
    Session = None  # type: ignore
    func = None  # type: ignore
    SQLALCHEMY_AVAILABLE = False

from schema import PsychometricScore, RoadmapRating, get_session_maker


SessionLocal = get_session_maker()


class PsychometricDB:
    """Database operations for psychometric scores."""
    
    @staticmethod
    def save_score(
        user_id: str,
        test_responses: Dict,
        archetype: str,
        learning_style: str,
        work_style: str,
        traits: List[str],
        description: str
    ) -> PsychometricScore:
        """Save psychometric test results."""
        db = SessionLocal()
        try:
            score = PsychometricScore(
                user_id=user_id,
                test_responses=test_responses,
                archetype=archetype,
                learning_style=learning_style,
                work_style=work_style,
                traits=traits,
                description=description
            )
            db.add(score)
            db.commit()
            db.refresh(score)
            return score
        finally:
            db.close()
    
    @staticmethod
    def get_score(user_id: str) -> Optional[PsychometricScore]:
        """Get most recent psychometric score for user."""
        db = SessionLocal()
        try:
            return db.query(PsychometricScore).filter(
                PsychometricScore.user_id == user_id
            ).order_by(PsychometricScore.created_at.desc()).first()
        finally:
            db.close()
    
    @staticmethod
    def get_all_scores(user_id: str) -> List[PsychometricScore]:
        """Get all psychometric scores for user."""
        db = SessionLocal()
        try:
            return db.query(PsychometricScore).filter(
                PsychometricScore.user_id == user_id
            ).order_by(PsychometricScore.created_at.desc()).all()
        finally:
            db.close()


class RoadmapRatingDB:
    """Database operations for roadmap ratings."""
    
    @staticmethod
    def save_rating(
        user_id: str,
        roadmap_id: str,
        grader_id: str,
        usefulness: Optional[int] = None,
        clarity: Optional[int] = None,
        factuality: Optional[int] = None,
        actionability: Optional[int] = None,
        overall_rating: Optional[int] = None,
        comments: Optional[str] = None
    ) -> RoadmapRating:
        """Save a grader's rating for a roadmap."""
        db = SessionLocal()
        try:
            rating = RoadmapRating(
                user_id=user_id,
                roadmap_id=roadmap_id,
                grader_id=grader_id,
                usefulness=usefulness,
                clarity=clarity,
                factuality=factuality,
                actionability=actionability,
                overall_rating=overall_rating,
                comments=comments
            )
            db.add(rating)
            db.commit()
            db.refresh(rating)
            return rating
        finally:
            db.close()
    
    @staticmethod
    def get_ratings(roadmap_id: str) -> List[RoadmapRating]:
        """Get all ratings for a roadmap."""
        db = SessionLocal()
        try:
            return db.query(RoadmapRating).filter(
                RoadmapRating.roadmap_id == roadmap_id
            ).order_by(RoadmapRating.created_at.desc()).all()
        finally:
            db.close()
    
    @staticmethod
    def get_user_ratings(user_id: str) -> List[RoadmapRating]:
        """Get all ratings by a user (as grader)."""
        db = SessionLocal()
        try:
            return db.query(RoadmapRating).filter(
                RoadmapRating.grader_id == user_id
            ).order_by(RoadmapRating.created_at.desc()).all()
        finally:
            db.close()
    
    @staticmethod
    def compute_summary(roadmap_id: str) -> Dict:
        """Compute average and std dev for all ratings dimensions."""
        db = SessionLocal()
        try:
            ratings = db.query(RoadmapRating).filter(
                RoadmapRating.roadmap_id == roadmap_id
            ).all()
            
            if not ratings:
                return {
                    'total_ratings': 0,
                    'usefulness': {'mean': None, 'stddev': None, 'count': 0},
                    'clarity': {'mean': None, 'stddev': None, 'count': 0},
                    'factuality': {'mean': None, 'stddev': None, 'count': 0},
                    'actionability': {'mean': None, 'stddev': None, 'count': 0},
                    'overall': {'mean': None, 'stddev': None, 'count': 0}
                }
            
            def compute_stats(values: List[int]) -> Tuple[Optional[float], Optional[float]]:
                """Compute mean and std dev. Returns (mean, stddev) or (None, None) if no values."""
                if not values:
                    return None, None
                mean = sum(values) / len(values)
                variance = sum((x - mean) ** 2 for x in values) / len(values)
                stddev = variance ** 0.5
                return mean, stddev
            
            usefulness_vals = [r.usefulness for r in ratings if r.usefulness is not None]
            clarity_vals = [r.clarity for r in ratings if r.clarity is not None]
            factuality_vals = [r.factuality for r in ratings if r.factuality is not None]
            actionability_vals = [r.actionability for r in ratings if r.actionability is not None]
            overall_vals = [r.overall_rating for r in ratings if r.overall_rating is not None]
            
            usefulness_mean, usefulness_stddev = compute_stats(usefulness_vals)
            clarity_mean, clarity_stddev = compute_stats(clarity_vals)
            factuality_mean, factuality_stddev = compute_stats(factuality_vals)
            actionability_mean, actionability_stddev = compute_stats(actionability_vals)
            overall_mean, overall_stddev = compute_stats(overall_vals)
            
            return {
                'total_ratings': len(ratings),
                'usefulness': {'mean': usefulness_mean, 'stddev': usefulness_stddev, 'count': len(usefulness_vals)},
                'clarity': {'mean': clarity_mean, 'stddev': clarity_stddev, 'count': len(clarity_vals)},
                'factuality': {'mean': factuality_mean, 'stddev': factuality_stddev, 'count': len(factuality_vals)},
                'actionability': {'mean': actionability_mean, 'stddev': actionability_stddev, 'count': len(actionability_vals)},
                'overall': {'mean': overall_mean, 'stddev': overall_stddev, 'count': len(overall_vals)}
            }
        finally:
            db.close()
