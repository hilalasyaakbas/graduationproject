import math

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

try:
    from surprise import Dataset, Reader, SVD
except ImportError:  # pragma: no cover
    Dataset = None
    Reader = None
    SVD = None

from models import Movie, Rating


class HybridRecommender:
    def __init__(self, session):
        self.session = session

    def _ratings_df(self) -> pd.DataFrame:
        ratings = self.session.query(Rating).all()
        return pd.DataFrame(
            [{"user_id": r.user_id, "movie_id": r.movie_id, "score": r.score} for r in ratings]
        )

    def _movies_df(self) -> pd.DataFrame:
        movies = self.session.query(Movie).order_by(Movie.id.asc()).all()
        return pd.DataFrame(
            [
                {
                    "movie_id": movie.id,
                    "title": movie.title,
                    "year": movie.year,
                    "quality": movie.quality,
                    "duration": movie.duration,
                    "poster_url": movie.poster_url,
                    "description": movie.description or "",
                    "genres": movie.genres or "",
                }
                for movie in movies
            ]
        )

    def _build_content_model(self, movies_df: pd.DataFrame):
        features = (
            movies_df["title"].fillna("")
            + " "
            + movies_df["genres"].fillna("")
            + " "
            + movies_df["description"].fillna("")
        )
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(features)
        similarity_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
        movie_index = {movie_id: idx for idx, movie_id in enumerate(movies_df["movie_id"].tolist())}
        return similarity_matrix, movie_index

    def _train_cf_model(self, ratings_df: pd.DataFrame):
        if ratings_df.empty or SVD is None or Dataset is None or Reader is None:
            return None

        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(ratings_df[["user_id", "movie_id", "score"]], reader)
        trainset = data.build_full_trainset()
        model = SVD(random_state=42)
        model.fit(trainset)
        return model

    def _predict_cf(self, model, user_id: int, movie_id: int, ratings_df: pd.DataFrame) -> float:
        if model is None:
            movie_mean = ratings_df.loc[ratings_df["movie_id"] == movie_id, "score"].mean()
            user_mean = ratings_df.loc[ratings_df["user_id"] == user_id, "score"].mean()
            global_mean = ratings_df["score"].mean() if not ratings_df.empty else 3.0
            values = [value for value in [movie_mean, user_mean, global_mean] if not np.isnan(value)]
            return float(sum(values) / len(values)) if values else 3.0
        return float(model.predict(user_id, movie_id).est)

    def _predict_cbf(self, user_id: int, movie_id: int, ratings_df: pd.DataFrame, similarity_matrix, movie_index) -> float:
        history = ratings_df.loc[ratings_df["user_id"] == user_id]
        if history.empty or movie_id not in movie_index:
            return 3.0

        target_idx = movie_index[movie_id]
        weighted_scores = []
        similarity_weights = []

        for row in history.itertuples(index=False):
            if row.movie_id == movie_id or row.movie_id not in movie_index:
                continue
            similarity = similarity_matrix[target_idx][movie_index[row.movie_id]]
            if similarity <= 0:
                continue
            weighted_scores.append(similarity * row.score)
            similarity_weights.append(similarity)

        if not similarity_weights:
            return float(history["score"].mean())
        return float(np.sum(weighted_scores) / np.sum(similarity_weights))

    def optimize_alpha(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        if ratings_df.empty or len(ratings_df) < 8:
            return {"alpha": 0.5, "rmse": 0.0, "cf_rmse": 0.0, "cbf_rmse": 0.0}

        train_rows = []
        test_rows = []
        for _, group in ratings_df.groupby("user_id"):
            group = group.sort_values("movie_id")
            if len(group) < 3:
                train_rows.append(group)
                continue
            split_idx = max(1, int(len(group) * 0.8))
            train_rows.append(group.iloc[:split_idx])
            test_rows.append(group.iloc[split_idx:])

        train_df = pd.concat(train_rows).reset_index(drop=True)
        test_df = pd.concat(test_rows).reset_index(drop=True) if test_rows else train_df.copy()

        similarity_matrix, movie_index = self._build_content_model(movies_df)
        cf_model = self._train_cf_model(train_df)

        actuals = []
        cf_preds = []
        cbf_preds = []
        for row in test_df.itertuples(index=False):
            actuals.append(row.score)
            cf_preds.append(self._predict_cf(cf_model, row.user_id, row.movie_id, train_df))
            cbf_preds.append(self._predict_cbf(row.user_id, row.movie_id, train_df, similarity_matrix, movie_index))

        cf_rmse = self._rmse(actuals, cf_preds)
        cbf_rmse = self._rmse(actuals, cbf_preds)

        best_alpha = 0.5
        best_rmse = math.inf
        for alpha in np.linspace(0.0, 1.0, 11):
            hybrid = [
                alpha * cf_pred + (1 - alpha) * cbf_pred
                for cf_pred, cbf_pred in zip(cf_preds, cbf_preds)
            ]
            rmse = self._rmse(actuals, hybrid)
            if rmse < best_rmse:
                best_rmse = rmse
                best_alpha = float(alpha)

        return {
            "alpha": best_alpha,
            "rmse": best_rmse,
            "cf_rmse": cf_rmse,
            "cbf_rmse": cbf_rmse,
        }

    def recommend_for_user(self, user_id: int, top_n: int = 8):
        ratings_df = self._ratings_df()
        movies_df = self._movies_df()
        metrics = self.optimize_alpha(ratings_df, movies_df)

        similarity_matrix, movie_index = self._build_content_model(movies_df)
        cf_model = self._train_cf_model(ratings_df)
        seen = set(ratings_df.loc[ratings_df["user_id"] == user_id, "movie_id"].tolist())

        recommendations = []
        for movie in self.session.query(Movie).order_by(Movie.year.desc()).all():
            if movie.id in seen:
                continue
            cf_score = self._predict_cf(cf_model, user_id, movie.id, ratings_df)
            cbf_score = self._predict_cbf(user_id, movie.id, ratings_df, similarity_matrix, movie_index)
            hybrid_score = metrics["alpha"] * cf_score + (1 - metrics["alpha"]) * cbf_score
            recommendations.append(
                {
                    "id": movie.id,
                    "title": movie.title,
                    "year": movie.year,
                    "quality": movie.quality,
                    "duration": movie.duration,
                    "poster_url": movie.poster_url,
                    "hybrid_score": hybrid_score,
                }
            )

        recommendations.sort(key=lambda item: item["hybrid_score"], reverse=True)
        metrics["recommendations"] = recommendations[:top_n]
        return metrics

    def get_similar_movies(self, movie_id: int, top_n: int = 4):
        movies_df = self._movies_df()
        if movies_df.empty or movie_id not in movies_df["movie_id"].values:
            return []

        similarity_matrix, movie_index = self._build_content_model(movies_df)
        idx = movie_index[movie_id]
        similarity_scores = list(enumerate(similarity_matrix[idx]))
        similarity_scores.sort(key=lambda item: item[1], reverse=True)

        movie_ids = []
        for index, _ in similarity_scores[1 : top_n + 1]:
            movie_ids.append(int(movies_df.iloc[index]["movie_id"]))

        return self.session.query(Movie).filter(Movie.id.in_(movie_ids)).all()

    def get_movie_statistics(self, movie_id: int):
        ratings = self.session.query(Rating).filter_by(movie_id=movie_id).all()
        if not ratings:
            return {"avg_rating": 0.0, "rating_count": 0}
        scores = [rating.score for rating in ratings]
        return {
            "avg_rating": sum(scores) / len(scores),
            "rating_count": len(scores),
        }

    @staticmethod
    def _rmse(actuals, predictions) -> float:
        if not actuals:
            return 0.0
        mse = np.mean([(actual - prediction) ** 2 for actual, prediction in zip(actuals, predictions)])
        return float(np.sqrt(mse))
