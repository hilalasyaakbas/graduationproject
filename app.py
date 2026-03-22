import os
from sqlalchemy import or_, text

from flask import Flask, flash, g, redirect, render_template, request, send_from_directory, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from imdb_service import fetch_imdb_movie_data
from models import Movie, Rating, User, db
from recommender import HybridRecommender


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MOVIE_SEEDS = [
    {
        "title": "Dune",
        "year": 2021,
        "quality": "4K",
        "duration": 155,
        "poster_url": "/static/assets/images/movie-1.png",
        "description": "Sci-fi epic on Arrakis with politics, prophecy and survival.",
        "genres": "Science Fiction Adventure Drama",
        "imdb_rating": 8.0,
        "imdb_url": "https://www.imdb.com/title/tt1160419/",
    },
    {
        "title": "Interstellar",
        "year": 2014,
        "quality": "HD",
        "duration": 169,
        "poster_url": "/static/assets/images/movie-2.png",
        "description": "Explorers travel through space to protect humanity's future.",
        "genres": "Science Fiction Drama Adventure",
        "imdb_rating": 8.7,
        "imdb_url": "https://www.imdb.com/title/tt0816692/",
    },
    {
        "title": "The Dark Knight",
        "year": 2008,
        "quality": "HD",
        "duration": 152,
        "poster_url": "/static/assets/images/movie-3.png",
        "description": "Batman confronts the Joker in a dark crime thriller.",
        "genres": "Action Crime Drama",
        "imdb_rating": 9.0,
        "imdb_url": "https://www.imdb.com/title/tt0468569/",
    },
    {
        "title": "Inception",
        "year": 2010,
        "quality": "HD",
        "duration": 148,
        "poster_url": "/static/assets/images/movie-4.png",
        "description": "A team enters dreams to plant an idea deep in the subconscious.",
        "genres": "Science Fiction Action Thriller",
        "imdb_rating": 8.8,
        "imdb_url": "https://www.imdb.com/title/tt1375666/",
    },
    {
        "title": "La La Land",
        "year": 2016,
        "quality": "HD",
        "duration": 128,
        "poster_url": "/static/assets/images/movie-5.png",
        "description": "A musician and an actress fall in love while chasing ambition.",
        "genres": "Romance Drama Music",
        "imdb_rating": 8.0,
        "imdb_url": "https://www.imdb.com/title/tt3783958/",
    },
    {
        "title": "Parasite",
        "year": 2019,
        "quality": "HD",
        "duration": 132,
        "poster_url": "/static/assets/images/movie-6.png",
        "description": "A sharp social satire about class tension and deception.",
        "genres": "Thriller Drama",
        "imdb_rating": 8.5,
        "imdb_url": "https://www.imdb.com/title/tt6751668/",
    },
    {
        "title": "Avengers: Endgame",
        "year": 2019,
        "quality": "4K",
        "duration": 181,
        "poster_url": "/static/assets/images/movie-7.png",
        "description": "Earth's mightiest heroes reunite for a final battle.",
        "genres": "Action Adventure Science Fiction",
        "imdb_rating": 8.4,
        "imdb_url": "https://www.imdb.com/title/tt4154796/",
    },
    {
        "title": "The Grand Budapest Hotel",
        "year": 2014,
        "quality": "HD",
        "duration": 100,
        "poster_url": "/static/assets/images/movie-8.png",
        "description": "A colorful hotel concierge becomes entangled in a mystery.",
        "genres": "Comedy Adventure Crime",
        "imdb_rating": 8.1,
        "imdb_url": "https://www.imdb.com/title/tt2278388/",
    },
    {
        "title": "Blade Runner 2049",
        "year": 2017,
        "quality": "4K",
        "duration": 164,
        "poster_url": "/static/assets/images/upcoming-1.png",
        "description": "A replicant hunter uncovers a secret that could change society.",
        "genres": "Science Fiction Mystery Drama",
        "imdb_rating": 8.0,
        "imdb_url": "https://www.imdb.com/title/tt1856101/",
    },
    {
        "title": "Whiplash",
        "year": 2014,
        "quality": "HD",
        "duration": 106,
        "poster_url": "/static/assets/images/upcoming-2.png",
        "description": "A young drummer is pushed to the edge by a ruthless mentor.",
        "genres": "Drama Music",
        "imdb_rating": 8.5,
        "imdb_url": "https://www.imdb.com/title/tt2582802/",
    },
    {
        "title": "Mad Max: Fury Road",
        "year": 2015,
        "quality": "4K",
        "duration": 120,
        "poster_url": "/static/assets/images/upcoming-3.png",
        "description": "A relentless desert chase fuels this action spectacle.",
        "genres": "Action Adventure Science Fiction",
        "imdb_rating": 8.1,
        "imdb_url": "https://www.imdb.com/title/tt1392190/",
    },
    {
        "title": "The Social Network",
        "year": 2010,
        "quality": "HD",
        "duration": 120,
        "poster_url": "/static/assets/images/upcoming-4.png",
        "description": "The origin of a social media empire and its fallout.",
        "genres": "Drama Biography",
        "imdb_rating": 7.8,
        "imdb_url": "https://www.imdb.com/title/tt1285016/",
    },
]


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "filmlane-dev-secret")
    app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv(
        "DATABASE_URL",
        "sqlite:///" + os.path.join(BASE_DIR, "app.db"),
    )
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db.init_app(app)

    with app.app_context():
        db.create_all()
        ensure_schema()
        seed_database()

    @app.before_request
    def load_current_user():
        user_id = session.get("user_id")
        g.current_user = User.query.get(user_id) if user_id else None

    @app.context_processor
    def inject_user():
        return {"current_user": g.get("current_user")}

    def require_user():
        if not g.get("current_user"):
            flash("Bu sayfayi gormek icin giris yapmalisin.", "error")
            return None
        return g.current_user

    @app.route("/")
    def home():
        featured = Movie.query.order_by(Movie.imdb_rating.desc().nullslast(), Movie.year.desc()).limit(12).all()
        return render_template("home.html", featured_movies=featured)

    @app.route("/find-movie")
    def find_movie():
        query = request.args.get("q", "").strip()
        movie_query = Movie.query
        if query:
            movie_query = movie_query.filter(Movie.title.ilike(f"%{query}%"))
        movies = movie_query.order_by(Movie.imdb_rating.desc().nullslast(), Movie.title.asc()).all()
        user_ratings = {}
        if g.get("current_user"):
            user_ratings = {rating.movie_id: rating.score for rating in g.current_user.ratings}
        return render_template(
            "search.html",
            movies=movies,
            query=query,
            user_ratings=user_ratings,
        )

    @app.route("/favicon.svg")
    def favicon():
        return send_from_directory(BASE_DIR, "favicon.svg")

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")

            if not username or not email or not password:
                flash("Tum alanlari doldurman gerekiyor.", "error")
            elif User.query.filter((User.username == username) | (User.email == email)).first():
                flash("Bu kullanici adi veya e-posta zaten kayitli.", "error")
            else:
                user = User(
                    username=username,
                    email=email,
                    password_hash=generate_password_hash(password),
                )
                db.session.add(user)
                db.session.commit()
                session["user_id"] = user.id
                flash("Kayit basarili. Cold start puanlamasina gecebilirsin.", "success")
                return redirect(url_for("cold_start"))

        return render_template("auth.html", mode="register")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            user = User.query.filter_by(email=email).first()

            if not user or not check_password_hash(user.password_hash, password):
                flash("Giris bilgileri gecersiz.", "error")
            else:
                session["user_id"] = user.id
                flash("Tekrar hos geldin.", "success")
                return redirect(url_for("recommendations"))

        return render_template("auth.html", mode="login")

    @app.route("/logout")
    def logout():
        session.clear()
        flash("Cikis yapildi.", "success")
        return redirect(url_for("home"))

    @app.route("/rate/<int:movie_id>", methods=["POST"])
    def rate_movie(movie_id: int):
        user = require_user()
        if not user:
            return redirect(url_for("login"))

        try:
            score = float(request.form.get("score", 0))
        except ValueError:
            score = 0

        if not 1 <= score <= 5:
            flash("Puan 1 ile 5 arasinda olmali.", "error")
        else:
            save_rating(user.id, movie_id, score)
            if len(user.ratings) >= 10:
                user.is_cold_start_done = True
                db.session.commit()
            flash("Puanin kaydedildi.", "success")

        next_url = request.form.get("next_url") or url_for("movie_detail", movie_id=movie_id)
        return redirect(next_url)

    @app.route("/cold-start", methods=["GET", "POST"])
    def cold_start():
        user = require_user()
        if not user:
            return redirect(url_for("login"))

        if request.method == "POST":
            saved = 0
            for key, value in request.form.items():
                if not key.startswith("rating_") or not value:
                    continue
                movie_id = int(key.split("_", 1)[1])
                save_rating(user.id, movie_id, float(value))
                saved += 1

            if len(user.ratings) >= 10:
                user.is_cold_start_done = True
                db.session.commit()
                flash("Cold start tamamlandi. Artik hibrit onerileri gorebilirsin.", "success")
                return redirect(url_for("recommendations"))

            if saved:
                flash(f"{saved} film puani kaydedildi. 10 filme ulasana kadar devam edebilirsin.", "success")
            else:
                flash("En az bir filme puan vermen gerekiyor.", "error")
            return redirect(url_for("cold_start", **request.args))

        genre = request.args.get("genre", "").strip()
        year_min = request.args.get("year_min", type=int)
        year_max = request.args.get("year_max", type=int)
        imdb_min = request.args.get("imdb_min", type=float)
        q = request.args.get("q", "").strip()

        movies_query = Movie.query
        if genre:
            movies_query = movies_query.filter(Movie.genres.ilike(f"%{genre}%"))
        if year_min:
            movies_query = movies_query.filter(Movie.year >= year_min)
        if year_max:
            movies_query = movies_query.filter(Movie.year <= year_max)
        if imdb_min:
            movies_query = movies_query.filter(Movie.imdb_rating >= imdb_min)
        if q:
            movies_query = movies_query.filter(Movie.title.ilike(f"%{q}%"))

        movies = movies_query.order_by(Movie.imdb_rating.desc().nullslast(), Movie.year.desc()).all()
        existing_ratings = {rating.movie_id: rating.score for rating in user.ratings}
        genres = sorted({genre_name.strip() for movie in Movie.query.all() for genre_name in movie.genres.split() if genre_name.strip()})

        return render_template(
            "cold_start.html",
            movies=movies,
            existing_ratings=existing_ratings,
            genres=genres,
            rated_count=len(user.ratings),
        )

    @app.route("/recommendations")
    def recommendations():
        user = require_user()
        if not user:
            return redirect(url_for("login"))
        if not user.is_cold_start_done or len(user.ratings) < 10:
            flash("Once cold start puanlamasini tamamlamalisin.", "error")
            return redirect(url_for("cold_start"))

        recommender = HybridRecommender(db.session)
        result = recommender.recommend_for_user(user.id, top_n=8)
        return render_template(
            "recommendations.html",
            recommended_movies=result["recommendations"],
            alpha=result["alpha"],
            rmse=result["rmse"],
            cf_rmse=result["cf_rmse"],
            cbf_rmse=result["cbf_rmse"],
        )

    @app.route("/movie/<int:movie_id>", methods=["GET", "POST"])
    def movie_detail(movie_id: int):
        movie = Movie.query.get_or_404(movie_id)
        user = g.get("current_user")

        if request.method == "POST":
            return rate_movie(movie_id)

        recommender = HybridRecommender(db.session)
        similar_movies = recommender.get_similar_movies(movie.id, top_n=4)
        stats = recommender.get_movie_statistics(movie.id)
        user_rating = None
        imdb_data = fetch_imdb_movie_data(movie.title, movie.year, movie.imdb_url)

        if imdb_data:
            if not movie.imdb_url and imdb_data.get("imdb_url"):
                movie.imdb_url = imdb_data["imdb_url"]
            if imdb_data.get("imdb_rating"):
                movie.imdb_rating = imdb_data["imdb_rating"]
            if imdb_data.get("description") and movie.description != imdb_data["description"]:
                movie.description = imdb_data["description"]
            if imdb_data.get("genres"):
                movie.genres = " ".join(imdb_data["genres"])
            if imdb_data.get("duration") and not movie.duration:
                movie.duration = imdb_data["duration"]
            db.session.commit()

        if user:
            rating = Rating.query.filter_by(user_id=user.id, movie_id=movie.id).first()
            user_rating = rating.score if rating else None

        return render_template(
            "movie_detail.html",
            movie=movie,
            similar_movies=similar_movies,
            stats=stats,
            user_rating=user_rating,
            imdb_data=imdb_data,
        )

    return app


def ensure_schema() -> None:
    inspector = db.inspect(db.engine)
    columns = {column["name"] for column in inspector.get_columns("movies")}
    statements = []
    if "imdb_rating" not in columns:
        statements.append("ALTER TABLE movies ADD COLUMN imdb_rating FLOAT")
    if "imdb_url" not in columns:
        statements.append("ALTER TABLE movies ADD COLUMN imdb_url VARCHAR(255)")

    for statement in statements:
        db.session.execute(text(statement))
    if statements:
        db.session.commit()


def save_rating(user_id: int, movie_id: int, score: float) -> None:
    rating = Rating.query.filter_by(user_id=user_id, movie_id=movie_id).first()
    if rating:
        rating.score = score
    else:
        db.session.add(Rating(user_id=user_id, movie_id=movie_id, score=score))
    db.session.commit()


def seed_database() -> None:
    if Movie.query.count() == 0:
        movies = [Movie(**movie_data) for movie_data in MOVIE_SEEDS]
        db.session.add_all(movies)
        db.session.commit()
    else:
        seed_map = {item["title"]: item for item in MOVIE_SEEDS}
        for movie in Movie.query.all():
            source = seed_map.get(movie.title)
            if not source:
                continue
            if movie.imdb_rating is None:
                movie.imdb_rating = source["imdb_rating"]
            if not movie.imdb_url:
                movie.imdb_url = source["imdb_url"]
        db.session.commit()

    if User.query.count() == 0:
        users = [
            User(username="neo", email="neo@example.com", password_hash=generate_password_hash("password"), is_cold_start_done=True),
            User(username="mia", email="mia@example.com", password_hash=generate_password_hash("password"), is_cold_start_done=True),
            User(username="arthur", email="arthur@example.com", password_hash=generate_password_hash("password"), is_cold_start_done=True),
            User(username="lena", email="lena@example.com", password_hash=generate_password_hash("password"), is_cold_start_done=True),
            User(username="omar", email="omar@example.com", password_hash=generate_password_hash("password"), is_cold_start_done=True),
        ]
        db.session.add_all(users)
        db.session.commit()

    if Rating.query.count() == 0:
        score_map = {
            1: {1: 5, 2: 5, 3: 4, 4: 5, 7: 4, 9: 5, 10: 3, 11: 4},
            2: {5: 5, 6: 4, 8: 4, 10: 5, 12: 4, 2: 3, 4: 4, 9: 3},
            3: {3: 5, 4: 4, 6: 5, 7: 4, 11: 5, 1: 3, 2: 4, 8: 3},
            4: {2: 5, 4: 5, 5: 4, 9: 5, 10: 4, 11: 4, 1: 4, 6: 3},
            5: {1: 4, 3: 5, 7: 5, 8: 4, 9: 4, 10: 3, 12: 5, 5: 2},
        }
        ratings = []
        for user_id, values in score_map.items():
            for movie_id, score in values.items():
                ratings.append(Rating(user_id=user_id, movie_id=movie_id, score=score))
        db.session.add_all(ratings)
        db.session.commit()


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
