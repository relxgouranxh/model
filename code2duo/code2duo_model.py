

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════
# 1. ELO RATING SYSTEM
# ═══════════════════════════════════════════════

class EloMatchmaker:
    """
    Elo system — Chess wala same formula.
    Har duel ke baad rating update hoti hai.
    """

    def __init__(self, k=32, initial_rating=1200):
        self.K = k                          # kitna fast rating change ho
        self.initial_rating = initial_rating
        self.ratings = {}                   # { user_id: rating }

    def get_rating(self, user_id):
        return self.ratings.get(user_id, self.initial_rating)

    def expected_score(self, rating_a, rating_b):
        """Probability ki A jeete B se"""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update_ratings(self, winner_id, loser_id):
        """Duel ke baad dono ki rating update karo"""
        ra = self.get_rating(winner_id)
        rb = self.get_rating(loser_id)

        ea = self.expected_score(ra, rb)    # winner ki expected prob
        eb = self.expected_score(rb, ra)    # loser ki expected prob

        # Winner → actual score = 1, Loser → actual score = 0
        self.ratings[winner_id] = ra + self.K * (1 - ea)
        self.ratings[loser_id]  = rb + self.K * (0 - eb)

        return self.ratings[winner_id], self.ratings[loser_id]

    def find_best_match(self, user_id, all_users, max_diff=200):
        """
        Similar rating wala opponent dhundho.
        max_diff → kitna rating difference acceptable hai
        """
        my_rating = self.get_rating(user_id)
        candidates = []

        for uid in all_users:
            if uid == user_id:
                continue
            their_rating = self.get_rating(uid)
            diff = abs(my_rating - their_rating)
            if diff <= max_diff:
                candidates.append((uid, diff))

        if not candidates:
            return None

        # Sabse close rating wala
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]

    def train_on_history(self, duel_history_df):
        """Past duel history se saari ratings compute karo"""
        print("🎮 Training Elo on duel history...")
        for _, row in duel_history_df.iterrows():
            winner = row['winner_id']
            loser  = row['user2_id'] if row['user1_won'] else row['user1_id']
            self.update_ratings(winner, loser)
        print(f"✅ Elo trained on {len(duel_history_df)} duels")
        print(f"   Total users rated: {len(self.ratings)}")


# ═══════════════════════════════════════════════
# 2. PROBLEM RECOMMENDER
# ═══════════════════════════════════════════════

class ProblemRecommender:
    """
    Content-based filtering.
    Duel ke liye fair + relevant problem suggest karo.
    """

    def __init__(self):
        self.problems_df = None
        self.tag_list = []

    def load_problems(self, csv_path='problems.csv'):
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=['rating', 'tags'])
        self.problems_df = df

        # Saare unique tags collect karo
        all_tags = set()
        for tags in df['tags']:
            for t in str(tags).split(', '):
                all_tags.add(t.strip())
        self.tag_list = sorted(list(all_tags))
        print(f"✅ {len(df)} problems loaded | {len(self.tag_list)} unique tags")

    def recommend(self, user1_rating, user2_rating,
                  user1_weak_tags=None, user2_weak_tags=None,
                  n=5):
        """
        Dono users ke liye problem suggest karo.

        Logic:
          - Difficulty → average of both ratings ± 100
          - Tags       → prefer common weak topics
        """
        avg_rating = (user1_rating + user2_rating) / 2
        low  = avg_rating - 100
        high = avg_rating + 100

        # Difficulty ke hisaab se filter
        pool = self.problems_df[
            (self.problems_df['rating'] >= low) &
            (self.problems_df['rating'] <= high)
        ].copy()

        if pool.empty:
            # Range badha do agar kuch nahi mila
            pool = self.problems_df[
                (self.problems_df['rating'] >= avg_rating - 200) &
                (self.problems_df['rating'] <= avg_rating + 200)
            ].copy()

        # Weak tags prefer karo (scoring)
        weak_tags = set()
        if user1_weak_tags:
            weak_tags.update(user1_weak_tags)
        if user2_weak_tags:
            weak_tags.update(user2_weak_tags)

        def tag_score(tags_str):
            tags = set(str(tags_str).split(', '))
            return len(tags & weak_tags)   # overlap kitna hai

        if weak_tags:
            pool['tag_score'] = pool['tags'].apply(tag_score)
            pool = pool.sort_values('tag_score', ascending=False)

        # Top N recommend karo
        recommended = pool.head(n)[['contest_id', 'index', 'name', 'rating', 'tags']]
        return recommended


# ═══════════════════════════════════════════════
# 3. WEAK TOPIC DETECTOR (K-Means)
# ═══════════════════════════════════════════════

class WeakTopicDetector:
    """
    K-Means clustering se detect karo:
    - User ka strong area kaunsa hai
    - Weak area kaunsa hai
    """

    TAGS = ['dp', 'greedy', 'graphs', 'math',
            'binary search', 'strings', 'trees', 'implementation']

    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.scaler = StandardScaler()

    def _user_to_vector(self, user_stats: dict):
        """
        User ke stats ko vector mein convert karo.
        user_stats = { 'dp': 10, 'greedy': 5, 'graphs': 0, ... }
        """
        return [user_stats.get(tag, 0) for tag in self.TAGS]

    def fit(self, users_stats: list):
        """
        users_stats = list of dicts
        [{ 'dp': 10, 'greedy': 5, ... }, ...]
        """
        X = np.array([self._user_to_vector(u) for u in users_stats])
        X_scaled = self.scaler.fit_transform(X)
        self.kmeans.fit(X_scaled)
        print(f"✅ K-Means trained | {self.n_clusters} clusters found")

    def get_weak_topics(self, user_stats: dict, top_n=3):
        """
        Ek user ke weak topics return karo.
        Weak = jinme solve count sabse kam hai.
        """
        vec = self._user_to_vector(user_stats)
        tag_scores = list(zip(self.TAGS, vec))
        tag_scores.sort(key=lambda x: x[1])    # ascending → weak pehle
        weak = [t for t, _ in tag_scores[:top_n]]
        return weak

    def get_cluster_label(self, user_stats: dict):
        """User kis group mein hai — beginner/mid/expert"""
        vec = np.array([self._user_to_vector(user_stats)])
        vec_scaled = self.scaler.transform(vec)
        cluster = self.kmeans.predict(vec_scaled)[0]
        labels = {0: 'Beginner', 1: 'Intermediate', 2: 'Expert'}
        return labels.get(cluster, f"Cluster {cluster}")


# ═══════════════════════════════════════════════
# DEMO — Sab ek saath chalao
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("   CODE2DUO — ML Pipeline Demo")
    print("=" * 55)

    # ── Load data ──
    duels_df   = pd.read_csv('duel_history.csv')
    users_df   = pd.read_csv('synthetic_users.csv')

    # ── 1. Elo System ──
    print("\n📊 ELO MATCHMAKING")
    print("-" * 30)
    elo = EloMatchmaker(k=32)
    elo.train_on_history(duels_df)

    # Sample match dhundho
    sample_user = users_df['user_id'].iloc[0]
    all_user_ids = users_df['user_id'].tolist()

    # Kuch users ko manually rate karo (demo ke liye)
    for uid in all_user_ids[:50]:
        row = users_df[users_df['user_id'] == uid].iloc[0]
        elo.ratings[uid] = float(row['rating'])

    match = elo.find_best_match(sample_user, all_user_ids[:50])
    print(f"   User:     {sample_user} (rating: {elo.get_rating(sample_user):.0f})")
    print(f"   Match:    {match} (rating: {elo.get_rating(match):.0f})")
    print(f"   Diff:     {abs(elo.get_rating(sample_user) - elo.get_rating(match)):.0f}")

    # ── 2. Problem Recommender ──
    print("\n🎯 PROBLEM RECOMMENDER")
    print("-" * 30)
    recommender = ProblemRecommender()
    recommender.load_problems('problems.csv')

    r1 = elo.get_rating(sample_user)
    r2 = elo.get_rating(match)
    recs = recommender.recommend(
        user1_rating=r1,
        user2_rating=r2,
        user1_weak_tags=['dp', 'graphs'],
        user2_weak_tags=['graphs', 'trees'],
        n=5
    )
    print(f"   Recommended problems for {r1:.0f} vs {r2:.0f} rating duel:")
    print(recs.to_string(index=False))

    # ── 3. Weak Topic Detector ──
    print("\n🔍 WEAK TOPIC DETECTOR")
    print("-" * 30)

    # Fake user stats banana (real mein submissions se aayega)
    np.random.seed(42)
    users_stats = []
    tags = ['dp', 'greedy', 'graphs', 'math',
            'binary search', 'strings', 'trees', 'implementation']
    for _ in range(200):
        stats = {tag: int(np.random.exponential(10)) for tag in tags}
        users_stats.append(stats)

    detector = WeakTopicDetector(n_clusters=3)
    detector.fit(users_stats)

    # Sample user ke weak topics
    my_stats = {
        'dp': 2, 'greedy': 15, 'graphs': 1, 'math': 8,
        'binary search': 20, 'strings': 3, 'trees': 1, 'implementation': 25
    }
    weak = detector.get_weak_topics(my_stats)
    level = detector.get_cluster_label(my_stats)
    print(f"   User level:   {level}")
    print(f"   Weak topics:  {weak}")
    print(f"   → Recommender inhi tags ki problems dega!")

    print("\n" + "=" * 55)
    print("✅ Pipeline complete!")
    print("   Next step: FastAPI backend mein integrate karo")
    print("=" * 55)
