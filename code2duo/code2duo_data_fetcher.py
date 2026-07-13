import requests
import pandas as pd
import time
import json
import random

BASE_URL = "https://codeforces.com/api"

# ─────────────────────────────────────────────
# 1. PROBLEMS FETCH 
# ─────────────────────────────────────────────
def fetch_problems():
    print("📦 Fetching problems from Codeforces...")
    resp = requests.get(f"{BASE_URL}/problemset.problems")
    data = resp.json()

    if data['status'] != 'OK':
        print("❌ Error:", data)
        return None

    problems = data['result']['problems']
    stats = data['result']['problemStatistics']

    stats_dict = {(s['contestId'], s['index']): s['solvedCount'] for s in stats}

    rows = []
    for p in problems:
        cid = p.get('contestId')
        idx = p.get('index')
        rows.append({
            'contest_id': cid,
            'index': idx,
            'name': p.get('name'),
            'rating': p.get('rating'),          
            'tags': ', '.join(p.get('tags', [])),
            'solved_count': stats_dict.get((cid, idx), 0)
        })

    df = pd.DataFrame(rows)
    df = df.dropna(subset=['rating'])           
    df = df[df['rating'] >= 800]              
    df.to_csv('problems.csv', index=False)
    print(f"✅ {len(df)} problems saved → problems.csv")
    return df


# ─────────────────────────────────────────────
# 2. REAL USERS FETCH 
# ─────────────────────────────────────────────

# Top competitive programmers + mix of average users
SEED_HANDLES = [
    "tourist", "Petr", "Um_nik", "ecnerwala", "jiangly",
    "ksun48", "Radewoosh", "scott_wu", "neal", "apiad",
    "Geothermal", "pajenegod", "SecondThread", "Benq", "maroonrk",
    "gamegame", "244mhq", "ko_osaga", "heno239", "LayCurse",
    "mojtaba", "vepifanov", "Vercingetorix", "Maksim1744", "Radewoosh"
]

def fetch_user_info(handles):
    """Batch mein users fetch karo (max 500 ek baar mein)"""
    chunk_size = 100
    all_users = []

    for i in range(0, len(handles), chunk_size):
        chunk = handles[i:i+chunk_size]
        handle_str = ';'.join(chunk)
        try:
            resp = requests.get(f"{BASE_URL}/user.info?handles={handle_str}")
            data = resp.json()
            if data['status'] == 'OK':
                all_users.extend(data['result'])
            time.sleep(0.5)  
        except Exception as e:
            print(f"⚠️ Error fetching chunk: {e}")

    return all_users

def fetch_user_rating_history(handle):
    """Ek user ka rating history fetch karo"""
    try:
        resp = requests.get(f"{BASE_URL}/user.rating?handle={handle}")
        data = resp.json()
        if data['status'] == 'OK':
            return data['result']
    except:
        pass
    return []

def fetch_users_dataset(handles):
    print(f"\n👥 Fetching {len(handles)} users...")
    users_raw = fetch_user_info(handles)

    rows = []
    for u in users_raw:
        rows.append({
            'handle': u.get('handle'),
            'rating': u.get('rating', 0),
            'max_rating': u.get('maxRating', 0),
            'rank': u.get('rank', 'unrated'),
            'max_rank': u.get('maxRank', 'unrated'),
            'contribution': u.get('contribution', 0),
            'friend_of_count': u.get('friendOfCount', 0),
        })

    df = pd.DataFrame(rows)
    df.to_csv('users.csv', index=False)
    print(f"✅ {len(df)} users saved → users.csv")
    return df


# ─────────────────────────────────────────────
# 3. SYNTHETIC MATCHMAKING DATA 
# ─────────────────────────────────────────────
def generate_matchmaking_data(n_users=1000):
    """
    Synthetic duel history generate karo for ML training.
    Real users ke ratings use karke realistic data banao.
    """
    import numpy as np
    print(f"\n🤖 Generating {n_users} synthetic users for matchmaking...")

    np.random.seed(42)

    # Rating distribution — Codeforces  (mostly 1000-1800)
    ratings = np.random.lognormal(mean=7.1, sigma=0.3, size=n_users).astype(int)
    ratings = np.clip(ratings, 800, 3500)

    users = pd.DataFrame({
        'user_id': [f"user_{i}" for i in range(n_users)],
        'rating': ratings,
        'problems_solved': np.random.randint(10, 800, n_users),
        'contests_participated': np.random.randint(1, 100, n_users),
        'win_rate': np.random.beta(2, 3, n_users).round(3),      # 0 to 1
        'avg_solve_time_min': np.random.normal(25, 10, n_users).clip(5, 60).round(1),
        'favorite_tag': np.random.choice(
            ['dp', 'greedy', 'graphs', 'math', 'binary search', 
             'strings', 'trees', 'implementation'], n_users
        )
    })

    # Duel history generate 
    duels = []
    for _ in range(5000):
        u1, u2 = np.random.choice(n_users, 2, replace=False)
        r1 = users.iloc[u1]['rating']
        r2 = users.iloc[u2]['rating']

        # Elo probability — higher rated player  probability
        prob_u1_wins = 1 / (1 + 10 ** ((r2 - r1) / 400))
        winner = u1 if np.random.random() < prob_u1_wins else u2

        duels.append({
            'user1_id': users.iloc[u1]['user_id'],
            'user2_id': users.iloc[u2]['user_id'],
            'user1_rating': r1,
            'user2_rating': r2,
            'rating_diff': abs(r1 - r2),
            'winner_id': users.iloc[winner]['user_id'],
            'user1_won': int(winner == u1),
            'problem_rating': int((r1 + r2) / 2),       # problem difficulty
        })

    users.to_csv('synthetic_users.csv', index=False)
    duels_df = pd.DataFrame(duels)
    duels_df.to_csv('duel_history.csv', index=False)

    print(f"✅ {n_users} users saved → synthetic_users.csv")
    print(f"✅ 5000 duels saved → duel_history.csv")
    return users, duels_df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("   CODE2DUO — Data Collection Pipeline")
    print("=" * 50)

    # 1. Problems
    problems_df = fetch_problems()

    # 2. Real users 
    users_df = fetch_users_dataset(SEED_HANDLES)

    # 3. Synthetic matchmaking data
    synthetic_users, duels = generate_matchmaking_data(n_users=1000)

    print("\n" + "=" * 50)
    print("✅ All data saved!")
    print("  📄 problems.csv        — problem bank")
    print("  📄 users.csv           — real CF users")
    print("  📄 synthetic_users.csv — 1000 fake users")
    print("  📄 duel_history.csv    — 5000 duel records")
    print("\nNext step: Elo matchmaking model train karo!")
    print("=" * 50)
