import os
import pandas as pd
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parents[1]

# --- OpenAI クライアント設定 ---
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ============================================================
# API リトライ付き呼び出し（Chat Completions）
# ============================================================
@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(10))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs, seed=42)

# ============================================================
# ① 質問に回答させる
# ============================================================
def generate_answer(question: str) -> str:
    system_prompt = "あなたは質問に対して、過不足なく説明する回答者です。"
    user_prompt = f"以下の質問に回答してください。\n\n# 質問\n{question}"

    resp = completion_with_backoff(
        model="gpt-5",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_completion_tokens=10000,
    )
    return resp.choices[0].message.content.strip()

# ============================================================
# ② 回答する上で重要だった部分を抽出（①の出力を assistant として渡す）
# ============================================================
def extract_important_points(question: str, answer_text: str) -> str:
    system_prompt = "あなたは回答を分析する立場です。"
    user_prompt = f"""直前の回答は、次の質問への回答です。

# 質問
{question}

この質問に回答する上で、特に重要だった部分を箇条書きで列挙してください。
"""

    resp = completion_with_backoff(
        model="gpt-5",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": answer_text},
            {"role": "user", "content": user_prompt},
        ],
        max_completion_tokens=10000,
    )
    return resp.choices[0].message.content.strip()

# ============================================================
# ③ 重要点を踏まえて評価基準を生成（②の出力を assistant として渡す）
# ============================================================
def generate_criterion(question: str, important_points_text: str) -> str:
    system_prompt = """
あなたは評価基準を設計する専門家です。
"""
    user_prompt = f"""
以下の質問に対する回答を1〜5点で評価するための評価基準を作成してください。

# 質問
{question}

# ルール
・加点の合計は4点（ベース点1点）
・一つの基準に一つの要素のみ
・回答に必須でない要素を要求しない
・各行は必ず「: +N点」で終わらせる
"""

    resp = completion_with_backoff(
        model="gpt-5",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": important_points_text},
            {"role": "user", "content": user_prompt},
        ],
        max_completion_tokens=10000,
    )
    return resp.choices[0].message.content.strip()

# ============================================================
# DataFrame の評価基準カラムを更新（3ターン構造）
# ============================================================
def update_criterion_column(
    df: pd.DataFrame,
    q_col: str = "質問",
    criterion_col: str = "評価基準",
    skip_existing: bool = False,
):
    criteria = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        if skip_existing and isinstance(row.get(criterion_col, ""), str) and row[criterion_col].strip():
            criteria.append(row[criterion_col])
            continue

        question = str(row[q_col])

        answer = generate_answer(question)
        important_points = extract_important_points(question, answer)
        criterion = generate_criterion(question, important_points)

        criteria.append(criterion)

    df[criterion_col] = criteria
    return df

# ============================================================
# CSV 入出力
# ============================================================
def process_csv(
    input_path: str,
    output_path: str,
    q_col: str = "質問",
    criterion_col: str = "評価基準",
    skip_existing: bool = False,
):
    df = pd.read_csv(input_path)

    if q_col not in df.columns or criterion_col not in df.columns:
        raise ValueError("CSV に必要なカラムが存在しません")

    df = update_criterion_column(
        df,
        q_col=q_col,
        criterion_col=criterion_col,
        skip_existing=skip_existing,
    )

    df.to_csv(output_path, index=False)
    print(f"出力完了: {output_path}")
    return df

if __name__ == "__main__":
    input_csv_path = PROJECT_DIR / "data2025/data_20.csv"
    output_csv_path = PROJECT_DIR / "scripts_generate_crit/crit/crit_multi20.csv"
    skip_existing = False

    process_csv(
        input_path=input_csv_path,
        output_path=output_csv_path,
        q_col="質問",
        criterion_col="評価基準",
        skip_existing=skip_existing,
    )
