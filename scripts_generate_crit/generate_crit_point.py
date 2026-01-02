import os
import pandas as pd
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parents[1]

# ============================================================
# OpenAI クライアント設定
# ============================================================
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ============================================================
# API リトライ付き呼び出し
# ============================================================
@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(10))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs, seed=42)


# ============================================================
# 評価基準生成関数
# ============================================================
def generate_criterion(question, gold_answer):
    system_prompt = """
# 目的

・「質問」が与えられます。
・その「質問」に対する 回答を1〜5点で評価するための評価基準 を作成してください。
・評価基準は 最終スコア別（5点 / 4点 / 3点 / 2点 / 1点） に分けて記述してください。

# 評価基準の記述形式（厳守）

以下の形式を必ず守ってください。

5点:
・……
・……

4点:
・……
・……

3点:
・……
・……

2点:
・……
・……

1点:
・……
・……


# 評価基準作成ルール

・各点数には、その点数に該当する 具体的かつ客観的に判定可能な条件 を箇条書きで記述してください。
・「良い」「十分」「適切」「概ね」など、評価者の解釈に依存する表現は禁止 です。
・評価者が 回答本文を読み、評価基準のみを参照して機械的に採点できる レベルの具体性を持たせてください。
・5点は「質問の要求をすべて満たしている状態」 と定義してください。
・4点以下は、5点との差分として「何が不足・欠落・不正確か」が明確に分かる ように記述してください。
・1つの箇条書き条件には 1つの評価観点のみ を記述してください（複数要素の同時要求は禁止）。
・「質問」への回答に 必須ではない要素（文体、分量、独自解釈など）を評価条件として含めないでください。
・「質問」文に含まれる語句・表現は、評価基準内でも 可能な限り同じ表現を使用 してください。

# 出力ルール
・評価基準のみ を出力してください。
・前置き、補足説明、注意書きは一切出力しないでください。
"""

    user_prompt = f"""
評価基準を生成してください。

# 質問文
{question}
"""

    response = completion_with_backoff(
        model="gpt-5",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_completion_tokens=10000
    )

    return response.choices[0].message.content.strip()


# ============================================================
# DataFrame の評価基準カラムのみを更新
# ============================================================
def update_criterion_column(
    df,
    q_col="質問",
    a_col="正解例",
    criterion_col="評価基準",
    skip_existing=False
):
    criteria = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        # スキップモード & 既に評価基準がある場合
        if (
            skip_existing
            and isinstance(row[criterion_col], str)
            and row[criterion_col].strip()
        ):
            criteria.append(row[criterion_col])
            continue

        criterion = generate_criterion(row[q_col], row[a_col])
        criteria.append(criterion)

    df[criterion_col] = criteria
    return df


# ============================================================
# CSV 入出力（評価基準のみ更新）
# ============================================================
def process_csv(
    input_path,
    output_path,
    q_col="質問",
    a_col="正解例",
    criterion_col="評価基準",
    skip_existing=False
):
    df = pd.read_csv(input_path)

    # カラムチェック
    for col in [q_col, a_col, criterion_col]:
        if col not in df.columns:
            raise ValueError(f"入力 CSV に必要なカラム '{col}' がありません")

    df = update_criterion_column(
        df,
        q_col=q_col,
        a_col=a_col,
        criterion_col=criterion_col,
        skip_existing=skip_existing
    )

    df.to_csv(output_path, index=False)
    print(f"出力完了: {output_path}")

    return df


# ============================================================
# ✨ CSV パスの指定（実行エントリ）
# ============================================================
if __name__ == "__main__":
    # --- 必要なパスをここで編集 ---
    input_csv_path = PROJECT_DIR / "data2025/data_20.csv"
    output_csv_path = PROJECT_DIR / "scripts_generate_crit/crit/crit_point20.csv"

    # 既存の評価基準がある行をスキップしたい場合は True
    skip_existing = False

    process_csv(
        input_path=input_csv_path,
        output_path=output_csv_path,
        q_col="質問",
        a_col="正解例",
        criterion_col="評価基準",
        skip_existing=skip_existing
    )
