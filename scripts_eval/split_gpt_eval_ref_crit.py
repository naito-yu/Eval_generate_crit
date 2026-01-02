import os
from openai import OpenAI
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm
from scipy.stats import pearsonr
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parents[1]

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ============================================================
# GPT-5 API with retry
# ============================================================
@retry(wait=wait_random_exponential(min=1, max=30),
       stop=stop_after_attempt(10))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs,seed=42)


# ============================================================
# ★ 評価基準1つを採点（Claude/Geminiと統一）
# ============================================================
def gpt5_eval_single_criterion(pred, input_text, output_text, criterion):

    system_message = f"""あなたは採点者です。
回答が評価基準を満たしているかだけを厳密に判断し採点してください。
回答を1,2,3,4,5の5段階で採点し、数字と採点理由を出力してください。

# 問題
{input_text}

# 正解例
{output_text}

# 評価基準
{criterion}

# 出力形式
1行目: 数字のみ
2行以降: 判断理由
"""
    user_prompt = f"# 回答\n{pred}"

    response = completion_with_backoff(
        model="gpt-5",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user",    "content": user_prompt},
        ],
        max_completion_tokens=3000
    )

    content = response.choices[0].message.content

    try:
        score_str, reason = content.split("\n", 1)
        score = int(score_str.strip())
    except:
        score = None
        reason = content

    return score, reason



# ============================================================
# ★ メイン処理（split → 個別採点 → 合算）
# ============================================================
def run_scoring(input_csv_path, output_csv_path):

    df = pd.read_csv(input_csv_path)

    # 出力列（Claude/Gemini と統一）
    for col in [
        "GPT5評価1", "GPT5理由1", "GPT5評価1_scores",
        "GPT5評価2", "GPT5理由2", "GPT5評価2_scores",
        "GPT5評価3", "GPT5理由3", "GPT5評価3_scores",
    ]:
        df[col] = None

    # ---------- GPT 採点 ----------
    for idx in tqdm(df.index, desc="GPT-5 採点中"):
        row = df.loc[idx]

        question = row["質問"]
        gold     = row["正解例"]

        # ★ 評価基準 split（すべてのLLM共通仕様）
        criteria_list = [
            c.strip() for c in str(row["評価基準"]).split("・") if c.strip()
        ]

        # ---------- 回答 1〜3 ----------
        for ai in [1, 2, 3]:

            pred = row[f"回答{ai}"]

            criterion_scores = []
            criterion_reasons = []

            # --- 基準ごとに GPT-5 評価 ---
            for criterion in criteria_list:
                score, reason = gpt5_eval_single_criterion(pred, question, gold, criterion)
                score = 0 if score is None else score

                criterion_scores.append(score)
                criterion_reasons.append(reason)

            # ★ 合算スコア（あなたの仕様）
            final_score = min(1 + sum(criterion_scores), 5)

            # 保存
            df.at[idx, f"GPT5評価{ai}"] = final_score
            df.at[idx, f"GPT5理由{ai}"] = str(criterion_reasons)
            df.at[idx, f"GPT5評価{ai}_scores"] = str(criterion_scores)

    # ============================================================
    # 相関 & Accuracy（元コード通り）
    # ============================================================
    gpt_all = pd.concat([
        df["GPT5評価1"], df["GPT5評価2"], df["GPT5評価3"]
    ], ignore_index=True)

    human_all = pd.concat([
        df["人手評価1"], df["人手評価2"], df["人手評価3"]
    ], ignore_index=True)

    gpt_all = pd.to_numeric(gpt_all, errors="coerce")
    human_all = pd.to_numeric(human_all, errors="coerce")

    valid = pd.concat([gpt_all, human_all], axis=1).dropna()

    gpt_valid   = valid.iloc[:, 0].values
    human_valid = valid.iloc[:, 1].values

    if len(gpt_valid) > 1:
        r, p = pearsonr(gpt_valid, human_valid)
    else:
        r = p = float("nan")

    accuracy = (gpt_valid == human_valid).mean() if len(gpt_valid) > 0 else float("nan")

    df["縦結合_相関_r"]  = r
    df["縦結合_p値"]    = p
    df["縦結合_Accuracy"] = accuracy

    # ---------- 保存 ----------
    df.to_csv(output_csv_path, index=False)

    print("\n===== 保存完了 =====")
    print(f"✓ 出力ファイル: {output_csv_path}")
    print(f"✓ 相関 r = {r}")
    print(f"✓ p値 = {p}")
    print(f"✓ Accuracy = {accuracy}")


# ============================================================
# ★ 実行部
# ============================================================
if __name__ == "__main__":

    INPUT_CSV = PROJECT_DIR / "scripts_generate_crit/crit/crit_split20.csv"
    OUTPUT_CSV = PROJECT_DIR / "scripts_eval/gpt_result/split_gpt_ref_crit20.csv"

    run_scoring(INPUT_CSV, OUTPUT_CSV)
