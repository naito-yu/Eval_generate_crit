import os
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm
from scipy.stats import pearsonr
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parents[1]

# ============================================================
# Google Gemini 認証設定（★GPT→Gemini）
# ============================================================
import google.generativeai as genai

genai.configure()

MODEL = genai.GenerativeModel("models/gemini-2.5-pro")


# ============================================================
# Gemini 応答テキスト抽出（安全版）
# ============================================================
def safe_extract_text(response):
    try:
        return response.text.strip()
    except Exception:
        pass
    try:
        parts = []
        for c in response.candidates:
            if hasattr(c, "content") and hasattr(c.content, "parts"):
                for p in c.content.parts:
                    if hasattr(p, "text"):
                        parts.append(p.text)
        if parts:
            return "".join(parts).strip()
    except:
        pass
    return ""


# ============================================================
# ★ 正解例なし採点関数（1評価基準ごとに使用）
# ============================================================
@retry(wait=wait_random_exponential(min=1, max=30),
       stop=stop_after_attempt(10))
def gpt5_eval_single_criterion(pred, input_text, criterion):

    system_message = f"""あなたは採点者です。
回答が評価基準を満たしているかだけを厳密に判断し採点してください。

# 問題
{input_text}

# 評価基準
{criterion}

# 出力形式
1行目: 数字のみ
2行以降: 判断理由
"""

    user_prompt = f"# 回答\n{pred}"

    response = MODEL.generate_content(
        [system_message, user_prompt],
        generation_config=genai.types.GenerationConfig(
            temperature=0.0,
            max_output_tokens=3000
        ),
    )

    text = safe_extract_text(response)
    if not text:
        return None, "Gemini の応答が空でした"

    try:
        score_str, reason = text.split("\n", 1)
        score = int(score_str.strip())
    except:
        score = None
        reason = text.strip()

    return score, reason



# ============================================================
# ★ メイン処理（評価基準 split → 個別採点 → 合算）
# ============================================================
def run_scoring(input_csv_path, output_csv_path):

    df = pd.read_csv(input_csv_path)

    # 出力列の初期化（Claude 版と互換）
    for col in ["GPT5評価1", "GPT5理由1",
                "GPT5評価2", "GPT5理由2",
                "GPT5評価3", "GPT5理由3",
                "GPT5評価1_scores", "GPT5評価2_scores", "GPT5評価3_scores"]:
        df[col] = None

    # ===================== 採点本体 =====================
    for idx in tqdm(df.index, desc="Gemini 採点（split対応）"):
        row = df.loc[idx]

        question = row["質問"]
        pred_answers = [row["回答1"], row["回答2"], row["回答3"]]

        # ✔ 評価基準 split
        criteria_list = [
            c.strip() for c in str(row["評価基準"]).split("・") if c.strip()
        ]

        for ai, pred in enumerate(pred_answers, start=1):

            criterion_scores = []
            criterion_reasons = []

            # --- 各基準ごとに Gemini 採点 ---
            for criterion in criteria_list:
                score, reason = gpt5_eval_single_criterion(pred, question, criterion)
                score = 0 if score is None else score

                criterion_scores.append(score)
                criterion_reasons.append(reason)

            # ✔ 合算スコア（あなたのルール）
            final_score = min(1 + sum(criterion_scores), 5)

            # --- 保存 ---
            df.at[idx, f"GPT5評価{ai}"] = final_score
            df.at[idx, f"GPT5理由{ai}"] = str(criterion_reasons)
            df.at[idx, f"GPT5評価{ai}_scores"] = str(criterion_scores)

    # ===================== 相関 & Accuracy（元コードのまま） =====================
    gpt_all = pd.concat([
        df["GPT5評価1"], df["GPT5評価2"], df["GPT5評価3"]
    ], ignore_index=True)

    human_all = pd.concat([
        df["人手評価1"], df["人手評価2"], df["人手評価3"]
    ], ignore_index=True)

    gpt_all = pd.to_numeric(gpt_all, errors="coerce")
    human_all = pd.to_numeric(human_all, errors="coerce")

    valid = pd.concat([gpt_all, human_all], axis=1).dropna()
    gpt_valid = valid.iloc[:, 0].astype(float).values
    human_valid = valid.iloc[:, 1].astype(float).values

    if len(gpt_valid) > 1:
        r, p = pearsonr(gpt_valid, human_valid)
    else:
        r, p = float("nan"), float("nan")

    accuracy = (gpt_valid == human_valid).mean() if len(gpt_valid) > 0 else float("nan")

    df["縦結合_相関_r"] = r
    df["縦結合_p値"] = p
    df["縦結合_Accuracy"] = accuracy

    df.to_csv(output_csv_path, index=False)

    print("\n===== 保存完了 =====")
    print("出力ファイル:", output_csv_path)
    print(f"相関 r = {r:.4f}")
    print(f"p値 = {p:.4e}")
    print(f"Accuracy = {accuracy:.4f}")


# ============================================================
# ★ パス指定
# ============================================================
if __name__ == "__main__":

    INPUT_CSV = PROJECT_DIR / "scripts_generate_crit/crit/crit_split20.csv"
    OUTPUT_CSV = PROJECT_DIR / "scripts_eval/gemini_result/split_gemini_crit20.csv"

    run_scoring(INPUT_CSV, OUTPUT_CSV)
