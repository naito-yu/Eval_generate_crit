import os
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm
from scipy.stats import pearsonr
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parents[1]

# ============================================================
# Google 認証設定（★GPT→Geminiへ変更）
# ============================================================
import google.generativeai as genai

genai.configure()

MODEL = genai.GenerativeModel("models/gemini-2.5-pro")


# ============================================================
# Gemini 応答から安全にテキスト抽出
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
# ★ Gemini 採点関数（元の gpt5_eval と同じ入出力に統一）
# ============================================================
@retry(wait=wait_random_exponential(min=1, max=30),
       stop=stop_after_attempt(10))
def gpt5_eval(pred, input_text):
    system_message = f"""あなたは採点者です。
回答を1,2,3,4,5の5段階で採点し、数字と採点理由を出力してください。

# 問題
{input_text}

# 出力形式
1行目: 数字のみ
2行目以降: 採点理由
"""

    user_prompt = f"# 回答\n{pred}"

    response = MODEL.generate_content(
        [system_message, user_prompt],
        generation_config=genai.types.GenerationConfig(
            temperature=0.0,
            max_output_tokens=3000,
        ),
    )

    text = safe_extract_text(response)

    # ★ 空応答対策
    if not text:
        return None, "Gemini の応答が空でした"

    # ★ 1行目: 数字, 2行目以降: 理由
    try:
        score_str, reason = text.split("\n", 1)
        score = int(score_str.strip())
    except:
        score = None
        reason = text.strip()

    return score, reason


# ============================================================
# ★ メイン処理（元コードのまま）
# ============================================================
def run_scoring(input_csv_path, output_csv_path):

    df = pd.read_csv(input_csv_path)

    for col in ["GPT5評価1", "GPT5理由1",
                "GPT5評価2", "GPT5理由2",
                "GPT5評価3", "GPT5理由3"]:
        df[col] = None

    for idx in tqdm(df.index, desc="Gemini 採点中"):
        row = df.loc[idx]

        score1, reason1 = gpt5_eval(row["回答1"], row["質問"])
        df.at[idx, "GPT5評価1"] = score1
        df.at[idx, "GPT5理由1"] = reason1

        score2, reason2 = gpt5_eval(row["回答2"], row["質問"])
        df.at[idx, "GPT5評価2"] = score2
        df.at[idx, "GPT5理由2"] = reason2

        score3, reason3 = gpt5_eval(row["回答3"], row["質問"])
        df.at[idx, "GPT5評価3"] = score3
        df.at[idx, "GPT5理由3"] = reason3

    gpt_all = pd.concat([df["GPT5評価1"], df["GPT5評価2"], df["GPT5評価3"]], axis=0, ignore_index=True)
    human_all = pd.concat([df["人手評価1"], df["人手評価2"], df["人手評価3"]], axis=0, ignore_index=True)

    gpt_all = pd.to_numeric(gpt_all, errors="coerce")
    human_all = pd.to_numeric(human_all, errors="coerce")

    valid = pd.concat([gpt_all, human_all], axis=1).dropna()
    gpt_valid = valid.iloc[:, 0].astype(float).values
    human_valid = valid.iloc[:, 1].astype(float).values

    if len(gpt_valid) > 1:
        r, p = pearsonr(gpt_valid, human_valid)
    else:
        r, p = float("nan"), float("nan")

    accuracy = (gpt_valid == human_valid).sum() / len(gpt_valid) if len(gpt_valid) > 0 else float("nan")

    df["縦結合_相関_r"] = r
    df["縦結合_p値"] = p
    df["縦結合_Accuracy"] = accuracy

    df.to_csv(output_csv_path, index=False)

    print("\n===== 保存完了 =====")
    print(f"✓ 出力ファイル: {output_csv_path}")
    print(f"✓ 相関 r = {r:.4f}")
    print(f"✓ p値 = {p:.4e}")
    print(f"✓ Accuracy = {accuracy:.4f}")


if __name__ == "__main__":

    INPUT_CSV = PROJECT_DIR / "scripts_generate_crit/crit/crit_base20.csv"
    OUTPUT_CSV = PROJECT_DIR / "scripts_eval/gemini_result/base_gemini_eval20.csv"

    run_scoring(INPUT_CSV, OUTPUT_CSV)
