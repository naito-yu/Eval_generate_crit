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
# GPT 採点関数
# ============================================================
def gpt5_eval(pred, input_text, output_text, criterion):
    system_message = f"""あなたは採点者です。
回答が評価基準を満たしているかを厳密に判断して採点してください。
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
            {"role": "user", "content": user_prompt},
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
# ★ メイン処理（CSV パスを後ろで指定）
# ============================================================
def run_scoring(input_csv_path, output_csv_path):

    # ---------- CSV 読み込み ----------
    df = pd.read_csv(input_csv_path)

    # GPT 出力カラム初期化
    for col in ["GPT5評価1", "GPT5理由1",
                "GPT5評価2", "GPT5理由2",
                "GPT5評価3", "GPT5理由3"]:
        df[col] = None

    # ---------- GPT 採点 ----------
    for idx in tqdm(df.index, desc="GPT-5 採点中"):
        row = df.loc[idx]

        score1, reason1 = gpt5_eval(row["回答1"], row["質問"], row["正解例"], row["評価基準"])
        df.at[idx, "GPT5評価1"] = score1
        df.at[idx, "GPT5理由1"] = reason1

        score2, reason2 = gpt5_eval(row["回答2"], row["質問"], row["正解例"], row["評価基準"])
        df.at[idx, "GPT5評価2"] = score2
        df.at[idx, "GPT5理由2"] = reason2

        score3, reason3 = gpt5_eval(row["回答3"], row["質問"], row["正解例"], row["評価基準"])
        df.at[idx, "GPT5評価3"] = score3
        df.at[idx, "GPT5理由3"] = reason3

    # ---------- 縦結合相関 & Accuracy ----------
    gpt_all = pd.concat([df["GPT5評価1"], df["GPT5評価2"], df["GPT5評価3"]], axis=0, ignore_index=True)
    human_all = pd.concat([df["人手評価1"], df["人手評価2"], df["人手評価3"]], axis=0, ignore_index=True)

    # 数値に強制変換（エラーは NaN に置換）
    gpt_all = pd.to_numeric(gpt_all, errors="coerce")
    human_all = pd.to_numeric(human_all, errors="coerce")

    # NaN を除去
    valid = pd.concat([gpt_all, human_all], axis=1).dropna()
    gpt_valid = valid.iloc[:, 0].astype(float).values
    human_valid = valid.iloc[:, 1].astype(float).values

    # Pearson 相関
    if len(gpt_valid) > 1:
        r, p = pearsonr(gpt_valid, human_valid)
    else:
        r, p = float("nan"), float("nan")

    # Accuracy
    accuracy = (gpt_valid == human_valid).sum() / len(gpt_valid) if len(gpt_valid) > 0 else float("nan")

    # ---------- 相関・Accuracy を列として追加 ----------
    df["縦結合_相関_r"] = r
    df["縦結合_p値"] = p
    df["縦結合_Accuracy"] = accuracy


    # ---------- CSV 保存 ----------
    df.to_csv(output_csv_path, index=False)

    print("\n===== 保存完了 =====")
    print(f"✓ 出力ファイル: {output_csv_path}")
    print(f"✓ 相関 r = {r:.4f}")
    print(f"✓ p値 = {p:.4e}")
    print(f"✓ Accuracy = {accuracy:.4f}")


# ============================================================
# ★ ここでパスを指定して実行（← ここだけ変更すればよい）
# ============================================================
if __name__ == "__main__":

    INPUT_CSV = PROJECT_DIR / "scripts_generate_crit/crit/crit_base20.csv"
    OUTPUT_CSV = PROJECT_DIR / "scripts_eval/gpt_result/gpt_ref_crit_point20.csv"

    run_scoring(INPUT_CSV, OUTPUT_CSV)
