import os
import json
import boto3
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm
from scipy.stats import pearsonr
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parents[1]


# ============================================================
# Bedrock Claude クライアント設定
# ============================================================
AWS_REGION = "us-east-1"
MODEL_ARN  = (
    "arn:aws:bedrock:us-east-1:794038215686:inference-profile/"
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
)

client = boto3.client("bedrock-runtime", region_name=AWS_REGION)


# ============================================================
# ★ Claude 採点関数（正解例なし）
# ============================================================
@retry(wait=wait_random_exponential(min=1, max=30),
       stop=stop_after_attempt(10))
def claude_eval_no_gold(pred, input_text, criterion):

    system_prompt = f"""あなたは採点者です。
回答が評価基準を満たしているかを厳密に判断して採点してください。
回答を1,2,3,4,5の5段階で採点し、数字と採点理由を出力してください。

# 問題
{input_text}

# 評価基準
{criterion}

# 出力形式
1行目: 数字のみ
2行以降: 判断理由
"""

    user_prompt = f"# 回答\n{pred}"

    payload = {
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}]
            }
        ],
        "max_tokens": 3000,
        "temperature": 0,
        "anthropic_version": "bedrock-2023-05-31"
    }

    response = client.invoke_model(
        modelId=MODEL_ARN,
        body=json.dumps(payload)
    )

    raw = response["body"].read().decode()
    data = json.loads(raw)
    content = data["content"][0]["text"]

    try:
        score_str, reason = content.split("\n", 1)
        score = int(score_str.strip())
    except:
        score = None
        reason = content

    return score, reason


# ============================================================
# ★ メイン処理（元コードそのまま）
# ============================================================
def run_scoring(input_csv_path, output_csv_path):

    df = pd.read_csv(input_csv_path)

    for col in ["GPT5評価1", "GPT5理由1",
                "GPT5評価2", "GPT5理由2",
                "GPT5評価3", "GPT5理由3"]:
        df[col] = None

    for idx in tqdm(df.index, desc="Claude 採点（正解例なし）"):
        row = df.loc[idx]

        s1, r1 = claude_eval_no_gold(row["回答1"], row["質問"], row["評価基準"])
        df.at[idx, "GPT5評価1"] = s1
        df.at[idx, "GPT5理由1"] = r1

        s2, r2 = claude_eval_no_gold(row["回答2"], row["質問"], row["評価基準"])
        df.at[idx, "GPT5評価2"] = s2
        df.at[idx, "GPT5理由2"] = r2

        s3, r3 = claude_eval_no_gold(row["回答3"], row["質問"], row["評価基準"])
        df.at[idx, "GPT5評価3"] = s3
        df.at[idx, "GPT5理由3"] = r3

    # ---------------- 相関 & Accuracy ----------------
    gpt_all = pd.concat([df["GPT5評価1"], df["GPT5評価2"], df["GPT5評価3"]],
                        axis=0, ignore_index=True)
    human_all = pd.concat([df["人手評価1"], df["人手評価2"], df["人手評価3"]],
                          axis=0, ignore_index=True)

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
    print("出力ファイル:", output_csv_path)
    print(f"相関 r = {r:.4f}")
    print(f"p値 = {p:.4e}")
    print(f"Accuracy = {accuracy:.4f}")


# ============================================================
# ★ パス指定（元のまま）
# ============================================================
if __name__ == "__main__":

    INPUT_CSV = PROJECT_DIR / "scripts_generate_crit/crit/crit_base20.csv"
    OUTPUT_CSV = PROJECT_DIR / "scripts_eval/claude_result/claude_crit_point20.csv"

    run_scoring(INPUT_CSV, OUTPUT_CSV)
