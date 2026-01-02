import os
import json
import boto3
import pandas as pd
import numpy as np
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm
from scipy.stats import pearsonr
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parents[1]


# ============================================================
# Bedrock Claude Client
# ============================================================
AWS_REGION = "us-east-1"
MODEL_ARN  = (
    "arn:aws:bedrock:us-east-1:794038215686:inference-profile/"
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
)

client = boto3.client("bedrock-runtime", region_name=AWS_REGION)



# ============================================================
# ★ Claude 採点関数（評価基準 split 対応版）
# ============================================================
@retry(wait=wait_random_exponential(min=1, max=30),
       stop=stop_after_attempt(10))
def claude_eval(pred, input_text, criterion):

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

    # Claude 4.5 出力形式（両対応）
    try:
        text = data["output"]["message"]["content"][0]["text"].strip()
    except:
        text = data["content"][0]["text"].strip()

    # --- スコア抽出 ---
    try:
        score_str, reason = text.split("\n", 1)
        score = int(score_str.strip())
    except:
        score = None
        reason = text

    return score, reason



# ============================================================
# ★ メイン処理：split + 合算 + 平均/分散 + 相関
# ============================================================
def run_scoring_with_criteria(input_csv_path, output_csv_path, n_repeats=1):

    df = pd.read_csv(input_csv_path)

    # 元コードの列 + 新しい多出力列
    for col in ["GPT5評価1", "GPT5理由1",
                "GPT5評価2", "GPT5理由2",
                "GPT5評価3", "GPT5理由3",
                "GPT5評価1_scores", "GPT5評価2_scores", "GPT5評価3_scores"]:
        df[col] = None

    # ============================================================
    # 行ごと処理
    # ============================================================
    for idx in tqdm(df.index, desc="Claude 採点（評価基準 split 対応版）"):

        question = df.at[idx, "質問"]
        criteria_raw = str(df.at[idx, "評価基準"])

        # --- 評価基準 split（"・" 区切り） ---
        criteria_list = [c.strip() for c in criteria_raw.split("・") if c.strip()]

        for ai in [1, 2, 3]:

            pred = df.at[idx, f"回答{ai}"]
            all_repeat_scores = []
            all_repeat_reasons = []

            for _ in range(n_repeats):

                criterion_scores = []
                criterion_reasons = []

                for criterion in criteria_list:
                    score, reason = claude_eval(pred, question, criterion)
                    score = 0 if score is None else score
                    criterion_scores.append(score)
                    criterion_reasons.append(reason)

                # --- 合算ルール（あなた指定のまま）---
                final_score = min(1 + sum(criterion_scores), 5)

                all_repeat_scores.append(final_score)
                all_repeat_reasons.append(criterion_reasons)

            # --- 保存 ---
            df.at[idx, f"GPT5評価{ai}"] = np.mean(all_repeat_scores)
            df.at[idx, f"GPT5理由{ai}"] = str(all_repeat_reasons)
            df.at[idx, f"GPT5評価{ai}_scores"] = str(all_repeat_scores)

    # ============================================================
    # 相関（縦結合）
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
    gpt_valid = valid.iloc[:, 0].values
    human_valid = valid.iloc[:, 1].values

    if len(gpt_valid) > 1:
        r, p = pearsonr(gpt_valid, human_valid)
        accuracy = (gpt_valid == human_valid).mean()
    else:
        r = p = accuracy = None

    df["縦結合_相関_r"] = r
    df["縦結合_p値"] = p
    df["縦結合_Accuracy"] = accuracy

    df.to_csv(output_csv_path, index=False)

    print("\n===== 保存完了 =====")
    print("出力ファイル:", output_csv_path)
    print(f"相関 r = {r}")
    print(f"p値 = {p}")
    print(f"Accuracy = {accuracy}")



# ============================================================
# 実行
# ============================================================
if __name__ == "__main__":
    INPUT = PROJECT_DIR / "scripts_generate_crit/crit/crit_split20.csv"
    OUTPUT = PROJECT_DIR / "scripts_eval/claude_result/split_claude_crit20.csv"

    run_scoring_with_criteria(INPUT, OUTPUT, n_repeats=1)
