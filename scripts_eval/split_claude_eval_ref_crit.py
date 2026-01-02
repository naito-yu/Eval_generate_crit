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
# Claude 採点関数（評価基準 split 対応）
# ============================================================
@retry(wait=wait_random_exponential(min=1, max=30),
       stop=stop_after_attempt(10))
def claude_eval_single_criterion(pred, input_text, output_text, criterion):

    system_prompt = f"""あなたは採点者です。
回答が評価基準を満たしているかだけを厳密に判断し採点してください。

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

    payload = {
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": f"# 回答\n{pred}"}]
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

    # Claude 4.5 フォーマット対応
    try:
        content = data["output"]["message"]["content"][0]["text"]
    except:
        content = data["content"][0]["text"]

    try:
        score_str, reason = content.split("\n", 1)
        score = int(score_str.strip())
    except:
        score = None
        reason = content

    return score, reason



# ============================================================
# ★ メイン処理（評価基準 split → 合算 対応版）
# ============================================================
def run_scoring(input_csv_path, output_csv_path):

    df = pd.read_csv(input_csv_path)

    # 新しい列も追加可能（基準ごとのスコア）
    for col in ["GPT5評価1", "GPT5理由1",
                "GPT5評価2", "GPT5理由2",
                "GPT5評価3", "GPT5理由3",
                "GPT5評価1_scores", "GPT5評価2_scores", "GPT5評価3_scores"]:
        df[col] = None

    for idx in tqdm(df.index, desc="Claude 採点中"):
        row = df.loc[idx]

        question = row["質問"]
        gold = row["正解例"]

        # --- 評価基準 split ---
        criteria = [c.strip() for c in str(row["評価基準"]).split("・") if c.strip()]

        # ---------------------------------------------------------
        # ★ 回答1〜3 をループ処理（スッキリ）
        # ---------------------------------------------------------
        for ai in [1, 2, 3]:

            pred = row[f"回答{ai}"]

            criterion_scores = []
            criterion_reasons = []

            # --- 基準ごとに採点 ---
            for criterion in criteria:
                score, reason = claude_eval_single_criterion(pred, question, gold, criterion)

                score = 0 if score is None else score
                criterion_scores.append(score)
                criterion_reasons.append(reason)

            # --- 合算ルール（あなた指定のまま）---
            final_score = min(1 + sum(criterion_scores), 5)

            # 保存（平均でなく合算結果）
            df.at[idx, f"GPT5評価{ai}"] = final_score
            df.at[idx, f"GPT5理由{ai}"] = str(criterion_reasons)
            df.at[idx, f"GPT5評価{ai}_scores"] = str(criterion_scores)

    # ============================================================
    # 相関・Accuracy（元コードのまま）
    # ============================================================
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
    print(f"✓ 出力ファイル: {output_csv_path}")
    print(f"✓ 相関 r = {r:.4f}")
    print(f"✓ p値 = {p:.4e}")
    print(f"✓ Accuracy = {accuracy:.4f}")


# ============================================================
# ★ 実行部
# ============================================================
if __name__ == "__main__":

    INPUT_CSV = PROJECT_DIR / "scripts_generate_crit/crit/crit_split20.csv"
    OUTPUT_CSV = PROJECT_DIR / "scripts_eval/claude_result/split_claude_ref_crit20.csv"

    run_scoring(INPUT_CSV, OUTPUT_CSV)
