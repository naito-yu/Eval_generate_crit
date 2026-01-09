## API Settings


```bash
# AWS Bedrock (Claude)
export AWS_BEARER_TOKEN_BEDROCK="your_bedrock_token"

# OpenAI (GPT)
export OPENAI_API_KEY="your_openai_api_key"

# Google Gemini
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service_account_key.json"
'''

## Code

```bash
### Eval

- **base**  
  基準を使用しない評価のベースコード

- **model_eval**  
  基準を使用する評価のベースコード
  必要に応じて使用する基準を差し替える

- **minus**  
  プロンプトに減点項目を追加したコード

- **split**  
  評価基準を split して個別に使用するコード
'''

