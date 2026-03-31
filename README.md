# 8-category Classifier

## Setup

・環境作成
```bash id="2h1p4y"
python -m venv .venv
```
※ 必要なパッケージをインストールしてください

・ベースモデルを事前にダウンロードして models/modernbert-base に配置

・ログ出力先ディレクトリを作成（任意のパス）

```bash id="gk1l2n"
mkdir -p <log_dir>
```

`.sh` 内でログ出力先を指定しているため、パスを適宜変更

```bash id="n6q2tx"
#SBATCH --output=<log_dir>/job.log

#SBATCH -o <log_dir>/multi_cl_%A_%a.out
#SBATCH -e <log_dir>/multi_cl_%A_%a.err
```

---

## Usage

### 分類器の学習

```bash id="6kz9bg"
cd /lustre/tohoku/tools/8category
sbatch scripts/classifier/run_3gpp.sh
```

### 分類器の実行

```bash id="6m9hql"
sbatch classify/8classify.sh
```
