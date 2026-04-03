# openclaw_gemini

Discord 上で自然会話しつつ、必要に応じて workspace 内でファイル操作やコマンド実行を行う AI エージェントです。

## 特徴

- 専用チャンネルでは自然言語で会話可能
- chat / dev_consult / agent_task を自動判定
- 小規模タスクは自動実行
- 大規模タスクは承認制
- デフォルトは Gemini
- `!provider` で OpenAI / Gemini / Anthropic に切替可能
- workspace 外操作は禁止

## 必要なもの

- Docker
- Discord Bot Token
- Gemini API Key
- 必要に応じて OpenAI / Anthropic API Key

## 環境変数

`.env.example` を `.env.openclaw` にコピーして編集してください。

## 起動

```bash
docker build -t openclaw_gemini .

docker run --rm -it \
-v $(pwd)/app:/app \
-v $(pwd)/workspace:/workspace \
--env-file .env.openclaw \
openclaw_gemini
