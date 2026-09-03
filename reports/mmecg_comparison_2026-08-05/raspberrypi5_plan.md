結論：ブラウザログインは最初の1回だけです
Tailscaleはデーモン（常駐サービス）として動くので、一度認証すればPi起動のたびに自動で繋がります。以下に全工程を時系列で整理します。

【Phase 1】SDカードへの書き込み（Windows PC上）
Raspberry Pi Imager（https://www.raspberrypi.com/software/）を使います。
① Imagerを起動
② 「Raspberry Pi 5」を選択
③ OSを選択（Raspberry Pi OS 64-bit 推奨）
④ SDカードを選択
⑤ 「次へ」→「OS カスタマイズを編集する」をクリック ← ここが重要

「OS カスタマイズ」画面で以下を設定：
タブ
設定項目
設定値の例
一般
ホスト名
raspberrypi
一般
ユーザー名 / パスワード
pi / raspberrypi
一般
Wi-Fi
SSIDとパスワードを入力
一般
タイムゾーン
Asia/Tokyo
サービス
SSHを有効にする
✅ オン（パスワード認証）

設定後「保存」→「はい」で書き込み開始。これでPi起動直後からSSH接続可能な状態になります。

【Phase 2】Pi初回起動〜WSL2からSSH接続
# ① SDカードをPiに挿して電源ON
# ② 1〜2分待つ（初回起動は少し時間がかかる）

# ③ WSL2のターミナルで接続確認
#    同一ネットワーク内であれば「ホスト名.local」で繋がる
ssh pi@raspberrypi.local

# 「Are you sure...?」→ yes
# パスワードを入力
# pi@raspberrypi:~$ と表示されれば成功

# ④ 接続確認できたら一旦抜ける
exit

.localで繋がらない場合はルーターの管理画面でPiのIPアドレスを確認してください。

【Phase 3】SSHキー認証の設定（パスワード不要化）
# WSL2で実行

# ① 鍵がなければ生成（すでにある場合はスキップ）
ls ~/.ssh/id_ed25519 || ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519

# ② 公開鍵をPiに登録（この1回だけパスワードが必要）
ssh-copy-id pi@raspberrypi.local

# ③ パスワードなしで繋がるか確認
ssh pi@raspberrypi.local "echo 'SSH key auth OK'"
# → SSH key auth OK と表示されれば完了


【Phase 4】PiにTailscaleをインストール・認証（1回だけ）
# WSL2からPiに入って作業
ssh pi@raspberrypi.local

# ---- ここからPi上での作業 ----

# ① Tailscaleインストール（1コマンド）
curl -fsSL https://tailscale.com/install.sh | sh

# ② 自動起動を有効化（インストール時に設定されているが念のため）
sudo systemctl enable tailscaled

# ③ 認証（この1回だけブラウザ操作が必要）
sudo tailscale up --accept-routes

# → 以下のようなURLが表示される
# To authenticate, visit: https://login.tailscale.com/a/xxxxxxxxxx

表示されたURLをPC・スマホどのブラウザでも開いてOK（Pi上でブラウザを開く必要はありません）。WSLのTailscaleと同じアカウントでログインします。
# 認証後、PiのTailscale IPを確認（メモしておく）
tailscale ip -4
# → 100.x.x.b のように表示される

exit   # Piから抜ける


【Phase 5】WSL2のSSH設定をTailscale IPで更新
# WSL2で実行
nano ~/.ssh/config

Host raspi
    HostName 100.x.x.b       # ← Phase4で確認したTailscale IP
    User pi
    IdentityFile ~/.ssh/id_ed25519

# 動作確認
ssh raspi "echo 'Tailscale経由でSSH接続成功'"


【Phase 6】Pi再起動後の自動接続確認
# WSL2からPiを再起動
ssh raspi "sudo reboot"

# 1分待ってから再接続（これが毎回の起動後の状態）
sleep 60
ssh raspi "hostname && tailscale ip -4"
# → raspberrypi
# → 100.x.x.b
# パスワードなし・ブラウザなしで繋がればセットアップ完了


起動後の接続フロー（完成形）
Piの電源ON
   ↓ 自動
Tailscaleデーモン起動・クラウドに自動接続  ← ブラウザ不要
   ↓ 自動
WSL2から ssh raspi でいつでも接続可能
   ↓
Claude Code が ssh raspi "コマンド" で
環境構築・推論実行を自動化


よくある疑問
Q: Tailscaleの認証は期限切れになりませんか？ デフォルトでは180日で期限切れになりますが、Tailscale管理画面（admin.tailscale.com）でデバイスの「Disable key expiry」をオンにすれば無期限になります。設定後は一切のログイン操作が不要になります。
Q: WSL2側はTailscaleがすでに入っているので追加作業は不要ですか？ はい、WSL2側は既存の設定のままで問題ありません。PiをTailscaleに追加するだけです。
ここまで完了したら、次はClaude Codeに「ssh raspiで接続できるRaspberry Pi 5に推論環境を構築して」と指示するだけで自動化が動き始めます。現在WSL2で使っているフレームワーク（PyTorchなど）を教えていただければ、その指示文も一緒に作れます。

