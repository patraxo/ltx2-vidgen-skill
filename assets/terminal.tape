# VHS tape for the ltx2-vidgen-skill terminal-cast demo.
# Render:  vhs assets/terminal.tape   (install: brew install vhs  /  go install github.com/charmbracelet/vhs@latest)
# Produces assets/terminal.gif — a ~15s loop of the Claude Code → video flow for the README.
# This is a representative cast (it does not spend GPU); swap the Sleep lines for a real
# `uv run ... submit_video.py` call if you want a true recording.

Output assets/terminal.gif

Set Shell "bash"
Set FontSize 19
Set Width 1280
Set Height 660
Set Theme "Catppuccin Mocha"
Set TypingSpeed 55ms
Set Padding 28

Hide
Type "clear" Enter
Show

Type "# Claude Code:  “turn this photo into a video — vertical reel, gentle motion”"
Enter
Sleep 1200ms

Type "uv run --with modal python3 submit_video.py --mode i2v --image portrait.jpg --format reel \\"
Enter
Type "  --prompt 'young woman, soft smile, hair drifting, warm light, photorealistic'"
Enter
Sleep 900ms

Type "[ltx2-video] mode=i2v  768x1280  97f  audio=on  -> ltx2-fast-inference (your Modal GPU)"
Enter
Sleep 700ms
Type "[ltx2-video] warm generation ~31s ..."
Enter
Sleep 1800ms

Type "SAVED  video_out/20260608_i2v.mp4   (1.9 MB, latency 31.4s, ~2.6 cents)"
Enter
Type "PREVIEW video_out/20260608_i2v.png"
Enter
Sleep 1200ms

Type "# your GPU, your bill — no per-clip meter. ~1,000 clips on Modal's free $30/mo."
Enter
Sleep 2500ms
