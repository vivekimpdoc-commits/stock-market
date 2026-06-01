import os
import re

base_dir = r"c:\Users\DELL\OneDrive\Desktop\ts project\stock market\stock-market"

# 1. Update login.html -> gateway.html
with open(os.path.join(base_dir, "login.html"), "r", encoding="utf-8") as f:
    content = f.read()

content = content.replace("Login |", "Gateway |")
content = content.replace("auth-card", "gateway-card")
content = content.replace("auth-header", "gateway-header")
content = content.replace("authCard", "gatewayCard")
content = content.replace("authForm", "gatewayForm")
content = content.replace("login info", "sensitive information")
content = content.replace("isLoggedIn", "gatewayPassed")

with open(os.path.join(base_dir, "gateway.html"), "w", encoding="utf-8") as f:
    f.write(content)

# 2. Update index.html
with open(os.path.join(base_dir, "index.html"), "r", encoding="utf-8") as f:
    idx_content = f.read()

idx_content = idx_content.replace("login.html", "gateway.html")
idx_content = idx_content.replace("isLoggedIn", "gatewayPassed")
idx_content = idx_content.replace("localStorage.removeItem(\"isLoggedIn\")", "localStorage.removeItem(\"gatewayPassed\")")

with open(os.path.join(base_dir, "index.html"), "w", encoding="utf-8") as f:
    f.write(idx_content)

# 3. Update app.py
with open(os.path.join(base_dir, "backend", "app.py"), "r", encoding="utf-8") as f:
    app_content = f.read()

app_content = app_content.replace("login.html", "gateway.html")
app_content = app_content.replace("/login", "/gateway")
app_content = app_content.replace("login_path", "gateway_path")

with open(os.path.join(base_dir, "backend", "app.py"), "w", encoding="utf-8") as f:
    f.write(app_content)

# 4. Remove login.html
try:
    os.remove(os.path.join(base_dir, "login.html"))
except Exception as e:
    print("Could not remove login.html:", e)

print("Migration completed successfully.")
