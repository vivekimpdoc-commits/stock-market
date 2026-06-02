import os
import shutil
import re

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# 1. Move root HTML files to frontend folder
index_src = os.path.join(BASE_DIR, "index.html")
if os.path.exists(index_src):
    shutil.move(index_src, os.path.join(FRONTEND_DIR, "index.html"))
    print("Moved index.html to frontend folder.")
    
# We already extracted gateway.html's CSS and JS into frontend, so the original gateway.html in root can be deleted
gateway_src = os.path.join(BASE_DIR, "gateway.html")
if os.path.exists(gateway_src):
    os.remove(gateway_src)
    print("Removed duplicate gateway.html from root.")

# 2. Reorganize Backend code
api_dir = os.path.join(BACKEND_DIR, "api")
services_dir = os.path.join(BACKEND_DIR, "services")
ml_models_dir = os.path.join(BACKEND_DIR, "ml_models")
core_dir = os.path.join(BACKEND_DIR, "core")
pipeline_dir = os.path.join(BACKEND_DIR, "pipeline")

for d in [api_dir, services_dir, ml_models_dir, core_dir, pipeline_dir]:
    os.makedirs(d, exist_ok=True)
    # Add __init__.py so Python treats them as packages
    open(os.path.join(d, "__init__.py"), 'a').close()

file_mappings = {
    "app.py": "api",
    "fetch_prices.py": "services",
    "fetch_fundamentals.py": "services",
    "fetch_sentiment.py": "services",
    "indicators.py": "core",
    "portfolio_opt.py": "core",
    "model_finbert.py": "ml_models",
    "model_lstm.py": "ml_models",
    "model_xgboost.py": "ml_models",
    "data_prep.py": "pipeline",
    "main.py": "pipeline",
    "run_pipeline.py": "pipeline",
    "backtest.py": "pipeline",
    "verify_pipeline.py": "pipeline"
}

print("Moving backend files to logical subfolders...")
for file_name, folder_name in file_mappings.items():
    src = os.path.join(BACKEND_DIR, file_name)
    dst = os.path.join(BACKEND_DIR, folder_name, file_name)
    if os.path.exists(src):
        shutil.move(src, dst)
        print(f"Moved {file_name} -> {folder_name}/{file_name}")

# 3. Update imports in app.py
app_py_path = os.path.join(api_dir, "app.py")
if os.path.exists(app_py_path):
    with open(app_py_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Update sys path append so it can find sibling directories
    content = content.replace("sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))", 
                              "sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))")
    
    # Update local imports
    content = content.replace("from fetch_prices", "from services.fetch_prices")
    content = content.replace("from fetch_fundamentals", "from services.fetch_fundamentals")
    content = content.replace("from fetch_sentiment", "from services.fetch_sentiment")
    content = content.replace("from indicators", "from core.indicators")
    content = content.replace("from portfolio_opt", "from core.portfolio_opt")
    
    with open(app_py_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Updated imports in app.py")

print("\nReorganization complete! Your codebase is now much cleaner.")
print("To run your server now, run: uvicorn backend.api.app:app --reload")
