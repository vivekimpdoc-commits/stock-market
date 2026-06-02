import os

index_path = r'c:\Users\DELL\OneDrive\Desktop\ts project\stock market\stock-market\index.html'

with open(index_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Extract CSS
style_start = content.find('<style>')
style_end = content.find('</style>')
css_content = content[style_start+7:style_end]

# Extract JS
script_start = content.find('<script>')
script_end = content.rfind('</script>')
js_content = content[script_start+8:script_end]

# Create directories
os.makedirs(r'c:\Users\DELL\OneDrive\Desktop\ts project\stock market\stock-market\frontend\css', exist_ok=True)
os.makedirs(r'c:\Users\DELL\OneDrive\Desktop\ts project\stock market\stock-market\frontend\js', exist_ok=True)

# Write CSS and JS
with open(r'c:\Users\DELL\OneDrive\Desktop\ts project\stock market\stock-market\frontend\css\style.css', 'w', encoding='utf-8') as f:
    f.write(css_content.strip())

with open(r'c:\Users\DELL\OneDrive\Desktop\ts project\stock market\stock-market\frontend\js\main.js', 'w', encoding='utf-8') as f:
    f.write(js_content.strip())

# Rewrite index.html
new_html = content[:style_start] + '<link rel="stylesheet" href="css/style.css">\n' + content[style_end+8:script_start] + '<script src="js/main.js"></script>\n' + content[script_end+9:]

with open(r'c:\Users\DELL\OneDrive\Desktop\ts project\stock market\stock-market\frontend\index.html', 'w', encoding='utf-8') as f:
    f.write(new_html)

# Clean up original index.html and gateway.html
os.remove(index_path)
os.remove(r'c:\Users\DELL\OneDrive\Desktop\ts project\stock market\stock-market\gateway.html')

print("Extraction complete!")
