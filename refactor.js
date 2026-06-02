const fs = require('fs');
const path = require('path');

const indexPath = path.join(__dirname, 'index.html');
let content = fs.readFileSync(indexPath, 'utf-8');

const styleStart = content.indexOf('<style>');
const styleEnd = content.indexOf('</style>');
const cssContent = content.substring(styleStart + 7, styleEnd);

const scriptStart = content.indexOf('<script>');
const scriptEnd = content.lastIndexOf('</script>');
// Wait, the first script tag is <script src="...">. We need the second <script> tag which contains the code.
const codeScriptStart = content.indexOf('<script>', styleEnd);
const codeScriptEnd = content.lastIndexOf('</script>');
const jsContent = content.substring(codeScriptStart + 8, codeScriptEnd);

fs.mkdirSync(path.join(__dirname, 'frontend', 'css'), { recursive: true });
fs.mkdirSync(path.join(__dirname, 'frontend', 'js'), { recursive: true });

fs.writeFileSync(path.join(__dirname, 'frontend', 'css', 'style.css'), cssContent.trim());
fs.writeFileSync(path.join(__dirname, 'frontend', 'js', 'main.js'), jsContent.trim());

const newHtml = content.substring(0, styleStart) + '<link rel="stylesheet" href="css/style.css">\n' + 
                content.substring(styleEnd + 8, codeScriptStart) + '<script src="js/main.js"></script>\n' + 
                content.substring(codeScriptEnd + 9);

fs.writeFileSync(path.join(__dirname, 'frontend', 'index.html'), newHtml);

try { fs.unlinkSync(indexPath); } catch(e) {}
try { fs.unlinkSync(path.join(__dirname, 'gateway.html')); } catch(e) {}

console.log('Extraction complete');
