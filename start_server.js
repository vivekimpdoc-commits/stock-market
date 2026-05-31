const fs = require('fs');
const { spawn } = require('child_process');
const path = require('path');

// Recursive search for python.exe
function findPythonRecursive(dir) {
    if (!fs.existsSync(dir)) return null;
    try {
        const files = fs.readdirSync(dir);
        for (const file of files) {
            const fullPath = path.join(dir, file);
            try {
                const stat = fs.statSync(fullPath);
                if (stat.isDirectory()) {
                    // Skip folders like Scripts, Lib, tcl, etc. to search faster
                    if (['Scripts', 'Lib', 'libs', 'tcl', 'share', 'include'].includes(file)) {
                        continue;
                    }
                    const found = findPythonRecursive(fullPath);
                    if (found) return found;
                } else if (file.toLowerCase() === 'python.exe') {
                    return fullPath;
                }
            } catch (e) {
                // Ignore errors for individual files
            }
        }
    } catch (e) {
        // Ignore read dir errors
    }
    return null;
}

let pythonPath = 'python'; // Default fallback

// List of base directories to search in Windows
const searchBases = [
    path.join('C:', 'Users', 'DELL', 'AppData', 'Local', 'Programs', 'Python'),
    path.join('C:', 'Users', 'DELL', 'anaconda3'),
    path.join('C:', 'Users', 'DELL', 'miniconda3'),
    path.join('C:', 'Users', 'DELL', 'AppData', 'Local', 'Programs', 'Thonny'),
    'C:\\'
];

console.log('[*] Searching for python.exe on your Windows system...');

for (const base of searchBases) {
    if (fs.existsSync(base)) {
        console.log(`Searching in: ${base}...`);
        
        // Special case for searching C:\ to avoid full system scanning (which takes too long)
        if (base === 'C:\\') {
            try {
                const dirs = fs.readdirSync(base);
                for (const d of dirs) {
                    if (d.toLowerCase().startsWith('python')) {
                        const fullD = path.join(base, d);
                        const p = path.join(fullD, 'python.exe');
                        if (fs.existsSync(p)) {
                            pythonPath = p;
                            break;
                        }
                    }
                }
            } catch (e) {}
            if (pythonPath !== 'python') break;
            continue;
        }
        
        const found = findPythonRecursive(base);
        if (found) {
            pythonPath = found;
            break;
        }
    }
}

if (pythonPath !== 'python') {
    console.log(`[✓] Located Python interpreter at: ${pythonPath}`);
} else {
    console.log('[!] python.exe not found in standard directories. Using default "python" command.');
}

console.log('[*] Starting FastAPI Uvicorn Server...');
const uvicorn = spawn(pythonPath, ['-m', 'uvicorn', 'app:app', '--host', '127.0.0.1', '--port', '8000'], {
    stdio: 'inherit',
    shell: true
});

uvicorn.on('error', (err) => {
    console.error('[X] Failed to start Uvicorn process:', err);
});

uvicorn.on('close', (code) => {
    console.log(`[*] Server process exited with code ${code}`);
});
