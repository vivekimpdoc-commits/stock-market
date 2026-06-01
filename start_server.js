const fs = require('fs');
const { spawn, execSync } = require('child_process');
const path = require('path');

// Test if python path has required modules
function testPython(pythonPath) {
    try {
        execSync(`"${pythonPath}" -c "import fastapi, pandas, uvicorn"`, { stdio: 'ignore' });
        return true;
    } catch (e) {
        return false;
    }
}

// Generate all candidate python executable paths
function getCandidates() {
    const userProfile = process.env.USERPROFILE || 'C:\\Users\\DELL';
    const candidates = [];

    // 0. Search in virtual environments in workspace and parent directories
    const workspaceDir = __dirname;
    const parent1 = path.resolve(workspaceDir, '..');
    const parent2 = path.resolve(workspaceDir, '../..');
    const parent3 = path.resolve(workspaceDir, '../../..');
    
    const venvs = ['.venv', 'venv', 'env', 'conda-env', 'stock-market-env', 'stock-env', 'pyenv'];
    for (const pDir of [workspaceDir, parent1, parent2, parent3]) {
        for (const venv of venvs) {
            candidates.push(path.join(pDir, venv, 'Scripts', 'python.exe'));
            candidates.push(path.join(pDir, venv, 'bin', 'python.exe'));
            candidates.push(path.join(pDir, venv, 'python.exe'));
        }
    }

    // 1. Search in PATH (excluding WindowsApps)
    const pathEnv = process.env.PATH || '';
    const paths = pathEnv.split(path.delimiter);
    for (const p of paths) {
        if (!p || p.toLowerCase().includes('microsoft\\windowsapps')) continue;
        try {
            const files = fs.readdirSync(p);
            for (const f of files) {
                if (f.toLowerCase() === 'python.exe' || f.toLowerCase() === 'python3.exe') {
                    candidates.push(path.join(p, f));
                }
            }
        } catch (e) {}
    }

    // 2. Common paths in User Profile AppData Local Programs
    for (let v = 7; v <= 15; v++) {
        candidates.push(path.join(userProfile, 'AppData', 'Local', 'Programs', 'Python', `Python3${v}`, 'python.exe'));
        candidates.push(path.join(userProfile, 'AppData', 'Local', 'Programs', 'Python', `Python${v}`, 'python.exe'));
    }

    // 3. Program Files
    for (let v = 7; v <= 15; v++) {
        candidates.push(`C:\\Program Files\\Python3${v}\\python.exe`);
        candidates.push(`C:\\Program Files\\Python${v}\\python.exe`);
        candidates.push(`C:\\Program Files (x86)\\Python3${v}\\python.exe`);
        candidates.push(`C:\\Program Files (x86)\\Python${v}\\python.exe`);
    }

    // 4. Anaconda, Miniconda, Thonny, etc.
    const bases = [
        path.join(userProfile, 'anaconda3'),
        path.join(userProfile, 'miniconda3'),
        'C:\\anaconda3',
        'C:\\miniconda3',
        path.join(userProfile, 'AppData', 'Local', 'Programs', 'Thonny'),
        path.join(userProfile, 'AppData', 'Local', 'Python', 'bin')
    ];
    for (const b of bases) {
        candidates.push(path.join(b, 'python.exe'));
    }

    // 5. C:\ python folders
    for (let v = 7; v <= 15; v++) {
        candidates.push(`C:\\Python3${v}\\python.exe`);
        candidates.push(`C:\\Python${v}\\python.exe`);
    }

    // Deduplicate and filter by exists
    const uniqueCandidates = [...new Set(candidates)];
    return uniqueCandidates.filter(p => {
        try {
            return fs.existsSync(p);
        } catch (e) {
            return false;
        }
    });
}

console.log('[*] Locating Python environment...');
const existingPythons = getCandidates();

let pythonPath = null;
let foundWorking = false;

console.log(`[*] Found ${existingPythons.length} Python executable(s) on your system.`);

// Check for one that has dependencies installed
for (const p of existingPythons) {
    console.log(`Testing: ${p}...`);
    if (testPython(p)) {
        pythonPath = p;
        foundWorking = true;
        console.log(`[✓] Selected working Python environment with required dependencies: ${p}`);
        break;
    }
}

// Fallback to first existing one
if (!pythonPath && existingPythons.length > 0) {
    pythonPath = existingPythons[0];
    console.log(`[!] No Python environment with all required modules (fastapi, pandas, uvicorn) found. Falling back to first found: ${pythonPath}`);
}

if (!pythonPath) {
    console.log('[!] python.exe not found on the system. Using default "python" command.');
    pythonPath = 'python';
}

console.log('[*] Starting FastAPI Uvicorn Server...');
const uvicorn = spawn(pythonPath, ['-m', 'uvicorn', 'backend.app:app', '--host', '127.0.0.1', '--port', '8000'], {
    stdio: 'inherit',
    shell: true
});

uvicorn.on('error', (err) => {
    console.error('[X] Failed to start Uvicorn process:', err);
});

uvicorn.on('close', (code) => {
    console.log(`[*] Server process exited with code ${code}`);
});
