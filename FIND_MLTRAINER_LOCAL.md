# 📍 Finding mlTrainer on Your Local System

## Quick Methods

### Method 1: Use the Python Script
Download and run the `find_mltrainer_local.py` script I created:

```bash
python3 find_mltrainer_local.py
```

This will:
- Search common project directories
- Check git repositories
- Find Cursor configuration
- Give you exact commands to open in Cursor

### Method 2: Platform-Specific Commands

#### 🍎 **macOS**
```bash
# Search entire home directory
find ~ -name "mlTrainer" -type d 2>/dev/null

# Search common locations only (faster)
find ~/Documents ~/Projects ~/Developer ~/Desktop ~/Code -name "mlTrainer" -type d 2>/dev/null

# If you use iCloud Drive
find ~/Library/Mobile\ Documents/com~apple~CloudDocs -name "mlTrainer" -type d 2>/dev/null
```

#### 🐧 **Linux**
```bash
# Search home directory
find ~ -name "mlTrainer" -type d 2>/dev/null

# Search with locate (faster if available)
locate -i mltrainer | grep -E "/mlTrainer$"

# Update locate database first if needed
sudo updatedb
```

#### 🪟 **Windows**

**PowerShell:**
```powershell
# Search common locations
Get-ChildItem -Path "$env:USERPROFILE\Documents", "$env:USERPROFILE\Desktop", "$env:USERPROFILE\Projects" -Filter "mlTrainer" -Directory -Recurse -ErrorAction SilentlyContinue

# Search entire C: drive (slower)
Get-ChildItem -Path C:\ -Filter "mlTrainer" -Directory -Recurse -ErrorAction SilentlyContinue | Select-Object FullName
```

**Command Prompt:**
```cmd
dir C:\mlTrainer /s /b
dir %USERPROFILE%\Documents\mlTrainer /s /b
```

### Method 3: Check Git History

```bash
# Show recent git clones
history | grep "git clone.*mlTrainer"

# Check git config for all repositories
find ~ -name ".git" -type d -exec dirname {} \; | grep mlTrainer
```

### Method 4: Check Cursor Recent Projects

Cursor stores recent projects. Check:

**macOS:**
- `~/Library/Application Support/Cursor/User/workspaceStorage/`
- Recent projects in Cursor: File → Open Recent

**Windows:**
- `%APPDATA%\Cursor\User\workspaceStorage\`
- Recent projects in Cursor: File → Open Recent

**Linux:**
- `~/.config/Cursor/User/workspaceStorage/`

## Opening in Native Cursor App

Once you find the directory, you have several options:

### Option 1: Command Line
```bash
cursor /path/to/mlTrainer
```

### Option 2: Cursor GUI
1. Open Cursor
2. File → Open Folder (or `Cmd+O` / `Ctrl+O`)
3. Navigate to the mlTrainer directory
4. Click "Open"

### Option 3: Drag and Drop
- Drag the mlTrainer folder onto the Cursor app icon

### Option 4: From Terminal in the Directory
```bash
cd /path/to/mlTrainer
cursor .
```

## If Not Found - Clone It

If you can't find mlTrainer, you might need to clone it:

```bash
# Choose a location
cd ~/Projects  # or wherever you keep projects

# Clone the repository
git clone https://github.com/hgw734/mlTrainer.git

# Open in Cursor
cursor mlTrainer
```

## Verify It's the Right Project

The mlTrainer directory should contain:
- `config/` directory with `models_config.py`
- `custom/` directory with model implementations
- `requirements.txt`
- `.git/` directory
- `modal_app.py` (for Modal deployment)

## Troubleshooting

1. **Permission Denied**: Some directories might be protected. The search will skip them.

2. **Not Found**: The project might be:
   - In a cloud drive (OneDrive, iCloud, Google Drive)
   - On an external drive
   - In a Docker container or WSL
   - Not cloned yet

3. **Multiple Copies**: You might have multiple clones. Check the git status and last commit to identify the most recent.

## Next Steps

Once you've located and opened mlTrainer in your native Cursor:

1. Pull the latest changes:
   ```bash
   git pull origin main
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Continue with Modal setup

Let me know once you've found the directory and opened it in Cursor!