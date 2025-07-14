# How to Keep Your Cursor Chat History

## The Problem
- Your chat history is stored in the container at `/workspace/.cursor/`
- When you open an empty local folder, you lose access to this chat history
- Each workspace in Cursor has its own separate chat history

## Solution 1: Stay in the Container (Recommended)
Don't open a local folder! Instead:

1. **In Cursor, use Command Palette** (`Cmd+Shift+P`)
2. Type: "Remote-Containers: Attach to Running Container"
3. Or look for "Reopen in Container" option
4. This keeps you connected to `/workspace` where your chat history lives

## Solution 2: If Already in Wrong Folder
If you've already opened an empty local folder:

1. **Close the current window**
2. **In Cursor**: 
   - File → Recent → Look for `/workspace`
   - Or File → Open Recent → Find the container workspace
3. **If that doesn't work**:
   - Look at Cursor's welcome screen
   - Check "Recent" section for container workspaces

## Solution 3: Force Reconnect to Container
1. In Cursor, open Command Palette (`Cmd+Shift+P`)
2. Search for one of these:
   - "Dev Containers: Open Folder in Container"
   - "Remote-Containers: Open Folder in Container"
   - "Reopen in Container"

## How to Identify the Right Workspace
The correct workspace will show:
- Path: `/workspace` (not a Mac path like `/Users/...`)
- Bottom status bar shows container/remote indicator
- Terminal shows Linux when you run `uname`
- Chat history shows our conversations

## Preventing This Issue
1. **Bookmark the container workspace** in Cursor
2. **Always use "Open Recent"** instead of "Open Folder"
3. **Check the status bar** - it should show container connection

## If All Else Fails
The chat history is stored in the container. To preserve it:

1. Stay in the current container session
2. Don't create new workspaces
3. Use the terminal even if file explorer is empty

## Important Note
Your chat history is tied to:
- The specific workspace path (`/workspace`)
- The container instance
- The `.cursor` directory in that workspace

Opening a different folder (even with same files) creates a NEW workspace with NO chat history.