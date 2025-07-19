#!/bin/bash
# Copy all mlTrainer3 files to workspace

echo "📁 Copying entire mlTrainer3 repository to workspace..."

# Create target directory
TARGET_DIR="/workspace/mlTrainer3_complete"
SOURCE_DIR="/tmp/mlTrainer3"

# Remove old directory if exists
rm -rf "$TARGET_DIR"

# Copy everything
cp -r "$SOURCE_DIR" "$TARGET_DIR"

echo "✅ Complete! All files copied to: $TARGET_DIR"

# Count files
echo ""
echo "📊 Statistics:"
echo "Total files: $(find $TARGET_DIR -type f | wc -l)"
echo "Python files: $(find $TARGET_DIR -name "*.py" | wc -l)"
echo "Directories: $(find $TARGET_DIR -type d | wc -l)"