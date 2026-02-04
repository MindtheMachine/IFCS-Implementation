@echo off
echo Committing cleanup of temporary files...
git add -A
git commit -m "Clean up temporary batch files used for git operations"
git push origin master
echo Cleanup committed and pushed!