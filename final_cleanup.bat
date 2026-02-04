@echo off
git add -A
git commit -m "Final cleanup: Remove temporary batch file"
git push origin master
del "%~f0"