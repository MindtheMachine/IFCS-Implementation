@echo off
echo Force adding all changes...
git add -A --force
git add taxonomy_36_results.json
git add run_all_36_cases.py
git add run_core_tests.py
git add trilogy_config.py
echo Checking what will be committed...
git status
echo Committing all implementation files...
git commit -m "IFCS Complete Implementation: All 36 taxonomy tests passing (100%% success) - Signal estimation, 3-gate pipeline, CP-2 topic gating with real LLM integration"
echo Pushing to remote...
git push origin master
echo Complete!