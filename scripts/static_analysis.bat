@echo off
REM == Cyclomatic complexity ==
radon cc -s -a backends scripts

REM == Import Linter (optional rules) ==
IF EXIST .importlinter (
  import-linter
) ELSE (
  echo (no .importlinter)
)
