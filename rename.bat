@echo off
setlocal EnableDelayedExpansion
set i=1
for %%a in (*.*) do (
    if "%%~xa" neq ".bat" (
        ren "%%a" "image (!i!)%%~xa"
        set /a i+=1
    )
)