@echo off
texcount %1 -out=wordcount.log

for /f "tokens=4 delims=: " %%i in ('findstr /c:"Words in text" wordcount.log') do (
    echo Main body word count: %%i > %~n1.wc
)
