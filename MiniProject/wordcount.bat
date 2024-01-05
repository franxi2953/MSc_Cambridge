@echo off
texcount %1 -out=wordcount.log

for /f "tokens=4 delims=: " %%i in ('findstr /c:"Words in text" wordcount.log') do (
    echo Word Count: %%i > %~n1.wc
)
