rd ".vs" /S/Q
del "*.suo" /S/Q/F/A
del "*.csproj.user" /S/Q/F/A

for /f "usebackq" %%f in (`"dir /ad/b/s .vs"`) do rd "%%f" /S/Q
for /f "usebackq" %%f in (`"dir /ad/b/s bin"`) do rd "%%f" /S/Q
for /f "usebackq" %%f in (`"dir /ad/b/s obj"`) do rd "%%f" /S/Q
for /f "usebackq" %%f in (`"dir /ad/b/s .tmp"`) do rd "%%f" /S/Q


