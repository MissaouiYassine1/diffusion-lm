Get-ChildItem -Recurse -Directory | Where-Object {
    -not (Get-ChildItem $_.FullName)
} | ForEach-Object {
    New-Item -Path (Join-Path $_.FullName ".gitkeep") -ItemType File
}